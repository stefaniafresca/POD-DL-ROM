"""

Stefania Fresca, MOX Laboratory, Politecnico di Milano
April 2020

"""

import tensorflow as tf
import numpy as np
import scipy.io as sio
import time
import os

import utils

seed = 374
np.random.seed(seed)

class Net:
    def __init__(self, config):
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.g_step = tf.Variable(0, dtype = tf.int32, trainable = False, name = 'global_step')

        self.n_data = config['n_data']
        self.n_train = int(0.8 * self.n_data)
        self.n_early_stopping = config['n_early_stopping']
        self.N_h = config['N_h']
        self.N = config['N']
        self.n_channels = config['n_channels']
        self.N_t = config['N_t']

        self.POD_mat = config['POD_mat']
        self.compute_POD = config['compute_POD']
        self.train_mat = config['train_mat']
        self.test_mat = config['test_mat']
        self.train_params = config['train_params']
        self.test_params = config['test_params']

        self.omega_h = config['omega_h']
        self.omega_N = config['omega_N']

        self.checkpoints_folder = config['checkpoints_folder']
        self.graph_folder = config['graph_folder']
        self.large_POD = config['large_POD']
        self.large = config['large']
        self.restart = config['restart']
        self.scaling = config['scaling']

    def get_data(self):
        with tf.name_scope('data'):
            self.X = tf.placeholder(tf.float32, shape = [None, self.n_channels * self.N]) # intrinsic coordinates - u_N
            self.Y = tf.placeholder(tf.float32, shape = [None, self.n_params]) # params - (mu, t)

            dataset = tf.data.Dataset.from_tensor_slices((self.X, self.Y))
            dataset = dataset.shuffle(self.n_data)
            dataset = dataset.batch(self.batch_size)

            iterator = dataset.make_initializable_iterator()
            self.init = iterator.initializer

            input, self.params = iterator.get_next()
            self.input = tf.reshape(input, shape = [-1, int(np.sqrt(self.N)), int(np.sqrt(self.N)), self.n_channels])

    def inference(self):
        raise NotImplementedError("Must be overridden with proper definition of forwaN path")

    def loss(self, u_N, u_n):
        with tf.name_scope('loss'):
            output = tf.reshape(self.input, shape = [-1, self.n_channels * self.N])
            self.loss_h = self.omega_h * tf.reduce_mean(tf.reduce_sum(tf.pow(output - u_N, 2), axis = 1))
            self.loss_N = self.omega_N * tf.reduce_mean(tf.reduce_sum(tf.pow(self.enc - u_n, 2), axis = 1))
            self.loss = self.loss_h + self.loss_N

    def optimize(self):
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step = self.g_step)

    def summary(self):
        with tf.name_scope('summaries'):
            self.summary = tf.summary.scalar('loss', self.loss)

    def build(self):
        self.get_data()
        self.inference()
        self.loss(self.u_N, self.u_n)
        self.optimize()
        self.summary()

    def train_one_epoch(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init, feed_dict = {self.X : self.S_train, self.Y : self.params_train})
        total_loss_h = 0
        total_loss_N = 0
        total_loss = 0
        n_batches = 0
        print('------------ TRAINING -------------', flush = True)
        try:
            while True:
                _, l_h, l_N, l, summary = sess.run([self.opt, self.loss_h, self.loss_N, self.loss, self.summary])
                writer.add_summary(summary, global_step = step)
                step += 1
                total_loss_h += l_h
                total_loss_N += l_N
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss_h at epoch {0} on training set: {1}'.format(epoch, total_loss_h / n_batches))
        print('Average loss_N at epoch {0} on training set: {1}'.format(epoch, total_loss_N / n_batches))
        print('Average loss at epoch {0} on training set: {1}'.format(epoch, total_loss / n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init, feed_dict = {self.X : self.S_val, self.Y : self.params_val})
        total_loss_h = 0
        total_loss_N = 0
        total_loss = 0
        n_batches = 0
        print('------------ VALIDATION ------------')
        try:
            while True:
                l_h, l_N, l, summary = sess.run([self.loss_h, self.loss_N, self.loss, self.summary])
                writer.add_summary(summary, global_step = step)
                total_loss_h += l_h
                total_loss_N += l_N
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        total_loss_mean = total_loss / n_batches
        if total_loss_mean < self.loss_best:
            saver.save(sess, self.checkpoints_folder + '/Net', step)
        print('Average loss_h at epoch {0} on validation set: {1}'.format(epoch, total_loss_h / n_batches))
        print('Average loss_N at epoch {0} on validation set: {1}'.format(epoch, total_loss_N / n_batches))
        print('Average loss at epoch {0} on validation set: {1}'.format(epoch, total_loss_mean))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return total_loss_mean

    def test_once(self, sess, init):
        start_time = time.time()
        sess.run(init, feed_dict = {self.X : self.S_test, self.Y : self.params_test})
        total_loss_h = 0
        total_loss_N = 0
        total_loss = 0
        n_batches = 0
        self.U_N = np.zeros(self.S_test.shape)
        print('------------ TESTING ------------')
        try:
            while True:
                l_h, l_N, l, u_N, enc, u_n = sess.run([self.loss_h, self.loss_N, self.loss, self.u_N, self.enc, self.u_n])
                self.U_N[self.batch_size * n_batches : self.batch_size * (n_batches + 1)] = u_N
                total_loss_h += l_h
                total_loss_N += l_N
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss_h on testing set: {0}'.format(total_loss_h / n_batches))
        print('Average loss_N on testing set: {0}'.format(total_loss_N / n_batches))
        print('Average loss on testing set: {0}'.format(total_loss / n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))

    #@profile
    def train_all(self, n_epochs):
        if (not self.restart):
            utils.safe_mkdir(self.checkpoints_folder)
        saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter('./' + self.graph_folder + '/train', tf.get_default_graph())
        test_writer = tf.summary.FileWriter('./' + self.graph_folder + '/val', tf.get_default_graph())

        print('Loading snapshot matrix...')
        if (self.large):
            S = utils.read_large_data(self.train_mat)
        else:
            S = utils.read_data(self.train_mat)

        if self.compute_POD == 'exact':
            self.V = utils.compute_SVD(np.transpose(S), self.N, self.N_h, self.n_channels)
        elif self.compute_POD == 'randomized':
            self.V = utils.compute_randomized_SVD(np.transpose(S), self.N, self.N_h, self.n_channels)
        else:
            self.V = utils.read_POD_data(self.POD_mat)

        idxs = np.random.permutation(S.shape[0])
        S = S[idxs]

        self.S_train, self.S_val = np.zeros((self.n_train, self.n_channels * self.N)), np.zeros((S.shape[0] - self.n_train, self.n_channels * self.N))
        for i in range(self.n_channels):
            self.S_train[:, i * self.N : (i + 1) * self.N] = np.matmul(S[:self.n_train, i * self.N_h : (i + 1) * self.N_h], self.V[i * self.N_h : (i + 1) * self.N_h])
            self.S_val[:, i * self.N : (i + 1) * self.N] = np.matmul(S[self.n_train:, i * self.N_h : (i + 1) * self.N_h], self.V[i * self.N_h : (i + 1) * self.N_h])
        del S

        if self.scaling:
            S_max, S_min = utils.max_min_componentwise(self.S_train, self.n_train, self.n_channels, self.N)
            utils.scaling_componentwise(self.S_train, S_max, S_min, self.n_channels, self.N)
            utils.scaling_componentwise(self.S_val, S_max, S_min, self.n_channels, self.N)

        print('Loading parameters...')
        params = utils.read_params(self.train_params)
        params = params[idxs]
        if self.scaling:
            params_max, params_min = utils.max_min_componentwise_params(params, self.n_train, params.shape[1])
            utils.scaling_componentwise_params(params, params_max, params_min, params.shape[1])
        self.params_train, self.params_val = params[:self.n_train], params[self.n_train:]
        del params

        self.loss_best = 1
        count = 0
        gpu_options = tf.GPUOptions(allow_growth = True)
        with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            if (self.restart):
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoints_folder + '/checkpoint'))
                if ckpt and ckpt.model_checkpoint_path:
                    print(ckpt.model_checkpoint_path)
                    saver.restore(sess, ckpt.model_checkpoint_path)

            step = self.g_step.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, self.init, train_writer, epoch, step)
                total_loss_mean = self.eval_once(sess, saver, self.init, test_writer, epoch, step)
                if total_loss_mean < self.loss_best:
                    self.loss_best = total_loss_mean
                    count = 0
                else:
                    count += 1
                # early - stopping criterion
                if count == self.n_early_stopping:
                    print('Stopped training due to early-stopping cross-validation')
                    break
            print('Best loss on validation set: {0}'.format(self.loss_best))

        train_writer.close()
        test_writer.close()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoints_folder + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)

            print('Loading testing snapshot matrix...')
            if (self.large):
                S_test = utils.read_large_data(self.test_mat)
            else:
                S_test = utils.read_data(self.test_mat)

            self.S_test = np.zeros((S_test.shape[0], self.n_channels * self.N))
            for i in range(self.n_channels):
                self.S_test[:, i * self.N : (i + 1) * self.N] = np.matmul(S_test[:, i * self.N_h : (i + 1) * self.N_h], self.V[i * self.N_h : (i + 1) * self.N_h])

            if self.scaling:
                utils.scaling_componentwise(self.S_test, S_max, S_min, self.n_channels, self.N)

            print('Loading testing parameters...')
            self.params_test = utils.read_params(self.test_params)
            if self.scaling:
                utils.scaling_componentwise_params(self.params_test, params_max, params_min, self.params_test.shape[1])

            self.test_once(sess, self.init)

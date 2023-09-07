"""

Stefania Fresca, MOX Laboratory, Politecnico di Milano
April 2020

"""

import tensorflow as tf
import numpy as np
import scipy.io as sio
import time
import os

from Net import Net
import utils

seed = 374
np.random.seed(seed)

class DecNet(Net):
    def __init__(self, config):
        Net.__init__(self, config)
        self.n = config['n']

        self.n_params = config['n_params']
        self.size = 5
        self.n_layers = 10
        self.n_neurons = 50
        self.n_h = config['n_h']

        self.export = config['export']

    def get_data(self):
        with tf.name_scope('data'):
            self.X = tf.placeholder(tf.float32, shape = [None, self.n_params]) # params - (mu, t)
            self.Y = tf.placeholder(tf.float32, shape = [None, self.n_channels * self.N]) # intrinsic coordinates - u_N

            dataset = tf.data.Dataset.from_tensor_slices((self.X, self.Y))
            dataset = dataset.batch(self.batch_size)

            iterator = dataset.make_initializable_iterator()
            self.init = iterator.initializer

            self.params, self.output = iterator.get_next()

    def inference(self):
        fc_n = tf.layers.dense(self.params,
                               self.n_neurons,
                               activation = tf.nn.elu,
                               kernel_initializer = tf.keras.initializers.he_uniform())
        for i in range(self.n_layers):
            fc_n = tf.layers.dense(fc_n,
                                   self.n_neurons,
                                   activation = tf.nn.elu,
                                   kernel_initializer = tf.keras.initializers.he_uniform())
        u_n = tf.layers.dense(fc_n,
                              self.n,
                              activation = tf.nn.elu,
                              kernel_initializer = tf.keras.initializers.he_uniform())
        fc1_t = tf.layers.dense(u_n, 64, activation = tf.nn.elu, kernel_initializer = tf.keras.initializers.he_uniform(), name = 'fc1_t')
        fc2_t = tf.layers.dense(fc1_t, self.N, activation = tf.nn.elu, kernel_initializer = tf.keras.initializers.he_uniform(), name = 'fc2_t')
        fc2_t = tf.reshape(fc2_t, [-1, self.n_h, self.n_h, 64])
        conv1_t = tf.layers.conv2d_transpose(inputs = fc2_t,
                                             filters = 64,
                                             kernel_size = [self.size, self.size],
                                             padding = 'SAME',
                                             strides = 2,
                                             kernel_initializer = tf.keras.initializers.he_uniform(),
                                             activation = tf.nn.elu,
                                             name = 'conv1_t')
        conv2_t = tf.layers.conv2d_transpose(inputs = conv1_t,
                                             filters = 32,
                                             kernel_size = [self.size, self.size],
                                             padding = 'SAME',
                                             strides = 2,
                                             kernel_initializer = tf.keras.initializers.he_uniform(),
                                             activation = tf.nn.elu,
                                             name = 'conv2_t')
        conv3_t = tf.layers.conv2d_transpose(inputs = conv2_t,
                                             filters = 16,
                                             kernel_size = [self.size, self.size],
                                             padding = 'SAME',
                                             strides = 2,
                                             kernel_initializer = tf.keras.initializers.he_uniform(),
                                             activation = tf.nn.elu,
                                             name = 'conv3_t')
        conv4_t = tf.layers.conv2d_transpose(inputs = conv3_t,
                                             filters = self.n_channels,
                                             kernel_size = [self.size, self.size],
                                             padding = 'SAME',
                                             strides = 1,
                                             kernel_initializer = tf.keras.initializers.he_uniform(),
                                             name = 'conv4_t')
        feature_dim_dec = conv4_t.shape[1] * conv4_t.shape[2] * conv4_t.shape[3]
        self.u_N = tf.reshape(conv4_t, [-1, feature_dim_dec])

    def loss(self, u_N):
        with tf.name_scope('loss'):
            self.loss = self.omega_h * tf.reduce_mean(tf.reduce_sum(tf.pow(self.output - u_N, 2), axis = 1))

    def build(self):
        self.get_data()
        self.inference()
        self.loss(self.u_N)

    def test_once(self, sess, init):
        start_time = time.time()
        sess.run(init, feed_dict = {self.X : self.params_test, self.Y : self.S_test})
        total_loss = 0
        n_batches = 0
        self.U_N = np.zeros(self.S_test.shape)
        print('------------ TESTING ------------')
        try:
            while True:
                l, u_N = sess.run([self.loss, self.u_N])
                self.U_N[self.batch_size * n_batches : self.batch_size * (n_batches + 1)] = u_N
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss on testing set: {0}'.format(total_loss / n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))

    #@profile
    def test_all(self):
        list = [v for v in tf.global_variables() if '_t' or 'dense' in v.name]
        saver = tf.train.Saver(var_list = list)

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

        if self.scaling:
            idxs = np.random.permutation(S.shape[0])
            S = S[idxs]

            S_train = np.zeros((self.n_train, self.n_channels * self.N))
            for i in range(self.n_channels):
                S_train[:, i * self.N : (i + 1) * self.N] = np.matmul(S[:self.n_train, i * self.N_h : (i + 1) * self.N_h], self.V[i * self.N_h : (i + 1) * self.N_h])
            S_max, S_min = utils.max_min_componentwise(S_train, self.n_train, self.n_channels, self.N)
            del S_train

            params = utils.read_params(self.train_params)
            params = params[idxs]
            params_max, params_min = utils.max_min_componentwise_params(params, self.n_train, params.shape[1])
            del params
        del S

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

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoints_folder + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                self.test_once(sess, self.init)

                if self.scaling:
                    utils.inverse_scaling_componentwise(self.U_N, S_max, S_min, self.n_channels, self.N)
                    utils.inverse_scaling_componentwise_params(self.params_test, params_max, params_min, self.params_test.shape[1])

                n_test = self.S_test.shape[0] // self.N_t
                err = np.zeros((n_test, 1))
                U_h = np.zeros((self.N_t, self.n_channels * self.N_h))
                for i in range(n_test):
                    for j in range(self.n_channels):
                        U_h[:, j * self.N_h : (j + 1) * self.N_h] = np.matmul(self.U_N[i * self.N_t : (i + 1) * self.N_t, j * self.N : (j + 1) * self.N], np.transpose(self.V[j * self.N_h : (j + 1) * self.N_h]))
                    num = np.sqrt(np.mean(np.linalg.norm(S_test[i * self.N_t : (i + 1) * self.N_t] - U_h, 2, axis = 1) ** 2))
                    den = np.sqrt(np.mean(np.linalg.norm(S_test[i * self.N_t : (i + 1) * self.N_t], 2, axis = 1) ** 2))
                    err[i] = num / den
                print('Relative error indicator: {0}'.format(np.mean(err)))

                if self.export:
                    sio.savemat('*.mat', {'S': U_h})

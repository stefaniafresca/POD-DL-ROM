"""

Stefania Fresca, MOX Laboratory, Politecnico di Milano
April 2020

"""

import tensorflow as tf
import numpy as np

from Net import Net

class ROMNet(Net):
    def __init__(self, config):
        Net.__init__(self, config)
        self.n = config['n']

        self.n_params = config['n_params']
        self.size = 5
        self.n_layers = 10
        self.n_neurons = 50
        self.n_h = config['n_h']

    def inference(self):
        # decoder function
        conv1 = tf.layers.conv2d(inputs = self.input,
                                 filters = 8,
                                 kernel_size = [self.size, self.size],
                                 padding = 'SAME',
                                 strides = 1,
                                 kernel_initializer = tf.keras.initializers.he_uniform(),
                                 activation = tf.nn.elu,
                                 name = 'conv1')
        conv2 = tf.layers.conv2d(inputs = conv1,
                                 filters = 16,
                                 kernel_size = [self.size, self.size],
                                 padding = 'SAME',
                                 strides = 2,
                                 kernel_initializer = tf.keras.initializers.he_uniform(),
                                 activation = tf.nn.elu,
                                 name = 'conv2')
        conv3 = tf.layers.conv2d(inputs = conv2,
                                 filters = 32,
                                 kernel_size = [self.size, self.size],
                                 padding = 'SAME',
                                 strides = 2,
                                 kernel_initializer = tf.keras.initializers.he_uniform(),
                                 activation = tf.nn.elu,
                                 name = 'conv3')
        conv4 = tf.layers.conv2d(inputs = conv3,
                                 filters = 64,
                                 kernel_size = [self.size, self.size],
                                 padding = 'SAME',
                                 strides = 2,
                                 kernel_initializer = tf.keras.initializers.he_uniform(),
                                 activation = tf.nn.elu,
                                 name = 'conv4')
        feature_dim_enc = conv4.shape[1] * conv4.shape[2] * conv4.shape[3]
        conv4 = tf.reshape(conv4, [-1, feature_dim_enc])
        fc1 = tf.layers.dense(conv4, 64, activation = tf.nn.elu, kernel_initializer = tf.keras.initializers.he_uniform(), name = 'fc1')
        self.enc = tf.layers.dense(fc1, self.n, activation = tf.nn.elu, kernel_initializer = tf.keras.initializers.he_uniform(), name = 'fc2')
        # feedforward neural network to model dynamics on the nonlinear manifold
        fc_n = tf.layers.dense(self.params,
                               self.n_neurons,
                               activation = tf.nn.elu,
                               kernel_initializer = tf.keras.initializers.he_uniform())
        for i in range(self.n_layers):
            fc_n = tf.layers.dense(fc_n,
                                   self.n_neurons,
                                   activation = tf.nn.elu,
                                   kernel_initializer = tf.keras.initializers.he_uniform())
        self.u_n = tf.layers.dense(fc_n,
                                   self.n,
                                   activation = tf.nn.elu,
                                   kernel_initializer = tf.keras.initializers.he_uniform())
        fc1_t = tf.layers.dense(self.u_n, 64, activation = tf.nn.elu, kernel_initializer = tf.keras.initializers.he_uniform(), name = 'fc1_t')
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

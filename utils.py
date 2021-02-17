import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import scipy.io as sio
import h5py
from scipy import linalg
from sklearn.utils import extmath
import time

def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass

def read_data(mat):
    data = sio.loadmat(mat)
    S = data['S'].squeeze()
    S = np.transpose(S)

    return S

def read_large_data(mat):
    file = h5py.File(mat, 'r')
    S = file['S'][:]

    return S

def read_params(mat):
    params = sio.loadmat(mat)
    params = params['I'].squeeze()

    return params

def zero_pad(S, n):
    paddings = np.zeros((S.shape[0], n))
    S = np.hstack((S, paddings))

    return S

def read_POD_data(mat):
    print('Loading POD basis...')
    data = sio.loadmat(mat)
    V = data['V'].squeeze()

    return V

def read_large_POD_data(mat):
    print('Loading POD basis...')
    file = h5py.File(mat, 'r')
    V = file['V'][:]
    V = np.transpose(V)

    return V

def compute_SVD(S, N, N_h, n_channels, name = ''):
    print('Computing exact POD...')
    U = np.zeros((n_channels * N_h, N))
    start_time = time.time()
    for i in range(n_channels):
        U, Sigma, Vh = linalg.svd(S[i * N_h : (i + 1) * N_h, :], full_matrices = False, overwrite_a = False, check_finite = False, lapack_driver = 'gesvd')
    print('Done... Took: {0} seconds'.format(time.time() - start_time))

    if name:
        sio.savemat(name, {'V': U[:, :N]})

    return U[:, :N]

def compute_randomized_SVD(S, N, N_h, n_channels, name = ''):
    print('Computing randomized POD...')
    U = np.zeros((n_channels * N_h, N))
    start_time = time.time()
    for i in range(n_channels):
        U[i * N_h : (i + 1) * N_h], Sigma, Vh = extmath.randomized_svd(S[i * N_h : (i + 1) * N_h, :], n_components = N, transpose = False, flip_sign =  False, random_state = 123)
    print('Done: computed matrix V of shape {0}... Took: {1} seconds'.format(U.shape, time.time() - start_time))

    if name:
        sio.savemat(name, {'V': U[:, :N]})

    return U

def max_min(S, n_train):
    S_max = np.max(np.max(S[:n_train], axis = 1), axis = 0) # np.max(S, axis = 1) -> max of each row
    S_min = np.min(np.min(S[:n_train], axis = 1), axis = 0)

    return S_max, S_min

def scaling(S, S_max, S_min):
    S[ : ] = (S - S_min)/(S_max - S_min)

def inverse_scaling(S, S_max, S_min):
    S[ : ] = (S_max - S_min) * S + S_min

def max_min_componentwise(S, n_train, n_components, N):
    S_max, S_min = np.zeros((n_components, 1)), np.zeros((n_components, 1))

    for i in range(n_components):
        S_max[i], S_min[i] = max_min(S[:, i * N : (i + 1) * N], n_train)

    return S_max, S_min

def scaling_componentwise(S, S_max, S_min, n_components, N):
    for i in range(n_components):
        scaling(S[:, i * N : (i + 1) * N], S_max[i], S_min[i])

def inverse_scaling_componentwise(S, S_max, S_min, n_components, N):
    for i in range(n_components):
        inverse_scaling(S[:, i * N : (i + 1) * N], S_max[i], S_min[i])

def max_min_componentwise_params(S, n_train, n_components):
    S_max, S_min = np.zeros((n_components, 1)), np.zeros((n_components, 1))

    for i in range(n_components):
        S_max[i], S_min[i] = max_min(S[:, i][np.newaxis], n_train)

    return S_max, S_min

def scaling_componentwise_params(S, S_max, S_min, n_components):
    for i in range(n_components):
        scaling(S[:, i], S_max[i], S_min[i])

def inverse_scaling_componentwise_params(S, S_max, S_min, n_components):
    for i in range(n_components):
        inverse_scaling(S[:, i][np.newaxis], S_max[i], S_min[i])

def print_trainable_variables(sess):
    variables = [v for v in tf.trainable_variables()]
    values = sess.run(variables)
    for k, v in zip(variables, values):
        print('Variable: ', k)
        print('Shape: ', v.shape)
        print(v)

def count_trainable_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        print(variable)
        local_parameters = 1
        shape = variable.get_shape()
        print(shape)
        for i in shape:
            local_parameters *= i.value
        total_parameters += local_parameters
    return total_parameters

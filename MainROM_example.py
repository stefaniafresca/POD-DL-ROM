"""

Stefania Fresca, MOX Laboratory, Politecnico di Milano
April 2020

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.stdout = open('*.out', 'w')

import utils

from ROMNet import ROMNet

if __name__ == '__main__':
    config = dict()
    config['n'] =                                                                 # reduced dimension - n
    config['n_params'] =                                                          # number of parameters + 1 - n_{mu} + 1
    config['lr'] =                                                                # starting learning rate - eta
    config['omega_h'] =
    config['omega_N'] =
    config['batch_size'] =                                                        # batch_size
    config['n_data'] =                                                            # number of training samples - N_{train} x N_t
    config['N_h'] =                                                               # FOM dimension - N_h
    config['N'] =                                                                 # rPOD dimension - N
    config['n_h'] =
    config['N_t'] =                                                               # number of time instances - N_t
    config['n_channels'] =                                                        # number of channels - d
    config['compute_POD'] =                                                       # options: '', 'exact', 'randomized'
    config['POD_mat'] =                                                           # POD matrix filename
    config['train_mat'] =                                                         # training snapshot matrix
    config['test_mat'] =                                                          # testing snapshot matrix
    config['train_params'] =                                                      # training parameter matrix
    config['test_params'] =                                                       # testing parameter matrix
    config['checkpoints_folder'] =                                                # checkpoints folder
    config['graph_folder'] =                                                      # graphs folder
    config['n_early_stopping'] =                                                  # number of epochs for early-stopping criterion
    config['large_POD'] =                                                         # True if POD matrix in .h5 format
    config['large'] =                                                             # True if snapshot matrices in .h5 format
    config['restart'] =                                                           # True if restart
    config['scaling'] =                                                           # True if data must be scaled

    model = ROMNet(config)
    model.build()
    model.train_all()                                                             # input: total numer of epochs - N_{epochs}

import os
import glob
import shutil
import time

import numpy as np

import tools.split_data as split_data
import parallel_GPLVM

P = 10
Q = 7
N = 1000
iterations = 500
num_inducing = 40
dropout_freq = [0, 0.01, 0.02, 0.03] # 1/P jumps
path = '/scratch/yg279/Drop_out_test_3/oildata'
dname = 'oildata'

# Prepare directories
reqdirs = ['inputs', 'embeddings', 'tmp', 'proc']
for dirname in reqdirs:
    if not os.path.exists(path + '/' + dirname):
        os.mkdir(path + '/' + dirname)

for repetition in range(0,10):
    np.random.seed(seed=repetition)
    for dropout in dropout_freq:
        # Load data
        Y = np.loadtxt(path + '/proc/oilflow', delimiter=',')
        print ('Dataset size:')
        print (Y.shape)
        print ('N:')
        print (N)
        print ('Dropout:')
        print (dropout)
        print ('Repetition:')
        print (repetition)

        # First delete all current inputs & embeddings
        split_data.clean_dir(path)

        perm = split_data.split_data(Y[:N], P, path, dname)
        np.save(path + '/permutation.npy', perm)

        # Run the Parallel GPLVM
        options = {}
        options['input'] = path + '/inputs/'
        options['embeddings'] = path + '/embeddings/'
        options['parallel'] = 'local'
        options['iterations'] = iterations
        options['statistics'] = path + '/tmp'
        options['tmp'] = path + '/tmp'
        options['M'] = num_inducing
        options['Q'] = Q
        options['D'] = 12
        options['fixed_embeddings'] = False
        options['keep'] = False
        options['load'] = False
        options['fixed_beta'] = True
        options['init'] = 'PCA'
        options['optimiser'] = 'SCG_adapted'
        options['fixed_beta'] = False
        options['drop_out_fraction'] = dropout

        options['local_no_pool'] = False

        parallel_GPLVM.main(options)

        # Copy output directory
        shutil.copytree(path, path + '_dropout_' + str(dropout)
                              + '_repetition_' + str(repetition) + '_' + str(time.time()))

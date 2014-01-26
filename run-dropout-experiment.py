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
iterations = 300
num_inducing = 40
path = './oildata'
dname = 'oildata'

for repetition in [1, 2, 3, 4, 5]:
    for dropout in [0, 0.2, 0.4, 0.6, 0.8]:
        # First delete all current inputs & embeddings
        split_data.clean_dir(path)

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

        # Prepare directories
        reqdirs = ['inputs', 'embeddings', 'tmp', 'proc']
        for dirname in reqdirs:
            if not os.path.exists(path + '/' + dirname):
                os.mkdir(path + '/' + dirname)

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

        options['local_no_pool'] = False

        parallel_GPLVM.main(options)

        # Copy output directory
        shutil.copytree(path, path + '_dropout_' + str(dropout)
                              + '_repetition_' + str(repetition) + '_' + str(time.time()))

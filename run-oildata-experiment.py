import os
import glob
import shutil
import time

import numpy as np

import tools.split_data as split_data
import parallel_GPLVM

P = 7
Q = 7
N = 1000
num_inducing = 40
path = './oildata/'
dname = 'oildata'

# First delete all current inputs & embeddings
split_data.clean_dir(path)

# Load data
Y = np.loadtxt(path + '/proc/oilflow', delimiter=',')
print ('Dataset size:')
print (Y.shape)
print ('N:')
print (N)
perm = split_data.split_data(Y[:N], P, path, dname)
np.save(path + 'permutation.npy', perm)

# Prepare directories
reqdirs = ['inputs', 'embeddings', 'tmp', 'proc']
for dirname in reqdirs:
    if not os.path.exists(path + '/' + dirname):
        os.mkdir(path + '/' + dirname)

# Run the Parallel GPLVM
options = {}
options['input'] = path + '/inputs/'
options['embeddings'] = path + '/embeddings/'
options['parallel'] = 'local'
options['iterations'] = 1000
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
shutil.copytree(path, '/scratch/mv310/results/' + dname + str(time.time()))

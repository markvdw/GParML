import os
import glob
import shutil
import time
import gzip
import cPickle

import numpy as np

import tools.split_data as split_data
import parallel_GPLVM

P = 60
Q = 20
num_inducing = 100
path = './mnist/'
dname = 'mnist'

# First delete all current inputs & embeddings
split_data.clean_dir(path)

# Load data
f = gzip.open(path + '/proc/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

Y = train_set[0][:N, :]

split_data.split_data(Y, P, path, dname)

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
options['D'] = Y.shape[1]
options['fixed_embeddings'] = False
options['keep'] = False
options['load'] = False
options['fixed_beta'] = True
options['init'] = 'PCA'
options['optimiser'] = 'SCG_adapted'
options['fixed_beta'] = False

start_time = time.time()
parallel_GPLVM.main(options)
print('Elapsed time:')
print(time.time() - start_time)

# Copy output directory
shutil.copytree(path, '/scratch/mv310/results/' + dname + str(time.time()))

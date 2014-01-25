import os
import glob
import shutil
import time
import cPickle

import numpy as np

import tools.split_data as split_data
import parallel_GPLVM

print('Starting...')

N = 100
P = 2
Q = 9
# num_inducing = min(900, N / P)
num_inducing = 40
path = './flight/'
dname = 'flight'

# 'Year', 'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'ArrTime', 'ArrDelay', 'AirTime', 'Distance', 'plane_age'

# First delete all current inputs & embeddings
split_data.clean_dir(path)

# Prepare directories
reqdirs = ['inputs', 'embeddings', 'tmp', 'proc']
for dirname in reqdirs:
    if not os.path.exists(path + '/' + dirname):
        os.mkdir(path + '/' + dirname)

# Load data
print('Loading data...')
Y = np.load('./flight/proc/flight_regression_output.npy')
X = np.load('./flight/proc/flight_regression_inputs.npy')

# Normalise data
X = X - np.mean(X)
X = X / np.std(X)
Y = Y - np.mean(Y)
Y = Y / np.std(Y)

print('Embeddings shape:')
print(X.shape)
print('Data shape:')
print(Y.shape)

print('Splitting data...')
perm = split_data.split_data(Y[:N], P, path, dname)
split_data.split_embeddings(X[:N], P, path, dname, perm=perm)

# Run the Parallel GPLVM
options = {}
options['input'] = path + '/inputs/'
options['embeddings'] = path + '/embeddings/'
options['parallel'] = 'local'
options['iterations'] = 100
options['statistics'] = path + '/tmp'
options['tmp'] = path + '/tmp'
options['M'] = num_inducing
options['Q'] = Q
options['D'] = 1
options['fixed_embeddings'] = True
options['keep'] = False
options['load'] = False
options['init'] = 'PCA'
options['optimiser'] = 'SCG_adapted'
options['fixed_beta'] = False

options['local_no_pool'] = False

print('Running...')
parallel_GPLVM.main(options)

# Copy output directory
shutil.copytree(path, '/scratch/mv310/results/' + dname + str(time.time()))

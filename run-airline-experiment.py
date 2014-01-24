import os
import glob
import shutil
import time

import numpy as np

import tools.split_data as split_data
import parallel_GPLVM

P = 10
Q = 7
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
Y = cPickle.load(open('./flight/proc/filtered_data.pickle'))

items = np.random.permutation(Y.shape[0])[:800000]

inputs = np.array(Y[['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'ArrTime', 'AirTime', 'Distance', 'plane_age']])[items][:]
output = np.array(Y['ArrDelay'])[items]

perm = split_data.split_data(outputs, P, path, dname)
split_data.split_embeddings(inputs, P, path, dname, perm)

# Normalise data
inputs = inputs - np.mean(inputs)
inputs = inputs / np.std(inputs)
output = output - np.mean(output)
output = output / np.std(output)

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
options['D'] = 1
options['fixed_embeddings'] = True
options['keep'] = False
options['load'] = True
options['init'] = 'PCA'
options['optimiser'] = 'SCG_adapted'
options['fixed_beta'] = False

parallel_GPLVM.main(options)

# Copy output directory
shutil.copytree(path, '/scratch/mv310/results/' + dname + str(time.time()))

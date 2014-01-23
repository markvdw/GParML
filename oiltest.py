import sys
import glob
import os
import random

import numpy as np
import numpy.linalg as linalg
import numpy.random as rnd
from mpl_toolkits.mplot3d.axes3d import Axes3D

import GPy

sys.path.append('../tools/')
sys.path.append('..')

import tools.split_data as split_data

import parallel_GPLVM

Q = 7
max_iters = 200
N = 1000
P = 6
num_inducing = min(40, N / P) - 1

# Get oildata and divide up
oildata = GPy.util.datasets.oil()
Y = oildata['X'][:N]
path = './oildata/'
split_data.clean_dir(path)
split_data.split_data(Y, P, path, 'oil')

# Run the Parallel GPLVM
options = {}
options['input'] = path + '/inputs/'
options['embeddings'] = path + '/embeddings/'
options['parallel'] = 'local'
options['iterations'] = max_iters
options['statistics'] = path + '/tmp'
options['tmp'] = path + '/tmp'
options['M'] = num_inducing
options['Q'] = Q
options['D'] = 12
options['fixed_embeddings'] = False
options['keep'] = True
options['load'] = False
options['fixed_beta'] = True
options['init'] = 'PCA'
options['optimiser'] = 'SCG_adapted'
options['fixed_beta'] = False

parallel_GPLVM.main(options)

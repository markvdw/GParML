'''
A bunch of support functions used for GD optimisation. They depend on the
parallel implementation framework, but may change for other optimisers.
'''

import glob
import numpy
from os.path import splitext
from local_MapReduce import load, save

'''
Initialisation for local statistics
'''
def embeddings_set_grads(folder):
    '''
    Sets the grads and other local statistics often needed for optimisation locally for
    each node. This is currently only implemented locally, but could easily be adapted
    to the MapReduce framework to be done on remote nodes in parallel. There's no real
    need to do this in parallel though, as the computations taking place are not that
    time consuming.
    '''
    input_files = sorted(glob.glob(folder + '/*.grad_latest.npy'))
    for file_name in input_files:
        grads = load(file_name)
        #print 'grads'
        #print grads
        # Save grad new as the latest grad evaluated
        new_file = splitext(splitext(file_name)[0])[0] + '.grad_now.npy'
        save(new_file, grads)
        # Save the direction as the negative grad
        new_file = splitext(splitext(file_name)[0])[0] + '.grad_d.npy'
        save(new_file, -1 * grads)

'''
Getters for local statistics
'''

def embeddings_get_grads_current_grad(folder):
    '''
    Get the sum over the inputs of the inner product of grad_now with itself
    '''
    current_grad = 0
    grad_now_files = sorted(glob.glob(folder + '/*.grad_now.npy'))
    for grad_now_file in grad_now_files:
        grad_now = load(grad_now_file)
        current_grad += numpy.sum(numpy.abs(grad_now))
    return current_grad

def embeddings_get_grads_max_gradnow(folder):
    '''
    Get the max abs element of the direction over all input files
    '''
    max_gradnow = 0
    grad_now_files = sorted(glob.glob(folder + '/*.grad_now.npy'))
    for grad_now_file in grad_now_files:
        grad_now = load(grad_now_file)
        max_gradnow = max(max_gradnow, numpy.max(numpy.abs(grad_now)))
    return max_gradnow

'''
Setters for local statistics
'''
def embeddings_set_grads_update_d(folder, gamma):
    '''
    Update the value of the direction for each input to be gamma (given) times the old direction
    minus grad_now
    '''
    grad_now_files = sorted(glob.glob(folder + '/*.grad_now.npy'))
    grad_d_files = sorted(glob.glob(folder + '/*.grad_d.npy'))
    for grad_now_file, grad_d_file in zip(grad_now_files, grad_d_files):
        grad_now = load(grad_now_file)
        grad_d = load(grad_d_file)
        save(grad_d_file,  - (grad_now + gamma * grad_d))
        #save(grad_d_file,  - (grad_now - gamma * grad_d))

def embeddings_set_grads_update_X(folder, step_size):
    '''
    Update the value of the local embeddings and variances themselves to be X + alpha * direction
    '''
    grad_d_files = sorted(glob.glob(folder + '/*.grad_d.npy'))
    X_mu_files = sorted(glob.glob(folder + '/*.embedding.npy'))
    X_S_files = sorted(glob.glob(folder + '/*.variance.npy'))
    for grad_d_file, X_mu_file, X_S_file in zip(grad_d_files, X_mu_files, X_S_files):
        grad_d = load(grad_d_file)
        grad_d_X_mu = grad_d[0]
        grad_d_X_S = grad_d[1]
        X_mu = load(X_mu_file)
        X_S = load(X_S_file)
        #print 'X_mu'
        #print X_mu
        #print 'X_S'
        #print X_S
        save(X_mu_file, X_mu + step_size * grad_d_X_mu)
        save(X_S_file, X_S + step_size * grad_d_X_S)

def embeddings_set_grads_update_grad_now(folder):
    '''
    Set grad_now to be grad_latest (a temp grad that keeps changing every evaluation)
    '''
    input_files = sorted(glob.glob(folder + '/*.grad_latest.npy'))
    for file_name in input_files:
        grads = load(file_name)
        # Save grad old as latest grad new
        new_file = splitext(splitext(file_name)[0])[0] + '.grad_now.npy'
        save(new_file, grads)

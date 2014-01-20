'''
A bunch of support functions used for SCG optimisation. They depend on the
parallel implementation framework, but may change for other optimisers.
'''

import glob
import time
import numpy
from os.path import splitext
from local_MapReduce import load, save

time_acc = {
    'embeddings_set_grads' : [],
    'embeddings_get_grads_mu' : [],
    'embeddings_get_grads_kappa' : [],
    'embeddings_get_grads_theta' : [],
    'embeddings_get_grads_current_grad' : [],
    'embeddings_get_grads_gamma' : [],
    'embeddings_get_grads_max_d' : [],
    'embeddings_set_grads_reset_d' : [],
    'embeddings_set_grads_update_d' : [],
    'embeddings_set_grads_update_X' : [],
    'embeddings_set_grads_update_grad_old' : [],
    'embeddings_set_grads_update_grad_new' : [],
}
'''
Initialisation for local statistics
'''
def embeddings_set_grads(folder):
    '''
    Sets the grads and other local statistics often needed for optimisation locally for
    each node. This is currently only implemented locally, but could easly be adapted
    to the MapReduce framework to be done on remote nodes in parallel. There's no real
    need to do this in parallel though, as the computaions taking place are not that
    time consuming.
    '''
    global time_acc
    start = time.time()
    input_files = sorted(glob.glob(folder + '/*.grad_latest.npy'))
    for file_name in input_files:
        grads = load(file_name)
        #print 'grads'
        #print grads
        # Save grad new as the latest grad evaluated
        new_file = splitext(splitext(file_name)[0])[0] + '.grad_new.npy'
        save(new_file, grads)
        # Init the old grad to be grad new
        new_file = splitext(splitext(file_name)[0])[0] + '.grad_old.npy'
        save(new_file, grads)
        # Save the direction as the negative grad
        new_file = splitext(splitext(file_name)[0])[0] + '.grad_d.npy'
        save(new_file, -1 * grads)
    end = time.time()
    time_acc['embeddings_set_grads'] += [end - start]

'''
Getters for local statistics
'''
def embeddings_get_grads_mu(folder):
    '''
    Get the sum over the inputs of the inner product of the direction and grad_new
    '''
    global time_acc
    start = time.time()
    mu = 0
    grad_new_files = sorted(glob.glob(folder + '/*.grad_new.npy'))
    grad_d_files = sorted(glob.glob(folder + '/*.grad_d.npy'))
    for grad_new_file, grad_d_file in zip(grad_new_files, grad_d_files):
        grad_new = load(grad_new_file)
        grad_d = load(grad_d_file)
        mu += (grad_new * grad_d).sum()
    end = time.time()
    time_acc['embeddings_get_grads_mu'] += [end - start]
    return mu

def embeddings_get_grads_kappa(folder):
    '''
    Get the sum over the inputs of the inner product of the direction with itself
    '''
    global time_acc
    start = time.time()
    kappa = 0
    grad_d_files = sorted(glob.glob(folder + '/*.grad_d.npy'))
    for grad_d_file in grad_d_files:
        grad_d = load(grad_d_file)
        kappa += (grad_d * grad_d).sum()
    end = time.time()
    time_acc['embeddings_get_grads_kappa'] += [end - start]
    return kappa

def embeddings_get_grads_theta(folder):
    '''
    Get the sum over the inputs of the inner product of the direction and grad_latest
    '''
    global time_acc
    start = time.time()
    theta = 0
    grad_new_files = sorted(glob.glob(folder + '/*.grad_new.npy'))
    grad_latest_files = sorted(glob.glob(folder + '/*.grad_latest.npy'))
    grad_d_files = sorted(glob.glob(folder + '/*.grad_d.npy'))
    for grad_latest_file, grad_d_file, grad_new_file in zip(grad_latest_files, grad_d_files, grad_new_files):
        grad_latest = load(grad_latest_file)
        grad_new = load(grad_new_file)
        grad_d = load(grad_d_file)
        theta += (grad_d * (grad_latest - grad_new)).sum()
    end = time.time()
    time_acc['embeddings_get_grads_theta'] += [end - start]
    return theta

def embeddings_get_grads_current_grad(folder):
    '''
    Get the sum over the inputs of the inner product of grad_new with itself
    '''
    global time_acc
    start = time.time()
    current_grad = 0
    grad_new_files = sorted(glob.glob(folder + '/*.grad_new.npy'))
    for grad_new_file in grad_new_files:
        grad_new = load(grad_new_file)
        current_grad += (grad_new * grad_new).sum()
    end = time.time()
    time_acc['embeddings_get_grads_current_grad'] += [end - start]
    return current_grad

def embeddings_get_grads_gamma(folder):
    '''
    Get the sum over the inputs of the inner product of grad_old and grad_new
    '''
    global time_acc
    start = time.time()
    gamma = 0
    grad_new_files = sorted(glob.glob(folder + '/*.grad_new.npy'))
    grad_old_files = sorted(glob.glob(folder + '/*.grad_old.npy'))
    for grad_new_file, grad_old_file in zip(grad_new_files, grad_old_files):
        grad_new = load(grad_new_file)
        grad_old = load(grad_old_file)
        gamma += (grad_new * grad_old).sum()
    end = time.time()
    time_acc['embeddings_get_grads_gamma'] += [end - start]
    return gamma

def embeddings_get_grads_max_d(folder, alpha):
    '''
    Get the max abs element of the direction over all input files
    '''
    global time_acc
    start = time.time()
    max_d = 0
    grad_d_files = sorted(glob.glob(folder + '/*.grad_d.npy'))
    for grad_d_file in grad_d_files:
        grad_d = load(grad_d_file)
        max_d = max(max_d, numpy.max(numpy.abs(alpha * grad_d)))
    end = time.time()
    time_acc['embeddings_get_grads_max_d'] += [end - start]
    return max_d

'''
Setters for local statistics
'''
def embeddings_set_grads_reset_d(folder):
    '''
    Reset the direction to be the negative of grad_new
    '''
    global time_acc
    start = time.time()
    input_files = sorted(glob.glob(folder + '/*.grad_new.npy'))
    for file_name in input_files:
        grads = load(file_name)
        # Save the direction as the negative grad
        new_file = splitext(splitext(file_name)[0])[0] + '.grad_d.npy'
        save(new_file, -1 * grads)
    end = time.time()
    time_acc['embeddings_set_grads_reset_d'] += [end - start]

def embeddings_set_grads_update_d(folder, gamma):
    '''
    Update the value of the direction for each input to be gamma (given) times the old direction
    minus grad_new
    '''
    global time_acc
    start = time.time()
    grad_new_files = sorted(glob.glob(folder + '/*.grad_new.npy'))
    grad_d_files = sorted(glob.glob(folder + '/*.grad_d.npy'))
    for grad_new_file, grad_d_file in zip(grad_new_files, grad_d_files):
        grad_new = load(grad_new_file)
        grad_d = load(grad_d_file)
        save(grad_d_file, gamma * grad_d - grad_new)
    end = time.time()
    time_acc['embeddings_set_grads_update_d'] += [end - start]

def embeddings_set_grads_update_X(folder, alpha):
    '''
    Update the value of the local embeddings and variances themselves to be X + alpha * direction
    '''
    global time_acc
    start = time.time()
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
        save(X_mu_file, X_mu + alpha * grad_d_X_mu)
        save(X_S_file, X_S + alpha * grad_d_X_S)
    end = time.time()
    time_acc['embeddings_set_grads_update_X'] += [end - start]

def embeddings_set_grads_update_grad_old(folder):
    '''
    Set grad_old to be grad_new
    '''
    global time_acc
    start = time.time()
    input_files = sorted(glob.glob(folder + '/*.grad_new.npy'))
    for file_name in input_files:
        grads = load(file_name)
        # Save grad old as latest grad new
        new_file = splitext(splitext(file_name)[0])[0] + '.grad_old.npy'
        save(new_file, grads)
    end = time.time()
    time_acc['embeddings_set_grads_update_grad_old'] += [end - start]

def embeddings_set_grads_update_grad_new(folder):
    '''
    Set grad_new to be grad_latest (a temp grad that keeps changing every evaluation)
    '''
    global time_acc
    start = time.time()
    input_files = sorted(glob.glob(folder + '/*.grad_latest.npy'))
    for file_name in input_files:
        grads = load(file_name)
        # Save grad old as latest grad new
        new_file = splitext(splitext(file_name)[0])[0] + '.grad_new.npy'
        save(new_file, grads)
    end = time.time()
    time_acc['embeddings_set_grads_update_grad_new'] += [end - start]

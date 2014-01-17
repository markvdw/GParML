'''
local_MapReduce.py
Implements the  map-reduce framework locally on a single machine and handles the file management for the different nodes.
'''
import tempfile
import os
from os.path import basename
from os.path import splitext
import numpy
import scipy
import multiprocessing
import glob
import collections
import itertools
'''
To Do: we currently have a bug where if an input file is one-dimensional, it will be read as
a single row array
'''
from numpy import genfromtxt
import kernels
import numpy.linalg as linalg
import partial_terms as pt
import supporting_functions as sp

'''
Initialise the inputs and embeddings
'''

def init(options):
    '''
    Init embeddings if 'load' was not given and we're not using fixed embeddings, 
    and work out N - the number of data points
    TODO: parallelise the initialisation
    '''
    input_files_names = os.listdir(options['input'] + '/')
    # N keeps track of the global number of inputs
    lengths = []
    ''' Find global number of inputs'''
    for file_name in input_files_names:
        # Count the number of lines in the input file
        length = 0
        input_file = options['input'] + '/' + file_name
        with open(input_file) as f:
            for line in f:
                if line.strip():
                    length += 1
        lengths += [length]
    options['N'] = sum(lengths)

    '''Initialise the embeddings and variances if needed'''
    if not options['fixed_embeddings'] and not options['load']:
        ''' 
        We're currently using PCA over the ENTIRE dataset locally since
        using PCA over subsets gives rise to rotation problems, and PPCA seems too noisy
        '''
        if options['init'] == 'PCA' or options['init'] == 'FA':
            # Load ALL data 
            input_file = options['input'] + '/' + input_files_names[0]
            Y = genfromtxt(input_file, delimiter=',')
            for file_name in input_files_names[1:]:
                input_file = options['input'] + '/' + file_name
                Y = numpy.concatenate((Y, genfromtxt(input_file, delimiter=',')))
            if options['init'] == 'PCA':
                # Collect statistics for PCA initialisation 
                X = sp.PCA(Y, options['Q'])
            elif options['init'] == 'FA':
                # Collect statistics for FA initialisation 
                [W, Psi, mean] = sp.FA_collect_statistics(Y, options, input_files_names)
        elif options['init'] == 'PPCA':
            # Collect statistics for PPCA initialisation 
            [W, sigma2, mean] = sp.PPCA_collect_statistics(options, input_files_names)

        for i, file_name in enumerate(input_files_names):
            input_file = options['input'] + '/' + file_name
            embedding_name = options['embeddings'] + '/' + file_name + '.embedding.npy'
            embedding_variance_name = options['embeddings'] + '/' + file_name + '.variance.npy'
            print 'Creating ' + embedding_name + ' with ' + str(lengths[i]) + ' points'
            remove(embedding_name)
            if options['init'] == 'PCA':
                save(embedding_name, X[sum(lengths[:i]):sum(lengths[:i])+lengths[i], :])
            elif options['init'] == 'PPCA':
                save(embedding_name, sp.PPCA(options, input_file, W, sigma2, mean))
            elif options['init'] == 'FA':
                save(embedding_name, sp.FA(options, input_file, W, Psi, mean))
            elif options['init'] == 'random':
                save(embedding_name, scipy.randn(lengths[i], options['Q']))
            print 'Creating ' + embedding_variance_name + ' with ' + str(lengths[i]) + ' points'
            # Initialise variance of data
            remove(embedding_variance_name)
            save(embedding_variance_name,
                sp.transformVar_back(numpy.clip(numpy.ones((lengths[i], options['Q'])) * 0.5
                            + 0.01 * scipy.randn(lengths[i], options['Q']),
                    0.001, 1)))
    if options['fixed_embeddings']:
        for i, file_name in enumerate(input_files_names):
            embedding_name = options['embeddings'] + '/' + file_name + '.embedding.npy'
            embedding_variance_name = options['embeddings'] + '/' + file_name + '.variance.npy'
            # If we are using fixed embeddings (i.e. doing sparse GPs)
            if not exists(embedding_name):
                raise Exception('No embedding file ' + embedding_name)
            print 'Creating ' + embedding_variance_name
            # Initialise variance of data
            save(embedding_variance_name, numpy.zeros((lengths[i], options['Q'])))
    return options



'''
Statistics Map-Reduce functions:
'''

def statistics_MR(options):
    '''
    Gets as input options and statistics to use in accumulation; returns as output partial sums. Writes files to /tmp/ to pass information between different nodes.
    '''
    input_files = sorted(glob.glob(options['input'] + '/*'))
    # Send both input_file_name and options to each mapper
    arguments = zip(input_files,itertools.repeat(options))
    pool = multiprocessing.Pool(len(input_files))
    map_responses = pool.map(statistics_mapper, arguments)
    pool.close()
    pool.join()
    # Code to debug locally because the trace from within the pool is not informative
    #map_responses = []
    #for arg in arguments:
    #    map_responses.append(statistics_mapper(arg))
    
    partitioned_data = partition(itertools.chain(*map_responses))
    
    arguments = zip(partitioned_data,itertools.repeat(options))
    pool = multiprocessing.Pool(len(input_files))
    reduced_values = pool.map(statistics_reducer, arguments)
    pool.close()
    pool.join()
    #reduced_values = []
    #for arg in arguments:
    #    reduced_values.append(statistics_reducer(arg))
    
    return reduced_values

def partition(mapped_values):
    '''
    Organize the mapped values by their key.
    Returns an unsorted sequence of tuples with a key and a sequence of values.
    '''
    partitioned_data = collections.defaultdict(list)
    for key, value in mapped_values:
        partitioned_data[key].append(value)
    return partitioned_data.items()

def statistics_mapper((input_file_name, options)):
    '''
    Maps inputs to temp files returning a dictionary of statistics and file names containing them
    '''
    # Load global statistics
    global_statistics = {}
    for key in options['global_statistics_names']:
        file_name = options['statistics'] + '/global_statistics_' + key + '_' + str(options['i']) + '.npy'
        global_statistics[key] = load(file_name)

    # Load inputs and embeddings
    embedding_name = options['embeddings'] + '/' + basename(input_file_name) + '.embedding.npy'
    embedding_variance_name = options['embeddings'] + '/' + basename(input_file_name) + '.variance.npy'
    Y = genfromtxt(input_file_name, delimiter=',')# os.path.join(INPUT_DIR,filename)
    X_mu = load(embedding_name)
    X_S = load(embedding_variance_name)

    if not options['fixed_embeddings']:
        # Check for existence of local grad info
        local_direction_name = options['embeddings'] + '/' + basename(input_file_name) + '.grad_d.npy'
        if exists(local_direction_name) and options['step_size'] != 0:
            d = load(local_direction_name)
            d_X_mu = d[0]
            d_X_S = d[1]
            X_mu += d_X_mu * options['step_size']
            X_S += d_X_S * options['step_size']
        
        # Transform the parameters that have to be positive to be positive
        X_S = sp.transformVar(X_S)

    #print X_mu
    #print X_S

    # Calculate partial statistics...
    partial_terms = load_partial_terms(options, global_statistics)
    # Load cached matrices
    load_cache(options, partial_terms)

    partial_terms.set_data(Y, X_mu, X_S, is_set_statistics=True)

    terms = partial_terms.get_local_statistics()
    accumulated_statistics = {
        'sum_YYT' : terms['sum_YYT'],
        'sum_exp_K_ii' : terms['sum_exp_K_ii'],
        'sum_exp_K_mi_K_im' : terms['sum_exp_K_mi_K_im'],
        'sum_exp_K_miY' : terms['exp_K_miY'],
        'sum_KL' : terms['KL'],
        'sum_d_exp_K_miY_d_Z' : partial_terms.dexp_K_miY_dZ(),
        'sum_d_exp_K_mi_K_im_d_Z' : partial_terms.dexp_K_mi_K_im_dZ(),
        'sum_d_exp_K_miY_d_alpha' : partial_terms.dexp_K_miY_dalpha(),
        'sum_d_exp_K_mi_K_im_d_alpha' : partial_terms.dexp_K_mi_K_im_dalpha(),
        'sum_d_exp_K_ii_d_sf2' : partial_terms.dexp_K_ii_dsf2(),
        'sum_d_exp_K_miY_d_sf2' : partial_terms.dexp_K_miY_dsf2(),
        'sum_d_exp_K_mi_K_im_d_sf2' : partial_terms.dexp_K_mi_K_im_dsf2()
    }

    file_names_list = []
    for key in accumulated_statistics.keys():
        file_name = tempfile.mktemp(dir=options['tmp'], suffix='.npy')
        save(file_name, accumulated_statistics[key])
        file_names_list.append((key, file_name))
    return file_names_list

def statistics_reducer((source_file_name_list, options)):
    '''
    Reduces a list of file names (of a single statistic) to a single file by summing them and deleting the old files
    '''
    statistic = source_file_name_list[0]
    files_names = source_file_name_list[1]

    target_file_name = options['statistics'] + '/accumulated_statistics_' + statistic + '_' + str(options['i']) + '.npy'
    if len(files_names) == 1:
        # Move to the statistics folder
        os.rename(files_names[0], target_file_name)
    else:
        accumulated_statistics = load(files_names[0])
        remove(files_names[0])
        for file_name in files_names[1:]:
            accumulated_statistics += load(file_name)
            remove(file_name)
        save(target_file_name, accumulated_statistics)

    return (statistic, target_file_name)


'''
Embeddings Map-Reduce functions:
'''

def embeddings_MR(options):
    '''
    Gets as input options and statistics to use in embeddings optimisation. Writes files to TMP 
    (given in options) to pass information between different nodes. This function is only called 
    if we are optimising the embeddings, so no further checks are made.
    '''

    input_files = sorted(glob.glob(options['input'] + '/*'))
    # Send options to each mapper
    arguments = zip(input_files,itertools.repeat(options))
    pool = multiprocessing.Pool(len(input_files))
    pool.map(embeddings_mapper, arguments)
    pool.close()
    pool.join()
    # Code to debug locally because the trace from within the pool is not informative
    #for arg in arguments:
    #    embeddings_mapper(arg)
    
def embeddings_mapper((input_file_name, options)):
    global_statistics = {}
    accumulated_statistics = {}

    for key in options['global_statistics_names']:
        file_name = options['statistics'] + '/global_statistics_' + key + '_' + str(options['i']) + '.npy'
        global_statistics[key] = load(file_name)
    for key in options['accumulated_statistics_names']:
        file_name = options['statistics'] + '/accumulated_statistics_' + key + '_' + str(options['i']) + '.npy'
        accumulated_statistics[key] = load(file_name)

    # Load inputs and embeddings
    embedding_name = options['embeddings'] + '/' + basename(input_file_name) + '.embedding.npy'
    embedding_variance_name = options['embeddings'] + '/' + basename(input_file_name) + '.variance.npy'
    Y = genfromtxt(input_file_name, delimiter=',')# os.path.join(INPUT_DIR,filename)
    X_mu = load(embedding_name)
    X_S_orig = load(embedding_variance_name)

    # Check for existence of local grad info
    local_direction_name = options['embeddings'] + '/' + basename(input_file_name) + '.grad_d.npy'
    if exists(local_direction_name) and options['step_size'] != 0:
        d = load(local_direction_name)
        d_X_mu = d[0]
        d_X_S = d[1]
        X_mu += d_X_mu * options['step_size']
        X_S_orig += d_X_S * options['step_size']
    
    # Transform the parameters that have to be positive to be positive
    X_S = sp.transformVar(X_S_orig)

    # Calculate partial statistics...
    partial_terms = load_partial_terms(options, global_statistics)
    # Load cached matrices
    load_cache(options, partial_terms)

    partial_terms.set_data(Y, X_mu, X_S, is_set_statistics=False)

    partial_terms.set_local_statistics(accumulated_statistics['sum_YYT'],
        accumulated_statistics['sum_exp_K_mi_K_im'],
        accumulated_statistics['sum_exp_K_miY'],
        accumulated_statistics['sum_exp_K_ii'],
        accumulated_statistics['sum_KL'])

    # Save the latest gradient
    grad_X_mu = partial_terms.grad_X_mu()
    grad_X_S = partial_terms.grad_X_S() * sp.transformVar_grad(X_S_orig)
    local_latest_grad_name = options['embeddings'] + '/' + basename(input_file_name) + '.grad_latest.npy'
    save(local_latest_grad_name, -1 * numpy.array([grad_X_mu, grad_X_S]))




'''
A bunch of support functions used for SCG optimisation. They are here because they depend on the 
parallel implementation framework, but may change for other optimisers.
'''

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

'''
Getters for local statistics
'''
def embeddings_get_grads_mu(folder):
    '''
    Get the sum over the inputs of the inner product of the direction and grad_new
    '''
    mu = 0
    grad_new_files = sorted(glob.glob(folder + '/*.grad_new.npy'))
    grad_d_files = sorted(glob.glob(folder + '/*.grad_d.npy'))
    for grad_new_file, grad_d_file in zip(grad_new_files, grad_d_files):
        grad_new = load(grad_new_file)
        grad_d = load(grad_d_file)
        mu += (grad_new * grad_d).sum()
    return mu

def embeddings_get_grads_kappa(folder):
    '''
    Get the sum over the inputs of the inner product of the direction with itself
    '''
    kappa = 0
    grad_d_files = sorted(glob.glob(folder + '/*.grad_d.npy'))
    for grad_d_file in grad_d_files:
        grad_d = load(grad_d_file)
        kappa += (grad_d * grad_d).sum()
    return kappa

def embeddings_get_grads_theta(folder):
    '''
    Get the sum over the inputs of the inner product of the direction and grad_latest
    '''
    theta = 0
    grad_new_files = sorted(glob.glob(folder + '/*.grad_new.npy'))
    grad_latest_files = sorted(glob.glob(folder + '/*.grad_latest.npy'))
    grad_d_files = sorted(glob.glob(folder + '/*.grad_d.npy'))
    for grad_latest_file, grad_d_file, grad_new_file in zip(grad_latest_files, grad_d_files, grad_new_files):
        grad_latest = load(grad_latest_file)
        grad_new = load(grad_new_file)
        grad_d = load(grad_d_file)
        theta += (grad_d * (grad_latest - grad_new)).sum()
    return theta

def embeddings_get_grads_current_grad(folder):
    '''
    Get the sum over the inputs of the inner product of grad_new with itself
    '''
    current_grad = 0
    grad_new_files = sorted(glob.glob(folder + '/*.grad_new.npy'))
    for grad_new_file in grad_new_files:
        grad_new = load(grad_new_file)
        current_grad += (grad_new * grad_new).sum()
    return current_grad

def embeddings_get_grads_gamma(folder):
    '''
    Get the sum over the inputs of the inner product of grad_old and grad_new
    '''
    gamma = 0
    grad_new_files = sorted(glob.glob(folder + '/*.grad_new.npy'))
    grad_old_files = sorted(glob.glob(folder + '/*.grad_old.npy'))
    for grad_new_file, grad_old_file in zip(grad_new_files, grad_old_files):
        grad_new = load(grad_new_file)
        grad_old = load(grad_old_file)
        gamma += (grad_new * grad_old).sum()
    return gamma

def embeddings_get_grads_max_d(folder, alpha):
    '''
    Get the max abs element of the direction over all input files
    '''
    max_d = 0
    grad_d_files = sorted(glob.glob(folder + '/*.grad_d.npy'))
    for grad_d_file in grad_d_files:
        grad_d = load(grad_d_file)
        max_d = max(max_d, numpy.max(numpy.abs(alpha * grad_d)))
    return max_d

'''
Setters for local statistics
'''
def embeddings_set_grads_reset_d(folder):
    '''
    Reset the direction to be the negative of grad_new
    '''
    input_files = sorted(glob.glob(folder + '/*.grad_new.npy'))
    for file_name in input_files:
        grads = load(file_name)
        # Save the direction as the negative grad
        new_file = splitext(splitext(file_name)[0])[0] + '.grad_d.npy'
        save(new_file, -1 * grads)

def embeddings_set_grads_update_d(folder, gamma):
    '''
    Update the value of the direction for each input to be gamma (given) times the old direction
    minus grad_new
    '''
    grad_new_files = sorted(glob.glob(folder + '/*.grad_new.npy'))
    grad_d_files = sorted(glob.glob(folder + '/*.grad_d.npy'))
    for grad_new_file, grad_d_file in zip(grad_new_files, grad_d_files):
        grad_new = load(grad_new_file)
        grad_d = load(grad_d_file)
        save(grad_d_file, gamma * grad_d - grad_new)

def embeddings_set_grads_update_X(folder, alpha):
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
        save(X_mu_file, X_mu + alpha * grad_d_X_mu)
        save(X_S_file, X_S + alpha * grad_d_X_S)

def embeddings_set_grads_update_grad_old(folder):
    '''
    Set grad_old to be grad_new
    '''
    input_files = sorted(glob.glob(folder + '/*.grad_new.npy'))
    for file_name in input_files:
        grads = load(file_name)
        # Save grad old as latest grad new
        new_file = splitext(splitext(file_name)[0])[0] + '.grad_old.npy'
        save(new_file, grads)

def embeddings_set_grads_update_grad_new(folder):
    '''
    Set grad_new to be grad_latest (a temp grad that keeps changing every evaluation)
    '''
    input_files = sorted(glob.glob(folder + '/*.grad_latest.npy'))
    for file_name in input_files:
        grads = load(file_name)
        # Save grad old as latest grad new
        new_file = splitext(splitext(file_name)[0])[0] + '.grad_new.npy'
        save(new_file, grads)





def save(file_name, obj):
    scipy.save(file_name, obj)

def load(file_name):
    return scipy.load(file_name)

def exists(file_name):
    return os.path.exists(file_name)

def remove(file_name):
    if exists(file_name):
        os.remove(file_name)

def cache(options, global_statistics):
    '''
    To Do: support Q=1 case where alpha squeeze is scalar
    '''
    # We can calculate the global statistics once for all nodes
    kernel = kernels.rbf(options['Q'], sf=float(global_statistics['sf2']**0.5), ard=numpy.squeeze(global_statistics['alpha'])**-0.5)
    Kmm = kernel.K(global_statistics['Z'])
    file_name = options['statistics'] + '/cache_Kmm_' + str(options['i']) + '.npy'
    save(file_name, Kmm)
    Kmm_inv = linalg.inv(Kmm)
    file_name = options['statistics'] + '/cache_Kmm_inv_' + str(options['i']) + '.npy'
    save(file_name, Kmm_inv)

def load_cache(options, partial_terms):
    file_name = options['statistics'] + '/cache_Kmm_' + str(options['i']) + '.npy'
    Kmm = load(file_name)
    file_name = options['statistics'] + '/cache_Kmm_inv_' + str(options['i']) + '.npy'
    Kmm_inv = load(file_name)
    partial_terms.set_global_statistics(Kmm, Kmm_inv)

def load_partial_terms(options, global_statistics):
    return pt.partial_terms(global_statistics['Z'],
                                float(global_statistics['sf2']),
                                numpy.squeeze(global_statistics['alpha']),
                                float(global_statistics['beta']),
                                options['M'], options['Q'],
                                options['N'], options['D'], update_global_statistics=False)
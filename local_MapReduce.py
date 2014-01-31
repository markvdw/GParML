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
import time
import collections
import itertools
import random
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
            if (len(Y.shape) == 1):
                Y = numpy.atleast_2d(Y).T
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


# Keep the dropped out nodes in a global variable to share between the different MRs
dropped_out_nodes = []
non_dropped_out_nodes = []

'''
Statistics Map-Reduce functions:
'''

def statistics_MR(options):
    '''
    Gets as input options and statistics to use in accumulation; returns as output partial sums. Writes files to /tmp/ to pass information between different nodes.
    '''
    global non_dropped_out_nodes, dropped_out_nodes
    input_files = sorted(glob.glob(options['input'] + '/*'))
    # Dropout drop_out_fraction of the nodes
    if 'drop_out_fraction' in options and options['drop_out_fraction'] > 0:
        drop_out = numpy.random.uniform(size=len(input_files)) < options['drop_out_fraction']
        dropped_out_nodes = numpy.arange(len(input_files))[drop_out]
        non_dropped_out_nodes = numpy.arange(len(input_files))[~drop_out]
        if len(non_dropped_out_nodes) == 0:
            print 'Warning: dropout fraction too high -- using at least one node'
            non_dropped_out_nodes = [random.randint(0, len(input_files) - 1)]
        input_files = [input_files[i] for i in non_dropped_out_nodes]

    # Send both input_file_name and options to each mapper
    arguments = zip(input_files,itertools.repeat(options))
    if (not (('local_no_pool' in options) and (options['local_no_pool']))):
        pool = multiprocessing.Pool(len(input_files))
        map_responses = pool.map(statistics_mapper, arguments)
        pool.close()
        pool.join()
    else:
        # Code to debug locally because the trace from within the pool is not informative
        map_responses = []
        for arg in arguments:
           map_responses.append(statistics_mapper(arg))

    # Extract the execution time from the map responses
    file_names_list = []
    statistics_mapper_time = []
    for map_response in map_responses:
        file_names_list +=  [map_response[0]]
        statistics_mapper_time += [map_response[1]]

    partitioned_data = partition(itertools.chain(*file_names_list))
    
    arguments = zip(partitioned_data,itertools.repeat(options))
    if (not (('local_no_pool' in options) and (options['local_no_pool']))):
        pool = multiprocessing.Pool(len(input_files))
        reduced_values = pool.map(statistics_reducer, arguments)
        pool.close()
        pool.join()
    else:
        reduced_values = []
        for arg in arguments:
            reduced_values.append(statistics_reducer(arg))

    # Extract the execution time from the reduced values
    file_names_list = []
    statistics_reducer_time = []
    for reduced_value in reduced_values:
        file_names_list +=  [reduced_value[0]]
        statistics_reducer_time += [reduced_value[1]]
    
    return file_names_list, statistics_mapper_time, statistics_reducer_time

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
    start = time.time()
    # Load global statistics
    global_statistics = {}
    for key in options['global_statistics_names']:
        file_name = options['statistics'] + '/global_statistics_' + key + '_' + str(options['i']) + '.npy'
        global_statistics[key] = load(file_name)

    # Load inputs and embeddings
    embedding_name = options['embeddings'] + '/' + basename(input_file_name) + '.embedding.npy'
    embedding_variance_name = options['embeddings'] + '/' + basename(input_file_name) + '.variance.npy'
    Y = genfromtxt(input_file_name, delimiter=',')# os.path.join(INPUT_DIR,filename)
    if (len(Y.shape) == 1):
        Y = numpy.atleast_2d(Y).T
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
    end = time.time()
    return file_names_list, end - start

def statistics_reducer((source_file_name_list, options)):
    '''
    Reduces a list of file names (of a single statistic) to a single file by summing them and deleting the old files
    '''
    global non_dropped_out_nodes, dropped_out_nodes
    start = time.time()
    statistic = source_file_name_list[0]
    files_names = source_file_name_list[1]

    target_file_name = options['statistics'] + '/accumulated_statistics_' + statistic + '_' + str(options['i']) + '.npy'
    if len(files_names) == 1:
        # Move to the statistics folder
        accumulated_statistics = load(files_names[0])
        if 'drop_out_fraction' in options and options['drop_out_fraction'] > 0:
            accumulated_statistics /= float(len(non_dropped_out_nodes)) / (len(non_dropped_out_nodes) + len(dropped_out_nodes))
        save(target_file_name, accumulated_statistics)
    else:
        accumulated_statistics = load(files_names[0])
        remove(files_names[0])
        for file_name in files_names[1:]:
            accumulated_statistics += load(file_name)
            remove(file_name)
        if 'drop_out_fraction' in options and options['drop_out_fraction'] > 0:
            accumulated_statistics /= float(len(non_dropped_out_nodes)) / (len(non_dropped_out_nodes) + len(dropped_out_nodes))
        save(target_file_name, accumulated_statistics)

    end = time.time()
    return (statistic, target_file_name), end - start


'''
Embeddings Map-Reduce functions:
'''

def embeddings_MR(options):
    '''
    Gets as input options and statistics to use in embeddings optimisation. Writes files to TMP 
    (given in options) to pass information between different nodes. This function is only called 
    if we are optimising the embeddings, so no further checks are made.
    '''
    global non_dropped_out_nodes
    input_files = sorted(glob.glob(options['input'] + '/*'))
    # Dropout drop_out_fraction of the nodes
    #if 'drop_out_fraction' in options and options['drop_out_fraction'] > 0:
    #    input_files = [input_files[i] for i in non_dropped_out_nodes]

    # Send options to each mapper
    arguments = zip(input_files,itertools.repeat(options))
    if (not (('local_no_pool' in options) and (options['local_no_pool']))):
        pool = multiprocessing.Pool(len(input_files))
        embeddings_MR_time = pool.map(embeddings_mapper, arguments)
        pool.close()
        pool.join()
    else:
        # Code to debug locally because the trace from within the pool is not informative
        for arg in arguments:
            embeddings_mapper(arg)

    return embeddings_MR_time
    
def embeddings_mapper((input_file_name, options)):
    start = time.time()
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
    if (len(Y.shape) == 1):
        Y = numpy.atleast_2d(Y).T
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

    end = time.time()
    return end - start


'''
Supporting functions
'''

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

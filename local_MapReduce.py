'''
local_MapReduce.py
Implements the  map-reduce framework locally on a single machine and handles the file management for the different nodes.
'''
import tempfile
import os
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
        if options['init'] == 'PCA':
            # Load ALL data for PCA
            input_file = options['input'] + '/' + input_files_names[0]
            Y = genfromtxt(input_file, delimiter=',')
            for file_name in input_files_names[1:]:
                input_file = options['input'] + '/' + file_name
                Y = numpy.concatenate((Y, genfromtxt(input_file, delimiter=',')))
            X = PCA(Y, options['Q'])
        elif options['init'] == 'PPCA':
            # Collect statistics for PPCA initialisation 
            [W, sigma2, mean] = PPCA_collect_statistics(options, input_files_names)
        elif options['init'] == 'FA':
            # Load ALL data for PCA
            input_file = options['input'] + '/' + input_files_names[0]
            Y = genfromtxt(input_file, delimiter=',')
            for file_name in input_files_names[1:]:
                input_file = options['input'] + '/' + file_name
                Y = numpy.concatenate((Y, genfromtxt(input_file, delimiter=',')))
            # Collect statistics for FA initialisation 
            [W, Psi, mean] = FA_collect_statistics(Y, options, input_files_names)

        for i, file_name in enumerate(input_files_names):
            input_file = options['input'] + '/' + file_name
            embedding_name = options['embeddings'] + '/' + file_name + '.embedding.npy'
            embedding_variance_name = options['embeddings'] + '/' + file_name + '.variance.npy'
            print 'Creating ' + embedding_name + ' with ' + str(lengths[i]) + ' points'
            remove(embedding_name)
            if options['init'] == 'PCA':
                save(embedding_name, X[sum(lengths[:i]):sum(lengths[:i])+lengths[i], :])
            elif options['init'] == 'PPCA':
                save(embedding_name, PPCA(options, input_file, W, sigma2, mean))
            elif options['init'] == 'FA':
                save(embedding_name, FA(options, input_file, W, Psi, mean))
            elif options['init'] == 'random':
                save(embedding_name, scipy.randn(lengths[i], options['Q']))
            print 'Creating ' + embedding_variance_name + ' with ' + str(lengths[i]) + ' points'
            # Initialise variance of data
            remove(embedding_variance_name)
            save(embedding_variance_name,
                numpy.clip(numpy.ones((lengths[i], options['Q'])) * 0.5
                            + 0.01 * scipy.randn(lengths[i], options['Q']),
                    0.001, 1))
    if options['fixed_embeddings']:
        for i, file_name in enumerate(input_files_names):
            embedding_name = options['embeddings'] + '/' + file_name + '.embedding.npy'
            embedding_variance_name = options['embeddings'] + '/' + file_name + '.variance.npy'
            # If we are using fixed embeddings (i.e. doing sparse GPs)
            if not os.path.exists(embedding_name):
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
    input_files = glob.glob(options['input'] + '/*')
    pool = multiprocessing.Pool(len(input_files))
    # Send both input_file_name and options to each mapper
    arguments = zip(input_files,itertools.repeat(options))
    map_responses = pool.map(statistics_mapper, arguments)
    # Code to debug locally because the trace from within the pool is not informative
    #map_responses = []
    #for arg in arguments:
    #    map_responses.append(statistics_mapper(arg))
    partitioned_data = partition(itertools.chain(*map_responses))
    arguments = zip(partitioned_data,itertools.repeat(options))
    reduced_values = pool.map(statistics_reducer, arguments)
    #reduced_values = []
    #for arg in arguments:
    #    reduced_values.append(statistics_reducer(arg))
    pool.close()
    pool.join()
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
    embedding_name = options['embeddings'] + '/' + os.path.basename(input_file_name) + '.embedding.npy'
    embedding_variance_name = options['embeddings'] + '/' + os.path.basename(input_file_name) + '.variance.npy'
    Y = genfromtxt(input_file_name, delimiter=',')# os.path.join(INPUT_DIR,filename)
    X_mu = load(embedding_name)
    X_S = load(embedding_variance_name)

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
    Gets as input options and statistics to use in embeddings optimisation. Writes files to TMP (given in options) to pass information between different nodes.
    '''

    input_files = glob.glob(options['input'] + '/*')
    pool = multiprocessing.Pool(len(input_files))
    # Send options to each mapper
    arguments = zip(input_files,itertools.repeat(options))
    pool.map(embeddings_mapper, arguments)
    # Code to debug locally because the trace from within the pool is not informative
    #map_responses = []
    #for arg in arguments:
    #    map_responses.append(embeddings_mapper(arg))
    pool.close()
    return pool

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
    embedding_name = options['embeddings'] + '/' + os.path.basename(input_file_name) + '.embedding.npy'
    embedding_variance_name = options['embeddings'] + '/' + os.path.basename(input_file_name) + '.variance.npy'
    Y = genfromtxt(input_file_name, delimiter=',')# os.path.join(INPUT_DIR,filename)
    X_mu = load(embedding_name)
    X_S = load(embedding_variance_name)

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

    # Actual optimisation of the embeddings
    (new_X_mu, new_X_S) = partial_terms.local_optimisation()
    save(embedding_name, new_X_mu)
    save(embedding_variance_name, new_X_S)



def embeddings_watcher(options, pool):
    pool.join()



'''
Supporting functions
'''

def linear_dim_reduction(options, input_files_names):
    mean = numpy.zeros((options['D']))
    # Collect mean of data over all files
    for file_name in input_files_names:
        input_file = options['input'] + '/' + file_name
        Y = genfromtxt(input_file, delimiter=',')
        mean += Y.sum(axis=0)
    mean /= options['N']

    S = numpy.zeros((options['D'], options['D']))
    # Collect covariance of data over all files
    for file_name in input_files_names:
        input_file = options['input'] + '/' + file_name
        Y = genfromtxt(input_file, delimiter=',')
        S += (Y.T - mean[:, None]).dot(Y - mean[None, :])
    S /= options['N']

    return [mean, S]

def PPCA_collect_statistics(options, input_files_names):
    '''
    Builds the PPCA matrices. Goes over the input files one by one not loading all together into memory, 
    thus having a small memory footprint
    '''
    [mean, S] = linear_dim_reduction(options, input_files_names)
    
    W = numpy.random.randn(options['D'], options['Q'])
    sigma2 = numpy.random.randn()

    ll_old = 0
    ll_new = numpy.inf
    while numpy.abs(ll_new - ll_old) > 1e-6:
        Minv = numpy.linalg.inv(sigma2 * numpy.eye(options['Q']) + W.T.dot(W))
        newW = S.dot(W.dot(numpy.linalg.inv(sigma2 * numpy.eye(options['Q']) + Minv.dot(W.T.dot(S.dot(W))))))
        sigma2 = 1.0 / options['D'] * numpy.trace(S - S.dot(W.dot(Minv.dot(newW.T))))
        W = newW
        # Calculate log-marginal likelihood for stopping criteria
        C = sigma2 * numpy.eye(options['D']) + W.dot(W.T)
        ll_old = ll_new
        ll_new = numpy.linalg.slogdet(C)[1] + numpy.trace(numpy.linalg.inv(C).dot(S))

    return [W, sigma2, mean]

def FA_collect_statistics(Y, options, input_files_names):
    '''
    Builds the FA matrices. Goes over the input files all together, thus having a large memory footprint.
    Can easily be re-written to go over the files one by one.
    '''
    [mean, S] = linear_dim_reduction(options, input_files_names)
    
    W = numpy.random.randn(options['D'], options['Q'])
    Psi = numpy.random.randn(options['D'])

    ll_old = 0
    ll_new = numpy.inf
    while numpy.abs(ll_new - ll_old) > 1e-6:
        Psi_inv = numpy.linalg.inv(numpy.diag(Psi))
        G = numpy.linalg.inv(numpy.eye(options['Q']) + W.T.dot(Psi_inv).dot(W))
        exp_z = G.dot(W.T).dot(Psi_inv).dot((Y - mean[None, :]).T).T
        exp_zz = G[None, :, :] + exp_z[:, None, :] * exp_z[:, :, None]
        W = ((Y - mean[None, :])[:, :, None] * exp_z[:, None, :]).sum(axis=0).dot(numpy.linalg.inv(exp_zz.sum(axis=0)))
        Psi = numpy.diag(S - W.dot(1.0 / Y.shape[0] * (exp_z[:, :, None] * (Y - mean[None, :])[:, None, :]).sum(axis=0)))
        # Calculate log-marginal likelihood for stopping criteria
        C = numpy.diag(Psi) + W.dot(W.T)
        ll_old = ll_new
        ll_new = numpy.linalg.slogdet(C)[1] + numpy.trace(numpy.linalg.inv(C).dot(S))

    return [W, Psi, mean]

def PPCA(options, Y_name, W, sigma2, mean):
    Y = genfromtxt(Y_name, delimiter=',')
    Minv = numpy.linalg.inv(sigma2 * numpy.eye(options['Q']) + W.T.dot(W))
    X = (Minv.dot(W.T).dot(Y.T - mean[:, None])).T
    v = X.std(axis=0)
    X /= v
    return X

def FA(options, Y_name, W, Psi, mean):
    Y = genfromtxt(Y_name, delimiter=',')
    Psi_inv = numpy.linalg.inv(numpy.diag(Psi))
    Sigma = numpy.linalg.inv(numpy.eye(options['Q']) + W.T.dot(Psi_inv).dot(W))
    X = (Sigma.dot(W.T).dot(Psi_inv).dot(Y.T - mean[:, None])).T
    v = X.std(axis=0)
    X /= v
    return X

def PCA(Y, input_dim):
    """
    Principal component analysis: maximum likelihood solution by SVD
    Adapted from GPy.util.linalg
    Arguments
    ---------
    :param Y: NxD np.array of data
    :param input_dim: int, dimension of projection

    Returns
    -------
    :rval X: - Nxinput_dim np.array of dimensionality reduced data
    W - input_dimxD mapping from X to Y
    """
    Z = numpy.linalg.svd(Y - Y.mean(axis=0), full_matrices=False)
    [X, W] = [Z[0][:, 0:input_dim], numpy.dot(numpy.diag(Z[1]), Z[2]).T[:, 0:input_dim]]
    v = X.std(axis=0)
    X /= v;
    W *= v;
    return X

def save(file_name, obj):
    scipy.save(file_name, obj)

def load(file_name):
    return scipy.load(file_name)

def remove(file_name):
    if os.path.exists(file_name):
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
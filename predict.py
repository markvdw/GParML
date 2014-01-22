import numpy
import glob
import os
import scipy
import scipy.spatial
import local_MapReduce
import parallel_GPLVM
from scg_adapted import SCG_adapted
import supporting_functions as sp

''' Some globals which are shared among all functions '''
options = {}
flat_statistics_bounds = []
shape = ()
global_statistics = {}
Y_test = []
accumulated_statistics = []

def test(options_, Y_test_, mask=None, is_random_init=False, random_iterations=100, random_restarts=100):
    '''
    Return mean and variance for test points given a trained model in 'options'.
    Very quick for random init, O(N) otherwise but with very small coefficient. Note that options['N'] has to be populated.
    '''
    global options, flat_statistics_bounds, shape, global_statistics, Y_test, accumulated_statistics
    options = options_.copy()
    Y_test = Y_test_
    ''' Load some stuff '''
    shape = (Y_test.shape[0], options['Q'])
    options['load'] = True
    options, global_statistics = parallel_GPLVM.init_statistics(local_MapReduce, options)
    options['i'] = 'f'
    accumulated_statistics = {}
    for key in options['accumulated_statistics_names']:
        file_name = options['statistics'] + '/accumulated_statistics_' + key + '_' + str(options['i']) + '.npy'
        accumulated_statistics[key] = local_MapReduce.load(file_name)

    ''' Init the mean and variance '''
    if is_random_init:
        index = numpy.random.randint(options['M'])
        Z = global_statistics['Z']
        X_mu = numpy.atleast_2d(Z[index])
        #X_mu = scipy.randn(Y_test.shape[0], options['Q'])
    else:
        #random_iterations = 100
        random_restarts = 0
        if mask == None:
            mask = range(Y_test.shape[1])
        #print Y_test[:, mask]
        file_names = glob.glob(options['input'] + '/*')
        Y_dists = [numpy.inf for i in xrange(shape[0])]
        X_mu = numpy.zeros(shape)
        for file_name in file_names:
            Y = numpy.genfromtxt(file_name, delimiter=',')
            embedding_name = options['embeddings'] + '/' + os.path.basename(file_name) + '.embedding.npy'
            X = scipy.load(embedding_name)

            tree = scipy.spatial.cKDTree(Y[:, mask],leafsize=100)
            (dist, ind) = tree.query(Y_test[:, mask], k=1, distance_upper_bound=6)

            for i in xrange(X_mu.shape[0]):
                if dist[i] < Y_dists[i]:
                    Y_dists[i] = dist[i]
                    X_mu[i, :] = X[ind[i], :]
    #print 'Initial X_mu'
    #print X_mu
    X_S = numpy.clip(numpy.ones(X_mu.shape) * 0.5 + 0.01 * scipy.randn(X_mu.shape[0], X_mu.shape[1]),
                0.001, 1)

    ''' Optimisation stuff '''
    flat_statistics_bounds = [(None, None) for i in range(numpy.prod(shape))]
    flat_statistics_bounds += [(0, None) for i in range(numpy.prod(shape))]

    ''' Run the optimisation itself '''
    x0 = numpy.concatenate((X_mu.flatten(), X_S.flatten()))
    x0 = numpy.array([sp.transform_back(b, x) for b, x in zip(flat_statistics_bounds, x0)])

    # We're setting fixed_embeddings to true because we want to optimise only over the 'globals' which are now the embeddings
    x = SCG_adapted(likelihood_and_gradient, x0, options['embeddings'], fixed_embeddings=True, display=False, maxiters=random_iterations)
    flat_array = x[0]
    flat_array_transformed = numpy.array([sp.transform(b, y) for b, y in zip(flat_statistics_bounds, flat_array)])
    X_mu = flat_array_transformed[:len(flat_array_transformed)/2].reshape(X_mu.shape)
    X_S = flat_array_transformed[len(flat_array_transformed)/2:].reshape(X_S.shape)
    likelihood = -x[1][-1]
    best_results = [X_mu, X_S, likelihood]
    #print 'best_results'
    #print best_results

    if is_random_init:
        ''' We often need many restarts because there are many many many local optima '''
        for i in xrange(random_restarts):
            index = numpy.random.randint(options['M'])
            Z = global_statistics['Z']
            X_mu = numpy.atleast_2d(Z[index])
            x0 = numpy.concatenate((X_mu.flatten(), X_S.flatten()))
            x0 = numpy.array([sp.transform_back(b, x) for b, x in zip(flat_statistics_bounds, x0)])
            # We're setting fixed_embeddings to true because we want to optimise only over the 'globals' which are now the embeddings
            x = SCG_adapted(likelihood_and_gradient, x0, options['embeddings'], fixed_embeddings=True, display=False, maxiters=random_iterations)
            if -x[1][-1] > best_results[-1]:
                flat_array = x[0]
                flat_array_transformed = numpy.array([sp.transform(b, y) for b, y in zip(flat_statistics_bounds, flat_array)])
                X_mu = flat_array_transformed[:len(flat_array_transformed)/2].reshape(X_mu.shape)
                X_S = flat_array_transformed[len(flat_array_transformed)/2:].reshape(X_S.shape)
                best_results = [X_mu, X_S, -x[1][-1]]
                #print 'best_results'
                #print best_results

    ''' BUG IN THE IMPLEMENTATION: set_data changes sum_YYT '''

    return best_results


''' Optimisation support function '''

def likelihood_and_gradient(flat_array, iteration=0, step_size=0):
    global options, flat_statistics_bounds, shape, global_statistics, Y_test, accumulated_statistics
    flat_array_transformed = numpy.array([sp.transform(b, x) for b, x in zip(flat_statistics_bounds, flat_array)])
    X_mu_ = flat_array_transformed[:len(flat_array_transformed)/2].reshape(shape)
    X_S_ = flat_array_transformed[len(flat_array_transformed)/2:].reshape(shape)
    # Calculate partial statistics for optimisation...
    partial_terms = local_MapReduce.load_partial_terms(options, global_statistics)
    # Load cached matrices
    local_MapReduce.load_cache(options, partial_terms)
    # We need to set the data for the optimisation
    partial_terms.set_data(Y_test, X_mu_, X_S_, is_set_statistics=True)
    # Get local statistics for new data points
    new_local_statistics = partial_terms.get_local_statistics()
    # ... but override local stats with global ones
    partial_terms.set_local_statistics(accumulated_statistics['sum_YYT'] + new_local_statistics['sum_YYT'],
        accumulated_statistics['sum_exp_K_mi_K_im'] + new_local_statistics['sum_exp_K_mi_K_im'],
        accumulated_statistics['sum_exp_K_miY'] + new_local_statistics['exp_K_miY'],
        accumulated_statistics['sum_exp_K_ii'] + new_local_statistics['sum_exp_K_ii'],
        accumulated_statistics['sum_KL'] + new_local_statistics['KL'])
    likelihood = partial_terms.logmarglik()
    gradient = numpy.concatenate((partial_terms.grad_X_mu().flatten(), partial_terms.grad_X_S().flatten()))
    #print 'likelihood'
    #print likelihood
    #print 'gradient'
    #print gradient
    # Transform the gradient parameters that have to be positive by multiplying
    # them by the gradient of the transform f:  g(f(x))' = g'(f(x))f'(x)
    gradient = numpy.array([g * sp.transform_grad(b, x) for b, x, g in zip(flat_statistics_bounds, flat_array, gradient)])
    return -1 * likelihood, -1 * gradient


''' If ran as main do some debugging '''

def main():
    # Parameters to adjust
    Q = 2
    num_inducing = 10

    options = {}
    options['input'] = './easydata1k/inputs/'
    options['embeddings'] = './easydata1k/embeddings/'
    options['parallel'] = 'local'
    options['statistics'] = './easydata1k/tmp'
    options['tmp'] = './easydata1k/tmp'
    options['M'] = num_inducing
    options['Q'] = 2
    options['D'] = 3
    options['keep'] = False
    options['init'] = 'PCA'
    options['fixed_beta'] = False
    options['optimiser'] = 'SCG_adapted'
    options['fixed_embeddings'] = False
    options['iterations'] = 0
    options['load'] = False

    options['N'] = 1000
    Y_test = numpy.array([[0,0,0]])
    #a = test(options, Y_test, is_random_init=True, random_iterations=10, random_restarts=10)
    a = test(options, Y_test)
    print a

if __name__ == '__main__':
    main()
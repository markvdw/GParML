import sys
import os.path
import scipy
import numpy
import partial_terms as pt
import kernels
from numpy.linalg.linalg import LinAlgError
from scg import SCG

D = 3
Q  = 2
M = 10
N = {}
Y = {}
X_mu = {}
X_S = {}
Kmm = {}
Kmm_inv = {}
accumulated_statistics = {}
flat_global_statistics_bounds = {}
global_statistics_names = {}
fix_beta = {}

def main():
    global N, Y, X_mu, X_S, flat_global_statistics_bounds, fix_beta, global_statistics_names

    iterations = 20

    # Load data
    Y = numpy.concatenate((
    numpy.genfromtxt('./easydata/inputs/easy_1', delimiter=','),
    numpy.genfromtxt('./easydata/inputs/easy_2', delimiter=','),
    numpy.genfromtxt('./easydata/inputs/easy_3', delimiter=','),
    numpy.genfromtxt('./easydata/inputs/easy_4', delimiter=',')))
    N = Y.shape[0]

    # We have several differet possible initialisations for the embeddings
    #X_mu = numpy.load(X_file)
    #X_mu = PCA(Y_file, Q)
    #X_mu = scipy.randn(N, Q)
    X_S = numpy.clip(numpy.ones((N, Q)) * 0.5
                        + 0 * scipy.randn(N, Q),
                    0.001, 1)
    #X_S = numpy.zeros((N, Q))

    X_mu = numpy.concatenate((
        scipy.load('./easydata/embeddings/easy_1.embedding.npy'),
        scipy.load('./easydata/embeddings/easy_2.embedding.npy'),
        scipy.load('./easydata/embeddings/easy_3.embedding.npy'),
        scipy.load('./easydata/embeddings/easy_4.embedding.npy')))
    '''
    X_S = numpy.concatenate((
        scipy.load('./easydata/embeddings/easy_1.variance.npy'),
        scipy.load('./easydata/embeddings/easy_2.variance.npy'),
        scipy.load('./easydata/embeddings/easy_3.variance.npy'),
        scipy.load('./easydata/embeddings/easy_4.variance.npy')))
    '''
    # Initialise the inducing points
    #Z = X_mu[numpy.random.permutation(N)[:M],:]
    Z = X_mu[:M,:]
    #Z += scipy.randn(M, Q) * 0.1


    global_statistics_names = {
        'Z' : (M, Q), 'sf2' : (1,1), 'alpha' : (1, Q), 'beta' : (1,1), 'X_mu' : (N, Q), 'X_S' : (N, Q)
    }
    # Initialise the global statistics
    global_statistics = {
        'Z' : Z, # see GPy models/bayesian_gplvm.py
        'sf2' : numpy.array([[1.0]]), # see GPy kern/rbf.py
        'alpha' : scipy.ones((1, Q)), # see GPy kern/rbf.py
        'beta' : numpy.array([[1.0]]), # see GPy likelihood/gaussian.py
        'X_Zmu' : X_mu,
        'X_S' : X_S,
    }

    # Initialise bounds for optimisation
    global_statistics_bounds = {
        'Z' : [(None, None) for i in range(M * Q)],
        'sf2' : [('sf2', 'sf2')],
        'alpha' : [('alpha', 'alpha') for i in range(Q)],
        'beta' : [('beta', 'beta')],
        'X_mu' : [(None, None) for i in range(N * Q)],
        'X_S' : [('X_S', 'X_S') for i in range(N * Q)],
    }
    flat_global_statistics_bounds = []
    for key, statistic in global_statistics_bounds.items():
        flat_global_statistics_bounds = flat_global_statistics_bounds+statistic


    ''' 
    Run the optimiser 
    '''
    x0 = flatten_global_statistics(global_statistics)
    # Transform the positiv parameters to be in the range (-Inf, Inf)
    x0 = numpy.array([transform_back(b, x) for b, x in zip(flat_global_statistics_bounds, x0)])
    
    ''' 
    SCG optimisation (adapted from GPy implementation to reduce function calls)
    The number of iterations might be greater than max_f_eval
    '''
    #fix_beta = True
    x = SCG(likelihood, gradient, x0, display=True, maxiters=iterations)
    #fix_beta = False
    #x = SC(likelihood_and_gradient, x[0], display=True, maxiters=iterations)
    flat_array = x[0]
    
    # Transform the parameters that have to be positive to be positive
    flat_array_transformed = numpy.array([transform(b, x) for b, x in zip(flat_global_statistics_bounds, flat_array)])
    global_statistics = rebuild_global_statistics(global_statistics_names, flat_array_transformed)
    print 'Final global_statistics'
    print global_statistics

def likelihood(x):
    try:
        return  likelihood_and_gradient(x)[0]
    except (LinAlgError, ZeroDivisionError, ValueError, Warning, AssertionError) as e:
        return  numpy.inf

def gradient(x):
    try:
        return  likelihood_and_gradient(x)[1]
    except (LinAlgError, ZeroDivisionError, ValueError, Warning, AssertionError) as e:
        return numpy.ones(x.shape)

'''
Likelihood and gradient functions
'''
def likelihood_and_gradient(flat_array, iteration=0):
    global Kmm, Kmm_inv, accumulated_statistics, N, Y, flat_global_statistics_bounds, fix_beta, global_statistics_names
    # Transform the parameters that have to be positive to be positive
    flat_array_transformed = numpy.array([transform(b, x) for b, x in zip(flat_global_statistics_bounds, flat_array)])
    global_statistics = rebuild_global_statistics(global_statistics_names, flat_array_transformed)
    
    #print 'global_statistics'
    #print global_statistics

    Z = global_statistics['Z']
    sf2 = float(global_statistics['sf2'])
    beta = float(global_statistics['beta'])
    alpha = numpy.squeeze(global_statistics['alpha'])
    X_mu = global_statistics['X_mu']
    X_S = global_statistics['X_S']

    # We can calculate the global statistics once
    kernel = kernels.rbf(Q, sf=sf2**0.5, ard=alpha**-0.5)
    Kmm = kernel.K(Z)
    Kmm_inv = numpy.linalg.inv(Kmm)

    # Calculate partial statistics...
    partial_terms = pt.partial_terms(Z, sf2, alpha, beta, M, Q, N, D, update_global_statistics=True)
    partial_terms.set_data(Y, X_mu, X_S, is_set_statistics=True)
    terms = partial_terms.get_local_statistics()
    accumulated_statistics = {
        'sum_YYT' : terms['sum_YYT'],
        'sum_exp_K_ii' : terms['sum_exp_K_ii'],
        'sum_exp_K_mi_K_im' : terms['sum_exp_K_mi_K_im'],
        'sum_exp_K_miY' : terms['exp_K_miY'],
        'sum_KL' : terms['KL'],
        'sum_d_Kmm_d_Z' : partial_terms.dKmm_dZ(),
        'sum_d_exp_K_miY_d_Z' : partial_terms.dexp_K_miY_dZ(),
        'sum_d_exp_K_mi_K_im_d_Z' : partial_terms.dexp_K_mi_K_im_dZ(),
        'sum_d_Kmm_d_alpha' : partial_terms.dKmm_dalpha(),
        'sum_d_exp_K_miY_d_alpha' : partial_terms.dexp_K_miY_dalpha(),
        'sum_d_exp_K_mi_K_im_d_alpha' : partial_terms.dexp_K_mi_K_im_dalpha(),
        'sum_d_Kmm_d_sf2' : partial_terms.dKmm_dsf2(),
        'sum_d_exp_K_ii_d_sf2' : partial_terms.dexp_K_ii_dsf2(),
        'sum_d_exp_K_miY_d_sf2' : partial_terms.dexp_K_miY_dsf2(),
        'sum_d_exp_K_mi_K_im_d_sf2' : partial_terms.dexp_K_mi_K_im_dsf2()
    }

    '''
    Calculates global statistics such as F and partial derivatives of F
    
    In the parallel code we create a new partial_terms object and 
    load the data into it. Here we use the previous one for debugging.
    '''
    partial_derivatives = {
        'F' : partial_terms.logmarglik(),
        'dF_dsum_exp_K_ii' : partial_terms.dF_dexp_K_ii(),
        'dF_dsum_exp_K_miY' : partial_terms.dF_dexp_K_miY(),
        'dF_dsum_exp_K_mi_K_im' : partial_terms.dF_dexp_K_mi_K_im(),
        'dF_dKmm' : partial_terms.dF_dKmm()
    }

    '''
    Evaluate the gradient for 'Z', 'sf2', 'alpha', and 'beta'
    '''
    grad_Z = partial_terms.grad_Z(partial_derivatives['dF_dKmm'],
        accumulated_statistics['sum_d_Kmm_d_Z'],
        partial_derivatives['dF_dsum_exp_K_miY'],
        accumulated_statistics['sum_d_exp_K_miY_d_Z'],
        partial_derivatives['dF_dsum_exp_K_mi_K_im'],
        accumulated_statistics['sum_d_exp_K_mi_K_im_d_Z'])
    grad_alpha = partial_terms.grad_alpha(partial_derivatives['dF_dKmm'],
        accumulated_statistics['sum_d_Kmm_d_alpha'],
        partial_derivatives['dF_dsum_exp_K_miY'],
        accumulated_statistics['sum_d_exp_K_miY_d_alpha'],
        partial_derivatives['dF_dsum_exp_K_mi_K_im'],
        accumulated_statistics['sum_d_exp_K_mi_K_im_d_alpha'])
    grad_sf2 = partial_terms.grad_sf2(partial_derivatives['dF_dKmm'],
        accumulated_statistics['sum_d_Kmm_d_sf2'],
        partial_derivatives['dF_dsum_exp_K_ii'],
        accumulated_statistics['sum_d_exp_K_ii_d_sf2'],
        partial_derivatives['dF_dsum_exp_K_miY'],
        accumulated_statistics['sum_d_exp_K_miY_d_sf2'],
        partial_derivatives['dF_dsum_exp_K_mi_K_im'],
        accumulated_statistics['sum_d_exp_K_mi_K_im_d_sf2'])
    if fix_beta:
        grad_beta = numpy.zeros(1)
    else:
        grad_beta = partial_terms.grad_beta()
    grad_X_mu = partial_terms.grad_X_mu()
    grad_X_S = partial_terms.grad_X_S()






    ####################################################################################################################
    # Debug comparison to GPy
    ####################################################################################################################
    import GPy
    gkern = GPy.kern.rbf(Q, global_statistics['sf2'].squeeze(), global_statistics['alpha'].squeeze()**-0.5, True)

    gpy = GPy.models.BayesianGPLVM(GPy.likelihoods.Gaussian(Y, global_statistics['beta']**-1), Q, X_mu, X_S, num_inducing=M, Z=global_statistics['Z'], kernel=gkern)
    GPy_lml = gpy.log_likelihood()
    GPy_grad = gpy._log_likelihood_gradients()
    dF_dmu = GPy_grad[0:(N * Q)].reshape(N, Q)
    dF_ds = GPy_grad[(N * Q):2*(N * Q)].reshape(N, Q)
    dF_dZ = GPy_grad[2*(N * Q):2*(N * Q)+(M*Q)].reshape(M, Q)
    dF_dsigma2 = GPy_grad[2*(N * Q)+(M*Q)]
    dF_dalpha = GPy_grad[2*(N * Q)+(M*Q)+1:2*(N * Q)+(M*Q)+3]
    dF_dbeta = GPy_grad[2*(N * Q)+(M*Q)+3:]

    dF_dmu2 = grad_X_mu
    dF_ds2 = grad_X_S
    dF_dZ2 = grad_Z
    dF_dalpha2 = grad_alpha * -2 * global_statistics['alpha']**1.5
    dF_dsigma22 = grad_sf2
    dF_dbeta2 = grad_beta * -1 * global_statistics['beta']**2

    if not numpy.sum(numpy.abs(dF_dmu - dF_dmu2)) < 10**-6:
        print '1'
    if not numpy.sum(numpy.abs(dF_dZ - dF_dZ2)) < 10**-6:
        print '2'
    if not numpy.sum(numpy.abs(dF_ds - dF_ds2)) < 10**-6:
        print '3'
    if not numpy.sum(numpy.abs(dF_dalpha - dF_dalpha2)) < 10**-6:
        print '4'
    if not numpy.sum(numpy.abs(dF_dsigma2 - dF_dsigma22))  < 10**-6:
        print '5'
    if not numpy.sum(numpy.abs(dF_dbeta - dF_dbeta2))  < 10**-6:
        print '6'
    if not numpy.abs(GPy_lml - partial_derivatives['F']) < 10**-6:
        print '7'
    
    
    
    #print 'gradient'
    #print gradient

    #gradient = {'Z' : dF_dZ,
    #    'sf2' : dF_dsigma2,
    #    'alpha' : dF_dalpha * -0.5 * global_statistics['alpha']**-1.5,
    #    'beta' : dF_dbeta * -1 * global_statistics['beta']**-2,
    #    'X_mu' : dF_dmu,
    #    'X_S' : dF_ds}
    #gradient = flatten_global_statistics(gradient)
    #likelihood = GPy_lml
    gradient = {'Z' : grad_Z,
        'sf2' : grad_sf2,
        'alpha' : grad_alpha,
        'beta' : grad_beta,
        'X_mu' : grad_X_mu,
        'X_S' : grad_X_S}
    gradient = flatten_global_statistics(gradient)
    likelihood = partial_derivatives['F']
    # Transform the gradient parameters that have to be positive by multiplying 
    # them by the gradeint of the transform f:  g(f(x))' = g'(f(x))f'(x)
    gradient = numpy.array([g * transform_grad(b, x) for b, x, g in zip(flat_global_statistics_bounds, flat_array, gradient)])
    return -1 * likelihood, -1 * gradient


def PCA(Y_name, input_dim):
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
    Y = numpy.genfromtxt(Y_name, delimiter=',')
    Z = numpy.linalg.svd(Y - Y.mean(axis=0), full_matrices=False)
    [X, W] = [Z[0][:, 0:input_dim], numpy.dot(numpy.diag(Z[1]), Z[2]).T[:, 0:input_dim]]
    v = X.std(axis=0)
    X /= v;
    W *= v;
    return X

def flatten_global_statistics(global_statistics):
    flat_array = numpy.array([])
    for key, statistic in global_statistics.items():
        flat_array = numpy.concatenate((flat_array, statistic.flatten()))
    return flat_array

def rebuild_global_statistics(global_statistics_names, flat_array):
    global_statistics = {}
    start = 0
    for key, shape in global_statistics_names.items():
        size = shape[0] * shape[1]
        global_statistics[key] = flat_array[start:start+size].reshape(shape)
        start = start + size
    return global_statistics


lim_val = -numpy.log(sys.float_info.epsilon)
# Transform a parameter to be in (0, inf) if the bound constraints it to be positive
def transform(b, x):
    if b == (0, None):
        if x > lim_val:
            return x
        elif x < -lim_val:
            return numpy.log(1 + numpy.exp(-lim_val))
        else:
            return numpy.log(1 + numpy.exp(x))
    elif b == ('sf2', 'sf2'):
        return x
    elif b == ('alpha', 'alpha'):
        return x**-2
    elif b == ('beta', 'beta'):
        return x**-1
    elif b == ('X_S', 'X_S'):
        return x
    elif b == (None, None):
        return x

# Transform a parameter back to be in (-inf, inf) if the bound constraints it to be positive
def transform_back(b, x):
    if b == (0, None):
        if x > lim_val:
            return x
        elif x <= sys.float_info.epsilon:
            return numpy.log(-1 + numpy.exp(sys.float_info.epsilon))
        else:
            return numpy.log(-1 + numpy.exp(x))
    elif b == ('sf2', 'sf2'):
        return x
    elif b == ('alpha', 'alpha'):
        return x**-0.5
    elif b == ('beta', 'beta'):
        return x**-1
    elif b == ('X_S', 'X_S'):
        return x
    elif b == (None, None):
        return x

# Gradient of the (0, inf) transform if the bound constraints it to be positive
def transform_grad(b, x):
    if b == (0, None):
        if x > lim_val:
            #return x
            return 1
        elif x < -lim_val:
            return numpy.exp(lim_val) / (numpy.exp(lim_val) + 1)
        else:
            ''' This should be reverted '''
            return numpy.exp(x) / (numpy.exp(x) + 1)
        #return 1. - 1. / (1 + numpy.exp(x))
    elif b == ('sf2', 'sf2'):
        return 1
    elif b == ('alpha', 'alpha'):
        return -2*x**3
    elif b == ('beta', 'beta'):
        return -x**2
    elif b == ('X_S', 'X_S'):
        return 1
    elif b == (None, None):
        return 1


if __name__ == '__main__':
    main()

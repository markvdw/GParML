import numpy
import sys

'''
Supporting functions
'''

def linear_dim_reduction(options, input_files_names):
    mean = numpy.zeros((options['D']))
    # Collect mean of data over all files
    for file_name in input_files_names:
        input_file = options['input'] + '/' + file_name
        Y = genfromtxt(input_file, delimiter=',')
        if (len(Y.shape) == 1):
            Y = numpy.atleast_2d(Y).T
        mean += Y.sum(axis=0)
    mean /= options['N']

    S = numpy.zeros((options['D'], options['D']))
    # Collect covariance of data over all files
    for file_name in input_files_names:
        input_file = options['input'] + '/' + file_name
        Y = genfromtxt(input_file, delimiter=',')
        if (len(Y.shape) == 1):
            Y = numpy.atleast_2d(Y).T
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
    if (len(Y.shape) == 1):
        Y = numpy.atleast_2d(Y).T
    Minv = numpy.linalg.inv(sigma2 * numpy.eye(options['Q']) + W.T.dot(W))
    X = (Minv.dot(W.T).dot(Y.T - mean[:, None])).T
    v = X.std(axis=0)
    X /= v
    return X

def FA(options, Y_name, W, Psi, mean):
    Y = genfromtxt(Y_name, delimiter=',')
    if (len(Y.shape) == 1):
        Y = numpy.atleast_2d(Y).T
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



lim_val = -numpy.log(sys.float_info.epsilon)
# Transform a parameter to be in (0, inf) if the bound constraints it to be positive
def transform(b, x):
    if b == (0, None):
        assert -lim_val < x < lim_val
        return numpy.log(1 + numpy.exp(x))
    elif b == (None, None):
        return x

# Transform a parameter back to be in (-inf, inf) if the bound constraints it to be positive
def transform_back(b, x):
    if b == (0, None):
        assert sys.float_info.epsilon < x < lim_val
        return numpy.log(-1 + numpy.exp(x))
    elif b == (None, None):
        return x

# Gradient of the (0, inf) transform if the bound constraints it to be positive
def transform_grad(b, x):
    if b == (0, None):
        assert -lim_val < x < lim_val
        return 1 / (numpy.exp(-x) + 1)
    elif b == (None, None):
        return 1


''' Transformation functions for arrays with a single costraint '''
# Transform a parameter to be in (0, inf) if the bound constraints it to be positive
def transformVar(x):
    assert numpy.all(-lim_val < x) and numpy.all(x < lim_val)
    val = numpy.log(1 + numpy.exp(x))
    return val

# Transform a parameter back to be in (-inf, inf) if the bound constraints it to be positive
def transformVar_back(x):
    assert numpy.all(sys.float_info.epsilon < x) and numpy.all(x < lim_val)
    val = numpy.log(-1 + numpy.exp(x))
    return val

# Gradient of the (0, inf) transform if the bound constraints it to be positive
def transformVar_grad(x):
    assert numpy.all(-lim_val < x) and numpy.all(x < lim_val)
    val = 1 / (numpy.exp(-x) + 1)
    return val


###############################################################################
# kernel_exp.py
# Calculate the expectations over kernel matrices w.r.t. q(X), i.e.
#  - <K_{m<i>}>_q(X_i)                     (4.14)   - Mx1 (column vector)
#  - <K_{m<i>}>_q(X_i) Y_i                 (above)  - MxD
#  - <K_{m<i>} K_{<i>m>}>_q(X_i)           (4.15)   - MxM matrix
#  - Y_i Y_i^T
# To be called from the first mapper function.
###############################################################################

import numpy as np

def calc_expect_K_mi_Y (Z, hyp_ard, X_mu, X_S, Y):
    '''
    calc_expect_K_mi
    Calculates the expectation of K_<i>m w.r.t the variational distribution
    over X's, q(X): <K_{m<i>}>_q(X_i).

    Args:
        Z       : MxQ matrix of inducting point locations.
        hyp_ard : GP hyperparameters for the ARD kernel.
        X_mu    : NxQ array of mean vectors for q(X).
        X_S     : NxQ array of diagonal entries for the covariance of q(X).
        Y       : NxD array of data.

    Returns:
        MxoutD matrix: <K_{m<i>}>_q(X_i) Y'.
    '''
    # input assertions
    assert np.all(X_S >= 0.0)
    assert np.all(hyp_ard.ard >= 0.0)
    assert X_mu.ndim == 2
    assert X_S.ndim == 2
    assert X_mu.shape == X_S.shape

    outD = Y.shape[1]
    M = Z.shape[0]
    Q = Z.shape[1]

    sf = hyp_ard.sf
    alpha = 1.0 / hyp_ard.ard**2

    ky_sum = np.zeros((M, outD))
    exp_K_mi = calc_expect_K_mi(Z, hyp_ard, X_mu, X_S)
    assert exp_K_mi.shape[0] == Y.shape[0]
    for expvect, y in zip(exp_K_mi, Y):
        ky_sum += np.outer(expvect, y)

    return ky_sum

def calc_expect_K_mi (Z, hyp_ard, X_mu, X_S):
    '''
    calc_expect_K_mi
    Calculates the expectations of K_<i>m w.r.t. the variational distribution
    over X's, q(X). It does not sum the result, but stores each vector in a row
    of the output matrix.

    Args:
        Z       : MxQ matrix of inducting point locations.
        hyp_ard : GP hyperparameters for the ARD kernel.
        X_mu    : NxQ array of mean vectors for q(X).
        X_S     : NxQ array of diagonal entries for the covariance of q(X).

    Returns:
        NxM matrix. Each row is one expectation vector.
    '''
    # input assertions
    assert np.all(X_S >= 0.0)
    assert np.all(hyp_ard.ard >= 0.0)
    assert X_mu.ndim == 2
    assert X_S.ndim == 2
    assert X_mu.shape == X_S.shape

    N = X_mu.shape[0]
    M = Z.shape[0]

    sf = hyp_ard.sf
    alpha = 1.0 / hyp_ard.ard**2

    res2 = (sf**2 / np.prod((X_S[:, :] * alpha[None, :] + 1.)**0.5, 1)[:, None]) * np.exp(-0.5*np.sum( ( (Z[None, :, :] - X_mu[:, None, :])**2 * alpha[None, None, :] ) / (alpha[None, None, :] * X_S[:, None, :] + 1.) , 2))

    return res2

def calc_expect_K_mi_K_im_old (Z, hyp_ard, X_mu, X_S):
    '''
    calc_expect_K_mi_K_im
    Calculates the expectation of the outer product of K_<i>m w.r.t the
    variational distribution over X's, q(X):
       <K_{m<i>} K_{<i>m>}>_q(X_i)
    Eqn (4.15).

    Args:
        Z       : MxQ matrix of inducting point locations.
        hyp_ard : GP hyperparameters for the ARD kernel.
        X_mu    : NxQ array of mean vectors for q(X).
        X_S     : NxQ array of diagonal entries for the covariance of q(X).

    Returns:
        MxM matrix: <K_{m<i>} K_{<i>m>}>_q(X_i)
    '''
    # input assertions
    assert np.all(X_S >= 0.0)
    assert np.all(hyp_ard.ard >= 0.0)
    assert X_mu.ndim == 2
    assert X_S.ndim == 2
    assert X_mu.shape[1] == X_S.shape[1]

    M = Z.shape[0]
    # Q = Z.shape[1]

    sf = hyp_ard.sf
    alpha = hyp_ard.ard**-2

    # Future optimisation: Can be vectorised
    res = np.zeros((M, M))
    for mu, s in zip(X_mu, X_S):
        const = sf**4 * np.prod(2*alpha*s + 1)**-0.5
        for mi in range(M):
            for mj in range(M):
                t1 = -np.sum( 0.25 * alpha * (Z[mi, :] - Z[mj, :])**2 )
                t2 = -np.sum( alpha * (mu - 0.5*Z[mi, :] - 0.5*Z[mj, :])**2 / (2 * alpha * s + 1) )
                res[mi, mj] += const * np.exp( t1 + t2 )

    return res

def calc_expect_K_mi_K_im (Z, hyp_ard, X_mu, X_S):
    # input assertions
    assert np.all(X_S >= 0.0)
    assert np.all(hyp_ard.ard >= 0.0)
    assert X_mu.ndim == 2
    assert X_S.ndim == 2
    assert X_mu.shape[1] == X_S.shape[1]

    M = Z.shape[0]
    # Q = Z.shape[1]

    sf = hyp_ard.sf
    alpha = hyp_ard.ard**-2

    res = np.zeros((M, M))

    for mu, s in zip(X_mu, X_S):
        const = sf**4 * np.prod(2*alpha*s + 1)**-0.5
        t1 = -0.25 * np.sum(alpha[None, None, :] * (Z[:, None, :] - Z)**2, 2)
        t2 = -np.sum( alpha[None, None, :] * (mu[None, None, :] - 0.5*Z[:, None, :] - 0.5*Z)**2 / (2*alpha*s+1)[None, None, :], 2)
        res += const * np.exp(t1 + t2)

    return res

if __name__ == '__main__':
    ###########################################################################
    # If run as main, do some tests and show some stuff
    ###########################################################################

    pass

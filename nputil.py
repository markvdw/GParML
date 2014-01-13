###############################################################################
# nputil.py
# Some functions that are generally useful.
###############################################################################

import numpy as np
import numpy.random as rnd

np.seterr(all='raise')

def check_grad(func, grad, X0, dmag=10**-8):
    '''
    check_grad
    Checks the validity of a function computing the gradient by comparing it
    with a finite difference method approximation. Slightly different behaviour
    than the version in Scipy.

    Args:
        func : A function which takes a Din length vector input and returns a
               Dout length vector.
        grad : Gradient evaluation function. Takes a Din length vector as input
               and returns a DinxDout matrix of derivatives.
        X0   : NxDin matrix of inputs where the gradient is to be tested.
        dmag : Magnitude of the finite difference.
    '''

    Din = X0.shape[1]

    maxpd = 0.0
    avgpd = 0.0

    for x0 in X0:
        Fa = func(x0)
        g = grad(x0)
        Dout = np.atleast_2d(Fa).shape[1]

        # Build matrix of derivatives
        fd = np.zeros((Dout, Din))
        for d in xrange(Din):
            offset = np.zeros(Din)
            offset[d] = dmag

            Fb = func(x0 + offset)
            fd[:, d] = (Fb - Fa) / dmag

        # Percentage difference between each derivative
        pd = (g - fd) / fd * 100.0

        avgpd += np.average(np.abs(pd))
        maxpd = np.max([maxpd, np.max(np.abs(pd))])

    avgpd /= len(X0)

    return (maxpd, pd, avgpd, g, fd)

def check_grad_old(func, grad, X0, dmag=10**-8, N = 10):
    '''
    check_grad_old
    Checks the gradients by comparing a function evaluation with a finite
    difference approximation. Slightly different behaviour than the one in
    Scipy.

    Args:
        func : A function which takes as input a D length vector.
        grad : The gradiant of func, which takes as in put a D length vector.
        x0   : NxD matrix of points to check the gradient at.

    Returns:
        Tuple of the average difference and the maximum difference for all
        inputs.

    Note:
        I've decided I don't like this version. To be phased out.
    '''
    # TODO: The correct thing to do, would be to make a finite difference in
    #       each direction of the input, and then compare the gradients.
    D = X0.shape[1]
    assert len(X0) == 1


    diffs = np.zeros(N)
    fdes = np.zeros(N)
    derivs = np.zeros(N)
    Fs = np.zeros(N)

    g = grad(X0[0, :])

    for n in xrange(N):
        d = rnd.randn(D) * dmag

        Fa = func(X0[0, :])
        Fb = func(X0[0, :] + d)
        fde = (Fb - Fa) / np.linalg.norm(d)
        deriv = g.dot(d) / np.linalg.norm(d)
        # print "fde   %d" % fde
        # print "deriv %d" % deriv
        Fs[n] = Fa
        diffs[n] = fde - deriv
        fdes[n] = fde
        derivs[n] = deriv

    err_prcnt = np.abs(diffs / fde * 100.0)
    max_err_prcnt = np.max(err_prcnt)
    max_idx = np.argmax(err_prcnt)
    avg_err_prcnt = np.average(err_prcnt)

    return (max_err_prcnt,
            avg_err_prcnt,
            np.max(np.abs(diffs)),
            diffs[max_idx],
            fdes[max_idx],
            derivs[max_idx],
            Fs[max_idx])

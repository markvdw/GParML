###############################################################################
# MLtools.py
# Some tools used for Machine Learning programming in Python.
#
# Mark van der Wilk (mv310@cam.ac.uk)
###############################################################################

from __future__ import division

import numpy as np
import numpy.linalg as linalg
import numpy.random as random
from scipy import constants
from scipy import stats

def _find_robust_range(data, prop=0.95):
    numpoints = np.prod(data.shape)
    threshold = prop * numpoints
    r = max([np.abs(np.max(data)), np.abs(np.min(data))])

    while(np.sum(np.abs(data) < r) > threshold):
        r *= 0.9

    return r, np.sum(np.abs(data) < r) / numpoints

def auto_axes_robust(ax, datax, datay, prop=0.95, verbose=False):
    '''
    auto_axes_robust
    Automatically adjust the axes of a plot, but be robust to outliers. Make
    sure that at most the proportion of the data given by 'prop' is actually
    displayed.
    '''
    rx, fx = _find_robust_range(datax, prop)
    ry, fy = _find_robust_range(datay, prop)
    ax.set_xlim(-rx, rx)
    ax.set_ylim(-ry, ry)

    if (verbose):
        print('At the least %f is displayed.' % (fx * fy))

class MultivariateNormal(object):
    def __init__(self, mu, S):
        self.D = len(mu)
        self.mu = mu
        self._S = S
        self._iS = linalg.inv(S)
        self._cS = linalg.cholesky(S)

    def logpdf(self, X):
        return mvnlogpdf_p(X, self.mu, self._iS)

    def pdf(self, X):
        return mvnpdf_p(X, self.mu, self._iS)

    def sample(self, N=1):
        return self._cS.dot(random.randn(self.D, N)).T

    @property
    def S(self):
        return self._S

    @S.setter
    def S(self, value):
        self._S = value
        self._cS = linalg.cholesky(S)
        self._iS = linalg.inv(S)

class Interval(object):
    def __init__(self, lower=0.0, upper=0.0):
        """
        Interval constructor

        Args:
            lower: Lower bound
            upper: Upper bound
        """
        self._lower = lower
        self._upper = upper
        self._range = upper - lower
        assert self._range >= 0.0

    @property
    def range(self):
        return self._range

    @property
    def lower(self):
        return self._lower

    @lower.setter
    def lower(self, value):
        self._lower = value
        self._range = self._upper - value

    @property
    def upper(self):
        return self._upper

    @upper.setter
    def upper(self, value):
        self._upper = value
        self._range = self._upper - self._lower

    def inside(self, x):
        return (x >= self._lower) and (x <= self.upper)

class MultivariateUniform(object):
    def __init__(self, hyper_rectangle):
        """
        MultivariateUniform constructor

        Args:
            hyper_rectangle: Array of Interval objects specifying the range of
                             the uniform distribution.
        """
        self._r = hyper_rectangle

        # Calculate density
        logdensity = 0
        for hr in hyper_rectangle:
            logdensity -= np.log(hr.range)

        self._ld = logdensity

    @property
    def D(self):
        return len(self._r)

    def logpdf(self, X):
        lpdf = np.empty(len(X))
        ninf = float('-inf')
        for n, x in enumerate(X):
            lpdf[n] = self._ld
            for val, i in zip(x, self._r):
                if not i.inside(val):
                    lpdf[n] = ninf
                    break
        return lpdf

    def pdf(self, X):
        return np.exp(self.logpdf(X))

    def sample(self, N=1):
        y = np.empty((N, self.D))

        for d, i in enumerate(self._r):
            for n in range(N):
                y[n, d] = random.uniform(i.lower, i.upper)

        return y

def _check_single_data(X, D):
    """
    Checks whether data given to one of the distributions is a single data
    point of dimension D, rather than a set of datapoints of dimension 1. This
    is necessary, as there is no distinction between row and column vectors in
    numpy (sadly).
    """

    # Alternatively, could use np.atleast_2d... Oh well.
    if (len(X.shape) == 1 and len(X) == D):
        return np.reshape(X, [1, D])
    else:
        return X

def mvnlogpdf_p (X, mu, PrecMat):
    """
    Multivariate Normal Log PDF

    Args:
        X      : NxD matrix of input data. Each ROW is a single sample.
        mu     : Dx1 vector for the mean.
        PrecMat: DxD precision matrix.

    Returns:
        Nx1 vector of log probabilities.
    """
    D = len(mu)
    X = _check_single_data(X, D)
    N = len(X)

    _, neglogdet = linalg.slogdet(PrecMat)
    normconst = -0.5 * (D * np.log(2 * constants.pi) - neglogdet)

    logpdf = np.empty((N, 1))
    for n, x in enumerate(X):
        d = x - mu
        logpdf[n] = normconst - 0.5 * d.dot(PrecMat.dot(d))

    return logpdf

def mvnlogpdf (X, mu, Sigma):
    """
    Multivariate Normal Log PDF

    Args:
        X    : NxD matrix of input data. Each ROW is a single sample.
        mu   : Dx1 vector for the mean.
        Sigma: DxD covariance matrix.

    Returns:
        Nx1 vector of log probabilities.
    """
    D = len(mu)
    X = _check_single_data(X, D)
    N = len(X)

    _, logdet = linalg.slogdet(Sigma)
    normconst = -0.5 * (D * np.log(2 * constants.pi) + logdet)

    iS = linalg.inv(Sigma)
    logpdf = np.empty((N, 1))
    for n, x in enumerate(X):
        d = x - mu
        logpdf[n] = normconst - 0.5 * d.dot(iS.dot(d))

    return logpdf

def mvnpdf_p (X, mu, PrecMat):
    """
    Multivariate Normal PDF

    Args:
        X    : NxD matrix of input data. Each ROW is a single sample.
        mu   : Dx1 vector for the mean.
        Sigma: DxD precision matrix.

    Returns:
        Nx1 vector of log probabilities.
    """
    D = len(mu)
    X = _check_single_data(X, D)
    N = len(X)

    normconst = 1.0 / ((2*constants.pi)**(0.5*D)*linalg.det(PrecMat)**-0.5)

    pdf = np.empty((N, 1))
    for n, x in enumerate(X):
        d = x - mu
        pdf[n] = normconst * np.exp(-0.5 * d.dot(PrecMat.dot(d)))

    return pdf

def mvnpdf (X, mu, Sigma):
    """
    Multivariate Normal PDF

    Args:
        X    : NxD matrix of input data. Each ROW is a single sample.
        mu   : Dx1 vector for the mean.
        Sigma: DxD covariance matrix.

    Returns:
        Nx1 vector of log probabilities.
    """
    D = len(mu)
    X = _check_single_data(X, D)
    N = len(X)

    normconst = 1.0 / ((2*constants.pi)**(0.5*D)*linalg.det(Sigma)**0.5)

    pdf = np.empty((N, 1))
    iS = linalg.inv(Sigma)
    for n, x in enumerate(X):
        d = x - mu
        pdf[n] = normconst * np.exp(-0.5 * d.dot(iS.dot(d)))

    return pdf

###############################################################################
# Unit Tests
# Simple tests to confirm the consistancy between these functions here and the
# standard pdf functions for single variables.
###############################################################################
if __name__ == '__main__':
    import unittest
    import argparse
    import time

###############################################################################
# Parse the arguments...
###############################################################################
    parser = argparse.ArgumentParser(description='Housekeeping for MLtools.py.')
    parser.add_argument('-b', '--benchmark', help="Benchmark the different functions.", action="store_true")
    parser.add_argument('-t', '--test', help="Run the unit tests.", action="store_true")
    args = parser.parse_args()

    class TestSequenceFunctions(unittest.TestCase):
        def setUp(self):
            self.tolerance = 10**-9

        def test_log_1d_eval(self):
            X = random.randn(100, 1)
            logpdf_tools = mvnlogpdf(X, [0], [[1]])
            logpdf_sp = np.log(stats.norm.pdf(X))

            diff = np.sum(np.absolute(logpdf_tools - logpdf_sp)) / len(X)

            self.assertTrue(diff < self.tolerance)

        def test_exp_1d_eval(self):
            X = random.randn(100, 1)
            pdf_tools = mvnpdf(X, [0], [[1]])
            pdf_sp = stats.norm.pdf(X)

            diff = np.sum(np.absolute(pdf_tools - pdf_sp)) / len(X)

            self.assertTrue(diff < self.tolerance)

        def test_log_diag(self):
            D = 100
            N = 3000
            X = random.randn(N, D)
            logpdf_tools = mvnlogpdf(X, [0] * D, np.eye(D)).T
            logpdf_sp = np.log(np.prod(stats.norm.pdf(X), axis=1))

            diff = np.sum(np.absolute(logpdf_tools - logpdf_sp)) / N

            self.assertTrue(diff < self.tolerance)

        def test_exp_diag(self):
            D = 3
            N = 3000
            X = random.randn(N, D)
            pdf_tools = mvnpdf(X, [0] * D, np.eye(D)).T
            pdf_sp = np.prod(stats.norm.pdf(X), axis=1)

            diff = np.sum(np.absolute(pdf_tools - pdf_sp)) / N

            self.assertTrue(diff < self.tolerance)

        def test_logpdf_vs_pdf(self):
            D = 9
            N = 100

            X = random.randn(N, D)
            S = random.randn(D, D)
            S = S.dot(S.T)
            logpdf = mvnlogpdf(X, [0]*D, S)
            pdf = mvnpdf(X, [0]*D, S)

            diff = np.sum(np.absolute(np.exp(logpdf) - pdf)) / len(X)

            self.assertTrue(diff < self.tolerance)

        def test_logpdf_precmat(self):
            D = 100
            N = 1000

            X = random.randn(N, D)
            S = random.randn(D, D)
            S = S.dot(S.T)
            logpdf = mvnlogpdf_p(X, [0]*D, linalg.inv(S))
            logpdf_s = mvnlogpdf(X, [0]*D, S)

            diff = np.sum(np.absolute(logpdf - logpdf_s)) / len(X)

            self.assertTrue(diff < self.tolerance)

        def test_pdf_precmat(self):
            D = 100
            N = 1000

            X = random.randn(N, D)
            S = random.randn(D, D)
            S = S.dot(S.T)
            pdf = mvnpdf_p(X, [0]*D, linalg.inv(S))
            pdf_s = mvnpdf(X, [0]*D, S)

            diff = np.sum(np.absolute(pdf - pdf_s)) / len(X)

            self.assertTrue(diff < self.tolerance)

        def test_interval(self):
            l = random.randn()
            u = np.absolute(random.randn()) + l

            i = Interval(l, u)

            self.assertAlmostEqual(i.range, u-l)

            l = random.randn()
            u = np.absolute(random.randn()) + l

            i.lower = l
            self.assertEqual(i.lower, l)
            i.upper = u
            self.assertEqual(i.upper, u)

            self.assertAlmostEqual(i.range, u-l)

        def test_mvnunif(self):
            homog = MultivariateUniform([Interval(-1.3, 1.3)] * 33)
            s = homog.sample(1000)
            self.assertLessEqual(np.max(s), 1.3)
            self.assertGreaterEqual(np.min(s), -1.3)

            hr = [Interval(random.randn() - 20, random.randn() + 20) for _ in range(13)]
            logp = 0
            for r in hr:
                logp -= np.log(r.range)

            u = MultivariateUniform(hr)
            self.assertAlmostEqual(logp, u._ld)

    if args.benchmark:
        print("Benchmarking...")

        D = 23
        N = 100000

        X = random.randn(N, D)
        S = random.randn(D, D)
        S = S.dot(S.T)

        # Benchmark mvnlogpdf
        start = time.time()
        logpdf = mvnlogpdf(X, [0]*D, S)
        print(time.time() - start)

        # Benchmark mvnpdf
        start = time.time()
        pdf = mvnpdf(X, [0]*D, S)
        print(time.time() - start)

        # Benchmark mvnlogpdf_p
        start = time.time()
        logpdf = mvnlogpdf_p(X, [0]*D, linalg.inv(S))
        print(time.time() - start)

    if args.test:
        suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceFunctions)
        unittest.TextTestRunner(verbosity=2).run(suite)

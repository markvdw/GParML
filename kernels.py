###############################################################################
# kernels.py
# Implement code to generate kernel matrices. So far, we only need the ARD
# squared exponential kernel. Technically, you could use any package to do this
# like GPy, but we decided to write our own function for some reason.
###############################################################################

import numpy as np
from scipy.spatial import distance

class ArdHypers(object):
    '''
    ArdHypers
    Container class for the hyperparameters of the ARD squared exponential
    kernel.
    '''
    def __init__(self, D, sf=1.0, ll=1.0, ard=None):
        self.D = D
        self.sf = sf
        if ard==None:
            self.ard = np.ones(D) * ll
        else:
            self.ard = np.array(ard).squeeze()
            assert self.ard.ndim == 1

    @property
    def ll(self):
        if (np.all(self.ard == self.ard[0])):
            return self.ard[0]
        else:
            raise ValueError("RBF kernel is not isotropic")

    @ll.setter
    def ll(self, value):
        self.ard = np.ones(self.D) * value

###############################################################################
# TODO: Use ArdHypers class in rbf.
#       Update view code.
###############################################################################
class rbf:
    def __init__(self, D, ll=1.0, sf=1.0, ard=None):
        '''
        __init__
        Constructor for the rbf kernel.

        Args:
            ll : Length scale parameter, if no ARD coefficients are used.
            sf : Marginal variance of the GP.
            ard: D element numpy array of length scales for each dimension.
        '''
        assert sf > 0.0

        self.D = D
        self.sf = sf
        if ard is None:
            self.ard = np.ones(D) * ll
        else:
            self.ard = np.array(ard)

    @property
    def ll(self):
        if (np.all(self.ard == self.ard[0])):
            return self.ard[0]
        else:
            raise ValueError("RBF kernel is not isotropic.")

    @ll.setter
    def ll(self, value):
        self.ard = np.ones(self.D) * value

    def K(self, X, X2=None):
        """
        rbf
        Implements the ARD RBF kernel.

        Args:
            X  : NxD numpy array of input points. N is the number of points, D
                 their dimension. I.e. each data point is a ROW VECTOR (same
                 convention as in GPML).
            X2 : Optional N2xD numpy array of input points. If it is given, the
                 cross-covariances are calculated.

        Returns:
            K_X1X2 or K_XX covariance matrix (NxN2 or NxN respectively).
        """
        # Assume we want to calculate the self-covariance if no second dataset is
        # given.
        if X2==None:
            X2 = X

        # Ensure that we can accept 1D arrays as well, where we assume the input
        # is 1 dimensional.
        if (X.ndim == 1):
            X = np.atleast_2d(X)
            X2 = np.atleast_2d(X2)

        # Get data shapes & dimensions etc.
        N = X.shape[0]
        D = X.shape[1]
        N2 = X2.shape[0]
        assert D == self.D

        # Dimensions must be equal. Assert for debug purposes.
        assert X.shape[1] == X2.shape[1]

        # Actually calculate the covariance matrix
        K = distance.cdist(X, X2, 'seuclidean', V=2*self.ard**2)
        assert K.shape == (N, N2)

        K = self.sf**2 * np.exp(-K**2)

        return K

###############################################################################
# If run as main, run either unit tests or just some visualisations.
###############################################################################
if __name__ == '__main__':
    import unittest
    import argparse
    import matplotlib.pyplot as plt
    import numpy.random as rnd
    import numpy.linalg as linalg

# Parse the arguments...
    parser = argparse.ArgumentParser(description='Housekeeping for kernels.py.')
    parser.add_argument('-v', '--view', help="View a covariance matrix.", action="store_true")
    parser.add_argument('-t', '--test', help="Run the unit tests.", action="store_true")
    args = parser.parse_args()

    class TestSequenceFunctions(unittest.TestCase):
        def setUp(self):
            pass

        def test_1(self):
            X = np.atleast_2d(rnd.uniform(-5.0, 5.0, 10*3)).T
            kern = rbf(1, 2.0, 4.0)
            K = kern.K(X)

            a = 3
            b = 5
            self.assertEqual(K[a, b], 16.0 * np.exp(-(X[a] - X[b])**2/4.0))

        def test_2(self):
            kern1 = rbf(3, ard=np.array([1.0, 1.0, float('inf')]))
            kern2 = rbf(2, ard=np.array([1.0, 1.0]))

            X1 = np.reshape(rnd.uniform(-5.0, 5.0, 10*3), (10, 3))
            X2 = np.reshape(rnd.uniform(-5.0, 5.0, 5*3), (5, 3))

            #Ka = rbf(X1, X2, ard=np.array([1.0, 1.0, float('inf')]))
            #Kb = rbf(X1[:, 0:2], X2[:, 0:2], ard=np.array([1.0, 1.0]))
            Ka = kern1.K(X1, X2)
            Kb = kern2.K(X1[:, 0:2], X2[:, 0:2])

            self.assertTrue(Ka.shape == (10, 5))
            self.assertTrue((Ka == Kb).all())

        def test_hypstruct(self):
            # Just needs to get through this with no errors
            m = ArdHypers(3, 3.0, ard=[1.0, 4.0, 3.0])
            m.ll = 3

    if args.view:
        X = rnd.uniform(-2.0, 2.0, 200)
        K = rbf(X, X)
        Xs = np.sort(X)
        Ks = rbf(Xs)

        fig = plt.figure(1)
        plt.clf()
        cax = plt.imshow(K, interpolation='none')
        fig.colorbar(cax)
        plt.draw()

        fig = plt.figure(2)
        plt.clf()
        cax = plt.imshow(Ks, interpolation='none')
        fig.colorbar(cax)
        plt.draw()

        # Draw a GP with the covariance matrix
        y = rnd.multivariate_normal(np.zeros(200), K)
        plt.figure(3)
        plt.plot(X, y, 'x')

        plt.show()

    if args.test:
        suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceFunctions)
        unittest.TextTestRunner(verbosity=2).run(suite)

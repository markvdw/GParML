import unittest
import argparse
import numpy as np
import numpy.random as rnd
import numpy.linalg as linalg
import matplotlib.pyplot as plt

import nputil
import partial_terms
import kernels

import sys
sys.path.append("/home/yg279/Downloads/GPy-master")
import GPy

###########################################################################
# If run as main, do some tests and show some stuff
###########################################################################
parser = argparse.ArgumentParser(description='Testing options.')
parser.add_argument('-v', '--verbose', help="Display information while running tests.", action="store_true")
args = parser.parse_args()

class TestSequenceFunctions(unittest.TestCase):
    def setUp(self):
        ###################################################################
        # Setup parameters and values to evaluate gradients at.
        self.D = 7                          # Output dimension
        self.Q = 2
        self.N = 5
        self.hyp = kernels.ArdHypers(self.Q, sf=0.5 + np.exp(rnd.randn(1)), ard=np.exp(rnd.randn(self.Q)))
        #self.hyp = kernels.ArdHypers(self.Q, sf=0.5 + np.exp(rnd.randn(1)), ard=1 + 0 * np.exp(rnd.randn(self.Q)))
        #self.hyp = kernels.ArdHypers(self.Q, sf=2.0, ard=1 + 0 * np.exp(rnd.randn(self.Q)))
        self.sn = rnd.uniform(0.01, 0.1)
        self.beta = self.sn**-2
        self.kernel = kernels.rbf(self.Q, sf=self.hyp.sf, ard=self.hyp.ard)

        # Inducing points
        self.M = 10

        self.genPriorData()
        self.Kmm = self.kernel.K(self.Z)
        self.Kmm_inv = linalg.inv(self.Kmm)
        self.partial_terms = partial_terms.partial_terms(self.Z, self.hyp.sf**2, self.hyp.ard**-2, self.beta, self.M, self.Q, self.N, self.D)
        self.partial_terms.set_global_statistics(self.Kmm, self.Kmm_inv)
        self.partial_terms.set_data(self.Y, self.X_mu, self.X_S, is_set_statistics=True)

    def genPriorData(self):
        # Generate from a GPLVM
        self.X = rnd.randn(self.N, self.Q)
        KXX = self.kernel.K(self.X)
        L = linalg.cholesky(KXX)
        self.Y = L.dot(rnd.randn(self.N, self.D)) + rnd.randn(self.N, self.D) * self.sn

        # Linear initialisation with a bit of noise.
        #self.Y = self.X.dot(rnd.randn(self.Q, self.D)) + rnd.randn(self.N, self.D) * self.sn

        self.Z = rnd.randn(self.M, self.Q)

        self.X_mu = self.X + 0.05 * rnd.randn(self.N, self.Q)
        self.X_S = 0.2 * np.ones((self.N, self.Q))

    def test_dF_dZ(self):
        '''
        test_dF_dZ
        Essentially one test to rule them all.

        Notes:
           - Values coming from Z are stable to smaller finite differences.
        '''

        def func_wrapper(Z):
            self.partial_terms.Z = Z.reshape(self.M, self.Q)
            self.partial_terms.set_data(self.Y, self.X_mu, self.X_S, is_set_statistics=True)
            self.partial_terms.update_global_statistics()
            lml = self.partial_terms.logmarglik()
            return lml

        def grad_wrapper(Z):
            self.partial_terms.Z = Z.reshape(self.M, self.Q)
            self.partial_terms.set_data(self.Y, self.X_mu, self.X_S, is_set_statistics=True)
            self.partial_terms.update_global_statistics()
            dF_dKmm = self.partial_terms.dF_dKmm()
            sum_d_Kmm_d_Z = self.partial_terms.dKmm_dZ()
            dF_dsum_exp_K_miY = self.partial_terms.dF_dexp_K_miY()
            sum_d_exp_K_miY_d_Z = self.partial_terms.dexp_K_miY_dZ()
            dF_dsum_exp_K_mi_K_im = self.partial_terms.dF_dexp_K_mi_K_im()
            sum_d_exp_K_mi_K_im_d_Z = self.partial_terms.dexp_K_mi_K_im_dZ()
            g = self.partial_terms.grad_Z(dF_dKmm, sum_d_Kmm_d_Z, dF_dsum_exp_K_miY, sum_d_exp_K_miY_d_Z,
                dF_dsum_exp_K_mi_K_im, sum_d_exp_K_mi_K_im_d_Z)
            return np.atleast_2d(g.flatten())

        maxpd, _, avgpd, _, _ = nputil.check_grad(func_wrapper, grad_wrapper, np.atleast_2d(self.Z.flatten()), 10**-4)
        self.assertTrue(maxpd < 1.0, 'max: %f\tavg:%f' % (maxpd, avgpd))


    def test_dF_dalpha(self):
        # According to GPy
        def gpy_calcs(ard):
            gkern = GPy.kern.rbf(self.Q, self.hyp.sf**2, ard, True)
            sp = GPy.core.SparseGP(self.X_mu, GPy.likelihoods.Gaussian(self.Y, self.sn**2), gkern, self.Z, self.X_S)
            sp._compute_kernel_matrices()
            sp._computations()
            gpyg = sp.dL_dtheta()
            L = sp.log_likelihood()

            return (L, gpyg)

        def our_calcs(ard):
            self.partial_terms.hyp.ard = ard
            self.partial_terms.set_data(self.Y, self.X_mu, self.X_S, is_set_statistics=True)
            self.partial_terms.update_global_statistics()

            L = self.partial_terms.logmarglik()
            dF_dKmm = self.partial_terms.dF_dKmm()
            dF_dexp_K_miY = self.partial_terms.dF_dexp_K_miY()
            dF_dexp_K_mi_K_im = self.partial_terms.dF_dexp_K_mi_K_im()
            dKmm_dalpha = self.partial_terms.dKmm_dalpha()
            dexp_K_miY_dalpha = self.partial_terms.dexp_K_miY_dalpha()
            dexp_K_mi_K_im_dalpha = self.partial_terms.dexp_K_mi_K_im_dalpha()
            g = self.partial_terms.grad_alpha(dF_dKmm, dKmm_dalpha, dF_dexp_K_miY,
                dexp_K_miY_dalpha, dF_dexp_K_mi_K_im, dexp_K_mi_K_im_dalpha) * -2 * ard**-3
            return (L, g)

        d = 10**-2

        # According to Finite Difference - GPy
        F1g, gradgpy = gpy_calcs(self.hyp.ard)
        F2g, _ = gpy_calcs(self.hyp.ard + [d, 0])
        F3g, _ = gpy_calcs(self.hyp.ard + [0, d])

        fd_gpy = np.array([(F2g - F1g) / d, (F3g - F1g) / d])

        # According to Finite Difference - ours
        F1, g = our_calcs(self.hyp.ard)
        F2, _ = our_calcs(self.hyp.ard + [d, 0])
        F3, _ = our_calcs(self.hyp.ard + [0, d])

        fd_our = np.array([(F2 - F1) / d, (F3 - F1) / d])

#         print('')
#         print 'gpy grad', gradgpy[1:]
#         print 'gpy fd  ', fd_gpy
#         print 'our fd  ', fd_our.squeeze()
#         print 'our g   ', g

        pd = np.max(np.abs((fd_our.squeeze() - g) / fd_our.squeeze())) * 100.0

        self.assertTrue(pd < 5.0, 'pd : %f' % pd)

    def test_dF_dsf2(self):
        gkern = GPy.kern.rbf(self.Q, self.hyp.sf**2, self.hyp.ard, True)
        sp = GPy.core.SparseGP(self.X_mu, GPy.likelihoods.Gaussian(self.Y, self.sn**2), gkern, self.Z, self.X_S)
        sp._compute_kernel_matrices()
        sp._computations()
        gpyg = sp.dL_dtheta()

        d = 10**-4

        dF_dKmm = self.partial_terms.dF_dKmm()
        dF_dexp_K_miY = self.partial_terms.dF_dexp_K_miY()
        dF_dexp_K_mi_K_im = self.partial_terms.dF_dexp_K_mi_K_im()
        dF_dexp_K_ii = self.partial_terms.dF_dexp_K_ii()
        dKmm_dsf2 = self.partial_terms.dKmm_dsf2()
        dexp_K_miY_dsf2 = self.partial_terms.dexp_K_miY_dsf2()
        dexp_K_mi_K_im_dsf2 = self.partial_terms.dexp_K_mi_K_im_dsf2()
        dexp_K_ii_dsf2 = self.partial_terms.dexp_K_ii_dsf2()
        g = self.partial_terms.grad_sf2(dF_dKmm, dKmm_dsf2, dF_dexp_K_ii, dexp_K_ii_dsf2,
            dF_dexp_K_miY, dexp_K_miY_dsf2, dF_dexp_K_mi_K_im, dexp_K_mi_K_im_dsf2) * 2*self.partial_terms.hyp.sf
        F1 = self.partial_terms.logmarglik()

        self.partial_terms.hyp.sf += d
        self.partial_terms.set_data(self.Y, self.X_mu, self.X_S, is_set_statistics=True)
        self.partial_terms.update_global_statistics()
        F2 = self.partial_terms.logmarglik()

        fd = (F2 - F1) / d

#         print g
#         print fd
#         print gpyg[0] * 2*self.partial_terms.hyp.sf

        pd = np.abs((g - fd) / fd)

        self.assertTrue(pd < 0.01, 'pd : %f' % pd)

    def test_dF_dbeta(self):
        d = 10**-4

        g = self.partial_terms.grad_beta()
        F1 = self.partial_terms.logmarglik()

        self.partial_terms.beta += d
        self.partial_terms.set_data(self.Y, self.X_mu, self.X_S, is_set_statistics=True)
        self.partial_terms.update_global_statistics()
        F2 = self.partial_terms.logmarglik()

        fd = (F2 - F1) / d

        pd = (fd - g) / fd * 100

        self.assertTrue(np.max(np.abs(pd)) < 1.0, 'pd : %f\ng  : %f\nfd : %f' % (np.max(np.abs(pd)), g, fd))

        gkern = GPy.kern.rbf(self.Q, self.hyp.sf**2, self.hyp.ard, True)
        sp = GPy.core.SparseGP(self.X_mu, GPy.likelihoods.Gaussian(self.Y, self.sn**2), gkern, self.Z, self.X_S)
        sp._compute_kernel_matrices()
        sp._computations()
        GPy_beta = sp.partial_for_likelihood

        pd = (g * -1 * self.sn**-4 - GPy_beta) / GPy_beta * 100

        self.assertLess(pd, 1.0, "Difference between GPy and our implementation: %f" % (pd))

#     def test_grad_Z_to_GPy(self):
#         gkern = GPy.kern.rbf(self.Q, self.hyp.sf**2, self.hyp.ard, True)
#         sp = GPy.core.SparseGP(self.X_mu, GPy.likelihoods.Gaussian(self.Y, self.sn**2), gkern, self.Z, self.X_S)
#         sp._compute_kernel_matrices()
#         sp._computations()
#         GPy_grad = sp.dL_dZ()
#
#         dF_dKmm = self.partial_terms.grad_dF_dKmm()
#         sum_d_Kmm_d_Z = self.partial_terms.grad_Z_dKmm_dZ()
#         dF_dsum_exp_K_miY = self.partial_terms.dF_dexp_K_miY()
#         sum_d_exp_K_miY_d_Z = self.partial_terms.grad_Z_dexp_K_miY_dZ()
#         dF_dsum_exp_K_mi_K_im = self.partial_terms.dF_dexp_K_mi_K_im()
#         sum_d_exp_K_mi_K_im_d_Z = self.partial_terms.grad_Z_dexp_K_mi_K_im_dZ()
#         par_grad = self.partial_terms.grad_Z(dF_dKmm, sum_d_Kmm_d_Z, dF_dsum_exp_K_miY, sum_d_exp_K_miY_d_Z,
#             dF_dsum_exp_K_mi_K_im, sum_d_exp_K_mi_K_im_d_Z)
#
#         pd = np.max((GPy_grad - par_grad) / par_grad)
#
#         self.assertLess(pd, 1.0, "Gradient w.r.t. Z differs from GPy result.")

    def test_F_to_GPy(self):
        gkern = GPy.kern.rbf(self.Q, self.hyp.sf**2, self.hyp.ard, True)
        sp = GPy.core.SparseGP(self.X_mu, GPy.likelihoods.Gaussian(self.Y, self.sn**2), gkern, self.Z, self.X_S)
        sp._compute_kernel_matrices()
        sp._computations()
        GPy_lml = sp.log_likelihood()

        lml = self.partial_terms.logmarglik()

        pd = (lml - GPy_lml) / GPy_lml * 100

        self.assertLess(pd, 1.0, "Difference between GPy and our implementation: %f" % (pd))

#     def test_GPy_gradcheck(self):
#         gkern = GPy.kern.rbf(self.Q, self.hyp.sf**2, self.hyp.ard, True)
#         sp = GPy.core.SparseGP(self.X_mu, GPy.likelihoods.Gaussian(self.Y, self.sn**2), gkern, self.Z, self.X_S)
#         sp._compute_kernel_matrices()
#         sp._computations()
#
#         def grad_wrapper(Z):
#             Z = np.reshape(Z, (self.M, self.Q))
#             sp.Z = Z
#             sp._compute_kernel_matrices()
#             sp._computations()
#             return sp.dL_dZ().flatten()
#
#         def lml_wrapper(Z):
#             Z = np.reshape(Z, (self.M, self.Q))
#             sp.Z = Z
#             sp._compute_kernel_matrices()
#             sp._computations()
#             return sp.log_likelihood()
#
#         maxpd, _, _, _, _ = nputil.check_grad(lml_wrapper, grad_wrapper, np.atleast_2d(self.Z.flatten()), 10**-6)
#
#         self.assertTrue(maxpd < 1.0, "Max pd: %f" % maxpd)

    def test_mu(self):
        def gw(d):
            return self.partial_terms.grad_X_mu().flatten()

        def fw(d):
            X_mu = np.reshape(d, self.X_mu.shape)
            self.partial_terms.set_data(self.Y, X_mu, self.X_S, is_set_statistics=True)

            return self.partial_terms.logmarglik()

        maxpd, pd, avgpd, g, fd = nputil.check_grad(fw, gw, np.atleast_2d(self.X_mu.flatten()), 10**-4)

        self.assertTrue(maxpd < 1.0, "maxpd : %f" % maxpd)

    def test_S(self):
        def gw(d):
            return self.partial_terms.grad_X_S().flatten()

        def fw(d):
            X_S = np.reshape(d, self.X_S.shape)
            self.partial_terms.set_data(self.Y, self.X_mu, X_S, is_set_statistics=True)

            return self.partial_terms.logmarglik()

        maxpd, pd, avgpd, g, fd = nputil.check_grad(fw, gw, np.atleast_2d(self.X_S.flatten()), 10**-4)

        self.assertTrue(maxpd < 1.0, "maxpd : %f" % maxpd)

rnd.seed()
suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceFunctions)
for _ in range(1):
    unittest.TextTestRunner(verbosity=2).run(suite)

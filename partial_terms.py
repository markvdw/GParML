###############################################################################
# partial_terms.py
# Calculate partial terms of the gradients and lml. In all functions here, we
# assume that the kernel expectations have already been computed.
#  - w.r.t. Z, the inducing point locations.
#  - w.r.t. alpha, the ARD hyperparameters
###############################################################################

import numpy as np
import numpy.linalg as linalg
from scipy import constants
import kernels
import kernel_exp

class partial_terms(object):
    def __init__(self, Z, sf2, alpha, beta, M, Q, N, D, update_global_statistics=True):
        '''
        Init the calculation of partial terms

        Args:

        '''
        # TODO: Take or assert M, Q, N, D from Z
        self.Z = Z

        self.M = M
        self.Q = Q
        self.N = N
        self.D = D

        self.beta = beta
        self.hyp = kernels.ArdHypers(self.Q, sf=sf2**0.5, ard=alpha**-0.5)
        self.kernel = kernels.rbf(self.Q, sf=self.hyp.sf, ard=self.hyp.ard)

        if update_global_statistics:
            self.update_global_statistics()

    def set_data(self, Y, X_mu, X_S, is_set_statistics=True):
        self.Y = Y
        self.sum_YYT = np.sum(np.array([y.dot(y) for y in self.Y]), 0)
        self.X_mu = X_mu
        self.X_S = X_S
        self.local_N = X_mu.shape[0]

        self.exp_K_mi_K_im = np.zeros((self.local_N, self.M, self.M))
        for i, (mu, s) in enumerate(zip(self.X_mu, self.X_S)):
            self.exp_K_mi_K_im[i, :, :] = kernel_exp.calc_expect_K_mi_K_im(self.Z, 
                self.hyp, np.atleast_2d(mu), np.atleast_2d(s))
        self.exp_K_mi = kernel_exp.calc_expect_K_mi(self.Z, self.hyp, self.X_mu, self.X_S)

        if is_set_statistics:
            self.update_local_statistics()

    def set_local_statistics(self, sum_YYT, sum_exp_K_mi_K_im, exp_K_miY, sum_exp_K_ii, KL):
        self.sum_YYT = sum_YYT
        self.sum_exp_K_mi_K_im = sum_exp_K_mi_K_im
        self.exp_K_miY = exp_K_miY
        self.sum_exp_K_ii = sum_exp_K_ii

        self.Kmm_plus_op_inv = linalg.inv(self.Kmm + self.beta*self.sum_exp_K_mi_K_im)
        self.KL = KL

    def get_local_statistics(self):
        return {'sum_YYT' : self.sum_YYT,
                'sum_exp_K_mi_K_im' : self.sum_exp_K_mi_K_im,
                'exp_K_miY' : self.exp_K_miY,
                'sum_exp_K_ii' : self.sum_exp_K_ii,
                'KL' : self.KL}

    def set_global_statistics(self, Kmm, Kmm_inv):
        self.Kmm = Kmm
        self.Kmm_inv = Kmm_inv

    def update_local_statistics(self):
        '''
        Update statistics for when X_mu or X_S have changed
        '''
        self.kernel = kernels.rbf(self.Q, sf=self.hyp.sf, ard=self.hyp.ard)
        self.sum_exp_K_mi_K_im = self.exp_K_mi_K_im.sum(0)
        self.exp_K_miY = kernel_exp.calc_expect_K_mi_Y(self.Z, self.hyp, self.X_mu, self.X_S, self.Y)
        self.sum_exp_K_ii = self.hyp.sf**2 * self.local_N
        self.Kmm_plus_op_inv = linalg.inv(self.Kmm + self.beta*self.sum_exp_K_mi_K_im)
        if not np.all(self.X_S == 0):
            mu_ip = np.array([x.dot(x) for x in self.X_mu])
            self.KL = 0.5 * np.sum(np.sum(self.X_S - np.log(self.X_S), 1) + mu_ip - self.Q)
        else: # We have fixed embeddings
            self.KL = 0

    def update_global_statistics(self):
        '''
        Update statistics for when Z changes
        '''
        self.kernel = kernels.rbf(self.Q, sf=self.hyp.sf, ard=self.hyp.ard)
        self.Kmm = self.kernel.K(self.Z)
        self.Kmm_inv = linalg.inv(self.Kmm)


    ###############################################################################
    # Partial gradients of F
    ###############################################################################

    def dF_dKmm(self):
        '''
        dF_dKmm

        Equations (5.7) & (5.46)
        '''
        dF_dKmm = ( 0.5*self.D*self.Kmm_inv +
                   -0.5*self.D*self.Kmm_plus_op_inv +
                   -0.5*self.beta*self.D*self.Kmm_inv.dot(self.sum_exp_K_mi_K_im.dot(self.Kmm_inv)) +
                   -0.5*self.beta**2*self.Kmm_plus_op_inv.dot(self.exp_K_miY.dot(self.exp_K_miY.T.dot(self.Kmm_plus_op_inv))))

        return dF_dKmm

    def dF_dexp_K_miY(self):
        '''
        dF_dexp_K_miY

        Eqn (5.9), or in detail, (5.27)
        '''
        return self.beta**2*self.Kmm_plus_op_inv.dot(self.exp_K_miY)

    def dF_dexp_K_mi_K_im(self):
        '''
        dF_dexp_K_mi_K_im
        Eqn (5.10), or in detail, (5.35)
        '''
        return (-0.5*self.beta*self.D*self.Kmm_plus_op_inv +
                0.5*self.beta*self.D*self.Kmm_inv +
               -0.5*self.beta**3 * (self.Kmm_plus_op_inv.dot(
                    self.exp_K_miY.dot(self.exp_K_miY.T.dot(self.Kmm_plus_op_inv)))))

    def dF_dexp_K_ii(self):
        '''
        dF_dexp_K_ii
        Equations (5.7) & (5.46)
        '''
        return -0.5 * self.beta * self.D

    ###############################################################################
    # grad_Z and necessary functions
    # Calculating the gradient of Z takes a lot of separate steps. I've split these
    # up into functions so they can all be individually tested.
    ###############################################################################

    def dKmm_dZ(self):
        '''
        grad_Z_dKmm_dZ
        Equation (5.49)

        Returns:
            MxQxM matrix. First two axes are the axes of Z, last indexes the
            non-zero elements of the derivative matrix.

        Status:
            Finished
            Tested
        '''
        alpha = 1.0 / self.hyp.ard**2

        # import time
        # t = time.time()
        # res = np.zeros((self.M, self.Q, self.M))
        # for j in xrange(self.M):
        #     for k in xrange(self.Q):
        #         for mp in xrange(self.M):
        #             res[j, k, mp] = self.kernel.K(self.Z[j, :], self.Z[mp, :])
        #             res[j, k, mp] *= -alpha[k]*(self.Z[j, k] - self.Z[mp, k])
        # print time.time() - t

        # t = time.time()
        K = self.kernel.K(self.Z, self.Z)
        res2 = K[:, None, :] * -alpha[None, :, None] * (self.Z[:, :, None] - self.Z.T[None, :, :])
        # print time.time() - t

        # assert np.sum(np.abs(res2 - res)) < 10**-12

        return res2

    def dexp_K_miY_dZ(self):
        '''
        grad_Z_dexp_K_miY_dZ
        Calculates the gradient of exp_K_mi w.r.t Z. Eqn (5.51).

        Eqn (5.9), (5.13) & (5.51)

        Args:

        Returns:


        Status:
             Confirmed correct by comparison to GPy.
        '''

        # Here, we're taking the derivative of exp_K_miY (2D matrix), with respect
        # to Z (also a 2D matrix). The result should be a 4D matrix, BUT, only one
        # vector will be non-zero. Therefore, we can summarise the whole result in
        # 3D Matrix.
        alpha = self.hyp.ard**-2

        res = np.zeros((self.M, self.Q, self.D))

        for j in range(self.M):
            for k in range(self.Q):
                n_factors = self.exp_K_mi[:, j] * alpha[k] * ((self.X_mu[:, k] - self.Z[j, k]) / (alpha[k] * self.X_S[:, k] + 1))
                res[j, k, :] = n_factors.dot(self.Y)

        return res

    def dexp_K_mi_K_im_dZ(self):
        '''
        grad_Z_dexp_K_mi_K_im_dZ
        Calculates the gradient of exp_K_mi_K_im w.r.t. Z. Eqn (5.52).

        Eqn (5.10), (5.14) & (5.52)

        Status:
             Confirmed correct by comparison to GPy.
        '''
        alpha = self.hyp.ard**-2

        # import time
        #
        # t = time.time()
        # res = np.zeros((self.M, self.Q, self.M))
        # # Need to sum over all input points
        # for n, (mu, s) in enumerate(zip(self.X_mu, self.X_S)):
        #     # Now calculate each element of the output
        #     for j in xrange(self.M):
        #         for k in xrange(self.Q):
        #             res[j, k, :] += (self.exp_K_mi_K_im[n, j, :] *
        #                              (-0.5*alpha[k]*(self.Z[j, k] - self.Z[:, k]) +
        #                                0.5*alpha[k]*(2*mu[k] - self.Z[j, k] - self.Z[:, k])/(2*alpha[k]*s[k] + 1) ))
        # print(time.time() - t)

        # t = time.time()
        res2 = np.sum( self.exp_K_mi_K_im[:, :, None, :] *
                       (-0.5*alpha[None, :, None]*(self.Z[:, :, None] - self.Z.T[None, :, :]) +
                        0.5*alpha[None, :, None]*(2.*self.X_mu[:, None, :, None] - self.Z[:, :, None] - self.Z.T[None, :, :])/(2.*alpha[None, :, None]*self.X_S[:, None, :, None] + 1)) , 0)
        # print(time.time() - t)

        # assert np.sum(np.abs(res - res2)) < 10**-13

        return res2

    def grad_Z(self, dF_dKmm, dKmm_dZ, dF_dexp_K_miY, dexp_K_miY_dZ, dF_dexp_K_mi_K_im, dexp_K_mi_K_im_dZ):
        # I think we need individual kernel_exp matrices here... So don't pass them
        # through as a sum.
        '''
        grad_Z
        Calculates the gradient of the log marginal likelihood w.r.t. Z.

        Args:

        Returns:
            MxQ matrix of gradients of the log marginal likelihood.
        '''

        # dF to store the overall result
        dF = np.zeros((self.M, self.Q))

        # Sum all the constituent parts
        for j in xrange(self.M):
            for k in xrange(self.Q):
                dKmm_dZjk = np.zeros((self.M, self.M))
                dKmm_dZjk[j, :] = dKmm_dZ[j, k, :]
                dKmm_dZjk[:, j] = dKmm_dZ[j, k, :]
                # Contribution of (5.7) - Confirmed correct by comparison to GPy,
                # though with errors up to 10**-5.
                dF[j, k] += np.sum(dF_dKmm * dKmm_dZjk)

                # Contribution of (5.9) - Confirmed correct by comparison to GPy.
                dF[j, k] += np.sum(dF_dexp_K_miY[j, :] * dexp_K_miY_dZ[j, k, :])

                # Contribution of (5.10) - Confirmed correct by comparison to GPy.
                # Multiplied by 2 based on GPy implementation. GPy.kern.kern.
                dF[j, k] += 2 * np.sum(dF_dexp_K_mi_K_im[j, :] * dexp_K_mi_K_im_dZ[j, k, :])

        return dF


    ###############################################################################
    # grad_alpha and necessary functions
    ###############################################################################

    def dKmm_dalpha(self):
        # Eqn (5.58)
        dKmm_dalpha = np.zeros((self.Q, self.M, self.M))
        for m in xrange(self.M):
            for md in xrange(self.M):
                dKmm_dalpha[:, m, md] = -0.5 * self.Kmm[m, md] * (self.Z[m, :] - self.Z[md, :])**2

        return dKmm_dalpha

    def dexp_K_miY_dalpha(self):
        alpha = self.hyp.ard**-2
        # Eqn (5.60)
        dexp_K_miY_dalpha = np.zeros((self.Q, self.M, self.D))

        for q in xrange(self.Q):
            for i in xrange(self.local_N):
                alphaS = alpha * self.X_S[i, :]
                # Correct by comparison to GPy:
                v = -0.5 * self.exp_K_mi[i, :] * (((self.X_mu[i, q] - self.Z[:, q]) / (alphaS[q] + 1.0))**2.0 + self.X_S[i, q] / (alphaS[q] + 1.0))
                dexp_K_miY_dalpha[q, :, :] += np.outer(v, self.Y[i, :])

        return dexp_K_miY_dalpha

    def dexp_K_mi_K_im_dalpha(self):
        import time
        alpha = self.hyp.ard**-2

        # Eqn (5.61)
        # TODO: Can easily vectorise (m, md) loop. Verify this first, then refactor.
        # timea = time.time()
        # dexp_K_mi_K_im_dalpha = np.zeros((self.Q, self.M, self.M))
        # for i in xrange(self.local_N):
        #     mu = self.X_mu[i, :]
        #     s = self.X_S[i, :]
        #     for q in xrange(self.Q):
        #         for m in xrange(self.M):
        #             for md in xrange(self.M):
        #                 dexp_K_mi_K_im_dalpha[q, m, md] += (self.exp_K_mi_K_im[i, m, md] *
        #                                                     (-0.25*(self.Z[m, q] - self.Z[md, q])**2 +
        #                                                      -0.25*((2.*mu[q] - self.Z[m, q] - self.Z[md, q]) / (2.*alpha[q]*s[q] + 1.))**2. +
        #                                                      -(s[q] / (2.*alpha[q]*s[q] + 1.)))
        #                                                    )
        # print (time.time() - timea)

        # Test alternative calculation
        # timec = time.time()
        dexp_K_mi_K_im_dalpha = np.sum(self.exp_K_mi_K_im[:, None, :, :] * (
                    (-0.25*np.rollaxis(self.Z[:, None, :] - self.Z[:, :], 2)**2) +
                    -0.25*np.rollaxis((2.*self.X_mu[:, None, None, :] - self.Z[:, None, :] - self.Z[:, :]) / (2.*alpha[None, None, :]*self.X_S[:, None, None, :] + 1.), 3, 1)**2
                    -np.rollaxis(self.X_S[:, None, None, :] / (2.*alpha[:]*self.X_S[0, None, None, :] + 1.), 3, 1)), 0)
        # print (time.time() - timec)

        # print (a.shape)
        # print np.sum(np.abs(a - dexp_K_mi_K_im_dalpha))

        return dexp_K_mi_K_im_dalpha

    def grad_alpha(self, dF_dKmm, dKmm_dalpha, dF_dexp_K_miY, dexp_K_miY_dalpha, dF_dexp_K_mi_K_im, dexp_K_mi_K_im_dalpha):
        # dF_dalpha = dF_dKmm (5.7) * dKmm_dalpha (5.58) +
        #             dF_dexp_K_ii * dexp_K_ii_dalpha ( = 0, 5.59) +
        #             dF_dexp_K_miY (5.27) * dexp_K_miY_dalpha (5.60) +
        #             dF_dexp_K_mi_K_im (5.35) * dexp_K_mi_K_im_dalpha (5.61)

        # Sum all the constituent parts to give the final gradient
        dF = np.zeros(self.Q)
        for q in xrange(self.Q):
            dF[q] = (np.sum(dF_dKmm * dKmm_dalpha[q, :, :]) +
                     np.sum(dF_dexp_K_miY * dexp_K_miY_dalpha[q, :, :]) +
                     np.sum(dF_dexp_K_mi_K_im * dexp_K_mi_K_im_dalpha[q, :, :]))

        return dF


    ###############################################################################
    # grad_sf2 and necessary functions
    ###############################################################################

    def dKmm_dsf2(self):
        # Eqn (5.54)
        return self.Kmm / self.hyp.sf**2

    def dexp_K_miY_dsf2(self):
        # Eqn (5.54)
        return self.exp_K_miY / self.hyp.sf**2

    def dexp_K_mi_K_im_dsf2(self):
        # Eqn (5.54)
        return 2.0 * self.sum_exp_K_mi_K_im / self.hyp.sf**2

    def dexp_K_ii_dsf2(self):
        # Eqn (5.54)
        return self.local_N

    def grad_sf2(self, dF_dKmm, dKmm_dsf2, dF_dexp_K_ii, dexp_K_ii_dsf2, dF_dexp_K_miY, dexp_K_miY_dsf2, dF_dexp_K_mi_K_im, dexp_K_mi_K_im_dsf2):
        # dF_dsf2 = dF_Kmm (5.7) * dKmm_dsf2 (5.54) +
        #           dF_dexp_K_ii (5.15) * dexp_K_ii_dsf2 (5.55) +
        #           dF_dexp_K_miY * dexp_K_miY_dsf2 (5.56) +
        #           dF_dexp_K_mi_K_im * dexp_K_mi_K_im_dsf2 (5.57)

        dF = (np.sum(dF_dKmm * dKmm_dsf2) +
              dF_dexp_K_ii * dexp_K_ii_dsf2 +
              np.sum(dF_dexp_K_miY * dexp_K_miY_dsf2) +
              np.sum(dF_dexp_K_mi_K_im * dexp_K_mi_K_im_dsf2))

        return dF


    ###############################################################################
    # grad_beta and necessary functions
    ###############################################################################

    def grad_beta(self):
        # dF_beta (5.74)
        N = self.N
        D = self.D
        beta = self.beta

        # Matrix calculated in (5.76). - a very informative name :-)
        mat576 = self.exp_K_miY.T.dot(self.Kmm_plus_op_inv.dot(self.exp_K_miY))
        if (mat576.ndim < 2):
            pass

        dF = ( 0.5*N*D/beta +
              -0.5*D*np.trace(self.Kmm_plus_op_inv.dot(self.sum_exp_K_mi_K_im)) +
              -0.5*self.sum_YYT +
              -0.5*D*self.sum_exp_K_ii +
               0.5*D*np.trace(self.Kmm_inv.dot(self.sum_exp_K_mi_K_im)) +
               beta*np.trace(mat576) +
              -0.5*beta**2*np.trace(self.exp_K_miY.T.dot(self.Kmm_plus_op_inv.dot(self.sum_exp_K_mi_K_im).dot(self.Kmm_plus_op_inv).dot(self.exp_K_miY)))
              )

        return dF


    ###############################################################################
    # grad_X_mu and grad_X_S
    ###############################################################################

    def grad_X_mu(self):
        # dF_dmu = dF_Kmm (5.7) * dKmm_dmu (= 0, 5.63) +
        #          dF_dexp_K_ii (5.15) * dexp_K_ii_dmu (= 0, 5.64) +
        #          dF_dexp_K_miY * dexp_K_miY_dmu (5.65) +
        #          dF_dexp_K_mi_K_im * dexp_K_mi_K_im_dmu (5.66) +
        #          dF_dKL * dKL_dmu (5.72)

        alpha = self.hyp.ard**-2

        # Shared partial derivatives - to be cached
        dF_dexp_K_miY = self.beta**2*self.Kmm_plus_op_inv.dot(self.exp_K_miY)
        dF_dexp_K_mi_K_im = (-0.5*self.beta*self.D*self.Kmm_plus_op_inv +
                              0.5*self.beta*self.D*self.Kmm_inv +
                             -0.5*self.beta**3 * (self.Kmm_plus_op_inv.dot(self.exp_K_miY.dot(self.exp_K_miY.T.dot(self.Kmm_plus_op_inv)))))

        dF = np.zeros((self.local_N, self.Q))
        for i in xrange(self.local_N):
            # Eqn (5.72)
            dF[i, :] += -self.X_mu[i, :]
            for q in xrange(self.Q):
                # Eqn (5.65)
                dexp_K_miY_dmu_iq = np.outer(self.exp_K_mi[i, :] * (-alpha[q]*(self.X_mu[i, q] - self.Z[:, q]) /
                                                                (alpha[q]*self.X_S[i, q] + 1.0)),
                                             self.Y[i, :])

                # Eqn (5.66)
                dexp_K_mi_K_im_dmu_iq = self.exp_K_mi_K_im[i, :, :] * -alpha[q]*(2*self.X_mu[i, q] - self.Z[:, None, q] - self.Z[None, :, q]) / (2*alpha[q] * self.X_S[i, q] + 1.0)

                dF[i, q] += (np.sum(dF_dexp_K_miY * dexp_K_miY_dmu_iq) +
                             np.sum(dF_dexp_K_mi_K_im * dexp_K_mi_K_im_dmu_iq))

        return dF

    def grad_X_S(self):
        # dF_ds = dF_Kmm (5.7) * dKmm_ds (= 0, 5.67) +
        #         dF_dexp_K_ii (5.15) * dexp_K_ii_ds (= 0, 5.68) +
        #         dF_dexp_K_miY * dexp_K_miY_ds (5.69) +
        #         dF_dexp_K_mi_K_im * dexp_K_mi_K_im_ds (5.70)
        #         dF_dKL * dKL_ds
        dF = np.zeros((self.local_N, self.Q))

        alpha = self.hyp.ard**-2

        # Shared partial derivatives - to be cached
        dF_dexp_K_miY = self.beta**2*self.Kmm_plus_op_inv.dot(self.exp_K_miY)
        dF_dexp_K_mi_K_im = (-0.5*self.beta*self.D*self.Kmm_plus_op_inv +
                              0.5*self.beta*self.D*self.Kmm_inv +
                             -0.5*self.beta**3 * (self.Kmm_plus_op_inv.dot(self.exp_K_miY.dot(self.exp_K_miY.T.dot(self.Kmm_plus_op_inv)))))

        for i in xrange(self.local_N):
            # Eqn (5.73)
            dF[i, :] = -0.5 * (1.0 - 1.0 / self.X_S[i, :])
            for q in xrange(self.Q):
                # Eqn (5.69)
                dexp_K_miY_ds_iq = np.outer(self.exp_K_mi[i, :] *
                                    ( 0.5 * ((alpha[q] * (self.X_mu[i, q] - self.Z[:, q])) / (alpha[q]*self.X_S[i, q] + 1.0))**2
                                     -0.5 * (alpha[q] / (alpha[q]*self.X_S[i, q] + 1))), self.Y[i, :])
                # Eqn (5.70)
                dexp_K_mi_K_im_ds_iq = self.exp_K_mi_K_im[i, :, :] * ( 2.0 * ((alpha[q] * (2*self.X_mu[i, q] - self.Z[:, None, q] - self.Z[None, :, q])) / (4.0*alpha[q]*self.X_S[i, q] + 2.0))**2 +
                                            (-alpha[q] / (2*alpha[q]*self.X_S[i, q] + 1.0)))

                dF[i, q] += (np.sum(dF_dexp_K_miY * dexp_K_miY_ds_iq) +
                             np.sum(dF_dexp_K_mi_K_im * dexp_K_mi_K_im_ds_iq))

        return dF

    ###############################################################################
    # Log marginal likelihood and necessary functions
    ###############################################################################
    def logmarglik(self):
        '''
        logmarglik
        Calculates the lower bound to log p(Y), the log marginal likelihood of the
        GPLVM, given the data. From the statistics calculated in parallel.

        Args:
            Kmm         : The covariance matrix of the inducing points.

        Returns:
            A single number, the log marginal likelihood.

        Status:
            Finished
            Tested
                - Corresponds with gradient w.r.t. Kmm.
        '''
        Kmm_plus_op = self.Kmm + self.beta*self.sum_exp_K_mi_K_im
        s1, Kmm_logdet = linalg.slogdet(self.Kmm)
        s2, Kmm_plus_op_logdet = linalg.slogdet(Kmm_plus_op)

        if __debug__:
            assert s1 >= 0.0
            assert s2 >= 0.0

        # Eqn (5.2)
        lml = (-0.5*self.N*self.D*np.log(2*constants.pi) +
                0.5*self.D*self.N*np.log(self.beta) +
                0.5*self.D*Kmm_logdet +
               -0.5*self.D*Kmm_plus_op_logdet +
               -0.5*self.beta*self.sum_YYT +
               -0.5*self.beta*self.D*self.sum_exp_K_ii +
                0.5*self.beta*self.D*np.trace(self.Kmm_inv.dot(self.sum_exp_K_mi_K_im)) +
                0.5*self.beta**2*np.trace(self.exp_K_miY.T.dot(linalg.inv(Kmm_plus_op).dot(self.exp_K_miY))) +
               -self.KL)
        return lml

###############################################################################
# calc_grad.py
# Take the statistics calculated in parallel (sum them, or maybe do this
# outside the class) and use them to calculate the gradients.
###############################################################################
class calc_grad(object):
    def __init__(self, exp_K_ii, exp_K_miY, sum_exp_K_mi_K_im):
        # We also need all other parameters.
        self.exp_K_ii = exp_K_ii
        self.exp_K_miY = exp_K_miY
        self.sum_exp_K_mi_K_im = sum_exp_K_mi_K_im

        self._calc_F_grads()

    def _calc_F_grads(self):
        '''
        _calc_F_grads
        Calculate the first parts of the chain of partial derivatives of F,
        i.e.:
          - dF_dexp_K_ii
          - dF_dexp_K_miY
          - dF_dexp_K_mi_K_im
        For this we need the statistics which have been calculated in parallel:
          - exp_K_ii (sum over all N)
          - exp_K_miY (sum over all N)
          - exp_K_mi_K_im (sum over all N)
        And some useful matrices derived from these:
          - Kmm_plus_op
          - Kmm_plus_op_inv
        '''
        pass
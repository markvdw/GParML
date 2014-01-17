import sys
import glob
import os
import parallel_GPLVM
sys.path.append('./tools/')
import show_embeddings
import shutil

iterations = 20

class empty:
	pass

disp_opt = empty()
disp_opt.verbose = True
disp_opt.dimension = [0, 1]
disp_opt.output_dimension = [0, 1, 2]
disp_opt.plot2d = True
disp_opt.plot3d = False
args = ['./easydata/']

# Parameters to adjust
Q = 2
num_inducing = 10

options = {}
options['input'] = './easydata/inputs/'
options['embeddings'] = './easydata/embeddings/'
options['parallel'] = 'local'
options['statistics'] = './easydata/tmp'
options['tmp'] = './easydata/tmp'
options['M'] = num_inducing
options['Q'] = 2
options['D'] = 3
options['keep'] = False
options['init'] = 'PCA'
options['fixed_beta'] = False

filelist = glob.glob("./easydata/embeddings/*")
for f in filelist:
    os.remove(f)

options['fixed_embeddings'] = False
options['iterations'] = 0
options['load'] = False
parallel_GPLVM.main(options)

shutil.rmtree('./easydata/embeddings_GPy_init')
shutil.copytree('./easydata/embeddings', './easydata/embeddings_GPy_init')

print('')
print('>>>> NOW STARTING THE OPTIMISATION FOR PARALLEL-GPLVM')
print('')

options['iterations'] = iterations
options['load'] = False
options['fixed_embeddings'] = False
parallel_GPLVM.main(options)

#show_embeddings.run(disp_opt, args)


import numpy
import scipy

####################################################################################################################
# Debug comparison to GPy
####################################################################################################################
import GPy
gkern = GPy.kern.rbf(options['Q'], 1, numpy.array([1,1]), True)

X_mu = numpy.concatenate((
    scipy.load('./easydata/embeddings_GPy_init/easy_1.embedding.npy'),
    scipy.load('./easydata/embeddings_GPy_init/easy_2.embedding.npy'),
    scipy.load('./easydata/embeddings_GPy_init/easy_3.embedding.npy'),
    scipy.load('./easydata/embeddings_GPy_init/easy_4.embedding.npy')))

X_S = numpy.clip(numpy.ones(X_mu.shape) * 0.5,
                    0.001, 1)

Y = numpy.concatenate((
    numpy.genfromtxt('./easydata/inputs/easy_1', delimiter=','),
    numpy.genfromtxt('./easydata/inputs/easy_2', delimiter=','),
    numpy.genfromtxt('./easydata/inputs/easy_3', delimiter=','),
    numpy.genfromtxt('./easydata/inputs/easy_4', delimiter=',')))


sp = GPy.models.BayesianGPLVM(GPy.likelihoods.Gaussian(Y, 1), options['Q'],
                              X_mu, X_S, num_inducing=options['M'], Z=X_mu[10:20], kernel=gkern)
# sp = GPy.core.SparseGP(X_mu, GPy.likelihoods.Gaussian(Y, 1))
#sp = GPy.models.SparseGPRegression(X_mu, Y, gkern, Z=X_mu[:10], num_inducing=options['M'], X_variance=X_S)
#sp.ensure_default_constraints() -- doesn't work
sp.optimize('scg', max_iters=iterations)

GPy_lml = sp.log_likelihood()
GPy_grad = sp._log_likelihood_gradients()
# dF_dmu = GPy_grad[0:(options['N'] * options['Q'])].reshape(options['N'], options['Q'])
# dF_ds = GPy_grad[(options['N'] * options['Q']):2*(options['N'] * options['Q'])].reshape(options['N'], options['Q'])
# dF_dZ = GPy_grad[2*(options['N'] * options['Q']):2*(options['N'] * options['Q'])+(options['M']*options['Q'])].reshape(options['M'], options['Q'])
# dF_dsigma2 = GPy_grad[2*(options['N'] * options['Q'])+(options['M']*options['Q'])]
# dF_dalpha = GPy_grad[2*(options['N'] * options['Q'])+(options['M']*options['Q'])+1:2*(options['N'] * options['Q'])+(options['M']*options['Q'])+3]

beta = scipy.load('./easydata/tmp/global_statistics_beta_f.npy')
alpha = scipy.load('./easydata/tmp/global_statistics_alpha_f.npy')
sf2 = scipy.load('./easydata/tmp/global_statistics_sf2_f.npy')
Z = scipy.load('./easydata/tmp/global_statistics_Z_f.npy')
F = scipy.load('./easydata/tmp/partial_derivatives_F_f.npy')

X_mu = numpy.concatenate((
    scipy.load('./easydata/embeddings/easy_1.embedding.npy'),
    scipy.load('./easydata/embeddings/easy_2.embedding.npy'),
    scipy.load('./easydata/embeddings/easy_3.embedding.npy'),
    scipy.load('./easydata/embeddings/easy_4.embedding.npy')))

X_S = numpy.concatenate((
    scipy.load('./easydata/embeddings/easy_1.variance.npy'),
    scipy.load('./easydata/embeddings/easy_2.variance.npy'),
    scipy.load('./easydata/embeddings/easy_3.variance.npy'),
    scipy.load('./easydata/embeddings/easy_4.variance.npy')))

print('All the stuff from PARALLEL GPLVM!')
print(F)
print(beta)
print(alpha)
print(sf2)
print(Z)
print X_mu[:20, :]
print X_S[:20, :]

print('')

print('All the stuff from GPy!')
print(sp.log_likelihood())
print(sp.likelihood.precision)
print(sp.kern.parts[0].lengthscale**-2)
print(sp.kern.parts[0].variance)
print(sp.Z)
print(sp.X[:20, :])
print(sp.X_variance[:20, :])

print('')

print('Difference between the two')
print(F - sp.log_likelihood())
print(beta - sp.likelihood.precision)
print(alpha - sp.kern.parts[0].lengthscale**-2)
print(sf2 - sp.kern.parts[0].variance)
print(Z - sp.Z)
print X_mu[:20, :] - sp.X[:20, :]
print X_S[:20, :] - sp.X_variance[:20, :]
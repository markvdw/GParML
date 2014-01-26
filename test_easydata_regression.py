import sys
import glob
import os
import parallel_GPLVM
sys.path.append('./tools/')
import tools.easy_dataset as easy_dataset
import tools.split_data as split_data
import shutil

iterations = 100
path = './easydata_regression/'
P = 4
N = 1000
dname = 'easydata_regression'
SVI_batchsize = 100
SVI_iterations = [iterations]

class empty:
    pass

disp_opt = empty()
disp_opt.verbose = True
disp_opt.dimension = [0, 1]
disp_opt.output_dimension = [0, 1, 2]
disp_opt.plot2d = True
disp_opt.plot3d = False
args = [path]

# Parameters to adjust
M = 20
Q = 1
D = 3

options = {}
options['input'] = './easydata_regression/inputs/'
options['embeddings'] = './easydata_regression/embeddings/'
options['parallel'] = 'local'
options['statistics'] = './easydata_regression/tmp'
options['tmp'] = './easydata_regression/tmp'
options['M'] = M
options['Q'] = Q
options['D'] = D
options['keep'] = False
options['init'] = 'PCA'
options['fixed_beta'] = False
options['optimiser'] = 'SCG_adapted'
options['drop_out_fraction'] = 0

split_data.clean_dir(path)
Y, X = easy_dataset.gen_easydata(N, Q, D)
perm = split_data.split_data(Y[:N, :], P, path, dname)
split_data.split_embeddings(X, P, path, dname, init_variance=0.0, perm=perm)

options['iterations'] = iterations
options['load'] = False
options['fixed_embeddings'] = True
options['local_no_pool'] = True
#parallel_GPLVM.main(options)



####################################################################################################################
# Debug comparison to GPy
####################################################################################################################
import numpy
import scipy
sys.path.append('../GPy-master_20140118')
import GPy
gkern = GPy.kern.rbf(options['Q'], 1, numpy.array([1]), True)

X_mu = numpy.concatenate((
    scipy.load('./easydata_regression/embeddings/easydata_regression_1.embedding.npy'),
    scipy.load('./easydata_regression/embeddings/easydata_regression_2.embedding.npy'),
    scipy.load('./easydata_regression/embeddings/easydata_regression_3.embedding.npy'),
    scipy.load('./easydata_regression/embeddings/easydata_regression_4.embedding.npy')))


Y = numpy.concatenate((
    numpy.genfromtxt('./easydata_regression/inputs/easydata_regression_1', delimiter=','),
    numpy.genfromtxt('./easydata_regression/inputs/easydata_regression_2', delimiter=','),
    numpy.genfromtxt('./easydata_regression/inputs/easydata_regression_3', delimiter=','),
    numpy.genfromtxt('./easydata_regression/inputs/easydata_regression_4', delimiter=',')))


def cb(foo):
    pass

Z = numpy.linspace(X_mu.min(),X_mu.max(),M)[:,None]
m = GPy.models.SVIGPRegression(X_mu,Y, batchsize=SVI_batchsize, Z=Z, kernel=gkern)
#m.ensure_default_constraints()

# Makes results even worse:
#m.constrain_fixed('noise_variance')
#m.constrain_fixed('.*lengthscale')
#m.optimize(SVI_iterations[0], callback=cb, callback_interval=1)

m.constrain_bounded('noise_variance',1e-3,1e-1)
#m.constrain_bounded('.*lengthscale',1e-3,1e0)
m.optimize(SVI_iterations[0], callback=cb, callback_interval=1)


GPy_lml = m.log_likelihood()
GPy_grad = m._log_likelihood_gradients()

beta = scipy.load('./easydata_regression/tmp/global_statistics_beta_f.npy')
alpha = scipy.load('./easydata_regression/tmp/global_statistics_alpha_f.npy')
sf2 = scipy.load('./easydata_regression/tmp/global_statistics_sf2_f.npy')
Z = scipy.load('./easydata_regression/tmp/global_statistics_Z_f.npy')
F = scipy.load('./easydata_regression/tmp/partial_derivatives_F_f.npy')

print('All the stuff from PARALLEL GPLVM!')
print(F)
print(beta)
print(alpha)
print(sf2)
print(Z)

print('')

print('All the stuff from GPy!')
print(m.log_likelihood())
print(m.likelihood.precision)
print(m.kern.parts[0].lengthscale**-2)
print(m.kern.parts[0].variance)
print(m.Z)

print('')

print('Difference between the two')
print(F - m.log_likelihood())
print(beta - m.likelihood.precision)
print(alpha - m.kern.parts[0].lengthscale**-2)
print(sf2 - m.kern.parts[0].variance)
print(Z - m.Z)
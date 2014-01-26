import numpy
import sys
sys.path.append('../GPy-master_20140118')
import GPy
import tools.easy_dataset as easy_dataset

M = 20
N = 1000
Q = 1
D = 3
Y, Xt = easy_dataset.gen_easydata(N, Q, D)
Z = numpy.linspace(Xt.min(),Xt.max(),M)[:,None]

def cb(foo):
    pass

gkern = GPy.kern.rbf(Q, 1, numpy.array([1]), True)
m = GPy.models.SVIGPRegression(Xt,Y, batchsize=10, Z=Z, kernel=gkern)
m.ensure_default_constraints()

m.constrain_bounded('noise_variance',1e-3,1e-1)
m.optimize(100, callback=cb, callback_interval=1)

# Makes things even worse:
#m.unconstrain('noise_variance')
#m.ensure_default_constraints()
#m.optimize(100, callback=cb, callback_interval=1)

# Should be +560
m.log_likelihood()


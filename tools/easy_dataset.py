###########################################################################
# easy_dataset.py
# Generate an easy dataset that is well initialised with PCA.
###########################################################################

import os
import numpy as np
import numpy.random as rnd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def gen_easydata(N, Q, D):
    X = rnd.randn(N, Q)
    Y = np.zeros((N, D))

    for d in xrange(D):
        # For every column in Y, i.e. dimension d for all datapoints.
        s = rnd.randn()
        Y[:, d] = (s*1.1 * X + np.sin(s * X + rnd.randn() * 5.0)).squeeze()

    Y += rnd.randn(N, D) * 0.05

    return (Y, X)

if __name__ == '__main__':
    Y, _ = gen_easydata(100, 1, 3)

    # Output data
    if not os.path.exists('../easydata/'):
        os.mkdir('../easydata/')
        os.mkdir('../easydata/inputs')
        os.mkdir('../easydata/embeddings/')
        os.mkdir('../easydata/tmp/')
        os.mkdir('../easydata/proc')

    np.savetxt('../easydata/proc/easy', Y, delimiter=',')
    os.system('python ../pre_process.py ../easydata/proc/easy 4')
    os.system('mv ../easydata/proc/easy_* ../easydata/inputs/')

    # Visualise data
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], 'x')

    plt.show()
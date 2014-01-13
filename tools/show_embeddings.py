#!/usr/bin/python
###########################################################################
# show_embeddings.py
# Display the X found by the Parallel GPLVM implementation.
###########################################################################

import itertools
import os
import glob
from optparse import OptionParser

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages

def run(opt, args):
    verbose = opt.verbose

    def vprint(s):
        if verbose:
            print(s)

    if (len(args) == 0):
        vprint('No X directory...')
        exit()
    elif (len(args) != 1):
        vprint('Ignoring superfluous positional arguments...')

    embdir = args[0]

    if not os.path.exists(embdir):
        print('Error: X path does not exist...')
        exit()

    # Latent space dimensions to plot
    dims = opt.dimension

    vprint("Displaying X in '%s'..."% embdir)

    ###########################################################################
    # Read the data
    ###########################################################################
    # Read embeddings
    embfiles = sorted(glob.glob(embdir + '/embeddings/*.embedding*'))
    varfiles = sorted(glob.glob(embdir + '/embeddings/*.variance*'))

    parGPLVM_X = None
    for emb, var in itertools.izip(embfiles, varfiles):
        embmat = sp.load(emb)
        if (parGPLVM_X is None):
            parGPLVM_X = embmat
        else:
            parGPLVM_X = np.vstack((parGPLVM_X, embmat))

    # Read the input data
    Y = None
    input_files = sorted(glob.glob(embdir + '/inputs/*'))
    for input_file in input_files:
        partialY = np.genfromtxt(input_file, delimiter=',')
        if (not Y is None):
            Y = np.vstack((Y, partialY))
        else:
            Y = partialY

    # Read relevant global statistics
    alpha_file = sorted(glob.glob(embdir + '/tmp/global_statistics_alpha_*'))[-1]
    beta_file = sorted(glob.glob(embdir + "/tmp/global_statistics_beta_*"))[-1]
    sf2_file = sorted(glob.glob(embdir + "/tmp/global_statistics_sf2_*"))[-1]

    alpha = sp.load(alpha_file).squeeze()
    beta = sp.load(beta_file).squeeze()
    sf2 = sp.load(sf2_file).squeeze()

    # Output global statistics
    print('alpha:'),
    print(alpha)
    print('beta :'),
    print(beta)
    print('sf2  :'),
    print(sf2)

    ###########################################################################
    # Visualise the data
    ###########################################################################
    if (dims is None):
        dims = alpha.argsort()[-2:]
    elif (len(dims) == 1):
        dims.append(np.argmax(alpha))

    pp = PdfPages('easydata.pdf')
    if (opt.plot2d):
        # Plot the X
        fig = plt.figure()
        plt.plot(parGPLVM_X[:, dims[0]], parGPLVM_X[:, dims[1]], 'x')
        pp.savefig(fig)
        plt.title('First two dimensions of the latent space.')

        for dy in opt.output_dimension:
            fig = plt.figure()
            plt.plot(parGPLVM_X[:, dims[0]], Y[:, dy], 'x')
            plt.title('First latent space dim vs data %i' % dy)


    # Plot the outputs as a func of the X
    # if (not opt.output_dimension is None):
    #     for dy in opt.output_dimension:
    #         fig = plt.figure()
    #         ax = fig.gca(projection='3d')
    #         ax.plot(X[:, dims[0]], X[:, dims[1]], Y[:, dy], 'x')
    #         ax.set_xlabel('Embedding dim %u' % dims[0])
    #         ax.set_ylabel('Embedding dim %u' % dims[1])
    #         ax.set_zlabel('Data dim %u' % dy)
    #         pp.savefig(fig)

    pp.close()

    if (opt.plot3d):
        vprint('3D plotting not implemented yet.')

    plt.show()

    return parGPLVM_X

if __name__ == '__main__':
    usage = "usage: %prog [options] data_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('-v', '--verbose', action='store_true', dest='verbose', help='Print output messages.', default=False)
    parser.add_option('-d', '--dimension', action='append', type='int', help='Embedding dimensions to display (max: 2/3).')
    parser.add_option('-y', '--output_dimension', action='append', type='int', help='Output dimensions to display.')
    parser.add_option('--plot2d', action='store_true', dest='plot2d', help='Plot things in 2D (default).', default=True)
    parser.add_option('--plot3d', action='store_true', dest='plot3d', help='Plot things in 3D.', default=False)

    (opt, args) = parser.parse_args()

    run(opt, args)

###########################################################################
# output_oilflow.py
# Saves the oilflow dataset to a file from GPy in a way that can be dealt
# with by pre_process.py.
###########################################################################

import sys
import numpy as np
import GPy

outfile = sys.argv[1]

print outfile

oildata = GPy.util.datasets.oil()
np.savetxt(outfile, oildata['X'], delimiter=',')

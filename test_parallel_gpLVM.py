import sys
import glob
import os
import parallel_GPLVM
sys.path.append('./tools/')
import show_embeddings

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
options['fixed_embeddings'] = False
options['keep'] = False
options['init'] = 'PCA'

filelist = glob.glob("./easydata/embeddings/*")
for f in filelist:
    os.remove(f)

options['iterations'] = 100
options['load'] = False
options['fixed_beta'] = False
parallel_GPLVM.main(options)

options['load'] = True
options['fixed_beta'] = False
for i in xrange(20):
	parallel_GPLVM.main(options)

show_embeddings.run(disp_opt, args)

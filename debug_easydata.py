import sys
import os
import glob
import parallel_GPLVM
sys.path.append('./tools/')
import show_embeddings

path = './easydata/'

# Parameters to adjust
Q = 2
num_inducing = 10

# First delete all current inputs & embeddings
filelist = glob.glob(path + "/inputs/*")
filelist.extend(glob.glob(path + "/embeddings/*"))
for f in filelist:
    os.remove(f)

options = {}
options['input'] = path + '/inputs/'
options['embeddings'] = path + '/embeddings/'
options['parallel'] = 'local'
options['iterations'] = 5
options['statistics'] = path + '/tmp'
options['tmp'] = path + '/tmp'
options['M'] = num_inducing
options['Q'] = Q
options['D'] = 3
options['fixed_embeddings'] = False
options['keep'] = True

filelist = (glob.glob(path + "/embeddings/*"))
for f in filelist:
    os.remove(f)

parallel_GPLVM.main(options)

class empty:
    pass
disp_opt = empty()
disp_opt.verbose = True
disp_opt.dimension = [0, 1]
disp_opt.output_dimension = [0, 1, 2]
disp_opt.plot2d = True
disp_opt.plot3d = False
args = [path]
show_embeddings.run(disp_opt, args)
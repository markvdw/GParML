# Copy the input data to a bunch of files.
P = 4
path = '../easydata/'

# First delete all current inputs & embeddings
filelist = glob.glob(path + "/inputs/*")
filelist.extend(glob.glob(path + "/embeddings/*"))
for f in filelist:
    os.remove(f)

# Open files for writing the divided dataset into
f = []
for p in xrange(1, P + 1):
    name = path + 'inputs/easy_' + str(p)
    f.append(open(name, 'w'))

# Divide up dataset
for y in Y:
    x_str = ",".join(np.char.mod('%f', y))
    randf = random.choice(f)
    randf.write(x_str)
    randf.write('\n')    
    
for fi in f:
    fi.close()

# Run the Parallel GPLVM
options = {}
options['input'] = path + '/inputs/'
options['embeddings'] = path + '/embeddings/'
options['parallel'] = 'local'
options['iterations'] = 100
options['statistics'] = path + '/tmp'
options['tmp'] = path + '/tmp'
options['M'] = num_inducing
options['Q'] = Q
options['D'] = 3
options['fixed_embeddings'] = False
options['keep'] = True
options['load'] = False
options['fixed_beta'] = True
options['init'] = 'PCA'
options['optimiser'] = 'SCG_adapted'
options['fixed_beta'] = False

import parallel_GPLVM
reload(parallel_GPLVM)

parallel_GPLVM.main(options)
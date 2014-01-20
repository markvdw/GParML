import glob
import os
import numpy as np

def clean_dir(path):
    filelist = glob.glob(path + "/inputs/*")
    filelist.extend(glob.glob(path + "/embeddings/*"))
    for f in filelist:
        os.remove(f)


def split_data(Y, P, path, dname):
    '''
    split_data
    Splits the data from one dataset into several different files for use with the parallel GPLVM.
    '''
    f = []
    for p in xrange(1, P+1):
        name = path + 'inputs/' + dname + '_' + str(p)
        f.append(open(name, 'w'))

    perm = np.random.permutation(Y.shape[0])
    for idx in perm:
        y = Y[idx, :]
        x_str = ','.join(np.char.mod('%f', y))
        outf = f[idx % P]
        outf.write(x_str)
        outf.write('\n')

    for fi in f:
        fi.close()
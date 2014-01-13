#!/usr/bin/python
'''
parallel_GPLVM.py
Main script to run, implements parallel inference for GPLVM for SGE (Sun Grid
Engine), Hadoop (Map Reduce framework), and a local parallel implementation.

Arguments:
-i, --input
    Folder containing files to be processed. One file will be processed per node. Files assumed to be in a comma-separated-value (CSV) format. (required))
-e, --embeddings
    Existing folder to store embeddings in. One file will be created for each input file. (required)
-p, --parallel
    Which parallel architecture to use (local (default), Hadoop, SGE)
-T, --iterations
    Number of iterations to run; default value is 100
-s, --statistics
    Folder to store statistics files in (default is /tmp)
-k, --keep
    Whether to keep statistics files or to delete them
-l, --load
    Whether to load statistics and embeddings from previous run or initialise new ones
-t, --tmp
    Shared folder to store tmp files in (default is /scratch/tmp)
--init
    Which Initialisation to use (PCA (default), PPCA (probabilistic PCA), FA (factor analysis), random)

Sparse GPs specific options
-M, --inducing_points
    Number of inducing points (default: 10)
-Q, --latent_dimensions
    umber of latent dimensions (default: 10)
-D, --output_dimensions
    Number of output dimensions given in Y (default value set to 10)
--fixed_embeddings
    If given, embeddings (X) are treated as fixed. Only makes sense when embeddings are given in the folder in advance
--fixed_beta
    If given, beta is treated as fixed.

SGE specific options
--simplejson
    SGE simplejson location

Hadoop specific options
--hadoop
    Hadoop folder
--jar
    Jar file for Hadoop streaming
'''
from optparse import OptionParser, OptionGroup
import os.path
import sys
import scipy
import numpy
from numpy import genfromtxt
import time
import random
import subprocess
import glob
import partial_terms as pt
import scg_adapted
from scg_adapted import SCG_adapted

options = {}
map_reduce = {}
def main(opt_param = None):
    global options, map_reduce

    if opt_param is None:
        options = parse_options()
    else:
        options = opt_param

    # Initialise the Map-Reduce parser
    if options['parallel'] == "local":
        import local_MapReduce
        map_reduce = local_MapReduce
    elif options['parallel'] == "SGE":
        import SGE_MapReduce
        map_reduce = SGE_MapReduce
    elif options['parallel'] == "Hadoop":
        raise Exception("Not implemented yet!")
    options = map_reduce.init(options)

    options, global_statistics = init_statistics(map_reduce, options)
    # Run the optimiser for T steps
    x0 = flatten_global_statistics(options, global_statistics)
    # Transform the positiv parameters to be in the range (-Inf, Inf)
    x0 = numpy.array([transform_back(b, x) for b, x in zip(options['flat_global_statistics_bounds'], x0)])
    # Todo: Nothing seems to be done with the end result of the global optimisation?
    x_opt = SCG_adapted(likelihood_and_gradient, local_optimisation, x0,
        display=True, maxiters=options['iterations'])
    
    flat_array = x_opt[0]
    # Transform the parameters that have to be positive to be positive
    flat_array_transformed = numpy.array([transform(b, x) for b, x in 
        zip(options['flat_global_statistics_bounds'], flat_array)])
    global_statistics = rebuild_global_statistics(options, flat_array_transformed)
    print 'Final global_statistics'
    print global_statistics

    # Clean unneeded files
    options['iteration'] = len(x_opt[1]) - 1
    clean(options)
    # We need to call this one last time to make sure that the search did not try any other 
    # vaues for the global statistics before finishing - ie the files are out of date
    likelihood_and_gradient(flat_array, 'f')
    file_name = options['statistics'] + '/partial_derivatives_F_f.npy'
    ''' We still have a bug where the reported F is not the same one as the one returned from test() '''
    print 'final F=' + str(float(map_reduce.load(file_name)))


def init_statistics(map_reduce, options):
    '''
    Initialise statistics and names of variables passed-around
    '''
    # Initialise the statistics that need to be handled on the master node
    options['global_statistics_names'] = {
        'Z' : (options['M'], options['Q']), 'sf2' : (1,1), 'alpha' : (1, options['Q']), 'beta' : (1,1)
    }
    options['accumulated_statistics_names'] = [
        'sum_YYT', 'sum_exp_K_mi_K_im', 'sum_exp_K_miY', 'sum_exp_K_ii', 'sum_KL',
        'sum_d_exp_K_miY_d_Z',
        'sum_d_exp_K_mi_K_im_d_Z',
        'sum_d_exp_K_miY_d_alpha',
        'sum_d_exp_K_mi_K_im_d_alpha',
        'sum_d_exp_K_ii_d_sf2',
        'sum_d_exp_K_miY_d_sf2',
        'sum_d_exp_K_mi_K_im_d_sf2'
    ]
    options['partial_derivatives_names'] = [
        'F', 'dF_dsum_exp_K_ii', 'dF_dKmm', 'dF_dsum_exp_K_miY', 'dF_dsum_exp_K_mi_K_im'
    ]
    options['cache_names'] = [
        'Kmm', 'Kmm_inv'
    ]

    # Initialise the global statistics to defaults
    if not options['load']:
        # Initialise Z (locations of inducing points)
        input_files_names = os.listdir(options['input'] + '/')
        input_files_names = sorted(input_files_names)
        embedding_name = options['embeddings'] + '/' + input_files_names[0] + '.embedding.npy'
        embeddings = map_reduce.load(embedding_name)
        # untested:
        if numpy.allclose(embeddings, embeddings.mean(axis=0)):
            print 'Warning: all embeddings identical'
        if embeddings.shape[0] < options['M']:
            raise Exception('Current implementation requires first file in the inputs folder (alphabetically) to have at least M data points')
        if not embeddings.shape[1] == options['Q']:
            raise Exception('Given Q does not equal existing embedding data dimensions!')
        # init embeddings using k-means (gives much better guess)
        import scipy.cluster.vq as cl
        Z = cl.kmeans(embeddings, options['M'])[0]
        while Z.shape[0] < options['M']:
            Z = cl.kmeans(embeddings, options['M'])[0]
        Z += scipy.randn(options['M'], options['Q']) * 0.1

        # Initialise the global statistics
        global_statistics = {
            'Z' : Z, # see GPy models/bayesian_gplvm.py
            'sf2' : numpy.array([[1.0]]), # see GPy kern/rbf.py
            'alpha' : scipy.ones((1, options['Q'])), # see GPy kern/rbf.py
            'beta' : numpy.array([[2.0]]) # see GPy likelihood/gaussian.py
        }
    else:
        # Load global statistics from previous run
        global_statistics = {}
        for key in options['global_statistics_names']:
            file_name = options['statistics'] + '/global_statistics_' + key + '_f.npy'
            global_statistics[key] = map_reduce.load(file_name)

    # Initialise bounds for optimisation
    global_statistics_bounds = {
        'Z' : [(None, None) for i in range(options['M'] * options['Q'])],
        'sf2' : [(0, None)],
        'alpha' : [(0, None) for i in range(options['Q'])],
        'beta' : [(0, None)]
    }
    flat_global_statistics_bounds = []
    for key, statistic in global_statistics_bounds.items():
        flat_global_statistics_bounds = flat_global_statistics_bounds+statistic
    options['flat_global_statistics_bounds'] = flat_global_statistics_bounds

    return options, global_statistics



'''
Calculate the likelihood and derivatives by sending jobs to the nodes
'''

def likelihood_and_gradient(flat_array, iteration):
    global options, map_reduce
    flat_array_transformed = numpy.array([transform(b, x) for b, x in 
        zip(options['flat_global_statistics_bounds'], flat_array)])
    global_statistics = rebuild_global_statistics(options, flat_array_transformed)
    options['i'] = iteration
    #print 'global_statistics'
    #print global_statistics

    # Clear unneeded files from previous iteration if we don't want to keep them. 
    clean(options)

    # Save into shared files so all node can access them
    for key in global_statistics.keys():
        file_name = options['statistics'] + '/global_statistics_' + key + '_' + str(options['i']) + '.npy'
        map_reduce.save(file_name, global_statistics[key])

    # Dispatch statistics Map-Reduce
    #print "Dispatching statistics Map-Reduce..."
    start = time.time()
    # Cache matrices that only need be calculated once
    map_reduce.cache(options, global_statistics)
    accumulated_statistics_files = map_reduce.statistics_MR(options)
    end = time.time()
    print "Done! statistics Map-Reduce took ", int(end - start), " seconds"

    # Calculate global statistics
    #print "Calculating global statistics..."
    start = time.time()
    partial_derivatives, accumulated_statistics, partial_terms = calculate_global_statistics(options,
        global_statistics, accumulated_statistics_files, map_reduce)
    end = time.time()
    print "Done! global statistics took ", int(end - start), " seconds"

    # Evaluate the gradient for 'Z', 'sf2', 'alpha', and 'beta'
    gradient = calculate_global_derivatives(options, partial_derivatives,
        accumulated_statistics, global_statistics, partial_terms)
    gradient = flatten_global_statistics(options, gradient)
    likelihood = partial_derivatives['F']

    gradient = numpy.array([g * transform_grad(b, x) for b, x, g in 
        zip(options['flat_global_statistics_bounds'], flat_array, gradient)])
    return -1 * likelihood, -1 * gradient


def local_optimisation(flat_array):
    if not options['fixed_embeddings']:
        flat_array_transformed = numpy.array([transform(b, x) for b, x in 
            zip(options['flat_global_statistics_bounds'], flat_array)])
        global_statistics = rebuild_global_statistics(options, flat_array_transformed)
    
        # Save into shared files so all node can access them
        for key in global_statistics.keys():
            file_name = options['statistics'] + '/global_statistics_' + key + '_' + str(options['i']) + '.npy'
            map_reduce.save(file_name, global_statistics[key])

        # Dispatch embeddings Map-Reduce to run in background if we're not using fixed embeddings
        #print "Dispatching embeddings Map-Reduce to run in background..."
        start = time.time()
        script_files = map_reduce.embeddings_MR(options)

        # Wait for embeddings Map-Reduce to finish if we're not using fixed embeddings
        #print "Waiting for embeddings Map-Reduce to finish..."
        map_reduce.embeddings_watcher(options, script_files)
        end = time.time()
        print "Done! embeddings Map-Reduce took ", int(end - start), " seconds"


'''
Supporting functions to pass the parameters in and out of the optimiser, and to calculate global statistics
'''

def flatten_global_statistics(options, global_statistics):
    flat_array = numpy.array([])
    for key, statistic in global_statistics.items():
        flat_array = numpy.concatenate((flat_array, statistic.flatten()))
    return flat_array

def rebuild_global_statistics(options, flat_array):
    global_statistics = {}
    start = 0
    for key, shape in options['global_statistics_names'].items():
        size = shape[0] * shape[1]
        global_statistics[key] = flat_array[start:start+size].reshape(shape)
        start = start + size
    return global_statistics


lim_val = -numpy.log(sys.float_info.epsilon) 
# Transform a parameter to be in (0, inf) if the bound constrains it to be positive 
def transform(b, x):
    if b == (0, None):
        if x > lim_val:
            return x
        elif x < -lim_val:
            return numpy.log(1 + numpy.exp(-lim_val))
        else:
            return numpy.log(1 + numpy.exp(x))
    elif b == (None, None):
        return x

# Transform a parameter back to be in (-inf, inf) if the bound constrains it to be positive 
def transform_back(b, x):
    if b == (0, None):
        if x > lim_val:
            return x
        elif x < -lim_val:
            return numpy.log(-1 + numpy.exp(-lim_val))
        else:
            return numpy.log(-1 + numpy.exp(x))
    elif b == (None, None):
        return x

# Gradient of the (0, inf) transform if the bound constrains it to be positive 
def transform_grad(b, x):
    if b == (0, None):
        if x > lim_val:
            #return x
            return 1
        elif x < -lim_val:
            return 1 - numpy.exp(lim_val)
        else:
            return 1 - numpy.exp(-x)
        #return 1. - 1. / (1 + numpy.exp(x))
    elif b == (None, None):
        return 1

def calculate_global_statistics(options, global_statistics, accumulated_statistics_files, map_reduce):
    '''
    Loads statistics into dictionaries and calculates global statistics such as F and partial derivatives of F
    '''
    # Load accumulated statistics
    accumulated_statistics = {}
    for (statistic, file_name) in accumulated_statistics_files:
        accumulated_statistics[statistic] = map_reduce.load(file_name)

    # Get F and partial derivatives for F
    partial_terms = map_reduce.load_partial_terms(options, global_statistics)
    # Load cached matrices
    map_reduce.load_cache(options, partial_terms)

    partial_terms.set_local_statistics(accumulated_statistics['sum_YYT'],
        accumulated_statistics['sum_exp_K_mi_K_im'],
        accumulated_statistics['sum_exp_K_miY'],
        accumulated_statistics['sum_exp_K_ii'],
        accumulated_statistics['sum_KL'])

    partial_derivatives = {
        'F' : partial_terms.logmarglik(),
        'dF_dsum_exp_K_ii' : partial_terms.dF_dexp_K_ii(),
        'dF_dsum_exp_K_miY' : partial_terms.dF_dexp_K_miY(),
        'dF_dsum_exp_K_mi_K_im' : partial_terms.dF_dexp_K_mi_K_im(),
        'dF_dKmm' : partial_terms.dF_dKmm()
    }

    for key in partial_derivatives.keys():
        file_name = options['statistics'] + '/partial_derivatives_' + key + '_' + str(options['i']) + '.npy'
        map_reduce.save(file_name, partial_derivatives[key])

    return partial_derivatives, accumulated_statistics, partial_terms

def calculate_global_derivatives(options, partial_derivatives, accumulated_statistics, global_statistics, partial_terms):
    '''
    Evaluate the gradient for 'Z', 'sf2', 'alpha', and 'beta'
    '''
    grad_Z = partial_terms.grad_Z(partial_derivatives['dF_dKmm'],
        partial_terms.dKmm_dZ(),
        partial_derivatives['dF_dsum_exp_K_miY'],
        accumulated_statistics['sum_d_exp_K_miY_d_Z'],
        partial_derivatives['dF_dsum_exp_K_mi_K_im'],
        accumulated_statistics['sum_d_exp_K_mi_K_im_d_Z'])
    grad_alpha = partial_terms.grad_alpha(partial_derivatives['dF_dKmm'],
        partial_terms.dKmm_dalpha(),
        partial_derivatives['dF_dsum_exp_K_miY'],
        accumulated_statistics['sum_d_exp_K_miY_d_alpha'],
        partial_derivatives['dF_dsum_exp_K_mi_K_im'],
        accumulated_statistics['sum_d_exp_K_mi_K_im_d_alpha'])
    grad_sf2 = partial_terms.grad_sf2(partial_derivatives['dF_dKmm'],
        partial_terms.dKmm_dsf2(),
        partial_derivatives['dF_dsum_exp_K_ii'],
        accumulated_statistics['sum_d_exp_K_ii_d_sf2'],
        partial_derivatives['dF_dsum_exp_K_miY'],
        accumulated_statistics['sum_d_exp_K_miY_d_sf2'],
        partial_derivatives['dF_dsum_exp_K_mi_K_im'],
        accumulated_statistics['sum_d_exp_K_mi_K_im_d_sf2'])
    gradient = {'Z' : grad_Z,
        'sf2' : grad_sf2,
        'alpha' : grad_alpha}
    if not options['fixed_beta']:
        gradient['beta'] = partial_terms.grad_beta()
    else: 
        gradient['beta'] = numpy.zeros((1,1))
    #print 'gradient'
    #print gradient
    return gradient


''' Clean unneeded files '''
def clean(options):
    # We assume that if the file does not exist the function simply returns. We also remove files from
    # the previous call to the function that used the same iteration
    if not options['keep'] and options['i'] != 'f':
        for key in options['global_statistics_names']:
            file_name = options['statistics'] + '/global_statistics_' + key + '_' + str(-1) + '.npy'
            map_reduce.remove(file_name)
            file_name = options['statistics'] + '/global_statistics_' + key + '_' + str(options['i'] - 1) + '.npy'
            map_reduce.remove(file_name)
            file_name = options['statistics'] + '/global_statistics_' + key + '_' + str(options['i']) + '.npy'
            map_reduce.remove(file_name)
        for key in options['accumulated_statistics_names']:
            file_name = options['statistics'] + '/accumulated_statistics_' + key + '_' + str(-1) + '.npy'
            map_reduce.remove(file_name)
            file_name = options['statistics'] + '/accumulated_statistics_' + key + '_' + str(options['i'] - 1) + '.npy'
            map_reduce.remove(file_name)
            file_name = options['statistics'] + '/accumulated_statistics_' + key + '_' + str(options['i']) + '.npy'
            map_reduce.remove(file_name)
        for key in options['partial_derivatives_names']:
            file_name = options['statistics'] + '/partial_derivatives_' + key + '_' + str(-1) + '.npy'
            map_reduce.remove(file_name)
            file_name = options['statistics'] + '/partial_derivatives_' + key + '_' + str(options['i'] - 1) + '.npy'
            map_reduce.remove(file_name)
            file_name = options['statistics'] + '/partial_derivatives_' + key + '_' + str(options['i']) + '.npy'
            map_reduce.remove(file_name)
        for key in options['cache_names']:
            file_name = options['statistics'] + '/cache_' + key + '_' + str(-1) + '.npy'
            map_reduce.remove(file_name)
            file_name = options['statistics'] + '/cache_' + key + '_' + str(options['i'] - 1) + '.npy'
            map_reduce.remove(file_name)
            file_name = options['statistics'] + '/cache_' + key + '_' + str(options['i']) + '.npy'
            map_reduce.remove(file_name)

def test(options, Y_test, is_random_init=False, iterations=10):
    ''' Return mean and variance for test points given a trained model in 'options' '''
    ''' Init the mean variance '''
    if is_random_init:
        X_mu = scipy.randn(Y_test.shape[0], options['Q'])
    else:
        import scipy.spatial
        file_names = glob.glob(options['input'] + '/*')
        Y_dists = [numpy.inf for i in xrange(Y_test.shape[0])]
        X_mu = numpy.zeros((Y_test.shape[0], options['Q']))
        for file_name in file_names:
            Y = genfromtxt(file_name, delimiter=',')
            embedding_name = options['embeddings'] + '/' + os.path.basename(file_name) + '.embedding.npy'
            X = scipy.load(embedding_name)

            tree = scipy.spatial.cKDTree(Y,leafsize=100)
            (dist, ind) = tree.query(Y_test, k=1, distance_upper_bound=6)

            for i in xrange(X_mu.shape[0]):
                if dist[i] < Y_dists[i]:
                    Y_dists[i] = dist[i]
                    X_mu[i, :] = X[ind[i], :]
    X_S = numpy.clip(numpy.ones((Y_test.shape[0], options['Q'])) * 0.5
                        + 0.01 * scipy.randn(Y_test.shape[0], options['Q']),
                0.001, 1)

    ''' Optimise over these values '''
    import local_MapReduce
    options['load'] = True
    options, global_statistics = init_statistics(local_MapReduce, options)
    options['i'] = 'f'
    accumulated_statistics = {}
    for key in options['accumulated_statistics_names']:
        file_name = options['statistics'] + '/accumulated_statistics_' + key + '_' + str(options['i']) + '.npy'
        accumulated_statistics[key] = local_MapReduce.load(file_name)

    # Calculate partial statistics...
    partial_terms = local_MapReduce.load_partial_terms(options, global_statistics)
    # Load cached matrices
    local_MapReduce.load_cache(options, partial_terms)

    partial_terms.set_data(Y_test, X_mu, X_S, is_set_statistics=False)

    ''' Get local statistics for new data points '''
    partial_terms_local = local_MapReduce.load_partial_terms(options, global_statistics)
    local_MapReduce.load_cache(options, partial_terms_local)
    partial_terms_local.set_data(Y_test, X_mu, X_S, is_set_statistics=True)
    new_local_statistics = partial_terms_local.get_local_statistics()

    partial_terms.set_local_statistics(accumulated_statistics['sum_YYT'] + new_local_statistics['sum_YYT'],
        accumulated_statistics['sum_exp_K_mi_K_im'] + new_local_statistics['sum_exp_K_mi_K_im'],
        accumulated_statistics['sum_exp_K_miY'] + new_local_statistics['exp_K_miY'],
        accumulated_statistics['sum_exp_K_ii'] + new_local_statistics['exp_K_ii'],
        accumulated_statistics['sum_KL'] + new_local_statistics['KL'])

    # Actual optimisation of the embeddings
    for i in xrange(iterations):
        (X_mu, X_S) = partial_terms.local_optimisation()

        partial_terms.set_data(Y_test, X_mu, X_S, is_set_statistics=False)

        ''' Get local statistics for new data points '''
        partial_terms_local.set_data(Y_test, X_mu, X_S, is_set_statistics=True)
        new_local_statistics = partial_terms_local.get_local_statistics()

        partial_terms.set_local_statistics(accumulated_statistics['sum_YYT'] + new_local_statistics['sum_YYT'],
            accumulated_statistics['sum_exp_K_mi_K_im'] + new_local_statistics['sum_exp_K_mi_K_im'],
            accumulated_statistics['sum_exp_K_miY'] + new_local_statistics['exp_K_miY'],
            accumulated_statistics['sum_exp_K_ii'] + new_local_statistics['exp_K_ii'],
            accumulated_statistics['sum_KL'] + new_local_statistics['KL'])
        ''' BUG IN THE IMPLEMENTATION: set_data changes sum_YYT '''

    return [X_mu, X_S, partial_terms.logmarglik()]

'''
Parse options[' Not interesting at all.
'''

def parse_options():
    parser = OptionParser("usage: \n%prog [options] --input <input folder> --embeddings <embeddings folder>\n%prog -h for help")
    parser.add_option("-i", "--input", dest="input",
                  help="Folder containing files to be processed. One file will be processed per node. Files assumed to be in a comma-separated-value (CSV) format. (required)")
    parser.add_option("-e", "--embeddings", dest="embeddings",
                  help="Existing folder to store embeddings in. One file will be created for each input file.")
    parser.add_option("-p", "--parallel",
                       type="choice",
                       choices=["local", "Hadoop", "SGE"], default="local",
                       help="Which parallel architecture to use (local (default), Hadoop, SGE)"
                      )
    parser.add_option("-T", "--iterations", dest="iterations",
                  help="Number of iterations to run; default value is 100", type="int", default="100")
    parser.add_option("-s", "--statistics", dest="statistics",
                  help="Folder to store statistics files in (default is /tmp)", default="/tmp")
    parser.add_option("-k", "--keep", action="store_true", dest="keep", default=False,
                  help="Whether to keep statistics files or to delete them")
    parser.add_option("-l", "--load", action="store_true", dest="load", default=False,
                  help="Whether to load statistics and embeddings from previous run or initialise new ones")
    parser.add_option("-t", "--tmp", dest="tmp",
                  help="Shared folder to store tmp files in (default is /scratch/tmp)", default="/scratch/tmp")
    parser.add_option("--init",
                       type="choice",
                       choices=["PCA", "PPCA", "FA", "random"], default="PCA",
                       help="Which Initialisation to use (PCA (default), PPCA (probabilistic PCA), FA (factor analysis), random)"
                      )

    # Sparse GPs specific options
    SparseGPs_group = OptionGroup(parser, "Sparse GPs Options")
    SparseGPs_group.add_option("-M", "--inducing_points", type=int, dest="M",
                  help="Number of inducing points (default: 10)", default="10")
    SparseGPs_group.add_option("-Q", "--latent_dimensions", type=int, dest="Q",
                  help="Number of latent dimensions (default: 10)", default="10")
    SparseGPs_group.add_option("-D", "--output_dimensions", type=int, dest="D",
                  help="Number of output dimensions given in Y (default value set to 10)", default="10")
    SparseGPs_group.add_option("--fixed_embeddings", action="store_true", dest="fixed_embeddings",
                  help="If given, embeddings (X) are treated as fixed. Only makes sense when embeddings are given in the folder in advance", default=False)
    SparseGPs_group.add_option("--fixed_beta", action="store_true", dest="fixed_beta",
                  help="If given, beta is treated as fixed.", default=False)
    parser.add_option_group(SparseGPs_group)

    # SGE specific options
    SGE_group = OptionGroup(parser, "SGE Options")
    SGE_group.add_option("--simplejson", dest="simplejson",
                  help="SGE simplejson location", default="/scratch/python/lib.linux-x86_64-2.5/")
    parser.add_option_group(SGE_group)

    # Hadoop specific options
    Hadoop_group = OptionGroup(parser, "Hadoop Options")
    Hadoop_group.add_option("--hadoop", dest="hadoop",
                  help="Hadoop folder", default="/usr/local/hadoop/bin/hadoop")
    Hadoop_group.add_option("--jar", dest="jar",
                  help="Jar file for Hadoop streaming", default="/usr/local/hadoop/share/hadoop/tools/lib/hadoop-*streaming*.jar")
    parser.add_option_group(Hadoop_group)

    # Check that the options are correct and create the required folders
    (options, args) = parser.parse_args()
    options = vars(options)
    if not options['input']:
        parser.error('Input folder not given')
    elif not os.path.exists(options['input']):
        raise Exception('Input folder does not exist')
    input_files_names = os.listdir(options['input'] + '/')
    if len(input_files_names) == 0:
        raise Exception('No input files!')

    if not options['embeddings']:
        parser.error('Embeddings folder not given')
    elif not os.path.exists(options['embeddings']):
        raise Exception('Folder to save embeddings in does not exist')

    if not os.path.exists(options['statistics']):
        raise Exception('Statistics folder ' + options['statistics'] + ' does not exist')

    if not os.path.exists(options['tmp']):
        raise Exception('TMP folder ' + options['tmp'] + ' does not exist')

    if options['parallel'] == "SGE":
        try:
            subprocess.call(["qstat"])
        except:
            raise Exception('Cannot call SGE''s qstat; please make sure the environment was set up correctly')
    if options['parallel'] == "SGE" and not os.path.exists(options['simplejson']):
        raise Exception('SGE simplejson ' + options['simplejson'] + ' does not exist')

    return options


if __name__ == '__main__':
    main()

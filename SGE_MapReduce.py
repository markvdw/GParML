'''
SGE_MapReduce.py
Implements the  map-reduce framework for an SGE cluster and handles the file management for the different nodes.
'''
#!/usr/bin/python
from __future__ import with_statement
import tempfile
import sys
import os
import glob
import collections
import itertools
import subprocess
import time
import numpy
import scipy
from numpy import genfromtxt
import kernels
import numpy.linalg as linalg
import partial_terms as pt

'''
Initialise the inputs and embeddings
'''

def init(options):
	'''
	Init shared folders and the embeddings if needed, and work out N, the number of data points
	'''
	sys.path.insert(0, options['simplejson'])
	import simplejson
	this_script = os.path.realpath(__file__)
	this_folder = os.path.dirname(this_script)
	if not os.path.exists(this_folder + '/logfiles'): os.makedirs(this_folder + '/logfiles')
	if not os.path.exists(this_folder + '/errorfiles'): os.makedirs(this_folder + '/errorfiles')
	if not os.path.exists(this_folder + '/scriptfiles'): os.makedirs(this_folder + '/scriptfiles')

	'''
	Init embeddings if needed, and work out N, the number of data points
	To Do: parallelise the initialisation
	'''
	# N keeps track of the global number of inputs
	N = 0
	input_files_names = os.listdir(options['input'] + '/')
	for file_name in input_files_names:
		''' Find global number of inputs'''
		# Count the number of lines in the input file
		length = 0
		input_file = options['input'] + '/' + file_name
		with open(input_file) as f:
			for line in f:
				if line.strip():
					length += 1
		N += length

		'''Initialise the inputs and embeddings'''
		embedding_name = options['embeddings'] + '/' + file_name + '.embedding.npy'
		embedding_variance_name = options['embeddings'] + '/' + file_name + '.variance.npy'
		if not options['fixed_embeddings']:
			if not os.path.exists(embedding_name):
				print 'Creating ' + embedding_name + ' with ' + str(length) + ' points'
				'''
				To Do: allow user to select init method. Currently hard-coded to do local PCA
				Random initialisation:
				# save(embedding_name, scipy.randn(length, options['Q']))
				PCA initialisation:
				# Here we perform PCA over the LOCAL dataset Y for each node, using the assumption
				# that the data is distributed uniformly among the nodes
				'''
				save(embedding_name, PCA(input_file, options['Q']))
			else:
				print 'Using existing embeddings...'
			if not os.path.exists(embedding_variance_name):
				print 'Creating ' + embedding_variance_name + ' with ' + str(length) + ' points'
				# Initialise variance of data
				save(embedding_variance_name, 
					numpy.clip(numpy.ones((length, options['Q'])) * 0.5
								+ 0.01 * scipy.randn(length, options['Q']),
						0.001, 1))
			else:
				print 'Using existing embedding variances...'
		else:
			# If we are using fixed embeddings (i.e. doing sparse GPs)
			if not os.path.exists(embedding_name):
				raise Exception('No embedding file ' + embedding_name)
			if not os.path.exists(embedding_variance_name):
				print 'Creating ' + embedding_variance_name
				# Initialise variance of data
				save(embedding_variance_name, numpy.zeros((length, options['Q'])))
	return N



'''
Statistics Map-Reduce functions:
'''

def statistics_MR(options):
	'''
	Gets as input options and statistics to use in accumulation; returns as output partial sums. Writes files to the shard folder options['tmp to pass information between different nodes.
	'''
	input_files = glob.glob(options['input'] + '/*')
	# We're passing only the relevant parameters
	arg = {'global_statistics_names' : options['global_statistics_names'],
			'statistics' : options['statistics'],
			'embeddings' : options['embeddings'], 'simplejson' : options['simplejson'], 
			'tmp' : options['tmp'], 'D' : options['D'], 
			'i' : options['i'], 'M' : options['M'], 'N' : options['N'], 'Q' : options['Q']}
	script_files = write_scripts(arg, input_files, 'statistics_mapper')
	map_responses = statistics_MR_map(options, script_files)
	partitioned_data = partition(itertools.chain(*map_responses))
	reduced_values = []
	for item in partitioned_data:
		(statistic, target_file_name) = statistics_reducer((item, options))
		reduced_values.append((statistic, target_file_name))
	return reduced_values

def partition(mapped_values):
	'''
	Organize the mapped values by their key.
	Returns an unsorted sequence of tuples with a key and a sequence of values.
	'''
	partitioned_data = collections.defaultdict(list)
	for key, value in mapped_values:
		partitioned_data[key].append(value)
	return partitioned_data.items()

def statistics_MR_map(options, script_files):
	'''
	Sends scripts to nodes to be processed
	'''
	# Submit jobs to SGE
	MR_map(options, script_files)
	wait_for_results_to_collect()
	return collect_results(script_files, return_responses = True)

def statistics_reducer((source_file_name_list, options)):
	'''
	Reduces a list of file names (of a single statistic) to a single file by summing them and deleting the old files
	'''
	statistic = source_file_name_list[0]
	files_names = source_file_name_list[1]
	
	target_file_name = options['statistics'] + '/accumulated_statistics_' + statistic + '_' + str(options['i']) + '.npy'
	if len(files_names) == 1:
		# Move to the statistics folder
		os.rename(files_names[0], target_file_name)
	else:
		accumulated_statistics = load(files_names[0])
		remove(files_names[0])
		for file_name in files_names[1:]:
			accumulated_statistics += load(file_name)	
			remove(file_name)
		save(target_file_name, accumulated_statistics)

	return (statistic, target_file_name)


'''
Embeddings Map-Reduce functions:
'''

def embeddings_MR(options):
	'''
	Gets as input options and statistics to use in embeddings optimisation. Writes files to shared folder options['statistics to pass information between different nodes.
	'''

	input_files = glob.glob(options['input'] + '/*')
	# We're passing only the relevant parameters
	arg = {'global_statistics_names' : options['global_statistics_names'],
			'accumulated_statistics_names' : options['accumulated_statistics_names'],
			'partial_derivatives_names' : options['partial_derivatives_names'],
			'statistics' : options['statistics'],
			'embeddings' : options['embeddings'], 'simplejson' : options['simplejson'], 
			'D' : options['D'], 'i' : options['i'], 'M' : options['M'], 'N' : options['N'], 'Q' : options['Q']}
	script_files = write_scripts(arg, input_files, 'embeddings_mapper')
	MR_map(options, script_files)
	return script_files

def embeddings_watcher(options, script_files):
	wait_for_results_to_collect()
	# Collect results
	collect_results(script_files, return_responses = False)



'''
functions to be called by nodes:
'''

def statistics_mapper((input_file_name, options)):
	'''
	Maps an input to temp files returning a dictionary of statistics and file names containing them
	'''
	# Load global statistics
	global_statistics = {}
	for key in options['global_statistics_names']:
		file_name = options['statistics'] + '/global_statistics_' + key + '_' + str(options['i']) + '.npy'
		global_statistics[key] = load(file_name)

	# Load inputs and embeddings
	embedding_name = options['embeddings'] + '/' + os.path.basename(input_file_name) + '.embedding.npy'
	embedding_variance_name = options['embeddings'] + '/' + os.path.basename(input_file_name) + '.variance.npy'
	Y = genfromtxt(input_file_name, delimiter=',')# os.path.join(INPUT_DIR,filename)
	X_mu = load(embedding_name)
	X_S = load(embedding_variance_name)

	# Calculate partial statistics...
	partial_terms = load_partial_terms(options, global_statistics)
	# Load cached matrices
	load_cache(options, partial_terms)

	partial_terms.set_data(Y, X_mu, X_S, is_set_statistics=True)

	terms = partial_terms.get_local_statistics()
	accumulated_statistics = {
		'sum_YYT' : terms['sum_YYT'], 
		'sum_exp_K_ii' : terms['exp_K_ii'], 
		'sum_exp_K_mi_K_im' : terms['sum_exp_K_mi_K_im'], 
		'sum_exp_K_miY' : terms['exp_K_miY'], 
		'sum_KL' : terms['KL'], 
		'sum_d_Kmm_d_Z' : partial_terms.dKmm_dZ(), 
		'sum_d_exp_K_miY_d_Z' : partial_terms.dexp_K_miY_dZ(), 
		'sum_d_exp_K_mi_K_im_d_Z' : partial_terms.dexp_K_mi_K_im_dZ(), 
		'sum_d_Kmm_d_alpha' : partial_terms.dKmm_dalpha(), 
		'sum_d_exp_K_miY_d_alpha' : partial_terms.dexp_K_miY_dalpha(), 
		'sum_d_exp_K_mi_K_im_d_alpha' : partial_terms.dexp_K_mi_K_im_dalpha(),
		'sum_d_Kmm_d_sf2' : partial_terms.dKmm_dsf2(), 
		'sum_d_exp_K_ii_d_sf2' : partial_terms.dexp_K_ii_dsf2(), 
		'sum_d_exp_K_miY_d_sf2' : partial_terms.dexp_K_miY_dsf2(), 
		'sum_d_exp_K_mi_K_im_d_sf2' : partial_terms.dexp_K_mi_K_im_dsf2()
	}

	file_names_list = []
	for key in accumulated_statistics.keys():
		file_name = tempfile.mktemp(dir=options['tmp'], suffix='.npy')
		save(file_name, accumulated_statistics[key])
		file_names_list.append((key, file_name))
	return file_names_list
	
def embeddings_mapper((input_file_name, options)):
	global_statistics = {}
	accumulated_statistics = {}
	
	for key in options['global_statistics_names']:
		file_name = options['statistics'] + '/global_statistics_' + key + '_' + str(options['i']) + '.npy'
		global_statistics[key] = load(file_name)
	for key in options['accumulated_statistics_names']:
		file_name = options['statistics'] + '/accumulated_statistics_' + key + '_' + str(options['i']) + '.npy'
		accumulated_statistics[key] = load(file_name)

	# Load inputs and embeddings
	embedding_name = options['embeddings'] + '/' + os.path.basename(input_file_name) + '.embedding.npy'
	embedding_variance_name = options['embeddings'] + '/' + os.path.basename(input_file_name) + '.variance.npy'
	Y = genfromtxt(input_file_name, delimiter=',')# os.path.join(INPUT_DIR,filename)
	X_mu = load(embedding_name)
	X_S = load(embedding_variance_name)

	# Calculate partial statistics...
	partial_terms = load_partial_terms(options, global_statistics)
	# Load cached matrices
	load_cache(options, partial_terms)

	partial_terms.set_data(Y, X_mu, X_S, is_set_statistics=False)

	partial_terms.set_local_statistics(accumulated_statistics['sum_YYT'],
		accumulated_statistics['sum_exp_K_mi_K_im'],
		accumulated_statistics['sum_exp_K_miY'],
		accumulated_statistics['sum_exp_K_ii'],
		accumulated_statistics['sum_KL'])

	# Actual optimisation of the embeddings
	(new_X_mu, new_X_S) = partial_terms.local_optimisation()
	save(embedding_name, new_X_mu)
	save(embedding_variance_name, new_X_S)



'''
Global map-reduce functions:
'''

def write_scripts(options, input_files, action):
	'''
	Write the scripts to be called by the SGE to a temp folder
	'''
	script_files = []
	this_script = os.path.realpath(__file__)
	this_folder = os.path.dirname(this_script)
	import simplejson
	for input_file in input_files:
		arguments = simplejson.dumps((input_file, options))
		script_name = tempfile.mktemp(dir=this_folder + '/scriptfiles', suffix='.sh')
		command = 'python \'' + this_script + '\' ' + action + ' \'' + arguments + '\' \'' + options['simplejson'] + '\'\necho\necho'
		with open(script_name, "w") as text_file:
			text_file.write(command)
		script_files.append(script_name)
	return script_files

def MR_map(options, script_files):
	this_script = os.path.realpath(__file__)
	this_folder = os.path.dirname(this_script)
	import simplejson
	# Submit jobs to SGE
	print 'Initiating ' + str(len(script_files)) + ' mappers...'
	for script_file in script_files:
		job = 'qsub -o ' + this_folder + '/logfiles/ -e ' + this_folder + '/errorfiles/ ' + script_file
		subprocess.Popen(job.split(' '), stdout=subprocess.PIPE)
		time.sleep(0.1)
	time.sleep(1)

def wait_for_results_to_collect():
	'''
	Wait for the results to collect
	'''
	for iii in range(0,8000): # ToDo replace with options['timeout / 5
		proc = subprocess.Popen('qstat', stdout=subprocess.PIPE)
		output = proc.stdout.read()
		lines = output.split('\n')
		if len(lines) == 1:
			print 'All mappers finished!'
			break
		proc = subprocess.Popen('qstat -s r'.split(' '), stdout=subprocess.PIPE)
		output = proc.stdout.read()
		running_tasks = output.split('\n')
		if len(running_tasks) > 1: running_tasks = len(running_tasks) - 3 
		else: running_tasks = 0
		proc = subprocess.Popen('qstat -s p'.split(' '), stdout=subprocess.PIPE)
		output = proc.stdout.read()
		pending_tasks = output.split('\n')
		if len(pending_tasks) > 1: pending_tasks = len(pending_tasks) - 3 
		else: pending_tasks = 0
		print str(running_tasks) + ' mappers still running, ' + str(pending_tasks) + ' pending...'
		time.sleep(2) # TODO: replace with options['sleep_time

def collect_results(script_files, return_responses):
	this_script = os.path.realpath(__file__)
	this_folder = os.path.dirname(this_script)
	import simplejson
	# Collect results
	map_responses = []
	for script_file in script_files:
		name = os.path.basename(script_file)
		log_file = glob.glob(this_folder + '/logfiles/' + name + '*')
		if return_responses:
			with open(log_file[0], "r") as text_file:
				line = text_file.readline().rstrip('\n')
				map_responses.append(simplejson.loads(line))
		error_file = glob.glob(this_folder + '/errorfiles/' + name + '*')
		# Check for errors
		line = ''
		with open(error_file[0], "r") as text_file:
			line = text_file.read()
		if not line == '':
			print 'Error in ' + error_file[0] + ':'
			print line
			continue
		# Clean up
		os.remove(log_file[0])
		os.remove(script_file)
		os.remove(error_file[0])
	return map_responses



'''
Supporting functions
'''

def PCA(Y_name, input_dim):
	"""
	Principal component analysis: maximum likelihood solution by SVD
	Adapted from GPy.util.linalg
	Arguments
	---------
	:param Y: NxD np.array of data
	:param input_dim: int, dimension of projection

	Returns
	-------
	:rval X: - Nxinput_dim np.array of dimensionality reduced data
	W - input_dimxD mapping from X to Y
	"""
	Y = genfromtxt(Y_name, delimiter=',')
	Z = numpy.linalg.svd(Y - Y.mean(axis=0), full_matrices=False)
	[X, W] = [Z[0][:, 0:input_dim], numpy.dot(numpy.diag(Z[1]), Z[2]).T[:, 0:input_dim]]
	v = X.std(axis=0)
	X /= v;
	W *= v;
	return X

def save(file_name, obj):
	scipy.save(file_name, obj)

def load(file_name):
	return scipy.load(file_name)

def remove(file_name):
	if os.path.exists(file_name):
		os.remove(file_name)

def cache(options, global_statistics):
	'''
	To Do: support Q=1 case where alpha squeeze is scalar
	'''
	# We can calculate the global statistics once for all nodes
	kernel = kernels.rbf(options['Q'], sf=float(global_statistics['sf2']**0.5), ard=numpy.squeeze(global_statistics['alpha']))
	Kmm = kernel.K(global_statistics['Z'])
	file_name = options['statistics'] + '/cache_Kmm_' + str(options['i']) + '.npy'
	save(file_name, Kmm)
	Kmm_inv = linalg.inv(Kmm)
	file_name = options['statistics'] + '/cache_Kmm_inv_' + str(options['i']) + '.npy'
	save(file_name, Kmm_inv)

def load_cache(options, partial_terms):
	file_name = options['statistics'] + '/cache_Kmm_' + str(options['i']) + '.npy'
	Kmm = load(file_name)
	file_name = options['statistics'] + '/cache_Kmm_inv_' + str(options['i']) + '.npy'
	Kmm_inv = load(file_name)
	partial_terms.set_global_statistics(Kmm, Kmm_inv)

def load_partial_terms(options, global_statistics):
	return pt.partial_terms(global_statistics['Z'],
								float(global_statistics['sf2']**0.5),
								numpy.squeeze(global_statistics['alpha']),
								float(global_statistics['beta']),
								options['M'], options['Q'],
								options['N'], options['D'], update_global_statistics=False)



if __name__ == '__main__':
	if len(sys.argv) != 4:
		raise Exception('This script is to be called by the remote nodes')

	action = sys.argv[1]
	arg = sys.argv[2]
	simplejson_folder = sys.argv[3]

	sys.path.insert(0, simplejson_folder)
	import simplejson

	arg = simplejson.loads(arg)

	res = []
	if action == 'statistics_mapper':
		res = statistics_mapper(arg)
	elif action== 'statistics_reducer':
		res = statistics_reducer(arg)
	elif action == 'embeddings_mapper':
		res = embeddings_mapper(arg)

	print simplejson.dumps(res)

This package contains the Python code used for the paper ``Distributed Variational Inference in Sparse Gaussian Process Regression and Latent Variable Models''. The supplementary material is given in the file ``Variational Inference in the Gaussian Process Latent Variable Model and Sparse GP Regression - Supplementary Material.pdf''.

The core and most important part of the code is the partial_terms.py file, implementing the partial sums presented in the paper and the tutorial, with many references to the equations in the tutorial (please note that the equation numbers need updating as additional equations were introduced to the tutorial for clarity as part of the review process). This file contains only 583 lines of Python.

The inference can be run sequentially using the simple example files (containing roughly 300 lines of code) where different optimisers are used (gd (gradient descent), scg (scaled conjugate gradient), and scg_adapted which has been optimised to use less function evaluations). These are:
gd-example.py
scg-example.py
scg_adapted-example.py

The parallel inference was implemented across several files as it was built to be modular and extendible. These are:
parallel_GPLVM.py
local_MapReduce.py
SGE_MapReduce.py (not up to date)
supporting_functions.py
These files implement sanity checks as well for different inputs and were used to run the experiments.

Unit tests are provided in test.py (although the generation of data for the tests often causes underflows and overflows). These implement finite differencing tests for the different functions in partial_terms.py as well as quantitative comparisons to GPy.

To run the inference (for GPLVM), create a new folder ('test') containing sub-folders 'inputs', 'embeddings', 'statistics', and 'tmp'. inputs contains (rather confusingly -- this will be changed in future versions) the observed outputs for the GPLVM, while embeddings contains the embeddings and variance files (which will be initialised by default using PCA). There are many options available for the inference which can be inspected using the command:

python parallel_GPLVM.py --help

These are given at the end of this document. To run inference in a minimal way for 5 iterations over a 4D dataset using 2 inducing points the following line can be used:

python parallel_GPLVM.py -i ./test/inputs/ -e ./test/embeddings/ --statistics ./test/statistics/ --tmp ./test/tmp/ -k -T 5 -M 2 -Q 2 -D 4

To run the profiler the following command can be used:
python -m cProfile -s cumtime test_parallel_gpLVM.py > profiler_test_parallel_gpLVM.txt

The file sizes for the provided code is as follows:
    353 gd-example.py
    105 gd_local_MapReduce.py
    127 gd.py
    205 kernel_exp.py
    191 kernels.py
    409 local_MapReduce.py
    113 nputil.py
    510 parallel_GPLVM.py
    583 partial_terms.py
    178 predict.py
     30 pre_process.py
    352 scg_adapted-example.py
    243 scg_adapted_local_MapReduce.py
    314 scg_adapted.py
    363 scg-example.py
    146 scg.py
    453 SGE_MapReduce.py
    169 supporting_functions.py
    301 test.py
   6102 total




The documentation for the code is as follows:

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
    Which initialisation to use (PCA (default), PPCA (probabilistic PCA), FA (factor analysis), random)
--optimiser
    Which optimiser to use (SCG_adapted (adapted scaled gradient descent - default), GD (gradient descent))
--drop_out_fraction
    Fraction of nodes to drop out  (default: 0)

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



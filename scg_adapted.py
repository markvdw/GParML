'''
Adapted version of SCG that makes use of a single objective_and_gradient function
for cases when it is cheaper to evaluate both together. Adapted from the original
from the GPy project.
'''

# Copyright I. Nabney, N.Lawrence and James Hensman (1996 - 2012)
# Scaled Conjuagte Gradients, originally in Matlab as part of the Netlab toolbox by I. Nabney, converted to python N. Lawrence and given a pythonic interface by James Hensman

#      THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
#      HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
#      EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
#      NOT LIMITED TO, THE IMPLIED WARRANTIES OF
#      MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#      PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#      REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
#      DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#      EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#      (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
#      OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#      DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#      HOWEVER CAUSED AND ON ANY THEORY OF
#      LIABILITY, WHETHER IN CONTRACT, STRICT
#      LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
#      OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#      OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#      POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import sys
from numpy.linalg.linalg import LinAlgError
import traceback
import scg_adapted_local_MapReduce as local_MapReduce
debug = False

def print_out(len_maxiters, display, fnow, current_grad, beta, iteration):
    if display:
        print 
        print '\r',
        print '{0:>0{mi}g}  {1:> 12e}  {2:> 12e}  {3:> 12e}'.format(iteration, float(fnow), float(beta), float(current_grad), mi=len_maxiters), # print 'Iteration:', iteration, ' Objective:', fnow, '  Scale:', beta, '\r',
        print 
        sys.stdout.flush()

_fail_count = 0
_allowed_failures = 100
def safe_f_and_grad_f(f_and_gradf, x, iteration=0, step_size=0, *optargs):
    '''
    Calls f and gradf and returns inf for f in case of warnings / assertion errors and so on.
    The returned gradf in that case is 0, which screws up SCG's momentum, so a re-start should be done
    '''
    global _fail_count, _allowed_failures
    try:
        [f, gradf] = f_and_gradf(x, iteration, step_size, *optargs)
        _fail_count = 0
    except (LinAlgError, ZeroDivisionError, ValueError, Warning, AssertionError) as e:
        if _fail_count >= _allowed_failures:
            print 'Too many errors...'
            raise e
        _fail_count += 1
        print

        _,_,tb = sys.exc_info()
        tbInfo = traceback.extract_tb(tb)
        filename,line,func,text = tbInfo[-1]
        print ('An error occurred on line ' + str(line) + ' in filename ' + filename)
        print 'Increasing failed count (' + str(_fail_count) + ') and returning nlml inf'
        #print 'x catch:'
        #print x
        #print 
        #[f, gradf] = f_and_gradf(x, iteration, *optargs)
        #print 'gradf catch:'
        #print gradf
        #print 
        f = np.inf
        gradf = np.ones(x.shape)
    return f, gradf

def SCG_adapted(f_and_gradf, x, tmp_folder, fixed_embeddings=False, optargs=(), maxiters=500, max_f_eval=500, display=True, xtol=None, ftol=None, gtol=None):
    """
    Optimisation through Scaled Conjugate Gradients (SCG)

    f: the objective function
    gradf : the gradient function (should return a 1D np.ndarray)
    x : the initial condition

    Returns
    x the optimal value for x
    flog : a list of all the objective values
    function_eval number of fn evaluations
    status: string describing convergence status
    """

    if xtol is None:
        xtol = 1e-6
    if ftol is None:
        ftol = 1e-6
    if gtol is None:
        gtol = 1e-5
    sigma0 = 1.0e-4
    f_gradf = safe_f_and_grad_f(f_and_gradf, x, iteration=0, step_size=0, *optargs)

    assert f_gradf[0] != float('inf')

    fold = f_gradf[0] # Initial function value.
    function_eval = 1
    fnow = fold
    gradnew = f_gradf[1] # Initial gradient.
    gradold = gradnew.copy()
    d = -gradnew # Initial search direction.
    if not fixed_embeddings:
        local_MapReduce.embeddings_set_grads(tmp_folder)
    current_grad = np.dot(gradnew, gradnew)
    if not fixed_embeddings:
        current_grad += local_MapReduce.embeddings_get_grads_current_grad(tmp_folder)

    success = True # Force calculation of directional derivs.
    nsuccess = 0 # nsuccess counts number of successes.
    beta = 1.0 # Initial scale parameter.
    betamin = 1.0e-60 # Lower bound on scale.
    betamax = 1.0e100 # Upper bound on scale.
    status = "Not converged"

    flog = [fold]

    iteration = 0

    len_maxiters = len(str(maxiters))
    if display:
        print ' {0:{mi}s}   {1:11s}    {2:11s}    {3:11s}'.format("I", "F", "Scale", "|g|", mi=len_maxiters)

    # Main optimization loop.
    if display:
        print 'Starting optimisation for ' + str(maxiters) + ' iterations'
    if debug:
        print 'fixed_embeddings'
        print fixed_embeddings
        print 'init x'
        print x
        print 'init gradnew'
        print gradnew
        print 'init d'
        print d
    while iteration < maxiters:

        # Calculate first and second directional derivatives.
        if success:
            mu = np.dot(d, gradnew)
            if not fixed_embeddings:
                mu += local_MapReduce.embeddings_get_grads_mu(tmp_folder)
            if mu >= 0:
                d = -gradnew
                if not fixed_embeddings:
                    local_MapReduce.embeddings_set_grads_reset_d(tmp_folder)
                mu = np.dot(d, gradnew)
                if not fixed_embeddings:
                    mu += local_MapReduce.embeddings_get_grads_mu(tmp_folder)
            kappa = np.dot(d, d)
            if not fixed_embeddings:
                kappa += local_MapReduce.embeddings_get_grads_kappa(tmp_folder)
            sigma = sigma0 / np.sqrt(kappa)
            xplus = x + sigma * d
            gplus = safe_f_and_grad_f(f_and_gradf, xplus, iteration=-1, step_size=sigma, *optargs)[1]
            theta = np.dot(d, gplus - gradnew)
            if not fixed_embeddings:
                theta += local_MapReduce.embeddings_get_grads_theta(tmp_folder)
            # Numerical stability thingy:
            theta = theta * np.sqrt(kappa) / sigma0
            if debug:
                print 'mu'
                print mu
                print 'gradnew'
                print gradnew
                print 'd'
                print d
                print 'kappa'
                print kappa
                print 'sigma'
                print sigma
                print 'xplus'
                print xplus
                print 'gplus'
                print gplus
                print 'theta'
                print theta

        # Increase effective curvature and evaluate step size alpha.
        delta = theta + beta * kappa
        if delta <= 0:
            delta = beta * kappa
            beta = beta - theta / kappa

        alpha = -mu / delta

        if debug:
            print 'delta'
            print delta
            print 'beta'
            print beta
            print 'alpha'
            print alpha

        # Calculate the comparison ratio.
        xnew = x + alpha * d
        f_gradf = safe_f_and_grad_f(f_and_gradf, xnew, iteration=iteration + 1, step_size=alpha, *optargs)
        fnew = f_gradf[0]
        function_eval += 1

        if debug:
            print 'xnew'
            print xnew
            print 'fnew'
            print fnew


        if function_eval >= max_f_eval:
            status = "Maximum number of function evaluations exceeded"
            break
            #return x, flog, function_eval, status

        Delta = 2.*(fnew - fold) / (alpha * mu)
        if Delta >= 0.:
            success = True
            nsuccess += 1
            x = xnew
            if not fixed_embeddings:
                local_MapReduce.embeddings_set_grads_update_X(tmp_folder, alpha)
            fnow = fnew
        else:
            success = False
            fnow = fold

        if debug:
            print 'Delta'
            print Delta
            print 'xnew'
            print xnew
            print
            print 'fnow'
            print fnow
            print 'fold'
            print fold

        # Store relevant variables
        flog.append(fnow) # Current function value
        iteration += 1
        print_out(len_maxiters, display, fnow, current_grad, beta, iteration)

        if success:
            # Test for termination
            max_alpha_d = np.max(np.abs(alpha * d))
            if not fixed_embeddings:
                max_alpha_d = max(max_alpha_d, local_MapReduce.embeddings_get_grads_max_d(tmp_folder, alpha))
            if debug:
                if not fixed_embeddings:
                    print 'local_MapReduce.embeddings_get_grads_max_d(tmp_folder, alpha)'
                    print local_MapReduce.embeddings_get_grads_max_d(tmp_folder, alpha)
                print 'np.max(np.abs(alpha * d))'
                print np.max(np.abs(alpha * d))
                print 'max_alpha_d'
                print max_alpha_d
            if (max_alpha_d < xtol) or (np.abs(fnew - fold) < ftol):
                status = 'converged'
                break
                #return x, flog, function_eval, status

            else:
                # Update variables for new position
                ''' A bug: this was called after the gradnew assignment and overridden in original implementation '''
                gradold = gradnew
                if not fixed_embeddings:
                    local_MapReduce.embeddings_set_grads_update_grad_old(tmp_folder)
                gradnew = f_gradf[1]
                if not fixed_embeddings:
                    local_MapReduce.embeddings_set_grads_update_grad_new(tmp_folder)
                current_grad = np.dot(gradnew, gradnew)
                if not fixed_embeddings:
                    current_grad += local_MapReduce.embeddings_get_grads_current_grad(tmp_folder)
                fold = fnew
                # If the gradient is zero then we are done.
                if current_grad <= gtol:
                    status = 'converged'
                    break
                    # return x, flog, function_eval, status

        # Adjust beta according to comparison ratio.
        if Delta < 0.25:
            beta = min(4.0 * beta, betamax)
        if Delta > 0.75:
            beta = max(0.5 * beta, betamin)

        if debug:
            print 'beta'
            print beta
            print 'f_gradf[1]'
            print f_gradf[1]

        # Update search direction using Polak-Ribiere formula, or re-start
        # in direction of negative gradient after nparams steps.
        if nsuccess == x.size:
            d = -gradnew
            if debug:
                print 'd'
                print d
            #beta = 1.  # TODO: betareset!!
            #if not fixed_embeddings:
            #    local_MapReduce.embeddings_set_grads_reset_d(tmp_folder)
            nsuccess = 0
        elif success:
            Gamma = (np.dot(gradold, gradnew) - current_grad) / (mu)
            if not fixed_embeddings:
                Gamma += local_MapReduce.embeddings_get_grads_gamma(tmp_folder) / (mu)
            d = Gamma * d - gradnew
            if not fixed_embeddings:
                local_MapReduce.embeddings_set_grads_update_d(tmp_folder, Gamma)
            if debug:
                print 'Gamma'
                print Gamma
                print 'new d'
                print d
                print 'accepted gradnew'
                print gradnew
                print
    else:
        # If we get here, then we haven't terminated in the given number of
        # iterations.
        status = "maxiter exceeded"

    if display:
        print_out(len_maxiters, display, fnow, current_grad, beta, iteration)
        print

    if display:
        print('')
        print(status)

    return x, flog, function_eval, status, local_MapReduce.time_acc

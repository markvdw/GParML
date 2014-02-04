
import numpy as np
from numpy.linalg.linalg import LinAlgError
import sys
import traceback
import gd_local_MapReduce as local_MapReduce


def print_out(len_maxiters, display, fnow, current_grad, beta, iteration):
    if display:
        print '\r',
        print '{0:>0{mi}g}  {1:> 12e}  {2:> 12e}  {3:> 12e}'.format(iteration, float(fnow), float(beta), float(current_grad), mi=len_maxiters), # print 'Iteration:', iteration, ' Objective:', fnow, '  Scale:', beta, '\r',
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
        f = np.inf
        gradf = np.ones(x.shape)
    return f, gradf

def GD(f_and_gradf, x, tmp_folder, fixed_embeddings=False, optargs=(), maxiters=500, max_f_eval=500, display=True, xtol=None, ftol=None, gtol=None):
    """
    Optimisation through Gradient Descent

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
        xtol = 1e-16
    if ftol is None:
        ftol = 1e-6
    if gtol is None:
        gtol = 1e-6

    len_maxiters = len(str(maxiters))

    step_size = 0.01
    mom_size = 0.0

    f_gradf = safe_f_and_grad_f(f_and_gradf, x, iteration=0, step_size=0, *optargs)
    fnow = f_gradf[0]
    flog = [fnow]
    gradnow = f_gradf[1]
    direction = - gradnow
    if not fixed_embeddings:
        local_MapReduce.embeddings_set_grads(tmp_folder)
    
    iteration = 0
    while iteration < maxiters:
        xprop = x + step_size * direction
        f_gradf = safe_f_and_grad_f(f_and_gradf, xprop, iteration=iteration, step_size=step_size, *optargs)
        fproposed = f_gradf[0]

        if (np.abs(fnow - fproposed) < ftol):
            break
            print 'converged due to ftol'
        if (np.abs(step_size) < xtol):
            break
            print 'converged due to xtol'

        if (fproposed <= fnow):
            fnow = fproposed
            flog += [fnow]
            gradnow = f_gradf[1]
            if not fixed_embeddings:
                local_MapReduce.embeddings_set_grads_update_grad_now(tmp_folder)
            x = xprop
            if not fixed_embeddings:
                local_MapReduce.embeddings_set_grads_update_X(tmp_folder, step_size)
            direction = - (gradnow + mom_size * step_size * direction)
            #direction = - (gradnow - mom_size * step_size * direction)
            if not fixed_embeddings:
                local_MapReduce.embeddings_set_grads_update_d(tmp_folder, mom_size * step_size)
            step_size *= 2.0
            iteration += 1

            max_abs_gradnow = np.max(np.abs(gradnow))
            if not fixed_embeddings:
                max_abs_gradnow = max(max_abs_gradnow, local_MapReduce.embeddings_get_grads_max_gradnow(tmp_folder))
            if (max_abs_gradnow < gtol):
                break
                print 'converged due to grad'
        else:
            step_size /= 2.0

        if display:
            print ' {0:{mi}s}   {1:11s}    {2:11s}    {3:11s}'.format("I", "F", "Scale", "|g|", mi=len_maxiters)
            current_grad = np.sum(np.abs(gradnow))
            if not fixed_embeddings:
                current_grad += local_MapReduce.embeddings_get_grads_current_grad(tmp_folder)
            print_out(len_maxiters, display, fnow, current_grad, step_size, iteration)


    if display:
        current_grad = np.sum(np.abs(gradnow))
        if not fixed_embeddings:
            current_grad += local_MapReduce.embeddings_get_grads_current_grad(tmp_folder)
        print_out(len_maxiters, display, fnow, current_grad, step_size, iteration)
        print ""
    return x, flog, None, 'converged... NOT'

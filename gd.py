# Copyright I. Nabney, N.Lawrence and James Hensman (1996 - 2012)
# Adapted by Mark van der Wilk and Yarin Gal

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


def print_out(len_maxiters, display, fnow, current_grad, beta, iteration):
    if display:
        print '\r',
        print '{0:>0{mi}g}  {1:> 12e}  {2:> 12e}  {3:> 12e}'.format(iteration, float(fnow), float(beta), float(current_grad), mi=len_maxiters), # print 'Iteration:', iteration, ' Objective:', fnow, '  Scale:', beta, '\r',
        sys.stdout.flush()

def GD(f, gradf, x, optargs=(), maxiters=500, max_f_eval=500, display=True, xtol=None, ftol=None, gtol=None):
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
        xtol = 1e-6
    if ftol is None:
        ftol = 1e-6
    if gtol is None:
        gtol = 1e-5

    len_maxiters = len(str(maxiters))

    step_size = 0.01
    mom_size = 0.1

    fnow = f(x, *optargs)
    lastmove = 0
    
    iteration = 0
    while iteration < maxiters:
        gradnow = gradf(x, *optargs)
        xprop = x - step_size * (gradnow + mom_size * lastmove)
        fproposed = f(xprop, *optargs)

        if (fproposed <= fnow):
            fnow = fproposed
            x = xprop
            step_size *= 2.0
            lastmove = - step_size * (gradnow + mom_size * lastmove)
            iteration += 1

            if (np.max(np.abs(gradnow)) < 10**-6):
                break
                print 'converged due to grad'
        else:
            step_size /= 2.0

        if display:
            print ' {0:{mi}s}   {1:11s}    {2:11s}    {3:11s}'.format("I", "F", "Scale", "|g|", mi=len_maxiters)
            print_out(len_maxiters, display, fnow, np.sum(np.abs(gradnow)), step_size, iteration)


    if display:
        print_out(len_maxiters, display, fnow, np.sum(np.abs(gradnow)), step_size, iteration)
        print ""
    return x, None, None, 'converged... NOT'

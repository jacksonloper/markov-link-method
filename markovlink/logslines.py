'''
Here we consider problems of the form

   max_x     f(x)=sum(c*log(x))
   subj      A@x=b

where c>=0.  This problem also admits the dual, unconstrained problem 

  min_lam   f*(lam) = sum(c*(log(c)-1)) - sum(c*log(A.T@lam)) + sum(lam*b)

In the comments we will also refer to the lagrangian,

    L(x,lam) = sum(c*log(x)) - sum(lam*(A@x-b))

'''

import numpy as np
import sys
import scipy as sp
import scipy.special
import itertools
from . import misc
import time
import scipy.sparse

class InfeasibleException(Exception):
    pass

def roundtrip_from_feasible_primal(c,A,b,x,tol=1e-10):
    '''
    Given a feasible x, we try to guess a value of lam, and then
    use lam to try to get a guess of x.  If x is optimal we will
    end up where we started.  
    '''
    if np.abs(A@x-b).max()>tol:
        raise InfeasibleException("x not feasible")

    return get_primal_from_dual(c,A,estimate_dual_from_primal(c,A,x))

def estimate_gap_from_feasible_primal(c,A,b,x,tol=1e-10):
    '''
    Given a feasible x, we guess a value of lam and try
    to use that to get a certificate of the duality gap
    '''
    lam = estimate_dual_from_primal(c,A,x)
    return dual_objective(c,A,b,lam) - primal_objective(c,A,b,x,tol=tol)

def primal_objective(c,A,b,x,tol=1e-10):
    '''
    Compute the primal objective, f
    '''
    if np.abs(A@x-b).max()>tol:
        raise InfeasibleException("x not feasible")
    if (x<=0).any():
        raise InfeasibleException("x negative")

    return np.sum(c*np.log(x))

def lagrangian(c,A,b,x,lam):
    return np.sum(c*np.log(x)) - np.sum(lam*(A@x-b))

def dual_objective(c,A,b,lam):
    '''
    Compute the dual objective, f*
    '''

    xi = A.T@lam

    if (xi>0).all():
        return np.sum(c*(np.log(c)-1))-np.sum(c*np.log(A.T@lam)) +np.sum(lam*b)
    else:
        raise InfeasibleException("lam not feasible")

def get_primal_from_dual(c,A,lam):
    '''
    We solve the problem 

       max_x L(x,lam) 
       subj  x>=0

    (assuming lam is feasible)
    '''

    x= c/ (A.T@lam)

    if (x>0).all():
        return x
    else:
        raise InfeasibleException("lam not feasible")

def estimate_dual_from_primal(c,A,x):
    '''
    Try to find a value of lam so that 

    x = get_primal_from_dual(c,a,lam)

    The problem is overconstrained, and I think perfectly solvable if x is optimal.  Nonetheless,
    we can try to find a solution.

    '''

    lam=np.linalg.lstsq(A.T,c/x,rcond=None)[0]

    if ((A.T @ lam)>0).all():
        return lam
    else:
        raise InfeasibleException("Couldn't guess a feasible lam")


def primal_feasible_grad(c,A,b,x,elimination=True):
    '''
    solve  

    (I A.T) (delta) = (g)
    (A  0 ) (m)       (0)
    '''

    m,n=A.shape

    if elimination:
        lam = np.linalg.lstsq(A.T,c/x)[0]
        return (c/x) - A.T@lam
    else:
        bigmat=np.zeros((n+m,n+m))
        bigmat[:n,:n]=np.eye(n)
        bigmat[:n,n:]=A.T
        bigmat[n:,:n]=A

        b=np.r_[c/x,np.zeros(m)]

        rez=np.linalg.solve(bigmat,b)

        return rez[:n]
        
def primal_grad_hess(c,A,b,x):
    return c/x,np.diag(-c/x**2)

def primal_newton_direction(c,A,b,x,method='eliminationII'):
    '''
    solve  

    (H A.T) (delta) = (g)
    (A  0 ) (m)       (0)
    '''

    m,n=A.shape

    if method=='eliminationI':
        grad=c/x
        hi=np.diag(-x**2/c)

        lam = np.linalg.solve(A@hi@A.T,A@hi@grad)
        sd=(x**2/c)*(grad - A.T@lam)
        return sd

    elif method=='eliminationII':
        grad=c/x
        sqrthi=np.diag(x/np.sqrt(c))
        lam = np.linalg.lstsq(sqrthi@A.T,sqrthi@grad,rcond=None)[0]
        sd=(x**2/c)*(grad - A.T@lam)
        return sd
    else:


        grad,hess=primal_grad_hess(c,A,b,x)
        bigmat=np.zeros((n+m,n+m))
        bigmat[:n,:n]=hess
        bigmat[:n,n:]=A.T
        bigmat[n:,:n]=A

        b=np.r_[grad,np.zeros(m)]

        rez=np.linalg.solve(bigmat,b)

        return -rez[:n]



def primal_simple_backtracking_for_stepsize(c,A,b,x,search_direction,alpha=.25,beta=.9,initsize=.95,
                    grad=None,feas_mult=.5,tol=1e-10):
    '''
    Consider the problem

       max f(x)

    For a given guess (feasible) x and search direction, we try to make
    progress in `search_direction`.  In particular, we start with a stepsize of initsize
    and 

    1) decrease (by beta) until lam is feasible
    2) multiply by feas_mult
    3) decrease (by beta) until f*(lam + s*direction) < objective(lam) + s * <grad,direction> 

    We optionally take the gradient as input if you have precomputed it elsewhere.

    '''

    if np.abs(A@x - b).max()>tol:
        raise InfeasibleException("Initial x not feasible")

    # compute where we stand now
    curobjective = primal_objective(c,A,b,x)

    # get the gradient
    if grad is None:
        grad = c/x

    # what kind of gradient might we hope to achieve moving in this search direction?
    hoped_improvement_per_s=alpha*np.sum(grad*search_direction)
    if hoped_improvement_per_s<=0:
        raise Exception("Search direction is not an ascent direction (%e)"%hoped_improvement_per_s)

    # initialize stepsize
    stepsize=initsize/feas_mult

    # decrease the stepsizes until we get something that is still at least positive
    while True:
        newx = x + search_direction*stepsize
        if (x>0).all(): # if we're still positive, good
            break
        else:
            stepsize=stepsize*beta

    # drop our size by feas_mult (the edge is a dangerous place!)
    stepsize=stepsize*feas_mult

    # decrease stepsizes until we get hoped-for improvement
    while True:
        newx = x + search_direction*stepsize
        newobjective=primal_objective(c,A,b,x)

        if newobjective > curobjective + stepsize*hoped_improvement_per_s: # we didn't get enough improvement.  that's bad
            stepsize=stepsize*beta
        elif np.abs(A@newx-b).max()>tol: # we left the equality constraints!  let's try going less far!
            stepsize=stepsize*beta
        else: # get enoug improvement and something feasible!  done.
            break 

    # done!
    return stepsize

def dual_grad_hess(c,A,b,lam):
    '''
    Get gradient and hessian of dual objective, f*
    '''

    xi= A.T@lam
    if (xi<=0).any():
        raise Exception("lam is not feasible")

    MAT = A/xi[None,:]

    return b-MAT@c,MAT@np.diag(c)@MAT.T

def dual_simple_backtracking_for_stepsize(c,A,b,lam,search_direction,alpha=.25,beta=.9,initsize=1.0,
                    grad=None,feas_mult=.5):
    '''
    Consider the problem

       min f*(lam)

    For a given guess (feasible) lam and search direction, we try to make
    progress in `search_direction`.  In particular, we start with a stepsize of initsize
    and 

    1) decrease (by beta) until lam is feasible
    2) multiply by feas_mult
    3) decrease (by beta) until f*(lam + s*direction) < objective(lam) + s * <grad,direction> 

    We optionally take the gradient as input if you have precomputed it elsewhere.

    '''

    # make sure our starting point is feasible
    if ((A.T@lam)<=0).any():
        raise InfeasibleException("initial lam is not feasible!")

    # compute where we stand now
    curobjective = dual_objective(c,A,b,lam)

    # get the gradient
    if grad is None:
        grad = dual_grad_hess(c,A,b,lam)[0]

    # what kind of gradient might we hope to achieve moving in this search direction?
    hoped_improvement_per_s=alpha*np.sum(grad*search_direction)
    if hoped_improvement_per_s>=0:
        raise Exception("Search direction is not a descent direction")

    # initialize stepsize
    stepsize=initsize

    # decrease the stepsizes until we get feasible
    while True:
        newlam = lam + search_direction*stepsize
        if ((A.T@newlam)>0).all():
            break
        else:
            stepsize=stepsize*beta

    # drop our size by feas_mult (the edge is a dangerous place!)
    feasible_stepsize=stepsize
    stepsize=feasible_stepsize*feas_mult

    # decrease stepsizes until we get hoped-for improvement
    while True:
        newlam = lam + search_direction*stepsize
        newobjective=dual_objective(c,A,b,newlam)

        if newobjective > curobjective + stepsize*hoped_improvement_per_s: # we didn't get enough improvement.  that's bad
            stepsize=stepsize*beta            
        else: # get enoug improvement and something feasible!  done.
            break 

    # done!
    return stepsize


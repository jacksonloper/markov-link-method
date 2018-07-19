'''
Tools for finding the value of q which is closest to the analytic center of the polytope defined by

          p@q = h
'''

import numpy as np
import sys
import scipy as sp
import scipy.special
import itertools
from . import misc
import time
import scipy.sparse
from . import logslines
from . import polytopes

def rundual(c,A,b,x,y,dualchunks,tol):
    for i in range(dualchunks):
        # get newton step
        grad,hess=logslines.dual_grad_hess(c,A,b,y)
        sd=-np.linalg.solve(hess,grad)

        # in some cases newton direction may not be a search direction
        # because of floating point error.  so we just give up.
        # there's no way we're going to get better because floating point
        # issues are going to slam us.
        if np.sum(sd*grad)>=0:
            return y
        else:
            stepsize=logslines.dual_simple_backtracking_for_stepsize(c,A,b,y,sd,initsize=1.0,grad=grad)
            y=y+sd*stepsize

            gap=logslines.dual_objective(c,A,b,y) - logslines.primal_objective(c,A,b,x)
            if gap<tol:
                return y

    return y

def runprimal(c,A,b,x,y,primalchunks,tol):
    for i in range(primalchunks):
        grad=c/x
        sd=logslines.primal_newton_direction(c,A,b,x,method='eliminationII')


        # in some cases newton direction may not be a search direction
        # because of floating point error. 
        # there's no way we're going to get better because floating point
        # issues are going to slam us.
        if np.sum(sd*grad)<=0:
            return x
        else:
            stepsize=logslines.primal_simple_backtracking_for_stepsize(c,A,b,x,sd,feas_mult=.9)
            x=x+sd*stepsize

            gap=logslines.dual_objective(c,A,b,y) - logslines.primal_objective(c,A,b,x)
            if gap<tol:
                return x

    return x

def centerq(p,hstar,q=None,tol=.0001,dualchunks=50,primalchunks=10,precondition=True):
    '''
    Solves the problem

    max sum(log(q))

    subject to pq=h

    We iterate until we get within (szX*szY)*tol of optimal.
    '''

    szL,szX=p.shape
    szL,szY=hstar.shape
    tol=tol*szX*szY

    # initialize
    if q is None:
        q = np.ones((szX,szY))/szY
    x=np.r_[q.ravel(),np.ones(szL)]
    y=initialize_dual_problem(p,hstar,q)

    # define the problem
    if precondition:
        A_orig,b_orig=polytopes.q_with_fixedph_centered(p,hstar,precondition=False)

        c=np.r_[np.ones(szX*szY),np.ones(szL)]
        A,b,preconditioner=polytopes.precondition_constraint(A_orig,b_orig,return_preconditioner=True)
        y = np.linalg.lstsq(A_orig.T@preconditioner.T,A_orig.T@y,rcond=None)[0]
    else:
        A,b=polytopes.q_with_fixedph_centered(p,hstar,precondition=False)


    # make suer we have feasible starting conditions:
    if ((A.T@y)<=0).any():
        raise Exception("Couldn't find feasible initial condition for dual!")


    # iterate
    gap=np.inf
    while True:
        y=rundual(c,A,b,x,y,dualchunks,tol)
        gap=logslines.dual_objective(c,A,b,y) - logslines.primal_objective(c,A,b,x)
        if gap<tol:
            return x[:szX*szY].reshape((szX,szY))

        x=runprimal(c,A,b,x,y,dualchunks,tol)
        gap=logslines.dual_objective(c,A,b,y) - logslines.primal_objective(c,A,b,x)
        if gap<tol:
            return x[:szX*szY].reshape((szX,szY))



def initialize_dual_problem(p,hstar,qtilde):
    szL,szX=p.shape
    szL,szY=hstar.shape

    lam = szY*(p@qtilde)
    xi = p.T@lam
    gamma=np.zeros(szX)    
    for x in range(szX):
        def normconst(gam):
            if ((gam-xi[x])<0).any():
                return np.inf
            return np.sum(1/(gam-xi[x]))-1

        mx=np.max(1/qtilde[x] + (p.T@lam)[x])
        
        gamma[x]=sp.optimize.bisect(normconst,0,mx)

    return np.r_[lam.ravel(),gamma]

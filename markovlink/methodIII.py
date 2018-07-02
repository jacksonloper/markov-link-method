import numpy as np
import sys
import scipy as sp
import scipy.special
import itertools
from . import misc
import time
from . import estimation

def guess_gamma(Nly,p,q,kappa):
    h=p@q

    szX,szY=q.shape
    szL,szX=p.shape
    p=p.reshape((szL,szX,1))
    q=q.reshape((1,szX,szY))
    Nly=Nly.reshape((szL,1,szY))
    pi = Nly*p*q
    pi = pi/np.sum(pi,axis=1,keepdims=True)
    newq=np.sum(pi*Nly,axis=0) + kappa
    return np.sum(newq,axis=1)

def gam_objective(Nly,p,q,kappa,gamma):
    return np.sum(Nly*np.log(p@q)) + kappa*np.sum(np.log(q)) - np.sum(q.T@gamma)

def calculate_surrogate_grad_and_hessian(Nly,p,q,kappa,gamma):
    szX,szY=q.shape
    szL,szX=p.shape

    h=p@q

    hess=np.zeros((szY,szX,szX))
    for y in range(szY):
        A = (np.sqrt(Nly)/h)[:,y][:,None]*p # ls x xs
        hess[y] = - kappa*np.diag(q[:,y]**(-2)) - A.T@A

    grad = p.T@(Nly/h) + (kappa/q) - gamma[:,None]

    # perversely, the gradient means we should take a gradient step in the direction of grad.T
    # because lam is (L x Y) whereas grad is (Y x L)

    return grad,hess

def calculate_surrogate_search_direction(Nly,p,q,kappa,gamma):
    grad,hess=calculate_surrogate_grad_and_hessian(Nly,p,q,kappa,gamma,)
    return -np.array([np.linalg.solve(h,g) for (h,g) in zip(hess,grad.T)]).T
    # return -np.array([sp.sparse.linalg.lsqr(h,g,damp=0.4,iter_lim=2000)[0] for (h,g) in zip(hess,grad.T)]).T

def update_q_by_surrogate(Nly,p,q,kappa,gamma,searchdirection,beta=.8):
    szX,szY=q.shape
    szL,szX=p.shape

    curobjective = gam_objective(Nly,p,q,kappa,gamma) # want to make this big

    # go forward, but not too much
    s=1.0
    working=False
    while not working:
        newq = q + searchdirection*s
        if (newq<=0).any(): # if we're negative, then we need to make s smaller
            s=s*beta
        elif gam_objective(Nly,p,q,kappa,gamma) < curobjective: # if we got worse then we need to make s smaller
            s=s*beta
        else: # okay, we have at least made some improvement.
            working=True

    return newq

def real_objective(Nly,p,q,kappa):
    q=np.abs(q)
    q=q/np.sum(q,axis=1,keepdims=True)
    return np.sum(Nly*np.log(p@q)) + kappa*np.sum(np.log(q))

def majorization_iteration(Nly,p,q,kappa,gamma=None,normalize=False):
    h = p@q

    szX,szY=q.shape
    szL,szX=p.shape
    p=p.reshape((szL,szX,1))
    q=q.reshape((1,szX,szY))
    Nly=Nly.reshape((szL,1,szY))
    pi = Nly*p*q
    pi = pi/np.sum(pi,axis=1,keepdims=True)
    newq=np.sum(pi*Nly,axis=0) + kappa

    if (gamma is not None) and (not normalize):
        return newq/gamma[:,None]
    elif normalize and (gamma is None):
        return newq/np.sum(newq,axis=1,keepdims=True)
    else:
        raise Exception("Must specify either gamma or normalize (%s %s)"%(gamma,normalize))

    # return  (q*(p.T @ (h/Nly))+kappa) / gamma[:,None]


def calculate_lam_grad_and_hessian(Nly,p,q,kappa,gamma,lam):
    szX,szY=q.shape
    szL,szX=p.shape

    hess=np.zeros((szY,szL,szL))
    grad=np.zeros((szY,szL))
    for y in range(szY):
        A = p / (gamma-p.T@lam[:,y])[None,:] # lams x xs
        hess[y] = -np.diag(Nly[:,y]/lam[:,y]**2) - kappa*(A@A.T) 
        grad[y] = (Nly[:,y]/lam[:,y]) - kappa * np.sum(A,axis=1)

    # perversely, the gradient means we should take a gradient step in the direction of grad.T
    # because lam is (L x Y) whereas grad is (Y x L)

    return grad,hess

def calculate_lam_search_direction(Nly,p,q,kappa,gamma,lam):
    grad,hess=calculate_lam_grad_and_hessian(Nly,p,q,kappa,gamma,lam)
    return -np.array([np.linalg.solve(h,g) for (h,g) in zip(hess,grad)]).T

def lam_objective(Nly,p,kappa,gamma,lam):
    qish = gamma[:,None] - p.T@lam

    if (qish<=0).any():
        return np.inf
    else:
        return np.sum(Nly*np.sum(lam)) + kappa * np.sum(np.log(qish))

def update_lam(Nly,p,q,kappa,gamma,lam,searchdirection,beta=.8):
    szX,szY=q.shape
    szL,szX=p.shape

    curobjective = lam_objective(Nly,p,kappa,gamma,lam) # want to make this big

    # go forward, but not too much
    s=1.0
    working=False
    while not working:
        newlam = lam + searchdirection*s
        newq = calculate_q_from_lam(Nly,p,kappa,gamma,lam)
        if (newq<=0).any(): # if we're negative, then we need to make s smaller
            s=s*beta
        elif lam_objective(Nly,p,kappa,gamma,newlam) < curobjective: # if we got worse then we need to make s smaller
            s=s*beta
        else: # okay, we have at least made some improvement.
            working=True

    return newlam

def calculate_q_from_lam(p,kappa,gamma,lam):
    return kappa / (gamma[:,None] - p.T@lam)

def guess_lam(Nly,p,q,ensure_feasible=True,beta=.9,kappa=None,gamma=None,minq=0.0):
    lam= Nly/(p@q)

    if ensure_feasible:
        q=calculate_q_from_lam(p,kappa,gamma,lam)
        while (q<=minq).any() or (q>=1-minq).any():
            lam=lam*beta
            q=calculate_q_from_lam(p,kappa,gamma,lam)

    return lam
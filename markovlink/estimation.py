import numpy as np
import sys
import scipy as sp
import scipy.special
import itertools
from . import misc
import time
import numpy.random as npr

def initialconds(Nlx,Nly,xytilde=None,pseudocount=1.0):
    szL,szX=Nlx.shape
    szL,szY=Nly.shape

    phat=(Nlx+pseudocount)/np.sum(Nlx+pseudocount,axis=1,keepdims=True)

    qhat=np.ones((szX,szY))/szY

    if xytilde is not None:
        xtilde,ytilde,val = xytilde
        qhat[xtilde]=(1-val)/(szY-1)
        qhat[xtilde,ytilde]=val

    return phat,qhat

def LL(Nlx,Nly,p,h):
    '''
    We return log likelihood of Nlx,Nly under p,h

        sum(Nlx*log(p)) + sum(Nly*log(h))

    with the convention that 0*log(0/0) = 0
    '''

    # compute the log likelihood for Nlx
    good=(Nlx>0)
    Nlxll = np.sum(Nlx[good]*np.log(p[good]))
    
    # compute the log likelihood for Nly    
    good=(Nly>0)
    Nlyll = np.sum(Nly[good]*np.log(h[good]))

    # done!
    return Nlxll+Nlyll

def random_estimate(Nlx,Nly,niter=100):
    '''
    - Grabs a sample from the (Dirichlet) posterior on p,h
    - Projects to MLM
    '''

    p=np.array([npr.dirichlet(x+1) for x in Nlx])
    hhat=np.array([npr.dirichlet(x+1) for x in Nly])

    return p,train_mlm_fixedp(hhat,niter,p)

def random_estimates(Nlx,Nly,niter=100,nsamps=100,alpha=.05):
    parms=[random_estimate(Nlx,Nly,niter=niter) for i in range(nsamps)]
    lls = np.array([LL(Nlx,Nly,p,p@q) for (p,q) in parms])

    good=np.where(lls>=np.percentile(lls,alpha*100))[0]

    return [parms[x] for x in good]

def pseudocount_point_estimate(Nlx,Nly,pseudocountX=1.0,pseudocountY=1.0):
    '''
    Gets point estimates for p,h using pseudocounted empiricals
    '''
    phat=(Nlx+pseudocountX)/np.sum(Nlx+pseudocountX,axis=1,keepdims=True)
    hhat=(Nly+pseudocountY)/np.sum(Nly+pseudocountX,axis=1,keepdims=True)

    return phat,hhat

def spl(Nlx,Nly):
    '''
    Gets number of samples per subpopulation (ell)
    '''
    return np.sum(Nlx,axis=1),np.sum(Nly,axis=1)


##############################################
#  _____          _                     
# |_   _| __ __ _(_)_ __   ___ _ __ ___ 
#   | || '__/ _` | | '_ \ / _ \ '__/ __|
#   | || | | (_| | | | | |  __/ |  \__ \
#   |_||_|  \__,_|_|_| |_|\___|_|  |___/
##############################################
                                      

#########3 fixed p nothing

def train_mlm(Nlx,Nly,p=None,q=None,tol=1e-5,maxtime=60,chunks=100):
    '''
    Tries to solve 

        max_{p,q}   sum(Nlx*log(p)) + sum(Nly*log(p@q)))

    - p,q provide initial conditions (uniform if not provided)

    This problem isn't convex.  So initial conditions matter here!
    '''

    szL,szX=Nlx.shape
    szL,szY=Nly.shape

    if p is None:
        p=np.ones((szL,szX))/szX
    if q is None:
        q=np.ones((szX,szY))/szY

    t=time.time()
    while True:
        for i in range(chunks):
            p,q=train_mlm_update(Nlx,Nly,p,q)

        # check convergence criteria
        newp,newq=train_mlm_update(Nlx,Nly,p,q)
        df = np.max([np.abs(newp-p).max(),np.abs(newq-q).max()])
        if df<tol:
            break
        elif time.time()-t>maxtime:
            raise Exception("Out of time!")

    assert not np.isnan(p).any()
    assert not np.isnan(q).any()

    return p,q

def train_mlm_update(Nlx,Nly,p,q):
    '''
    Uses a minorization technique to find a new value of p,q with a higher value of 
    
        lam*(sum(Nlx*log(p)) + sum(Nly*log(p@q))) + sum(c*log(q))
    
    starting from p,q as an initial guess.

    '''

    # compute Nly / h, with convention that 0/0=0
    Noh=np.zeros(Nly.shape) 
    good=Nly>0
    Noh[good]=Nly[good]/((p@q)[good])

    # form the unnormalized update
    newp = Nlx + p*(Noh@q.T) 
    newq = q*(p.T@Noh) 

    # normalize
    newq = newq/np.sum(newq,axis=1,keepdims=True)
    newp = newp/np.sum(newp,axis=1,keepdims=True)

    # done!
    return newp,newq

#########3 fixed p

def train_mlm_fixedp(Nly,p,q=None,tol=1e-5,maxtime=60,chunks=100):
    '''
    Tries to solve 

        max_{q}   sum(Nly*log(p@q)))

    - q provides initial conditions (uniform if not provided)

    This problem is convex. 
    '''

    szL,szX=p.shape
    szL,szY=Nly.shape

    if p is None:
        p=np.ones((szL,szX))/szX
    if q is None:
        q=np.ones((szX,szY))/szY

    t=time.time()
    while True:
        for i in range(chunks):
            q=train_mlm_update_fixedp(Nly,p,q)

        # check convergence criteria
        newq=train_mlm_update_fixedp(Nly,p,q)
        df = np.abs(newq-q).max()
        if df<tol:
            break
        elif time.time()-t>maxtime:
            raise Exception("Out of time!")

    assert not np.isnan(q).any()

    return q

def train_mlm_update_fixedp(Nly,p,q):
    '''
    Uses a minorization technique to find a new value of p,q with a higher value of 
    
        lam*(sum(Nlx*log(p)) + sum(Nly*log(p@q))) + sum(c*log(q))
    
    starting from p,q as an initial guess.

    '''

    # compute Nly / h, with convention that 0/0=0
    Noh=np.zeros(Nly.shape) 
    good=Nly>0
    Noh[good]=Nly[good]/((p@q)[good])

    # form the unnormalized update
    newq = q*(p.T@Noh) 

    # normalize
    newq = newq/np.sum(newq,axis=1,keepdims=True)

    # done!
    return newq


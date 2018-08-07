import numpy as np
import sys
import scipy as sp
import scipy.stats
import scipy.special
import itertools
from . import misc
import time
from . import logslines
from . import polytopes
import numpy.random as npr
from . import estimation




#  _ _                        _                __ 
# | | |_ __    __ _ _ __   __| |  _ __  _ __  / _|
# | | | '__|  / _` | '_ \ / _` | | '_ \| '_ \| |_ 
# | | | |    | (_| | | | | (_| | | |_) | |_) |  _|
# |_|_|_|     \__,_|_| |_|\__,_| | .__/| .__/|_|  
#                                |_|   |_|        

def LLR(Nlx,Nly,p,h):
    '''
    log likelihood ratio:

        LLR =  sum(Nlx*log(phat/p)) + sum(Nly*log(hhat/h))

    with the convention that 0*log(0/0) = 0
    '''

    # compute empirical averages
    phat = Nlx/np.sum(Nlx,axis=1,keepdims=True)
    hhat = Nly/np.sum(Nly,axis=1,keepdims=True)

    # compute the log likelihood ratios for Nlx
    good=(Nlx>0)
    Nlxratio = np.sum(Nlx[good]*np.log(phat[good]/p[good]))
    
    # compute the log likelihood ratios for Nly    
    good=(Nly>0)
    Nlyratio = np.sum(Nly[good]*np.log(hhat[good]/h[good]))

    # done!
    return Nlxratio+Nlyratio

def estimate_PPF_chisq(alpha,phat,hhat,n_samps=1000):
    '''
    Use asymptotic normality to get an approximate estimate for the value of PPF so that 

        P(LLR(Nlx,Nly,phat,hhat) > PPF) = alpha

    In a way that doesn't depend upon phat,hhat

    '''

    szL,szX=phat.shape
    szL,szY=hhat.shape

    dof=szL*(szX+szY-2)

    return sp.stats.chi2.ppf(alpha,dof)


def estimate_PPF(alpha,splX,splY,phat,hhat,n_samps=1000):
    '''
    Use data to estimate for the value of PPF so that 

        P(LLR(Nlx,Nly,phat,hhat) > PPF) = alpha

    when Nlx is drawn from phat (with splX[ell] samples per subpop ell)
         Nly is drawn from hhat (with splY[ell] samples per subpop ell)
    '''

    samps=PPF_samples(phat,hhat,splX,splY,n_samps=n_samps) 
    return np.percentile(samps,100-100*alpha)

def PPF_samples(phat,hhat,splX,splY,n_samps=1000,pseudocount=1.0):
    '''
    Returns samples of LLR(Nlx,Nly,phat,hhat) where

    when Nlx is drawn from phat (with samp_n[ell] samples per subpop ell)
         Nly is drawn from hhat (with samp_m[ell] samples per subpop ell)
    '''

    samps=np.zeros(n_samps)
    for i in range(n_samps):
        Nlx2=misc.sample(splX,phat)
        Nly2=misc.sample(splY,hhat)
        
        samps[i]=LLR(Nlx2,Nly2,phat,hhat)

    return samps

def confidence_region_LLR(Nlx,Nly,alpha,n_samps=1000):
    '''
    Consider the region

        R={p,q: LLR(Nlx,Nly,p,p@q) < k}

    We attempt to choose k so R contains the truth with probability at least (1-alpha)
    '''

    phat,hhat=estimation.pseudocount_point_estimate(Nlx,Nly,pseudocountX=1.0,pseudocountY=1.0)
    splX,splY=estimation.spl(Nlx,Nly)
    return estimate_PPF(alpha,splX,splY,phat,hhat,n_samps=n_samps)


#  _______  _______ ____  _____ __  __    _    _     
# | ____\ \/ /_   _|  _ \| ____|  \/  |  / \  | |    
# |  _|  \  /  | | | |_) |  _| | |\/| | / _ \ | |    
# | |___ /  \  | | |  _ <| |___| |  | |/ ___ \| |___ 
# |_____/_/\_\ |_| |_| \_\_____|_|  |_/_/   \_\_____|
                                                   

def find_extremal_in_confidence_interval(Nlx,Nly,c,PPF,minlam=1e-10,maxlam=1000,maxtime=120,niter=None):
    '''
    We attempt to maximize

        sum(c*log(q))

    within the space of (p,q) so that LLR(Nlx,Nly,p,p@q) < PPF.

    The problem is not convex, so the answer may depend on the initial condition
    for (p,q).  
    '''

    szL,szX=Nlx.shape
    szL,szY=Nly.shape

    # ==== Get niter if necessary
    if niter is None:
        newp,newq,niter=find_feasible_point(Nlx,Nly,PPF,maxtime=maxtime)
        niter=niter*2

    # === DEFINE FUNCTION TO HELP US FIT LAM TO THE PPF
    def lval(lam):
        newp,newq=train_extremal(Nlx,Nly,c,lam,niter)
        return LLR(Nlx,Nly,newp,newp@newq)-PPF
    # In terms of this, our task is to find the smallest lam>=0 so that lval(lam) < 0

    # ==== CHECK THAT OUR CONSIDERED LAMS ARE GOOD
    minlamppf = lval(minlam)
    maxlamppf = lval(maxlam)

    # ==== FIND RIGHT LAM
    if minlamppf < 0:  # if even with negligible lambda we still achieve lval < 0, return that
        return train_extremal(Nlx,Nly,c,minlam,niter)
    elif maxlamppf > 0: # if even with large lambda we can't achieve feasible, yelp
        raise Exception("Couldn't find feasible solution",dict(smalllam=minlamppf+PPF,biglam=maxlamppf+PPF,target=PPF))
    else: # otherwise, find the smallest lambda that gives us lval < 0
        bestlam = sp.optimize.bisect(lval,minlam,maxlam)
        return train_extremal(Nlx,Nly,c,bestlam,niter)


def train_extremal(Nlx,Nly,c,lam,iterations,p=None,q=None):
    '''
    Tries to solve 

        max_{p,q}   lam*(sum(Nlx*log(p)) + sum(Nly*log(p@q))) + sum(c*log(q))

    By running `iteration` minorized updates from the initial conditions.
    This problem isn't convex.  So initial conditions matter here!
    '''

    assert (c>=0).all()
    assert lam>0

    # initialize
    pguess,qguess=estimation.initialconds(Nlx,Nly)
    if p is None:
        p=pguess
    if q is None:
        q=qguess

    # iterate
    for i in range(iterations):
        p,q=train_extremal_update(Nlx,Nly,p,q,c,lam)

    assert not np.isnan(p).any()
    assert not np.isnan(q).any()

    return p,q


def train_extremal_update(Nlx,Nly,p,q,c,lam):
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
    newq = c + lam*q*(p.T@Noh) 

    # normalize
    newq = newq/np.sum(newq,axis=1,keepdims=True)
    newp = newp/np.sum(newp,axis=1,keepdims=True)

    # done!
    return newp,newq


#   __ _           _    __                _ _     _      
#  / _(_)_ __   __| |  / _| ___  __ _ ___(_) |__ | | ___ 
# | |_| | '_ \ / _` | | |_ / _ \/ _` / __| | '_ \| |/ _ \
# |  _| | | | | (_| | |  _|  __/ (_| \__ \ | |_) | |  __/
# |_| |_|_| |_|\__,_| |_|  \___|\__,_|___/_|_.__/|_|\___|
                                                       

def find_feasible_point(Nlx,Nly,PPF,maxtime=10,plusiter=1):
    '''
    Find (p,q) in the confidence interval given by PPF

    - p : initial condition for p
    - q : intial condtion for q
    - PPF : value to be beat

    returns
    - (p,q) : feasible point
    - niter : number of iterations required to find the feasible point
    '''

    szL,szX=Nlx.shape
    szL,szY=Nly.shape

    p,q=estimation.initialconds(Nlx,Nly)

    t=time.time()
    niter=0
    while True:
        niter=niter+100
        p,q=estimation.train_mlm(Nlx,Nly,100,p,q)
        if LLR(Nlx,Nly,p,p@q)<PPF:
            break
        if time.time()-t>maxtime:
            raise OutOfTime("Search for a calibration in the confidence region terminated, due to insufficient time.  This can happen if the model is wrong or the sample size is very large.  Consider increasing maxtime.",
                dict(curLLR=LLR(Nlx,Nly,p,p@q),targetPPF=PPF,niter=niter,timespent=time.time()-t,p=p,q=q))


    p,q=estimation.train_mlm(Nlx,Nly,niter*plusiter,p,q)

    return p,q,niter*(1+plusiter)
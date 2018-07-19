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

class OutOfTime(Exception):
    pass
class OutOfIterations(Exception):
    pass

def LL_ph(Nlx,Nly,p,h):
    '''
    We return log likelihood of Nlx,Nly under p,h

        sum(Nlx*log(p)) + sum(Nly*log(h))

    with the convention that 0*log(0/0) = 0
    '''


    # compute the log likelihood for Nlx
    good=(Nlx>0)
    Nlxratio = np.sum(Nlx[good]*np.log(p[good]))
    
    # compute the log likelihood for Nly    
    good=(Nly>0)
    Nlyratio = np.sum(Nly[good]*np.log(h[good]))

    # done!
    return Nlxratio+Nlyratio

def LLR(Nlx,Nly,p,h):
    '''
    LLR =  mean(Nlx*log(phat/p)) + mean(Nly*log(hhat/h))

    with the convention that 0*log(0/0) = 0
    '''

    # compute empirical averages
    phat = Nlx/np.sum(Nlx,axis=1,keepdims=True)
    hhat = Nly/np.sum(Nly,axis=1,keepdims=True)

    # compute the log likelihood ratios for Nlx
    good=(Nlx>0)
    Nlxratio = np.mean(Nlx[good]*np.log(phat[good]/p[good]))
    
    # compute the log likelihood ratios for Nly    
    good=(Nly>0)
    Nlyratio = np.mean(Nly[good]*np.log(hhat[good]/h[good]))

    # done!
    return Nlxratio+Nlyratio

def estimate_PPF(Nlx,Nly,alpha,n_samps=1000):
    '''
    Get an approximate estimate for the value of PPF so that 

        P(LLR(Nlx,Nly,pstar,hstar) > PPF) = alpha

    when Nly is drawn from pstar 
         Nly is drawn from hstar

    '''

    # we assume the TRUTH is given by pseudocounted empirical distributions
    # (the pseudocounts end up giving us a more conservative estimator)
    # we then look at samples from the distribution of LLR
    samps=PPF_samples(Nlx,Nly)

    # we estimate our object of interest based on these samples
    return np.percentile(samps,100-100*alpha)

def estimate_alpha(Nlx,Nly,PPF,n_samps=1000):
    '''
    Get an approximate estimate for 

        P(LLR(Nlx,Nly,pstar,hstar) > PPF)

    when Nly is drawn from pstar 
         Nly is drawn from hstar

    '''

    # we assume the TRUTH is given by pseudocounted empirical distributions
    # (the pseudocounts end up giving us a more conservative estimator)
    # we then look at samples from the distribution of LLR
    samps=PPF_samples(Nlx,Nly)

    # we estimate our object of interest based on these samples
    return np.mean(samps>PPF)

def PPF_samples(Nlx,Nly,n_samps=1000,pseudocount=1.0):
    '''
    Returns samples of LLR(Nlx2,Nly2,pstar,hstar) where 

    - pstar,hstar are pseudocounted empirical distributions from Nlx,Nly
    - Nlx,Nly2 are drawn from multinomials of pstar,hstar with as many samples per row as Nlx,Nly
    '''

    # we assume the TRUTH is given by pseudocounted empirical distributions
    # (the pseudocounts end up giving us a more conservative estimator)
    phat=(Nlx+pseudocount)/np.sum(Nlx+pseudocount,axis=1,keepdims=True)
    hhat=(Nly+pseudocount)/np.sum(Nly+pseudocount,axis=1,keepdims=True)

    # if this WAS the truth, what would be the
    # distribution of LLR(Nlx,Nly,phat,hhat)?
    # we can draw samples to get a sense
    samps=np.zeros(n_samps)
    for i in range(n_samps):
        Nlx2=misc.sample(np.sum(Nlx,axis=1),phat)
        Nly2=misc.sample(np.sum(Nly,axis=1),hhat)
        
        samps[i]=LLR(Nlx2,Nly2,phat,hhat)

    return samps


def find_feasible_point(Nlx,Nly,PPF, p=None,q=None,maxtime=10):
    '''
    Find (p,q) so that LLR(Nlx,Nly,p,p@q) < PPF

    - p : initial condition for p
    - q : intial condtion for q
    - PPF : value to be beat

    returns
    - (p,q) : feasible point
    - niter : number of iterations required to find the feasible point
    '''

    szL,szX=Nlx.shape
    szL,szY=Nly.shape

    if p is None:
        p=np.ones((szL,szX))/szX
    if q is None:
        q=np.ones((szX,szY))/szY

    t=time.time()
    niter=0
    while True:
        niter=niter+100
        p,q=train_mlm_minorized(Nlx,Nly,p,q,0,1,100)
        if LLR(Nlx,Nly,p,p@q)<PPF:
            break
        if time.time()-t>maxtime:
            raise OutOfTime("Search for a calibration in the confidence region terminated, due to insufficient time.  This can happen if the model is wrong or the sample size is very large.  Consider increasing maxtime.",
                dict(curLLR=LLR(Nlx,Nly,p,p@q),targetPPF=PPF,niter=niter,timespent=time.time()-t,p=p,q=q))

    return p,q,niter


def find_extremal_in_confidence_interval(Nlx,Nly,p,q,c,alpha,minlam=1e-10,maxlam=1000,maxtime=120):
    '''
    We determine a likelihood-ratio-based approximate confidence interval for p,q
    with nominal coverage (1-alpha).  

    Within this confidence interval, we attempt to maximize

        sum(c*log(q))

    The problem is not convex, so the answer may depend on the initial condition
    for (p,q).  
    '''

    szL,szX=Nlx.shape
    szL,szY=Nly.shape


    '''
    === CHOOSING THE PPF

    Confidence interval is defined as the set of p,q such that

        LLR(p,p@q) < PPF

    We try to choose PPF so that the coverage is about right:
    '''
    PPF=estimate_PPF(Nlx,Nly,alpha)

    '''
    === HOW MANY ITERATIONS?

    We will use `train_mlm_minorized` in our inner loop,
    but its not clear how many iterations we need.
    We determine this by finding out how many iterations
    of training we need to find a single feasible point
    in the confidence interval.  We then double that number.

    When later we are actually optimizing sum(c*log(q))
    within this confidence interval, in practice we find that
    need fewer iterations; the objective function has 
    some kind of smoothing effect.  
    '''

    newp,newq,niter=find_feasible_point(Nlx,Nly,PPF,p=p,q=q,maxtime=maxtime)
    niter=niter*2

    # === DEFINE FUNCTION TO HELP US FIT LAM TO THE PPF
    def lval(lam):
        newp,newq=train_mlm_minorized(Nlx,Nly,p,q,c,lam,niter)
        return LLR(Nlx,Nly,newp,newp@newq)-PPF
    '''
    We need to find the smallest lam so that

        lval(lam) < 0
    '''


    # ==== GET INITIAL BOUNDS
    minlamppf = lval(minlam)
    maxlamppf = lval(maxlam)

    # FIND LAM
    if minlamppf < 0:  # if even with negligible lambda we still achieve lval < 0, return that
        return train_mlm_minorized(Nlx,Nly,p,q,c,minlam,niter)
    elif maxlamppf > 0: # if even with large lambda we can't achieve feasible, yelp
        raise Exception("Couldn't find feasible solution",dict(smalllam=minlamppf+PPF,biglam=maxlamppf+PPF,target=PPF))
    else: # otherwise, find the smallest lambda that gives us lval < 0
        bestlam = sp.optimize.bisect(lval,minlam,maxlam)
        return train_mlm_minorized(Nlx,Nly,p,q,c,bestlam,niter)



def train_mlm_minorized(Nlx,Nly,p,q,c,lam,iterations):
    '''
    Tries to solve 

        max_{p,q}   lam*(sum(Nlx*log(p)) + sum(Nly*log(p@q))) + sum(c*log(q))

    By running `iteration` minorized updates from the initial conditions.
    This problem isn't convex.  So initial conditions matter here!
    '''

    for i in range(iterations):
        p,q=train_mlm_minorized_update(Nlx,Nly,p,q,c,lam)

    assert not np.isnan(p).any()
    assert not np.isnan(q).any()

    return p,q


def train_mlm_minorized_update(Nlx,Nly,p,q,c,lam):
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
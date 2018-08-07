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
from . import globalconf
import functools
import bisect




###############################################
#  _ _                        _                __ 
# | | |_ __    __ _ _ __   __| |  _ __  _ __  / _|
# | | | '__|  / _` | '_ \ / _` | | '_ \| '_ \| |_ 
# | | | |    | (_| | | | | (_| | | |_) | |_) |  _|
# |_|_|_|     \__,_|_| |_|\__,_| | .__/| .__/|_|  
#                                |_|   |_|        
###############################################

def LL_prof(Nlx,Nly,niter,xytilde,pseudocount=1.0):
    '''
    We return (approximate) max log likelihood of Nlx,Nly of

        sum(Nlx*log(p)) + sum(Nly*log(h))

    with the convention that 0*log(0/0) = 0, subject
    to the constraint that q[xtilde,ytilde]=val
    '''

    szL,szX=Nlx.shape
    szL,szY=Nly.shape
    xtilde,ytilde,val=xytilde


    # get initial guess
    p,q = estimation.initialconds(Nlx,Nly,xytilde,pseudocount=pseudocount)

    # train
    secondbestp,secondbestq=estimation.train_mlm_fixedpqxy(Nly,niter,xytilde,p,q=q)

    # get ll
    ll=estimation.LL(Nlx,Nly,secondbestp,secondbestp@secondbestq)

    # done!
    return secondbestp,secondbestq,ll

def estimate_PPF(alpha,xtilde,ytilde,splX,splY,pstar,qstar,niter,n_samps=100):
    '''
    Say pstar,qstar is true.

    Say you are going to produce a confidence interval defined by

                C = {val: LL_prof(bestval)-LL_prof(val) < PPF}

    What value of kk should you pick so that with probability (1-alpha) the
    confidence interval contains the low and hi possibilities defined by pstar,qstar?

    We estimate this quantity.  
    '''

    szL,szX=pstar.shape
    szX,szY=qstar.shape
    hstar=pstar@qstar

    # determine qlo and qhi
    direc=np.zeros(qstar.shape)
    direc[xtilde,ytilde]=1
    qlow=polytopes.find_extremal(pstar,qstar,direc)[xtilde,ytilde]
    qhi=polytopes.find_extremal(pstar,qstar,-direc)[xtilde,ytilde]

    # sometimes numerical issues make qlow and qhi slightly outside the parameter space
    # we fix it
    qlow=np.clip(qlow,0,1)
    qhi=np.clip(qhi,0,1)

    # generate surrogate datasets
    surrogate_Nlx=np.zeros((n_samps,szL,szX))
    surrogate_Nly=np.zeros((n_samps,szL,szY))
    for i in range(n_samps):
        Nlx,Nly=misc.sample(splX,pstar),misc.sample(splY,hstar) 
        surrogate_Nlx[i]=Nlx
        surrogate_Nly[i]=Nly

    # evaluate LL_prof at bestq, qlo, and qhi
    LL_profs=np.zeros((n_samps,3))
    for i in range(n_samps):
        optrez=sp.optimize.minimize_scalar(
            lambda v: -LL_prof(surrogate_Nlx[i],surrogate_Nly[i],niter,(xtilde,ytilde,v))[-1],
            method='bounded',bounds=(0,1))
        LL_profs[i,0] = -optrez['fun']
        LL_profs[i,1]=LL_prof(surrogate_Nlx[i],surrogate_Nly[i],niter,(xtilde,ytilde,qlow))[-1]
        LL_profs[i,2]=LL_prof(surrogate_Nlx[i],surrogate_Nly[i],niter,(xtilde,ytilde,qhi))[-1]
    dfs=LL_profs[:,0]-np.min(LL_profs[:,1:],axis=1)

    # we want to find k such that dfs<=k for (1-alpha) of the samples
    PPF =np.percentile(dfs,(1-alpha)*100)
    assert PPF>=0,"PPF should be positive!!"

    return PPF

###############################################
#
#                   __ _       _   
#   ___ ___  _ __  / _(_)_ __ | |_ 
#  / __/ _ \| '_ \| |_| | '_ \| __|
# | (_| (_) | | | |  _| | | | | |_ 
#  \___\___/|_| |_|_| |_|_| |_|\__|
#                                
###############################################

def confinterval_fixedk(Nlx,Nly,xtilde,ytilde,niter,PPF):
    '''
    Returns lower and upper bounds of the confidence interval defined by

                C = {val: LL_prof(bestval)-LL_prof(val) < PPF}

    
    '''

    assert PPF>0

    # define database of findings
    lls={}
    def check(val):
        if val not in lls:
            lls[val]=LL_prof(Nlx,Nly,niter,(xtilde,ytilde,val))[-1]
        return -lls[val]

    # get best ll
    bestx=sp.optimize.minimize_scalar(check,method='bounded',bounds=(0,1))['x']

    # we now define the confidence interval, in terms of check2:
    def check2(val): # val is feasible if check2<0
        return lls[bestx]+check(val)-PPF

    # now figure out the extremes of the confidence itnerval thus defined
    lower=0
    upper=1

    if check2(0)<0: # if 0 is feasible
        lower=0
    else:
        lower = sp.optimize.bisect(check2,0,bestx)

    if check2(1)<0: # if 1 is feasibel
        upper=1
    else:
        upper = sp.optimize.bisect(check2,bestx,1)

    vals=np.sort(np.array(list(lls.keys())))
    lls=lls[bestx]-np.array([lls[x] for x in vals])

    return (lower,upper),(vals,lls)

def confinterval(Nlx,Nly,xtilde,ytilde,alpha=.05,CI_niter=100,postiter=1,n_samps=100,bestp=None,bestq=None):

    # get estimate
    if (bestp is None) or (bestq is None):
        global_LLR_lim=globalconf.confidence_region_LLR(Nlx,Nly,.05)
        bestp,bestq,niter=globalconf.find_feasible_point(Nlx,Nly,global_LLR_lim,maxtime=10)
        bestp,bestq=estimation.train_mlm(Nlx,Nly,niter*postiter,p=bestp,q=bestq)

    # get PPF based on that estimate
    splX,splY=estimation.spl(Nlx,Nly)
    PPF=estimate_PPF(alpha,xtilde,ytilde,splX,splY,bestp,bestq,CI_niter,n_samps=n_samps)

    # get CI based on that PPF
    return confinterval_fixedk(Nlx,Nly,xtilde,ytilde,CI_niter,PPF)[0]


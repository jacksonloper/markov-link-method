from . import estimation
from . import exploration
from . import misc
import numpy as np
from . import minorization

def hypothesis_test(Nlx,Nly,alpha=.05,maxtime=120):
    p,q=point_estimates(Nlx,Nly)
    PPF=minorization.LLR(Nlx,Nly,p,p@q)
    PPF = minorization.estimate_PPF(Nlx,Nly,alpha)

    p,q,niter=minorization.find_feasible_point(Nlx,Nly,PPF,maxtime=maxtime)

    return "fail to reject"

def point_estimates(Nlx,Nly,maxtime=10):
    '''
    Get point estimates for the calibration
    '''

    # find a q inside an approximate 95% confidence interval
    PPF = minorization.estimate_PPF(Nlx,Nly,.05)
    p,q,niter=minorization.find_feasible_point(Nlx,Nly,PPF,maxtime=maxtime)

    # keep seeking a q that better optimizes the likelihood of the data,
    # for twice the number of iterations it took to find the feasible point:
    p,q = minorization.train_mlm_minorized(Nlx,Nly,p,q,0,1,niter*2)

    return p,q

def uncertainty_assessment(Nlx,Nly,alpha=.05,verbose=True,n_extremals=50):
    '''
    Get a measure of our uncertainty about the calibration, in the form of a collection of 
    calibrations lying within an approximate (1-alpha) confidence interval
    '''

    return exploration.construct_interval_samples(Nlx,Nly,n_extremals=n_extremals,alpha=alpha,verbose=verbose)
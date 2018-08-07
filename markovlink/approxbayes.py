from . import estimation
from . import misc
import numpy as np
import numpy.random as npr
from . import polytopes

def posterior_samples(Nlx,Nly,nsamps=100,verbose=False,clipthresh=1e-6):
    estimates=[]
    if verbose:
        misc.pnn("\n %d samples to produce:"%nsamps)
    for i in range(nsamps):
        if verbose:
            misc.pnn(i)
        p=np.array([npr.dirichlet(x+1) for x in Nlx])
        htilde=np.array([npr.dirichlet(x+1) for x in Nly])

        qtilde=estimation.train_mlm_fixedp(htilde,p)

        # get the central q, for uniqueness
        q = polytopes.find_central(p,qtilde,clipthresh=1e-6)
        
        estimates.append((p,q))

    return estimates


def credible_interval(estimates,xtilde,ytilde,alpha=.05,verbose=False):
    lower,upper=get_extremes(estimates,xtilde,ytilde,verbose=verbose)

    lower=np.percentile(lower[:,xtilde,ytilde],100*(alpha*.5))
    upper=np.percentile(upper[:,xtilde,ytilde],100*(1-alpha*.5))

    return np.clip(lower,0,1),np.clip(upper,0,1)

def get_extremes(estimates,xtilde,ytilde,verbose=False):
    szL,szX=estimates[0][0].shape
    szX,szY=estimates[0][1].shape

    lower=np.ones((len(estimates),szX,szY))
    upper=np.ones((len(estimates),szX,szY))

    good=np.ones(len(estimates),dtype=np.bool)

    for i,(ptilde,qtilde) in enumerate(estimates):
        if verbose:
            misc.pnn(i)

        direc=np.zeros((szX,szY))
        direc[xtilde,ytilde]=1

        lower[i]=polytopes.find_extremal(ptilde,qtilde,direc)
        upper[i]=polytopes.find_extremal(ptilde,qtilde,-direc)

    return lower,upper

def credible_intervals(estimates,xys,alpha=.05,verbose=False,onfail='throw'):
    CIs=np.ones((len(xys),2))*np.nan

    if verbose:
        misc.pnn("\n %d parameters look at:"%len(xys))

    for i,(x,y) in enumerate(xys):
        if verbose:
            misc.pnn("%d"%i)

        if onfail=='warn':
            try:
                CIs[i] = credible_interval(estimates,x,y,alpha=alpha)
            except Exception as e:
                misc.pnn("(fail %s)"%str(e))
        else:
            CIs[i] = credible_interval(estimates,x,y,alpha=alpha)
    return CIs

def estimates(Nlx,Nly,alpha=.05,nsamps=100,verbose=False,subCIs=None,onfail='throw'):
    szL,szX=Nlx.shape
    szL,szY=Nly.shape

    # get posterior samples
    estimates=posterior_samples(Nlx,Nly,nsamps=nsamps,verbose=verbose)

    # get estimator
    qhat=np.mean([q for (p,q) in estimates],axis=0)

    # get credible intervals
    if subCIs is None:
        Y,X=np.meshgrid(np.r_[0:szY],np.r_[0:szX])
        xys=np.c_[X.ravel(),Y.ravel()]
        CIs=credible_intervals(estimates,xys,alpha=alpha,verbose=verbose,onfail=onfail)
        CIs=CIs.reshape((szX,szY,2))
    else:
        CIs=credible_intervals(estimates,subCIs,alpha=alpha,verbose=verbose,onfail=onfail)
    
    return qhat,CIs

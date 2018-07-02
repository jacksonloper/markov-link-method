from . import estimation
from . import exploration
from . import misc
import numpy as np
from . import majorization

def point_estimates(Nlx,Nly,pseudocount=1.0):
    '''
    Get point estimates by the Markov Link Method
    '''

    p = (Nlx + pseudocount) / np.sum(Nlx+pseudocount,axis=1,keepdims=True)

    kappa=.001
    q=majorization.train_q_by_majorization(p,Nly+pseudocount,kappa,n_train=500)

    h=p@q

    return p,q,p@q

def bootstrap(Nlx,Nly,phat,qhat,n_straps=50,verbose=False):
    '''
    Draws surrogate datasets for Nlx and Nly, with replacement, and then gets point estimates
    for the surrogate datasets using the Markov Link Method.

    Also estimates the Deltap,Deltaq,Deltah
    '''

    hhat=phat @ qhat

    straps=[]

    dsts=np.zeros((n_straps,3))

    if verbose:
        misc.pnn("%d bootstraps:"%n_straps)
    for i in range(n_straps):
        if verbose:
            misc.pnn(i)
        Nlx2=exploration.resample(Nlx)
        Nly2=exploration.resample(Nly)

        p,q,h =point_estimates(Nlx2,Nly2,pseudocount=0.0)

        dsts[i,0] = .5*np.sum(np.abs(p-phat))
        dsts[i,1] = .5*np.sum(np.abs(q-qhat))
        dsts[i,2] = .5*np.sum(np.abs(h-hhat))

        straps.append(dict(Nlx=Nlx2,Nly=Nly2,estimates=(p,q,h)))

    return straps,np.mean(dsts,axis=0)

def diameter_estimates(straps,verbose=False):
    diams=[]

    if verbose:
        misc.pnn("%d bootstraps to consider:"%len(straps))
    for i,strap in enumerate(straps):
        if verbose:
            misc.pnn(i)
        p,q,h = strap['estimates']
        diams.append(exploration.diameter_estimation(p,q)[0])

    return diams

def qualitative_one_row(straps,x,verbose=False):
    Q1Rs=[]
    if verbose:
        misc.pnn("%d bootstraps to consider:"%len(straps))
    for i,strap in enumerate(straps):
        if verbose:
            misc.pnn(i)
        p,q,h = strap['estimates']
        Q1Rs.append(exploration.qualitative_one_row(p,q,x))
    return np.array(Q1Rs)

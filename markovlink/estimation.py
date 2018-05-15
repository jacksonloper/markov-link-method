import numpy as np

import sys

def emupdate(p,q,kappa,Nly):
    d1,d2=p.shape
    d2,d3=q.shape
    pi = p.reshape((d1,d2,1)) * q.reshape((1,d2,d3))
    pi=pi/np.sum(pi,axis=1,keepdims=True)
    
    L=pi*Nly.reshape((d1,1,d3))
    L=np.sum(L,axis=0) + kappa
    
    L = L / np.sum(L,axis=1,keepdims=True)
    
    return L


def check_transition_matrix(a):
    return (np.abs(np.sum(a,axis=1) - 1).max()<1e-9) and (np.min(a)>-1e-9)

def train_q(p,Nly,iterations=5000,kappa=.0001,verbose=False):
    '''
    Uses EM algorithm to find the qhat which optimizes of the markov link method objective.

    - p -- estimate empirical distribution of X|l (each row sums to 1.0)
    - Nly -- Nly[l,y] is the number of times samples from subpopulation l yielded Y=y 
        (each row sums to the number of samples for subpopulation l under technique II)
    - kappa -- regularization 
    - verbose -- if true, prints out updates as it trains to indicate progress
    '''

    Nly = np.require(Nly,dtype=np.float)
    p=np.require(p,dtype=np.float)

    assert len(Nly.shape)==2,"Nly should be a matrix"
    assert len(p.shape)==2,"p should be a matrix"
    d1,d2=Nly.shape
    d1p,d3=Nly.shape
    assert d1==d1p,'The first dimensions of Nly and p should be the same'
    assert np.abs(np.sum(p,axis=1)-1).max()<1e-9,"The rows of p should sum to 1 and all entries should be positive"

    # initialize qxy
    qxy = np.ones((p.shape[1],Nly.shape[1])) / Nly.shape[1]

    # run emupdate repeatedly
    if verbose:
        # ersatz tqdm replacement (didn't want to include tqdm as a requirement)

        update_interval=1+iterations//20
        for i in range(iterations): 
            if i%update_interval==0:
                sys.stdout.write('(%d/%d)'%(i,iterations))
                sys.stdout.flush()
            qxy=emupdate(p,qxy,kappa,Nly)
    else:
        for i in range(iterations):
            qxy=emupdate(p,qxy,kappa,Nly)

    return qxy

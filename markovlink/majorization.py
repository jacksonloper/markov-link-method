import numpy as np
import sys
import scipy as sp
import scipy.special
import itertools
from . import misc
import time

def train_q_by_majorization(p,Nly,kappa,q=None,n_train=10):
    '''
    Maximizes

    L(q) = sum_{l,y} Nly log(sum_x plx qxy) + sum_{x,y} kappa log(qxy)

    by a majorization algorithm

    '''

    # initialize
    Nly=np.require(Nly)
    l,x =p.shape
    l,y = Nly.shape

    assert misc.check_transition_matrix(p),"The rows of p should sum to 1 and all entries should be positive"
    if q is None:
        q=np.ones((x,y),dtype=np.float64)/y  
    else:
        assert misc.check_transition_matrix(q),"The rows of q should sum to 1 and all entries should be positive"

    # run iterations
    for i in range(n_train):
        # update
        q = train_mlm_em_update(Nly,p,q,kappa)
       
    return q

def train_mlm_em_update(Nly,p,q,kappa):
    '''
    Uses a majorization technique to find a new value of q with a higher value of 
    
        L(q) = np.sum(Nlx * np.log(p @ q)) 
    
    starting from q as an initial guess.

    '''
    l,x =p.shape
    x,y = q.shape
    
    # compute the majorization
    pi = p.reshape((l,x,1)) * q.reshape((1,x,y))
    pi = pi /np.sum(pi,axis=1,keepdims=True)
    
    # make updates
    newq = np.sum(Nly.reshape((l,1,y)) * pi,axis=0).reshape(x,y)+kappa
    newq = newq/np.sum(newq,axis=1,keepdims=True)
    
    return newq.reshape((x,y))

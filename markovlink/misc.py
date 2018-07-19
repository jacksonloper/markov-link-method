
import numpy as np
import sys
import scipy as sp
import numpy.random as npr


def pnn(x): 
    sys.stdout.write(str(x) + " ")
    sys.stdout.flush()

def check_transition_matrix(a):
    return (np.abs(np.sum(a,axis=1) - 1).max()<1e-9) and (np.min(a)>-1e-9)

def totalvardist(a,b):
    '''
    Returns the average total variation distance between two discrete conditional
    distributions, parameterized by a transition matrix.  
    '''
    return .5*np.mean(np.sum(np.abs(a-b),axis=1),axis=0)

def log_likelihood(Nlx,p):
    good=(Nlx>0)
    return np.sum(Nlx[good]*np.log(p[good]))


def resample(tmx):
    '''
    Resamples each row of tmx, with replacement
    '''
    tmx2=np.zeros(tmx.shape)
    for z in range(len(tmx)):
        p=tmx[z]
        p=p/np.sum(p)
        tmx2[z]=npr.multinomial(np.sum(tmx[z]),p)
    return tmx2


def sample(N,p):
    '''
    Returns a matrix formed by sampling

        Multinomial(N[1],p[1])
        Multinomial(N[2],p[2])
                .
                .
                .
    '''
    tmx2=np.zeros(p.shape)
    for z in range(len(N)):
        tmx2[z]=npr.multinomial(np.sum(N[z]),p[z])
    return tmx2


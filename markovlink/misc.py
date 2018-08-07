
import numpy as np
import sys
import scipy as sp
import numpy.random as npr

def newdataset(Nlx,Nly,p,h):
    '''
    Constructs a new dataset based on the number of samples in Nlx,Nly
    and the distributions in p,h
    '''

    szL,szX=Nlx.shape
    szL,szY=Nly.shape

    Nlx2=np.array([npr.multinomial(np.sum(Nlx[ell]),p[ell]) for ell in range(szL)])
    Nly2=np.array([npr.multinomial(np.sum(Nly[ell]),h[ell]) for ell in range(szL)])

    return Nlx2,Nly2

def empiricals(Nlx,Nly):
    return Nlx/np.sum(Nlx,axis=1,keepdims=True),Nly/np.sum(Nly,axis=1,keepdims=True)

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


def train_test_split(Nly,m):
    '''
    Input:
    - an array of positive integers, Nly

    Output:
    - an array train,test

    For each row, we hold out m of the counts from Nly
    '''

    assert (np.sum(Nly,axis=1)>=m).all(),"Not all rows of Nly have %d total count; can't hold out that much"%m

    Nly=np.require(Nly)
    Nshp=Nly.shape

    # sample (l,Y) events, uniformly without replacement
    # from the (l,Y) events indicated by Nly

    # we iteratively select a random point from Nly
    # and move it into our new datset
    Nly=Nly.copy().ravel() # this will hold train
    Nnew = np.zeros(len(Nly)) # this will hold test
    for l in range(Nshp[0]):
        for i in range(m):
            # choose an event in the lth row of Nly to sample
            idxs=np.r_[0:Nshp[1]]+Nshp[1]*l
            idx=npr.choice(idxs,size=1,p=Nly[idxs]/np.sum(Nly[idxs]))[0]
            Nly[idx]=Nly[idx]-1
            Nnew[idx]=Nnew[idx]+1
    Nnew=Nnew.reshape(Nshp)

    return Nly.reshape(Nshp),Nnew.reshape(Nshp)



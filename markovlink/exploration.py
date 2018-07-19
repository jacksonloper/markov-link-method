'''
Some tools for bootstrap and exploring the polytopes of interest
'''


import scipy as sp
import scipy.optimize
import numpy as np
import numpy.random as npr
import sys

from . import misc
from . import minorization

def construct_interval_samples(Nlx,Nly,n_extremals=50,alpha=.05,verbose=True):
    '''
    Construct extremal samples to help us explore the confidence interval
    '''
    szL,szX=Nlx.shape
    szL,szY=Nly.shape

    Ms=np.zeros((n_extremals,szX,szY))

    if verbose:
        misc.pnn("%d samples:"%n_extremals)
    for i in range(n_extremals):
        if verbose:
            misc.pnn(i)

        # random initial conditions
        p=np.array([npr.dirichlet(np.ones(szX)) for l in range(szL)])
        q=np.array([npr.dirichlet(np.ones(szY)) for l in range(szX)])

        # random direction of search
        c=npr.exponential(size=(szX,szY))

        # find the most extreme q in the confidence interval in that direction
        p,q=minorization.find_extremal_in_confidence_interval(Nlx,Nly,p,q,c,alpha)
        Ms[i]=q

    return Ms



##################################
########## TEST TRAIN SPLIT ######
##################################


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


import numpy as np
import numpy.random as npr


def project_to_polytope(p,qdef,newq):
    '''
    Given 
    - polytope defined by p @ q = p @ qdef and q>=0
    - a point newq

    Output
    - the point in the polytope which is closest to newq

    '''

    try:
        import cvxopt

        raveln= np.prod(newq.shape)

        # equality constraint is that A q = b
        A,b = formulate_polytope(p,qdef)

        # inequality constraint is that 
        # (-I)q<=0
        G=-np.eye(raveln)
        h=np.zeros(raveln)

        # thing to minimize is
        #   .5 |q - newq|^2
        # = .5 |q|^2 - <newq,q> + ...
        # = .5 q^T I q + (-newq^T) q + ...
        P = np.eye(raveln)
        d = -newq.ravel()

        '''
        min   |q - qxy|^2
        subj  p q = p qdef

        Has become
        
        min   .5 q^T P q + d^T q
        subj  Aq = b
              Gq <= h

        '''
        
        mtx=lambda x: cvxopt.matrix(x)
        
        rez=cvxopt.solvers.qp(mtx(P),mtx(d),G=mtx(G),h=mtx(h),A=mtx(A),b=mtx(b))
        
        if rez['status']=='optimal':
            return np.array(rez['x']).reshape(newq.shape)
        else:
            raise Exception("Polytope projection failed!")
    except ModuleNotFoundError:
        raise("You don't seem to have CVXOPT.  We depend on CVXOPT for polytope projection :(.  Try conda install cvxopt?")
    

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
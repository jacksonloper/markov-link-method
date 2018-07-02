import scipy as sp
import scipy.optimize
import numpy as np
import numpy.random as npr
import sys

from . import misc

###################################
########## QUALITATIVE-1-ROW ######
###################################

def qualitative_one_row(phat,qhat,x):
    szL,szX=phat.shape
    szX,szY=qhat.shape

    block=np.zeros((szY,szY))

    for y in range(szY):
        direc=np.zeros((szX,szY))
        direc[x,y]=1
        block[y] = find_extremal(phat,qhat,-direc)[x]

    return block

##################################
########## DIAMETER ESTIMAT ######
##################################

def diameter_estimation(phat,qhat,n_samps=50):
    '''
    Estimates the total variation diameter (that is, one half of the L1 diameter) of the polytope

    Theta(phat,hhat) = {q: phat@q= hhat}

    where hhat = phat @ qhat


    '''

    szL,szX=phat.shape
    szX,szY=qhat.shape

    hhat= phat@qhat

    diam=0
    samps=[]

    for i in range(n_samps):
        direc=(npr.randn(szX,szY)>0)*2-1 # L1 test directions
        newq1 = find_extremal(phat,qhat,direc)
        newq2 = find_extremal(phat,qhat,-direc)
        samps.append([newq1,newq2])
        dst=misc.totalvardist(newq1,newq2)
        diam =np.max([diam,dst])

    return diam,samps

##################################
########## EXTREMAL FINDING ######
##################################

def find_extremal(plx,qxy,c,use_cvxopt=True):
    '''
    Consider the polytope of transition matrices q such that plx @ q = plx @ qxy.
    Find the most extremal vertex of this polytope in the direction of c.
    '''

    # make sure inputs are in a good form
    plx = np.require(plx,dtype=np.float)
    qxy=np.require(qxy,dtype=np.float)
    c = np.require(c,dtype=np.float).ravel()
    assert c.shape[0]==np.prod(qxy.shape),"c should be the same shape as qxy"
    
    # polytope can be defined by A_eq q = b_eq, q>=0:
    A_eq,b_eq = formulate_polytope_fixedph(plx,plx@qxy)

    # find relevant extremal point with interior point method   
    if use_cvxopt:
        return cvxopt_linprog(c,A_eq,b_eq,qxy.ravel()).reshape(qxy.shape)
    else:
        return scipy_linprog(c,A_eq,b_eq).reshape(qxy.shape)


def scipy_linprog(c,A_eq,b_eq):
    rez=sp.optimize.linprog(c,A_eq=A_eq,b_eq=b_eq,method='interior-point',options=dict(sym_pos=True,lstsq=True))
    if rez['success']:
        return rez['x']
    else:
        raise Exception("Linear programming failed!")    

def cvxopt_linprog(c,A_eq,b_eq,initialcondition):
    try:
        import cvxopt
    except:
        raise Exception("Failed to import cvxopt.  Maybe try with use_cvxopt=False?  That will use scipy instead, which is sometimes a bit slower though.  Or try conda install cvxopt.")

    G=-np.eye(A_eq.shape[1])
    g_eq = np.zeros(A_eq.shape[1])

    com=lambda x: cvxopt.matrix(x)
    rez = cvxopt.solvers.lp(com(c),com(G),com(g_eq),A=com(A_eq),b=com(b_eq),
                            primalstart=dict(x=com(initialcondition),s=com(initialcondition)),
                           solver='glpk')

    '''
    minimize <c, x>

    subj to x  = s
           Ax  = b
            s >= 0
    '''

    
    if rez['status']=='optimal':
        return np.array(rez['x'])
    else:
        return Exception("Linear programming failed!") 


##################################
########## DEFINING THETA ######
##################################

def formulate_polytope_fixedph(plx,hly):
    '''
    make explicit the equality constraints 
    that define {q: p @ q = p @ qstar}
    '''
    plx = np.require(plx,dtype=np.float)
    hly=np.require(hly,dtype=np.float)

    assert len(plx.shape)==2,"plx should be a matrix"
    assert len(hly.shape)==2,"hly should be a matrix"
    szL,szX=plx.shape
    szL,szY=hly.shape
    assert misc.check_transition_matrix(plx),"The rows of plx should sum to 1 and all entries should be positive"
    assert misc.check_transition_matrix(hly),"The rows of hly should sum to 1 and all entries should be positive"

    # plx @ q = hzy (note we skip the last column of hzy)
    A_eq1 = np.zeros((szL,szY-1,szX,szY))
    for z in range(szL):
        for y in range(szY-1):
            A_eq1[z,y,:,y] = plx[z,:]
    A_eq1 = A_eq1.reshape((szL*(szY-1),szX*szY))
    b_eq1 = hly[:,:-1].ravel()

    # q is a transition matrix 
    A_eq2 = np.zeros((szX,szX,szY))
    b_eq2=np.ones(szX)
    for x in range(szX):
        A_eq2[x,x]=1
    A_eq2=A_eq2.reshape((szX,szX*szY))
    b_eq2=b_eq2.ravel()

    # all eqs
    A_eq = np.concatenate([A_eq1,A_eq2])
    b_eq = np.concatenate([b_eq1,b_eq2])
    
    return A_eq,b_eq

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

##################################
########## RESAMPLE STUFF ######
##################################


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


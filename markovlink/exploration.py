import scipy as sp
import scipy.optimize
import numpy as np
import numpy.random as npr
import sys

from . import estimation

def sample_prux(plx,qxy,n):
    '''
    Consider the polytope of transition matrices q such that plx @ q = plx @ qxy.

    Choose a direction uniformly at random, and find the pair of vertices in that polytope
    which are most extremal along that axis.  Calculate the distance between those vertices
    projected along the direction, as well as their distance in absolute terms.  

    Repeat n times.

    Returns a list of the form

    [
        (direction,samp1,samp2,distance_along_axis,absolute_distance),
        (direction,samp1,samp2,distance_along_axis,absolute_distance),
        (direction,samp1,samp2,distance_along_axis,absolute_distance),
    ]
    '''

    lst=[]
    for i in range(n):
        direc=npr.randn(*qxy.shape)
        direc=direc/np.linalg.norm(direc)
        samp1=find_extremal(plx,qxy,direc)
        samp2=find_extremal(plx,qxy,direc2)
        lst.append((direc,samp1,samp2,np.sum(direc*(samp1-samp2)),np.linalg.norm(samp1-samp2)))
    return lst

def sample_rux(plx,qxy,n,verbose=True,update_interval=1):
    '''
    Consider the polytope of transition matrices q such that plx @ q = plx @ qxy.

    Choose a direction uniformly at random, and find vertex in that polytope
    which is most extremal along that axis.  

    Repeat n times.

    Returns a list of samples.
    '''

    lst=[]
    for i in range(n):
        if verbose and i%update_interval==0:  # ersatz tqdm replacement (didn't want to include tqdm as a requirement)
            sys.stdout.write('(%d/%d)'%(i,n))
            sys.stdout.flush()
        direc=npr.randn(*qxy.shape)
        samp1=find_extremal(plx,qxy,direc)
        lst.append(samp1)
    return np.array(lst)

def find_extremal(plx,qxy,c):
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
    A_eq,b_eq = formulate_polytope(plx,qxy)

    # find relevant extremal poitn with interior point method    
    rez=sp.optimize.linprog(c,A_eq=A_eq,b_eq=b_eq,method='interior-point',options=dict(sym_pos=True,lstsq=True))
    
    if rez['success']:
        return rez['x'].reshape(qxy.shape)
    else:
        raise Exception("Linear programming failed!")

def sample_uniform(plx,qxy,n,dikinshrink=2,verbose=False):
    '''
    Consider the polytope of transition matrices q such that plx @ q = plx @ qxy.

    Attempt to obtain sampes from the uniform distribution on this polytope by taking n
    steps in a metropolis hastings sampler, starting from qxy.

    Inputs:
    - plx -- a transition matrix
    - qxy -- a transition matrix whose first dimension is the same as plx's second
    - n -- number of steps to take
    - dikinshrink -- a parameter of the mcmc method, greater than 1.  larger value means more accepts and smaller steps.
    - verbose -- if true, prints out updates as it mixes to indicate progress

    Outputs
    - steps -- position at each step in the evolution of the chain

    '''

    # make sure inputs are in a good form
    plx = np.require(plx,dtype=np.float)
    qxy=np.require(qxy,dtype=np.float)
    
    # polytope can be defined by A_eq q = b_eq, q>=0:
    A_eq,b_eq = formulate_polytope(plx,qxy)

    # let's reduce the degrees of freedom to only those that are truly free:
    ker=kernel_and_support(A_eq)[0]
    ndims=ker.shape[0]
    A2=-ker.T
    B2 = qxy.ravel()

    # internally, we represent our state
    # as TRANSLATIONS relative to our inital qxy
    # in the basis indicated by the kernel:
    steps=[np.zeros(ndims)]

    # we compute the Dikin ellipsoid (as represented by Atilde)
    # around our initial point, along with its volume
    # (volume computed up to a constant)
    Atilde = get_dikin_Atilde(A2,B2,steps[-1],shrink=dikinshrink)
    current_logvol=-.5*np.linalg.slogdet(Atilde.T @ Atilde)[1]

    update_interval=1+n//20


    for i in range(n):
        if verbose and i%update_interval==0:  # ersatz tqdm replacement (didn't want to include tqdm as a requirement)
            sys.stdout.write('(%d/%d)'%(i,n))
            sys.stdout.flush()
            
        # sample uniformly inside the dikin ellipsoid at point steps[-1]
        y0=steps[-1]
        y = y0+sample_ellipsoid_lstsq(Atilde)
        
        # get new ellipsoid and volume around point y
        Atilde2 = get_dikin_Atilde(A2,B2,y,shrink=dikinshrink)
        new_logvol = -.5*np.linalg.slogdet(Atilde2.T @ Atilde2)[1]
        
        # check whether steps[-1] is inside the dikin ellipsoid
        # at position y
        magchange = np.linalg.norm(Atilde2 @ (y-y0))
        
        if magchange<1: # if so, we accept with probability based on their relative volume
            accept_probability = np.exp(new_logvol-current_logvol)
            if npr.rand()<=accept_probability: # we move to the new point
                Atilde = Atilde2
                current_logvol=new_logvol
                steps.append(y)
            else:
                steps.append(steps[-1]) # we stay where we are
        else:
            steps.append(steps[-1]) # we stay where we are

    # translate back into the original space
    steps = [(qxy+ (ker.T @ x).reshape(qxy.shape)) for x in steps]
                
    return steps


def get_dikin_Atilde(A,b,x0,shrink=1):
    # get dikin ellipsoid around point x0 in the polytope defined by Ax<=b
    dsts=b-A@x0
    dstmeasure=dsts.min()
    assert dstmeasure>0
    Atilde=A / dsts.reshape((-1,1))
    
    return Atilde*shrink

def sample_gaussian_lstsq(Atilde):
    '''
    Samples gaussian with covariance (Atilde^T Atilde)^-1
    '''
    
    Z=npr.randn(len(Atilde))
    samp = np.linalg.lstsq(Atilde,Z,rcond=None)[0]
    
    return samp

def sample_ellipsoid_lstsq(Atilde):
    '''
    Uniform sample from x such that |Atilde x|^2 <= 1
    '''
    
    # calculate scaling
    scaling=(npr.rand()**(1.0/Atilde.shape[0]))
    
    # calculate gaussian
    samp = sample_gaussian_lstsq(Atilde)
    
    return scaling * samp / np.linalg.norm(Atilde @ samp)


def kernel_and_support(p):
    '''
    Input
    - p, a list of vectors, a basis for some subspace

    Output
    - kernel, an orthonormal basis for the subspace's orthogonal complement
    - support, an orthonormal basis for the subspace
    '''
    U,e,V=np.linalg.svd(p)
    
    support=V[:len(e)]
    kernel=V[len(e):]
    
    return kernel,support

def formulate_polytope(plx,qxy):
    plx = np.require(plx,dtype=np.float)
    qxy=np.require(qxy,dtype=np.float)

    assert len(plx.shape)==2,"plx should be a matrix"
    assert len(qxy.shape)==2,"qxy should be a matrix"
    d1,d2=plx.shape
    d2p,d3=qxy.shape
    assert d2==d2p,'The second dimensions of plx should be the same as the first dimension of qxy'
    assert estimation.check_transition_matrix(plx),"The rows of plx should sum to 1 and all entries should be positive"
    assert estimation.check_transition_matrix(qxy),"The rows of qxy should sum to 1 and all entries should be positive"

    d1,d2 = plx.shape
    d2,d3 = qxy.shape

    # plx @ q = hzy (note we skip the last column of hzy)
    A_eq1 = np.zeros((d1,d3-1,d2,d3))
    for z in range(d1):
        for y in range(d3-1):
            A_eq1[z,y,:,y] = plx[z,:]
    A_eq1 = A_eq1.reshape((d1*(d3-1),d2*d3))
    b_eq1 = (plx @ qxy)[:,:-1].ravel()

    # q is a transition matrix
    A_eq2 = np.zeros((d2,d2,d3))
    b_eq2=np.ones(d2)
    for x in range(d2):
        A_eq2[x,x]=1
    A_eq2=A_eq2.reshape((d2,d2*d3))
    b_eq2=b_eq2.ravel()

    # all eqs
    A_eq = np.concatenate([A_eq1,A_eq2])
    b_eq = np.concatenate([b_eq1,b_eq2])
    
    return A_eq,b_eq
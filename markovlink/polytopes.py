'''
Defining and exploring polytopes involved in the MLM
'''

import scipy as sp
import scipy.optimize
import numpy as np
import numpy.random as npr
import sys
from . import misc

class LinearProgrammingFailure(Exception):
    pass

def precondition_constraint(A,b,tol=1e-10,return_preconditioner=False):
    '''
    Transforms the constraint 

       Ax=b --> Vx=b2

    where U is unitary.  We do this by writing 

        Ax= UEVx = b

    And replacing this with Vx = E^-1 U^T b.  That is, we premultiply both sides by

        E^-1 U^T

    We drop singular values less than tol
    '''

    U,e,V = np.linalg.svd(A)

    e=e[e>tol]

    V=V[:len(e)]

    if return_preconditioner:
        return V, (U.T@b)[:len(e)]/e,np.diag(1/e)@U.T[:len(e)]
    else:
        return V, (U.T@b)[:len(e)]/e




def q_with_fixedph(plx,hly,precondition=True,preconditiontol=1e-10):
    '''
    make explicit the equality constraints 
    that define {q: p @ q = p @ qstar, q@1=1}
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
    b_eq2=-np.ones(szX)
    for x in range(szX):
        A_eq2[x,x]=-1
    A_eq2=A_eq2.reshape((szX,szX*szY))
    b_eq2=b_eq2.ravel()

    # all eqs
    A_eq = np.concatenate([A_eq1,A_eq2])
    b_eq = np.concatenate([b_eq1,b_eq2])
    
    if precondition:
        return precondition_constraint(A_eq,b_eq,tol=preconditiontol)
    else:
        return A_eq,b_eq

def q_with_fixedph_centered(plx,hly,precondition=True,preconditiontol=1e-10):
    '''
    make explicit the equality constraints 
    that define {q,v: p @ q = p @ qstar, q@1=v}
    '''
    plx = np.require(plx,dtype=np.float)
    hly=np.require(hly,dtype=np.float)

    assert len(plx.shape)==2,"plx should be a matrix"
    assert len(hly.shape)==2,"hly should be a matrix"
    szL,szX=plx.shape
    szL,szY=hly.shape
    assert misc.check_transition_matrix(plx),"The rows of plx should sum to 1 and all entries should be positive"
    assert misc.check_transition_matrix(hly),"The rows of hly should sum to 1 and all entries should be positive"

    # plx @ q = hzy 
    A_eq1A = np.zeros((szL,szY,szX,szY))
    for z in range(szL):
        for y in range(szY):
            A_eq1A[z,y,:,y] = -plx[z,:]
    A_eq1B = np.zeros((szL,szY,szL))
    for z in range(szL):
        for y in range(szY):
            A_eq1B[z,y,z] = (1/szY)
    b_eq1 = ((1.0/szY)-hly)

    # q is a transition matrix 
    A_eq2 = np.zeros((szX,szX,szY))
    b_eq2=np.ones(szX)
    for x in range(szX):
        A_eq2[x,x]=1
    b_eq2=b_eq2

    # combine
    A_eq=np.zeros((szL*szY+szX,szX*szY+szL))
    A_eq[:szL*szY,:szX*szY] = A_eq1A.reshape((szL*szY,szX*szY))
    A_eq[:szL*szY,szX*szY:] = A_eq1B.reshape((szL*szY,szL))
    A_eq[szL*szY:,:szX*szY] = A_eq2.reshape((szX,szX*szY))
    b_eq=np.r_[b_eq1.ravel(),b_eq2.ravel()]


    if precondition:
        return precondition_constraint(A_eq,b_eq,tol=preconditiontol)
    else:
        return A_eq,b_eq

def qh_with_fixed_p(p,szY,precondition=True,preconditiontol=1e-10):
    '''
    This function defines the space of values of (q,h) which
    are consistent with the fact that the rows of q sum to 1
    and p@q =h.  It does so by articulating this constraint as

    A @ np.r_[q.ravel(),h.ravel()] = b
    '''

    szL,szX =p.shape

                #tm  pq=h     q      h
    A=np.zeros((szX+szL*szY,szX*szY+szL*szY))
    b=np.zeros((szX+szL*szY))
    b[:szX]=-1

    # transition matrix polytope constraint
    for x in range(szX):
        for y in range(szY):
            A[x,np.ravel_multi_index((x,y),(szX,szY))]=-1
            
    # p@q = h constraint
    # each one is sum_x p_lx q_xy = h_ly
    for l in range(szL):
        for y in range(szY):
            for x in range(szX):
                A[szX+np.ravel_multi_index((l,y),(szL,szY)),np.ravel_multi_index((x,y),(szX,szY))]=p[l,x]
            A[szX+np.ravel_multi_index((l,y),(szL,szY)),szX*szY+np.ravel_multi_index((l,y),(szL,szY))]=-1


    if precondition:
        return precondition_constraint(A,b,tol=preconditiontol)
    else:
        return A,b


##################################
########## EXTREMAL FINDING ######
##################################

def find_extremal(plx,qxy,c,use_cvxopt=True,precondition=False):
    '''
    Consider the polytope of transition matrices q such that plx @ q = plx @ qxy.
    Find the most extremal vertex of this polytope in the direction of c.
    '''

    # make sure inputs are in a good form
    plx = np.require(plx,dtype=np.float)
    qxy=np.require(qxy,dtype=np.float)
    c = np.require(c,dtype=np.float).ravel()
    assert c.shape[0]==np.prod(qxy.shape),"c should be the same shape as qxy"
    
    if qxy.shape[0] <= plx.shape[0]: # <-- trivial polytope
        return qxy

    # polytope can be defined by A_eq q = b_eq, q>=0:
    A_eq,b_eq = q_with_fixedph(plx,plx@qxy,precondition=precondition)

    # find relevant extremal point with interior point method   
    if use_cvxopt:
        return cvxopt_linprog(c,A_eq,b_eq,qxy.ravel()).reshape(qxy.shape)
    else:
        return scipy_linprog(c,A_eq,b_eq).reshape(qxy.shape)


def find_central(plx,qxy,precondition=False,clipthresh=1e-6):
    # make sure inputs are in a good form
    plx = np.require(plx,dtype=np.float)
    qxy=np.require(qxy,dtype=np.float)
    
    # define feasible polytope:
    #       A_eq @ q = b_eq, q>=0
    # where A_eq,b_eq are given by
    A_eq,b_eq = q_with_fixedph(plx,plx@qxy,precondition=precondition)

    # two cases
    if qxy.shape[0]<plx.shape[0]:
        # it may be that q has no degrees of freedom within the polytope
        # the only way that q might still have degrees of freedom
        # is if some of the constraints are secretly redundant
        # we assume that everything is in general position, so that is impossible.  
        # we then obtain that the polytope contains only one feasible point,
        # namely qxy

        return qxy

    else:
        # q has at least one degree of freedom (in terms of the equality constraints)
        qxy=cvxopt_project_to_polytope(A_eq,b_eq,np.zeros(A_eq.shape[1])).reshape(qxy.shape)

        # cvxopt sometimes has minor numerical errors.  in some cases qxy can have negative values.
        # we push it away from 0 to ensure numerical stability.
        qxy=np.clip(qxy,clipthresh,1)
        qxy=qxy/np.sum(qxy,axis=1,keepdims=True)
        return qxy

def project_to_polytope(A,b,y,feasiblepoint=None):

    return cvxopt_quadprog(np.eye(A_eq.shape[1]),A_eq,b_eq).reshape(qxy.shape)

def scipy_linprog(c,A_eq,b_eq):
    rez=sp.optimize.linprog(c,A_eq=A_eq,b_eq=b_eq,method='interior-point',options=dict(sym_pos=True,lstsq=True))
    if rez['success']:
        return rez['x']
    else:
        raise Exception("Linear programming failed!")    

def cvxopt_linprog(c,A_eq,b_eq,initialcondition,gaptol=1e-5,verbose=False):
    '''
    minimize <c, x>
    subj to Ax = b
            x >= 0

    In cvxopt land, this becomes

    minimize <c, x>
    subj to -x + s  = 0
           Ax  = b
            s >= 0

    '''

    try:
        import cvxopt
    except:
        raise Exception("Failed to import cvxopt.  Maybe try with use_cvxopt=False?  That will use scipy instead, which is sometimes a bit slower though.  Or try conda install cvxopt.")

    if verbose:
        p = misc.pnn
    else:
        p= lambda x: x

    G=-np.eye(A_eq.shape[1])
    g_eq = np.zeros(A_eq.shape[1])

    com=lambda x: cvxopt.matrix(x)

    p("cond=%f"%np.linalg.svd(A_eq)[1].min())

    # # first try glpk
    # p("gpkl")
    # rez = cvxopt.solvers.lp(com(c),com(G),com(g_eq),A=com(A_eq),b=com(b_eq),
    #                         primalstart=dict(x=com(initialcondition),s=com(initialcondition)),
    #                        solver='glpk')
    # if rez['status']=='optimal':
    #     p("suc")
    #     return np.array(rez['x'])

    # okay glpk didn't work.  try cvxopt's coneopt
    p("coneopt")
    rez = cvxopt.solvers.lp(com(c),com(G),com(g_eq),A=com(A_eq),b=com(b_eq),
                        primalstart=dict(x=com(initialcondition),s=com(initialcondition)))
    if rez['status']=='optimal':
        p("suc")
        return np.array(rez['x'])

    # okay cvxopt didn't return optimal.  That doesn't mean all is lost.
    # cvxopt has high standards for numerical precision.
    # Let's check the primal and dual infeasibilities
    p("nonopt")
    if rez['gap'] is not None and rez['gap']<gaptol:
        p("suc")
        return np.array(rez['x'])

    # fail.        
    raise LinearProgrammingFailure("Couldn't solve the LP")


    # else: #glpk worked
        


    # # return
    # if rez['x'] is None:
    #     raise LinearProgrammingFailure("Linear programming failed! (%s)"%(rez['status'])) 
    # else:
    #     return np.array(rez['x'])



def cvxopt_project_to_polytope(A,b,y,feasiblepoint=None):
    '''
    Project y to the polytope

    Ax=b
    x>=0
    '''

    '''
    cvxopt wants this in the form

    qp(P, q, G=None, h=None, A=None, b=None)

    minimize    (1/2)*x'*P*x + q'*x 
    subject to  G*x <= h      
                A*x = b.

    In our case we have 

    - G=-I_n
    - h=-0_n
    - A =A
    - b=b
    - P=I_n
    - q = -y

    ''' 

    import cvxopt

    cvxopt.solvers.options['show_progress'] = False

    szM,szN=A.shape
    com=lambda x: cvxopt.matrix(x)
    
    ident=np.eye(szN)
    h=np.zeros((szN,1))
    q=-y.reshape((-1,1))


    if feasiblepoint is not None:
        rez = cvxopt.solvers.qp(com(ident),com(q),com(-ident),com(h),com(A),com(b),
            initvals=dict(x=com(feasiblepoint)))
    else:
        rez = cvxopt.solvers.qp(com(ident),com(q),com(-ident),com(h),com(A),com(b))

    if rez['status']=='optimal':
        return np.array(rez['x'])
    else:
        raise Exception("Quadratic programming failed!")    
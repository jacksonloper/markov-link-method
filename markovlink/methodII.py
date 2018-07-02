import numpy as np
import sys
import scipy as sp
import scipy.special
import itertools
from . import misc
import time
from . import estimation

def train_q_primaldual_initialize(p,Nly,kappa,q):
    szL,szY=Nly.shape

    lam = Nly / (p@q)
    gamma = -np.sum(p.T*(q@lam.T),axis=1) - szY*kappa

    return gamma,lam

def train_q_primaldual_residual(p,Nly,kappa,q,gamma,lam):
    szL,szY=Nly.shape
    szL,szX=p.shape

    return np.r_[(p.T@lam+gamma.reshape((-1,1))).ravel()+kappa/q.ravel(),np.sum(q,axis=1)-1,(lam*(p@q)-Nly).ravel()]

def train_q_primaldual_update(q,gamma,lam,direc,s):
    szL,szY=lam.shape
    szX,szY=q.shape

    q=q+s*direc[:szX*szY].reshape(q.shape)
    gamma=gamma+s*direc[szX*szY:szX*szY+szX]
    lam=lam+s*direc[szX*szY+szX:].reshape(lam.shape)

    return q,gamma,lam

def train_q_primaldual_jacobian(p,Nly,kappa,q,gamma,lam):
    szL,szY=Nly.shape
    szL,szX=p.shape

    def iQ(x,y):
        return np.ravel_multi_index((x,y),q.shape)
    def iG(x):
        return szX*szY+x
    def iL(l,y):
        return szX*szY + szX + np.ravel_multi_index((l,y),Nly.shape)

    jac = np.zeros((szX*szY + szX + szL*szY,szX*szY + szX + szL*szY))

    # rQ
    for x in range(szX):
        for y in range(szY):
            jac[iQ(x,y),iQ(x,y)] = -kappa/q[x,y]**2
            jac[iQ(x,y),iG(x)] = 1
            for l in range(szL):
                jac[iQ(x,y),iL(l,y)] = p[l,y]

    # rG
    for x in range(szX):
        for y in range(szY):
            jac[iG(x),iQ(x,y)]=1

    # rL
    for l in range(szL):
        for y in range(szY):
            jac[iL(l,y),iL(l,y)]=np.sum(p[l]*q[:,y])
            for x in range(szX):
                jac[iL(l,y),iQ(x,y)]=p[l,x]*lam[l,y]

    return jac


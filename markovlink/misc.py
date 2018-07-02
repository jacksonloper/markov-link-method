
import numpy as np
import sys
import scipy as sp

def pnn(x): 
    sys.stdout.write(str(x) + " ")
    sys.stdout.flush()

def check_transition_matrix(a):
    return (np.abs(np.sum(a,axis=1) - 1).max()<1e-9) and (np.min(a)>-1e-9)

def totalvardist(a,b):
    return .5*np.sum(np.abs(a-b))
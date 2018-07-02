import numpy as np
import sys
import scipy as sp
import scipy.special
import itertools
from . import misc
import time

def pseudocount_point_estimate(Nlx,pseudocount=1.0):
    return (Nlx+pseudocount)/np.sum(Nlx+pseudocount,axis=1,keepdims=True)
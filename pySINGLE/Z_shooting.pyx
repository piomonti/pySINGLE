# python shooting algorithm
#
# wihtout separate softThresholding function

import numpy
#import math
cimport numpy as np

from libc.math cimport copysign as CS
from libc.math cimport fabs as fabs

cimport cython
@cython.boundscheck(False)

def Z_shooting(np.ndarray[np.float64_t, ndim=1] B, np.ndarray[np.float64_t, ndim=1] y, np.float64_t l1, np.float64_t l2, np.float64_t tol=.01, np.int_t max_iter=5):
    """Shooting algorithm for Z approximation step
    
    INPUT:
	 - B: intial estimate for coefficients (eg LS estimate)
	 - y: actual time series
	 - l1, l2: penalty terms
    
    OUTPUT:
	 - sparse & smoothened B coefficients
    
    """
    
    cdef int iter_=0
    cdef Py_ssize_t i
    cdef int lenB = len(B)
    cdef int convergence = 0
    cdef np.ndarray[np.float64_t] norm_ = numpy.ones(len(B))*(1+4*l2) #numpy.empty(lenB)
    cdef np.float64_t x
    
    Bold = numpy.copy(B)
    #convergence = False
    #iter_ = 0
    
    #norm_ = numpy.ones(len(B))*(1+4*l2)
    norm_[0] = norm_[-1] = (1+2*l2)
    cdef int n = lenB
    
    while (convergence==0) & (iter_ < max_iter):
        for i in xrange(lenB):
            if i==0:
                x = y[i] + 2.*l2*(  B[i+1] )
                B[i] = CS(1,x) * max(0, fabs(x)-l1)/norm_[i]
            elif i== n-1:
                x = y[i] + 2.*l2*(  B[i-1] )
                B[i] = CS(1,x) * max(0, fabs(x)-l1)/norm_[i]
            else:
                x = y[i] + 2.*l2*( B[i-1] + B[i+1] )
                B[i] = CS(1,x) * max(0, fabs(x)-l1)/norm_[i]
        if numpy.sum(abs(Bold-B))<tol:
            convergence=1
        else:
            Bold = numpy.copy(B)
            iter_ +=1
    return B 
 

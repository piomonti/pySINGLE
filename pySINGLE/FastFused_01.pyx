# ADMM implementation of Fused Lasso

import numpy
cimport numpy as np

from libc.math cimport copysign as CS
from libc.math cimport fabs as fabs
#from libc.math cimport sqrt as sqrt

cimport cython
@cython.boundscheck(False)


cdef inline int evalObjective2( np.ndarray[np.float64_t, ndim=1] resp, np.ndarray[np.float64_t, ndim=1] beta1, np.ndarray[np.float64_t, ndim=1] beta2, np.float64_t l1):
    """Compare objective function values for two different beta vectors. 
    This is to make sure the 
    objective is descreasing - FISTA it NOT a descent algorithm, so it may not always decrease!
    
    We are evaluating the following:
     0.5 * || y - A *beta_i  ||^2 + l1 * ||x||_1 for i=1,2
    
    """
    cdef int n = len(beta1)+1
    cdef np.ndarray[np.float64_t, ndim=1] Us1 = numpy.append( (numpy.cumsum((numpy.linspace(n-1,1,n-1) * beta1/float(n))[::-1]))[::-1], 0)
    cdef np.ndarray[np.float64_t, ndim=1] Ls1 = numpy.append( 0, numpy.cumsum(beta1 * numpy.linspace(1, n-1, n-1))*(1./n) )
    
    cdef np.ndarray[np.float64_t, ndim=1] Us2 = numpy.append( (numpy.cumsum((numpy.linspace(n-1,1,n-1) * beta2/float(n))[::-1]))[::-1], 0)
    cdef np.ndarray[np.float64_t, ndim=1] Ls2 = numpy.append( 0, numpy.cumsum(beta2 * numpy.linspace(1, n-1, n-1))*(1./n) )
    
    cdef float diff_term = 0.5*((resp - Ls1 + Us1)**2).sum() + l1 * abs(beta1).sum() - 0.5*((resp - Ls2 + Us2)**2).sum() - l1 * abs(beta2).sum()
    cdef int I = 0
    
    if diff_term < 0:
        I = 1
    
    return I


def getGrad( np.ndarray[np.float64_t, ndim=1] x):
    """O(n) calculation of the gradient
    
    Define it once to be used in function Pmethod - in main function we simply include the code
    
    """
    
    cdef int n= len(x) + 1
    
    cdef np.ndarray[np.float64_t, ndim=1] UT = numpy.append( numpy.cumsum( (numpy.linspace(n-2,1, n-2)*(1./n) * x[1:])[::-1])[::-1] * numpy.linspace(1, n-2, n-2), 0) # upper tri sum
    cdef np.ndarray[np.float64_t, ndim=1] LT = numpy.append(0, numpy.cumsum( numpy.linspace(1,n-2,n-2)*(1./n) * x[:-1]) * numpy.linspace(n-2,1,n-2) )
    cdef np.ndarray[np.float64_t, ndim=1] DT = numpy.linspace(n-1, 1, n-1) * (1./n) * numpy.linspace(1,n-1,n-1) * x
    
    return UT+LT+DT    
    
cdef inline float Pmethod( int n, int miter=100, float tol=0.01):
    """Calculate largest eigenvalue using power method
    
    See http://en.wikipedia.org/wiki/Power_iteration for details
    
    INPUT:
	 - n: length of response vector
    
    OUTPUT:
        - l: largest evalue
    """
    
    if n%2==0:
        # n is even:
        b = numpy.concatenate(( numpy.linspace(1, n/2, n/2)/n, (numpy.linspace(1, n/2, n/2)/n)[::-1])) * -1 # initial guess
    else:
        b = numpy.concatenate(( numpy.linspace(1, n/2, n/2)/n, numpy.array([0.5]),  (numpy.linspace(1, n/2, n/2)/n)[::-1])) * -1 # initial guess

    cdef np.ndarray[np.float64_t, ndim=1] bold = numpy.copy(b) # for convergence
    cdef int conv = 0
    cdef int iter_ = 0
    while( (conv==0) & (iter_ < miter)):
        b = getGrad(b)
        b /= ((sum(b**2))**(0.5))
        if (sum((b-bold)**2) < tol):
            conv = True
        else:
            iter_ += 1
            bold = numpy.copy(b)
    #print "Converged in " + str(iter_) + " iterations"
    return (sum(getGrad(b)**2))**(0.5)
    
   
def SoftThres(float x, float l1):
    """softthresholding function"""
    return CS(1,x) * max(0, fabs(x)-l1)
    
def SoftThresVec(np.ndarray[np.float64_t, ndim=1] x, float l1 ):
    cdef Py_ssize_t i
    cdef int lenX = len(x)
    for i in xrange(lenX):
        x[i] = CS(1,x[i]) * max(0, fabs(x[i])-l1)
    return x

def fastFLSAc( np.ndarray[np.float64_t, ndim=1] resp, np.float64_t l1, np.float64_t l2, np.int_t miter=5000, np.float64_t tol=1e-3):
    """
    Fast Fused Lasso Signal Approximator (FLSA)
    This is fast in the sense that each iteration costs O(n) and that it is an accelerated algorithm - convergence is O(k^-2)
    
    Approach taken here is to reformulate as a FISTA problem. The exploit the nature of this problem to obtain cheap iterations
    
    INPUT:
	 - resp: response vector
	 - l1: sparsity penalty
	 - l2: smoothness penalty
	 - miter: maximum number of iterations before convergence is declared
	 - tol: convergence tolerance
    
    OUTPUT:
	 - beta: solution to FLSA 
	 - iter_: iteration count
	 
    see pure python code for improved readability
    """
    
    # define variables:
    cdef int n = len(resp)
    cdef float respBar = numpy.mean(resp) # will be needed later
    resp -= respBar
    
    # get A^ty vector (this only needs to be done once):
    cdef np.ndarray[np.float64_t, ndim=1] Aresp = (numpy.cumsum( resp[1:][::-1] ))[::-1] * numpy.linspace(1, n-1, n-1)/float(n) - numpy.cumsum(resp[:-1])*numpy.linspace(n-1,1,n-1)/float(n)
    
    # use power method to find largest evalue (again only need to do this once)
    cdef float lam_stepsize = 1./Pmethod(n-1) 
    
    cdef np.ndarray[np.float64_t, ndim=1] beta =  numpy.ones(n-1) # arbitrary intial guess
    cdef np.ndarray[np.float64_t, ndim=1] betaOld = numpy.ones(n-1) # for convergence checking
    cdef np.ndarray[np.float64_t, ndim=1] y_iter = numpy.ones(n-1) # for explolation step
    
    cdef np.ndarray[np.float64_t, ndim=1] grad = numpy.ones(n-1) # gradient vector
    cdef np.ndarray[np.float64_t, ndim=1] u = numpy.ones(n-1) # additional vector used to check objective is decreasing (as FISTA is not a descent algorithm)
    
    # some extra variables which I will use later:
    cdef float coef1 = 0.0
    cdef np.ndarray[np.float64_t, ndim=1] betaOrig = numpy.ones(n)
    cdef np.ndarray[np.float64_t, ndim=1] UT = numpy.zeros(n)
    cdef np.ndarray[np.float64_t, ndim=1] LT = numpy.zeros(n)
    cdef np.ndarray[np.float64_t, ndim=1] DT = numpy.zeros(n)
    cdef Py_ssize_t i # used for softthresholding
    cdef double aux = 0.0
    
    cdef int iter_= 0
    cdef int convergence = 0 
    
    cdef np.float w = 0.0
    
    # ok so now we're ready to FISTA:
    
    while( (convergence==0) & (iter_ < miter)):
        w = iter_/(iter_ + 3.)
        
        y_iter = beta + w*(beta - betaOld)
        betaOld = beta.copy() # this might break - check!
        
        # calculate gradient:
        UT = numpy.append( numpy.cumsum( (numpy.linspace(n-2,1, n-2)*(1./n) * y_iter[1:])[::-1])[::-1] * numpy.linspace(1, n-2, n-2), 0) # upper tri sum
        LT = numpy.append(0, numpy.cumsum( numpy.linspace(1,n-2,n-2)*(1./n) * y_iter[:-1]) * numpy.linspace(n-2,1,n-2) )
        DT = numpy.linspace(n-1, 1, n-1) * (1./n) * numpy.linspace(1,n-1,n-1) * y_iter
        
        grad = UT+LT+DT #getGrad(y_iter) # O(n) calculation of gradient step
        for i in xrange(n-1):
            aux = y_iter[i] - lam_stepsize * (grad[i] - Aresp[i]) # store auxiliary variable to make next line more readable
            u[i] = CS(1,aux) * max(0, fabs(aux)-lam_stepsize*l2)   
        
        #u = SoftThresVec( y_iter - lam_stepsize * (grad - Aresp), lam_stepsize*l2)

        # add descending check! FISTA is not a descent algorithm!
        if evalObjective2(resp=resp, beta1=u, beta2=betaOld, l1=l2)==0:  #evalObjective(resp=resp, beta=u, l1=l2) > evalObjective(resp=resp, beta=betaOld, l1=l2):
            beta = betaOld.copy() # to ensure always reducing
            #print iter_
        else:
            beta = u.copy()
        
        if ( ((u-betaOld)**2).sum() < tol):
            convergence = 1
        else:
            iter_ += 1
            #print iter_
        
    #print "Converged in " + str(iter_) + " iterations"
    # clean up and change basis one last time:
    
    # following variables have not been defined - this will slow down performance, but not by much (I hope)
    coef1 = respBar - numpy.dot(numpy.linspace(n-1,1,n-1)/float(n), beta)
    beta = numpy.append(coef1, beta)
    
    # transform back to original basis:
    betaOrig = numpy.cumsum(beta)
    
    betaOrig = SoftThresVec(betaOrig, l1) # soft threshold for sparsity (l1) penalty
    return betaOrig
    
    
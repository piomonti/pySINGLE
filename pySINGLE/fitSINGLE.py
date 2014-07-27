## For given values of lambda1 and lambda2 we minimise the SINGLE objective function and calculate AIC ##


#import pandas
import math
import numpy
import os
import spams
from scipy.linalg import solveh_banded
import multiprocessing
from operator import add, sub
import itertools
from FastFused_01 import *
    
def fitSINGLE(S, data, l1, l2, parallel=0, obs=1, rho=1., max_iter=500, tol=0.001):
    """Solves SIGL for given covariance estimates S
    Input:
	- S = covariance matricies. S is a list where S[i] is the estimate of the covariance at time i
	- data: data array, used to fit AIC
	- l1 and l2 are the penalties in the SINGLE cost function
	- parallel: run in parallel (multiprocessing), 1 indicates multiprocessing, 0 indicates serial processing
	- Approx: use approximation
    Output:
	- Z = a list where Z[i] is an esitmate of the precision at time i"""
    
    # define auxiliary variables:
    theta = [numpy.identity(S[0].shape[0])] * len(S)
    Z = [numpy.zeros((S[0].shape[0], S[0].shape[0]))] * len(S)
    U = Z[:]
    Zold = Z[:] # used to confirm convergence
    convergence = False
    iter_ = 0
    if isinstance(obs, int):
	obs = [1.]*len(S)
    
    # run ADMM algorithm:
    if parallel==1:
	# initialise pool - only do this once!
	pool_size = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(processes=pool_size)
    
    while (convergence==False) & (iter_ < max_iter):
	# Theta update:
	A = [0.]*len(S)
	
	if parallel==1:
	    inputs = [S[i] - (rho/obs[i]) *Z[i] + (rho/obs[i])*U[i] for i in range(len(S))]
	    theta = pool.map(minimize_theta, inputs)
	    A = map(add, theta, U) # used in Z update step
	else:
	    for i in range(len(S)):
		theta[i] = minimize_theta(S_ = S[i] - (rho/obs[i]) *Z[i] + (rho/obs[i])*U[i], rho=rho, obs=obs[i] )
		A[i] = theta[i] + U[i] # will be used in Z update step
	# Z update:
	if parallel==1:
	    # prepare input:
	    # convert A into an array:
	    p = A[0].shape[0]
	    A_ = numpy.zeros((len(A), A[0].shape[0], A[0].shape[0]))
	    for i in range(len(A)):
		A_[i,:,:] = A[i]
	    
	    Zinputs = [A_[:,i,j] for i in range(p) for j in range(i,p)]
	    sudoZ = pool.map(Z_parallel_helper, itertools.izip(Zinputs, itertools.repeat(l1), itertools.repeat(l2)))
	    
	    # now I need to put it back together!
	    Zarray = numpy.zeros((len(A), A[0].shape[0], A[0].shape[0]))
	    counter = 0
	    for i in range(p):
		for j in range(i,p):
		    Zarray[:,i,j] = sudoZ[counter]
		    Zarray[:,j,i] = sudoZ[counter]
		    counter += 1
	    
	    # put back into a list:
	    Z = [None] * len(A)
	    for i in range(len(A)):
		Z[i] = Zarray[i,:,:]
	else:
	    # serial implementation:
	    A_ = numpy.zeros((len(A), A[0].shape[0], A[0].shape[0]))
	    for i in range(len(A)):
		A_[i,:,:] = A[i]
	    sudoZ = A_[:]
	    for i in range(A[0].shape[0]):
		for j in range(i, A[0].shape[0]):
		    resp = A_[:,i,j]
		    beta_hat = fastFLSAc(resp=resp, l1=l1, l2=l2, miter=1000, tol=1e-6)
		    sudoZ[:,i,j] = beta_hat
		    sudoZ[:,j,i] = beta_hat
	    #Z = [None] * len(A)
	    for i in range(len(A)):
		Z[i] = sudoZ[i,:,:]
	
	# U update:
	for i in range(len(S)):
	    U[i] = U[i] + theta[i] - Z[i]
	    
	# check for convergence
	convergence = check_conv(theta, Z, Zold, tol=tol)
	iter_ += 1
	Zold = Z[:]
	print iter_
	#if convergence:
	#    print 'CONVERGED HURRAH'
    
    # clean up:
    if parallel==1:
	pool.close()
	pool.join()
    
    # calculate AIC
    ncol_ = data.shape[1]
    theta = numpy.array(theta).reshape(data.shape[0], ncol_, ncol_)
    Z = numpy.array(Z).reshape(data.shape[0], ncol_, ncol_) # conver to array to make plotting etc easier
    AIC = get_aic(data, P=theta)
	
    return Z, convergence, iter_, AIC


def minimize_theta(S_, rho=1, obs=1):
    """1st step: Minimize theta step of the ADMM algorithm for solving SIGL
    input:
	- S_ = S_i - rho/obs * Z_i + rho/obs * U_i where S_i is ith entry of S (our list of covariance estimates)
    output:
	- new update of theta_i"""
    D, V = numpy.linalg.eig(S_)
    
    D_ = numpy.identity(len(D)) * [obs/(2. * rho) * (-x + math.sqrt(x*x + 4.*rho/obs)) for x in D]

    return numpy.dot(numpy.dot(V, D_), V.T)

def minimize_Z_parallel(resp, l1, l2):
    """Parallelisable implementation of Z step
    
    INPUT:
	  - resp: vector for (i,j) partial correlation estimates 
	  - l1, l2, rho: penalty parameters
    
    Note that FastFLSAc is the cython implementation of O(n) FLSA algorithm
    
    """
    
    return fastFLSAc(resp = resp, l1=l1, l2=l2, miter=1000, tol=1e-6)    
    
    
# auxiliary funciton to make it work
def Z_parallel_helper(args):
    return minimize_Z_parallel(*args)
    
def check_conv(theta, Z, Zold, tol):
    """Check convergence of the ADMM algorithm"""
    cond1 = True
    cond2 = True
    for i in range(0, len(theta)):
	if ( ((abs(theta[i]-Z[i]))**2).sum() >= tol):
	    cond1 = False
	if ( ((abs(Z[i] - Zold[i]))**2).sum() >= tol):
	    cond2 = False
    return cond1 & cond2

def LLaic(data, P):
    """Given array of estimated precisions, P, and data we estimate the log likelihood (assuming multivariate Gaussian"""
    
    LL = 0
    for i in range(data.shape[0]):
	LL += -0.5 * (math.log(numpy.linalg.det(P[i,:,:])) + (numpy.matrix(data[i,:])*numpy.matrix(P[i,:,:]).I*numpy.matrix(data[i,:]).T)[0,0])
	
    return LL
    
def get_aic(data, P):
    """Calculate the AIC of a given array of precision matrices
    
    Input:
	 - data: array of observations
	 - P: array of precision matrices
    
    """
    
    ncol_  = data.shape[1]
    param_no = 0
    for i in range(ncol_):
	for j in range(i+1, ncol_):
	    param_no += len(numpy.unique(P[:,i,j]))

    LL_est = LLaic(data, P=P)
    return 2*(param_no - LL_est)
    
    
    

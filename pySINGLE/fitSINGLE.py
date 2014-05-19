## For given values of lambda1 and lambda2 we minimise the SINGLE objective function and calculate AIC ##


import pandas
import math
import numpy
import os
import spams
from scipy.linalg import solveh_banded
import multiprocessing
from operator import add, sub
import itertools
import Z_shooting

    
def fitSINGLE(S, data, l1, l2, pen_type=1, parallel=0, Approx=False, obs=1, rho=1., max_iter=50, tol=0.001, C=1):
    """Solves SIGL for given covariance estimates S
    Input:
	- S = covariance matricies. S is a list where S[i] is the estimate of the covariance at time i
	- data: data array, used to fit AIC
	- l1 and l2 are the penalties in the SINGLE cost function
	- parallel: run in parallel (multiprocessing)
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
	    #print "Theta step in parallel!"
	    inputs = [S[i] - (rho/obs[i]) *Z[i] + (rho/obs[i])*U[i] for i in range(len(S))]
	
	    #pool_size = multiprocessing.cpu_count() - 1
	    #pool = multiprocessing.Pool(processes=pool_size, maxtasksperchild=5)
	    theta = pool.map(minimize_theta, inputs)
	    
	    #pool.close()
	    #pool.join()
	    
	    #jobs = []
	    #p = Process(target=minimize_theta, args=(S[i] - (rho/obs[i]) *Z[i] + (rho/obs[i])*U[i],))
	    #jobs.append(p)
	    #p.start()
	
	    #xr = range(len(S))
	    #pool = Pool()
	    #theta = pool.map(minimize_theta_parallel, xr)
	    
	    A = map(add, theta, U) # used in Z update step
	    
	    #for i in range(len(S)):
		#A[i] = theta[i] + U[i]
	    
	else:
	    for i in range(len(S)):
		theta[i] = minimize_theta(S_ = S[i] - (rho/obs[i]) *Z[i] + (rho/obs[i])*U[i], rho=rho, obs=obs[i] )
		A[i] = theta[i] + U[i] # will be used in Z update step
	# Z update:
	if pen_type==1:
	    Z = minimize_Z_fused(A, l1=l1, l2=l2, rho=rho)
	else:
	    if parallel==1:
		# prepare input:
		p = A[0].shape[0]
		# prepare tridogonal problem which we will solve effciently using solveh_banded:
		n = len(A)
		Bmat = numpy.zeros((2,n))
		Bmat[0,:] = -2.*l2/rho
		Bmat[1,1:n-1] = 1 +4.*l2/rho
		Bmat[1,0] = Bmat[1,n-1] = 1 + 2.*l2/rho
		
		# convert A into an array:
		A_ = numpy.zeros((len(A), A[0].shape[0], A[0].shape[0]))
		for i in range(len(A)):
		    A_[i,:,:] = A[i]
		
		Zinputs = [A_[:,i,j] for i in range(p) for j in range(i,p)]
		#sudoZ = pool.map(Z_parallel_helper_Approx, itertools.izip(Zinputs, itertools.repeat(Bmat), itertools.repeat(l1)))
		if Approx:
		    sudoZ = pool.map(Z_parallel_helper_Approx, itertools.izip(Zinputs, itertools.repeat(Bmat), itertools.repeat(l1)))
		else:
		    if C==0:
			sudoZ = pool.map(Z_parallel_helper, itertools.izip(Zinputs, itertools.repeat(Bmat), itertools.repeat(l1), itertools.repeat(l2))) # cython free implementation
		    else:
			sudoZ = pool.map(Z_parallel_helper_cython, itertools.izip(Zinputs, itertools.repeat(Bmat), itertools.repeat(l1), itertools.repeat(l2))) # cython implementation
			#print "cython multiprocessing!!"
		
		#sudoZ = pool.map(Z_parallel_helper, itertools.izip(Zinputs, itertools.repeat(Bmat), itertools.repeat(l1), itertools.repeat(l2)))
		#print "Z step succesfully done in parallel MUTHAFUKA!!"
		
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
		if C==0:
		    Z = minimize_Z_EL2(A, l1=l1, l2=l2, rho=rho) # run cython free implementation
		else:
		    Z = minimize_Z_EL_cython(A, l1=l1, l2=l2, rho=rho) # run cython implementation
	
	# U update:
	for i in range(len(S)):
	    U[i] = U[i] + theta[i] - Z[i]
	    
	# check for convergence
	convergence = check_conv(theta, Z, Zold, tol=tol)
	iter_ += 1
	Zold = Z[:]
	#print iter_
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
	
	
def minimize_Z_fused(A, l1, l2, rho):
    """2nd step: Minimize Z step of the ADMM algorithm for solving SIGL
    input:
	- A is a list such that A[i] = theta[i] + U[i]
    outout:
	- new update of Z (ie a list)"""
    
    # some definitions:
    param = {'numThreads' : -1, # number of processors/cores to use (-1 => all cores)
         'pos' : False,
         'mode': 5, # FLSA
         'lambda1' : l1,
         'lambda2' : l2,
         'lambda3' : 0}
	
    # convert A into an array:
    A_ = numpy.zeros((len(A), A[0].shape[0], A[0].shape[0]))
    for i in range(len(A)):
	A_[i,:,:] = A[i]
    
    sudoZ = A_[:]
    for i in range(A[0].shape[0]):
	for j in range(i, A[0].shape[0]):
	    resp = A_[:,i,j]
	    resp.shape = (len(resp),1)
	    resp = numpy.asfortranarray(resp)
	    beta_hat = spams.sparseProject(resp, **param)
	    beta_hat.shape = (len(beta_hat),)
	    sudoZ[:,i,j] = beta_hat
	    sudoZ[:,j,i] = beta_hat

    # return to a list (terribly inefficient! I have to change this!)
    Z_ = [None] * len(A)
    for i in range(len(A)):
	Z_[i] = sudoZ[i,:,:]
	
    return Z_
	

	
def minimize_Z_EL(A, l1, l2, rho):
    """2nd step: Minimize Z step of the ADMM algorithm for solving SIGL
    input:
	- A is a list such that A[i] = thetaZ_parallel_helper_Approx[i] + U[i]
    outout:
	- new update of Z (ie a list)"""
    
    # build banded matrix:
    n = len(A)
    Bmat = numpy.zeros((2,n))
    Bmat[0,:] = -2*l2/rho
    Bmat[1,1:n-1] = 1 +4*l2/rho
    Bmat[1,0] = Bmat[1,n-1] = 1 + 2*l2/rho
    
    # convert A into an array:
    A_ = numpy.zeros((len(A), A[0].shape[0], A[0].shape[0]))
    for i in range(len(A)):
	A_[i,:,:] = A[i]
    
    sudoZ = A_[:]
    for i in range(A[0].shape[0]):
	for j in range(i, A[0].shape[0]):
	    resp = A_[:,i,j]
	    beta_hat = solveh_banded(Bmat, resp, overwrite_ab=True, overwrite_b=True)
	    
	    # apply soft thresholding:
	    beta_hat = [math.copysign(1,x) * max(0, abs(x)-l1) for x in beta_hat]
	    
	    sudoZ[:,i,j] = beta_hat
	    sudoZ[:,j,i] = beta_hat

    # return to a list (terribly inefficient! I have to change this!)
    Z_ = [None] * len(A)
    for i in range(len(A)):
	Z_[i] = sudoZ[i,:,:]
	
    return Z_	
    
def minimize_Z_EL_parallel(resp, Bmat, l1, l2):
    """Parallel implementation of Z_EL step
    
    INPUT:
	  - resp: vector for (i,j) partial correlation estimates 
	  - Bmat: tridiagonal matrix used to solve L2 problem
	  - l1, l2, rho: penalty parameters
    
    """
    
    beta_hat = solveh_banded(Bmat, resp, overwrite_ab=True, overwrite_b=True)
    beta_hat = Z_shooting_old(B=beta_hat, y=resp, l1=l1, l2=l2, tol=.01, max_iter=100)
    
    return beta_hat
    
# auxiliary funciton to make it work
def Z_parallel_helper(args):
    return minimize_Z_EL_parallel(*args)

    
def minimize_Z_EL_parallel_cython(resp, Bmat, l1, l2):
    """Parallel implementation of Z_EL step
    
    INPUT:
	  - resp: vector for (i,j) partial correlation estimates 
	  - Bmat: tridiagonal matrix used to solve L2 problem
	  - l1, l2, rho: penalty parameters
    
    """
    
    beta_hat = solveh_banded(Bmat, resp, overwrite_ab=True, overwrite_b=True)
    beta_hat = Z_shooting.Z_shooting(B=beta_hat, y=resp, l1=l1, l2=l2, tol=.1, max_iter=100)
    
    return beta_hat

def Z_parallel_helper_cython(args):
    return minimize_Z_EL_parallel_cython(*args)

def minimize_Z_EL_parallel_Approx(resp, Bmat, l1):
    """Parallel implementation of Z_EL step
    
    INPUT:
	  - resp: vector for (i,j) partial correlation estimates 
	  - Bmat: tridiagonal matrix used to solve L2 problem
	  - l1, l2, rho: penalty parameters
    
    """
    
    beta_hat = stVec(solveh_banded(Bmat, resp, overwrite_ab=True, overwrite_b=True), l1)
    
    return beta_hat    
    
def Z_parallel_helper_Approx(args):
    return minimize_Z_EL_parallel_Approx(*args)
    
def minimize_Z_EL2(A, l1, l2, rho):
    """"""
    
    # build banded matrix:
    n = len(A)
    Bmat = numpy.zeros((2,n))
    Bmat[0,:] = -2*l2/rho
    Bmat[1,1:n-1] = 1 +4*l2/rho
    Bmat[1,0] = Bmat[1,n-1] = 1 + 2*l2/rho
    
    # convert A into an array:
    A_ = numpy.zeros((len(A), A[0].shape[0], A[0].shape[0]))
    for i in range(len(A)):
	A_[i,:,:] = A[i]
    
    sudoZ = A_[:]
    for i in range(A[0].shape[0]):
	for j in range(i, A[0].shape[0]):
	    resp = A_[:,i,j]
	    # get LS solution:
	    beta_hat = solveh_banded(Bmat, resp, overwrite_ab=True, overwrite_b=True)
	    
	    # shooting algorithm:
	    beta_hat = Z_shooting_old(B=beta_hat, y=resp, l1=l1, l2=l2, tol=0.01, max_iter=200)
	    
	    sudoZ[:,i,j] = beta_hat
	    sudoZ[:,j,i] = beta_hat

    # return to a list (terribly inefficient! I have to change this!)
    Z_ = [None] * len(A)
    for i in range(len(A)):
	Z_[i] = sudoZ[i,:,:]
	
    return Z_	

    
    
def minimize_Z_EL_cython(A, l1, l2, rho):
    """"""
    
    # build banded matrix:
    n = len(A)
    Bmat = numpy.zeros((2,n))
    Bmat[0,:] = -2*l2/rho
    Bmat[1,1:n-1] = 1 +4*l2/rho
    Bmat[1,0] = Bmat[1,n-1] = 1 + 2*l2/rho
    
    # convert A into an array:
    A_ = numpy.zeros((len(A), A[0].shape[0], A[0].shape[0]))
    for i in range(len(A)):
	A_[i,:,:] = A[i]
    
    sudoZ = A_[:]
    for i in range(A[0].shape[0]):
	for j in range(i, A[0].shape[0]):
	    resp = A_[:,i,j]
	    # get LS solution:
	    beta_hat = solveh_banded(Bmat, resp, overwrite_ab=True, overwrite_b=True)
	    
	    # shooting algorithm:
	    beta_hat = Z_shooting.Z_shooting(B=beta_hat, y=resp, l1=l1, l2=l2, tol=0.1, max_iter=100)
	    
	    sudoZ[:,i,j] = beta_hat
	    sudoZ[:,j,i] = beta_hat

    # return to a list (terribly inefficient! I have to change this!)
    Z_ = [None] * len(A)
    for i in range(len(A)):
	Z_[i] = sudoZ[i,:,:]
	
    return Z_
    
def softThres(x, l1):
    """Soft thresholding function"""
    
    return math.copysign(1,x) * max(0, abs(x)-l1)

stVec = numpy.vectorize(softThres)    
    
    
def Z_shooting_old(B, y, l1, l2, tol=.01, max_iter=5):
    """Shooting algorithm for Z approximation step
    
    INPUT:
	 - B: intial estimate for coefficients (eg LS estimate)
	 - y: actual time series
	 - l1, l2: penalty terms
    
    OUTPUT:
	 - sparse & smoothened B coefficients
    
    """
    
    Bold = numpy.copy(B)
    convergence = False
    iter_ = 0
    
    norm_ = numpy.ones(len(B))*(1+4*l2)
    norm_[0] = norm_[-1] = (1+2*l2)
    n = len(B)
    
    while (convergence==False) & (iter_ < max_iter):
	# need to iterate through each entry:
	for i in range(len(B)):
	    if i==0:
		B[i] = softThres( y[i] + 2.*l2*(  B[i+1] ), l1)/norm_[i]
	    elif i== n-1:
		B[i] = softThres( y[i] + 2.*l2*(  B[i-1] ), l1)/norm_[i]
	    else:
		B[i] = softThres( y[i] + 2.*l2*( B[i-1] + B[i+1] ), l1)/norm_[i]
    
	#Bmin1 = numpy.insert(B[:-1], 0,0)
	#Bplus1 = numpy.insert(B[1:], len(B)-1,0)
	
	#B = stVec(y+2*l2*(Bmin1+Bplus1), l1)/norm_
	
	#B = [math.copysign(1,x) * max(0, abs(x)-l1) for x in y+2*l2*(Bmin1+Bplus1)]
	
	if numpy.sum(abs(Bold-B))<tol:
	    convergence=True
	else:
	    Bold = numpy.copy(B)
	    iter_ +=1
	    #print iter_
    
    return B
    
    
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
    
    
    

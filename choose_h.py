## Kernel Smoothing estimates of mean and covariance ##

# Note that it is assumed data is temporally arranged.

import pandas
import math
import numpy
import multiprocessing
from operator import add, sub
from itertools import repeat

def choose_h(data, rad_list, samples=None, parallel=0):
    """function to choose best performing value of h based on LOO cross validation
    
    Input:
	 - data: data for which to estimate radius
	 - rad_list: a list of values of values to try for the Gaussian kernel radius
	 - samples: Index of time points over which to cross validate. Can be empty, if provided will only cross validate over these indices
    
    """
    
    if samples==None:
	samples = range(1,data.shape[0])
    
    for x in range(data.shape[1]):
	if x in samples:
	    samples.pop(samples.index(x))
	if data.shape[1]-x in samples:
	    samples.pop(samples.index(data.shape[1]-x))

    if parallel==1:
	pool_size = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(processes=pool_size)
	results = pool.map(CV_LL_parallel, zip(repeat(data), rad_list, repeat(samples)))
	pool.close()
	pool.join()
	
    else:
	results = [ CV_LL(data, radius=x, samples=samples) for x in rad_list]
    
    return rad_list[results.index(max(results))]
    

def CV_LL(data, radius, samples=-1):
    """Cross-validate likelihood for a given radius
    
    samples can be used to provide a subset of the data (index over time) in order to reduce computational cost
    e.g., samples = range(50, 100)
    If samples is not provided, all observations will be used - this can be computationally expensive"""
    
    if samples==-1:
	samples = range(data.shape[0])

    mean_ = get_kern_mean(data, radius=radius) # needed to center data in order to estimate covariance
    CV_log_lik = [ estLL(data, ID=x, radius=radius, mean_=mean_) for x in samples]
    
    return sum(CV_log_lik)
    
def CV_LL_parallel((data, radius, samples)):
    """Cross-validate likelihood for a given radius - code has been amended slighty to allow for parallel implementation (just added some parentisis)
    
    samples can be used to provide a subset of the data (index over time) in order to reduce computational cost
    e.g., samples = range(50, 100)
    If samples is not provided, all observations will be used - this can be computationally expensive"""
    
    if samples==-1:
	samples = range(data.shape[0])

    mean_ = get_kern_mean(data, radius=radius) # needed to center data in order to estimate covariance
    CV_log_lik = [ estLL(data, ID=x, radius=radius, mean_=mean_) for x in samples]
    
    return sum(CV_log_lik)    
    
def estLL(data, ID, radius, mean_):
    """Estimate Log likelihood at a given index (ID) for a given radius"""
    
    mu_est = get_kern_mean_ind(data, ID=ID, radius=radius, CV=True)
    cov_est = get_kern_cov_ind(data, ID=ID, mean_=mean_, radius=radius, CV=True)
    
    log_lik = math.log(numpy.linalg.det(cov_est)) + (numpy.matrix(data[ID,:]-mu_est)*(numpy.matrix(cov_est).I)*numpy.matrix(data[ID,:]-mu_est).T)[0,0]

    return -0.5*log_lik
    
    
    
def gaussian_kern(a,b,radius=1):
    return math.exp(-(a-b)*(a-b)/(2.*radius*radius))

    
def get_kern_mean_ind(data, ID, radius=1, CV=False):
    """Add details"""
    
    ncol_ = data.shape[1]
    nrow_ = data.shape[0]
    
    norm_ = numpy.array([gaussian_kern(x, ID, radius) for x in range(1, nrow_+1)]).reshape(-1,1)
    if CV: norm_[ID-1] = 0
   
    return sum(norm_ * data)/sum(norm_)  

    
def get_kern_mean(data, radius=1, CV=False):
    """Calculates mean based on Gaussian kernel for each datapoint
    Input:
	- data is a numpy array
	- radius is the radius of the Gaussian kernel
    Output:
	- kern_mean is a numpy array"""
    
    kern_mean = numpy.zeros((data.shape[0], data.shape[1]))
    
    for i in range(0, data.shape[0]):
	kern_mean[i,:] = get_kern_mean_ind(data, ID=i+1, radius=radius, CV=CV)
	
    return kern_mean


def sqr(x):
    return x*x
    
def get_kern_cov_ind2(data, ID, mean_, radius, CV=False):
    """this is actually slower, so we dont use this function - see below for the function"""
    
    ncol_ = data.shape[1]
    nrow_ = data.shape[0]
    
    # remove mean:
    data = data - mean_
    
    # get norm:
    norm_ = numpy.array([math.sqrt(gaussian_kern(x, ID, radius)) for x in range(1, nrow_+1)]).reshape(-1,1) # need to take sqrt as will be taking inner product (ie multiplying twice)
    if CV: norm_[ID-1] = 0
    
    data = norm_ * data 
    
    cov_ = numpy.array([numpy.outer(data[i,:], data[i,:]).reshape(-1,1) for i in range(data.shape[0])]).reshape(nrow_, ncol_*ncol_)
    
    cov_ = numpy.sum(cov_, axis = 0).reshape(ncol_, ncol_)/sum(map(sqr, norm_))
    
    return cov_
    
    
def get_kern_cov_ind(data, ID, mean_, radius, CV=False):
    """calcuate covariance for a given datapoint based on Gaussian kernel
    Input:
	- data - a numpy array
	- ID is used to identify point of interested
	- mean_ is the mean estimated using the same Gaussian kernel
	- radius is the radius of the Gaussian kernel"""
    
    cov_ = numpy.zeros((data.shape[1], data.shape[1]))
    norm_ = 0

    for i in range(0, data.shape[0]):
	cov_ = cov_ + gaussian_kern(ID,i+1,radius=radius) * numpy.outer(data[i,:]-mean_[i,:], data[i,:]-mean_[i,:])
	norm_ = norm_ + gaussian_kern(ID,i+1, radius=radius)
	
    if CV:
	cov_ = cov_ - numpy.outer(data[ID-1,:]-mean_[ID-1,:], data[ID-1,:]-mean_[ID-1,:])
	norm_ -= 1
    return cov_/norm_
	
	
def get_kern_cov(data, radius=1, CV=False):
    """Calculates covariances based on Gaussian kernel for each datapoint
    Input:
	- data is a numpy array
	- radius is the radius of the Gaussian kernel
    Output:
	- kern_cov is a numpy array"""
    
    kern_cov = numpy.zeros((data.shape[0], data.shape[1], data.shape[1]))
    mean_ = get_kern_mean(data, radius=radius, CV=CV)
    for i in range(0, data.shape[0]):
	kern_cov[i,:,:] = get_kern_cov_ind(data, ID=i, mean_=mean_, radius=radius, CV=CV)
    
    return kern_cov
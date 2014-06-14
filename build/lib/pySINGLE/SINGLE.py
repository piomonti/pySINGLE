## Python Implementation of Smooth Incremental Graphical Lasso Estimation (SINGLE) algorithm ## 
#
#
#

from choose_h import *
from fitSINGLE import *
from plotSINGLE import plotSINGLE

import numpy

class SINGLE():
    '''Class for SINGLE objects
    
    INPUT:
        - data: numpy array of multivariate data
        - h: radius of Gaussian kernel - can be empty in which case it will be estimated using cross validation
        - pen_type: for smoothness/temporal homogeneity penalty. "Fused"=Fused Lasso, "Elastic"=Elastic Fused Lasso (squared difference). Use pen_type="Elastic" for faster computational speed
        - parallel: boolean indicating whether to run code in parallel. 
        - l1, l2: penalty coefficients for sparsity and temporal homogeneity respectively. Can be empty and will be estimated using AIC
        - norm: boolean indicating whether to normalise resulting precision matrices
        - tol: convergence tolerance
        - max_iter: maximum number of iterations
        
    '''
    
    def __init__(self, data, h=None, pen_type="Fused", parallel=True, l1=None, l2=None, Approx=False, norm=True, tol=0.001, max_iter=100):
	self.data = data
	self.h = h
	self.pen_type = int(pen_type=="Fused")
	self.parallel = int(parallel==True)
	self.l1 = l1
	self.l2 = l2
	self.Approx = Approx
	self.norm=norm
	self.C_ = None
	self.tol = tol
	self.max_iter = max_iter
	self.P = None # estimate precision matrices
	self.iter_ = None
	self.AIC = None
	
	if self.pen_type==1:
	    import spams # only import if needed
	
	
    def __repr__(self):
	if self.pen_type==1:
	    mes = " ### SINGLE ###\n"
	else:
	    mes = " ### EL-SINGLE ###\n"
	if self.h!=None: mes += " # h:  " + str(self.h) + "\n"
	if self.l1!=None: mes += " # l1: " + str(self.l1) + "\n"
	if self.l2!=None: mes += " # l2: " + str(self.l2) + "\n"
	if self.parallel==1:
	    mes += " # Implemented using multiprocessing "
	return mes
	
    def fit_radius(self, h_vals, samples):
	"""Estimate radius of Gaussian kernel using cross-validation"""
	self.h = choose_h(data=self.data, rad_list=h_vals, samples=samples, parallel=self.parallel)
	# function to choose h (kernel width)
	
    def est_S(self):
	"""Once Gaussian kernel radius, h, has been estimated we can estimate sample covariance matrices"""
	
	if self.h==None:
	    raise Exception("Need to either provide Gaussian kernel radius, h, or estimate this using the fit_radius method")	
	C = get_kern_cov(self.data, radius=self.h)
	self.C_ = [None]*len(C)
	for i in range(len(C)):
	    self.C_[i] = C[i,:,:]
	
	
    def tune_params(self, l1=None, l2=None):
	"""Estimate parameters l1 and l2 based on AIC"""
	if l1==None:
	    l1 = numpy.linspace(numpy.percentile(abs(self.data), 25), numpy.percentile(abs(self.data), 50), 4)
	if l2==None:
	    l2 = numpy.linspace(.1, .5, 3)
	if self.h==None:
	    raise Exception("Need to either provide Gaussian kernel radius, h, or estimate this using the fit_radius method")
	if self.C_==None:
	    self.est_S()
	AIC_results = numpy.zeros((len(l1),len(l2)))
	for i in range(len(l1)):
	    for j in range(len(l2)):
		a,b,c,AIC_results[i,j] = fitSINGLE(S=self.C_, data=self.data, l1=l1[i], l2=l2[j], pen_type=self.pen_type, parallel=self.parallel)
		
	index = numpy.argmin(AIC_results)
	self.l1 = l1[ index % len(l1)]
	self.l2 = l2[ index % len(l2)]
    
    def fit(self):
    	#C = get_kern_cov(self.data, radius=self.h)
	#C_ = [None]*len(C)
	#for i in range(len(C)):
	    #C_[i] = C[i,:,:]
	if self.C_==None:
	    self.est_S()
	self.P, a, self.iter_, self.AIC = fitSINGLE(S=self.C_, data=self.data, l1=self.l1, l2=self.l2, pen_type=self.pen_type, parallel=self.parallel, Approx=self.Approx, max_iter=self.max_iter, tol=self.tol)
	# fit and normalise:
	if self.norm:
	    for i in range(len(self.P)):
		d = numpy.sqrt(numpy.diagonal(self.P[i,:,:]))
		self.P[i,:,:] /= numpy.outer(d,d)
    
    def plot(self, index, ncol_=None):
	"""add details and code"""
	if self.AIC==None:
	    raise Exception("First need to fit model using fit() function")
	plotSINGLE(self.P, ii=index, ncol_=ncol_) 
	
    
    
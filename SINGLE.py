## Python Implementation of Smooth Incremental Graphical Lasso Estimation (SINGLE) algorithm ## 
#
#
#

class SINGLE():
    '''Class for SINGLE objects
    
    Add more details here
    
    '''
    
    def __init__(self, data, h=None, k_type='gaussian', l1=None, l2=None, tol=0.001, max_iter=100):
	self.data = data
	self.h = h
	self.k_type = k_type
	self.l1 = l1
	self.l2 = l2
	self.tol = tol
	self.max_iter = max_iter
	self.P = None # estimate precision matrices
	self.iter_ = None
	self.AIC = None
	
    def fit_radius(self, h_vals, samples):
	self.h = choose_h(data=self.data, rad_list=h_vals, samples=samples)
	# function to choose h (kernel width)
	
    def tune_params(self, l1=None, l2=None):
	if l1==None:
	    l1 = numpy.linspace(numpy.percentile(abs(self.data), 25), numpy.percentile(abs(self.data), 50), 4)
	if l2==None:
	    l2 = numpy.linspace(.1, .5, 5)
	if self.h==None:
	    raise Exception("Need to either provide Gaussian kernel radius, h, or estimate this using the choose_h function")
	
	C = get_kern_cov(data, radius=self.h)
	C_ = [None]*len(C)
	for i in range(len(C)):
	    C_[i] = C[i,:,:]
	AIC_results = numpy.zeros((len(l1),len(l2)))
	for i in range(len(l1)):
	    for j in range(len(l2)):
		a,b,c,AIC_results[i,j] = fitSINGLE(S=C_, l1=l1[i], l2=l2[j])
		
	index = numpy.argmin(AIC_results)
	self.l1 = l1[ index % len(l1)]
	self.l2 = l2[ index % len(l2)]
    
    def fit():
    	C = get_kern_cov(data, radius=self.h)
	C_ = [None]*len(C)
	for i in range(len(C)):
	    C_[i] = C[i,:,:]
	self.P, a, self.iter_, self.AIC = fitSINGLE(S=C_, l1=self.l1, l2=self.l2)
    
	
    
    
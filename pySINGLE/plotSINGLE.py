## functions to plot results of SINGLE algorithm ##

from pylab import *
import math
import numpy

def plotSINGLE(P, ii, ncol_=None):
    """
    Plot the results after estimating sparse precision matrices, P, according to the SINGLE algorithm.
    
    INPUT:
         - P: array of estimated precision matrices
         - ii: list of node indices, we will plot all possible pairs
    
    OUTPUT:
         - plot (add details)
    
    """
    
    if ncol_==None: ncol_=4
    nrow_ = numpy.ceil((len(ii)*(len(ii)-1)/2.) / ncol_)
    
    fig, axes = matplotlib.pyplot.subplots(ncols=int(ncol_), nrows=int(nrow_), sharex=True, sharey=False)
    
    counter_col = 0
    counter_row = 0
    for i in range(len(ii)):
	for j in range(i+1, len(ii)):
	    axes[counter_row,counter_col].plot(P[:, ii[i], ii[j]])
	    axes[counter_row, counter_col].set_title("PC nodes "+str(ii[i])+" and  "+str(ii[j]))
	    counter_col = (counter_col+1) % ncol_
	    counter_row += int(counter_col==0)
	    
    
    show()


import numpy as np
import matplotlib.pyplot as plt

def plotData(X, y):
    #PLOTDATA Plots the data points X and y into a new figure 
    #   PLOTDATA(x,y) plots the data points with + for the positive examples
    #   and o for the negative examples. X is assumed to be a Mx2 matrix.
    #
    # Note This was slightly modified such that it expects y = 1 or y = 0
    
    # Find Indices of Positive and Negative Examples
    
    
    pos = np.where(y.flatten()==1)
    neg = np.where(y.flatten()==0)
    
    # Plot Examples
    plt.plot(X[pos, 0], X[pos, 1], 'k+',linewidth=1, markersize=7)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', color='y', markersize=7)
    

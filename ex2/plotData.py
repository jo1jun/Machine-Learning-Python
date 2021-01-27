import numpy as np
import matplotlib.pyplot as plt

def plotData(X, y):
    #PLOTDATA Plots the data points X and y into a new figure 
    #   PLOTDATA(x,y) plots the data points with + for the positive examples
    #   and o for the negative examples. X is assumed to be a Mx2 matrix.
    
    # Find Indices of Positive and Negative Examples 
    pos = np.where(y==1)            #조건에 맞는 원소의 index 추출
    neg = np.where(y==0)

    # Plot Examples 
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', linewidths=2)
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', c='y')
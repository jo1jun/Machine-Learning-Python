import numpy as np
import matplotlib.pyplot as plt

def plotDataPoints(X, idx, K):
    #PLOTDATAPOINTS plots data points in X, coloring them so that those with the same
    #index assignments in idx have the same color
    #   PLOTDATAPOINTS(X, idx, K) plots data points in X, coloring them so that those 
    #   with the same index assignments in idx have the same color
    
    # Create palette
    colors = ['r', 'g', 'b']
    
    # Plot the data
    # label 에 맞는 색깔을 입힌다.
    for i in range(K):
        idxs = np.where(idx == i)
        plt.scatter(X[idxs,0], X[idxs,1], s=15, c=colors[i])
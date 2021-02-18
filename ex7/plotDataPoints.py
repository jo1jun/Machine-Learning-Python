import matplotlib.pyplot as plt

def plotDataPoints(X, idx, K):
    #PLOTDATAPOINTS plots data points in X, coloring them so that those with the same
    #index assignments in idx have the same color
    #   PLOTDATAPOINTS(X, idx, K) plots data points in X, coloring them so that those 
    #   with the same index assignments in idx have the same color
    
    # Plot the data
    
    # reference : https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html?highlight=scatter#matplotlib.pyplot.scatter
    # cmap : https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html#sphx-glr-tutorials-colors-colormaps-py
    plt.scatter(X[:,0], X[:,1], s=15, c=idx, cmap='gist_rainbow')

from plotData import plotData
import numpy as np
import matplotlib.pyplot as plt

def visualizeBoundary(X, y, model):
    #VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
    #   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision 
    #   boundary learned by the SVM and overlays the data on it
    
    # Plot the training data on top of the boundary
    plotData(X, y)
    
    # Make classification predictions over a grid of values
    x1plot = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100).reshape(-1,1)
    x2plot = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100).reshape(-1,1)

    X1, X2 = np.meshgrid(x1plot, x2plot)    #X1.shape = X2.shape = (100, 100)
    vals = np.zeros_like(X1)

    for i in range(X2.shape[1]):
       this_X = np.append(X1[:, i].reshape(-1,1), X2[:, i].reshape(-1,1), axis=1)
       vals[:, i] = model.predict(this_X)
       
    # Plot the SVM boundary
    plt.contour(X1, X2, vals, colors='b')

    
import numpy as np
import matplotlib.pyplot as plt
from plotData import plotData

def visualizeBoundaryLinear(X, y, model):
    #VISUALIZEBOUNDARYLINEAR plots a linear decision boundary learned by the
    #SVM
    #   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary 
    #   learned by the SVM and overlays the data on it
    
    w = model.coef_.flatten()
    b = model.intercept_.flatten()
    xp = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    yp = - (w[0] * xp + b) / w[1]
    plotData(X, y)
    plt.plot(xp, yp, '-b')
    
    # reference : https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    
    #coef_ : ndarray of shape (n_classes * (n_classes - 1) / 2, n_features)
    #Weights assigned to the features (coefficients in the primal problem). This is only available in the case of a linear kernel.
    
    #intercept_ : ndarray of shape (n_classes * (n_classes - 1) / 2,)
    #Constants in decision function.
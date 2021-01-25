import numpy as np

def computeCostMulti(X, y, theta):
    
    m = y.shape[0]
    
    J = np.sum(np.square((np.dot(X,theta) - y))) / (2*m)
    
    return J

    
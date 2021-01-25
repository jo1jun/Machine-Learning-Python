import numpy as np

def computeCost(X, y, theta):
    
    m = y.shape[0]
    J = 0
    for i in range(m):
        J += ((np.dot(X[i,:], theta) - y[i]) ** 2)

    J /= (2*m)
    
    return J

    #need to vectorize -> computeCostMulti
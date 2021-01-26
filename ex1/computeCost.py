import numpy as np

#COMPUTECOST Compute cost for linear regression
#   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
#   parameter for linear regression to fit the data points in X and y

def computeCost(X, y, theta):
    
    # Initialize some useful values
    m = y.shape[0]
    # You need to return the following variables correctly 
    J = 0
    
    for i in range(m):
        J += ((np.dot(X[i,:], theta) - y[i]) ** 2)
    J /= (2*m)
    
    return J

    # **need to vectorize -> computeCostMulti**
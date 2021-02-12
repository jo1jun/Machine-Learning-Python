import numpy as np
def computeCost(X, y, theta):
    #COMPUTECOST Compute cost for linear regression
    #   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    #   parameter for linear regression to fit the data points in X and y
 
    # Initialize some useful values
    m = y.shape[0]
    # You need to return the following variables correctly 
    J = 0
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.
    
    # 처음이니까 for loop 으로. vectorize 하자. -> computeCostMulti 에서 구현.
    for i in range(m):
        J += ((np.dot(X[i,:], theta) - y[i]) ** 2)
    J /= (2*m)
    
    return J
    
    # =========================================================================
def computeCostMulti(X, y, theta):
    #COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
    #   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
    #   parameter for linear regression to fit the data points in X and y
    
    # Initialize some useful values
    m = y.shape[0]  # number of training examples
    
    # You need to return the following variables correctly 
    J = 0
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.
    
    #J = np.sum(np.square((np.dot(X,theta) - y))) / (2*m)
    
    # fully vectorized. more efficient!
    
    J = (X @ theta - y).T @ (X @ theta - y) / (2*m)
    
    return J

    # =========================================================================
    
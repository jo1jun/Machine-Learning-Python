import numpy as np

def linearRegCostFunction(X,y,theta,_lambda):
    #LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
    #regression with multiple variables
    #   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
    #   cost of using theta as the parameter for linear regression to fit the 
    #   data points in X and y. Returns the cost in J and the gradient in grad
    
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    theta = theta.reshape(X.shape[1],1)
    
    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros_like(theta)
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost and gradient of regularized linear 
    #               regression for a particular choice of theta.
    #
    #               You should set J to the cost and grad to the gradient.
    #
    
    J = ((X @ theta - y).T @ (X @ theta - y) + _lambda*(theta[1:,:].T @ theta[1:,:])) / (2 * m)
    
    grad = (X.T @ (X @ theta - y))/m
    grad[1:,:] += (_lambda / m) * theta[1:,:]
    
    grad = grad.flatten()
    return J, grad

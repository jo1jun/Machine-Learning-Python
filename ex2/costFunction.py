import numpy as np
from sigmoid import sigmoid

def costFunction(theta, X, y):

    #COSTFUNCTION Compute cost and gradient for logistic regression
    #   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    #   parameter for logistic regression and the gradient of the cost
    #   w.r.t. to the parameters.
    
    # Initialize some useful values
    
    m = y.shape[0] # number of training examples
    
    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros_like(theta)
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Note: grad should have the same dimensions as theta
    #
    
    #fully vectorize
    
    J = -(y.T @ np.log(sigmoid(X @ theta)) + (1 - y).T @ np.log(1 - sigmoid(X @ theta))) / m
    
    grad = X.T @ (sigmoid(X @ theta) - y) / m
    
    return J, grad
    
    # =============================================================

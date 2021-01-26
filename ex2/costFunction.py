import numpy as np
from sigmoid import sigmoid

def costFunction(theta, X, y):

    #COSTFUNCTION Compute cost and gradient for logistic regression
    #   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    #   parameter for logistic regression and the gradient of the cost
    #   w.r.t. to the parameters.
    
    # Initialize some useful values
    
    m,n = X.shape # number of training examples
    
    # You need to return the following variables correctly 
    J = 0
    theta = np.array(theta)[:, np.newaxis]
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
    
    h = sigmoid(X @ theta)
    
    dummy = 1e-7    #log 안에 0 값을 방지
    
    J = -(y.T @ np.log(h + dummy) + (1 - y).T @ np.log(1 - h + dummy)) / m

    grad = X.T @ (h - y) / m
    
    grad = grad.flatten()
    
    return J, grad
    
    # =============================================================

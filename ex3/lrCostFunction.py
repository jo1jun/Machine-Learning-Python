import numpy as np
from sigmoid import sigmoid

def lrCostFunction(theta, X, y, _lambda):
    #LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
    #regularization
    #   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters. 
    
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    # You need to return the following variables correctly 
    J = 0
    theta = theta.reshape((np.size(theta),1))
    grad = np.zeros_like(theta)
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Hint: The computation of the cost function and gradients can be
    #       efficiently vectorized. For example, consider the computation
    #
    #           sigmoid(X * theta)
    #
    #       Each row of the resulting matrix will contain the value of the
    #       prediction for that example. You can make use of this to vectorize
    #       the cost function and gradient computations. 
    #
    # Hint: When computing the gradient of the regularized cost function, 
    #       there're many possible vectorized solutions, but one solution
    #       looks like:
    #           grad = (unregularized gradient for logistic regression)
    #           temp = theta; 
    #           temp(1) = 0;   # because we don't add anything for j = 0  
    #           grad = grad + YOUR_CODE_HERE (using the temp variable)
    #
    
    
    J = (-y.T @ np.log(sigmoid(X @ theta)) - (1-y).T @ np.log(1-sigmoid(X @ theta))) / m 
    J += (_lambda / (2*m)) * (theta[1:].T @ theta[1:]) #np.sum(np.square(theta)) 를 theta 끼리의 dot product 로 구할 수 있다.
    #0 번 index 는 제외.
    grad = (X.T @ (sigmoid(X @ theta) - y)) / m
    grad[1:] += (_lambda / m) * theta[1:] #0 번 index 는 제외.
    
    grad = grad.flatten()
    
    return J, grad
    
    # =============================================================
    
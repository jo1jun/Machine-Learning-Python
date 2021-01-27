import numpy as np
from sigmoid import sigmoid

def costFunctionReg(theta, X, y, _lambda):

    #COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    #   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters. 

    # Initialize some useful values
    m, n = X.shape # number of training examples

    # You need to return the following variables correctly 
    J = 0
    y = y[:,np.newaxis]
    theta = theta[:,np.newaxis]
    grad = np.zeros_like(theta)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta

    J = -(y.T @ np.log(sigmoid(X @ theta)) + (1-y).T @ np.log(1-sigmoid(X @ theta))) / m
    
    J_regterm = _lambda/(2*m) * np.sum(np.square(theta[1:]))  #theta_0 는 regularization term 에 포함 x
    
    J += J_regterm
    
    grad = (X.T @ (sigmoid(X @ theta) - y)) / m

    g_regterm = _lambda/m * theta

    grad[1:] += g_regterm[1:]   #theta_0 는 regularization term 에 포함 x
    
    grad = grad.flatten()              #optimizer 을 사용하기 위해 평탄화.

    return J,grad

    # =============================================================


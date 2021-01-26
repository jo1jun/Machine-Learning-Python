import numpy as np
from costFunction import costFunction
from sigmoid import sigmoid

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = y.shape[0] # number of training examples
    
    for i in range(num_iters):
    
        theta -= alpha * X.T @ (sigmoid(X @ theta) - y) / m 
    
    J,_ = costFunction(theta, X, y)
    
    return theta, J

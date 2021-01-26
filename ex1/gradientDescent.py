import numpy as np
from computeCost import computeCost

#GRADIENTDESCENT Performs gradient descent to learn theta
#   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
#   taking num_iters gradient steps with learning rate alpha

def gradientDescent(X, y, theta, alpha, num_iters):
    
    # Initialize some useful values
    m = y.shape[0]  # number of training examples
    J_history = np.zeros((num_iters,1))
    
    for idx in range(num_iters):
        delta = np.zeros_like(theta, float)
        for i in range(m):
            for j in range (theta.shape[0]):
                delta[j,0] += (np.dot(X[i,:], theta) - y[i]) * X[i,j]
        delta *= alpha
        delta /= m
        theta -= delta
            
        J_history[idx] = computeCost(X, y, theta)
        
    return theta, J_history

    #   **need to vectorize -> gradientDescentMulti**
import numpy as np
from computeCostMulti import computeCostMulti

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = y.shape[0] # number of training examples
    J_history = np.zeros((num_iters, 1));
    
    for i in range(num_iters):
    
        theta -= alpha * np.dot(X.T, (np.dot(X, theta) - y.reshape(m,1))) / m
                # (3,m) x ((m,3) x (3,1) - (m,1)) = (3,1)

        # Save the cost J in every iteration    
        J_history[i,0] = computeCostMulti(X, y, theta);
    
    
    return theta, J_history

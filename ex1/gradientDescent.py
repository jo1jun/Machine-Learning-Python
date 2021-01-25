import numpy as np

def gradientDescent(X, y, theta, alpha, num_iters):
    
    m = y.shape[0]
    
    for idx in range(num_iters):
        delta = np.zeros_like(theta, float)
        for i in range(m):
            for j in range (theta.shape[0]):
                delta[j,0] += (np.dot(X[i,:], theta) - y[i]) * X[i,j]
        delta *= alpha
        delta /= m
        theta -= delta
        

    return theta

    #need to vectorize -> gradientDescentMulti
import numpy as np
from sigmoid import sigmoid

def sigmoidGradient(X):

    #SIGMOIDGRADIENT returns the gradient of the sigmoid function
    #evaluated at z
    #   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
    #   evaluated at z. This should work regardless if z is a matrix or a
    #   vector. In particular, if z is a vector or matrix, you should return
    #   the gradient for each element.
    
    X = np.array(X)
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of the sigmoid function evaluated at
    #               each value of z (z can be a matrix, vector or scalar).
    
    #직접 미분해보면 아래와 같은 식을 쉽게 얻을 수 있다.
    return sigmoid(X)*(1-sigmoid(X))
    # =============================================================

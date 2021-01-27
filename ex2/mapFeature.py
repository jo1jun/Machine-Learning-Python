import numpy as np

def mapFeature(X1, X2):
    
    # MAPFEATURE Feature mapping function to polynomial features
    #
    #   MAPFEATURE(X1, X2) maps the two input features
    #   to quadratic features used in the regularization exercise.
    #
    #   Returns a new feature array with more features, comprising of 
    #   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    #
    #   Inputs X1, X2 must be the same size
    #

    degree = 6
    
    if np.size(X1) != 1:
        X1 = X1[:,np.newaxis]
        X2 = X2[:,np.newaxis]
        out = np.ones_like(X1)
    
        for i in range(degree):
            for j in range((i+1) + 1):
                out = np.append(out, ((X1 ** ((i+1)-j)) * (X2 ** j)), axis=1)
            
    else:
        out = np.ones_like(X1)
        for i in range(degree):
            for j in range((i+1) + 1):
                out = np.append(out, ((X1 ** ((i+1)-j)) * (X2 ** j)))

    return out
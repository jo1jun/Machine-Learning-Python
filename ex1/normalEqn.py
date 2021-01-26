import numpy as np

#NORMALEQN Computes the closed-form solution to linear regression 
#   NORMALEQN(X,y) computes the closed-form solution to linear 
#   regression using the normal equations.

def normalEqn(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

#np.linalg.inv(matrix) : 역행렬 return
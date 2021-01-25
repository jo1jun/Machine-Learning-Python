import numpy as np

def normalEqn(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

#np.linalg.inv(matrix) : 역행렬 return
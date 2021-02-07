import numpy as np

def pca(X):
    #PCA Run principal component analysis on the dataset X
    #   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
    #   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
    #
    
    # Useful values
    m, n = X.shape
    
    # You need to return the following variables correctly.
    U = np.zeros(n)
    S = np.zeros(n)
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: You should first compute the covariance matrix. Then, you
    #               should use the "svd" function to compute the eigenvectors
    #               and eigenvalues of the covariance matrix. 
    #
    # Note: When computing the covariance matrix, remember to divide by m (the
    #       number of examples).
    #

    Sigma = (1 / m) * X.T @ X
    
    # reference : https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html?highlight=svd#numpy.linalg.svd
    U, S, V = np.linalg.svd(Sigma)
    
    
    return U, S
    
    # =========================================================================

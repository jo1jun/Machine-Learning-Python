import numpy as np

def findClosestCentroids(X, centroids):
    #FINDCLOSESTCENTROIDS computes the centroid memberships for every example
    #   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
    #   in idx for a dataset X where each row is a single example. idx = m x 1 
    #   vector of centroid assignments (i.e. each entry in range [1..K])
    #
    
    # Set K
    K = centroids.shape[0]
    
    # You need to return the following variables correctly.
    idx = np.zeros((X.shape[0], 1)) # lecture 에서 c 를 의미한다.
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Go over every example, find its closest centroid, and store
    #               the index inside idx at the appropriate location.
    #               Concretely, idx(i) should contain the index of the centroid
    #               closest to example i. Hence, it should be a value in the 
    #               range 1..K
    #
    # Note: You can use a for-loop over the examples to compute this.
    #
    
    m , n = X.shape
    temp = np.zeros((m,K))

    for i in range(K):
        temp[:,i] = np.sum(np.square((X - centroids[i,:])), axis = 1)
    
    idx = np.argmin(temp, axis=1)
    return idx

# =============================================================



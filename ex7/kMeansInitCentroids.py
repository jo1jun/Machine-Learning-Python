import numpy as np
def kMeansInitCentroids(X,K):
    #KMEANSINITCENTROIDS This function initializes K centroids that are to be 
    #used in K-Means on the dataset X
    #   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
    #   used with the K-Means on the dataset X
    #
    
    # You should return this values correctly
    centroids = np.zeros((K, X.shape[1]))
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: You should set centroids to randomly chosen examples from
    #               the dataset X
    #
    idxs = np.arange(X.shape[0])
    np.random.shuffle(idxs)
    centroids = X[idxs[:K]]
    
    return centroids
    # =============================================================


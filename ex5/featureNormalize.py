import numpy as np

def featureNormalize(X): 
    #FEATURENORMALIZE Normalizes the features in X 
    #   FEATURENORMALIZE(X) returns a normalized version of X where
    #   the mean value of each feature is 0 and the standard deviation
    #   is 1. This is often a good preprocessing step to do when
    #   working with learning algorithms.
    mu = np.mean(X,axis=0).reshape(1,-1)
    X_norm = X - mu
    
    sigma = np.std(X_norm,axis=0, ddof=1).reshape(1,-1)
    #ddof = 0 (default) : n 으로 나눔 / ddof = 1 : n-1 로 나눔.
    X_norm = X_norm / sigma
    
    return X_norm, mu, sigma
    # ============================================================
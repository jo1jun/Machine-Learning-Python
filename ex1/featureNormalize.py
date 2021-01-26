import numpy as np

#FEATURENORMALIZE Normalizes the features in X 
#   FEATURENORMALIZE(X) returns a normalized version of X where
#   the mean value of each feature is 0 and the standard deviation
#   is 1. This is often a good preprocessing step to do when
#   working with learning algorithms.

def featureNormalize(X):

    # You need to set these values correctly
    X_norm = X
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))    

    mu = np.mean(X,axis=0).reshape(1,X.shape[1])        #axis = 0 -> shape[0](row(행)) 에 해당하는 모든 원소들
    sigma = np.std(X,axis=0).reshape(1,X.shape[1])      #broadcast 하기 위해 reshape

    #broadcast 로 한번에.            
    X_norm -= mu
    X_norm /= sigma
            
    return X, mu, sigma
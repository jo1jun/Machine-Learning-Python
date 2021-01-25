import numpy as np

def featureNormalize(X):

    # You need to set these values correctly
    X_norm = X
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))    

    mu = np.mean(X,axis=0).reshape(1,X.shape[1])        #axis = 0 -> shape[0] 에 해당하는 모든 원소들
    sigma = np.std(X,axis=0).reshape(1,X.shape[1])

    #broadcast 로 한번에.            
    X_norm -= mu
    X_norm /= sigma
            
    return X, mu, sigma
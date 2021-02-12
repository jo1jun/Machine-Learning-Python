import numpy as np

def predictSoftmaxWithLoss(theta, X):
    
    m = X.shape[0]
    
    p = np.zeros((m, 1))
    
    theta = theta.reshape((X.shape[1],10))
    
    p = np.argmax(X @ theta, axis=1) + 1 #index 가 0부터 시작하므로 + 1 로 보정.
        
    return p
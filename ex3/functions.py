import numpy as np

def softmax(x):

    if x.ndim == 2:
        x -= x.max(axis = 1).reshape(x.shape[0],1)      #broadcast 를 위해 reshape
        return np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims=True)   #broadcast 를 위해 차원 유지
    
    x -= x.max()
    return np.exp(x) / np.sum(np.exp(x))
    
def cross_entropy_error(a,y):
    
    m = a.shape[0]
    y = y.reshape(m)
    return -np.sum(np.log(a[np.arange(m), y - 1] + 1e-7)) / m
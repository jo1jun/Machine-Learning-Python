import numpy as np

from functions import softmax, cross_entropy_error

def lrSoftmaxWithLoss(theta, X, y, _lambda):

    m = y.shape[0]
    
    J = 0
    
    #Deep Learning from Scratch 에서 학습했던 Softmax & cross entropy error 를 활용하면 단 한 번의 loop 로  학습이 가능!
    
    theta = theta.reshape((X.shape[1],10))
    grad = np.zeros_like(theta)
    a = softmax(X @ theta)
    J = cross_entropy_error(a, y) + 0.5 * (_lambda/m) * np.sum(np.square(theta[1:,:]))  # regularization term 추가.
    
    print('J : ', J)
    temp = np.zeros_like(a)
    y = y.reshape(m)
    temp[np.arange(m),y-1] = 1  #원 핫 인코딩**

    dout = 1
    dout /= m
    dout = dout * (a - temp)    #a 와 temp 모두 one hot 인코딩**
    grad = X.T @ dout
    grad[1:] += (_lambda/m) * theta[1:]     # regularization term 추가.
    
    grad = grad.flatten()
    
    return J,grad
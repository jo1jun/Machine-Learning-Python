import numpy as np
from lrSoftmaxWithLoss import lrSoftmaxWithLoss

def trainSoftmaxWithLoss(X, y, num_labels, _lambda):
    
    import scipy.optimize as op
    
    #Deep Learning from Scratch 에서 학습했던 Softmax & cross entropy error 를 활용하면 단 한 번의 loop 로  학습이 가능!
    initial_theta =  0.01 * np.random.randn(X.shape[1],10).flatten()
    
    result = op.minimize(fun=lrSoftmaxWithLoss,x0=initial_theta,args=(X,y,_lambda),method='L-BFGS-B',jac=True,options={'maxiter':50})
    
    return result.x
    
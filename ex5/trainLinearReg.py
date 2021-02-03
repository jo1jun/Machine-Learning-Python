import numpy as np
from linearRegCostFunction import linearRegCostFunction
import scipy.optimize as op

def trainLinearReg(X, y, _lambda):
    #TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
    #regularization parameter lambda
    #   [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
    #   the dataset (X, y) and regularization parameter lambda. Returns the
    #   trained parameters theta.
    #

    # Initialize Theta
    initial_theta = np.zeros(X.shape[1]) 
    
    # Create "short hand" for the cost function to be minimized    
    costFunction = lambda t: linearRegCostFunction(X, y, t, _lambda)

    # Now, costFunction is a function that takes in only one argument    
    result = op.minimize(fun=costFunction,x0=initial_theta,method='L-BFGS-B',\
                         jac=True,options={'maxiter':15})
    #maxiter 를 200 이 아니라 15 번으로 해야 ex5.pdf 에 나와있는 그림처럼 그려진다.
    
    return result.x

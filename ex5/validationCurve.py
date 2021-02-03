import numpy as np
from trainLinearReg import trainLinearReg
from linearRegCostFunction import linearRegCostFunction

def validationCurve(X, y, Xval, yval):
    #VALIDATIONCURVE Generate the train and validation errors needed to
    #plot a validation curve that we can use to select lambda
    #   [lambda_vec, error_train, error_val] = ...
    #       VALIDATIONCURVE(X, y, Xval, yval) returns the train
    #       and validation errors (in error_train, error_val)
    #       for different values of lambda. You are given the training set (X,
    #       y) and validation set (Xval, yval).
    #
    
    # Selected values of lambda (you should not change this)
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    # You need to return these variables correctly.
    error_train = np.zeros_like(lambda_vec)
    error_val = np.zeros_like(lambda_vec)
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return training errors in 
    #               error_train and the validation errors in error_val. The 
    #               vector lambda_vec contains the different lambda parameters 
    #               to use for each calculation of the errors, i.e, 
    #               error_train(i), and error_val(i) should give 
    #               you the errors obtained after training with 
    #               lambda = lambda_vec(i)
    #
    # Note: You can loop over lambda_vec with the following:
    #
    #       for i = 1:length(lambda_vec)
    #           lambda = lambda_vec(i)
    #           # Compute train / val errors when training linear 
    #           # regression with regularization parameter lambda
    #           # You should store the result in error_train(i)
    #           # and error_val(i)
    #           ....
    #           
    #       end
    
    #reg term 이 달린 J와 train set으로 학습
    #학습으로 구한 theta, reg term 을 뺀 J로 train set error, CV set error 구한다. 
    for i in range(np.size(lambda_vec)):
        _lambda = lambda_vec[i]
        theta = trainLinearReg(X,y,_lambda)
        error_train[i],_ = linearRegCostFunction(X,y,theta,0)
        error_val[i],_ = linearRegCostFunction(Xval,yval,theta,0)
    
    return lambda_vec, error_train, error_val
    
    # =========================================================================

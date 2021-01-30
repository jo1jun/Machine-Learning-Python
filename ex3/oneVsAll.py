import numpy as np
from lrCostFunction import lrCostFunction

def oneVsAll(X, y, num_labels, _lambda):
    
    #ONEVSALL trains multiple logistic regression classifiers and returns all
    #the classifiers in a matrix all_theta, where the i-th row of all_theta 
    #corresponds to the classifier for label i
    #   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
    #   logistic regression classifiers and returns each of these classifiers
    #   in a matrix all_theta, where the i-th row of all_theta corresponds 
    #   to the classifier for label i
    
    # Some useful variables
    m , n = X.shape
    
    # You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1))
    
    # Add ones to the X data matrix
    X = np.append(np.ones((m,1)), X, axis=1)
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the following code to train num_labels
    #               logistic regression classifiers with regularization
    #               parameter lambda. 
    #
    # Hint: theta(:) will return a column vector.
    #
    # Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
    #       whether the ground truth is true/false for this class.
    #
    # Note: For this assignment, we recommend using fmincg to optimize the cost
    #       function. It is okay to use a for-loop (for c = 1:num_labels) to
    #       loop over the different classes.
    #
    #       fmincg works similarly to fminunc, but is more efficient when we
    #       are dealing with large number of parameters.
    #
    # Example Code for fmincg:
    #
    #     # Set Initial theta
    #     initial_theta = zeros(n + 1, 1);
    #     
    #     # Set options for fminunc
    #     options = optimset('GradObj', 'on', 'MaxIter', 50);
    # 
    #     # Run fmincg to obtain the optimal theta
    #     # This function will return theta and the cost 
    #     [theta] = ...
    #         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
    #                 initial_theta, options);
    #fmincg는 fminunc와 유사하게 작동하지만 많은 수의 매개 변수를 처리하는 데 더 효율적입니다.
    # y 는 1 또는 0 이 아닌 정수값. (10개의 class 이므로 1 ~ 10)
    # 따라서 logistic regression 의 cost function 을 사용하기 위해서 10번 반복 (lrCostFunction 이 0 또는 1 의 label 을 처리하기 때문)
    # 해당 class 정수가 맞으면 1 아니면 0 으로 변환해서 advanced optimizer 에 넘겨줌.
    
    #  octave 의 fmincg 대신, python 에서 동작하는 프레임 워크(scipy.optimize.minimize)를 활용
    #  reference : https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    #입출력 theta 는 반드시 평탄화가 되어있어야 한다.
    #costFunction 내부에서 평탄화를 다시 reshaping 하면 된다.
    #argument 로 method = '특정 method' 로 지정할 수도 있다.
    #If jac is a Boolean and is True, fun(costFunction) is assumed to return and objective and gradient as an (f, g) tuple.
    
    import scipy.optimize as op
    
    for c in range(num_labels):
        c = c+1 #0번 index 시작을 1번 index 시작으로.
        y_t = np.asarray(y==c,int)    #class number 에서 0,1 로 변환
        initial_theta = np.zeros(n+1)
        result = op.minimize(fun=lrCostFunction,x0=initial_theta,args=(X,y_t,_lambda),method='L-BFGS-B',jac=True,options={'maxiter':50})
        theta = np.reshape(result.x, (1,n+1))
        all_theta[c-1,:] = theta
        print('[', c ,'/', num_labels, ' completed..]')
    
    return all_theta
    
    # =========================================================================

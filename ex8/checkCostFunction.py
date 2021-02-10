import numpy as np
from computeNumericalGradient import computeNumericalGradient
from cofiCostFunc import cofiCostFunc

def checkCostFunction(_lambda = 0):
    #CHECKCOSTFUNCTION Creates a collaborative filering problem 
    #to check your cost function and gradients
    #   CHECKCOSTFUNCTION(_lambda) Creates a collaborative filering problem 
    #   to check your cost function and gradients, it will output the 
    #   analytical gradients produced by your code and the numerical gradients 
    #   (computed using computeNumericalGradient). These two gradient 
    #   computations should result in very similar values.
    
    ## Create small problem
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)
    
    # Zap out most entries
    Y = X_t @ Theta_t.T
    Y[(np.random.rand(np.size(Y)) > 0.5).reshape(Y.shape)] = 0  # 배열 을 논리연산하면 bool 배열 반환된다.
    R = np.zeros_like(Y)                                        # bool 배열을 index 로 사용 가능
    R[Y != 0] = 1

    ## Run Gradient Checking
    X = np.random.randn(X_t.shape[0], X_t.shape[1])
    Theta = np.random.randn(Theta_t.shape[0], Theta_t.shape[1])
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]
    
    costFunc = lambda t: cofiCostFunc(t, Y, R, num_users, num_movies, num_features, _lambda)
    
    numgrad = computeNumericalGradient(costFunc, np.append(X.flatten(), Theta.flatten()))

    cost, grad = costFunc(np.append(X.flatten(), Theta.flatten()))
    
    print(np.append(numgrad.reshape(-1,1), grad.reshape(-1,1), axis=1))
    print('The above two columns you get should be very similar.\n'\
             '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n')
    
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print('If your cost function implementation is correct, then \n'\
             'the relative difference will be small (less than 1e-9). \n'\
             '\nRelative Difference: ', diff, '\n')
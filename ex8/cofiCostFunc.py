import numpy as np

def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, _lambda):
    #COFICOSTFUNC Collaborative filtering cost function
    #   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
    #   num_features, _lambda) returns the cost and gradient for the
    #   collaborative filtering problem.
    #
    
    # Unfold the U and W matrices from params
    X = np.reshape(params[:num_movies*num_features], (num_movies, num_features))
    Theta = np.reshape(params[num_movies*num_features:], \
                    (num_users, num_features))
    
    # You need to return the following values correctly
    J = 0
    X_grad = np.zeros_like(X)
    Theta_grad = np.zeros_like(Theta)
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost function and gradient for collaborative
    #               filtering. Concretely, you should first implement the cost
    #               function (without regularization) and make sure it is
    #               matches our costs. After that, you should implement the 
    #               gradient and use the checkCostFunction routine to check
    #               that the gradient is correct. Finally, you should implement
    #               regularization.
    #
    # Notes: X - num_movies  x num_features matrix of movie features
    #        Theta - num_users  x num_features matrix of user features
    #        Y - num_movies x num_users matrix of user ratings of movies
    #        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
    #            i-th movie was rated by the j-th user
    #
    # You should set the following variables correctly:
    #
    #        X_grad - num_movies x num_features matrix, containing the 
    #                 partial derivatives w.r.t. to each element of X
    #        Theta_grad - num_users x num_features matrix, containing the 
    #                     partial derivatives w.r.t. to each element of Theta
    #
    
    # R 이 1인 idx 를 찾아서 for 문을 돌려도 되지만 R 을 element wise 곱해서 연산하면
    # R 이 0인 부분은 0이 되므로 연산해도 0이된다. 즉, 포함되지 않는다. R 이 1인 부분만 연산된다.
    # R 을 element wise 로 곱하여 1인 경우에만 제곱합을 하게끔 한다.
    J = (1 / 2) * np.sum(np.square((X @ Theta.T - Y) * R))
    J_reg = (_lambda / 2) * (np.sum(np.square(Theta)) + np.sum(np.square(X)))
    J += J_reg

    X_grad = ((X @ Theta.T - Y) * R) @ Theta + _lambda * X
    Theta_grad = ((X @ Theta.T - Y) * R).T @ X + _lambda * Theta
    
    # 위 reg term 의 bias 에 대해서 주석 추가.
    
    # =============================================================
    
    grad = np.append(X_grad.flatten(), Theta_grad.flatten())
    
    return J, grad
import numpy as np
def normalEqn(X, y):
    #NORMALEQN Computes the closed-form solution to linear regression 
    #   NORMALEQN(X,y) computes the closed-form solution to linear 
    #   regression using the normal equations.

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the code to compute the closed form solution
    #               to linear regression and put the result in theta.
    #
    
    # ---------------------- Sample Solution ----------------------

    #np.linalg.inv(matrix) : 역행렬 return
    return np.linalg.inv(X.T @ X) @ X.T @ y

    # -------------------------------------------------------------


    # ============================================================

import numpy as np

def computeNumericalGradient(J, theta):
    #COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
    #and gives us a numerical estimate of the gradient.
    #   numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
    #   gradient of the function J around theta. Calling y = J(theta) should
    #   return the function value at theta.
    
    # Notes: The following code implements numerical gradient checking, and 
    #        returns the numerical gradient.It sets numgrad(i) to (a numerical 
    #        approximation of) the partial derivative of J with respect to the 
    #        i-th input argument, evaluated at theta. (i.e., numgrad(i) should 
    #        be the (approximately) the partial derivative of J with respect 
    #        to theta(i).)
    #                
    
    numgrad = np.zeros(np.size(theta))
    perturb = np.zeros(np.size(theta))
    e = 1e-4
    for p in range(np.size(theta)):         #(cost function J 는 theta 를 unroll 된 상태로 받아들임!)
        # Set perturbation vector
        perturb[p] = e
        loss1,_ = J(theta - perturb)        #모든 theta를 하나씩 조금씩 변경하여 수치적 편미분을 구한다.
        loss2,_ = J(theta + perturb)
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e)    
        perturb[p] = 0                      #반드시 조금 변경한 하나의 theta 를 원래대로! 편미분을 제대로 수행하기 위해서다.
        
    return numgrad
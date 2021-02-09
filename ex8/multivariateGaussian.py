import numpy as np
import math

def multivariateGaussian(X, mu, sigma2):
    #MULTIVARIATEGAUSSIAN Computes the probability density function of the
    #multivariate gaussian distribution.
    #    p = MULTIVARIATEGAUSSIAN(X, mu, Sigma2) Computes the probability 
    #    density function of the examples X under the multivariate gaussian 
    #    distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
    #    treated as the covariance matrix. If Sigma2 is a vector, it is treated
    #    as the \sigma^2 values of the variances in each dimension (a diagonal
    #    covariance matrix)
    #
    
    '''
    # case1 : Original Gaussian model
    # 이것을 적용해도 잘 나온다.
    p = (1 / np.sqrt(2 * math.pi * sigma2)) * np.exp(-(((X - mu) ** 2) / (2 * sigma2)))
    # 각각의 feature 에 대한 p(x) 들을 구하고 이들을 전부 곱한다.(모든 열을 곱한다.)
    # reduce reference : https://numpy.org/doc/stable/reference/generated/numpy.ufunc.reduce.html
    p = np.multiply.reduce(p,axis=1)
    '''
    
    # case2 : MultivariateGaussian
    k = np.size(mu)
    if(sigma2.ndim == 1):
        sigma2 = np.diag(sigma2)
    # bsxfun 은 python 에서의 broadcasting. 앞선 exercise 에서 언급했었다.
    # 아래 공식은 당연히 암기할 필요 x
    X = X - mu
    p = (2 * math.pi)**(-k / 2) * np.linalg.det(sigma2)**(-0.5) *\
        np.exp(-0.5 * np.sum(X @ np.linalg.pinv(sigma2) * X, axis=1))
    
    return p
    
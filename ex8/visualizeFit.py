import numpy as np
from multivariateGaussian import multivariateGaussian
import matplotlib.pyplot as plt

def visualizeFit(X,  mu, sigma2):
    #VISUALIZEFIT Visualize the dataset and its estimated distribution.
    #   VISUALIZEFIT(X, p, mu, sigma2) This visualization shows you the 
    #   probability density function of the Gaussian distribution. Each example
    #   has a location (x1, x2) that depends on its feature values.
    #

    X1, X2 = np.meshgrid(np.arange(0,30,0.5), np.arange(0,30,0.5))
    
    Z = multivariateGaussian(np.append(X1.reshape(-1,1), X2.reshape(-1,1), axis=1),mu,sigma2)
    Z = np.reshape(Z,(X1.shape))
    
    # Do not plot if there are infinities
    # np.isinf(X) 는 inf 인 element 에 해당하는 자리에 True, otherwise, False
    if(np.sum(np.isinf(Z)) == 0):
        plt.contour(X1, X2, Z, 10.0 ** np.arange(-20, 0, 3))

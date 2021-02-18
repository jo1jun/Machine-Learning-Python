from plotData_ex2 import plotData
import matplotlib.pyplot as plt
import numpy as np
from mapFeature import mapFeature

def plotDecisionBoundary(theta, X, y):
    #PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
    #the decision boundary defined by theta
    #   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
    #   positive examples and o for the negative examples. X is assumed to be 
    #   a either 
    #   1) Mx3 matrix, where the first column is an all-ones column for the 
    #      intercept.
    #   2) MxN, N>3 matrix, where the first column is all-ones
    
    # Plot Data
    plotData(X[:,1:3], y)
    
    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = [np.min(X[:,1]) - 2,  np.max(X[:,1]) + 2]  #x축 기준 가장 작은 spot - 2, 가장 큰 spot + 2 (양 끝 x 값 설정)

        # Calculate the decision boundary line
        plot_y = (-1/theta[2])*(theta[1]*plot_x + theta[0])  #(theta_0 * x_0) + (theta_1 * x_1) + (theta_2 * x_2) = 0 방정식을 풀자.

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)
        # Legend, specific for the exercise
        plt.legend(['Decision Boundary','Admitted', 'Not admitted'])
        plt.axis([30, 100, 30, 100])    #축 범위 지정
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
    
        z = np.zeros((np.size(u), np.size(v)))
        # Evaluate z = theta*x over the grid
        for i in range(np.size(u)):
            for j in range(np.size(v)):
                z[i,j] = mapFeature(u[i], v[j]) @ theta

        z = z.T # important to transpose z before calling contour
    
        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        plt.contour(u, v, z, 0, linewidths=2)    #중간에 0 은 등고선(line) 을 하나만 그리겠다는 의미. (n+1)
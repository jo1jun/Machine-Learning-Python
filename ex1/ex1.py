import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
import numpy as np
import matplotlib.pyplot as plt
from warmUpExercise import warmUpExercise
from load import load
from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent

## Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     warmUpExercise.m
#     plotData.m
#     gradientDescent.m
#     computeCost.m
#     gradientDescentMulti.m
#     computeCostMulti.m
#     featureNormalize.m
#     normalEqn.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
#

## ==================== Part 1: Basic Function ====================

# Complete warmUpExercise.py
print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')
print(warmUpExercise(), '\n')


## ======================= Part 2: Plotting =======================

print('Plotting Data ...\n')

data = load('ex1data1.txt')
X = data[:,0]
y = data[:,1]
m = y.shape[0] # number of training examples
y = y[:,np.newaxis] #y 를 1차원 tuple 에서 2차원 배열로 (m,) -> (m,1)

# Plot Data
# Note: You have to complete the code in plotData.py
plotData(X, y)

## =================== Part 3: Cost and Gradient descent ===================

X = np.append(np.ones((m,1)), data[:,0].reshape(m,1), axis = 1) # Add a column of ones to x

theta = np.zeros((2, 1), float) # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...\n')
# compute and display initial cost
J = computeCost(X,y,theta)
print('With theta = [[0],[0]]\nCost computed = ', J, '\n')
print('Expected cost value (approx) 32.07\n')

# further testing of the cost function
theta = np.array([[-1],[2]], float)
J = computeCost(X, y, theta)
print('\nWith theta = [[-1],[2]]\nCost computed = ', J, '\n')
print('Expected cost value (approx) 54.24\n')

print('\nRunning Gradient Descent ...\n')
# run gradient descent
theta, _ = gradientDescent(X, y, theta, alpha, iterations)


# print theta to screen
print('Theta found by gradient descent:\n')
print(theta, '\n')
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

# Plot the linear fit
plt.plot(X[:,1], np.dot(X,theta), label='Linear regression')
plt.legend()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([1, 3.5], theta)
print('For population = 35,000, we predict a profit of ',predict1*10000, '\n')
predict2 = np.dot([1, 7], theta)
print('For population = 70,000, we predict a profit of ', predict2*10000, '\n')


## ============= Part 4: Visualizing J(theta_0, theta_1) =============

print('Visualizing J(theta_0, theta_1) ...\n')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)     #-10 ~ 10 까지 100 개의 원소 생성
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

# Fill out J_vals
for i in range(theta0_vals.shape[0]):
    for j in range(theta1_vals.shape[0]):
        t = [[theta0_vals[i]], [theta1_vals[j]]]
        J_vals[i,j] = computeCost(X, y, t)
        

# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T

# Surface plot
# reference : https://matplotlib.org/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html?highlight=plot_surface#mpl_toolkits.mplot3d.axes3d.Axes3D.plot_surface
fig = plt.figure(figsize=(8,6))
ax3d = fig.add_subplot(111,projection='3d')
X,Y = np.meshgrid(theta0_vals,theta1_vals)
Z = J_vals
ax3d.plot_surface(Y, X, Z, cmap='plasma')
ax3d.set_xlabel('theta_0')
ax3d.set_ylabel('theta_1')

# Contour plot
# reference : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.contour.html?highlight=pyplot%20contour#matplotlib.pyplot.contour
plt.figure()
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.plot(theta[0], theta[1], 'rx', markersize=10, linewidth=2)

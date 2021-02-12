import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
import numpy as np
import matplotlib.pyplot as plt
from load import load
from featureNormalize import featureNormalize
from gradientDescentMulti import gradientDescentMulti
from normalEqn import normalEqn

## Machine Learning Online Class
#  Exercise 1: Linear regression with multiple variables
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear regression exercise. 
#
#  You will need to complete the following functions in this 
#  exericse:
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
#  For this part of the exercise, you will need to change some
#  parts of the code below for various experiments (e.g., changing
#  learning rates).
#

## Initialization

## ================ Part 1: Feature Normalization ================

print('Loading data ...\n')

## Load Data
data = load('ex1data2.txt')
X = data[:, 0:2]
y = np.array(data[:, 2])
y = y[:,np.newaxis]
m = y.shape[0]

# Print out some data points
print('First 10 examples from the dataset: \n')      

for i in range(10):
    print('x = ',X[i,:], ' y = ',y[i])

# Scale features and set them to zero mean
print('Normalizing Features ...\n');

X, mu, sigma = featureNormalize(X)

# Add intercept term to X
X = np.append(np.ones((m,1)), X, axis = 1) # Add a column of ones to x

## ================ Part 2: Gradient Descent ================

print('Running gradient descent ...\n');

# Choose some alpha value
alpha = 0.01;
num_iters = 400;

# Init Theta and Run Gradient Descent 
theta = np.zeros((3, 1));
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters);

# Plot the convergence graph

plt.plot(range(num_iters), J_history)
plt.xlabel('Number of iterations');
plt.ylabel('Cost J');

# Display gradient descent's result
print('Theta computed from gradient descent: \n');
print(theta, '\n')

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.

price = 0 # You should change this

plt.figure()
theta = np.zeros((3,1))
theta, J1 = gradientDescentMulti(X, y, theta, alpha, num_iters)
plt.plot(range(50), J1[0:50], 'b')
alpha = 0.03
theta = np.zeros((3,1))
theta, J2 = gradientDescentMulti(X, y, theta, alpha, num_iters)
plt.plot(range(50), J2[0:50], 'r')
alpha = 0.1
theta = np.zeros((3,1))
theta, J3 = gradientDescentMulti(X, y, theta, alpha, num_iters)
plt.plot(range(50), J3[0:50], 'k')
alpha = 0.3
theta = np.zeros((3,1))
theta, J4 = gradientDescentMulti(X, y, theta, alpha, num_iters)
plt.plot(range(50), J4[0:50], 'c')
alpha = 1
theta, J5 = gradientDescentMulti(X, y, theta, alpha, num_iters)
plt.plot(range(50), J5[0:50], 'g')
# 여기서 alpha 를 다시 3배하면 J 가 발산한다. 
#alpha = 3
#theta = np.zeros((3,1))
#theta, J6 = gradientDescentMulti(X, y, theta, alpha, num_iters)
#plt.plot(range(50), J6[0:50], 'y')

price = np.dot(np.array([1 ,(1650-mu[0,0])/sigma[0,0], (3-mu[0,1])/sigma[0,1]]), theta);

# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n ', price, '\n')

## ================ Part 3: Normal Equations ================

print('Solving with normal equations...\n');

## Load Data
data = load('ex1data2.txt')
X = data[:,0:2]
y = data[:, 2]
y = y[:,np.newaxis]
m = y.shape[0]

# Add intercept term to X
X = np.append(np.ones((m,1)), X, axis = 1) # Add a column of ones to x

# Calculate the parameters from the normal equation
theta = normalEqn(X, y);

# Display normal equation's result
print('Theta computed from the normal equations: \n');
print(theta, '\n');


# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
price = 0   # You should change this

price = [1, 1650, 3] @ theta

# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations) : ', price)
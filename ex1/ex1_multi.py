import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
import numpy as np
import matplotlib.pyplot as plt
from load import load
from featureNormalize import featureNormalize
from gradientDescentMulti import gradientDescentMulti
from normalEqn import normalEqn

## ================ Part 1: Feature Normalization ================

print('Loading data ...\n')

## Load Data
data = load('ex1data2.txt')
X = data[:, 0:2]
y = np.array(data[:, 2])
m = y.shape[0]
y = y[:,np.newaxis]                 #y 를 1차원 tuple 에서 2차원 배열로 (m,) -> (m,1)

# Print out some data points
print('First 10 examples from the dataset: \n')      

for i in range(10):
    print('x = ',X[i,:], ' y = ',y[i])

# Scale features and set them to zero mean
print('Normalizing Features ...\n');

X, mu, sigma = featureNormalize(X)

# Add intercept term to X
X = np.ones((m,1))
X = np.append(X, data[:,0:2], axis = 1) # Add a column of ones to x

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
alpha = 1
theta, J4 = gradientDescentMulti(X, y, theta, alpha, num_iters)
plt.plot(range(50), J4[0:50], 'g')
theta = np.zeros((3,1))
theta, J5 = gradientDescentMulti(X, y, theta, alpha, num_iters)
plt.plot(range(50), J5[0:50], 'y')
# 여기서 alpha 를 다시 3배하면 J 가 발산한다. 

price = np.dot(np.array([1 ,(1650-mu[0,0])/sigma[0,0], (3-mu[0,1])/sigma[0,1]]), theta);

# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n ', price, '\n')

## ================ Part 3: Normal Equations ================

print('Solving with normal equations...\n');

## Load Data
data = load('ex1data2.txt')
X = data[:,0:2]
y = data[:, 2]
m = y.shape[0]
y = y[:,np.newaxis]

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



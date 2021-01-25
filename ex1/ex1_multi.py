import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
import numpy as np
import matplotlib.pyplot as plt
from loadMulti import loadMulti
from featureNormalize import featureNormalize
from plotData import plotData
from computeCost import computeCost
from gradientDescentMulti import gradientDescentMulti

## ================ Part 1: Feature Normalization ================

print('Loading data ...\n')

## Load Data
data = loadMulti('ex1data2.txt')    #dataset 가 두개 뿐이니 범용성 고려하지 않았다. (귀찮아서..)
X = data[:, 0:2]
y = data[:, 2]
m = y.shape[0]

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
#rice = 0; % You should change this

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

## Machine Learning Online Class - Exercise 2: Logistic Regression
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
from load import load
import numpy as np
import matplotlib.pyplot as plt
from plotData import plotData
from costFunction import costFunction
from gradientDescentMulti import gradientDescentMulti
from plotDecisionBoundary import plotDecisionBoundary
from sigmoid import sigmoid
from predict import predict

## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.
data = load('ex2data1.txt')
X = data[:, :2]
y = data[:, 2]
y = y[:,np.newaxis]

## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')

plotData(X, y)
plt.show()

## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in 
#  costFunction.py

#  Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape

# Add intercept term to x and X_test
X = np.append(np.ones((m, 1)), X, axis=1)

# Initialize fitting parameters
initial_theta = np.zeros((n + 1, 1))

# Compute and display initial cost and gradient

cost, grad = costFunction(initial_theta, X, y)

print('Cost at initial theta (zeros): ', cost, '\n')
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): \n')
print(grad, '\n')
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

# Compute and display cost and gradient with non-zero theta
test_theta = [[-24], [0.2], [0.2]]
cost, grad = costFunction(test_theta, X, y)

print('\nCost at test theta: ', cost, '\n')
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta: \n')
print(grad, '\n')
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')


## ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.
#  python 에서 동작하는 프레임 워크를 사용.
#  미구현 그대신 gradientDescentMulti 를 이용.

'''
theta, J = gradientDescentMulti(X, y, initial_theta, alpha=0.01, num_iters=700000)   #이정도 alpha 와 반복수 이어야 얼추 비슷. (비효율적..)
# Print theta to screen
print('Cost at theta found by gradientDescentMulti: ', cost, '\n')
print('theta: \n')
print(theta, '\n')
'''

theta = np.array([[-25.161272],[0.206233],[0.201470]])  #   fminuc 으로 학습한 theta 값.

# Plot Boundary
plotDecisionBoundary(theta, X, y)
plt.show()

## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and 
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of 
#  our model.
#
#  Your task is to complete the code in predict.py

#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 

prob = sigmoid([1, 45, 85] @ theta)
print('For a student with scores 45 and 85, we predict an admission probability of ', prob);
print('Expected value: 0.775 +/- 0.002\n\n');

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: ', np.mean(p == y) * 100, '\n')
print('Expected accuracy (approx): 89.0\n')


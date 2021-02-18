## Machine Learning Online Class - Exercise 2: Logistic Regression
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
from load import load
import numpy as np
import matplotlib.pyplot as plt
from plotData_ex2 import plotData
from mapFeature import mapFeature
from costFunctionReg import costFunctionReg
from plotDecisionBoundary import plotDecisionBoundary
from predict import predict

## Load Data
# The first two columns contains the X values and the third column
# contains the label (y).

data = load('ex2data2.txt')
X = data[:, :2]
y = data[:, 2]

plotData(X, y)

plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

# Specified in plot order
plt.legend(['y = 1', 'y = 0'])

## =========== Part 1: Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic
#  regression to classify the data points.
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).
#
print('==================== Part 1: Regularized Logistic Regression ====================')

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled

X = mapFeature(X[:,0], X[:,1])

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
_lambda = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
cost, grad = costFunctionReg(initial_theta, X, y, _lambda)

print('Cost at initial theta (zeros): ', cost, '\n')
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros) - first five values only:\n')
print(grad[:5],'\n')
print('Expected gradients (approx) - first five values only:\n')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')


# Compute and display cost and gradient
# with all-ones theta and lambda = 10
test_theta = np.ones(X.shape[1])
cost, grad = costFunctionReg(test_theta, X, y, 10)

print('\nCost at test theta (with lambda = 10): ', cost, '\n')
print('Expected cost (approx): 3.16\n')
print('Gradient at test theta - first five values only:\n')
print(grad[:5],'\n')
print('Expected gradients (approx) - first five values only:\n')
print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n')


## ============= Part 2: Regularization and Accuracies =============
#  Optional Exercise:
#  In this part, you will get to try different values of lambda and
#  see how regularization affects the decision coundart
#
#  Try the following values of lambda (0, 1, 10, 100).
#
#  How does the decision boundary change when you vary lambda? How does
#  the training set accuracy vary?
#
print('==================== Part 2: Regularization and Accuracies ====================')


# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1 (you should vary this)
_lambda = 1

#  octave 의 fminunc 대신, python 에서 동작하는 프레임 워크(scipy.optimize.minimize)를 활용
#  reference : https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

import scipy.optimize as op

print ('Executing minimize function...\n')

result = op.minimize(fun=costFunctionReg,x0=initial_theta,args=(X,y,_lambda),jac=True,options={'maxiter':400})
#입출력 theta 는 반드시 평탄화가 되어있어야 한다.
#costFunction 내부에서 평탄화를 다시 reshaping 하면 된다.
#argument 로 method = '특정 method' 로 지정할 수도 있다.
#If jac is a Boolean and is True, fun(costFunction) is assumed to return and objective and gradient as an (f, g) tuple.

# Plot Boundary
plotDecisionBoundary(result.x, X, y)

plt.title('lambda = 1')

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

plt.legend(['y = 1', 'y = 0', 'Decision boundary'])

# Compute accuracy on our training set
p = predict(result.x, X)

print('lambda = 1\n')
print('Train Accuracy: ', float(np.mean((p == y))) * 100, '\n')
print('Expected accuracy (with lambda = 1): 83.1 (approx)\n')

# No regularization (Overfitting) (λ = 0)
_lambda = 0
result = op.minimize(fun=costFunctionReg,x0=initial_theta,args=(X,y,_lambda),jac=True,options={'maxiter':400})
plotDecisionBoundary(result.x, X, y)

plt.title('lambda = 0')

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(['y = 1', 'y = 0', 'Decision boundary'])

# Compute accuracy on our training set
p = predict(result.x, X)

print('lambda = 0\n')
print('Train Accuracy: ', float(np.mean((p == y))) * 100, '\n')

#Too much regularization (Underfitting) (λ = 100)
_lambda = 100
result = op.minimize(fun=costFunctionReg,x0=initial_theta,args=(X,y,_lambda),jac=True,options={'maxiter':400})
plotDecisionBoundary(result.x, X, y)

plt.title('lambda = 100')

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(['y = 1', 'y = 0', 'Decision boundary'])

# Compute accuracy on our training set
p = predict(result.x, X)

print('lambda = 100\n')
print('Train Accuracy: ', float(np.mean((p == y))) * 100, '\n')

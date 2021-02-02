import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from linearRegCostFunction import linearRegCostFunction
from trainLinearReg import trainLinearReg
from learningCurve import learningCurve
from polyFeatures import polyFeatures
from featureNormalize import featureNormalize
from plotFit import plotFit
## Machine Learning Online Class
#  Exercise 5 | Regularized Linear Regression and Bias-Variance
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     linearRegCostFunction.m
#     learningCurve.m
#     validationCurve.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')

# Load from ex5data1: 
# You will have X, y, Xval, yval, Xtest, ytest in your environment
mat = scipy.io.loadmat('ex5data1.mat')
X, y = mat['X'], mat['y']
Xval, yval = mat['Xval'], mat['yval']
Xtest, ytest = mat['Xtest'], mat['ytest']

#print(X.shape) #(12,1)
#print(y.shape) #(12,1)
#print(Xval.shape) #(21,1)
#print(yval.shape) #(21,1)

# m = Number of examples
m = X.shape[0]

# Plot training data
plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')


## =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear 
#  regression. 
#

theta = np.array([1,1])

J,_ = linearRegCostFunction(np.insert(X,0,1,axis=1), y, theta, 1)

print('Cost at theta = [1 ; 1]: {} \n(this value should be about 303.993192)\n'.format(J))


## =========== Part 3: Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear 
#  regression.
#

theta = np.array([1,1])
J, grad = linearRegCostFunction(np.insert(X,0,1,axis=1), y, theta, 1)

print('Gradient at theta = [1 ; 1]:  [{0}; {1}] \
      \n(this value should be about [-15.303016; 598.250744])\n'.format(grad[0], grad[1]))

## =========== Part 4: Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train 
#  regularized linear regression.
# 
#  Write Up Note: The data is non-linear, so this will not give a great 
#                 fit.
#

#  Train linear regression with lambda = 0
_lambda = 0
theta = trainLinearReg(np.insert(X,0,1,axis=1), y, _lambda)

#  Plot fit over the data
plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')

plt.plot(X, np.insert(X,0,1,axis=1) @ theta, '--', 'LineWidth', 2)



## =========== Part 5: Learning Curve for Linear Regression =============
#  Next, you should implement the learningCurve function. 
#
#  Write Up Note: Since the model is underfitting the data, we expect to
#                 see a graph with "high bias" -- Figure 3 in ex5.pdf 
#

error_train, error_val = \
    learningCurve(np.insert(X,0,1,axis=1), y, np.insert(Xval,0,1,axis=1), yval, _lambda)

plt.figure()
plt.plot(list(range(m)), error_train)
plt.plot(list(range(m)), error_val)
plt.title('Learning curve for linear regression')
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 150])

print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
    print('  \t{0}\t\t{1}\t{2}\n'.format(i, error_train[i], error_val[i]))


## =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers
#

p = 8

#octave 의 bsxfun 은 numpy 의 broadcast 이다. numpy 개발자분들 thank you!!!

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize(X_poly)   # Normalize
X_poly = np.insert(X_poly,0,1,axis=1)          # Add Ones

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test / sigma
X_poly_test = np.insert(X_poly_test,0,1,axis=1) # Add Ones

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val / sigma
X_poly_val = np.insert(X_poly_val,0,1,axis=1)   # Add Ones

print('Normalized Training Example 1:\n');
print('  {}  \n'.format(X_poly[0, :].reshape(-1,1)))

## =========== Part 7: Learning Curve for Polynomial Regression =============
#  Now, you will get to experiment with polynomial regression with multiple
#  values of lambda. The code below runs polynomial regression with 
#  lambda = 0. You should try running the code with different values of
#  lambda to see how the fit and learning curve change.
#

_lambda = 0
theta = trainLinearReg(X_poly, y, _lambda)

# Plot training data and fit
plt.figure()
plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plotFit(np.min(X), np.max(X), mu, sigma, theta, p)
plt.xlabel('Change in water level (x)');
plt.ylabel('Water flowing out of the dam (y)');
plt.title('Polynomial Regression Fit (lambda = {})'.format(_lambda))

plt.figure()
error_train, error_val = \
    learningCurve(X_poly, y, X_poly_val, yval, _lambda)
plt.plot(list(range(m)), error_train)
plt.plot(list(range(m)), error_val)

plt.title('Polynomial Regression Learning Curve (lambda = {})'.format(_lambda))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 100])
plt.legend(['Train', 'Cross Validation'])

print('Polynomial Regression (lambda = {})\n\n'.format(_lambda))
print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
    print('  \t{0}\t\t{1}\t{2}\n'.format(i, error_train[i], error_val[i]))
          
    
## =========== Part 8: Validation for Selecting Lambda =============
#  You will now implement validationCurve to test various values of 
#  lambda on a validation set. You will then use this to select the
#  "best" lambda value.
#

'''
[lambda_vec, error_train, error_val] = ...
    validationCurve(X_poly, y, X_poly_val, yval);

close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' #f\t#f\t#f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

'''
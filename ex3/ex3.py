import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
import numpy as np
import scipy.io
from displayData import displayData
from lrCostFunction import lrCostFunction
from oneVsAll import oneVsAll
from predictOneVsAll import predictOneVsAll

## Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     lrCostFunction.m (logistic regression cost function)
#     oneVsAll.m
#     predictOneVsAll.m
#     predict.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Setup the parameters you will use for this part of the exercise
input_layer_size  = 400   # 20x20 Input Images of Digits
num_labels = 10           # 10 labels, from 1 to 10
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')

#MATLAB 파일 읽기
mat = scipy.io.loadmat('ex3data1.mat') # training data stored in arrays X, y
X, y = mat['X'], mat['y']
#print(X.shape)
#print(y.shape)

print("'y' shape: ", mat['y'].shape, "Unique elements in y: " ,np.unique(mat['y']))
print("'X' shape: ",X.shape,"X[0] shape: ",X[0].shape)

m = X.shape[0]

# Randomly select 100 data points to display
rand_indices = np.random.choice(m,100,replace=False)    #비복원 추출
sel = X[rand_indices, :]

displayData(sel)

## ============ Part 2a: Vectorize Logistic Regression ============
#  In this part of the exercise, you will reuse your logistic regression
#  code from the last exercise. You task here is to make sure that your
#  regularized logistic regression implementation is vectorized. After
#  that, you will implement one-vs-all classification for the handwritten
#  digit dataset.
#

# Test case for lrCostFunction
print('\nTesting lrCostFunction() with regularization')

theta_t = np.array([-2, -1, 1, 2])

X_t = np.append(np.ones((5,1)) , np.linspace(1,15,15).reshape((3,5)).T / 10 , axis=1)
y_t = np.asarray(np.array([[1],[0],[1],[0],[1]]) >= 0.5, int)

lambda_t = 3

J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)

print('\nCost: ',J,'\n')
print('Expected cost: 2.534819\n')
print('Gradients:\n')
print(grad, '\n')
print('Expected gradients:\n')
print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n')


## ============ Part 2b: One-vs-All Training ============
print('\nTraining One-vs-All Logistic Regression...\n')

_lambda = 0.1
all_theta = oneVsAll(X, y, num_labels, _lambda)

## ================ Part 3: Predict for One-Vs-All ================

pred = predictOneVsAll(all_theta, X)
#print(y.shape)
#print(pred.shape)
pred = pred[:,np.newaxis]   #y 와 동일한 shape 으로 변형.

print('\nTraining Set Accuracy: ', np.mean(pred == y) * 100,'\n')

#minimize 의 method 를 default(BFGS) 로 두면 93.24 가 나온다.
#method 를 Newton-CG 로 바꾸니 너무 오래걸리는 듯.
#method 를 CG 로 바꾸니 빠르고 95.16 으로 더 정확하게 나온다.
#method 를 L-BFGS-B 로 바꾸니 더 빠르고 96.16 으로 더 정확하게 나온다.
#여러가지 method reference : https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

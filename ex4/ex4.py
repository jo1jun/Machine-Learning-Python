import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from displayData import displayData
from nnCostFunction import nnCostFunction
from sigmoidGradient import sigmoidGradient
from randInitializeWeights import randInitializeWeights
from checkNNGradients import checkNNGradients
import scipy.optimize as op
from predict import predict

## Machine Learning Online Class - Exercise 4 Neural Network Learning

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     sigmoidGradient.m
#     randInitializeWeights.m
#     nnCostFunction.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                         # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')

mat = scipy.io.loadmat('ex4data1.mat')
X, y = mat['X'], mat['y']
#print(X.shape)
#print(y.shape)

m = X.shape[0]

# Randomly select 100 data points to display
rand_indices = np.random.choice(m,100,replace=False)    #비복원 추출
sel = X[rand_indices, :]

plt.figure()
displayData(sel)

## ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('\nLoading Saved Neural Network Parameters ...\n')

# Load the weights into variables Theta1 and Theta2
mat = scipy.io.loadmat('ex4weights.mat')

Theta1 = mat['Theta1']
Theta2 = mat['Theta2']

print('Theta1.shape : ', Theta1.shape)
print('Theta2.shape : ', Theta2.shape)

# Unroll parameters
nn_params = np.append(Theta1.reshape(-1), Theta2.reshape(-1))   # -1 : 나머지 원소들로 알아서.
#print(nn_params.shape)

## ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.m to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#
print('\nFeedforward Using Neural Network ...\n')

# Weight regularization parameter (we set this to 0 here).
_lambda = 0

J,_ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)

print('Cost at parameters (loaded from ex4weights): %f\n(this value should be about 0.287629)\n' %J)

## =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
#

print('\nChecking Cost Function (w/ Regularization) ... \n')

# Weight regularization parameter (we set this to 1 here).
_lambda = 1

J,_ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)

print('Cost at parameters (loaded from ex4weights): %f\n(this value should be about 0.383770)\n'% J)


## ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#

print('\nEvaluating sigmoid gradient...\n')

g = sigmoidGradient([-1, -0.5, 0, 0.5, 1])
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ')
print(g)
print('\n\n')


## ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.m)

print('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.append(initial_Theta1.reshape(-1), initial_Theta2.reshape(-1))


## =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.m to return the partial
#  derivatives of the parameters.
#
print('\nChecking Backpropagation... \n')

#  Check gradients by running checkNNGradients
_lambda = 0
checkNNGradients(_lambda)

## =============== Part 8: Implement Regularization ===============
#  Once your backpropagation implementation is correct, you should now
#  continue to implement the regularization with the cost and gradient.
#

print('\nChecking Backpropagation (w/ Regularization) ... \n')

#  Check gradients by running checkNNGradients
_lambda = 3
checkNNGradients(_lambda)

# Also output the costFunction debugging values
debug_J,_  = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)

print('\n\nCost at (fixed) debugging parameters (w/ lambda = ',_lambda,'): ',debug_J,' '\
         '\n(for lambda = 3, this value should be about 0.576051)\n\n')


## =================== Part 8: Training NN ===================
#  You have now implemented all the code necessary to train a neural 
#  network. To train your neural network, we will now use "fmincg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.
#
print('\nTraining Neural Network... \n')

#  After you have completed the assignment, change the MaxIter to a larger
#  value to see how more training helps.
#  options = optimset('MaxIter', 50)

#  You should also try different values of lambda
_lambda = 1

# Create "short hand" for the cost function to be minimized
costFunction = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)

# Now, costFunction is a function that takes in only one argument (the
# neural network parameters)
result = op.minimize(fun=costFunction,x0=initial_nn_params,method='L-BFGS-B',jac=True,options={'maxiter':50})

# Obtain Theta1 and Theta2 back from nn_params
Theta1 = result.x[:(input_layer_size + 1) * hidden_layer_size] #bias 고려하여 +1
Theta2 = result.x[np.size(Theta1):]
Theta1 = Theta1.reshape(hidden_layer_size, -1)
Theta2 = Theta2.reshape(num_labels, -1)

## ================= Part 9: Visualize Weights =================
#  You can now "visualize" what the neural network is learning by 
#  displaying the hidden units to see what features they are capturing in 
#  the data.

print('\nVisualizing Neural Network... \n')

plt.figure()
displayData(Theta1[:,1:])

## ================= Part 10: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X)
pred = pred.reshape(-1,1)

print('\nTraining Set Accuracy: ', np.mean((pred == y)) * 100, '\n')
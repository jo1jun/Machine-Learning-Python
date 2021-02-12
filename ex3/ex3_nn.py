import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from displayData import displayData
from predict_nn import predict

## Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks

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

#MATLAB 파일 읽기
mat = scipy.io.loadmat('ex3data1.mat') # training data stored in arrays X, y
X, y = mat['X'], mat['y']

m = X.shape[0]

# Randomly select 100 data points to display
rand_indices = np.random.choice(m,100,replace=False)    #비복원 추출
sel = X[rand_indices, :]

displayData(sel)

## ================ Part 2: Loading Pameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('\nLoading Saved Neural Network Parameters ...\n')

# Load the weights into variables Theta1 and Theta2
mat = scipy.io.loadmat('ex3weights.mat')

Theta1 = mat['Theta1']
Theta2 = mat['Theta2']

print('Theta1.shape : ', Theta1.shape)
print('Theta2.shape : ', Theta2.shape)
## ================= Part 3: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X)
pred = pred.reshape(-1,1)

print('\nTraining Set Accuracy: ', np.mean(pred==y) * 100, '\n')

#  To give you an idea of the network's output, you can also run
#  through the examples one at the a time to see what it is predicting.

#  Randomly permute examples
rp = np.random.choice(m,m,False)    #비복원 추출
for i in range(5):
    # Display 
    print('\nDisplaying Example Image\n')
    x = X[rp[i],:].reshape((1,X.shape[1]))
    plt.figure()
    displayData(x)
    pred = predict(Theta1, Theta2, x)
    plt.title('pred : %s' %(pred%10))
    print('\nNeural Network Prediction: ',pred%10, ' (digit ',y[rp[i]]%10,')\n')
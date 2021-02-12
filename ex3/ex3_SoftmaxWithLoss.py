import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
import numpy as np
import scipy.io
from trainSoftmaxWithLoss import trainSoftmaxWithLoss
from predictSoftmaxWithLoss import predictSoftmaxWithLoss

input_layer_size  = 400
num_labels = 10

print('Loading and Visualizing Data ...\n')

mat = scipy.io.loadmat('ex3data1.mat')
X, y = mat['X'], mat['y']

m = X.shape[0]

#Deep Learning from Scratch 에서 학습했던 Softmax & cross entropy error 를 활용하면 단 한 번의 loop 로  학습이 가능!
print('Training Affine-Softmax-crossEntropy Logistic Regression...\n')

_lambda = 0.1

X = np.append(np.ones((m,1)), X, axis = 1)
theta = trainSoftmaxWithLoss(X, y, num_labels, _lambda)

## ================ Predict for Affine-Softmax-crossEntropy ================

pred = predictSoftmaxWithLoss(theta, X)
pred = pred[:,np.newaxis]   #y 와 동일한 shape 으로 변형.

print('\nTraining Set Accuracy: ', np.mean(pred == y) * 100,'\n')

#wow!! 매우 빠르다!! accuracy(97) 도 좋고!!
#maxiter 을 500 으로 하면 정확도가 98.84 까지 나온다. (그러나 test set 으로 predict 하지 않음 검증이 필요)
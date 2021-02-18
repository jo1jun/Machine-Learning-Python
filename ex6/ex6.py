import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
import scipy.io
from plotData import plotData
import sklearn.svm as svm
from visualizeBoundaryLinear import visualizeBoundaryLinear
import matplotlib.pyplot as plt
from gaussianKernel import gaussianKernel
from visualizeBoundary import visualizeBoundary
from dataset3Params import dataset3Params
## Machine Learning Online Class
#  Exercise 6 | Support Vector Machines
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     gaussianKernel.m
#     dataset3Params.m
#     processEmail.m
#     emailFeatures.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## =============== Part 1: Loading and Visualizing Data ================
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#
print('=============== Part 1: Loading and Visualizing Data ================')
print('Loading and Visualizing Data ...\n')

# Load from ex6data1: 
# You will have X, y in your environment
mat = scipy.io.loadmat('ex6data1.mat')
X, y = mat['X'], mat['y'].flatten()
#print(X.shape) #(51, 2)
#print(y.shape) #(51,)

# Plot training data
plotData(X, y)

## ==================== Part 2: Training Linear SVM ====================
#  The following code will train a linear SVM on the dataset and plot the
#  decision boundary learned.
#
print('==================== Part 2: Training Linear SVM ====================')
# Load from ex6data1: 
# You will have X, y in your environment
mat = scipy.io.loadmat('ex6data1.mat')
X, y = mat['X'], mat['y'].flatten()


print('Training Linear SVM ...')

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)
# sklearn 을 사용해서 구현
# reference : https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

C = 1
classifier = svm.SVC(C=C, kernel='linear', tol=1e-3)
model = classifier.fit(X, y)
#.fit(X, y[, sample_weight]) Fit the SVM model according to the given training data.

plt.figure()
plt.title('C = 1')
visualizeBoundaryLinear(X, y, model)
#C 가 크지 않은 경우, margin 을 크게 남기려 하는 경향이 적다. 따라서 outlier 를 크게 신경쓰지 않는다.

C = 100
classifier = svm.SVC(C=C, kernel='linear', tol=1e-3)
model = classifier.fit(X, y)

plt.figure()
plt.title('C = 100')
visualizeBoundaryLinear(X, y, model)
#C 가 매우 큰 경우, margin 을 크게 남겨야 비용함수의 크기를 줄일 수 있기 때문에 outlier 를 크게 신경쓴다.

## =============== Part 3: Implementing Gaussian Kernel ===============
#  You will now implement the Gaussian kernel to use
#  with the SVM. You should complete the code in gaussianKernel.m
#
print('=============== Part 3: Implementing Gaussian Kernel ===============')
print('\nEvaluating the Gaussian Kernel ...\n')

x1 = [1, 2, 1] 
x2 = [0, 4, -1] 
sigma = 2
sim = gaussianKernel(x1, x2, sigma)

print('Gaussian Kernel between x1 = [1 2 1], x2 = [0 4 -1], sigma = {} :' \
         '\n\t{}\n(for sigma = 2, this value should be about 0.324652)\n'.format(sigma, sim))


## =============== Part 4: Visualizing Dataset 2 ================
#  The following code will load the next dataset into your environment and 
#  plot the data. 
#
print('=============== Part 4: Visualizing Dataset 2 ================')
print('Loading and Visualizing Data ...\n')

# Load from ex6data2: 
# You will have X, y in your environment
mat = scipy.io.loadmat('ex6data2.mat')
X, y = mat['X'], mat['y'].flatten()   #X.shape = (863,2) , y.shape = (863,)
# Plot training data
plt.figure()
plotData(X, y)


## ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
#  After you have implemented the kernel, we can now use it to train the 
#  SVM classifier.
# 
print('========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========')
print('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n')

# Load from ex6data2: 
# You will have X, y in your environment
mat = scipy.io.loadmat('ex6data2.mat')
X, y = mat['X'], mat['y'].flatten()   #X.shape = (863,2) , y.shape = (863,)

# SVM Parameters
C = 1 
sigma = 0.1

# We set the tolerance and max_passes lower here so that the code will run
# faster. However, in practice, you will want to run the training to
# convergence.

# kernel='rbf' 는 exp(-gamma*||x-x'||^2) 을 따른다. 따라서 gamma 만 가우시안 커널 공식에 맞게 맞춰주면 된다.
# reference : https://scikit-learn.org/stable/modules/svm.html 에서 kernel function 부분

g = 1 / (2 * sigma ** 2)
classifier = svm.SVC(C=C, kernel='rbf', tol=1e-3, gamma = g)
model = classifier.fit(X, y)
visualizeBoundary(X, y, model)

## =============== Part 6: Visualizing Dataset 3 ================
#  The following code will load the next dataset into your environment and 
#  plot the data. 
#
print('=============== Part 6: Visualizing Dataset 3 ================')
print('Loading and Visualizing Data ...\n')

# Load from ex6data3: 
# You will have X, y in your environment
mat = scipy.io.loadmat('ex6data3.mat')
X, y = mat['X'], mat['y'].flatten()   #X.shape = (211,2) , y.shape = (211,)
# Plot training data
plt.figure()
plotData(X, y)

## ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

#  This is a different dataset that you can use to experiment with. Try
#  different values of C and sigma here.
# 
print('========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========')
# Load from ex6data3: 
# You will have X, y in your environment
mat = scipy.io.loadmat('ex6data3.mat')
X, y = mat['X'], mat['y'].flatten()   #X.shape = (211,2) , y.shape = (211,)
Xval, yval = mat['Xval'], mat['yval'].flatten()   #Xval.shape = (200,2) , yval.shape = (200,)

# Try different SVM Parameters here
C, sigma = dataset3Params(X, y, Xval, yval)

# Train the SVM
classifier = svm.SVC(C=C, kernel='rbf', tol=1e-3, gamma = g)
model = classifier.fit(X, y)
visualizeBoundary(X, y, model)
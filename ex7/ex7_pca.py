import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from featureNormalize import featureNormalize
from pca import pca
from drawLine import drawLine
from projectData import projectData
from recoverData import recoverData
from displayData import displayData
import imageio
from kMeansInitCentroids import kMeansInitCentroids
from runkMeans import runkMeans
from plotDataPoints import plotDataPoints
## Machine Learning Online Class
#  Exercise 7 | Principle Component Analysis and K-Means Clustering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     pca.m
#     projectData.m
#     recoverData.m
#     computeCentroids.m
#     findClosestCentroids.m
#     kMeansInitCentroids.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easily to
#  visualize
#
print('================== Part 1: Load Example Dataset  ===================')
print('Visualizing example dataset for PCA.\n\n')

#  The following command loads the dataset. You should now have the 
#  variable X in your environment
mat = scipy.io.loadmat('ex7data1.mat')
X = mat['X']

#  Visualize the example dataset
plt.plot(X[:, 0], X[:, 1], 'bo')
plt.axis([0.5, 6.5, 2, 8])

## =============== Part 2: Principal Component Analysis ===============
#  You should now implement PCA, a dimension reduction technique. You
#  should complete the code in pca.m
#
print('================== Part 2: Principal Component Analysis ===================')
print('\nRunning PCA on example dataset.\n\n')

#  Before running PCA, it is important to first normalize X
X_norm, mu, sigma = featureNormalize(X)

#  Run PCA
U, S = pca(X_norm)

#  Compute mu, the mean of the each feature

#  Draw the eigenvectors centered at mean of data. These lines show the
#  directions of maximum variations in the dataset.
varargin = {'fmt' : '-k', 'linewidth' : 2}
drawLine(mu, mu + 1.5 * S[0] * U[:,0].T, varargin)
drawLine(mu, mu + 1.5 * S[1] * U[:,1].T, varargin)

print('Top eigenvector: \n')
print(' U[:,0] = {} {} \n'.format(U[0,0], U[1,0]))
print('\n(you should expect to see -0.707107 -0.707107)\n')

## =================== Part 3: Dimension Reduction ===================
#  You should now implement the projection step to map the data onto the 
#  first k eigenvectors. The code will then plot the data in this reduced 
#  dimensional space.  This will show you what the data looks like when 
#  using only the corresponding eigenvectors to reconstruct it.
#
#  You should complete the code in projectData.m
#
print('================== Part 3: Dimension Reduction ===================')
print('\nDimension reduction on example dataset.\n\n')

#  Plot the normalized dataset (returned from pca)
plt.figure()
plt.plot(X_norm[:, 0], X_norm[:, 1], 'bo')
plt.axis([-4 ,3 ,-4 ,3])

#  Project the data onto K = 1 dimension
K = 1
Z = projectData(X_norm, U, K)
print('Projection of the first example: ', Z[0], '\n')
print('\n(this value should be about 1.481274)\n\n')

X_rec  = recoverData(Z, U, K)
print('Approximation of the first example: {} {}\n'.format(X_rec[0, 0], X_rec[0, 1]))
print('\n(this value should be about  -1.047419 -1.047419)\n\n')

#  Draw lines connecting the projected points to the original pointshold on
plt.plot(X_rec[:, 0], X_rec[:, 1], 'ro')
varargin = {'fmt' : '--k', 'linewidth' : 1}
for i in range(X_norm.shape[0]):
    drawLine(X_norm[i,:], X_rec[i,:], varargin)


## =============== Part 4: Loading and Visualizing Face Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment
#
print('================== Part 4: Loading and Visualizing Face Data ===================')
print('\nLoading face dataset.\n\n')

#  Load Face dataset
mat = scipy.io.loadmat('ex7faces.mat')
X = mat['X']

#  Display the first 100 faces in the dataset
plt.figure()
displayData(X[:100, :])

## =========== Part 5: PCA on Face Data: Eigenfaces  ===================
#  Run PCA and visualize the eigenvectors which are in this case eigenfaces
#  We display the first 36 eigenfaces.
#
print('================== Part 5: PCA on Face Data: Eigenfaces ===================')
print('\nRunning PCA on face dataset.\n' \
         '(this might take a minute or two ...)\n\n')

#  Before running PCA, it is important to first normalize X by subtracting 
#  the mean value from each feature
X_norm, mu, sigma = featureNormalize(X)

#  Run PCA
U, S = pca(X_norm)

#  Visualize the top 36 eigenvectors found
plt.figure()
displayData(U[:, :36].T)


## ============= Part 6: Dimension Reduction for Faces =================
#  Project images to the eigen space using the top k eigenvectors 
#  If you are applying a machine learning algorithm 
print('================== Part 6: Dimension Reduction for Faces ===================')
print('\nDimension reduction for face dataset.\n\n')

K = 100
Z = projectData(X_norm, U, K)

print('The projected data Z has a size of: ')
print(np.size(Z))

## ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
#  Project images to the eigen space using the top K eigen vectors and 
#  visualize only using those K dimensions
#  Compare to the original input, which is also displayed

print('\nVisualizing the projected (reduced dimension) faces.\n\n')

K = 100
X_rec  = recoverData(Z, U, K)

# Display normalized data
plt.figure()
plt.subplot(1, 2, 1)
displayData(X_norm[:100,:])
plt.title('Original faces')

# Display reconstructed data from only k eigenfaces
plt.subplot(1, 2, 2)
displayData(X_rec[:100,:])
plt.title('Recovered faces')

## === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
#  One useful application of PCA is to use it to visualize high-dimensional
#  data. In the last K-Means exercise you ran K-Means on 3-dimensional 
#  pixel colors of an image. We first visualize this output in 3D, and then
#  apply PCA to obtain a visualization in 2D.

A = imageio.imread('bird_small.png')

# If imread does not work for you, you can try instead
#   load ('bird_small.mat')

A = A / 255
img_size = A.shape
X = np.reshape(A, (img_size[0] * img_size[1], 3))
K = 16 
max_iters = 10
initial_centroids = kMeansInitCentroids(X, K)
centroids, idx = runkMeans(X, initial_centroids, max_iters)

#  Sample 1000 random indexes (since working with all the data is
#  too expensive. If you have a fast computer, you may increase this.
sel = np.floor(np.random.rand(1000, 1) * X.shape[0])
sel = sel.flatten().astype(int) #index 로 활용하기 위한 전처리

# 3d scatter reference : https://matplotlib.org/3.2.1/gallery/mplot3d/scatter3d.html#sphx-glr-gallery-mplot3d-scatter3d-py
# 3d scatter's parameters : https://matplotlib.org/3.2.1/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html?highlight=axes3d#mpl_toolkits.mplot3d.axes3d.Axes3D
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X[sel,0],X[sel,1],X[sel,2],s=10,c=idx[sel], cmap='gist_rainbow')

## === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
# Use PCA to project this cloud to 2D for visualization

# Subtract the mean to use PCA
X_norm, mu, sigma = featureNormalize(X)

# PCA and project the data to 2D
U, S = pca(X_norm)
Z = projectData(X_norm, U, 2)

# Plot in 2D
plt.figure()
plotDataPoints(Z[sel, :], idx[sel], K)
plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')


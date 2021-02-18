import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from cofiCostFunc import cofiCostFunc
from checkCostFunction import checkCostFunction
from loadMovieList import loadMovieList
from normalizeRatings import normalizeRatings
import scipy.optimize as op

## Machine Learning Online Class
#  Exercise 8 | Anomaly Detection and Collaborative Filtering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     estimateGaussian.m
#     selectThreshold.m
#     cofiCostFunc.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## =============== Part 1: Loading movie ratings dataset ================
#  You will start by loading the movie ratings dataset to understand the
#  structure of the data.
#  
print('=============== Part 1: Loading movie ratings dataset ================')
print('Loading movie ratings dataset.\n\n')

#  Load data
mat = scipy.io.loadmat('ex8_movies.mat')
R, Y = mat['R'], mat['Y']

#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 
#  943 users
#
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i

#  From the matrix, we can compute statistics like average rating.
print('Average rating for movie 1 (Toy Story): {} / 5\n\n'\
        .format(np.mean(Y[0, R[0, :]])))

#  We can "visualize" the ratings matrix by plotting it with imagesc
plt.imshow(Y)
plt.ylabel('Movies')
plt.xlabel('Users')


## ============ Part 2: Collaborative Filtering Cost Function ===========
#  You will now implement the cost function for collaborative filtering.
#  To help you debug your cost function, we have included set of weights
#  that we trained on that. Specifically, you should complete the code in 
#  cofiCostFunc.m to return J.
print('============ Part 2: Collaborative Filtering Cost Function ===========')
#  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
mat = scipy.io.loadmat('ex8_movieParams.mat')
num_features, num_movies, num_users = mat['num_features'], mat['num_movies'], mat['num_users']
Theta, X = mat['Theta'], mat['X']

#  Reduce the data set size so that this aruns faster
num_users = 4
num_movies = 5
num_features = 3
X = X[:num_movies, :num_features]
Theta = Theta[:num_users, :num_features]
Y = Y[:num_movies, :num_users]
R = R[:num_movies, :num_users]

#  Evaluate cost function
J,_ = cofiCostFunc(np.append(X.flatten(), Theta.flatten()), Y, R, num_users, num_movies, \
               num_features, 0)
           
print('Cost at loaded parameters: {} '\
         '\n(this value should be about 22.22)\n'.format(J))

## ============== Part 3: Collaborative Filtering Gradient ==============
#  Once your cost function matches up with ours, you should now implement 
#  the collaborative filtering gradient function. Specifically, you should 
#  complete the code in cofiCostFunc.m to return the grad argument.
#  

print('============== Part 3: Collaborative Filtering Gradient ==============')

#  Check gradients by running checkNNGradients
checkCostFunction()

## ========= Part 4: Collaborative Filtering Cost Regularization ========
#  Now, you should implement regularization for the cost function for 
#  collaborative filtering. You can implement it by adding the cost of
#  regularization to the original cost computation.
#  
print('======= Part 4: Collaborative Filtering Cost Regularization ======')
#  Evaluate cost function
J,_ = cofiCostFunc(np.append(X.flatten(), Theta.flatten()), Y, R, num_users, num_movies,\
               num_features, 1.5)
           
print('Cost at loaded parameters (lambda = 1.5): {} '\
         '\n(this value should be about 31.34)\n'.format(J))



## ======= Part 5: Collaborative Filtering Gradient Regularization ======
#  Once your cost matches up with ours, you should proceed to implement 
#  regularization for the gradient. 
#
print('======= Part 5: Collaborative Filtering Gradient Regularization ======')
#
print('\nChecking Gradients (with regularization) ... \n')

#  Check gradients by running checkNNGradients
checkCostFunction(1.5)



## ============== Part 6: Entering ratings for a new user ===============
#  Before we will train the collaborative filtering model, we will first
#  add ratings that correspond to a new user that we just observed. This
#  part of the code will also allow you to put in your own ratings for the
#  movies in our dataset!
#
print('============== Part 6: Entering ratings for a new user ===============')
movieList = loadMovieList()

#  Initialize my ratings
my_ratings = np.zeros(1682)

# Check the file movie_idx.txt for id of each movie in our dataset
# For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings[0] = 4

# Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings[97] = 2

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[6] = 3
my_ratings[11]= 5
my_ratings[53] = 4
my_ratings[63]= 5
my_ratings[65]= 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354]= 5

print('\n\nNew user ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0: 
        print('Rated {} for {}\n'.format(my_ratings[i], movieList[i]))


## ================== Part 7: Learning Movie Ratings ====================
#  Now, you will train the collaborative filtering model on a movie rating 
#  dataset of 1682 movies and 943 users
#
print('============== Part 7: Learning Movie Ratings ===============')
print('\nTraining collaborative filtering...\n')

#  Load data
mat = scipy.io.loadmat('ex8_movies.mat')
R, Y = mat['R'], mat['Y']
#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 
#  943 users
#
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i

#  Add our own ratings to the data matrix
Y = np.append(my_ratings.reshape(-1,1), Y, axis=1)
R = np.append((my_ratings != 0).reshape(-1,1), R, axis=1)

#  Normalize Ratings
# Mean Normalization 을 위한 작업.
Ynorm, Ymean = normalizeRatings(Y, R)

#  Useful Values
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10

# Set Initial Parameters (Theta, X)

X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)
'''
X = np.zeros((num_movies, num_features)) + 1
Theta = np.zeros((num_users, num_features)) + 2
'''
# 전부 0으로 초기화 하면 갱신이 이루어지지 않는다.
# 각각 혹은 전체가 전부 동일한 값이면 갱신이 원활하게 이루어지지 않는다.
# 초기값을 X는 1, Theta는 2로 초기화 하면
# X, Theta 를 보면 행마다 전부 동일한 값으로 되어있는 것을 볼 수 있다.

initial_parameters = np.append(X.flatten(), Theta.flatten())


# optimize
_lambda = 10
result = op.minimize(fun=cofiCostFunc,x0=initial_parameters,\
                     args=(Ynorm, R, num_users, num_movies, num_features, _lambda),\
                         method='L-BFGS-B',jac=True,options={'maxiter':100})

theta = result.x
print('cost : ', result.fun)

# Unfold the returned theta back into U and W
X = np.reshape(theta[:num_movies*num_features], (num_movies, num_features))
Theta = np.reshape(theta[num_movies*num_features:], (num_users, num_features))

print('Recommender system learning completed.\n')

## ================== Part 8: Recommendation for you ====================
#  After training the model, you can now make recommendations by computing
#  the predictions matrix.
#
print('============== Part 8: Recommendation for you ===============')
p = X @ Theta.T
my_predictions = p[:,0].reshape(-1,1) + Ymean
my_predictions = my_predictions.flatten()

movieList = loadMovieList()

ix = np.argsort(my_predictions)[::-1] # descend sort

print('\nTop recommendations for you:\n')
for i in range(10):
    j = ix[i]
    print('Predicting rating {0:0.1f} for movie {1}\n'.format(my_predictions[j], movieList[j]))

print('\n\nOriginal ratings provided:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated {} for {}\n'.format(my_ratings[i], movieList[i]))

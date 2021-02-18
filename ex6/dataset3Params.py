import numpy as np
import sklearn.svm as svm

def dataset3Params(X, y, Xval, yval):
    #DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
    #where you select the optimal (C, sigma) learning parameters to use for SVM
    #with RBF kernel
    #   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
    #   sigma. You should complete this function to return the optimal C and 
    #   sigma based on a cross-validation set.
    #
    
    # You need to return the following variables correctly.
    C = 1
    sigma = 0.3
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return the optimal C and sigma
    #               learning parameters found using the cross validation set.
    #               You can use svmPredict to predict the labels on the cross
    #               validation set. For example, 
    #                   predictions = svmPredict(model, Xval);
    #               will return the predictions on the cross validation set.
    #
    #  Note: You can compute the prediction error using 
    #        mean(double(predictions ~= yval))
    #
    
    values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    means = np.zeros((64, 1))
    idxs = np.zeros((64,2))
    
    k = 0
    for i in range(np.size(values)):
      for j in range(np.size(values)):
        C = values[i]
        sigma = values[j]
        g = 1 / (2 * sigma ** 2)
        classifier = svm.SVC(C=C, kernel='rbf', tol=1e-3, gamma = g)
        model = classifier.fit(X, y)
        predictions = model.predict(Xval).reshape(-1,1)
        means[k] = np.mean(predictions != yval)
        idxs[k,0] = i
        idxs[k,1] = j
        k += 1

    idx = np.argmin(means[:,0])
    C = values[int(idxs[idx,0])]
    sigma = values[int(idxs[idx,1])]
    
    print('error : ', means[idx])
    print('C : ', C)
    print('sigma : ', sigma)
    
    return C, sigma
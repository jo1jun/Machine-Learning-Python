import numpy as np
import sklearn.svm as svm

def dataset3Params(X, y, Xval, yval):
    
    values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    means = np.zeros((64, 1))
    
    k = 0
    for i in range(np.size(values)):
      for j in range(np.size(values)):
        C = values[i]
        sigma = values[j]
        g = 1.0 / (2.0 * sigma ** 2)
        classifier = svm.SVC(C=C, kernel='rbf', tol=1e-3, max_iter=200, gamma = g)
        model = classifier.fit(X, y)
        predictions = model.predict(Xval)
        means[k,0] = np.mean(predictions != yval)
        k += 1
    
    idx = np.argmin(means)
    
    if idx < 8 :
      i = 0
      j = idx
    elif idx < 16:
      i = 1
      j = idx - 7
    elif idx < 24:
      i = 2
      j = idx - 15
    elif idx < 32:
      i = 3
      j = idx - 23
    elif idx < 40:
      i = 4
      j = idx - 31
    elif idx < 48:
      i = 5
      j = idx - 39
    elif idx < 56:
      i = 6
      j = idx - 47
    elif idx < 64:
      i = 7
      j = idx - 55
    
    
    C = values[i]
    sigma = values[j]
    
    print('C : ', C)
    print('sigma : ', sigma)
    
    return C, sigma
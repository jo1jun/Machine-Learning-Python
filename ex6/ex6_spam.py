import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
from readFile import readFile
from processEmail import processEmail
import sklearn.svm as svm
import scipy.io
from emailFeatures import emailFeatures
from getVocabList import getVocabList
import numpy as np

## Machine Learning Online Class
#  Exercise 6 | Spam Classification with SVMs
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions
#
#     gaussianKernel.m
#     dataset3Params.m
#     processEmail.m
#     emailFeatures.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## ==================== Part 1 Email Preprocessing ====================
#  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
#  to convert each email into a vector of features. In this part, you will
#  implement the preprocessing steps for each email. You should
#  complete the code in processEmail.m to produce a word indices vector
#  for a given email.
print('==================== Part 1 Email Preprocessing ====================')
print('\nPreprocessing sample email (emailSample1.txt)\n')

# Extract Features
file_contents = readFile('emailSample1.txt')
word_indices  = processEmail(file_contents)

# Print Stats
print('Word Indices \n')
print(word_indices)
print('\n\n')

## ==================== Part 2 Feature Extraction ====================
#  Now, you will convert each email into a vector of features in R^n. 
#  You should complete the code in emailFeatures.m to produce a feature
#  vector for a given email.
print('==================== Part 2 Feature Extraction ====================')
print('\nExtracting features from sample email (emailSample1.txt)\n')

# Extract Features
file_contents = readFile('emailSample1.txt')
word_indices  = processEmail(file_contents)
features      = emailFeatures(word_indices)

# Print Stats
print('Length of feature vector ', len(features), '\n')
print('Number of non-zero entries ', sum(features > 0), '\n')

## =========== Part 3 Train Linear SVM for Spam Classification ========
#  In this section, you will train a linear classifier to determine if an
#  email is Spam or Not-Spam.
print('=========== Part 3 Train Linear SVM for Spam Classification ========')
# Load the Spam Email dataset
# You will have X, y in your environment
mat = scipy.io.loadmat('spamTrain.mat')
X, y = mat['X'], mat['y'].flatten()

print('\nTraining Linear SVM (Spam Classification)\n')
print('(this may take 1 to 2 minutes) ...\n')

C = 0.1
classifier = svm.SVC(C=C, kernel='linear', tol=1e-3)
model = classifier.fit(X, y)
p = model.predict(X).reshape(-1,1)
print('Training Accuracy ', np.mean(p == y) * 100, '\n')

## =================== Part 4 Test Spam Classification ================
#  After training the classifier, we can evaluate it on a test set. We have
#  included a test set in spamTest.mat
print('=================== Part 4 Test Spam Classification ================')

# Load the test dataset
# You will have Xtest, ytest in your environment
mat = scipy.io.loadmat('spamTest.mat')
Xtest, ytest = mat['Xtest'], mat['ytest'].flatten()

print('\nEvaluating the trained Linear SVM on a test set ...\n')

p = model.predict(Xtest).reshape(-1,1)
ytest = ytest.reshape(-1,1)

print('Test Accuracy: ', np.mean(p == ytest) * 100, '\n')

## ================= Part 5 Top Predictors of Spam ====================
#  Since the model we are training is a linear SVM, we can inspect the
#  weights learned by the model to understand better how it is determining
#  whether an email is spam or not. The following code finds the words with
#  the highest weights in the classifier. Informally, the classifier
#  'thinks' that these words are the most likely indicators of spam.
#
print('================= Part 5 Top Predictors of Spam ====================')

# Sort the weights and obtin the vocabulary list
weight = model.coef_.flatten()
weight_reverse = np.sort(weight)[::-1]          #역방향 정렬
index_reverse = np.argsort(weight)[::-1]        #역방향 정렬
vocabList = getVocabList()

print('\nTop predictors of spam \n')

for i in range(15):
    print(' {} ({}) \n'.format(vocabList[index_reverse[i]], weight_reverse[i]))

## =================== Part 6 Try Your Own Emails =====================
#  Now that you've trained the spam classifier, you can use it on your own
#  emails! In the starter code, we have included spamSample1.txt,
#  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples. 
#  The following code reads in one of these emails and then uses your 
#  learned SVM classifier to determine whether the email is Spam or 
#  Not Spam
print('=================== Part 6 Try Your Own Emails =====================')

# Set the file to be read in (change this to spamSample2.txt,
# emailSample1.txt or emailSample2.txt to see different predictions on
# different emails types). Try your own emails as well!

filename = 'emailSample2.txt'
#filename = 'emailSample3.txt' #평소 실제로 사용했던 email 을 emailSample3.txt 에 저장해서 실행

# spamSample 은 1, emailSample 은 0으로 잘 분류된다. 직접 작성한 email 을 적용시켜봐도 잘 나온다.

# Read and predict
file_contents = readFile(filename)
word_indices  = processEmail(file_contents)
x             = emailFeatures(word_indices)
p             = model.predict(x.T)

print('\nProcessed {}\n\nSpam Classification {}\n'.format(filename, p))
print('(1 indicates spam, 0 indicates not spam)\n\n')

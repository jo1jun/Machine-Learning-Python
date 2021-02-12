import numpy as np
from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
    #PREDICT Predict the label of an input given a trained neural network
    #   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    #   trained weights of a neural network (Theta1, Theta2)
    
    # Useful values
    m = X.shape[0]
    
    # You need to return the following variables correctly 
    p = np.zeros((m,1))
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned neural network. You should set p to a 
    #               vector containing labels between 1 to num_labels.
    #
    # Hint: The max function might come in useful. In particular, the max
    #       function can also return the index of the max element, for more
    #       information see 'help max'. If your examples are in rows, then, you
    #       can use max(A, [], 2) to obtain the max for each row.
    #
    X = np.append(np.ones((m,1)), X, axis=1)
    
    a2 = sigmoid(X @ Theta1.T)  #(m * n) * (n * h) = (m * h)
    a2 = np.append(np.ones((a2.shape[0],1)), a2, axis=1) #bias 추가
    a3 = sigmoid(a2 @ Theta2.T) # 이것이 가설의 최종 값. #(m * h) * (h * 10) = (m * 10)
    p = np.argmax(a3, axis=1) + 1 #index 가 0부터 시작하므로 + 1 로 보정.
    
    return p
    # =========================================================================
import numpy as np
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda):
    #NNCOSTFUNCTION Implements the neural network cost function for a two layer
    #neural network which performs classification
    #   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
    #   X, y, lambda) computes the cost and gradient of the neural network. The
    #   parameters for the neural network are "unrolled" into the vector
    #   nn_params and need to be converted back into the weight matrices. 
    # 
    #   The returned parameter grad should be a "unrolled" vector of the
    #   partial derivatives of the neural network.
    #
    
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = nn_params[:(input_layer_size + 1) * hidden_layer_size] #bias 고려하여 +1
    Theta2 = nn_params[np.size(Theta1):]
    
    # print(Theta1.shape)
    # print(Theta2.shape)
    
    Theta1 = Theta1.reshape(hidden_layer_size, -1)
    Theta2 = Theta2.reshape(num_labels, -1)
    
    # print(Theta1.shape)
    # print(Theta2.shape)
    
    # Setup some useful variables
    m = X.shape[0]
    
    # You need to return the following variables correctly 
    J = 0
    Theta1_grad = np.zeros_like(Theta1)
    Theta2_grad = np.zeros_like(Theta2)
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the code by working through the
    #               following parts.
    #
    # Part 1: Feedforward the neural network and return the cost in the
    #         variable J. After implementing Part 1, you can verify that your
    #         cost function computation is correct by verifying the cost
    #         computed in ex4.m
    #
    # Part 2: Implement the backpropagation algorithm to compute the gradients
    #         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    #         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    #         Theta2_grad, respectively. After implementing Part 2, you can check
    #         that your implementation is correct by running checkNNGradients
    #
    #         Note: The vector y passed into the function is a vector of labels
    #               containing values from 1..K. You need to map this vector into a 
    #               binary vector of 1's and 0's to be used with the neural network
    #               cost function.
    #
    #         Hint: We recommend implementing backpropagation using a for-loop
    #               over the training examples if you are implementing it for the 
    #               first time.
    #
    # Part 3: Implement regularization with the cost function and gradients.
    #
    #         Hint: You can implement this around the code for
    #               backpropagation. That is, you can compute the gradients for
    #               the regularization separately and then add them to Theta1_grad
    #               and Theta2_grad from Part 2.
    #
    
    #forward
    #affine - sigmoid - affine - sigmoid - cost 로 구성된다.
    a1 = np.append(np.ones((m,1)), X, axis=1)               #+bias #5000 * 401
    z2 = a1 @ Theta1.T                                      #affine #5000 * 25
    a2 = sigmoid(z2)                                        #sigmoid #5000 * 25
    a2 = np.append(np.ones((a2.shape[0],1)), a2, axis=1)    #+bias #5000 * 26
    z3 = a2 @ Theta2.T                                      #affine #5000 * 40
    a3 = sigmoid(z3)                                        #sigmoid #5000 * 10
    h = a3

    #print(y.shape)
    y_t = np.zeros((m,num_labels))
    y_t[np.arange(m),y.reshape(-1) - 1] = 1                 #y 를 [0 0 ... 1 0 0] 형태로 변환. (+index 보정)
    #print(y_t.shape)
    J = np.sum((-y_t*np.log(h) - (1 - y_t)*np.log(1 - h))) / m
    #logistic regression(binary class) 에서는 y 가 (m,1), h가 (m,1) 이었으므로 y.T @ np.log(h) 로 표현해도 되었던 것. 지금은 multi class.
    J_regterm = _lambda/(2*m) * (np.sum(np.square(Theta1[:,1:])) + np.sum(np.square(Theta2[:,1:]))) #Theta 는 bias 부분 제거 후 연산
    J += J_regterm
    
    #lecture note 에서는 마지막에 Theta grad 에 1/m 을 곱해주는데 d3 에 1/m 을 곱해줘도 상수 계수 이므로 앞으로 전파되어서 문제없다.   
    #역전파 할 때 뒷 층의 bias 에 대한 가중치는 앞 층으로 전달될 error(delta) 를 계산할 때 활용하지 않는다.
    #왜냐하면 앞 층으로 전달될 error(delta) 는 앞 층의 가중치들을 갱신하기 위해서 전달하는 것인데 뒷 층의 bias 는 앞 층의 가중치들에
    #의해 전혀 영향을 받지 않는 고정값 1 이기 때문이다. bias 를 제외한 뒷 층의 모든 a(z 에서 activation function 통과), z 는
    #앞 층의 가중치로 연산하여 나온 결과의므로 이들과 관련된 가중치들 만을 활용하여 앞 층에 전달할 delta 를 계산한다.
    #deep learning from scratch 에서는 이에 대한 고민이 필요없었다. 왜냐면 bias vector 가 따로 있었기 때문.
    #여기서는 bias 가 theta_0 로 같이 붙어있다.
    
    '''
    #backward (bias 제외) (for loop)
    #원래 식 활용하면서 m 번 반복 할 수 있지만 비효율적.
    for i in range(m):
        a1 = X[i].reshape(-1,1)                         #(400,1)
        a1 = np.append(1, a1)                           #(401,1)
        z2 = (Theta1 @ a1).reshape(-1,1)                #(25,1) #행렬곱 하면 차원이 풀어짐.
        a2 = sigmoid(z2)
        a2 = np.append(1, a2)                           #(26,1)
        z3 = (Theta2 @ a2).reshape(-1,1)                #(10,1)
        a3 = sigmoid(z3)
        h = a3
        
        d3 = (h - y_t[i].reshape(-1,1)) / m                                 #(10,1) #batch size 가 m 이므로 m 으로 나누기.
        Theta2_grad += d3 @ a2.reshape(-1,1).T                              #(10,26)
        d2 = Theta2[:,1:].T @ d3 * sigmoidGradient(z2).reshape(-1,1)        #(25,1)
        Theta1_grad += d2 @ a1.reshape(-1,1).T                              #(25,401)
    
    Theta2_grad[:,1:] += (_lambda/m) * Theta2[:,1:]
    Theta1_grad[:,1:] += (_lambda/m) * Theta1[:,1:]
    '''
    
    #backward (fully vectorize)                                                 #원래 식을 vectorize 하기 위해 변형
    #행렬 곱셈을 transpose 로 잘 맞추면 된다.
    d3 = (h - y_t) / m                                                          #변환한 y 를 사용, batch size 가 m 이므로 m 으로 나누기.
    Theta2_grad = d3.T @ a2                                                     #batch 단위로 vectorize 했으므로 m 으로 나눠야함.
    Theta2_grad[:,1:] += (_lambda/m) * Theta2[:,1:]                             #bias 항에는 regterm 제외
    
    d2 = d3 @ Theta2[:,1:] * sigmoidGradient(z2)                                #bias 부분 제거 이후 오차 역전파
    Theta1_grad = d2.T @ a1
    Theta1_grad[:,1:] += (_lambda/m) * Theta1[:,1:]
    
    grad = np.append(Theta1_grad.reshape(-1), Theta2_grad.reshape(-1))
    
    return J, grad
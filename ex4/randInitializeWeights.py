import numpy as np

def randInitializeWeights(L_in,L_out):
    #RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
    #incoming connections and L_out outgoing connections
    #   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
    #   of a layer with L_in incoming connections and L_out outgoing 
    #   connections. 
    #
    #   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
    #   the first column of W handles the "bias" terms
    #
    
    # You need to return the following variables correctly 
    W = np.zeros((L_out, 1 + L_in)) # +1 은 bias 를 위함
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Initialize W randomly so that we break the symmetry while
    #               training the neural network.
    #
    # Note: The first column of W corresponds to the parameters for the bias unit
    #
    
    #When training neural networks, it is important to randomly initialize the parameters for symmetry breaking.
    #You should use 0.12 This range of values ensures that the parameters are kept small and makes the learning more efficient.
    
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

    #지금은 깊게 알 필요 x 요점은 symmetry breaking 을 위해서 random initialize 해야하고 주어진 std 값을 쓰라는 것.
    return W    
    # =========================================================================
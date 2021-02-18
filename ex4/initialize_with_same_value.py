# README 참고
import numpy as np

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda):

    Theta1 = nn_params[:(input_layer_size) * hidden_layer_size]
    Theta2 = nn_params[np.size(Theta1):]
    
    Theta1 = Theta1.reshape(hidden_layer_size, -1)
    Theta2 = Theta2.reshape(num_labels, -1)

    m = 2
    
    J = 0
    Theta1_grad = np.zeros_like(Theta1)
    Theta2_grad = np.zeros_like(Theta2)


    z2 = X @ Theta1.T                                     
    z3 = z2 @ Theta2.T                                  
    J = z3

    d3 = (J - y) / m                                                      
    Theta2_grad = d3.T @ z2                                               
    d2 = d3 @ Theta2
    Theta1_grad = d2.T @ X

    grad = np.append(Theta1_grad.reshape(-1), Theta2_grad.reshape(-1))

    return J, grad

Theta1 = np.zeros((2,3)) + 2
Theta2 = np.zeros((1,2)) + 2
nn_params = np.append(Theta1.reshape(-1), Theta2.reshape(-1))
J,g = nnCostFunction(nn_params, 3, 2, 1, np.array([[1,2,3],[2,3,1]]), 40, 0)
Theta1 = g[:6]
Theta2 = g[np.size(Theta1):]
Theta1 = Theta1.reshape(2, -1)
Theta2 = Theta2.reshape(1, -1)
print(Theta1)
print(Theta2)
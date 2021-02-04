import numpy as np
def gaussianKernel(x1, x2, sigma):
    #RBFKERNEL returns a radial basis function kernel between x1 and x2
    #   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
    #   and returns the value in sim
    
    # Ensure that x1 and x2 are column vectors
    x1 = np.array(x1)
    x2 = np.array(x2)

    # You need to return the following variables correctly.
    sim = 0
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return the similarity between x1
    #               and x2 computed using a Gaussian kernel with bandwidth
    #               sigma
    #
    #
    
    sim = np.exp(-np.sum(np.square(x1 - x2)) / ( 2 * (sigma ** 2)))
    
    
    return sim
    
    # =============================================================

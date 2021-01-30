import math
import numpy as np
import matplotlib.pyplot as plt

def displayData(X):
    #DISPLAYDATA Display 2D data in a nice grid
    #   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
    #   stored in X in a nice grid. It returns the figure handle h and the 
    #   displayed array if requested.
    # Set example_width automatically if not passed in
        
    example_width = round(math.sqrt(X.shape[1]))    #round = 반올림(float -> int)
    
    # Gray Image
    plt.set_cmap('gray')
        
    # Compute rows, cols
    m, n = X.shape
    example_height = int((n / example_width))

    # Compute number of items to display
    display_rows = int(math.floor(np.sqrt(m)))
    display_cols = int(math.ceil(m / display_rows))

    # Between images padding
    pad = 1
    
    # Setup blank display
    display_array = - np.ones((pad + display_rows * (example_height + pad), 
                           pad + display_cols * (example_width + pad)))
    
    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in np.arange(display_rows):
        for i in np.arange(display_cols):
            if curr_ex >= m:
                break
            # Copy the patch
            
            # Get the max value of the patch
            max_val = np.max(np.abs(X[curr_ex, :]))
            
            x = pad + j * (example_height + pad)
            y = pad + i * (example_width + pad)
        
            display_array[x : x + example_height, y : y+ example_width] = \
            X[curr_ex, :].reshape((example_height, example_width)).T / max_val
            
            curr_ex = curr_ex + 1
            
            if curr_ex >= m:
                break
                       
    # Display Image
    h = plt.imshow(display_array)
    
    # Do not show axis
    plt.axis('off')

    
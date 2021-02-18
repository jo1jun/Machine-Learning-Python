import numpy as np

def selectThreshold(yval, pval):
    #SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
    #outliers
    #   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
    #   threshold to use for selecting outliers based on the results from a
    #   validation set (pval) and the ground truth (yval).
    #
    
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0
    
    stepsize = (max(pval) - min(pval)) / 1000
    for epsilon in np.arange(min(pval),max(pval),stepsize):
        # ====================== YOUR CODE HERE ======================
        # Instructions: Compute the F1 score of choosing epsilon as the
        #               threshold and place the value in F1. The code at the
        #               end of the loop will compare the F1 score for this
        #               choice of epsilon and set it to be the best epsilon if
        #               it is better than the current choice of epsilon.
        #               
        # Note: You can use predictions = (pval < epsilon) to get a binary vector
        #       of 0's and 1's of the outlier predictions
        
        tp, fp, fn = 0,0,0
        prec, rec = 0,0
        
        predictions = pval < epsilon
        predictions = np.reshape(predictions, (yval.shape))

        # predict anomaly & actual anomaly
        tp = sum((predictions == 1) & (yval == 1))
        
        # predict anomaly & actual nonanomaly
        fp = sum((predictions == 1) & (yval == 0))
        
        # predict nonanomaly & actual anomaly
        fn = sum((predictions == 0) & (yval == 1))

        # RuntimeWarning: invalid value encountered in true_divide
        # 0으로 나누는 것을 방지하기 위해 dummy 를 분모에 더해준다.
        dummy = 1e-7

        prec = tp / (tp + fp + dummy)
        rec = tp / (tp + fn + dummy)
        F1 = 2 * prec * rec / (prec + rec + dummy)
        # =============================================================
        
        if F1 > bestF1:
           bestF1 = F1
           bestEpsilon = epsilon
           
    return bestEpsilon, bestF1
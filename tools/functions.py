import numpy as np

def sigmoid(x):
    return 1.0 / (1+np.exp(-1 * x))

def threshold(data, threshold=0.5, true_val=1, false_val=0, incl_thresh_in_upper = True):
    
    if data > threshold:
        return true_val
    if data == threshold:
        if incl_thresh_in_upper:
            return true_val
    return false_val
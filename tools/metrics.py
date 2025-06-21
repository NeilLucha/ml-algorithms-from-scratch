import numpy as np

def mse(y_true, y_pred):
    '''
    Calculates Mean Squared Error between true and predicted values
    '''
    return np.mean((y_true - y_pred) ** 2)

def binary_crossentropy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    clip_const = 10**-10 # Clipping values outside a range to stabilize log calculation
    y_pred = np.clip(y_pred, clip_const, 1-clip_const)
    return -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))

def accuracy_score(y_true, y_pred):
    return np.sum(np.array(y_true)==np.array(y_pred))/len(y_true)

def binary_confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    true_pos = np.sum((y_true==1) & (y_pred==1))
    true_neg = np.sum((y_true==0) & (y_pred==0))
    false_pos = np.sum((y_true==0) & (y_pred==1))
    false_neg = np.sum((y_true==1) & (y_pred==0))
    return [[true_neg, false_pos],[false_neg, true_pos]]
import numpy as np

def mse(y_true, y_pred):
    '''
    Calculates Mean Squared Error between true and predicted values
    '''
    return np.mean((y_true - y_pred) ** 2)
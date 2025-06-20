import numpy as np

def normalize(X):
    return (X - np.mean(X)) / np.std(X)
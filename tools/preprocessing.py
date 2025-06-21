import numpy as np

def standardize(X):
    return (X - np.mean(X)) / np.std(X)


def train_test_split(X, y, train_size=0.8, seed=42):
    np.random.seed(seed=seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    idx_split = int(train_size*len(indices))
    train_indices = indices[:idx_split]
    test_indices = indices[idx_split:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train =  y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test
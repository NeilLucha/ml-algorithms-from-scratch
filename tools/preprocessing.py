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

class LabelEncoder:
    def __init__(self):
        self.mapping = None
        self.inverse_mapping = None
    
    def fit(self, X):
        labels = np.unique(X, return_counts=False).flatten()
        self.mapping = {label: mapping for mapping, label in enumerate(labels)}
        self.inverse_mapping = {mapping: label for mapping, label in enumerate(labels)}
        
    def transform(self, X):
        return np.array([self.mapping[dat] for dat in X])
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
        
    def inverse_transform(self, y):
        return np.array([self.inverse_mapping[dat] for dat in y])
        
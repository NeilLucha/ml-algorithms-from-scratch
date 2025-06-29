import numpy as np

class KNN:
    
    def __init__(self, k=3, distance_metric = 'euclidean'):
        supported_distance_metrics = ['euclidean', 'manhattan', 'cosine']
        self.k = k
        if distance_metric not in supported_distance_metrics:
            raise ValueError(f"Distance Metric {distance_metric} is not Supported")
        self.distance_metric = distance_metric
    
    def fit(self, X, y):
        if self.k > len(X):
            raise ValueError(f'k cannot exceed the number of training samples')
        self.X = np.array(X)
        self.y = np.array(y)
        
    def get_distance(self, x, y):
        if self.distance_metric == 'euclidean':
            return np.linalg.norm(x-y)
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x-y))
        
        elif self.distance_metric == 'cosine':
            norm_x = np.linalg.norm(x)
            norm_y = np.linalg.norm(y) 
            if norm_x==0 or norm_y==0:
                return 1.0
            return 1 - np.dot(x,y)/(norm_x*norm_y)
    
    def predict(self, X):
        '''
        Predicts the class label for single input data point X based on the K Nearest Neightbors of X
        '''
        
        distances = [self.get_distance(X, x) for x in self.X]
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y[k_nearest_indices]
        
        classified_label = None

        while True:
            labels, counts = np.unique(k_nearest_labels, return_counts=True)
            sorted_counts = np.sort(counts)
            
            if len(labels) == 1:
                return k_nearest_labels[0]
            
            if sorted_counts[-1]>sorted_counts[-2]:
                classified_label = labels[np.argmax(counts)]
                return classified_label
            
            else:
                k_nearest_labels = k_nearest_labels[:-1]
                
    def predict_all(self, X):
        X = np.array(X)
        return np.array([self.predict(dat) for dat in X])

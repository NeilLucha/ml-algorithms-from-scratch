import numpy as np

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, tolerance=0.0001):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self._labels = None
        self.centroids = None
        self.tolerance = tolerance
        
    def _init_centroids(self, X):
        # X=np.array(X)
        self.centroids = X[np.random.choice(a=len(X), size=self.n_clusters, replace=False)]
        
    def _assign_clusters(self, X):
        self._labels = np.zeros(len(X), dtype=int)
        for i, dat in enumerate(X):
            distances = np.linalg.norm(dat - self.centroids, axis=1)
            cluster_center_index = np.argmin(distances)
            self._labels[i] = cluster_center_index
            
    def _centroid_update(self, X):
        prev_centroids = self.centroids.copy()
        for i in range(len(self.centroids)):
            cluster = X[self._labels==i]
            if len(cluster)>0:
                self.centroids[i] = np.mean(cluster, axis=0)
        
        return np.linalg.norm(prev_centroids-self.centroids, axis=1)
    
    def fit(self, X):
        self._init_centroids(X)
        for i in range(self.max_iter):
            self._assign_clusters(X)
            shift = self._centroid_update(X)
            if np.max(shift)<self.tolerance:
                break
        
        inertia = sum(np.linalg.norm(dat-self.centroids[self._labels[i]])**2 for i,dat in enumerate(X))
        return inertia
import numpy as np

class MyReducer:
    """
    Implement your own dimensionality reduction to 10 dimensions using PCA via SVD.
    
    Projects high-dimensional data to 10 dimensions while preserving maximum variance
    and pairwise Euclidean distance structure for k-NN classification.
    """
    
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
    
    def fit(self, X):
        """
        Learn the principal components from training data using SVD.
        
        Args:
            X: Training data, shape (n_samples, n_features)
        
        Returns:
            self
        """
        # 1. Compute and store feature-wise mean for centering
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 2. Compute Singular Value Decomposition (SVD) of centered X
        # X_centered = U * S * Vt
        # The rows of Vt (columns of V) are the principal directions/components
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # 3. Store top n_components principal axes (shape: n_components, n_features)
        self.components_ = Vt[:self.n_components]
        return self
    
    def transform(self, X):
        """
        Apply the learned PCA reduction to data.
        
        Args:
            X: Data to transform, shape (n_samples, n_features)
        
        Returns:
            X_reduced: shape (n_samples, 10)
        """
        # Center using training mean and project onto top principal components
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
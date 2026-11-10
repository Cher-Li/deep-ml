import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """
    # copy of weights to avoid modifying the input arr
    w = weights.copy()
    n_samples = X.shape[0]
    
    for i in range(n_iter):
        idx = i % n_samples
        xi = X[idx]
        yi = y[idx]
        
        # 1. prediction
        y_pred = np.dot(xi, w)
        
        # 2. per-sample MSE gradient (using 2 * (y_pred - yi) * xi)
        error = y_pred - yi
        gradient = 2 * error * xi
        
        # 3. weight update 
        w = w - learning_rate * gradient
        
    return w.tolist()
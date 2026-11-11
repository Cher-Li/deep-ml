import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    # * mini-batch data
    X_batch = X[batch_indices]
    y_batch = y[batch_indices]
    m = len(batch_indices)
    
    # predictions for the mini-batch
    y_pred = X_batch @ weights + bias
    
    error = y_pred - y_batch
    
    # L = (1/m) * sum((Xw + b - y)^2)
    grad_weights = (2 / m) * (X_batch.T @ error)
    grad_bias = (2 / m) * np.sum(error)
    
    # gradient descent update
    updated_weights = weights - lr * grad_weights
    updated_bias = bias - lr * grad_bias
    
    return np.append(updated_weights, updated_bias)
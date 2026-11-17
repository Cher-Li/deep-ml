import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], list[float]]:
    """
    Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    
    N, d = X.shape
    
    # prepend col of ones for bias
    X_b = np.hstack([np.ones((N, 1)), X])
    
    # init all param to 0
    theta = np.zeros(d + 1)
    
    losses = []
    
    for _ in range(iterations):
        # y_hat = sigma(X_b @ theta)
        z = X_b @ theta
        y_hat = 1 / (1 + np.exp(-z))
        
        # loss (sum over N samples)
        loss = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        losses.append(loss)
        
        # theta <- theta - eta * X_b.T @ (y_hat - y)
        gradient = X_b.T @ (y_hat - y)
        theta = theta - learning_rate * gradient
        
    coefficients = [round(float(val), 4) for val in theta]
    rounded_losses = [round(float(l), 4) for l in losses]
    
    return coefficients, rounded_losses
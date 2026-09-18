import numpy as np

def gaussian_mle(data: np.ndarray) -> tuple:
    """
    Compute Maximum Likelihood Estimates for Gaussian distribution parameters.
    
    Args:
        data: 1D numpy array of observations
        
    Returns:
        Tuple of (mean_mle, variance_mle)
    """
    mean_mle = np.mean(data)
    variance_mle = np.mean((data - mean_mle) ** 2)
    
    return (mean_mle, variance_mle)

# import numpy as np

# def gaussian_mle(data: np.ndarray) -> tuple:
#     """
#     Compute Maximum Likelihood Estimates for Gaussian distribution parameters.
    
#     Args:
#         data: 1D numpy array of observations
        
#     Returns:
#         Tuple of (mean_mle, variance_mle)
#     """
#     n = len(data)
    
#     # MLE for mean: sample mean
#     mean_mle = np.sum(data) / n
    
#     # MLE for variance: biased estimator (divide by n, not n-1)
#     variance_mle = np.sum((data - mean_mle) ** 2) / n
    
#     return (mean_mle, variance_mle)
import numpy as np

def classify_critical_point(hessian: np.ndarray, tol: float = 1e-10):
    """
    Classify a critical point using the Hessian eigenvalues.

    Returns:
        -1 : local minimum  (all eigenvalues > 0)
         1 : local maximum  (all eigenvalues < 0)
         0 : saddle point   (mixed signs)
        None : inconclusive (any eigenvalue approximately zero)
    """
    
    # Compute eigenvalues of the Hessian
    eigenvalues = np.linalg.eigvals(hessian)

    # Check for near-zero eigenvalues
    if np.any(np.abs(eigenvalues) <= tol):
        return None

    # All positive -> local minimum
    if np.all(eigenvalues > tol):
        return -1

    # All negative -> local maximum
    if np.all(eigenvalues < -tol):
        return 1

    # Mixed signs -> saddle point
    return 0

# import numpy as np

# def classify_critical_point(hessian: np.ndarray, tol: float = 1e-10):
#     hessian = np.array(hessian, dtype=float)
#     eigenvalues = np.linalg.eigvalsh(hessian)

#     positive = np.sum(eigenvalues > tol)
#     negative = np.sum(eigenvalues < -tol)
#     zero = np.sum(np.abs(eigenvalues) <= tol)

#     n = len(eigenvalues)

#     if zero > 0:
#         return None
#     elif positive == n:
#         return -1
#     elif negative == n:
#         return 1
#     else:
#         return 0
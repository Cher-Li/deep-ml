import numpy as np

def check_positive_definite(matrix: list) -> dict:
    """
    Check if a matrix is positive definite and compute its eigenvalues.
    
    Args:
        matrix: A 2D list representing a square matrix
        
    Returns:
        dict with:
            - 'is_positive_definite' (bool)
            - 'eigenvalues' (list of floats sorted ascending)
    """
    A = np.array(matrix, dtype=float)

    # Validate that the matrix is square
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input must be a square matrix.")

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(A)

    # Remove tiny imaginary parts caused by numerical errors
    eigenvalues = np.real_if_close(eigenvalues)

    # Sort and round eigenvalues
    eigenvalues_sorted = sorted(float(ev) for ev in eigenvalues)
    eigenvalues_rounded = [round(ev, 4) for ev in eigenvalues_sorted]

    # A matrix is positive definite if:
    # 1. It is symmetric
    # 2. All eigenvalues are greater than 1e-10
    is_symmetric = np.allclose(A, A.T)
    is_positive_definite = is_symmetric and all(ev > 1e-10 for ev in eigenvalues_sorted)

    return {
        'is_positive_definite': is_positive_definite,
        'eigenvalues': eigenvalues_rounded
    }

# import numpy as np

# def check_positive_definite(matrix: list) -> dict:
#     """
#     Check if a matrix is positive definite and compute its eigenvalues.
    
#     Args:
#         matrix: A 2D list representing a square matrix
        
#     Returns:
#         dict with 'is_positive_definite' (bool) and 'eigenvalues' (list of floats sorted ascending)
#     """
#     A = np.array(matrix, dtype=float)
    
#     # Check symmetry (A must equal its transpose)
#     is_symmetric = np.allclose(A, A.T)
    
#     if is_symmetric:
#         # Use eigvalsh for symmetric matrices (more numerically stable)
#         eigenvalues = np.linalg.eigvalsh(A)
#     else:
#         # For non-symmetric, use eigvals and take real parts
#         eigenvalues = np.real(np.linalg.eigvals(A))
    
#     eigenvalues_sorted = sorted(eigenvalues)
    
#     # Positive definite requires: symmetric AND all eigenvalues > 0
#     is_positive_definite = is_symmetric and all(ev > 1e-10 for ev in eigenvalues_sorted)
    
#     return {
#         'is_positive_definite': is_positive_definite,
#         'eigenvalues': [round(float(ev), 4) for ev in eigenvalues_sorted]
#     }
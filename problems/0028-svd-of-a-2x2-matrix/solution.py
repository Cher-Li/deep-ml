import numpy as np

def svd_2x2(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using closed-form solution.
    
    Args:
        A: 2x2 numpy array
    
    Returns:
        U: 2x2 orthogonal matrix (left singular vectors)
        s: 1D array of singular values
        V: 2x2 matrix (right singular vectors)
    """
    # Compute intermediate values <- auxiliary values
    y1, x1 = (A[1, 0] + A[0, 1]), (A[0, 0] - A[1, 1])
    y2, x2 = (A[1, 0] - A[0, 1]), (A[0, 0] + A[1, 1])

    # norms: 
    h1 = np.sqrt(y1**2 + x1**2)
    h2 = np.sqrt(y2**2 + x2**2)

    # * Handle edge case where h1 or h2 is zero
    t1 = x1 / h1 if h1 > 1e-10 else 1.0
    t2 = x2 / h2 if h2 > 1e-10 else 1.0

    # Compute rotation components
    cc = np.sqrt((1.0 + t1) * (1.0 + t2))
    ss = np.sqrt((1.0 - t1) * (1.0 - t2))
    cs = np.sqrt((1.0 + t1) * (1.0 - t2))
    sc = np.sqrt((1.0 - t1) * (1.0 + t2))

    c1, s1 = (cc - ss) / 2.0, (sc + cs) / 2.0
    U = np.array([[-c1, -s1], [-s1, c1]])

    # Singular values
    s = np.array([(h1 + h2) / 2.0, abs(h1 - h2) / 2.0])

    # Compute V, handling near-zero singular values
    s_inv = np.array([1.0/s[0] if s[0] > 1e-10 else 0.0,
                      1.0/s[1] if s[1] > 1e-10 else 0.0])
    V = np.diag(s_inv) @ U.T @ A

    return U, s, V
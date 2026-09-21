# * 
import numpy as np

def pairwise_cosine_similarity(X):
    X = np.array(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    X_normed = X / safe_norms # this is X tilde i 
    X_normed = np.where(norms == 0, 0.0, X_normed)
    S = X_normed @ X_normed.T
    S = np.round(S, 4)
    return S.tolist()
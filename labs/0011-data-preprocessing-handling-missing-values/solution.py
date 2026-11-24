# --- cell 1 ---
import numpy as np


def impute(X: np.ndarray) -> np.ndarray:
    """
    Fill in missing values (NaN) in the input array.

    Args:
        X: Array with possible NaN values, shape (n_samples, n_features)

    Returns:
        X_clean: Array with no NaN values, same shape as X
    """
    X_clean = np.asarray(X, dtype=float).copy()

    if X_clean.ndim == 1:
        X_clean = X_clean.reshape(-1, 1)

    for col_idx in range(X_clean.shape[1]):
        col = X_clean[:, col_idx]
        nan_mask = np.isnan(col)

        if np.any(nan_mask):
            valid_values = col[~nan_mask]
            fill_value = float(np.median(valid_values)) if valid_values.size > 0 else 0.0 # fill with col wise median, 0.0 if col missing
            col[nan_mask] = fill_value

    return X_clean
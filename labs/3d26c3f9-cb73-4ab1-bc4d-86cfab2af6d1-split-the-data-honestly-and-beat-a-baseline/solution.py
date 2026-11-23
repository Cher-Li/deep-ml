# --- cell 1 ---
import numpy as np


def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):
    """Split rows into train/val/test folds, fit something on the training fold, predict the test fold.

    Returns
    -------
    test_predictions : np.ndarray, shape (len(test_idx),)
    train_idx, val_idx, test_idx : 1-D integer arrays forming a partition of range(len(y))
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError(f"X and y must have the same number of rows; got {len(X)} and {len(y)}.")

    n = len(y)
    fracs = np.array([train_frac, val_frac, test_frac], dtype=float)

    # normalize fractions so rounding always covers all n rows
    norm_fracs = fracs / fracs.sum()
    n_train = int(np.floor(n * norm_fracs[0]))
    n_val = int(np.floor(n * norm_fracs[1]))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).astype(int)

    # * splitting 
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    if len(train_idx) == 0:
        raise ValueError("train_idx is empty; increase train_frac or n.")

    # baseline on train only
    y_train = y[train_idx]
    baseline_value = np.mean(y_train, axis=0) # avg of y_train along rows
    pred_dtype = np.result_type(baseline_value.dtype, np.float64)
    
    # build output arr for test split
    test_predictions = np.full((len(test_idx),) + y.shape[1:], baseline_value, dtype=pred_dtype) 
    # constant mean baseline predicts same value for unseen ex 

    return test_predictions, train_idx, val_idx, test_idx
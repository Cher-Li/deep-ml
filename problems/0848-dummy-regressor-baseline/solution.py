import numpy as np

def dummy_regressor(y_train, n_test, strategy='mean', constant=None, quantile=None):
    """
    Baseline regressor that predicts a constant value derived from y_train.

    Args:
        y_train: 1D array-like of training target values.
        n_test: number of test predictions to return (int >= 0).
        strategy: one of 'mean', 'median', 'quantile', 'constant'.
        constant: required when strategy='constant'.
        quantile: required when strategy='quantile', must be in [0, 1].

    Returns:
        List[float] of length n_test, all equal to the chosen summary value.
    """
    y_train = np.asarray(y_train)
    
    if strategy == 'mean':
        p = float(np.mean(y_train))
    elif strategy == 'median':
        p = float(np.median(y_train))
    elif strategy == 'quantile':
        if quantile is None or not (0 <= quantile <= 1):
            raise ValueError("Quantile must be provided and be between 0 and 1 inclusive.")
        p = float(np.quantile(y_train, quantile))
    elif strategy == 'constant':
        if constant is None:
            raise ValueError("Constant value must be provided when strategy='constant'.")
        p = float(constant)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
        
    return [p] * n_test
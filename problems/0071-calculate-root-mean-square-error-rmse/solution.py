
import numpy as np

def rmse(y_true, y_pred):    
    # empty arr edge case
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Inputs cannot be empty.")
        
    # check if shapes match
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true shape {y_true.shape} must match y_pred shape {y_pred.shape}")
        
    # use np.mean instead of manually calc n
    rmse_res = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return round(rmse_res, 3)
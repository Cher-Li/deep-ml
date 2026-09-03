import numpy as np
from scipy import stats

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    arr = np.asarray(data)
    
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    mode_result = stats.mode(arr, keepdims=True)
    mode = float(mode_result.mode.item()) if arr.size > 0 else None

    variance = float(np.var(arr)) 
    sd = float(np.std(arr))
    
    quarter, half, three_quarter = np.percentile(arr, [25, 50, 75])
    iqr = three_quarter - quarter

    return {
        'mean': mean, 
        'median': median,
        'mode': mode,
        'variance': variance,
        'standard_deviation': sd,
        '25th_percentile': float(quarter),
        '50th_percentile': float(half),
        '75th_percentile': float(three_quarter),
        'interquartile_range': float(iqr)
    }
def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    # Your code here
    min_value = min(x) 
    max_value = max(x) 

    return [(val - min_value) / (max_value - min_value) for val in x] # * this way of going through all elmts
import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    # Your code here
    n = len(data)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    
    train_end = int(n * train_frac)
    val_end = train_end + int(n * validation_frac)
    
    train_data = data[indices[:train_end]]
    val_data = data[indices[train_end:val_end]]
    test_data = data[indices[val_end:]]
    
    return [train_data, val_data, test_data]
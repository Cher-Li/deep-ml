import numpy as np

def simulate_clt(distribution: str, n: int, runs: int = 10000, seed: int = 42) -> dict:
    """
    Simulate the Central Limit Theorem.

    Args:
        distribution (str): The distribution to sample from ('uniform', 'exponential', 'bernoulli').
        n (int): Sample size.
        runs (int): Number of repeated experiments.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: {'mean': float, 'std': float} of the standardized sample means.
    """
    np.random.seed(seed)
    # Your implementation here
    # * just directly use np.random._ for each
    if distribution == 'uniform':
        data = np.random.uniform(0, 1, size=(runs, n))
        mu, sigma = 0.5, np.sqrt(1/12)
    elif distribution == 'exponential':
        data = np.random.exponential(1.0, size=(runs, n))
        mu, sigma = 1.0, 1.0
    elif distribution == 'bernoulli':
        p = 0.3
        data = (np.random.rand(runs, n) < p).astype(float)
        mu, sigma = p, np.sqrt(p*(1-p))
    else:
        raise ValueError('Unsupported distribution')


    # *
    x_bar = data.mean(axis = 1) 
    # draw samples, compute means (np.mean?), standarize to Z scores, then return mean and std

    Z = (x_bar - mu) / (sigma / np.sqrt(n))
    return {'mean': float(Z.mean()), 'std': float(Z.std())} # get the mean and std of Z, yea they're functions
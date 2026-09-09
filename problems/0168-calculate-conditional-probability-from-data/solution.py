def conditional_probability(data, x, y):
    """
    Returns the probability P(Y=y|X=x) from list of (X, Y) pairs.
    Args:
      data: List of (X, Y) tuples
      x: value of X to condition on
      y: value of Y to check
    Returns:
      float: conditional probability, rounded to 4 decimal places
    """
    # Your code here
    
    # list of pairs so, directly return count of (X = x, Y = y) / count of X = x? 
    numerator = 0
    denom = 0
    for color, animal in data:
      if color == x and animal == y:
        numerator += 1
      if color == x:
        denom += 1
      
    if denom == 0:
      return 0
    return numerator / denom
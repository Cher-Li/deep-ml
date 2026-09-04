import numpy as np

def is_linearly_independent(vectors: list[list[float]]) -> bool:
    """
    Check if a set of vectors is linearly independent.
    
    Args:
        vectors: List of vectors, where each vector is a list of floats.
                 All vectors must have the same dimension.
        
    Returns:
        True if vectors are linearly independent, False otherwise.
    """
    # Empty set is linearly independent by convention
    if not vectors:
        return True

    # Create matrix with vectors as columns
    matrix = np.array(vectors, dtype=float).T

    # Vectors are independent iff rank equals number of vectors
    return np.linalg.matrix_rank(matrix) == len(vectors)

# import numpy as np

# def is_linearly_independent(vectors: list[list[float]]) -> bool:
#     """
#     Check if a set of vectors is linearly independent.
    
#     Args:
#         vectors: List of vectors, where each vector is a list of floats.
#                  All vectors must have the same dimension.
        
#     Returns:
#         True if vectors are linearly independent, False otherwise.
#     """
#     if not vectors:
#         return True
    
#     matrix = np.array(vectors, dtype=float)
#     num_vectors = len(vectors)
#     rank = np.linalg.matrix_rank(matrix)
    
#     return rank == num_vectors
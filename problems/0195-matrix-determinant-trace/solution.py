import numpy as np

def matrix_determinant_and_trace(matrix: list[list[float]]) -> tuple[float, float]:
	"""
	Compute the determinant and trace of a square matrix.
	
	Args:
		matrix: A square matrix (n x n) represented as list of lists
	
	Returns:
		Tuple of (determinant, trace)
	"""
	# Your code here
	# a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
	# return (a * d - b * c, a + d)

	det = np.linalg.det(matrix)
	trace = 0
	for i in range(len(matrix)):
		trace += matrix[i][i]
	
	return (det, trace)
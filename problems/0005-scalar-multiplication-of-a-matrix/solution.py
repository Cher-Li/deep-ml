def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	# Your code here

	# yea not this
	# for i in range(len(matrix)):
	# 	matrix[i] *= scalar
	# return matrix 

	for i in range(len(matrix)):
		for j in range(len(matrix[0])):
			matrix[i][j] *= scalar

	return matrix
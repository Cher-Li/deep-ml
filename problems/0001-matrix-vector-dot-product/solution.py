def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	
	# -1 if incompatible so check that first
	# num col in a = len(b)
	if len(a[0]) != len(b):
		return -1
	
	final = []
	# calc dot product
	for row in range(len(a)):
		sum = 0
		for col in range(len(a[0])):
			sum += a[row][col] * b[col]
		final.append(sum)
	
	return final
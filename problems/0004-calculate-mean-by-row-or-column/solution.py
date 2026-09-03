def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	answer = []

	if mode == 'row':
		# just go across, calc mean for each row
		for i in range(len(matrix)):
			cur_row = 0
			for j in range(len(matrix[0])):
				cur_row += matrix[i][j]
			
			answer.append(cur_row / len(matrix[0]))
	
	elif mode == 'column':
		# reverse operation, take the same col
		for j in range(len(matrix[0])):
			cur_col = 0
			for i in range(len(matrix)):
				cur_col += matrix[i][j]
			
			answer.append(cur_col / len(matrix))

	return answer
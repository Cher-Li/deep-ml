# *
import numpy as np

# iterations 
def gauss_seidel_it(A, b, x):
    rows, cols = A.shape
    for i in range(rows):
        x_new = b[i]
        for j in range(cols):
            if i != j:
				# b[i] - the sums then after that * 1/aii
                x_new -= A[i, j] * x[j]
        x[i] = x_new / A[i, i]
    return x

def gauss_seidel(A, b, n, x_ini=None):
    x = x_ini or np.zeros_like(b)
    for _ in range(n):
        x = gauss_seidel_it(A, b, x)
    return x

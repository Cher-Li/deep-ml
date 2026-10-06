# *
import numpy as np

def low_rank_approximation(delta_W: np.ndarray, r: int) -> list:
	"""
	Compute the best rank-r approximation of delta_W via truncated SVD.

	Args:
		delta_W: matrix of shape (m, n)
		r: target rank (1 <= r <= min(m, n))

	Returns:
		The rank-r approximation as a nested Python list of shape (m, n).
	"""
	# Your code here
	A = np.asarray(delta_W, dtype=float)
	U, S, Vt = np.linalg.svd(A, full_matrices=False)
	U_r = U[:, :r]
	S_r = S[:r]
	Vt_r = Vt[:r, :]
	A_r = (U_r * S_r) @ Vt_r
	return A_r.tolist()
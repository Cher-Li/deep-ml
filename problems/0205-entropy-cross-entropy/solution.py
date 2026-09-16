import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
	"""
	Compute entropy of P and cross-entropy between P and Q.
	
	Args:
		P: True probability distribution
		Q: Predicted probability distribution
	
	Returns:
		Tuple of (entropy H(P), cross-entropy H(P,Q))
	"""
	# Your code here
	cross_entropy = entropy = 0

	for i in range(len(P)):
		if P[i] == 0:
			entropy += 0
		else:
			entropy += P[i] * np.log(P[i])
		cross_entropy += P[i] * np.log(Q[i])
	entropy = -entropy
	cross_entropy = -cross_entropy

	return (entropy, cross_entropy)
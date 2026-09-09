def conditional_probability(joint_distribution: dict) -> float:
	"""
	Compute conditional probability P(A|B) from a joint probability distribution.

	Args:
		joint_distribution (dict): dictionary with keys ('A','B'), ('A','`B'), ('`A','B'), ('`A','`B')

	Returns:
		float: Conditional probability P(A|B)
	"""
	# Your code here
	
	# P(B) = values of ('A','B') + ('`A','B') 
	# P(A ∩ B) is ('A','B') 
	# P(A | B) = P(A ∩ B) / P(B)
	P_B = joint_distribution.get(('A','B')) + joint_distribution.get(('`A','B') ) 
	return joint_distribution.get(('A','B')) / P_B
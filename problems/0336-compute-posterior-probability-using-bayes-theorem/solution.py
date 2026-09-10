def bayes_theorem(priors: list[float], likelihoods: list[float]) -> list[float]:
    """
    Calculate posterior probabilities using Bayes' Theorem.
    
    Args:
        priors: Prior probabilities P(H_i) for each hypothesis
        likelihoods: Likelihoods P(E|H_i) for each hypothesis
        
    Returns:
        Posterior probabilities P(H_i|E) for each hypothesis
    """
    if len(priors) != len(likelihoods):
        raise ValueError("priors and likelihoods must have the same length")

    # Calculate unnormalized posteriors
    numerators = [p * l for p, l in zip(priors, likelihoods)]

    # Total probability of the evidence
    evidence_prob = sum(numerators)

    if evidence_prob == 0:
        raise ValueError("Total evidence probability is zero")

    # Normalize to get posterior probabilities
    return [n / evidence_prob for n in numerators]

# def bayes_theorem(priors: list[float], likelihoods: list[float]) -> list[float]:
# 	"""
# 	Calculate posterior probabilities using Bayes' Theorem.
	
# 	Args:
# 		priors: Prior probabilities P(H_i) for each hypothesis
# 		likelihoods: Likelihoods P(E|H_i) for each hypothesis
		
# 	Returns:
# 		Posterior probabilities P(H_i|E) for each hypothesis
# 	"""
# 	# Calculate numerators: P(E|H_i) * P(H_i) for each hypothesis
# 	numerators = [p * l for p, l in zip(priors, likelihoods)]
	
# 	# Calculate total evidence probability (denominator)
# 	evidence = sum(numerators)
	
# 	# Handle edge case where evidence is zero
# 	if evidence == 0:
# 		return [0.0] * len(priors)
	
# 	# Calculate posterior probabilities
# 	posteriors = [n / evidence for n in numerators]
	
# 	return posteriors
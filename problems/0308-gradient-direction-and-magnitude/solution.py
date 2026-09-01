import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
    """
    Calculate the magnitude and direction of a gradient vector.
    
    Args:
        gradient: A list representing the gradient vector
    
    Returns:
        Dictionary containing:
        - magnitude: The L2 norm of the gradient
        - direction: Unit vector in direction of steepest ascent
        - descent_direction: Unit vector in direction of steepest descent
    """
    
    # Convert input to NumPy array
    grad = np.array(gradient, dtype=float)
    
    # Compute L2 norm (magnitude)
    magnitude = np.linalg.norm(grad)
    
    # Handle zero vector case
    if magnitude == 0:
        zero_vector = [0.0] * len(gradient)
        return {
            'magnitude': 0.0,
            'direction': zero_vector,
            'descent_direction': zero_vector
        }
    
    # Compute unit direction vector
    direction = (grad / magnitude).tolist()
    
    # Compute descent direction
    descent_direction = (-grad / magnitude).tolist()
    
    return {
        'magnitude': float(magnitude),
        'direction': direction,
        'descent_direction': descent_direction
    }

# import numpy as np

# def gradient_direction_magnitude(gradient: list) -> dict:
# 	"""
# 	Calculate the magnitude and direction of a gradient vector.
	
# 	Args:
# 		gradient: A list representing the gradient vector
	
# 	Returns:
# 		Dictionary containing:
# 		- magnitude: The L2 norm of the gradient
# 		- direction: Unit vector in direction of steepest ascent
# 		- descent_direction: Unit vector in direction of steepest descent
# 	"""
# 	grad = np.array(gradient, dtype=float)
	
# 	# Calculate magnitude (L2 norm)
# 	magnitude = np.linalg.norm(grad)
	
# 	# Handle zero gradient case
# 	if magnitude < 1e-10:
# 		direction = np.zeros_like(grad)
# 		descent_direction = np.zeros_like(grad)
# 	else:
# 		# Direction is the unit vector (steepest ascent)
# 		direction = grad / magnitude
# 		# Steepest descent is negative of gradient direction
# 		descent_direction = -direction
	
# 	return {
# 		'magnitude': float(magnitude),
# 		'direction': direction.tolist(),
# 		'descent_direction': descent_direction.tolist()
# 	}
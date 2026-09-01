def poly_term_derivative(c: float, x: float, n: float) -> float:
    # Your code here
    
    # c * x^n so deriv is n * c * x ^ (n-1)

    return n * c * x ** (n - 1)
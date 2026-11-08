import numpy as np
from collections import Counter
import math

def dummy_classifier(y_train, n_test, strategy, constant=None):
    """
    Produce baseline predictions of length n_test using the given strategy.
    Returns a Python list of predicted labels.
    """
    if not y_train:
        raise ValueError("y_train cannot be empty.")
        
    sorted_classes = sorted(list(set(y_train)))
    
    if strategy == "constant":
        return [constant] * n_test
    elif strategy == "most_frequent":
        counts = Counter(y_train)
        # most frequent class, break ties by smallest class label
        best_class = min(sorted_classes, key=lambda c: (-counts[c], c))
        return [best_class] * n_test
    elif strategy == "uniform":
        k = len(sorted_classes)
        return [sorted_classes[i % k] for i in range(n_test)]
    elif strategy == "stratified":
        # each class c, assign floor() predictions
        N = len(y_train)
        counts = Counter(y_train)
        
        allocations = {}
        fractional_parts = {}
        
        for c in sorted_classes:
            fc = counts[c] / N
            product = n_test * fc
            base = math.floor(product)
            allocations[c] = base
            fractional_parts[c] = product - base
            
        total_allocated = sum(allocations.values())
        remainder = n_test - total_allocated
        
        if remainder > 0:
            # sort by largest fractional part, breaking ties by smallest class label again
            sorted_for_rem = sorted(sorted_classes, key=lambda c: (-fractional_parts[c], c))
            for i in range(remainder):
                c = sorted_for_rem[i]
                allocations[c] += 1
                
        # Output predictions grouped in sorted class order
        predictions = []
        for c in sorted_classes:
            predictions.extend([c] * allocations[c])
            
        return predictions
        
    else:
        raise ValueError(f"Unknown strategy: '{strategy}'")
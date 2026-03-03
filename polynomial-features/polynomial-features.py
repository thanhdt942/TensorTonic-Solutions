import numpy as np
def polynomial_features(values, degree):
    """
    Generate polynomial features for each value up to the given degree.
    """
    # Write code here
    result = []
    
    for x in values:
        row = []
        for d in range(degree + 1):
            row.append(x ** d)
        result.append(row)
    
    return result
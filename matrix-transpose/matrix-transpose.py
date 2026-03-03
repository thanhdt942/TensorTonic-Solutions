import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.array(A)

    # return A.T
    # return A.transpose()
    # return np.transpose(A)
    return np.einsum('ij->ji', A)

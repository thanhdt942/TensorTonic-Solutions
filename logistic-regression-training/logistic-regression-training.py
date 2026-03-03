import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    n_samples, n_features = X.shape
    
    # Initialize
    w = np.zeros(n_features)
    b = 0.0
    
    # Gradient descent loop
    for _ in range(steps):
        
        # Linear part
        z = X @ w + b
        
        # Sigmoid
        y_hat = _sigmoid(z)
        
        # Error
        error = y_hat - y
        
        # Gradients
        dw = (1/n_samples) * (X.T @ error)
        db = (1/n_samples) * np.sum(error)
        
        # Update
        w -= lr * dw
        b -= lr * db
    
    return w, b
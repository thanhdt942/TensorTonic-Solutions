import numpy as np

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code here
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    
    # Cosine similarity
    cos_sim = np.dot(x1, x2) / (
        np.linalg.norm(x1) * np.linalg.norm(x2) + 1e-12
    )
    
    if label == 1:
        loss = 1 - cos_sim
    else:  # label == -1
        loss = max(0.0, cos_sim - margin)
    
    return loss
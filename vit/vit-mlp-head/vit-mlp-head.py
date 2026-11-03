import numpy as np

def classification_head(encoder_output: np.ndarray,
                        W_head: np.ndarray) -> np.ndarray:
    """
    Returns float64 class logits with shape (B, C).
    """
    # 1. Extract CLS token at position 0: (B, N, D) -> (B, D)
    h = encoder_output[:, 0, :]

    # 2. LayerNorm over embedding dimension D
    mean = np.mean(h, axis=-1, keepdims=True)
    var = np.var(h, axis=-1, keepdims=True)
    h_norm = (h - mean) / np.sqrt(var + 1e-6)

    # 3. Linear projection: (B, D) @ (D, C) -> (B, C)
    logits = h_norm @ W_head

    return logits.astype(np.float64)
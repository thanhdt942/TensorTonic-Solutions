import numpy as np

def patch_embed(image: np.ndarray, patch_size: int,
                W_proj: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Returns float64 patch embeddings with shape (B, N, D).
    """
    image = np.asarray(image, dtype=np.float64)
    W_proj = np.asarray(W_proj, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)

    B, H, W, C = image.shape

    P = patch_size
    
    if H % P != 0 or W % P != 0:
        H = (H // P) * P 
        W = (W // P) * P
        image = image[:, :H, :W, :]

    image = image.reshape(
        B,
        H // P, P,
        W // P, P,
        C
    )

    image = image.transpose(0, 1, 3, 2, 4, 5)

    N = (H // P) * (W // P)

    patch_dim = P * P * C
    
    patches = image.reshape(B, N, patch_dim)

    embeddings = patches @ W_proj + bias

    return embeddings.astype(np.float64)
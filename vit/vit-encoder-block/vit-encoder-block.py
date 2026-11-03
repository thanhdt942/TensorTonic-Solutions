import numpy as np

def vit_encoder_block(x: np.ndarray, num_heads: int,
                      Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                      Wo: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """
    Returns the float64 output of one pre-normalized ViT encoder block.
    """
    x = np.asarray(x, dtype=np.float64)
    Wq, Wk, Wv, Wo, W1, W2 = [
        np.asarray(w, dtype=np.float64)
        for w in (Wq, Wk, Wv, Wo, W1, W2)
    ]

    B, N, D = x.shape

    if num_heads <= 0 or D % num_heads != 0:
        raise ValueError("D must be divisible by num_heads.")

    head_dim = D // num_heads
    eps = 1e-6

    # LayerNorm: normalize each token over the embedding dimension.
    def layer_norm(t):
        mean = np.mean(t, axis=-1, keepdims=True)
        var = np.var(t, axis=-1, keepdims=True)
        return (t - mean) / np.sqrt(var + eps)

    # GELU activation (exact formulation).
    def gelu(t):
        return 0.5 * t * (
            1.0 + np.tanh(
                np.sqrt(2.0 / np.pi) * (t + 0.044715 * t**3)
            )
        )

    # 1. Pre-LN before Multi-Head Self-Attention.
    x_norm = layer_norm(x)

    # 2. Linear projections: Q, K, V. 
    # x (B, N, D) 
    # wq (D, D)
    # Q = (B, N, D) @ (D, D) = (B, N, D)
    Q = x_norm @ Wq
    K = x_norm @ Wk
    V = x_norm @ Wv

    # 3. Split into attention heads: (B, H, N, head_dim).
    Q = Q.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)

    # 4. Scaled dot-product attention.
    scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)

    # Numerically stable softmax over the key-token dimension.
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    attn = np.exp(scores)
    attn = attn / np.sum(attn, axis=-1, keepdims=True)

    context = attn @ V

    # 5. Concatenate heads and apply output projection.
    context = context.transpose(0, 2, 1, 3).reshape(B, N, D)
    attn_output = context @ Wo

    # 6. First residual connection.
    y = x + attn_output

    # 7. Pre-LN + two-layer MLP with GELU.
    y_norm = layer_norm(y)
    mlp_output = gelu(y_norm @ W1) @ W2

    # 8. Second residual connection.
    z = y + mlp_output

    return z.astype(np.float64)
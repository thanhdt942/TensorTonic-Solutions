import numpy as np

def add_position_embedding(patches: np.ndarray,
                           pos_embed: np.ndarray) -> np.ndarray:
    """
    Returns the float64 tokens after adding position embeddings.
    """
    patches = np.asarray(patches, dtype=np.float64)
    pos_embed = np.asarray(pos_embed, dtype=np.float64)

    tokens = patches + pos_embed

    return tokens.astype(np.float64)
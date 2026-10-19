import numpy as np

def prepend_class_token(patches: np.ndarray,
                        cls_token: np.ndarray) -> np.ndarray:
    """
    Returns the float64 sequence with the class token at position zero.
    """
    patches = np.asarray(patches, dtype=np.float64)
    cls_token = np.asarray(cls_token, dtype=np.float64)

    B = patches.shape[0]

    cls_tokens = np.tile(cls_token, (B, 1, 1))

    tokens = np.concatenate(
        [cls_tokens, patches],
        axis=1
    )

    return tokens

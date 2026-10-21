"""Small educational demonstration of scaled dot-product attention.

This is not a full Transformer implementation. It demonstrates the core
attention calculation using hand-written, tiny matrices and NumPy.
"""
import numpy as np

def scaled_dot_product_attention(query, key, value):
    """Return attention output and attention weights."""
    query = np.asarray(query, dtype=float)
    key = np.asarray(key, dtype=float)
    value = np.asarray(value, dtype=float)
    if query.ndim != 2 or key.ndim != 2 or value.ndim != 2:
        raise ValueError("Query, key, and value must be 2D matrices.")
    if query.shape[1] != key.shape[1]:
        raise ValueError("Query and key dimensions must match.")
    if key.shape[0] != value.shape[0]:
        raise ValueError("Key and value sequence lengths must match.")
    scores = query @ key.T / np.sqrt(query.shape[1])
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ value, weights

if __name__ == "__main__":
    # Three toy token vectors with two features each.
    Q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    V = np.array([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]])
    output, weights = scaled_dot_product_attention(Q, K, V)
    print("Attention weights:")
    print(np.round(weights, 3))
    print("\\nAttention output:")
    print(np.round(output, 3))

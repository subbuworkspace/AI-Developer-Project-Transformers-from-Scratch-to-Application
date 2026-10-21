import pytest
from transformer_app import analyze_sentiment
from attention_demo import scaled_dot_product_attention
import numpy as np

def test_empty_text_is_rejected_before_model_loading():
    with pytest.raises(ValueError):
        analyze_sentiment("   ")

def test_attention_output_shapes_and_weights_sum_to_one():
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    k = np.array([[1.0, 0.0], [0.0, 1.0]])
    v = np.array([[2.0, 0.0], [0.0, 2.0]])
    output, weights = scaled_dot_product_attention(q, k, v)
    assert output.shape == (2, 2)
    assert weights.shape == (2, 2)
    assert np.allclose(weights.sum(axis=1), 1.0)

def test_attention_rejects_incompatible_dimensions():
    with pytest.raises(ValueError):
        scaled_dot_product_attention([[1, 2]], [[1, 2, 3]], [[1, 2]])

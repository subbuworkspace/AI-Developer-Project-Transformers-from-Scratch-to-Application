"""Inference utilities for a pretrained Transformer sentiment classifier."""
from functools import lru_cache
from transformers import pipeline

DEFAULT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

@lru_cache(maxsize=2)
def load_classifier(model_name: str = DEFAULT_MODEL):
    """Load and cache the selected Hugging Face text-classification pipeline."""
    return pipeline("text-classification", model=model_name)

def analyze_sentiment(text: str, model_name: str = DEFAULT_MODEL) -> dict:
    """Classify a non-empty text string and return label and score."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please provide non-empty text.")
    classifier = load_classifier(model_name)
    result = classifier(text.strip(), truncation=True, max_length=512)[0]
    return {"label": str(result["label"]), "score": float(result["score"])}

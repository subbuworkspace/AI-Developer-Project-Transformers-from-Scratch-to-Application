"""Transformer Sentiment Analyzer: Streamlit UI."""
import streamlit as st
from transformer_app import analyze_sentiment

st.set_page_config(page_title="Transformer Sentiment Analyzer", page_icon="🧠")
st.title("🧠 Transformer Sentiment Analyzer")
st.write("Explore how a pretrained Transformer classifies text sentiment.")

with st.sidebar:
    st.header("Model settings")
    model_name = st.text_input(
        "Hugging Face model",
        value="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        help="The model and tokenizer are downloaded on first run.",
    )
    st.caption("This checkpoint is trained for English positive/negative sentiment.")

examples = [
    "The service was excellent and the staff were helpful.",
    "The delivery was late and the product was damaged.",
    "The product arrived today.",
]
choice = st.selectbox("Try an example", ["Write my own", *examples])
default_text = "" if choice == "Write my own" else choice
text = st.text_area("Enter text", value=default_text, height=120,
                    placeholder="Type a review, feedback message, or short sentence...")
if st.button("Analyze sentiment", type="primary", disabled=not bool(text.strip())):
    try:
        with st.spinner("Loading model and analyzing text..."):
            result = analyze_sentiment(text, model_name=model_name)
        st.subheader("Prediction")
        col1, col2 = st.columns(2)
        col1.metric("Label", result["label"])
        col2.metric("Model confidence", f"{result['score']:.1%}")
        st.progress(min(max(result["score"], 0.0), 1.0))
        st.caption("Confidence is the model's score, not a guarantee of correctness.")
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.info("Check your internet connection, model name, and installed dependencies.")

st.divider()
st.subheader("What happens inside?")
st.markdown(
    "1. **Tokenization:** text becomes token IDs.  \n"
    "2. **Embeddings:** token and position information become vectors.  \n"
    "3. **Self-attention:** each token representation incorporates contextual information.  \n"
    "4. **Transformer layers:** attention and feed-forward blocks refine the representations.  \n"
    "5. **Classification head:** produces sentiment scores."
)
st.caption("Educational demo. Do not use sentiment predictions alone for consequential decisions.")
''',
"transformer_app.py": r'''
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

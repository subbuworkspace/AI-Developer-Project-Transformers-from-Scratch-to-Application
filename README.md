# 🚀 Transformer Architecture — AI Developer Project

A hands-on project to understand **Transformer architecture, Self-Attention, Query-Key-Value (QKV), and pre-trained language models** using Python, PyTorch, and Hugging Face.

This repository combines simple implementations with a practical NLP application to help beginners understand how modern AI models process text.

## 🎯 Project Objectives

- Understand Transformer architecture from fundamentals to practical applications.
- Learn how Self-Attention and scaled dot-product attention work.
- Explore tokenization, embeddings, positional information, and attention weights.
- Build a sentiment-analysis application using a pre-trained Transformer model.
- Practice core concepts through Python code and interview questions.

## 🧠 What Is a Transformer?

A Transformer is a neural network architecture introduced in the research paper *Attention Is All You Need* (2017).

Unlike traditional recurrent neural networks (RNNs), Transformers can process token representations in parallel during training and use attention mechanisms to model relationships between tokens.

Transformers are widely used in:

- Large Language Models (LLMs)
- Text classification and sentiment analysis
- Machine translation
- Text summarization
- Question answering
- Code generation
- Multimodal AI systems

## 🏗️ Transformer Architecture

A traditional encoder-decoder Transformer contains two major components:

**1. Encoder**
- Converts input tokens into contextual representations.
- Uses Multi-Head Self-Attention.
- Applies feed-forward neural networks.
- Uses residual connections and layer normalization.

**2. Decoder**
- Generates output tokens autoregressively.
- Uses masked self-attention to prevent access to future tokens during generation.
- Uses encoder-decoder attention in the original encoder-decoder architecture.

### Processing Pipeline

```text
Input Text
    ↓
Tokenization
    ↓
Token Embeddings + Positional Information
    ↓
Transformer Layers
    ├── Multi-Head Attention
    ├── Feed-Forward Network
    ├── Residual Connections
    └── Layer Normalization
    ↓
Contextual Representations
    ↓
Task-Specific Output
```

> Note: Modern Transformer models differ. BERT-style models primarily use encoders, GPT-style models use decoder-only architectures, and T5-style models use encoder-decoder architectures.

## 🔍 Core Concepts

| Concept | Explanation |
|---|---|
| Tokenization | Splits text into tokens or subword units. |
| Embeddings | Represents tokens as numerical vectors. |
| Positional information | Provides information about token positions or order. |
| Self-Attention | Determines how tokens relate to other tokens in context. |
| Query (Q) | Represents what a token is looking for. |
| Key (K) | Represents what a token offers for matching. |
| Value (V) | Represents information that contributes to the output. |
| Multi-Head Attention | Learns relationships through multiple attention heads. |
| Feed-Forward Network | Transforms each token representation independently within a layer. |
| Residual Connection | Adds a layer's input to its output. |
| Layer Normalization | Helps stabilize model training. |
| Fine-tuning | Adapts a pre-trained model to a specific task or dataset. |

## 📐 Scaled Dot-Product Attention

The core attention calculation is:

\[
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

Where:

- \(Q\): Query matrix
- \(K\): Key matrix
- \(V\): Value matrix
- \(d_k\): Dimension of each key vector

The dot products measure compatibility between queries and keys. Softmax converts the scores into attention weights, which are used to combine the value vectors.

## 🛠️ Projects Included

### Project 1: Sentiment Analysis with a Pre-trained Transformer

**File:** `main.py` and `transformer_app.py`

Build an interactive application that classifies text sentiment using Hugging Face Transformers.

**Features**
- Accepts user-provided text.
- Uses a pre-trained DistilBERT sentiment classifier.
- Returns the predicted sentiment label and confidence score.
- Provides a simple Streamlit interface.

**Model:** `distilbert/distilbert-base-uncased-finetuned-sst-2-english`

### Project 2: Scaled Dot-Product Attention from Scratch

**File:** `attention_demo.py`

Implement the mathematical attention mechanism using NumPy.

**What you learn**
- Query, Key, and Value matrices
- Matrix multiplication
- Scaling attention scores
- Softmax normalization
- Attention weights and weighted value outputs

This is a small educational implementation, not a complete Transformer model.

### Project 3: Automated Tests

**File:** `tests/test_transformer_app.py`

Tests basic input validation and the attention implementation's output shapes, normalized weights, and invalid dimensions.

## 📁 Repository Structure

```text
transformer-ai-developer-project/
│
├── main.py
├── transformer_app.py
├── attention_demo.py
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
├── INTERVIEW_QUESTIONS.md
├── LINKEDIN_POST.txt
├── outputs/
│   └── example_run.txt
└── tests/
    └── test_transformer_app.py
```

## ⚙️ Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/transformer-ai-developer-project.git
cd transformer-ai-developer-project
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 2: Create a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch installation can vary by operating system and hardware. If necessary, follow the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### Step 4: Run the attention demonstration

```bash
python attention_demo.py
```

The script displays example attention weights and the resulting weighted value representations.

### Step 5: Run the sentiment-analysis application

```bash
streamlit run main.py
```

Open the local URL displayed in the terminal, enter a sentence, and inspect the predicted sentiment.

The first run may download the model weights and tokenizer, requiring an internet connection and additional disk space.

### Step 6: Run tests

```bash
pytest -q
```

## 🧪 Example

Input:

```text
The product is excellent and the service was amazing.
```

Expected type of output:

```text
Predicted sentiment: POSITIVE
Score: [model-generated confidence score]
```

The score depends on the model's actual inference. This example illustrates the expected output format; it is not a measured benchmark result.

## 🔄 How the Sentiment Application Works

1. The user enters a sentence.
2. The tokenizer converts the text into token IDs and model inputs.
3. DistilBERT processes the input using Transformer attention layers.
4. A classification head produces sentiment scores.
5. The application displays the predicted label and score.

This project uses a pre-trained, fine-tuned model. It does **not** train a Transformer from scratch.

## ⚖️ Transformer vs. RNN vs. LSTM

| Feature | RNN | LSTM | Transformer |
|---|---|---|---|
| Core mechanism | Recurrent state | Gated recurrent state | Attention |
| Sequential processing | Yes | Yes | Parallelizable during training |
| Long-range relationships | Can be difficult | Improved over basic RNN | Attention can directly connect token positions |
| Training efficiency | Often sequential | Often sequential | Often better parallelism |
| Common applications | Sequential data | Time series and sequences | NLP, LLMs, vision and multimodal AI |

Transformers can still have substantial computational and memory costs, particularly for long sequences.

## 💼 Real-World Applications

- Banking customer feedback analysis
- Customer support ticket classification
- Fraud-related text triage
- Document classification
- Semantic search
- Summarization and question answering
- AI assistants and chatbots

For banking applications, model predictions should be evaluated against representative data and supported by privacy, fairness, security, and human-review controls.

## 🎤 Interview Preparation

Review the following questions before interviews:

1. What is a Transformer, and why was it introduced?
2. What problem does Self-Attention solve?
3. What are Query, Key, and Value?
4. Why do we divide attention scores by \(\sqrt{d_k}\)?
5. What is Multi-Head Attention?
6. Why is positional information needed?
7. What is the difference between an encoder and a decoder?
8. How does masked attention work?
9. What is the difference between BERT and GPT?
10. What is the difference between pre-training and fine-tuning?
11. What is DistilBERT, and how does it relate to BERT?
12. What does a tokenizer do?
13. How do you evaluate a sentiment classifier?
14. What is overfitting in fine-tuning?
15. What are the limitations of attention for long sequences?

See [`INTERVIEW_QUESTIONS.md`](INTERVIEW_QUESTIONS.md) for the expanded question bank.

## 📚 Learning Resources

- Original research paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- PyTorch Transformer API: [torch.nn.Transformer](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
- Hugging Face documentation: [Transformers](https://huggingface.co/docs/transformers)
- Text classification guide: [Hugging Face task guide](https://huggingface.co/docs/transformers/tasks/sequence_classification)
- Example model: [DistilBERT SST-2](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)

## 🚀 Future Improvements

- Add a custom labeled dataset and fine-tune a classifier.
- Compare DistilBERT with a second model.
- Evaluate accuracy, precision, recall, and F1-score.
- Add confusion-matrix visualization.
- Build a REST API using FastAPI.
- Containerize the application with Docker.
- Experiment with parameter-efficient fine-tuning methods such as LoRA.

## 👨‍💻 Skills Demonstrated

`Python` · `NumPy` · `PyTorch` · `Hugging Face` · `NLP` · `Transformers` · `Attention Mechanism` · `Streamlit` · `Unit Testing` · `GitHub`

## 📄 License

This project is distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

**Learning by building:** Understand the mathematics, implement attention, run a pre-trained Transformer, and gradually progress toward fine-tuning and production-ready AI applications.

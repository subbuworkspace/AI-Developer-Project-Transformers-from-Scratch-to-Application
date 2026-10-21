# Transformer Interview Questions and Answers

## Fundamentals

1. **What is a Transformer?** A neural network architecture that uses attention mechanisms and feed-forward layers to process sequences without recurrence.
2. **Which paper introduced the Transformer?** *Attention Is All You Need* by Vaswani et al., published in 2017.
3. **Why were Transformers a major advance over RNNs?** They support parallel processing across sequence positions during training and model long-range relationships through attention.
4. **What is a token?** A unit of text processed by a tokenizer; it may be a word, subword, character, or other unit.
5. **What is tokenization?** Converting raw text into tokens and then token IDs that the model can process.
6. **What are embeddings?** Learned dense vectors representing token identities and, in some systems, positions or other information.
7. **Why is positional information needed?** Self-attention alone does not inherently encode token order; position information helps the model distinguish sequence order.
8. **What is self-attention?** A mechanism that lets each token representation weigh information from other tokens in the same sequence.
9. **What are Query, Key, and Value?** Learned projections of token representations: queries seek relevant information, keys are matched against queries, and values carry the information combined into outputs.
10. **What is scaled dot-product attention?** `softmax(QKᵀ / √d_k)V`, where scaling helps control score magnitude.

## Architecture

11. **What is multi-head attention?** Several attention operations run in different learned representation subspaces; their outputs are combined.
12. **Why divide by √d_k?** Dot products tend to grow in magnitude with dimension; scaling helps keep softmax gradients in a useful range.
13. **What is a feed-forward network in a Transformer block?** A position-wise neural network applied to each token representation after attention.
14. **What is residual connection?** A skip connection that adds a block's input to its output, helping information and gradients flow.
15. **What is LayerNorm?** A normalization operation over features that helps stabilize training.
16. **What is dropout?** A regularization method that randomly drops activations during training.
17. **What is an attention mask?** A constraint that prevents attention to certain positions, such as padding tokens or future tokens in causal language modeling.
18. **What is causal attention?** Attention that blocks access to future tokens so generation can proceed left to right.
19. **What is an encoder-only Transformer?** A model such as BERT that builds bidirectional contextual representations and is commonly used for classification or token-level tasks.
20. **What is a decoder-only Transformer?** A model such as GPT that predicts next tokens using causal attention.
21. **What is an encoder-decoder Transformer?** An architecture with an encoder for input representations and a decoder that generates outputs, often used in translation and summarization.
22. **BERT vs GPT?** BERT is encoder-based and commonly pretrained with masked-token objectives; GPT-style models are decoder-based and trained for autoregressive next-token prediction.
23. **What is cross-attention?** Attention in which queries come from one sequence and keys/values come from another, as in encoder-decoder models.
24. **What is an LM head?** A projection that converts hidden states into vocabulary logits for token prediction.
25. **What are logits?** Raw model scores before a probability transformation such as softmax.

## Training and inference

26. **What is pretraining?** Learning general representations or language patterns from large datasets before task-specific adaptation.
27. **What is fine-tuning?** Further training a pretrained model on a task- or domain-specific dataset.
28. **What is masked language modeling?** Predicting selected masked tokens from surrounding context, commonly associated with BERT-style pretraining.
29. **What is next-token prediction?** Learning to predict the next token given previous tokens, commonly used for causal language models.
30. **What is teacher forcing?** Training a sequence model using the ground-truth previous tokens as decoder inputs.
31. **What is perplexity?** An exponentiated average negative log-likelihood measure often used to assess language-model predictions; lower is better on the same evaluation setup.
32. **What is temperature in generation?** A sampling parameter that changes the sharpness of the token probability distribution.
33. **What is context length?** The maximum number of tokens a model can process in one context window, subject to its configuration.
34. **Why can long contexts be expensive?** Standard full self-attention has roughly quadratic attention-score computation and memory growth with sequence length.
35. **What is teacher-free inference?** At generation time, the model uses its own previous outputs rather than ground-truth next tokens.

## Practical and advanced

36. **What is DistilBERT?** A smaller distilled Transformer designed to retain much of a larger model's capability with reduced size and faster inference.
37. **What is transfer learning?** Reusing knowledge learned from a large dataset or pretrained model for a related task.
38. **What is Hugging Face Transformers?** An open-source library for loading, training, and running many pretrained Transformer models.
39. **What is a pipeline?** A high-level API that packages preprocessing, model inference, and postprocessing for a task.
40. **What does the sentiment project do?** Tokenizes input, runs a pretrained sequence classifier, and returns a sentiment label and score.
41. **What is attention complexity?** Standard self-attention's score computation scales approximately as O(n²d) with sequence length n and hidden dimension d.
42. **What is an attention head?** One learned attention subspace with its own query, key, and value projections.
43. **What is knowledge distillation?** Training a smaller student model to imitate a larger teacher model's behavior or output distributions.
44. **What is quantization?** Representing model weights or activations with lower precision to reduce memory and sometimes improve speed.
45. **What is LoRA?** Low-Rank Adaptation, a parameter-efficient fine-tuning approach that learns small low-rank updates rather than updating all base-model weights.
46. **What is a hallucination?** An output that is unsupported or incorrect; Transformer architecture alone does not guarantee factuality.
47. **How do you evaluate sentiment classification?** Use a held-out labeled dataset and metrics such as accuracy, precision, recall, F1, and confusion matrix.
48. **How do you avoid data leakage?** Split data before fitting task-specific preprocessing or selecting hyperparameters, and keep test data isolated until final evaluation.
49. **What are Transformer limitations?** Compute and memory demands, context limits, data bias, sensitivity to distribution shifts, and potentially unreliable predictions.
50. **When would you use a simpler model?** When a linear model or small neural network meets quality, latency, cost, and interpretability requirements.

## Scenario questions

- Explain Q, K, and V using a sentence example.
- Why can a model misclassify sarcasm?
- How would you fine-tune DistilBERT on labeled customer feedback?
- How would you reduce inference latency on CPU?
- How do you compare BERT, DistilBERT, and a decoder-only LLM for classification?
- What changes are needed to handle long documents?
- Why is a high confidence score not proof that a prediction is correct?
- How would you monitor model drift after deployment?

## Recommended answer structure

**Concept → mechanism → use case → trade-off → evaluation.** State whether a result is measured or only an illustrative example.

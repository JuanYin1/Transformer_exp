
# Concept table
| Module | Primary Topic | Core Concepts | Model Architectures | Training and Evaluation Methods | Mathematical Components | Source |
| --- | --- | --- | --- | --- | --- | --- |
| Modern LMs: Key Ingredients | Transformers & Self-Attention | Parallel computation, context-sensitive representations, Query/Key/Value paradigm, Multi-head attention | Transformer Encoder, Transformer Decoder, BERT, GPT | Beam Search, Greedy Decoding, Nucleus (Top-p) Sampling, Layer Normalization, Residual Connections | Scaled Dot-Product Attention, Softmax, Positional Encoding (Sine/Cosine), GeLU | [1-4] |
| Modern LMs: Key Ingredients | Pretrained Language Models | Transfer learning, bi-directional vs. unidirectional context, masked language modeling | ELMo, BERT, GPT-n, RoBERTa, ELECTRA, DeBERTa, T5 | Pre-training, Fine-tuning, Parameter-efficient fine-tuning (PEFT), Reinforcement Learning from Human Feedback (RLHF) | Masked language model objective, Causal masking, Cross-Entropy | [1, 5, 6] |
| Modern LMs: Background | Recurrent Neural Networks | Processing variable-length sequences, summarization of context in hidden states, vanishing/exploding gradients | Vanilla RNN, Bidirectional RNN, Multi-layer RNN, LSTM, GRU | Perplexity, Gradient Clipping, Backpropagation Through Time (BPTT) | tanh, Gating mechanisms (Input, Forget, Output gates), Sigmoid | [1, 2, 7] |
| Modern LMs: Background | Sequence-to-Sequence Models | Conditional language modeling, mapping one sequence to another, information bottleneck | Seq2Seq Encoder-Decoder RNN | Teacher Forcing, BLEU score evaluation | Autoregressive conditional probability | [1, 2, 8] |
| Intro to Neural NLP | Feedforward Neural Networks | Adding nonlinearity to linear models, composition of logistic regressions, model capacity | Feedforward Neural Networks (FFN), Deep Averaging Networks (DAN) | Backpropagation, Dropout, L2 Weight Decay, Learning Rate Schedules (Step, Cosine) | ReLU, Sigmoid, tanh, GeLU activation functions | [1, 6, 9] |
| Intro to Neural NLP | Word Embeddings | Distributional Semantics (words in similar contexts have similar meanings), dense vs. sparse vectors | Word2Vec (Skip-gram, CBOW), GloVe, FastText | Intrinsic Evaluation (Clustering, Analogies), Extrinsic Evaluation (Downstream tasks), Negative Sampling | Dot product similarity, L2 Regularization, Skip-gram loss function | [1, 2, 5, 10] |
| Intro to Neural NLP | Linear & Softmax Models | Categorization of text into classes, unnormalized scores to probabilities, Maximum Likelihood Estimation (MLE) | Linear Classifier, Softmax Classifier | Stochastic Gradient Descent (SGD), Minibatch Gradient Descent, Maximum Likelihood Estimation (MLE) | Softmax function, Cross-Entropy Loss, Dot product | [1, 11-13] |
| Intro to Neural NLP | Tokenization | Translation layer between raw text and IDs, Out-of-Vocabulary (OOV) problem, subword units | Tokenizer module (independent of main model) | Byte Pair Encoding (BPE) frequency-based merging | Frequency statistics, Byte-level BPE |  |


# Key Words

### encoder: 
1. position embedding -> encoder block
2. encoder block include: (softmax transfer dot-product simularity score to attention weight which sums to 1) multihead self-attention with Add/Norm, FFNN wth Add/Norm, 

### decoder:
1. position embedding -> decoder block -> projection layer -> softmax layer for probabilities
2. decoder block inclaude: masked multihead self-attention with Add/Norm, multihead cross-attention with Add/Norm, FFNN wth Add/Norm

Add/Norm include: residual connection, Layer Norm / Batch Norm


### position encoding:

#### Traditional approach:
  - self.pos_embedding = nn.Embedding(self.block_size, self.n_embed) - learnable position embeddings
  - pos_emb = self.pos_embedding(pos) - look up position embeddings
  - x = tok_emb + pos_emb - add token + position embeddings

#### AliBi approach:
  - Only token embeddings (no positional embeddings with AliBi)
  - x = self.embed_dropout(tok_emb) - NO position embeddings added
  - Position info comes from AliBi biases in attention instead

#### Summary of the Three Approaches:

  | Approach                | Location            | How Position is Encoded                       |
  |-------------------------|---------------------|-----------------------------------------------|
  | Traditional (Parts 1&2) | tok_emb + pos_emb   | Learnable position embeddings added to tokens |
  | AliBi (Part 3)          | scores + alibi_bias | Linear biases added to attention scores       |
  | Local Window (Part 3)   | tok_emb + pos_emb   | Traditional + limited attention window        |


#### Complete Positional Encoding Comparison

  | Type                 | Memory | Computation | Extrapolation | Best Use Case                     |
  |----------------------|--------|-------------|---------------|-----------------------------------|
  | Learnable Embeddings | High   | Low         | Poor          | Fixed length sequences            |
  | Sinusoidal           | Low    | Low         | Good          | Variable length, interpretability |
  | AliBi                | None   | Medium      | Excellent     | Long sequences, efficiency        |
  | RoPE                 | None   | Medium      | Excellent     | Long sequences, rotation          |
  | Relative Position    | Medium | High        | Good          | Local patterns                    |


#### Detailed Analysis:
```
  1. Learnable Position Embeddings (Current Parts 1&2)

  self.pos_embedding = nn.Embedding(block_size, n_embed)
  Pros: Simple, can learn task-specific patterns
  Cons: Fixed max length, lots of parameters, poor extrapolation

  2. Sinusoidal Position Encoding (Original Transformer)

  def sinusoidal_encoding(seq_len, d_model):
      position = torch.arange(seq_len).unsqueeze(1)
      div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
      encoding = torch.zeros(seq_len, d_model)
      encoding[:, 0::2] = torch.sin(position * div_term)
      encoding[:, 1::2] = torch.cos(position * div_term)
      return encoding
  Pros: No parameters, good extrapolation, interpretable
  Cons: Fixed pattern, not learnable

  3. AliBi (Your Part 3 implementation)

  Position as attention bias
  alibi_bias = slopes * (i - j)  # Linear distance bias
  Pros: No position parameters, excellent extrapolation, efficient
  Cons: Linear assumption, limited expressiveness

  4. RoPE (Rotary Position Embedding) (Not implemented, but very popular)

  Rotates query and key vectors based on position.
  Pros: No parameters, excellent for long sequences, used in GPT-J, LLaMA
  Cons: More complex implementation

  How to Choose the Right One?

  For Your Assignment Context:

  1. Short sequences (≤32 tokens): Learnable embeddings work fine
  2. Need efficiency: AliBi (your Part 3) is best
  3. Variable length: Sinusoidal or AliBi
  4. Long sequences: AliBi or RoPE
```
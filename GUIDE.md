# Model type
 - Skip-gram Model: we predict the surrounding context words given the current center word. Used in Word2Vec, skip-gram models predict surrounding words given a target word. They are effective for capturing semantic relationships between words.
 - N-gram Model: Predicting the center word from context is called CBOW (Continuous Bag of Words).
 - Neural networks based language models: RNN / LSTM / GRUs / seq2seq
 - BERT: Encoder-only, uses bidirectional context, great for sentence classification. (with their ability to remember long-range dependencies, are well-suited for machine translation applications, where context and sequence memory are crucial.)
 - GPT: Decoder-only, uses causal (left-to-right) context, used for generation
 - seq2seq: Encoder-Decoder, neural network architecture designed to transform one sequence into another. It is widely used in tasks such as machine translation, text summarization, speech recognition, and image captioning.
 ### How Seq2Seq Works
 The process involves two phases:

 Encoding: The encoder processes the input sequence token by token, updating its internal state at each step. After processing the entire sequence, it outputs a context vector summarizing the input. (one vector only for the model)

 Decoding: The decoder uses the context vector to generate the output sequence token by token. During training, techniques like teacher forcing are used, where the actual target token is provided as input to the decoder instead of its previous prediction.
 ### RNN and LSTM
 Increasing the learning rate actually often leads to exploding gradients (where weights become huge and unstable), not fixing vanishing ones. The "Vanishing Gradient" problem means the gradient signal becomes virtually zero as it travels back through long sequences, so the model "forgets" early inputs.

 Why LSTMs? LSTMs were explicitly invented to solve this. They use gating mechanisms (forget, input, and output gates) that create a "gradient superhighway," allowing error signals to flow backward through time without vanishing.
 
 ### Self-attention
 The attention matrix calculates a score for every query against every key, resulting in an $N \times N$ matrix
 
 Impact on Long Docs in self-attention: While "time" is a factor, the bigger killer is Memory (RAM). Because the complexity is quadratic, doubling the sequence length quadruples the memory required. Processing a whole book (e.g., 50,000 tokens) would create an attention matrix with 2.5 billion entries, likely causing the GPU to run out of memory (OOM) immediately.


# Cards
| Key | Concept |
| --- | --- |
| Vanishing Gradient | Derivative of Sigmoid near 0 at extremes; weight updates stop. |
| $PP$ / Perplexity Formula | $e^{H(p,q)}$ (exponential of cross-entropy). If a model has a cross-entropy loss of $L$, the perplexity is defined as $e^L$ (or $2^L$ depending on the log base).|
| Causal Mask | Sets future token scores to $-\infty$ so $e^{-\infty} = 0$. |
| RoPE | Rotary position; rotates $Q$ and $K$ based on index; similarity decays with distance. |
| Gradient Checkpointing | Saves Memory, Costs Time (+33% compute) by re-calculating activations. |
| BPE Merging | Iteratively merges the most frequent adjacent pair of tokens. |
| Teacher Forcing | Feeding the ground-truth token as the next input during training. |
| Logit Temperature | $T < 1$ (Deterministic/Sharp), $T > 1$ (Diverse/Flat). |





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


# Sampling Method

#### Current Architecture:
``` 
Input → Transformer Layers → Language Model Head → Logits
                                                        ↓
                                            🎯 SAMPLING METHODS GO HERE
                                                        ↓
                                                Next Token

```

```python
# 1. Temperature Sampling

  def apply_temperature(logits, temperature=1.0):
      """Apply temperature scaling to logits"""
      if temperature == 0:
          return torch.argmax(logits, dim=-1)
      return logits / temperature

# 2. Top-k Sampling

  def top_k_sampling(logits, k=50, temperature=1.0):
      """Sample from top-k most likely tokens"""
      # Apply temperature
      logits = logits / temperature

      # Get top-k values and indices
      top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)

      # Create mask for top-k
      mask = torch.full_like(logits, float('-inf'))
      mask.scatter_(-1, top_k_indices, top_k_values)

      # Sample from top-k distribution
      probs = F.softmax(mask, dim=-1)
      return torch.multinomial(probs, 1)

# 3. Top-p (Nucleus) Sampling

  def top_p_sampling(logits, p=0.9, temperature=1.0):
      """Sample from tokens with cumulative probability <= p"""
      # Apply temperature
      logits = logits / temperature

      # Sort logits in descending order
      sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

      # Calculate cumulative probabilities
      cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

      # Create mask for tokens with cumsum > p
      sorted_indices_to_remove = cumulative_probs > p
      # Keep at least the first token
      sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
      sorted_indices_to_remove[..., 0] = 0

      # Scatter mask back to original order
      indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
      logits[indices_to_remove] = float('-inf')

      # Sample from filtered distribution
      probs = F.softmax(logits, dim=-1)
      return torch.multinomial(probs, 1)

# 4. Beam Search

  def beam_search(model, prompt_tokens, beam_size=5, max_length=50):
      """Beam search for sequence generation"""
      batch_size = prompt_tokens.size(0)
      seq_len = prompt_tokens.size(1)

      # Initialize beams
      beams = [(prompt_tokens, 0.0)]  # (sequence, score)

      for _ in range(max_length - seq_len):
          new_beams = []

          for seq, score in beams:
              # Get logits from model
              with torch.no_grad():
                  logits = model(seq)[:, -1, :]  # Last token logits

              # Get top beam_size candidates
              log_probs = F.log_softmax(logits, dim=-1)
              top_log_probs, top_indices = torch.topk(log_probs, beam_size)

              # Create new sequences
              for i in range(beam_size):
                  new_seq = torch.cat([seq, top_indices[:, i:i+1]], dim=1)
                  new_score = score + top_log_probs[0, i].item()
                  new_beams.append((new_seq, new_score))

          # Keep top beam_size beams
          new_beams.sort(key=lambda x: x[1], reverse=True)
          beams = new_beams[:beam_size]

      return beams[0][0]  # Return best sequence

  
  # Add to main.py PART3 section

  # After training your EnhancedLanguageModelingDecoder:
  print("\n--- Part 3.4: Text Generation with Different Sampling Methods ---")

  def generate_text(model, tokenizer, prompt, method='greedy', **kwargs):
      """Generate text using different sampling methods"""
      model.eval()
      prompt_tokens = torch.tensor([tokenizer.encode(prompt)]).to(device)

      with torch.no_grad():
          if method == 'greedy':
              return greedy_generate(model, prompt_tokens, **kwargs)
          elif method == 'top_k':
              return top_k_generate(model, prompt_tokens, **kwargs)
          elif method == 'top_p':
              return top_p_generate(model, prompt_tokens, **kwargs)
          # etc.

  # Test different sampling methods
  test_prompt = "The president"

  methods = [
      ('greedy', {}),
      ('top_k', {'k': 50, 'temperature': 0.8}),
      ('top_p', {'p': 0.9, 'temperature': 0.8}),
  ]

  for method, params in methods:
      generated = generate_text(enhanced_lm_model, tokenizer, test_prompt, method, **params)
      decoded = tokenizer.decode(generated[0].tolist())
      print(f"{method.upper()}: {decoded}")

```
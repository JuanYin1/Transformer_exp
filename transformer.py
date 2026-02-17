# # add all  your Encoder and Decoder code here


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

######################################################### Part 1: Encoder Implementation #########################################################
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, n_head, dropout=0.1):
        super().__init__()
        assert n_embed % n_head == 0

        self.n_head = n_head
        self.n_embed = n_embed
        self.head_dim = n_embed // n_head

        # Q, K, V projections
        self.q_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.k_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.v_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.out_proj = nn.Linear(n_embed, n_embed)

        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj_dropout(self.out_proj(out))

        return out, attn_weights

class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),  # Standard transformer FF ratio
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, n_embed, eps=1e-5):
        super().__init__()
        self.n_embed = n_embed
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(n_embed))
        self.beta = nn.Parameter(torch.zeros(n_embed))
    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mu) / (sigma + self.eps) + self.beta

# Block with residual connections and layer normalization
class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_head, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(n_embed, n_head, dropout)
        self.ffn = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN architecture with residual connections
        attn_out, attn_weights = self.attn(self.ln1(x), mask)
        x = x + self.dropout1(attn_out)

        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout2(ffn_out)

        return x, attn_weights

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, n_embed=64, n_layer=4, n_head=2, block_size=32):
        super().__init__()
        # Use hyperparameters passed as arguments
        self.n_embed = n_embed  # 64
        self.n_layer = n_layer  # 4
        self.n_head = n_head   # 2 (NOT 8!)
        self.block_size = block_size  # 32

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, self.n_embed)
        self.pos_embedding = nn.Embedding(self.block_size, self.n_embed) # learnable position embeddings
        self.embed_dropout = nn.Dropout(0.1)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(self.n_embed, self.n_head, dropout=0.1)
            for _ in range(self.n_layer)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(self.n_embed)

    def forward(self, input_ids):
        B, T = input_ids.size()
        assert T <= self.block_size

        # Position indices
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        # Embeddings
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(pos)
        x = self.embed_dropout(tok_emb + pos_emb)

        # Pass through transformer blocks
        attention_maps = []
        for block in self.blocks:
            x, attn_weights = block(x)
            attention_maps.append(attn_weights)

        # Final layer norm
        x = self.ln_f(x)

        return x, attention_maps

class SpeechClassifier(nn.Module):
    def __init__(self, vocab_size, n_embed=64, n_layer=4, n_head=2, block_size=32, n_hidden=100, n_output=3):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, n_embed, n_layer, n_head, block_size)

        # Use classifier dimensions passed as arguments
        self.classifier = nn.Sequential(
            nn.Linear(n_embed, n_hidden),  # 64 -> 100
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(n_hidden, n_output)  # 100 -> 3
        )

    def forward(self, input_ids):
        encoder_out, attention_maps = self.encoder(input_ids)

        # Mean pooling across sequence dimension (as required)
        pooled = encoder_out.mean(dim=1)

        # Classification
        logits = self.classifier(pooled)

        return logits, attention_maps


######################################################### Part 2: Decoder Implementation #########################################################
# Self-Attention → Cross-Attention → Feed-Forward with residual connections
# cross-attention is used to attend to the encoder's output for decoder
class CrossAttention(nn.Module):
    def __init__(self, n_embed, n_head, dropout=0.1):
        super().__init__()
        assert n_embed % n_head == 0
        
        self.n_embed = n_embed
        self.n_head = n_head
        self.head_dim = n_embed // n_head
        self.dropout = dropout
        
        self.q_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.k_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.v_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.out_proj = nn.Linear(n_embed, n_embed)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, x, encoder_out, src_mask=None, tgt_mask=None):
        B_tgt, T_tgt, C_tgt = x.size()
        B_src, T_src, C_src = encoder_out.size()
        
        q = self.q_proj(x).view(B_tgt, T_tgt, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(encoder_out).view(B_src, T_src, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(encoder_out).view(B_src, T_src, self.n_head, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if src_mask is not None:
            scores = scores.masked_fill(src_mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B_tgt, T_tgt, C_tgt)
        out = self.proj_dropout(self.out_proj(out))
        
        return out, attn_weights


# Decoder implementation 
class DecoderBlock(nn.Module):
    def __init__(self,n_embed, n_head, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_embed, n_head, dropout)
        self.cross_attn = CrossAttention(n_embed, n_head, dropout)
        self.ffn = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ln3 = nn.LayerNorm(n_embed)

    def forward(self, x, encoder_out, src_mask=None, tgt_mask=None):
        # Self-attention with residual connection and layer norm
        attn_out, _ = self.self_attn(self.ln1(x), tgt_mask)
        x = x + attn_out
        
        # Cross-attention with residual connection and layer norm  
        cross_attn_out, _ = self.cross_attn(self.ln2(x), encoder_out, src_mask, tgt_mask)
        x = x + cross_attn_out
        
        # Feed-forward with residual connection and layer norm
        ffn_out = self.ffn(self.ln3(x))
        x = x + ffn_out
        
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, n_embed=64, n_head=2, n_layer=4, dropout=0.1):
        super().__init__()
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(n_embed, n_head, dropout) for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(n_embed)

    def forward(self, x, encoder_out, src_mask=None, tgt_mask=None):
        for block in self.decoder_blocks:
            x = block(x, encoder_out, src_mask, tgt_mask)
        return self.ln(x)

class OutputHead(nn.Module):
    def __init__(self, n_embed=64, n_output=3):
        super().__init__()
        self.linear = nn.Linear(n_embed, n_output)
    def forward(self, x):
        return torch.log_softmax(self.linear(x), dim=-1)


class SpeechClassifierDecoder(nn.Module):
    def __init__(self, vocab_size, n_embed=64, n_head=2, n_layer=4, block_size=32, n_hidden=100, n_output=3, dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, n_embed, n_layer, n_head, block_size)
        self.decoder = TransformerDecoder(n_embed, n_head, n_layer, dropout)
        
        # Classification head with hidden layer
        self.classifier = nn.Sequential(
            nn.Linear(n_embed, n_hidden),  # 64 -> 100
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_output)  # 100 -> 3
        )
        
        # Decoder embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding = nn.Embedding(block_size, n_embed)
        self.embed_dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids):
        # Encode the input
        encoder_out, encoder_attention_maps = self.encoder(input_ids)
        
        # Use the input as "target" for the decoder (teacher forcing style)
        # In a real seq2seq task, this would be different
        B, T = input_ids.size()
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(pos)
        tgt_embedded = self.embed_dropout(tok_emb + pos_emb)
        
        # Decode
        decoder_out = self.decoder(tgt_embedded, encoder_out)
        
        # Mean pooling and classification
        pooled = decoder_out.mean(dim=1)
        logits = self.classifier(pooled)
        
        # Return format compatible with main.py: (logits, attention_maps)
        return logits, encoder_attention_maps


#Part 2.2: Language Modeling Pretraining - Pure Decoder-Only Architecture (GPT-like)
class DecoderOnlyBlock(nn.Module):
    """Pure decoder-only block with only causal self-attention (no cross-attention)"""
    def __init__(self, n_embed, n_head, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_embed, n_head, dropout)
        self.ffn = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None):
        # Only causal self-attention (no cross-attention like GPT)
        attn_out, attn_weights = self.self_attn(self.ln1(x), causal_mask)
        x = x + self.dropout1(attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout2(ffn_out)
        
        return x, attn_weights

class LanguageModelingDecoder(nn.Module):
    """GPT-like decoder-only language model for autoregressive generation"""
    def __init__(self, vocab_size, n_embed=64, n_head=2, n_layer=4, block_size=32, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        
        # Input embeddings for language modeling
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding = nn.Embedding(block_size, n_embed)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Stack of decoder-only blocks (no cross-attention)
        self.blocks = nn.ModuleList([
            DecoderOnlyBlock(n_embed, n_head, dropout) for _ in range(n_layer)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(n_embed)
        
        # Language modeling head
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
    def forward(self, input_ids):
        B, T = input_ids.size()
        assert T <= self.block_size
        
        # Input embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(pos)
        x = self.embed_dropout(tok_emb + pos_emb)
        
        # Create causal mask for self-attention (lower triangular)
        causal_mask = torch.tril(torch.ones(T, T, device=input_ids.device)).view(1, 1, T, T)
        
        # Pass through decoder-only blocks with causal self-attention
        attention_maps = []
        for block in self.blocks:
            x, attn_weights = block(x, causal_mask)
            attention_maps.append(attn_weights)
        
        # Final layer norm and language modeling head
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits


######################################################### Part 3: Architectural Exploration ################################################ 

# AliBi (Attention with Linear Biases) Implementation
class AliBiMultiHeadAttention(nn.Module):
    """Multi-head attention with AliBi (linear position biases)"""
    def __init__(self, n_embed, n_head, dropout=0.1, max_seq_len=512):
        super().__init__()
        assert n_embed % n_head == 0

        self.n_head = n_head
        self.n_embed = n_embed
        self.head_dim = n_embed // n_head
        self.max_seq_len = max_seq_len

        # Q, K, V projections
        self.q_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.k_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.v_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.out_proj = nn.Linear(n_embed, n_embed)

        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Create AliBi slopes for each head
        self.register_buffer('slopes', self._get_alibi_slopes())

    def _get_alibi_slopes(self):
        """Generate AliBi slopes for each attention head"""
        # Correct AliBi formula from original paper: geometric sequence
        def get_slopes_power_of_2(n):
            # Start with 1/(2^1), 1/(2^2), ..., 1/(2^n)
            return [1.0 / (2 ** (i + 1)) for i in range(n)]

        # For non-power of 2, use closest power of 2 and interpolate
        if self.n_head <= 1:
            return torch.tensor([0.5], dtype=torch.float32)
            
        if math.log2(self.n_head).is_integer():
            slopes = get_slopes_power_of_2(self.n_head)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(self.n_head))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            # Add extra slopes by interpolating
            extra_base = 2*closest_power_of_2
            extra_slopes = get_slopes_power_of_2(extra_base)[1::2]  # Every other slope
            slopes = slopes + extra_slopes[:self.n_head - closest_power_of_2]

        return torch.tensor(slopes, dtype=torch.float32)

    def _get_alibi_bias(self, seq_len, device):
        """Generate AliBi bias matrix"""
        # Create position matrix: i - j for each pair (i,j)  
        position_diffs = torch.arange(seq_len, device=device).unsqueeze(1) - torch.arange(seq_len, device=device).unsqueeze(0)
        
        # Apply slopes to get bias for each head
        alibi_bias = self.slopes.view(-1, 1, 1).to(device) * position_diffs.unsqueeze(0)
        
        # For causal language modeling, mask out future positions
        # AliBi handles causality through large negative biases for future positions
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        alibi_bias = alibi_bias.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        
        return alibi_bias

    def forward(self, x, mask=None):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add AliBi bias
        alibi_bias = self._get_alibi_bias(T, x.device)
        scores = scores + alibi_bias.unsqueeze(0)  # Add batch dimension

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj_dropout(self.out_proj(out))

        return out, attn_weights


# Local Window Attention (Sparse Attention)
class LocalWindowAttention(nn.Module):
    """Multi-head attention with local window (sparse attention)"""
    def __init__(self, n_embed, n_head, window_size=8, dropout=0.1):
        super().__init__()
        assert n_embed % n_head == 0

        self.n_head = n_head
        self.n_embed = n_embed
        self.head_dim = n_embed // n_head
        self.window_size = window_size

        # Q, K, V projections
        self.q_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.k_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.v_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.out_proj = nn.Linear(n_embed, n_embed)

        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _create_local_mask(self, seq_len, device):
        """Create local window mask"""
        mask = torch.zeros(seq_len, seq_len, device=device)
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 1
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions

    def forward(self, x, mask=None):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply local window mask
        local_mask = self._create_local_mask(T, x.device)
        scores = scores.masked_fill(local_mask == 0, float('-inf'))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj_dropout(self.out_proj(out))

        return out, attn_weights


# Enhanced Transformer Blocks for Part 3 exploration
class AliBiTransformerBlock(nn.Module):
    """Transformer block using AliBi attention"""
    def __init__(self, n_embed, n_head, dropout=0.1):
        super().__init__()
        self.attn = AliBiMultiHeadAttention(n_embed, n_head, dropout)
        self.ffn = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN architecture with residual connections
        attn_out, attn_weights = self.attn(self.ln1(x), mask)
        x = x + self.dropout1(attn_out)

        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout2(ffn_out)

        return x, attn_weights


class LocalWindowTransformerBlock(nn.Module):
    """Transformer block using local window attention"""
    def __init__(self, n_embed, n_head, window_size=8, dropout=0.1):
        super().__init__()
        self.attn = LocalWindowAttention(n_embed, n_head, window_size, dropout)
        self.ffn = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN architecture with residual connections
        attn_out, attn_weights = self.attn(self.ln1(x), mask)
        x = x + self.dropout1(attn_out)

        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout2(ffn_out)

        return x, attn_weights


# Enhanced Transformer Encoder for Part 3
class EnhancedTransformerEncoder(nn.Module):
    """Enhanced Transformer Encoder with AliBi and no positional embeddings"""
    def __init__(self, vocab_size, n_embed=64, n_layer=4, n_head=2, block_size=32):
        super().__init__()
        self.n_embed = n_embed
        self.n_layer = n_layer
        self.n_head = n_head
        self.block_size = block_size

        # Only token embeddings (no positional embeddings with AliBi)
        self.token_embedding = nn.Embedding(vocab_size, self.n_embed)
        self.embed_dropout = nn.Dropout(0.1)

        # Transformer blocks with AliBi
        self.blocks = nn.ModuleList([
            AliBiTransformerBlock(self.n_embed, self.n_head, dropout=0.1)
            for _ in range(self.n_layer)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(self.n_embed)

    def forward(self, input_ids):
        B, T = input_ids.size()
        assert T <= self.block_size

        # Only token embeddings (AliBi handles position)
        tok_emb = self.token_embedding(input_ids)
        x = self.embed_dropout(tok_emb)

        # Pass through transformer blocks
        attention_maps = []
        for block in self.blocks:
            x, attn_weights = block(x)
            attention_maps.append(attn_weights)

        # Final layer norm
        x = self.ln_f(x)

        return x, attention_maps


class LocalWindowTransformerEncoder(nn.Module):
    """Transformer Encoder with Local Window Attention"""
    def __init__(self, vocab_size, n_embed=64, n_layer=4, n_head=2, block_size=32, window_size=8):
        super().__init__()
        self.n_embed = n_embed
        self.n_layer = n_layer
        self.n_head = n_head
        self.block_size = block_size

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, self.n_embed)
        self.pos_embedding = nn.Embedding(self.block_size, self.n_embed)
        self.embed_dropout = nn.Dropout(0.1)

        # Transformer blocks with local window attention
        self.blocks = nn.ModuleList([
            LocalWindowTransformerBlock(self.n_embed, self.n_head, window_size, dropout=0.1)
            for _ in range(self.n_layer)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(self.n_embed)

    def forward(self, input_ids):
        B, T = input_ids.size()
        assert T <= self.block_size

        # Position indices
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        # Embeddings
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(pos)
        x = self.embed_dropout(tok_emb + pos_emb)

        # Pass through transformer blocks
        attention_maps = []
        for block in self.blocks:
            x, attn_weights = block(x)
            attention_maps.append(attn_weights)

        # Final layer norm
        x = self.ln_f(x)

        return x, attention_maps


# Enhanced Speech Classifiers for Part 3
class EnhancedSpeechClassifier(nn.Module):
    """Speech Classifier with AliBi positional encoding"""
    def __init__(self, vocab_size, n_embed=64, n_layer=4, n_head=2, block_size=32, n_hidden=100, n_output=3):
        super().__init__()
        self.encoder = EnhancedTransformerEncoder(vocab_size, n_embed, n_layer, n_head, block_size)

        # Enhanced classifier with deeper architecture
        self.classifier = nn.Sequential(
            nn.Linear(n_embed, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_hidden, n_hidden // 2),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(n_hidden // 2, n_output)
        )

    def forward(self, input_ids):
        encoder_out, attention_maps = self.encoder(input_ids)

        # Mean pooling across sequence dimension
        pooled = encoder_out.mean(dim=1)

        # Classification
        logits = self.classifier(pooled)

        return logits, attention_maps


class LocalWindowSpeechClassifier(nn.Module):
    """Speech Classifier with Local Window Attention"""
    def __init__(self, vocab_size, n_embed=64, n_layer=4, n_head=2, block_size=32, n_hidden=100, n_output=3, window_size=8):
        super().__init__()
        self.encoder = LocalWindowTransformerEncoder(vocab_size, n_embed, n_layer, n_head, block_size, window_size)

        # Use classifier dimensions passed as arguments
        self.classifier = nn.Sequential(
            nn.Linear(n_embed, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(n_hidden, n_output)
        )

    def forward(self, input_ids):
        encoder_out, attention_maps = self.encoder(input_ids)

        # Mean pooling across sequence dimension
        pooled = encoder_out.mean(dim=1)

        # Classification
        logits = self.classifier(pooled)

        return logits, attention_maps


# Enhanced Language Model for Part 3 - decoder 
class EnhancedLanguageModelingDecoder(nn.Module):
    """Enhanced Language modeling decoder with AliBi and improved architecture"""
    def __init__(self, vocab_size, n_embed=64, n_head=2, n_layer=4, block_size=32, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        
        # Input embeddings for language modeling (no positional embeddings with AliBi)
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Transformer blocks with AliBi
        self.blocks = nn.ModuleList([
            AliBiTransformerBlock(n_embed, n_head, dropout)
            for _ in range(n_layer)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(n_embed)
        
        # Language modeling head with weight sharing
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)
        # Tie weights with token embedding (but initialize separately first)
        with torch.no_grad():
            self.lm_head.weight.normal_(0.0, 0.02)  # Proper initialization
        # Note: Weight tying can cause instability, let's disable it temporarily
        # self.lm_head.weight = self.token_embedding.weight
        
    def forward(self, input_ids):
        B, T = input_ids.size()
        assert T <= self.block_size
        
        # Input embeddings (AliBi handles position)
        x = self.token_embedding(input_ids)
        x = self.embed_dropout(x)
        
        # AliBi handles causality internally - no need for external causal mask
        # Pass through transformer blocks  
        for block in self.blocks:
            x, _ = block(x, mask=None)  # AliBi attention handles causality
            
        # Final layer norm and language modeling head
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits



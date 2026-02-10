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
        self.pos_embedding = nn.Embedding(self.block_size, self.n_embed)
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


#Part 2.2: Language Modeling Pretraining - decoder only model
class DecoderOnlyLM(nn.Module):
    """Decoder-only language model for pretraining"""
    def __init__(self, vocab_size, n_embed=64, n_head=2, n_layer=4, block_size=32, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding = nn.Embedding(block_size, n_embed)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Decoder blocks (with causal self-attention)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(n_embed, n_head, dropout) for _ in range(n_layer)
        ])
        
        # Final layer norm and language modeling head
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
    def forward(self, input_ids, targets=None):
        B, T = input_ids.size()
        assert T <= self.block_size
        
        # Embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(pos)
        x = self.embed_dropout(tok_emb + pos_emb)
        
        # Create causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(T, T, device=input_ids.device)).view(1, 1, T, T)
        
        # Pass through decoder blocks
        for block in self.decoder_blocks:
            # For language modeling, we don't need encoder output
            # We'll modify DecoderBlock to handle this case
            x = self._decoder_block_lm(block, x, causal_mask)
        
        # Final layer norm and projection to vocabulary
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            # Shift targets: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_targets.view(-1))
            
        return logits, loss
    
    def _decoder_block_lm(self, block, x, mask):
        """Use only self-attention part of decoder block for language modeling"""
        # Self-attention with causal mask
        attn_out, _ = block.self_attn(block.ln1(x), mask)
        x = x + attn_out
        
        # Skip cross-attention (not needed for language modeling) since we are building the encoder-only model
        
        # Feed-forward
        ffn_out = block.ffn(block.ln3(x))
        x = x + ffn_out
        
        return x


######################################################### Part 3: Architectural Exploration ################################################ 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        x = self.dropout(x)
        return x + self.encoding[:, :x.size(1)]



    
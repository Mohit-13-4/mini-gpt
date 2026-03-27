"""
transformer.py - GPT Decoder-Only Transformer Architecture
Author: You
Date: 2026

This file contains the core GPT architecture:
1. Token + Position Embeddings
2. Multi-Head Causal Self-Attention with KV Cache
3. Feed-Forward Networks
4. Layer Normalization
5. Full Decoder Block
6. Complete GPT Model

Each component is explained in detail.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================================
# PART 1: CONFIGURATION (Hyperparameters)
# ============================================================================

class GPTConfig:
    """Configuration class for GPT hyperparameters."""
    
    def __init__(self):
        # Model architecture
        self.vocab_size = 1000     # Size of vocabulary (from tokenizer)
        self.block_size = 128       # Maximum context length (sequence length)
        self.n_layer = 6            # Number of transformer blocks (layers)
        self.n_head = 6             # Number of attention heads
        self.n_embd = 384           # Embedding dimension (model size)
        
        # Training
        self.dropout = 0.1          # Dropout rate (prevents overfitting)
        self.bias = False           # Use bias in Linear layers? (False = no bias)
        
        # Initialization
        self.init_std = 0.02         # Standard deviation for weight init
        
        print(f"🔧 GPT Configuration:")
        print(f"   Vocabulary size: {self.vocab_size}")
        print(f"   Context length: {self.block_size}")
        print(f"   Layers: {self.n_layer}, Heads: {self.n_head}")
        print(f"   Embedding dim: {self.n_embd}")


# ============================================================================
# PART 2: LAYER NORMALIZATION
# ============================================================================

class LayerNorm(nn.Module):
    """
    Layer Normalization module.
    
    Why layer norm instead of batch norm?
    - Works with variable sequence lengths
    - Independent across tokens (important for autoregressive models)
    - GPT uses LayerNorm before attention (pre-norm architecture)
    """
    
    def __init__(self, ndim: int, bias: bool = False):
        super().__init__()
        
        # gamma: learnable scale parameter (initialized to 1)
        self.weight = nn.Parameter(torch.ones(ndim))
        
        # beta: learnable shift parameter (initialized to 0)
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        
        # Epsilon for numerical stability (prevents division by zero)
        self.eps = 1e-5
    
    def forward(self, x):
        """
        Forward pass: Apply layer normalization.
        
        Args:
            x: Input tensor [batch_size, seq_len, ndim]
        
        Returns:
            Normalized tensor [batch_size, seq_len, ndim]
        """
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        if self.bias is not None:
            return self.weight * x_norm + self.bias
        else:
            return self.weight * x_norm


# ============================================================================
# PART 3: CAUSAL SELF-ATTENTION with KV Cache
# ============================================================================

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention mechanism with KV Cache support.
    
    KV Cache stores previous keys and values for faster inference.
    This is used in real LLMs like GPT, Llama, etc.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        # Validate that embedding dimension is divisible by number of heads
        assert config.n_embd % config.n_head == 0, \
            "Embedding dimension must be divisible by number of heads"
        
        # Store configuration
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # Key, Query, Value projections for all heads in one batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Dropout for regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask to ensure attention is only applied to the left
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
        
        # KV Cache for faster inference
        self.k_cache = None
        self.v_cache = None
        
        # Flash attention check
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("⚠️  Flash attention not available. Using manual implementation.")
    
    def forward(self, x, use_cache=False):
        """
        Forward pass of causal self-attention with KV cache support.
        
        Args:
            x: Input tensor [batch_size, seq_len, n_embd]
            use_cache: Whether to use KV cache for faster inference
        
        Returns:
            Attended tensor [batch_size, seq_len, n_embd]
        """
        batch_size, seq_len, n_embd = x.shape
        
        # Step 1: Project input to Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Step 2: Reshape for multi-head attention
        head_dim = n_embd // self.n_head
        q = q.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        
        # Step 3: Apply KV Cache
        if use_cache and self.k_cache is not None:
            k = torch.cat([self.k_cache, k], dim=2)
            v = torch.cat([self.v_cache, v], dim=2)
        
        # Update cache for next iteration
        if use_cache:
            self.k_cache = k
            self.v_cache = v
        else:
            # Reset cache when not using (e.g., during training)
            self.k_cache = None
            self.v_cache = None
        
        # Step 4: Compute attention scores
        if self.flash:
            # Flash attention (efficient, memory-optimized)
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0,
                is_causal=True
            )
        else:
            # Manual attention implementation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            
            # Apply causal mask
            causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
            att = att.masked_fill(causal_mask == 0, float('-inf'))
            
            # Softmax and dropout
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            
            # Apply attention to values
            y = att @ v
        
        # Step 5: Reassemble all heads
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
        
        # Step 6: Output projection
        y = self.resid_dropout(self.c_proj(y))
        
        return y
    
    def reset_cache(self):
        """Reset KV cache (useful for new generation sessions)."""
        self.k_cache = None
        self.v_cache = None


# ============================================================================
# PART 4: MLP (Multi-Layer Perceptron)
# ============================================================================

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network).
    
    In GPT: Linear → GELU → Linear → Dropout
    Dimension: n_embd → 4*n_embd → n_embd
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        # First linear layer: expand dimension by 4x
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        
        # GELU activation (non-linearity)
        self.gelu = nn.GELU()
        
        # Second linear layer: project back to original dimension
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# ============================================================================
# PART 5: TRANSFORMER BLOCK
# ============================================================================

class Block(nn.Module):
    """
    One transformer decoder block.
    
    Architecture:
    x -> LayerNorm -> Self-Attention -> + (residual) -> 
    LayerNorm -> MLP -> + (residual) -> output
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        # First layer norm (pre-attention)
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        
        # Causal self-attention
        self.attn = CausalSelfAttention(config)
        
        # Second layer norm (pre-MLP)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        
        # MLP
        self.mlp = MLP(config)
    
    def forward(self, x, use_cache=False):
        """
        Forward pass through transformer block with KV cache support.
        
        Args:
            x: Input [batch, seq, n_embd]
            use_cache: Whether to use KV cache
        
        Returns:
            Output [batch, seq, n_embd]
        """
        # Attention with residual connection
        x = x + self.attn(self.ln_1(x), use_cache=use_cache)
        
        # MLP with residual connection
        x = x + self.mlp(self.ln_2(x))
        
        return x


# ============================================================================
# PART 6: POSITIONAL EMBEDDINGS
# ============================================================================

class PositionalEmbedding(nn.Module):
    """Learnable positional embeddings."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pos_emb[:, :seq_len, :]


# ============================================================================
# PART 7: COMPLETE GPT MODEL
# ============================================================================

class GPT(nn.Module):
    """
    GPT Decoder-Only Transformer with KV Cache support.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Positional embeddings
        self.position_embedding = PositionalEmbedding(config)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        
        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights
        self.token_embedding.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Special initialization for certain layers
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=config.init_std / math.sqrt(2 * config.n_layer))
        
        print(f"\n✅ GPT Model Created:")
        print(f"   Total parameters: {self.get_num_params():,}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """Configure AdamW optimizer with weight decay."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        n_decay = sum(p.numel() for p in decay_params)
        n_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"   Optimizer: {n_decay:,} decay params, {n_nodecay:,} no-decay params")
        
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    
    def forward(self, idx, targets=None, use_cache=False):
        """
        Forward pass through entire GPT.
        
        Args:
            idx: Input token IDs [batch, seq_len]
            targets: Optional target token IDs for training [batch, seq_len]
            use_cache: Whether to use KV cache (for inference)
        
        Returns:
            logits: Predictions [batch, seq_len, vocab_size]
            loss: Cross-entropy loss (if targets provided)
        """
        batch_size, seq_len = idx.shape
        assert seq_len <= self.config.block_size, \
            f"Cannot forward sequence of length {seq_len}, block size is {self.config.block_size}"
        
        # Token and position embeddings
        token_emb = self.token_embedding(idx)
        x = self.position_embedding(token_emb)
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language model head
        logits = self.lm_head(x)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            logits_flat = logits.view(batch_size * seq_len, self.config.vocab_size)
            targets_flat = targets.view(batch_size * seq_len)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new text autoregressively with KV cache.
        
        Args:
            idx: Starting token IDs [batch, seq_len]
            max_new_tokens: How many tokens to generate
            temperature: Controls randomness
            top_k: If set, only sample from top k tokens
        
        Returns:
            Generated token IDs including input
        """
        # Reset cache before generation
        for block in self.blocks:
            block.attn.reset_cache()
        
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass with KV cache enabled
            logits, _ = self(idx_cond, use_cache=True)
            
            # Focus on last token
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Safety check - stop if we hit block size
            if idx.size(1) >= self.config.block_size:
                break
        
        return idx
    
    @torch.no_grad()
    def generate_advanced(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        Advanced generation with Top-K and Top-P (nucleus) sampling.
        
        Args:
            idx: Starting token IDs [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Controls randomness (1.0=normal, <1=more deterministic)
            top_k: Only sample from top k tokens (e.g., 50)
            top_p: Nucleus sampling - sample from top p probability mass (e.g., 0.9)
        """
        # Reset cache before generation
        for block in self.blocks:
            block.attn.reset_cache()
        
        for _ in range(max_new_tokens):
            # Crop context to block_size if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass with KV cache enabled
            logits, _ = self(idx_cond, use_cache=True)
            logits = logits[:, -1, :] / temperature
            
            # Top-K sampling
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Top-P (nucleus) sampling
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, -float('Inf'))
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Safety check - stop if we hit block size
            if idx.size(1) >= self.config.block_size:
                break
        
        return idx


# ============================================================================
# PART 8: TEST THE MODEL
# ============================================================================

def test_gpt():
    """Test the GPT model with dummy data."""
    print("\n" + "="*60)
    print("🧪 TESTING GPT MODEL")
    print("="*60)
    
    # Create config
    config = GPTConfig()
    config.vocab_size = 100
    config.block_size = 32
    config.n_layer = 2
    config.n_head = 2
    config.n_embd = 64
    
    # Create model
    model = GPT(config)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"\n📦 Model moved to: {device}")
    
    # Test forward pass
    print("\n📝 Testing forward pass...")
    batch_size = 4
    seq_len = 16
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    
    logits, loss = model(idx, targets)
    print(f"   Input shape: {idx.shape}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Loss: {loss.item():.4f}")
    
    # Test generation
    print("\n📝 Testing generation...")
    start_idx = torch.tensor([[1, 2, 3]]).to(device)
    generated = model.generate(start_idx, max_new_tokens=10, temperature=0.8, top_k=40)
    print(f"   Start shape: {start_idx.shape}")
    print(f"   Generated shape: {generated.shape}")
    print(f"   Generated tokens: {generated[0].tolist()}")
    
    # Test KV cache
    print("\n📝 Testing KV cache...")
    start_idx = torch.tensor([[1, 2, 3]]).to(device)
    import time
    start_time = time.time()
    _ = model.generate(start_idx, max_new_tokens=50, temperature=0.8)
    elapsed = time.time() - start_time
    print(f"   Generation with cache: {elapsed:.2f}s")
    
    # Count parameters
    n_params = model.get_num_params()
    print(f"\n📊 Total parameters: {n_params:,}")
    
    # Memory usage
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1e6
        print(f"   GPU memory: {mem_allocated:.1f} MB")
    
    print("\n" + "="*60)
    print("✅ GPT TEST COMPLETE!")
    print("="*60)
    
    return model


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    model = test_gpt()
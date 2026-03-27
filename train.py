"""
train.py - Training loop for GPT on Shakespeare
Author: You
Date: 2026

This file handles:
1. Loading and tokenizing Shakespeare text
2. Creating training batches
3. Training loop with optimization
4. Validation and checkpointing
5. Text generation during training

Each line is explained in detail.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import time
import math
import os
from datetime import datetime

# Import your modules
from tokenizer import BPETokenizer
from transformer import GPT, GPTConfig

# ============================================================================
# PART 1: CONFIGURATION
# ============================================================================

class TrainConfig:
    """Training configuration - optimized for RTX 3050 4GB."""
    
    def __init__(self):
        # Data
        self.data_file = 'data/tiny_shakespeare.txt'  # Shakespeare dataset
        self.vocab_size = 1000  # Match tokenizer from earlier
        
        # Model (using your tested config)
        self.block_size = 128    # Context length (from your test)
        self.n_layer = 6         # Number of transformer layers
        self.n_head = 6          # Number of attention heads
        self.n_embd = 384        # Embedding dimension
        
        # Training hyperparameters (optimized for your GPU!)
        self.batch_size = 64      # From your profiling - sweet spot!
        self.learning_rate = 3e-4  # Standard for Adam
        self.max_iters = 5000      # Total training iterations
        self.eval_interval = 250   # How often to evaluate
        self.eval_iters = 50       # How many batches for eval
        self.warmup_iters = 100    # Warmup steps for learning rate
        
        # Optimizer
        self.weight_decay = 1e-1   # AdamW weight decay
        self.beta1 = 0.9           # Adam beta1
        self.beta2 = 0.95          # Adam beta2
        self.grad_clip = 1.0        # Gradient clipping threshold
        
        # Checkpointing
        self.checkpoint_dir = 'checkpoints'
        self.log_interval = 10      # Print loss every N steps
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print("="*60)
        print("🎯 TRAINING CONFIGURATION")
        print("="*60)
        print(f"   Batch size: {self.batch_size} (your optimal!)")
        print(f"   Max iterations: {self.max_iters}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Model size: {self.n_layer} layers, {self.n_head} heads, {self.n_embd} dim")
        print(f"   Checkpoints: {self.checkpoint_dir}")
        print("="*60)


# ============================================================================
# PART 2: DATA LOADING
# ============================================================================

class ShakespeareData:
    """
    Handles loading and batching of Shakespeare text.
    
    This creates the training examples for GPT:
    - Takes the tokenized text
    - Creates chunks of block_size
    - Sets up inputs and targets for next token prediction
    """
    
    def __init__(self, config: TrainConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
        # Load and tokenize the entire text
        print("\n📖 Loading Shakespeare dataset...")
        with open(config.data_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"   Loaded {len(text):,} characters")
        
        # Train tokenizer on this text if not already trained
        if len(tokenizer.vocab) == 0:
            print("   Training tokenizer on Shakespeare...")
            tokenizer.train(text, verbose=False)
        
        # Tokenize entire text
        print("   Tokenizing text...")
        self.tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        
        print(f"   Created {len(self.tokens):,} tokens")
        print(f"   Vocabulary size: {len(tokenizer)}")
        
        # Calculate number of batches
        self.n_tokens = len(self.tokens)
        self.n_batches = self.n_tokens // (config.batch_size * config.block_size)
        print(f"   Total batches: {self.n_batches}")
    
    def get_batch(self, split: str = 'train'):
        """
        Get a random batch of data for training or evaluation.
        
        Args:
            split: 'train' or 'val' (90/10 split)
        
        Returns:
            x: Input tokens [batch_size, block_size]
            y: Target tokens [batch_size, block_size] (shifted by 1)
        """
        # Use 90% for training, 10% for validation
        data = self.tokens
        n_train = int(0.9 * len(data))
        
        if split == 'train':
            # Random starting indices in training portion
            start_indices = torch.randint(0, n_train - self.config.block_size, 
                                         (self.config.batch_size,))
        else:
            # Random starting indices in validation portion
            start_indices = torch.randint(n_train, len(data) - self.config.block_size,
                                         (self.config.batch_size,))
        
        # Prepare batches
        x = torch.stack([data[i:i+self.config.block_size] for i in start_indices])
        y = torch.stack([data[i+1:i+self.config.block_size+1] for i in start_indices])
        
        # Move to GPU if available
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        
        return x, y


# ============================================================================
# PART 3: LEARNING RATE SCHEDULER
# ============================================================================

def get_lr(iteration: int, config: TrainConfig):
    """
    Cosine learning rate schedule with warmup.
    
    Why this schedule?
    - Warmup: Prevents instability at start
    - Cosine decay: Gradually reduces learning rate
    - Helps model converge better
    
    Args:
        iteration: Current training iteration
        config: Training configuration
    
    Returns:
        Learning rate for this iteration
    """
    # 1. Linear warmup for warmup_iters steps
    if iteration < config.warmup_iters:
        return config.learning_rate * iteration / config.warmup_iters
    
    # 2. If past max_iters, use minimum learning rate
    if iteration > config.max_iters:
        return config.learning_rate * 0.1  # 10% of initial
    
    # 3. Cosine decay from warmup_iters to max_iters
    decay_ratio = (iteration - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Cosine: 1 → 0
    
    return config.learning_rate * coeff


# ============================================================================
# PART 4: MODEL INITIALIZATION
# ============================================================================

def init_model(config: TrainConfig):
    """
    Create and initialize the GPT model.
    
    Args:
        config: Training configuration
    
    Returns:
        model: GPT model
        optimizer: AdamW optimizer
    """
    print("\n🤖 Initializing GPT model...")
    
    # Create model config
    model_config = GPTConfig()
    model_config.vocab_size = config.vocab_size
    model_config.block_size = config.block_size
    model_config.n_layer = config.n_layer
    model_config.n_head = config.n_head
    model_config.n_embd = config.n_embd
    
    # Create model
    model = GPT(model_config)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model has {n_params:,} parameters")
    
    # Move to GPU
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"   Model moved to GPU: {torch.cuda.get_device_name(0)}")
    
    # Create optimizer (AdamW with weight decay)
    optimizer = model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2)
    )
    
    return model, optimizer


# ============================================================================
# PART 5: EVALUATION
# ============================================================================

@torch.no_grad()
def estimate_loss(model, data, config: TrainConfig):
    """
    Estimate loss on train and validation sets.
    
    Args:
        model: GPT model
        data: ShakespeareData object
        config: Training configuration
    
    Returns:
        Dictionary with 'train' and 'val' losses
    """
    out = {}
    model.eval()  # Set to evaluation mode
    
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        
        for k in range(config.eval_iters):
            # Get batch
            x, y = data.get_batch(split)
            
            # Forward pass
            logits, loss = model(x, y)
            losses[k] = loss.item()
        
        out[split] = losses.mean()
    
    model.train()  # Back to training mode
    return out


# ============================================================================
# PART 6: TEXT GENERATION
# ============================================================================

@torch.no_grad()
def generate_sample(model, tokenizer, config: TrainConfig, prompt: str = "ROMEO:"):
    """
    Generate text from the model to see progress.
    
    Args:
        model: GPT model
        tokenizer: BPE tokenizer
        config: Training configuration
        prompt: Starting text
    
    Returns:
        Generated text
    """
    model.eval()
    
    # Encode prompt
    context = tokenizer.encode(prompt)
    context = torch.tensor(context, dtype=torch.long).unsqueeze(0)
    
    if torch.cuda.is_available():
        context = context.cuda()
    
    # Generate
    generated = model.generate(
        context,
        max_new_tokens=100,
        temperature=0.8,
        top_k=40
    )
    
    # Decode
    generated_text = tokenizer.decode(generated[0].tolist())
    
    model.train()
    return generated_text


# ============================================================================
# PART 7: CHECKPOINTING
# ============================================================================

def save_checkpoint(model, optimizer, iteration, loss, config: TrainConfig):
    """
    Save model checkpoint.
    
    Args:
        model: GPT model
        optimizer: Optimizer
        iteration: Current iteration
        loss: Current loss
        config: Training configuration
    """
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,
    }
    
    filename = f"{config.checkpoint_dir}/gpt_iter_{iteration:06d}.pt"
    torch.save(checkpoint, filename)
    print(f"💾 Checkpoint saved: {filename}")


def load_checkpoint(filename, model, optimizer):
    """
    Load model checkpoint.
    
    Args:
        filename: Checkpoint file
        model: GPT model
        optimizer: Optimizer
    
    Returns:
        iteration: Loaded iteration
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']


# ============================================================================
# PART 8: MAIN TRAINING LOOP
# ============================================================================

def train():
    """Main training function."""
    
    # ===== SETUP =====
    print("\n" + "="*60)
    print("🚀 STARTING GPT TRAINING ON SHAKESPEARE")
    print("="*60)
    
    # Configuration
    config = TrainConfig()
    
    # Initialize tokenizer
    print("\n🔤 Initializing tokenizer...")
    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    
    # Load data
    data = ShakespeareData(config, tokenizer)
    
    # Initialize model
    model, optimizer = init_model(config)
    
    # ===== TRAINING LOOP =====
    print("\n" + "="*60)
    print("🎮 TRAINING BEGINS")
    print("="*60)
    
    # Tracking variables
    best_val_loss = float('inf')
    iter_num = 0
    train_losses = []
    val_losses = []
    start_time = time.time()
    
    # Main loop
    for iter_num in range(config.max_iters):
        
        # ===== LEARNING RATE SCHEDULING =====
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # ===== FORWARD BACKWARD =====
        # Get batch
        x, y = data.get_batch('train')
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        # Optimizer step
        optimizer.step()
        
        # ===== LOGGING =====
        if iter_num % config.log_interval == 0:
            elapsed = time.time() - start_time
            print(f"\nIteration {iter_num}/{config.max_iters} | "
                  f"Loss: {loss.item():.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Time: {elapsed:.1f}s")
        
        # ===== EVALUATION =====
        if iter_num % config.eval_interval == 0 and iter_num > 0:
            # Estimate losses
            losses = estimate_loss(model, data, config)
            train_loss = losses['train']
            val_loss = losses['val']
            
            train_losses.append((iter_num, train_loss))
            val_losses.append((iter_num, val_loss))
            
            print(f"\n📊 EVALUATION at iter {iter_num}:")
            print(f"   Train loss: {train_loss:.4f}")
            print(f"   Val loss: {val_loss:.4f}")
            
            # Generate sample text
            if iter_num % (config.eval_interval * 2) == 0:
                sample = generate_sample(model, tokenizer, config, "ROMEO:")
                print(f"\n📝 GENERATED SAMPLE:\n{sample}\n")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, iter_num, val_loss, config)
                print(f"🏆 New best model! Val loss: {val_loss:.4f}")
    
    # ===== TRAINING COMPLETE =====
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Final train loss: {loss.item():.4f}")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Checkpoints saved in: {config.checkpoint_dir}")
    
    # Generate final sample
    print("\n📝 FINAL GENERATED SAMPLE:")
    final_sample = generate_sample(model, tokenizer, config, "ROMEO:")
    print(final_sample)
    
    # Save final model
    save_checkpoint(model, optimizer, config.max_iters, loss.item(), config)
    
    return model, tokenizer, train_losses, val_losses


# ============================================================================
# PART 9: ADD OPTIMIZER CONFIGURATION TO GPT CLASS
# ============================================================================
# 
# We need to add this method to the GPT class in transformer.py
# Add this to your GPT class in transformer.py
# ============================================================================

"""
Add this method to your GPT class in transformer.py:

def configure_optimizers(self, weight_decay, learning_rate, betas):
    # Get all parameters
    param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
    
    # Parameters with dimension >= 2 get weight decay (like weights)
    # Parameters with dimension < 2 don't (like biases and layer norms)
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    print(f"   Decay params: {len(decay_params)} | No decay params: {len(nodecay_params)}")
    
    # Create AdamW optimizer
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer
"""


# ============================================================================
# PART 10: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run training
    model, tokenizer, train_losses, val_losses = train()
    
    print("\n✨ Training script completed!")
    print("   You can now use inference.py to generate more text!")
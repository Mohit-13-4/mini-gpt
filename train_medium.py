"""
train_medium.py - Train a larger GPT model (25-30M parameters) on WikiText-103
Optimized for RTX 3050 4GB
"""

import torch
import torch.nn as nn
import time
import math
import os
from datetime import datetime
from tokenizer import BPETokenizer
from transformer import GPT, GPTConfig
from data_loader import DataLoader

# ============================================================================
# CONFIGURATION
# ============================================================================

class MediumModelConfig:
    """Configuration for 25-30M parameter model optimized for RTX 3050 4GB"""
    
    def __init__(self):
        # Model architecture (25-30M parameters)
        self.vocab_size = 1500        # WikiText vocabulary
        self.block_size = 128          # Keep same for memory
        self.n_layer = 10              # 6 → 10 layers
        self.n_head = 8                # Changed from 10 to 8 (512/8=64, divisible!)
        self.n_embd = 512              # 384 → 512
        
        # Training hyperparameters
        self.batch_size = 32           # Reduced from 64 for larger model
        self.learning_rate = 3e-4
        self.max_iters = 10000         # Train longer for better results
        self.eval_interval = 500
        self.eval_iters = 50
        self.warmup_iters = 200
        
        # Advanced features
        self.grad_clip = 1.0
        self.use_mixed_precision = True
        self.weight_decay = 0.1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.dropout = 0.1
        
        # Checkpointing
        self.checkpoint_dir = 'checkpoints_medium'
        self.log_dir = f'runs/medium_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Calculate approximate parameter count
        # Rough estimate: n_layer * n_embd * n_embd * 12 / 1e6
        param_estimate = (self.n_layer * self.n_embd * self.n_embd * 12) / 1e6
        print("="*60)
        print("🎯 MEDIUM GPT MODEL CONFIGURATION")
        print("="*60)
        print(f"   Model: {self.n_layer} layers, {self.n_head} heads, {self.n_embd} dim")
        print(f"   Head dimension: {self.n_embd // self.n_head} (must divide evenly)")
        print(f"   Estimated parameters: ~{param_estimate:.0f}M")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Max iterations: {self.max_iters}")
        print(f"   Mixed precision: {self.use_mixed_precision}")
        print(f"   Checkpoint dir: {self.checkpoint_dir}")
        print("="*60)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_tokenizer(config):
    """Load or create tokenizer for WikiText-103."""
    print("\n🔤 Loading tokenizer...")
    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    
    tokenizer_path = f'{config.checkpoint_dir}/tokenizer.json'
    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
        print(f"   Loaded existing tokenizer: {len(tokenizer)} tokens")
        return tokenizer
    
    print("   Training tokenizer on WikiText-103 (this may take a moment)...")
    from data_loader import DataLoader
    wiki_data = DataLoader('wikitext', None, None)
    text = wiki_data._load_wikitext()
    tokenizer.train(text, verbose=False)
    tokenizer.save(tokenizer_path)
    print(f"   Trained tokenizer: {len(tokenizer)} tokens")
    
    return tokenizer

def create_data_loader(config, tokenizer):
    """Create data loader for WikiText-103."""
    print("\n📖 Loading WikiText-103 dataset...")
    from data_loader import DataLoader
    data = DataLoader('wikitext', config, tokenizer)
    print(f"   Total tokens: {data.n_tokens:,}")
    return data

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

def create_model(config):
    """Create the medium-sized GPT model."""
    print("\n🤖 Creating medium GPT model...")
    
    # Create model config
    model_config = GPTConfig()
    model_config.vocab_size = config.vocab_size
    model_config.block_size = config.block_size
    model_config.n_layer = config.n_layer
    model_config.n_head = config.n_head
    model_config.n_embd = config.n_embd
    model_config.dropout = config.dropout
    model_config.bias = False
    
    # Create model
    model = GPT(model_config)
    
    # Move to GPU
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"   Model moved to GPU: {torch.cuda.get_device_name(0)}")
    
    # Count actual parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Actual parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    
    return model

def create_optimizer(model, config):
    """Create AdamW optimizer with weight decay."""
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    n_decay = sum(p.numel() for p in decay_params)
    n_nodecay = sum(p.numel() for p in nodecay_params)
    print(f"   Optimizer: {n_decay:,} decay params, {n_nodecay:,} no-decay params")
    
    optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=(config.beta1, config.beta2))
    return optimizer

# ============================================================================
# LEARNING RATE SCHEDULER
# ============================================================================

def get_lr(iteration, config):
    """Cosine learning rate schedule with warmup."""
    if iteration < config.warmup_iters:
        return config.learning_rate * iteration / config.warmup_iters
    if iteration > config.max_iters:
        return config.learning_rate * 0.1
    decay_ratio = (iteration - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.learning_rate * coeff

# ============================================================================
# EVALUATION
# ============================================================================

@torch.no_grad()
def estimate_loss(model, data, config):
    """Estimate loss on train and validation sets."""
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            x, y = data.get_batch(split)
            with torch.cuda.amp.autocast(enabled=config.use_mixed_precision):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train():
    """Main training function."""
    
    # Load configuration
    config = MediumModelConfig()
    
    # Load tokenizer and data
    tokenizer = load_tokenizer(config)
    data = create_data_loader(config, tokenizer)
    
    # Create model and optimizer
    model = create_model(config)
    optimizer = create_optimizer(model, config)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_mixed_precision)
    
    # Training loop
    print("\n" + "="*60)
    print("🎮 TRAINING BEGINS")
    print("="*60)
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    for iter_num in range(config.max_iters):
        # Update learning rate
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Get batch
        x, y = data.get_batch('train')
        
        # Forward/backward with mixed precision
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=config.use_mixed_precision):
            _, loss = model(x, y)
        
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Logging
        if iter_num % 50 == 0:
            elapsed = time.time() - start_time
            mem_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"\nIter {iter_num}/{config.max_iters} | Loss: {loss.item():.4f} | "
                  f"LR: {lr:.2e} | Time: {elapsed:.1f}s | GPU: {mem_used:.2f}GB")
        
        # Evaluation
        if iter_num % config.eval_interval == 0 and iter_num > 0:
            losses = estimate_loss(model, data, config)
            train_loss = losses['train']
            val_loss = losses['val']
            train_ppl = math.exp(train_loss)
            val_ppl = math.exp(val_loss)
            
            print(f"\n📊 EVALUATION at iter {iter_num}:")
            print(f"   Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
            print(f"   Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'iteration': iter_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'config': config
                }
                torch.save(checkpoint, f'{config.checkpoint_dir}/best_model.pt')
                print(f"🏆 New best model! Val loss: {val_loss:.4f}")
            
            # Save periodic checkpoint
            if iter_num % 1000 == 0:
                torch.save(checkpoint, f'{config.checkpoint_dir}/iter_{iter_num:06d}.pt')
                print(f"💾 Checkpoint saved: iter_{iter_num:06d}.pt")
    
    # Training complete
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Checkpoints: {config.checkpoint_dir}")
    
    return model, tokenizer

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 TRAINING MEDIUM GPT MODEL (25-30M PARAMETERS)")
    print("="*60)
    print("   This model will take 30-45 minutes to train")
    print("   Make sure you have ~3.5GB free GPU memory")
    print("="*60)
    
    model, tokenizer = train()
    
    print("\n✨ Training script completed!")
    print("   You can now use app_medium.py to generate text!")
"""
train_fixed.py - Train a balanced 15-20M model on FULL WikiText-103 dataset
Optimized for RTX 3050 4GB with proper regularization
"""

import torch
import torch.nn as nn
import time
import math
import os
from datetime import datetime
from tokenizer import BPETokenizer
from transformer import GPT, GPTConfig

# ============================================================================
# CONFIGURATION
# ============================================================================

class FixedModelConfig:
    """Balanced 15-20M model with strong regularization for FULL dataset"""
    
    def __init__(self):
        # Model architecture (balanced for 15-20M parameters)
        self.vocab_size = 5000
        self.block_size = 128
        self.n_layer = 8               # 8 layers
        self.n_head = 8                # 8 heads (512/8=64)
        self.n_embd = 512              # 512 embedding dim
        
        # Strong regularization to prevent overfitting
        self.dropout = 0.2             # Higher dropout
        self.weight_decay = 0.2        # Higher weight decay
        self.bias = False
        
        # Training hyperparameters
        self.batch_size = 24           # Slightly smaller for full dataset
        self.gradient_accumulation = 2  # Effective batch size = 48
        self.learning_rate = 2e-4      # Lower learning rate
        self.max_iters = 15000          # Train longer
        self.eval_interval = 500
        self.eval_iters = 50
        self.warmup_iters = 500
        
        # Advanced features
        self.grad_clip = 0.5            # Tighter gradient clipping
        self.use_mixed_precision = True
        self.beta1 = 0.9
        self.beta2 = 0.95
        
        # Use FULL WikiText-103 dataset
        self.use_full_dataset = True
        self.dataset = 'wikitext_full'
        
        # Checkpointing
        self.checkpoint_dir = 'checkpoints_fixed'
        self.log_dir = f'runs/fixed_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Calculate parameter estimate
        param_estimate = (self.n_layer * self.n_embd * self.n_embd * 12) / 1e6
        effective_batch = self.batch_size * self.gradient_accumulation
        
        print("="*60)
        print("🎯 FIXED MODEL CONFIGURATION (15-20M)")
        print("="*60)
        print(f"   Model: {self.n_layer} layers, {self.n_head} heads, {self.n_embd} dim")
        print(f"   Head dimension: {self.n_embd // self.n_head}")
        print(f"   Estimated parameters: ~{param_estimate:.0f}M")
        print(f"   Dropout: {self.dropout} | Weight decay: {self.weight_decay}")
        print(f"   Batch size: {self.batch_size} x {self.gradient_accumulation} = {effective_batch}")
        print(f"   Max iterations: {self.max_iters}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Dataset: FULL WikiText-103")
        print(f"   Checkpoint dir: {self.checkpoint_dir}")
        print("="*60)

# ============================================================================
# DATA LOADING - FULL WIKITEXT
# ============================================================================

def load_full_wikitext_text():
    """Load FULL WikiText-103 dataset (not just subset)."""
    print("\n📖 Loading FULL WikiText-103 dataset...")
    
    try:
        from datasets import load_dataset
        print("   Using HuggingFace datasets...")
        
        # Load the FULL training set
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
        
        # Get all text (filter empty lines)
        texts = [text for text in dataset['text'] if text.strip()]
        text = '\n'.join(texts)
        
        print(f"   ✅ Loaded {len(text):,} characters from {len(texts):,} examples")
        print(f"   (Using FULL dataset - ~180K articles!)")
        
        return text
        
    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
        print("   Falling back to subset...")
        return load_wikitext_subset()

def load_wikitext_subset():
    """Fallback to subset if full dataset fails."""
    from data_loader import DataLoader
    wiki_data = DataLoader('wikitext', None, None)
    return wiki_data._load_wikitext()

def load_tokenizer(config):
    """Load or create tokenizer for FULL WikiText."""
    print("\n🔤 Loading tokenizer...")
    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    
    tokenizer_path = f'{config.checkpoint_dir}/tokenizer.json'
    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
        print(f"   Loaded existing tokenizer: {len(tokenizer)} tokens")
        return tokenizer
    
    print("   Training tokenizer on FULL WikiText-103 (this may take a minute)...")
    text = load_full_wikitext_text()
    tokenizer.train(text, verbose=False)
    tokenizer.save(tokenizer_path)
    print(f"   Trained tokenizer: {len(tokenizer)} tokens")
    
    return tokenizer

def create_data_loader(config, tokenizer):
    """Create data loader for FULL WikiText."""
    print("\n📖 Creating data loader...")
    
    # Load full text
    text = load_full_wikitext_text()
    
    # Tokenize
    print("   Tokenizing text...")
    tokens = tokenizer.encode(text)
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    print(f"   Created {len(tokens_tensor):,} tokens")
    
    # Create simple data class
    class WikiTextData:
        def __init__(self, tokens, config):
            self.tokens = tokens
            self.config = config
            self.n_tokens = len(tokens)
            print(f"   Total tokens: {self.n_tokens:,}")
        
        def get_batch(self, split='train'):
            n_train = int(0.9 * self.n_tokens)
            data = self.tokens
            
            if split == 'train':
                start = torch.randint(0, n_train - self.config.block_size, 
                                     (self.config.batch_size,))
            else:
                start = torch.randint(n_train, self.n_tokens - self.config.block_size,
                                     (self.config.batch_size,))
            
            x = torch.stack([data[i:i+self.config.block_size] for i in start])
            y = torch.stack([data[i+1:i+self.config.block_size+1] for i in start])
            
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            
            return x, y
    
    return WikiTextData(tokens_tensor, config)

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

def create_model(config):
    """Create the balanced GPT model."""
    print("\n🤖 Creating balanced GPT model...")
    
    model_config = GPTConfig()
    model_config.vocab_size = config.vocab_size
    model_config.block_size = config.block_size
    model_config.n_layer = config.n_layer
    model_config.n_head = config.n_head
    model_config.n_embd = config.n_embd
    model_config.dropout = config.dropout
    model_config.bias = config.bias
    
    model = GPT(model_config)
    
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"   Model moved to GPU: {torch.cuda.get_device_name(0)}")
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Actual parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    
    return model

def create_optimizer(model, config):
    """Create AdamW optimizer with strong weight decay."""
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
    
    return torch.optim.AdamW(optim_groups, lr=config.learning_rate, 
                             betas=(config.beta1, config.beta2))

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
# TRAINING LOOP WITH GRADIENT ACCUMULATION
# ============================================================================

def train():
    """Main training function with gradient accumulation."""
    
    config = FixedModelConfig()
    
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
    print(f"   Gradient accumulation: {config.gradient_accumulation} steps")
    print(f"   Effective batch size: {config.batch_size * config.gradient_accumulation}")
    print("="*60)
    
    best_val_loss = float('inf')
    start_time = time.time()
    accum_step = 0
    
    for iter_num in range(config.max_iters):
        # Update learning rate
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Get batch
        x, y = data.get_batch('train')
        
        # Forward/backward with accumulation
        with torch.cuda.amp.autocast(enabled=config.use_mixed_precision):
            _, loss = model(x, y)
            loss = loss / config.gradient_accumulation
        
        scaler.scale(loss).backward()
        accum_step += 1
        
        # Update weights after accumulation steps
        if accum_step % config.gradient_accumulation == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accum_step = 0
        
        # Logging
        if iter_num % 50 == 0:
            elapsed = time.time() - start_time
            mem_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"\nIter {iter_num}/{config.max_iters} | Loss: {loss.item() * config.gradient_accumulation:.4f} | "
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
            
            # Periodic checkpoint
            if iter_num % 2000 == 0:
                torch.save(checkpoint, f'{config.checkpoint_dir}/iter_{iter_num:06d}.pt')
                print(f"💾 Checkpoint saved: iter_{iter_num:06d}.pt")
    
    # Training complete
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Best perplexity: {math.exp(best_val_loss):.2f}")
    print(f"   Checkpoints: {config.checkpoint_dir}")
    
    return model, tokenizer

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 TRAINING FIXED MODEL (15-20M) ON FULL WIKITEXT")
    print("="*60)
    print("   This will take 1-2 hours to train")
    print("   The model will learn proper English!")
    print("="*60)
    
    model, tokenizer = train()
    
    print("\n✨ Training complete! Run app_fixed.py to generate text!")
"""
train_wikitext_fixed.py - Train a working WikiText model from scratch
Uses WikiText-2 (smaller vocabulary) and proper BPE
"""

import torch
import torch.nn as nn
import time
import math
import os
import glob
from datetime import datetime
from tokenizer import BPETokenizer
from transformer import GPT, GPTConfig

# ============================================================================
# CONFIGURATION
# ============================================================================

class WikiTextConfig:
    def __init__(self):
        # Model - Balanced for RTX 3050 4GB
        self.vocab_size = 5000
        self.block_size = 128
        self.n_layer = 6
        self.n_head = 6
        self.n_embd = 384
        self.dropout = 0.2
        self.bias = False
        
        # Training - Optimized for longer run
        self.batch_size = 32
        self.gradient_accumulation = 2
        self.learning_rate = 2e-4  # Lower LR for longer training (was 3e-4)
        self.max_iters = 20000      # Extended to 20k iterations
        self.eval_interval = 500
        self.eval_iters = 50
        self.warmup_iters = 500      # Extended warmup
        
        # Advanced
        self.grad_clip = 1.0
        self.use_mixed_precision = True
        self.weight_decay = 0.1
        
        # Dataset
        self.dataset = 'wikitext2'  # Use WikiText-2!
        
        # Checkpointing
        self.checkpoint_dir = 'checkpoints_wikitext_fixed'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        param_est = self.n_layer * self.n_embd * self.n_embd * 12 / 1e6
        print("="*60)
        print("🎯 WIKITEXT FIXED CONFIGURATION (20K ITERATIONS)")
        print("="*60)
        print(f"   Model: {self.n_layer} layers, {self.n_head} heads, {self.n_embd} dim")
        print(f"   Parameters: ~{param_est:.0f}M")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Max iterations: {self.max_iters}")
        print(f"   Dataset: WikiText-2 (small vocabulary!)")
        print(f"   Batch size: {self.batch_size} x {self.gradient_accumulation} = {self.batch_size * self.gradient_accumulation}")
        print("="*60)

# ============================================================================
# DATA LOADING - WikiText-2
# ============================================================================

def load_wikitext2():
    """Load WikiText-2 dataset (much smaller vocabulary)."""
    print("\n📖 Loading WikiText-2 dataset...")
    
    try:
        from datasets import load_dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        texts = [text for text in dataset['text'] if text.strip()]
        text = '\n'.join(texts)  # Use ALL examples for better training
        print(f"   Loaded {len(text):,} characters from {len(texts):,} examples")
        print(f"   Unique characters: {len(set(text))}")
        return text
    except Exception as e:
        print(f"   Error: {e}")
        return None

def create_data_loader(config, tokenizer):
    """Create data loader for WikiText-2."""
    print("\n📖 Creating data loader...")
    
    text = load_wikitext2()
    if text is None:
        return None
    
    print("   Tokenizing text (this may take a moment)...")
    tokens = tokenizer.encode(text)
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    print(f"   Created {len(tokens_tensor):,} tokens")
    
    class WikiData:
        def __init__(self, tokens, config):
            self.tokens = tokens
            self.config = config
            self.n_tokens = len(tokens)
        
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
    
    return WikiData(tokens_tensor, config)

# ============================================================================
# TOKENIZER
# ============================================================================

def load_tokenizer(config):
    """Load or create tokenizer with proper BPE."""
    print("\n🔤 Loading tokenizer...")
    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    
    tokenizer_path = f'{config.checkpoint_dir}/tokenizer.json'
    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
        print(f"   Loaded tokenizer: {len(tokenizer)} tokens, {len(tokenizer.merges)} merges")
        return tokenizer
    
    print("   Training tokenizer on WikiText-2...")
    text = load_wikitext2()
    if text is None:
        return None
    
    tokenizer.train(text, verbose=True)
    tokenizer.save(tokenizer_path)
    print(f"   Trained tokenizer: {len(tokenizer)} tokens, {len(tokenizer.merges)} merges")
    
    return tokenizer

# ============================================================================
# MODEL
# ============================================================================

def create_model(config):
    """Create model with correct configuration."""
    print("\n🤖 Creating model...")
    
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
        print(f"   Model on GPU: {torch.cuda.get_device_name(0)}")
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    
    return model

def create_optimizer(model, config):
    """Create AdamW optimizer."""
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    return torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=(0.9, 0.95))

# ============================================================================
# TRAINING
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

@torch.no_grad()
def estimate_loss(model, data, config):
    model.eval()
    out = {}
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

def resume_from_checkpoint(model, optimizer, config):
    """Resume training from latest checkpoint if exists."""
    checkpoint_files = sorted(glob.glob(f'{config.checkpoint_dir}/iter_*.pt'))
    if checkpoint_files:
        latest = checkpoint_files[-1]
        print(f"\n📂 Found checkpoint: {os.path.basename(latest)}")
        checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_iter = checkpoint.get('iteration', 0)
        best_val_loss = checkpoint.get('loss', float('inf'))
        
        print(f"   Resuming from iteration {start_iter}")
        print(f"   Best validation loss so far: {best_val_loss:.4f}")
        
        return start_iter, best_val_loss
    
    return 0, float('inf')

def train():
    """Main training function."""
    
    config = WikiTextConfig()
    
    # Load tokenizer and data
    tokenizer = load_tokenizer(config)
    if tokenizer is None:
        print("Failed to load tokenizer")
        return None, None
    
    data = create_data_loader(config, tokenizer)
    if data is None:
        print("Failed to load data")
        return None, None
    
    # Create model and optimizer
    model = create_model(config)
    optimizer = create_optimizer(model, config)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_mixed_precision)
    
    # Resume from checkpoint if exists
    start_iter, best_val_loss = resume_from_checkpoint(model, optimizer, config)
    
    print("\n" + "="*60)
    print("🎮 TRAINING BEGINS")
    print("="*60)
    print(f"   Starting from iteration: {start_iter}")
    print(f"   Target iterations: {config.max_iters}")
    print("="*60)
    
    start_time = time.time()
    accum_step = 0
    
    for iter_num in range(start_iter, config.max_iters):
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        x, y = data.get_batch('train')
        
        with torch.cuda.amp.autocast(enabled=config.use_mixed_precision):
            _, loss = model(x, y)
            loss = loss / config.gradient_accumulation
        
        scaler.scale(loss).backward()
        accum_step += 1
        
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
            mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            current_loss = loss.item() * config.gradient_accumulation
            print(f"\nIter {iter_num}/{config.max_iters} | Loss: {current_loss:.4f} | LR: {lr:.2e} | Time: {elapsed:.1f}s | GPU: {mem:.2f}GB")
        
        # Evaluation
        if iter_num % config.eval_interval == 0 and iter_num > 0:
            losses = estimate_loss(model, data, config)
            train_ppl = math.exp(losses['train'])
            val_ppl = math.exp(losses['val'])
            
            print(f"\n📊 EVALUATION at iter {iter_num}:")
            print(f"   Train Loss: {losses['train']:.4f} | Train PPL: {train_ppl:.2f}")
            print(f"   Val Loss: {losses['val']:.4f} | Val PPL: {val_ppl:.2f}")
            
            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = {
                    'iteration': iter_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': losses['val'],
                }
                torch.save(checkpoint, f'{config.checkpoint_dir}/best_model.pt')
                print(f"🏆 New best model! Val loss: {losses['val']:.4f}")
            
            # Save periodic checkpoint
            torch.save(checkpoint, f'{config.checkpoint_dir}/iter_{iter_num:06d}.pt')
            print(f"💾 Checkpoint saved: iter_{iter_num:06d}.pt")
    
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Best perplexity: {math.exp(best_val_loss):.2f}")
    print(f"   Checkpoints: {config.checkpoint_dir}")
    
    return model, tokenizer

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 TRAINING WIKITEXT MODEL (20K ITERATIONS)")
    print("="*60)
    model, tokenizer = train()
    if model:
        print("\n✨ Training complete! Run app_wikitext_fixed.py to generate text!")
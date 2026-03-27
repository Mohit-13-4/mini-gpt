"""
train_advanced.py - Enhanced training with WikiText-103 dataset
"""

import torch
import torch.nn as nn
import time
import os
from datetime import datetime

# Import your modules
from tokenizer import BPETokenizer
from transformer import GPT, GPTConfig
from data_loader import DataLoader

class AdvancedTrainConfig:
    """Enhanced training configuration."""
    
    def __init__(self, dataset='wikitext'):  # Change to 'wikitext' or 'shakespeare'
        # Dataset selection
        self.dataset = dataset  # 'shakespeare' or 'wikitext'
        
        # Model (same as before)
        self.vocab_size = 1500  # Increased vocab size for WikiText-103
        self.block_size = 128
        self.n_layer = 6
        self.n_head = 6
        self.n_embd = 384
        
        # Training (optimized for your GPU)
        self.batch_size = 64
        self.learning_rate = 3e-4
        self.max_iters = 5000
        self.eval_interval = 250
        self.eval_iters = 50
        self.warmup_iters = 100
        
        # Advanced features
        self.grad_clip = 1.0
        self.use_mixed_precision = True
        self.weight_decay = 0.1
        
        # Logging
        self.log_dir = f'runs/{dataset}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.checkpoint_dir = f'checkpoints_{dataset}'
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        print("="*60)
        print("🎯 ADVANCED TRAINING CONFIGURATION")
        print("="*60)
        print(f"   Dataset: {dataset}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Max iterations: {self.max_iters}")
        print(f"   Mixed precision: {self.use_mixed_precision}")
        print(f"   Checkpoints: {self.checkpoint_dir}")
        print("="*60)

def train():
    """Main training function with WikiText-103 support."""
    
    # Configuration
    config = AdvancedTrainConfig(dataset='wikitext')  # Change to 'shakespeare' to go back
    
    # Initialize tokenizer
    print("\n🔤 Initializing tokenizer...")
    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    
    # Load data (NEW: uses WikiText-103!)
    data = DataLoader(config.dataset, config, tokenizer)
    stats = data.get_stats()
    print(f"\n📊 Dataset Stats:")
    print(f"   {stats}")
    
    # Create model
    print("\n🤖 Initializing GPT model...")
    model_config = GPTConfig()
    model_config.vocab_size = config.vocab_size
    model_config.block_size = config.block_size
    model_config.n_layer = config.n_layer
    model_config.n_head = config.n_head
    model_config.n_embd = config.n_embd
    
    model = GPT(model_config)
    model = model.cuda() if torch.cuda.is_available() else model
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model has {n_params:,} parameters")
    
    # Optimizer with weight decay
    optimizer = model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(0.9, 0.95)
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_mixed_precision)
    
    # Learning rate scheduler
    def get_lr(iteration):
        if iteration < config.warmup_iters:
            return config.learning_rate * iteration / config.warmup_iters
        if iteration > config.max_iters:
            return config.learning_rate * 0.1
        decay_ratio = (iteration - config.warmup_iters) / (config.max_iters - config.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config.learning_rate * coeff
    
    # Training loop
    print("\n🎮 TRAINING BEGINS")
    print("="*60)
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    for iter_num in range(config.max_iters):
        # Update learning rate
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Get batch
        x, y = data.get_batch('train')
        
        # Forward/backward with mixed precision
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=config.use_mixed_precision):
            logits, loss = model(x, y)
        
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Logging
        if iter_num % 50 == 0:
            elapsed = time.time() - start_time
            print(f"\nIteration {iter_num}/{config.max_iters} | "
                  f"Loss: {loss.item():.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Time: {elapsed:.1f}s")
        
        # Evaluation
        if iter_num % config.eval_interval == 0 and iter_num > 0:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for _ in range(config.eval_iters):
                    x_val, y_val = data.get_batch('val')
                    with torch.cuda.amp.autocast(enabled=config.use_mixed_precision):
                        _, loss_val = model(x_val, y_val)
                    val_loss += loss_val.item()
            
            val_loss /= config.eval_iters
            train_loss = loss.item()
            
            # Calculate perplexity
            train_ppl = torch.exp(torch.tensor(train_loss)).item()
            val_ppl = torch.exp(torch.tensor(val_loss)).item()
            
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
            
            model.train()
    
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Checkpoints: {config.checkpoint_dir}")
    
    return model, tokenizer

if __name__ == "__main__":
    import math
    model, tokenizer = train()
# inspect_checkpoint.py
import torch
import numpy as np
import os

# ============================================================================
# DUMMY CONFIG CLASSES FOR CHECKPOINT LOADING
# ============================================================================

class FixedModelConfig:
    """Dummy config class for loading checkpoint."""
    def __init__(self, **kwargs):
        pass

class AdvancedTrainConfig:
    """Dummy config class for loading checkpoint."""
    def __init__(self, **kwargs):
        pass

# ============================================================================
# INSPECT
# ============================================================================

print("="*60)
print("🔍 INSPECTING CHECKPOINT")
print("="*60)

# Find checkpoint
checkpoint_path = 'checkpoints_fixed/best_model.pt'
if not os.path.exists(checkpoint_path):
    import glob
    checkpoints = sorted(glob.glob('checkpoints_fixed/iter_*.pt'))
    if checkpoints:
        checkpoint_path = checkpoints[-1]
        print(f"Using latest checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint found!")
        exit(1)

print(f"\n📂 Loading: {checkpoint_path}")

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("\n📦 Checkpoint keys:")
for key in checkpoint.keys():
    print(f"   {key}")

# Check model state dict
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print(f"\n📊 Model state dict has {len(state_dict)} keys")
    
    # Check a few key parameters
    print("\n🔍 Parameter values:")
    
    # Token embedding
    if 'token_embedding.weight' in state_dict:
        param = state_dict['token_embedding.weight']
        print(f"\n   token_embedding.weight:")
        print(f"      Shape: {param.shape}")
        print(f"      Mean: {param.mean().item():.6f}")
        print(f"      Std: {param.std().item():.6f}")
        print(f"      Min: {param.min().item():.6f}")
        print(f"      Max: {param.max().item():.6f}")
        print(f"      First 10 values: {param.flatten()[:10].tolist()}")
    
    # Attention weights
    if 'blocks.0.attn.c_attn.weight' in state_dict:
        param = state_dict['blocks.0.attn.c_attn.weight']
        print(f"\n   blocks.0.attn.c_attn.weight:")
        print(f"      Shape: {param.shape}")
        print(f"      Mean: {param.mean().item():.6f}")
        print(f"      Std: {param.std().item():.6f}")
    
    # LM head
    if 'lm_head.weight' in state_dict:
        param = state_dict['lm_head.weight']
        print(f"\n   lm_head.weight:")
        print(f"      Shape: {param.shape}")
        print(f"      Mean: {param.mean().item():.6f}")
        print(f"      Std: {param.std().item():.6f}")

# Check config in checkpoint
if 'config' in checkpoint:
    print(f"\n⚙️ Config in checkpoint:")
    config = checkpoint['config']
    if hasattr(config, '__dict__'):
        for k, v in config.__dict__.items():
            if not k.startswith('_'):
                print(f"   {k}: {v}")

# Check loss
if 'loss' in checkpoint:
    print(f"\n📈 Validation loss: {checkpoint['loss']:.4f}")
    print(f"   Perplexity: {torch.exp(torch.tensor(checkpoint['loss'])).item():.2f}")

if 'iteration' in checkpoint:
    print(f"   Iteration: {checkpoint['iteration']}")

print("\n" + "="*60)
print("✅ Inspection complete!")
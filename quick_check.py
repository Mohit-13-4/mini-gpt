# quick_check.py
import torch
import os
from tokenizer import BPETokenizer
from transformer import GPT, GPTConfig

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
# CHECK
# ============================================================================

print("="*60)
print("🔍 CHECKING MODEL AND TOKENIZER")
print("="*60)

# Check tokenizer
tokenizer_path = 'checkpoints_fixed/tokenizer.json'
if os.path.exists(tokenizer_path):
    tokenizer = BPETokenizer(vocab_size=5000)
    tokenizer.load(tokenizer_path)
    print(f"\n✅ Tokenizer loaded:")
    print(f"   Vocabulary size: {len(tokenizer)}")
    print(f"   Number of merges: {len(tokenizer.merges)}")
    print(f"   First 10 tokens: {list(tokenizer.vocab.items())[:10]}")
    print(f"   Last 10 tokens: {list(tokenizer.vocab.items())[-10:]}")
else:
    print(f"\n❌ Tokenizer not found at {tokenizer_path}")

# Check model
checkpoint_path = 'checkpoints_fixed/best_model.pt'
if os.path.exists(checkpoint_path):
    print(f"\n✅ Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"   Keys in checkpoint: {list(checkpoint.keys())}")
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"\n📊 Model State Dict Info:")
        for name, param in state_dict.items():
            if 'token_embedding.weight' in name:
                print(f"   Token embedding shape: {param.shape}")
                vocab_size = param.shape[0]
                emb_dim = param.shape[1]
                print(f"   → Vocabulary size: {vocab_size}")
                print(f"   → Embedding dimension: {emb_dim}")
            if 'lm_head.weight' in name:
                print(f"   LM head shape: {param.shape}")
            if 'blocks.0.attn.c_attn.weight' in name:
                print(f"   Attention weight shape: {param.shape}")
    
    if 'loss' in checkpoint:
        print(f"\n📈 Training Metrics:")
        print(f"   Validation loss: {checkpoint['loss']:.4f}")
        print(f"   Perplexity: {torch.exp(torch.tensor(checkpoint['loss'])).item():.2f}")
    
    if 'iteration' in checkpoint:
        print(f"   Iteration: {checkpoint['iteration']}")
    
    if 'config' in checkpoint:
        print(f"\n⚙️ Training Config:")
        config_dict = checkpoint['config']
        if hasattr(config_dict, '__dict__'):
            for k, v in config_dict.__dict__.items():
                if not k.startswith('_'):
                    print(f"   {k}: {v}")
else:
    print(f"\n❌ Model checkpoint not found at {checkpoint_path}")
    
    # Try to find any checkpoint
    import glob
    checkpoints = glob.glob('checkpoints_fixed/iter_*.pt')
    if checkpoints:
        print(f"\n📁 Found other checkpoints:")
        for cp in sorted(checkpoints)[-5:]:
            print(f"   {os.path.basename(cp)}")

print("\n" + "="*60)
print("🔍 VOCABULARY SIZE CHECK")
print("="*60)

# Verify tokenizer and model vocab match
if os.path.exists(tokenizer_path) and os.path.exists(checkpoint_path):
    try:
        # Get model vocab size from checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        for name, param in state_dict.items():
            if 'token_embedding.weight' in name:
                model_vocab = param.shape[0]
                break
        else:
            model_vocab = None
        
        tokenizer_vocab = len(tokenizer)
        
        print(f"\n   Tokenizer vocabulary size: {tokenizer_vocab}")
        print(f"   Model vocabulary size: {model_vocab}")
        
        if tokenizer_vocab == model_vocab:
            print(f"\n   ✅ MATCH! Tokenizer and model vocabularies are aligned.")
        else:
            print(f"\n   ❌ MISMATCH! This will cause gibberish output.")
            print(f"      Tokenizer: {tokenizer_vocab}, Model: {model_vocab}")
            
    except Exception as e:
        print(f"\n   Error checking vocab: {e}")

print("\n" + "="*60)
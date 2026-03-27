# test_earlier_checkpoint.py
import torch
import os
import glob
from tokenizer import BPETokenizer
from transformer import GPT, GPTConfig

# Dummy config classes
class FixedModelConfig:
    def __init__(self, **kwargs):
        pass

class AdvancedTrainConfig:
    def __init__(self, **kwargs):
        pass

print("="*60)
print("🧪 TESTING EARLIER CHECKPOINTS")
print("="*60)

# Load tokenizer
tokenizer = BPETokenizer(vocab_size=5000)
tokenizer.load('checkpoints_fixed/tokenizer.json')
print(f"Tokenizer vocab: {len(tokenizer)}")

# Create model with correct config
config = GPTConfig()
config.vocab_size = 5000
config.block_size = 128
config.n_layer = 8
config.n_head = 8
config.n_embd = 512
config.bias = False

print("\n📐 Model Config:")
print(f"   vocab: {config.vocab_size}, layers: {config.n_layer}, heads: {config.n_head}, embd: {config.n_embd}")

model = GPT(config)

# Find all checkpoints
checkpoints = sorted(glob.glob('checkpoints_fixed/iter_*.pt'))
print(f"\n📁 Found {len(checkpoints)} checkpoints")

# Test the 3 most recent checkpoints
for cp_path in checkpoints[-3:]:
    print(f"\n{'='*50}")
    print(f"Testing: {os.path.basename(cp_path)}")
    
    try:
        checkpoint = torch.load(cp_path, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Test generation
        prompt = "The history of"
        context = tokenizer.encode(prompt)
        context_tensor = torch.tensor(context).unsqueeze(0)
        
        with torch.no_grad():
            generated = model.generate_advanced(context_tensor, max_new_tokens=30, temperature=0.7)
        
        text = tokenizer.decode(generated[0].tolist())
        print(f"   Generated: {text[:150]}")
        
        if 'loss' in checkpoint:
            print(f"   Loss: {checkpoint['loss']:.4f}")
            
    except Exception as e:
        print(f"   Error: {e}")

print("\n" + "="*60)
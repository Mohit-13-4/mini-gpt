# fix_tokenizer.py - Expand tokenizer to 5000 tokens
import torch
import os
from tokenizer import BPETokenizer

print("="*60)
print("🔧 FIXING TOKENIZER VOCABULARY SIZE")
print("="*60)

# Load existing tokenizer
tokenizer = BPETokenizer(vocab_size=5000)  # Target size
tokenizer.load('checkpoints_fixed/tokenizer.json')

print(f"Current tokenizer vocab size: {len(tokenizer)}")
print(f"Target vocab size: 5000")

# Add padding tokens to reach 5000
current_size = len(tokenizer)
needed = 5000 - current_size

if needed > 0:
    print(f"\nAdding {needed} padding tokens...")
    
    # Add dummy tokens to fill the gap
    for i in range(needed):
        token = f"<PAD_{i}>"
        tokenizer.vocab[token] = current_size + i
        tokenizer.reverse_vocab[current_size + i] = token
    
    print(f"New tokenizer vocab size: {len(tokenizer)}")
    
    # Save fixed tokenizer
    tokenizer.save('checkpoints_fixed/tokenizer_fixed.json')
    print(f"\n✅ Fixed tokenizer saved to: checkpoints_fixed/tokenizer_fixed.json")
    
    # Also overwrite the original
    tokenizer.save('checkpoints_fixed/tokenizer.json')
    print(f"✅ Original tokenizer overwritten with fixed version")
else:
    print("Tokenizer already matches target size")

print("\n" + "="*60)
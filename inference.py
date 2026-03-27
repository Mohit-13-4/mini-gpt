"""
inference.py - Generate Shakespeare text with trained GPT model
"""

import torch
import os
from tokenizer import BPETokenizer
from transformer import GPT, GPTConfig

# ============================================================================
# DUMMY CONFIG CLASS FOR CHECKPOINT LOADING
# ============================================================================

class AdvancedTrainConfig:
    """Dummy config class for loading checkpoint. We don't actually use this."""
    def __init__(self, **kwargs):
        pass


# ============================================================================
# LOAD FUNCTIONS
# ============================================================================

def create_shakespeare_tokenizer():
    """
    Create a tokenizer trained on Shakespeare (same as during training).
    This ensures the tokenizer matches what the model expects.
    """
    print("\n🔤 Creating Shakespeare tokenizer...")
    tokenizer = BPETokenizer(vocab_size=1000)
    
    # Load Shakespeare text
    shakespeare_path = 'data/tiny_shakespeare.txt'
    if not os.path.exists(shakespeare_path):
        raise FileNotFoundError(f"Shakespeare file not found at {shakespeare_path}")
    
    with open(shakespeare_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"   Loaded {len(text):,} characters of Shakespeare")
    
    # Train tokenizer (this may take a moment)
    tokenizer.train(text, verbose=False)
    print(f"   Vocabulary size: {len(tokenizer)}")
    print(f"   Number of merges: {len(tokenizer.merges)}")
    
    return tokenizer


def load_best_model(checkpoint_path='checkpoints_shakespeare/best_model.pt'):
    """Load the best model from training."""
    # Create model config (must match training config)
    config = GPTConfig()
    config.vocab_size = 1000
    config.block_size = 128
    config.n_layer = 6
    config.n_head = 6
    config.n_embd = 384
    
    # Initialize model
    model = GPT(config)
    
    # Load checkpoint
    print(f"\n📂 Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("✅ Model loaded on GPU")
    else:
        print("✅ Model loaded on CPU")
    
    # Print training info
    print(f"   Trained for: {checkpoint.get('iteration', 'unknown')} iterations")
    print(f"   Validation loss: {checkpoint.get('loss', 'unknown'):.4f}")
    
    return model


def generate_text(model, tokenizer, prompt="ROMEO:", max_tokens=200, temperature=0.8, top_k=40):
    """
    Generate text from a prompt.
    
    Args:
        model: The GPT model
        tokenizer: The tokenizer
        prompt: Starting text
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling (limit to top k tokens)
    """
    # Encode prompt
    context = tokenizer.encode(prompt)
    context = torch.tensor(context, dtype=torch.long).unsqueeze(0)
    
    if torch.cuda.is_available():
        context = context.cuda()
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode
    text = tokenizer.decode(generated[0].tolist())
    return text


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("🎭 SHAKESPEARE GPT GENERATOR")
    print("="*60)
    
    # Load tokenizer (trained on Shakespeare)
    tokenizer = create_shakespeare_tokenizer()
    
    # Load best model
    print("\n🤖 Loading best model...")
    model = load_best_model('checkpoints_shakespeare/best_model.pt')
    
    # Generate with different prompts
    prompts = [
        "ROMEO:",
        "JULIET: O Romeo, Romeo,",
        "MERCUTIO: A plague on both your houses!",
        "HAMLET: To be or not to be,",
        "KING HENRY: Once more unto the breach,",
    ]
    
    print("\n" + "="*60)
    print("📝 GENERATED SHAKESPEARE")
    print("="*60)
    
    for prompt in prompts:
        print(f"\n{'='*50}")
        print(f"🎭 Prompt: {prompt}")
        print(f"{'='*50}")
        
        generated = generate_text(
            model, 
            tokenizer, 
            prompt=prompt, 
            max_tokens=150, 
            temperature=0.8,
            top_k=40
        )
        
        print(generated)
        print()
    
    # Also try a lower temperature for more focused generation
    print("\n" + "="*60)
    print("🎭 GENERATION WITH LOWER TEMPERATURE (more focused)")
    print("="*60)
    
    print("\nPrompt: ROMEO:")
    generated = generate_text(
        model, 
        tokenizer, 
        prompt="ROMEO:", 
        max_tokens=150, 
        temperature=0.5,  # Lower temperature = more focused
        top_k=40
    )
    print(generated)


    temperatures = [0.3, 0.7, 1.0, 1.5]

    for temp in temperatures:
        print(f"\n--- Temperature {temp} ---")
        text = generate_text(model, tokenizer, "ROMEO:", 
                            max_tokens=50, temperature=temp)
        print(text)
"""
app_fixed.py - Gradio interface for the 28M WikiText-trained GPT model
FIXED VERSION - Proper weight loading
"""

import torch
import gradio as gr
import os
import glob
from tokenizer import BPETokenizer
from transformer import GPT, GPTConfig

# ============================================================================
# DUMMY CONFIG CLASSES FOR CHECKPOINT LOADING
# ============================================================================

class FixedModelConfig:
    def __init__(self, **kwargs):
        pass

class AdvancedTrainConfig:
    def __init__(self, **kwargs):
        pass

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_best_model():
    """Load the best trained model with CORRECT architecture and weights."""
    
    print("="*60)
    print("🎭 LOADING 28M WIKITEXT GPT MODEL")
    print("="*60)
    
    # Find checkpoint
    checkpoint_path = 'checkpoints_fixed/best_model.pt'
    if not os.path.exists(checkpoint_path):
        checkpoints = sorted(glob.glob('checkpoints_fixed/iter_*.pt'))
        if checkpoints:
            checkpoint_path = checkpoints[-1]
            print(f"   Using latest checkpoint: {os.path.basename(checkpoint_path)}")
        else:
            print("   ❌ No checkpoint found!")
            return None, None
    
    # Load checkpoint to get architecture info
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get model architecture from checkpoint
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Detect architecture from state dict
    for name, param in state_dict.items():
        if 'token_embedding.weight' in name:
            model_vocab_size = param.shape[0]
            model_embd = param.shape[1]
            print(f"   Detected: vocab={model_vocab_size}, embd={model_embd}")
            break
    
    # Detect number of layers
    n_layers = 0
    for name in state_dict.keys():
        if 'blocks.' in name:
            layer_num = int(name.split('.')[1])
            n_layers = max(n_layers, layer_num + 1)
    print(f"   Detected layers: {n_layers}")
    
    # Detect number of heads (from attention weight shape)
    n_heads = 8  # Default
    for name, param in state_dict.items():
        if 'attn.c_attn.weight' in name:
            # c_attn weight shape is [3 * n_embd, n_embd]
            # We can infer n_heads from this
            if model_embd % 64 == 0:
                n_heads = model_embd // 64
            break
    print(f"   Detected heads: {n_heads}")
    
    # Load tokenizer
    print("\n📚 Loading tokenizer...")
    tokenizer = BPETokenizer(vocab_size=model_vocab_size)
    
    tokenizer_path = 'checkpoints_fixed/tokenizer.json'
    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
        print(f"   Tokenizer vocabulary size: {len(tokenizer)}")
    else:
        print(f"   ❌ Tokenizer not found!")
        return None, None
    
    # Create model config with detected values
    print("\n🤖 Creating model with correct architecture...")
    
    # Create config directly (bypass GPTConfig defaults)
    config = GPTConfig()
    config.vocab_size = model_vocab_size
    config.block_size = 128
    config.n_layer = n_layers
    config.n_head = n_heads
    config.n_embd = model_embd
    config.bias = False
    config.dropout = 0.1
    
    print(f"\n📐 Model Configuration:")
    print(f"   vocab_size: {config.vocab_size}")
    print(f"   block_size: {config.block_size}")
    print(f"   n_layer: {config.n_layer}")
    print(f"   n_head: {config.n_head}")
    print(f"   n_embd: {config.n_embd}")
    
    # Create model
    model = GPT(config)
    
    # Load weights - THIS IS THE CRITICAL PART
    print("\n📂 Loading trained weights...")
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Move to GPU
    if torch.cuda.is_available():
        model = model.cuda()
        print("   ✅ Model loaded on GPU")
    else:
        print("   ✅ Model loaded on CPU")
    
    # Verify weights loaded correctly by checking a parameter
    print("\n🔍 Verifying weights...")
    first_param = next(model.parameters())
    print(f"   First parameter mean: {first_param.mean().item():.6f}")
    print(f"   First parameter std: {first_param.std().item():.6f}")
    
    # Model stats
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model Statistics:")
    print(f"   Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    
    if 'iteration' in checkpoint:
        print(f"   Trained for: {checkpoint['iteration']:,} iterations")
    if 'loss' in checkpoint:
        loss_val = checkpoint['loss']
        ppl = torch.exp(torch.tensor(loss_val)).item()
        print(f"   Validation loss: {loss_val:.4f}")
        print(f"   Perplexity: {ppl:.2f}")
    
    return model, tokenizer

# ============================================================================
# TEXT GENERATION
# ============================================================================

def generate_text(prompt, max_tokens, temperature, top_k, top_p):
    """Generate coherent Wikipedia-style text."""
    
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return "Error: Model not loaded properly."
    
    if not prompt or prompt.strip() == "":
        prompt = "The history of"
    
    try:
        # Encode prompt
        context = tokenizer.encode(prompt)
        if len(context) == 0:
            context = tokenizer.encode("The")
        
        print(f"   Prompt: '{prompt[:50]}...' -> {len(context)} tokens")
        
        # Truncate if too long
        if len(context) > 96:
            context = context[:96]
            print(f"   Truncated to {len(context)} tokens")
        
        context_tensor = torch.tensor(context, dtype=torch.long).unsqueeze(0)
        
        if torch.cuda.is_available():
            context_tensor = context_tensor.cuda()
        
        # Calculate max tokens
        max_gen_tokens = min(int(max_tokens), 128 - len(context))
        if max_gen_tokens < 5:
            return f"Prompt too long ({len(context)} tokens). Please use a shorter prompt."
        
        print(f"   Generating {max_gen_tokens} tokens...")
        
        # Generate with advanced sampling
        with torch.no_grad():
            top_k_val = int(top_k) if top_k > 0 else None
            top_p_val = float(top_p) if top_p < 1.0 else None
            
            generated = model.generate_advanced(
                context_tensor,
                max_new_tokens=max_gen_tokens,
                temperature=float(temperature),
                top_k=top_k_val,
                top_p=top_p_val
            )
        
        # Decode and clean
        text = tokenizer.decode(generated[0].tolist())
        text = text.replace('<UNK>', '?')
        
        return text
        
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_generation():
    """Quick test to verify model works."""
    print("\n" + "="*60)
    print("🧪 QUICK TEST")
    print("="*60)
    
    test_prompt = "The history of"
    print(f"Testing prompt: '{test_prompt}'")
    
    result = generate_text(test_prompt, 50, 0.7, 40, 0.9)
    print(f"Result: {result}")
    
    return result

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Load model
model, tokenizer = load_best_model()

if model is None:
    print("\n❌ Failed to load model. Exiting...")
    exit(1)

# Run a quick test to verify
print("\n" + "="*60)
print("🧪 RUNNING QUICK TEST...")
print("="*60)
test_result = test_generation()
print("\n✅ Quick test complete!")
print(f"Test output: {test_result[:200]}...")

# Create interface
with gr.Blocks(title="WikiText GPT - 28M") as demo:
    gr.Markdown("""
    # 🎭 WikiText GPT - 28 Million Parameters
    
    This GPT model was trained from scratch on the **full WikiText-103 dataset**!
    
    ## 🏆 Results
    | Metric | Value |
    |--------|-------|
    | **Perplexity** | **3.70** |
    | **Parameters** | **28M** |
    | **Layers** | **8** |
    | **Heads** | **8** |
    | **Embedding Dim** | **512** |
    
    ## 💡 Generation Tips
    - **Temperature 0.7-0.9** → Creative text
    - **Temperature 0.3-0.5** → Focused text
    - **Top-K 40 + Top-P 0.9** → Best balance
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="📝 Prompt",
                value="The history of",
                placeholder="Enter your prompt here...",
                lines=3
            )
            
            max_tokens_slider = gr.Slider(
                label="📏 Max Tokens",
                minimum=20,
                maximum=100,
                value=60,
                step=10
            )
            
            temperature_slider = gr.Slider(
                label="🌡️ Temperature",
                minimum=0.1,
                maximum=1.5,
                value=0.7,
                step=0.05
            )
            
            top_k_slider = gr.Slider(
                label="🎯 Top-K",
                minimum=0,
                maximum=100,
                value=40,
                step=5
            )
            
            top_p_slider = gr.Slider(
                label="🎲 Top-P (Nucleus)",
                minimum=0.5,
                maximum=1.0,
                value=0.9,
                step=0.05
            )
            
            generate_btn = gr.Button("🎭 Generate Text", variant="primary")
        
        with gr.Column(scale=3):
            output_text = gr.Textbox(
                label="📖 Generated Text",
                lines=20,
                placeholder="Your generated text will appear here..."
            )
    
    generate_btn.click(
        fn=generate_text,
        inputs=[prompt_input, max_tokens_slider, temperature_slider, top_k_slider, top_p_slider],
        outputs=output_text
    )
    
    prompt_input.submit(
        fn=generate_text,
        inputs=[prompt_input, max_tokens_slider, temperature_slider, top_k_slider, top_p_slider],
        outputs=output_text
    )

if __name__ == "__main__":
    print("\n🚀 Launching Gradio interface...")
    print("   Open http://localhost:7860 in your browser")
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
"""
app_wikitext_fixed.py - Gradio interface for the WikiText GPT model
Memory-optimized version
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

class WikiTextConfig:
    def __init__(self, **kwargs):
        pass

class AdvancedTrainConfig:
    def __init__(self, **kwargs):
        pass

# ============================================================================
# MODEL LOADING
# ============================================================================

def get_best_checkpoint():
    checkpoint_dir = 'checkpoints_wikitext_fixed'
    best_path = os.path.join(checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_path):
        return best_path
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, 'iter_*.pt')))
    if checkpoints:
        return checkpoints[-1]
    return None

def load_wikitext_model():
    """Load the best trained WikiText model with memory optimization."""
    
    print("="*60)
    print("🎭 LOADING WIKITEXT GPT MODEL")
    print("="*60)
    
    checkpoint_path = get_best_checkpoint()
    if checkpoint_path is None:
        print("   ❌ No checkpoint found!")
        return None, None
    
    print(f"   Loading checkpoint: {os.path.basename(checkpoint_path)}")
    
    # Load checkpoint on CPU first to save GPU memory
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get model architecture from checkpoint
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Detect architecture
    for name, param in state_dict.items():
        if 'token_embedding.weight' in name:
            vocab_size = param.shape[0]
            embd_dim = param.shape[1]
            print(f"   Detected: vocab={vocab_size}, embd={embd_dim}")
            break
    else:
        vocab_size = 5000
        embd_dim = 384
    
    # Detect number of layers
    n_layers = 0
    for name in state_dict.keys():
        if 'blocks.' in name:
            layer_num = int(name.split('.')[1])
            n_layers = max(n_layers, layer_num + 1)
    print(f"   Detected layers: {n_layers}")
    
    n_heads = embd_dim // 64
    print(f"   Detected heads: {n_heads}")
    
    # Load tokenizer
    print("\n📚 Loading tokenizer...")
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    
    tokenizer_path = 'checkpoints_wikitext_fixed/tokenizer.json'
    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
        print(f"   Tokenizer vocabulary: {len(tokenizer)} tokens")
        print(f"   Number of BPE merges: {len(tokenizer.merges)}")
    else:
        print(f"   ❌ Tokenizer not found")
        return None, None
    
    # Create model
    print("\n🤖 Creating model...")
    config = GPTConfig()
    config.vocab_size = vocab_size
    config.block_size = 128
    config.n_layer = n_layers
    config.n_head = n_heads
    config.n_embd = embd_dim
    config.bias = False
    config.dropout = 0.1
    
    model = GPT(config)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Move to GPU (this will allocate memory)
    if torch.cuda.is_available():
        model = model.cuda()
        print("\n   ✅ Model loaded on GPU")
        # Clear cache
        torch.cuda.empty_cache()
    
    # Model stats
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model Statistics:")
    print(f"   Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    
    if 'loss' in checkpoint:
        print(f"   Validation loss: {checkpoint['loss']:.4f}")
        print(f"   Perplexity: {torch.exp(torch.tensor(checkpoint['loss'])).item():.2f}")
    
    return model, tokenizer

# ============================================================================
# MEMORY-OPTIMIZED TEXT GENERATION
# ============================================================================

def generate_text(prompt, max_tokens, temperature, top_k, top_p, repetition_penalty):
    """Memory-optimized text generation."""
    
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return "Error: Model not loaded properly."
    
    if not prompt or prompt.strip() == "":
        prompt = "The history of"
    
    try:
        # Clear GPU cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Encode prompt
        context = tokenizer.encode(prompt)
        if len(context) == 0:
            context = tokenizer.encode("The")
        
        print(f"   Prompt: '{prompt[:50]}...' -> {len(context)} tokens")
        
        # Limit context length to save memory
        if len(context) > 64:
            context = context[:64]
            print(f"   Truncated to {len(context)} tokens")
        
        # Limit max tokens to prevent memory issues
        max_gen_tokens = min(int(max_tokens), 50)  # Reduced from 100 to 50
        if max_gen_tokens < 5:
            return f"Prompt too long ({len(context)} tokens). Please use a shorter prompt."
        
        print(f"   Generating {max_gen_tokens} tokens...")
        
        context_tensor = torch.tensor(context, dtype=torch.long).unsqueeze(0)
        
        if torch.cuda.is_available():
            context_tensor = context_tensor.cuda()
        
        # Generate with repetition penalty
        generated = context_tensor
        generated_tokens = context_tensor[0].tolist()
        
        for step in range(max_gen_tokens):
            # Clear cache periodically
            if step % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logits, _ = model(generated, use_cache=True)
            logits = logits[:, -1, :] / temperature
            
            # Apply repetition penalty
            penalty_val = float(repetition_penalty)
            if penalty_val > 1.0:
                recent_tokens = set(generated_tokens[-15:])  # Smaller window
                for token in recent_tokens:
                    logits[:, token] /= penalty_val
            
            # Top-K sampling
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Top-P sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, -float('Inf'))
            
            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            generated_tokens.append(next_token.item())
        
        # Decode
        text = tokenizer.decode(generated[0].tolist())
        text = text.replace('<UNK>', '?')
        
        # Final cache clear
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return text
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return "Error: GPU out of memory. Please try:\n- Reducing Max Tokens (try 30-40)\n- Using a shorter prompt\n- Lowering temperature or top-k"
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Load model
model, tokenizer = load_wikitext_model()

if model is None:
    print("\n❌ Failed to load model. Exiting...")
    exit(1)

# Create interface
with gr.Blocks(title="WikiText GPT - 12.6M Parameters") as demo:
    gr.Markdown("""
    # 🎭 WikiText GPT - 12.6 Million Parameters
    
    This GPT model was trained from scratch on **WikiText-2** with proper BPE tokenization!
    
    ## 🏆 Final Results
    | Metric | Value |
    |--------|-------|
    | **Best Validation Loss** | **2.06** |
    | **Perplexity** | **7.84** |
    | **Parameters** | **12.6M** |
    | **Training Time** | **32 minutes** |
    | **BPE Merges** | **4,073** |
    
    ## ⚠️ Memory Note
    Due to GPU memory constraints, generation is limited to **50 tokens**.
    
    ## 💡 Generation Tips
    - **Temperature 0.7-0.9** → Creative text
    - **Temperature 0.3-0.5** → Focused text
    - **Top-K 40 + Top-P 0.9** → Best balance
    - **Repetition Penalty 1.1-1.3** → Prevents repetition
    
    ## 📝 Example Prompts
    - `The history of`
    - `According to scientists`
    - `In the year 1900`
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="📝 Prompt",
                value="The history of",
                placeholder="Enter your prompt here...",
                lines=2
            )
            
            max_tokens_slider = gr.Slider(
                label="📏 Max Tokens (max 50)",
                minimum=10,
                maximum=50,
                value=40,
                step=5
            )
            
            temperature_slider = gr.Slider(
                label="🌡️ Temperature",
                minimum=0.1,
                maximum=1.5,
                value=0.8,
                step=0.05
            )
            
            top_k_slider = gr.Slider(
                label="🎯 Top-K (0 = disabled)",
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
            
            penalty_slider = gr.Slider(
                label="🔁 Repetition Penalty",
                minimum=1.0,
                maximum=1.5,
                value=1.2,
                step=0.05
            )
            
            generate_btn = gr.Button("🎭 Generate Text", variant="primary")
        
        with gr.Column(scale=3):
            output_text = gr.Textbox(
                label="📖 Generated Text",
                lines=15,
                placeholder="Your generated text will appear here..."
            )
    
    generate_btn.click(
        fn=generate_text,
        inputs=[
            prompt_input, 
            max_tokens_slider, 
            temperature_slider, 
            top_k_slider, 
            top_p_slider,
            penalty_slider
        ],
        outputs=output_text
    )
    
    prompt_input.submit(
        fn=generate_text,
        inputs=[
            prompt_input, 
            max_tokens_slider, 
            temperature_slider, 
            top_k_slider, 
            top_p_slider,
            penalty_slider
        ],
        outputs=output_text
    )
    
    # Example prompts
    gr.Markdown("### 📝 Quick Examples")
    example_prompts = [
        "The history of",
        "According to scientists",
        "In the year 1900",
    ]
    
    for prompt in example_prompts:
        gr.Button(prompt, size="sm").click(
            fn=lambda p=prompt: p,
            inputs=[],
            outputs=prompt_input
        )
    
    gr.Markdown("""
    ### ⚙️ Recommended Settings
    
    | Setting | Creative | Factual |
    |---------|----------|---------|
    | Temperature | 0.8-0.9 | 0.4-0.6 |
    | Top-K | 40-50 | 30-40 |
    | Top-P | 0.9-0.95 | 0.85-0.9 |
    | Repetition Penalty | 1.1-1.2 | 1.2-1.3 |
    """)

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Launching Gradio interface...")
    print("   Open http://localhost:7860 in your browser")
    print("   Press Ctrl+C to stop the server")
    print("="*60)
    print("\n💡 Tips:")
    print("   - Max tokens limited to 50 for memory stability")
    print("   - Keep prompts under 64 tokens")
    print("   - Use repetition penalty 1.2 to avoid repeats")
    print("="*60)
    
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
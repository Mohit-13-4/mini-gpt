"""
app_shakespeare_final.py - Working Shakespeare GPT Interface
UPDATED with better generation defaults
"""

import torch
import gradio as gr
import os
from tokenizer import BPETokenizer
from transformer import GPT, GPTConfig

# ============================================================================
# DUMMY CONFIG FOR CHECKPOINT LOADING
# ============================================================================

class AdvancedTrainConfig:
    def __init__(self, **kwargs):
        pass

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_shakespeare_model():
    """Load the working Shakespeare model."""
    
    print("="*60)
    print("🎭 LOADING SHAKESPEARE GPT MODEL (WORKING!)")
    print("="*60)
    
    # Load tokenizer trained on Shakespeare
    print("\n📚 Loading tokenizer...")
    tokenizer = BPETokenizer(vocab_size=1000)
    
    # Train tokenizer on Shakespeare
    with open('data/tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer.train(text, verbose=False)
    print(f"   Tokenizer vocabulary size: {len(tokenizer)}")
    
    # Load model
    print("\n🤖 Loading model...")
    config = GPTConfig()
    config.vocab_size = 1000
    config.block_size = 128
    config.n_layer = 6
    config.n_head = 6
    config.n_embd = 384
    config.bias = False
    
    model = GPT(config)
    
    checkpoint_path = 'checkpoints_shakespeare/best_model.pt'
    if not os.path.exists(checkpoint_path):
        print(f"   ❌ Checkpoint not found at {checkpoint_path}")
        return None, None
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("   ✅ Model loaded on GPU")
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model Statistics:")
    print(f"   Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"   Validation loss: {checkpoint.get('loss', 'unknown'):.4f}")
    
    return model, tokenizer

# ============================================================================
# TEXT GENERATION WITH BETTER DEFAULTS
# ============================================================================

def generate_shakespeare(prompt, max_tokens, temperature, top_k, top_p):
    """Generate Shakespearean text with improved sampling."""
    
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return "Error: Model not loaded properly."
    
    if not prompt or prompt.strip() == "":
        prompt = "ROMEO:"
    
    try:
        # Encode prompt
        context = tokenizer.encode(prompt)
        if len(context) == 0:
            context = tokenizer.encode("ROMEO:")
        
        context_tensor = torch.tensor(context, dtype=torch.long).unsqueeze(0)
        
        if torch.cuda.is_available():
            context_tensor = context_tensor.cuda()
        
        # Adjust temperature if it's too low (causes repetition)
        if temperature < 0.5:
            print(f"   ⚠️ Low temperature ({temperature}) may cause repetition. Consider 0.7-0.9 for better results.")
        
        # Generate with advanced sampling
        with torch.no_grad():
            top_k_val = int(top_k) if top_k > 0 else None
            top_p_val = float(top_p) if top_p < 1.0 else None
            
            generated = model.generate_advanced(
                context_tensor,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_k=top_k_val,
                top_p=top_p_val
            )
        
        # Decode
        text = tokenizer.decode(generated[0].tolist())
        text = text.replace('<UNK>', '?')
        
        # Check for repetition and warn
        words = text.split()
        if len(words) > 20:
            # Check if the same word repeats too much
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                text = text + "\n\n⚠️ Note: Output shows repetition. Try increasing temperature (0.7-0.9) or adjusting Top-K/Top-P."
        
        return text
        
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================================================
# REPETITION PENALTY FUNCTION (ADVANCED)
# ============================================================================

def generate_with_penalty(prompt, max_tokens, temperature, top_k, top_p, repetition_penalty=1.2):
    """Generate with repetition penalty to reduce repeats."""
    
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return "Error: Model not loaded properly."
    
    if not prompt or prompt.strip() == "":
        prompt = "ROMEO:"
    
    try:
        context = tokenizer.encode(prompt)
        if len(context) == 0:
            context = tokenizer.encode("ROMEO:")
        
        context_tensor = torch.tensor(context, dtype=torch.long).unsqueeze(0)
        
        if torch.cuda.is_available():
            context_tensor = context_tensor.cuda()
        
        # Custom generation with repetition penalty
        generated = context_tensor
        generated_tokens = context_tensor[0].tolist()
        
        for _ in range(int(max_tokens)):
            logits, _ = model(generated, use_cache=True)
            logits = logits[:, -1, :] / temperature
            
            # Apply repetition penalty
            for token in set(generated_tokens[-20:]):  # Look at last 20 tokens
                logits[:, token] /= repetition_penalty
            
            # Top-K
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Top-P
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, -float('Inf'))
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            generated_tokens.append(next_token.item())
        
        text = tokenizer.decode(generated[0].tolist())
        text = text.replace('<UNK>', '?')
        return text
        
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Load model
model, tokenizer = load_shakespeare_model()

if model is None:
    print("\n❌ Failed to load model. Exiting...")
    exit(1)

# Create interface with better defaults
with gr.Blocks(title="Shakespeare GPT - Working Model", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎭 Shakespeare GPT - 11 Million Parameters
    
    This GPT model was trained from scratch on **Shakespeare's complete works**!
    
    ## 🏆 Results
    | Metric | Value |
    |--------|-------|
    | **Parameters** | **11M** |
    | **Layers** | **6** |
    | **Heads** | **6** |
    | **Embedding Dim** | **384** |
    
    ## 💡 Generation Tips
    - **Temperature 0.8-1.0** → Creative, varied text (BEST for avoiding repetition)
    - **Temperature 0.5-0.7** → Balanced
    - **Temperature 0.3-0.5** → Focused, predictable (may cause repetition)
    - **Top-K 40 + Top-P 0.9** → Best balance
    
    ## ⚠️ If you see repetition:
    - **Increase Temperature** to 0.8-1.0
    - **Decrease Top-K** to 30-40
    - **Try Top-P** around 0.85-0.95
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="📝 Prompt",
                value="ROMEO:",
                placeholder="Enter your prompt here...",
                lines=3
            )
            
            max_tokens_slider = gr.Slider(
                label="📏 Max Tokens",
                minimum=20,
                maximum=150,
                value=80,
                step=10
            )
            
            temperature_slider = gr.Slider(
                label="🌡️ Temperature (Higher = More Creative)",
                minimum=0.1,
                maximum=1.5,
                value=0.9,  # Changed from 0.8 to 0.9 for better variety
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
                label="🎲 Top-P (Nucleus, 1.0 = disabled)",
                minimum=0.5,
                maximum=1.0,
                value=0.9,
                step=0.05
            )
            
            repetition_penalty_slider = gr.Slider(
                label="🔁 Repetition Penalty (1.0 = none, higher = less repetition)",
                minimum=1.0,
                maximum=1.5,
                value=1.2,
                step=0.05
            )
            
            use_penalty = gr.Checkbox(
                label="Use Repetition Penalty (helps avoid repeats)",
                value=True
            )
            
            generate_btn = gr.Button("🎭 Generate Shakespeare", variant="primary")
        
        with gr.Column(scale=3):
            output_text = gr.Textbox(
                label="📖 Generated Text",
                lines=20,
                placeholder="Your Shakespearean text will appear here..."
            )
    
    # Generation function with penalty option
    def generate_wrapper(prompt, max_tokens, temperature, top_k, top_p, use_penalty, penalty_val):
        if use_penalty and penalty_val > 1.0:
            return generate_with_penalty(prompt, max_tokens, temperature, top_k, top_p, penalty_val)
        else:
            return generate_shakespeare(prompt, max_tokens, temperature, top_k, top_p)
    
    generate_btn.click(
        fn=generate_wrapper,
        inputs=[prompt_input, max_tokens_slider, temperature_slider, top_k_slider, top_p_slider, use_penalty, repetition_penalty_slider],
        outputs=output_text
    )
    
    prompt_input.submit(
        fn=generate_wrapper,
        inputs=[prompt_input, max_tokens_slider, temperature_slider, top_k_slider, top_p_slider, use_penalty, repetition_penalty_slider],
        outputs=output_text
    )
    
    # Example prompts
    gr.Markdown("### 📝 Try these examples (Click to use):")
    examples = [
        "ROMEO:",
        "JULIET: O Romeo, Romeo,",
        "MERCUTIO: A plague on both your houses!",
        "HAMLET: To be or not to be,",
        "KING HENRY: Once more unto the breach,",
    ]
    
    for example in examples:
        gr.Button(example, size="sm").click(
            fn=lambda p=example: p,
            inputs=[],
            outputs=prompt_input
        )

if __name__ == "__main__":
    print("\n🚀 Launching Gradio interface...")
    print("   Open http://localhost:7860 in your browser")
    print("   Recommended settings: Temperature 0.9, Top-K 40, Top-P 0.9")
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
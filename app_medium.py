"""
app_medium.py - Gradio interface for 32.3M parameter GPT model
"""

import torch
import gradio as gr
import os
from tokenizer import BPETokenizer
from transformer import GPT, GPTConfig

# ============================================================================
# DUMMY CONFIG CLASS FOR CHECKPOINT LOADING
# ============================================================================

class MediumModelConfig:
    """Dummy config class for loading checkpoint."""
    def __init__(self, **kwargs):
        pass

class AdvancedTrainConfig:
    """Dummy config class for loading checkpoint."""
    def __init__(self, **kwargs):
        pass

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_medium_model():
    """Load the 32.3M parameter model."""
    
    print("📚 Loading tokenizer...")
    tokenizer = BPETokenizer(vocab_size=1500)
    
    # Load the tokenizer from checkpoints_medium
    tokenizer_path = 'checkpoints_medium/tokenizer.json'
    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
        print(f"   Vocabulary size: {len(tokenizer)}")
    else:
        print("   Tokenizer not found! Please train the model first.")
        return None, None
    
    print("🤖 Loading 32.3M parameter model...")
    
    # Create model config that matches the trained model
    config = GPTConfig()
    config.vocab_size = 1500
    config.block_size = 128
    config.n_layer = 10      # 10 layers (from training)
    config.n_head = 8        # 8 heads
    config.n_embd = 512      # 512 embedding dim
    config.bias = False
    config.dropout = 0.1
    
    # Create model
    model = GPT(config)
    
    # Load checkpoint
    checkpoint_path = 'checkpoints_medium/best_model.pt'
    if not os.path.exists(checkpoint_path):
        print(f"   Checkpoint not found at {checkpoint_path}")
        return None, None
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("✅ Model loaded on GPU")
    else:
        print("✅ Model loaded on CPU")
    
    # Print model info
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Best validation loss: {checkpoint.get('loss', 'unknown'):.4f}")
    print(f"   Perplexity: {torch.exp(torch.tensor(checkpoint.get('loss', 1.68))):.2f}")
    print(f"   Trained for: {checkpoint.get('iteration', 'unknown')} iterations")
    
    return model, tokenizer

# ============================================================================
# TEXT GENERATION
# ============================================================================

def generate_text(prompt, max_tokens, temperature, top_k, top_p):
    """Generate text with the medium model."""
    
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return "Error: Model not loaded properly. Please check checkpoint files."
    
    if not prompt or prompt.strip() == "":
        prompt = "The history of"
    
    try:
        context = tokenizer.encode(prompt)
        if len(context) == 0:
            context = tokenizer.encode("The")
        
        # Truncate if too long
        if len(context) > 64:
            context = context[:64]
        
        context_tensor = torch.tensor(context, dtype=torch.long).unsqueeze(0)
        
        if torch.cuda.is_available():
            context_tensor = context_tensor.cuda()
        
        # Ensure we don't exceed block size
        max_gen_tokens = min(int(max_tokens), 128 - len(context))
        if max_gen_tokens < 1:
            return f"Prompt too long ({len(context)} tokens). Please use shorter prompt."
        
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
        
        text = tokenizer.decode(generated[0].tolist())
        text = text.replace('<UNK>', '?')
        
        return text
        
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Load model
print("="*60)
print("🎭 MEDIUM GPT MODEL (32.3M PARAMETERS)")
print("="*60)
model, tokenizer = load_medium_model()

if model is None:
    print("\n❌ Failed to load model. Please ensure training completed successfully.")
    print("   Checkpoints should be in: checkpoints_medium/")
    exit(1)

# Create Gradio interface
with gr.Blocks(title="Medium GPT - 32.3M Parameters") as demo:
    gr.Markdown("""
    # 🎭 Medium GPT - 32.3 Million Parameters
    
    This model was trained from scratch on WikiText-103 for 10,000 iterations!
    
    **Model Specs:**
    - Parameters: **32.3 million**
    - Layers: 10
    - Heads: 8
    - Embedding Dimension: 512
    - Training: 10,000 iterations (57 minutes on RTX 3050)
    
    **Results:**
    - Best Validation Loss: **1.68**
    - Perplexity: **5.37**
    - 11.5% better than the 11M model!
    
    **Tips:**
    - Use **Temperature 0.7-0.9** for creative text
    - Use **Temperature 0.3-0.5** for focused, factual text
    - Try prompts about history, science, technology
    - **Top-K 40 + Top-P 0.9** works well
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
                value=80,
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
    
    # Example prompts
    gr.Markdown("### 📝 Try these examples:")
    examples = [
        "The history of",
        "In the year 1900",
        "According to scientists",
        "The most important invention",
        "During the Middle Ages",
        "Albert Einstein was",
    ]
    
    for example in examples:
        gr.Button(example).click(
            fn=lambda p=example: p,
            inputs=[],
            outputs=prompt_input
        )

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    print("\n🚀 Launching Gradio interface...")
    print("   Open http://localhost:7860 in your browser")
    print("   Press Ctrl+C to stop the server")
    
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
"""
app.py - Gradio web interface for Shakespeare GPT
Run with: python app.py
"""

import torch
import gradio as gr
import os
from tokenizer import BPETokenizer
from transformer import GPT, GPTConfig

# ============================================================================
# MODEL LOADING
# ============================================================================

class AdvancedTrainConfig:
    """Dummy config for checkpoint loading."""
    def __init__(self, **kwargs):
        pass

def load_model_and_tokenizer(checkpoint_path='checkpoints_shakespeare/best_model.pt'):
    """Load the trained model and Shakespeare tokenizer."""
    
    # Create tokenizer trained on Shakespeare
    print("📚 Loading tokenizer...")
    tokenizer = BPETokenizer(vocab_size=1000)
    
    # Train tokenizer on Shakespeare
    with open('data/tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer.train(text, verbose=False)
    
    print(f"   Vocabulary size: {len(tokenizer)}")
    
    # Create model
    print("🤖 Loading model...")
    config = GPTConfig()
    config.vocab_size = 1000
    config.block_size = 128
    config.n_layer = 6
    config.n_head = 6
    config.n_embd = 384
    
    model = GPT(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("✅ Model loaded on GPU")
    else:
        print("✅ Model loaded on CPU")
    
    return model, tokenizer

def generate_text(prompt, max_tokens, temperature, top_k, top_p):
    """Generate text with adjustable parameters."""
    
    global model, tokenizer
    
    # Handle empty prompt
    if not prompt or prompt.strip() == "":
        prompt = "ROMEO:"
    
    try:
        # Encode prompt
        context = tokenizer.encode(prompt)
        if len(context) == 0:
            context = tokenizer.encode("ROMEO:")
        
        # TRUNCATE if context is too long (leave room for generation)
        max_context_len = 64  # Leave half for generation
        if len(context) > max_context_len:
            context = context[-max_context_len:]
            print(f"   Truncated prompt to {len(context)} tokens")
        
        context = torch.tensor(context, dtype=torch.long).unsqueeze(0)
        
        if torch.cuda.is_available():
            context = context.cuda()
        
        # Ensure we don't exceed block size
        max_tokens = min(max_tokens, 128 - len(context))
        if max_tokens < 1:
            return "Error: Prompt is too long. Please use a shorter prompt."
        
        # Generate
        with torch.no_grad():
            # Handle top_k=0 meaning disabled
            top_k_val = int(top_k) if top_k > 0 else None
            # Handle top_p=1.0 meaning disabled
            top_p_val = float(top_p) if top_p < 1.0 else None
            
            generated = model.generate_advanced(
                context,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_k=top_k_val,
                top_p=top_p_val
            )
        
        # Decode
        text = tokenizer.decode(generated[0].tolist())
        
        # Clean up
        text = text.replace('<UNK>', '?')
        
        return text
        
    except Exception as e:
        return f"Error: {str(e)}\n\nPlease try again with different settings."

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Load model once at startup
print("="*60)
print("🎭 SHAKESPEARE GPT - GRADIO INTERFACE")
print("="*60)
model, tokenizer = load_model_and_tokenizer()

# Example prompts - each example must have exactly 5 values matching the function parameters
examples = [
    ["ROMEO:", 150, 0.8, 40, 0.9],
    ["JULIET: O Romeo, Romeo,", 150, 0.8, 40, 0.9],
    ["MERCUTIO: A plague on both your houses!", 150, 0.8, 40, 0.9],
    ["HAMLET: To be or not to be,", 150, 0.7, 50, 0.85],
    ["KING HENRY: Once more unto the breach,", 150, 0.9, 30, 0.95],
]

# Create the interface
with gr.Blocks(title="Shakespeare GPT") as demo:
    gr.Markdown("""
    # 🎭 Shakespeare GPT - From Scratch
    
    This GPT model was trained from scratch on Shakespeare's complete works!
    
    **Model Specs:**
    - Architecture: Decoder-only Transformer (GPT-style)
    - Parameters: **11 million**
    - Training: 1000 iterations on Shakespeare
    - Hardware: NVIDIA RTX 3050 4GB
    
    ### Tips:
    - Use **Temperature 0.7-0.9** for creative Shakespeare
    - Use **Temperature 0.3-0.5** for focused, predictable text
    - Try prompts like **ROMEO:**, **JULIET:**, **HAMLET:**
    - **Top-K 40 + Top-P 0.9** works well together
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="🎭 Prompt",
                value="ROMEO:",
                placeholder="Enter your prompt here...",
                lines=3
            )
            
            max_tokens_slider = gr.Slider(
                label="📏 Max Tokens",
                minimum=50,
                maximum=300,
                value=150,
                step=10
            )
            
            temperature_slider = gr.Slider(
                label="🌡️ Temperature",
                minimum=0.1,
                maximum=1.5,
                value=0.8,
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
            
            generate_btn = gr.Button("🎭 Generate Shakespeare", variant="primary")
        
        with gr.Column(scale=3):
            output_text = gr.Textbox(
                label="🎭 Generated Shakespeare",
                lines=20,
                placeholder="Your Shakespearean text will appear here..."
            )
    
    # Connect the generate button - pass inputs in correct order
    generate_btn.click(
        fn=generate_text,
        inputs=[prompt_input, max_tokens_slider, temperature_slider, top_k_slider, top_p_slider],
        outputs=output_text
    )
    
    # Also allow pressing Enter in the prompt box
    prompt_input.submit(
        fn=generate_text,
        inputs=[prompt_input, max_tokens_slider, temperature_slider, top_k_slider, top_p_slider],
        outputs=output_text
    )
    
    # Add examples with proper formatting
    gr.Markdown("### 📝 Try these examples:")
    for example in examples:
        gr.Examples(
            examples=[example],
            inputs=[prompt_input, max_tokens_slider, temperature_slider, top_k_slider, top_p_slider],
            outputs=output_text,
            fn=generate_text,
            cache_examples=False,
            label=f"Example: {example[0]}"
        )

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    print("\n🚀 Launching Gradio interface...")
    print("   Open http://localhost:7860 in your browser")
    print("   Press Ctrl+C to stop the server")
    
    # Remove share=True to avoid connection issues
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
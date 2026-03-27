"""
app.py - Gradio web interface for GPT (now using WikiText-trained model!)
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

def load_model_and_tokenizer(checkpoint_path='checkpoints_wikitext/best_model.pt'):
    """Load the WikiText-trained model and tokenizer."""
    
    # Create tokenizer with correct vocab size for WikiText
    print("📚 Loading WikiText tokenizer...")
    tokenizer = BPETokenizer(vocab_size=1500)  # IMPORTANT: 1500 for WikiText!
    
    # Check if we have a saved tokenizer
    tokenizer_path = 'checkpoints_wikitext/tokenizer.json'
    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
        print(f"   Loaded saved tokenizer: {len(tokenizer)} tokens")
    else:
        # Train tokenizer on WikiText
        print("   Training tokenizer on WikiText-103...")
        from data_loader import DataLoader
        wiki_data = DataLoader('wikitext', None, None)
        text = wiki_data._load_wikitext()
        tokenizer.train(text, verbose=False)
        tokenizer.save(tokenizer_path)
        print(f"   Trained tokenizer: {len(tokenizer)} tokens")
    
    # Create model with matching vocab size
    print("🤖 Loading WikiText-trained model...")
    config = GPTConfig()
    config.vocab_size = 1500  # IMPORTANT: Must match tokenizer!
    config.block_size = 128
    config.n_layer = 6
    config.n_head = 6
    config.n_embd = 384
    
    model = GPT(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("✅ Model loaded on GPU")
    else:
        print("✅ Model loaded on CPU")
    
    # Print training info
    print(f"   Best validation loss: {checkpoint.get('loss', 'unknown'):.4f}")
    
    return model, tokenizer

def generate_text(prompt, max_tokens, temperature, top_k, top_p):
    """Generate text with adjustable parameters."""
    
    global model, tokenizer
    
    # Handle empty prompt
    if not prompt or prompt.strip() == "":
        prompt = "The history of"
    
    try:
        # Encode prompt
        context = tokenizer.encode(prompt)
        if len(context) == 0:
            context = tokenizer.encode("The")
        
        # Truncate if too long
        if len(context) > 64:
            context = context[-64:]
        
        context_tensor = torch.tensor(context, dtype=torch.long).unsqueeze(0)
        
        if torch.cuda.is_available():
            context_tensor = context_tensor.cuda()
        
        # Ensure we don't exceed block size
        max_gen_tokens = min(int(max_tokens), 128 - len(context))
        if max_gen_tokens < 1:
            return f"Prompt too long ({len(context)} tokens). Please use a shorter prompt (max 64 tokens)."
        
        # Generate
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
        
        # Decode
        text = tokenizer.decode(generated[0].tolist())
        text = text.replace('<UNK>', '?')
        
        return text
        
    except Exception as e:
        return f"Error: {str(e)}\n\nPlease try again with different settings."

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Load model once at startup
print("="*60)
print("🎭 GPT - WikiText-103 Model")
print("="*60)
model, tokenizer = load_model_and_tokenizer()

# Example prompts for WikiText
examples = [
    ["The history of", 80, 0.8, 40, 0.9],
    ["In the year 1900", 80, 0.8, 40, 0.9],
    ["According to scientists", 80, 0.7, 50, 0.85],
    ["The most important invention", 80, 0.8, 40, 0.9],
    ["During the Middle Ages", 80, 0.75, 45, 0.88],
]

# Create the interface
with gr.Blocks(title="WikiText GPT", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎭 GPT from Scratch - WikiText-103 Model
    
    This GPT was trained from scratch on **WikiText-103**, a large dataset of Wikipedia articles!
    
    **Model Specs:**
    - Architecture: Decoder-only Transformer (GPT-style)
    - Parameters: **11 million**
    - Training: 5,000 iterations on WikiText-103
    - Best Validation Loss: **1.80** (Perplexity: **6.07**)
    - Hardware: NVIDIA RTX 3050 4GB
    
    ### 📊 Results:
    - 🚀 **52% better perplexity** than Shakespeare model!
    - 📚 Understands modern English, not just Shakespeare
    - 🎯 Much more coherent and diverse outputs
    
    ### Tips:
    - Use **Temperature 0.7-0.9** for creative text
    - Use **Temperature 0.3-0.5** for focused, factual text
    - Try prompts like **"The history of"**, **"In the year"**, **"According to"**
    - **Top-K 40 + Top-P 0.9** works well together
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
                minimum=10,
                maximum=100,
                value=80,
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
            
            generate_btn = gr.Button("🎭 Generate Text", variant="primary")
        
        with gr.Column(scale=3):
            output_text = gr.Textbox(
                label="📖 Generated Text",
                lines=20,
                placeholder="Your generated text will appear here..."
            )
    
    # Connect the generate button
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
    
    # Add examples section
    gr.Markdown("### 📝 Try these examples:")
    for example in examples:
        with gr.Row():
            gr.Button(example[0]).click(
                fn=lambda p=example[0]: p,
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
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.txt)
[![Stars](https://img.shields.io/github/stars/Mohit-13-4/mini-gpt.svg?style=social)](https://github.com/Mohit-13-4/mini-gpt/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/Mohit-13-4/mini-gpt.svg)](https://github.com/Mohit-13-4/mini-gpt/issues)
[![GitHub forks](https://img.shields.io/github/forks/Mohit-13-4/mini-gpt.svg?style=social)](https://github.com/Mohit-13-4/mini-gpt/network)

<div align="center">
  <h1>🎭 GPT from Scratch: Complete Implementation</h1>
  <p><em>Build your own GPT from scratch - BPE tokenizer, transformer architecture, and trained models</em></p>
</div>




# 🎭 GPT From Scratch — Full Stack LLM Implementation

A **complete, from-scratch implementation of a GPT-style Large Language Model** built using PyTorch — covering **tokenization → architecture → training → inference → evaluation → deployment**.

> ⚡ This project demonstrates deep understanding of **LLM internals, training dynamics, and real-world system design**.



# 🚀 Key Highlights

* ✅ **GPT architecture implemented from scratch** (no HuggingFace models)
* ✅ **Custom BPE tokenizer (4,073 merges)** trained on WikiText-2
* ✅ Trained **multiple models (11M → 32M parameters)**
* ✅ **Advanced decoding**: Top-K, Top-P, temperature, repetition penalty
* ✅ **KV Cache optimization** for faster inference
* ✅ **Mixed precision training + gradient accumulation**
* ✅ **Full evaluation suite + benchmarking system**
* ✅ **Interactive Gradio web app for real-time generation**



# 📊 Model Performance

| Model               | Params    | Val Loss | Perplexity | Dataset          | Training Time |
| ------------------- | --------- | -------- | ---------- | ---------------- | ------------- |
| Shakespeare         | 11M       | 2.53     | 12.55      | Tiny Shakespeare | 14 min        |
| **WikiText (Best)** | **12.6M** | **2.06** | **7.84**   | WikiText-2       | 32 min        |
| Medium              | 32M       | 1.68     | 5.37       | WikiText-103     | 57 min        |

> ⚠️ Larger models showed **overfitting on smaller datasets**, highlighting the importance of **model-data scaling balance**.



# 🧠 Architecture Overview


Text → BPE Tokenizer → Embeddings + Positional Encoding
     → Transformer Blocks (Causal Self-Attention)
     → Feed Forward Network
     → LayerNorm + Residual Connections
     → Output Probabilities


### Model Specs (Best Model)

* Layers: 6
* Heads: 6
* Embedding: 384
* Context Length: 128
* Parameters: 12.6M



# ⚙️ Core Features

## 🧩 Tokenization

* BPE from scratch (4,073 merges)
* Custom vocabulary (5,000 tokens)
* Special tokens + save/load



## 🧠 Transformer Model

* Decoder-only GPT architecture
* Multi-head causal self-attention
* Pre-norm layer normalization
* GELU feed-forward network
* KV cache for efficient inference



## 🏋️ Training System

* Mixed precision (FP16)
* Gradient accumulation
* Cosine LR scheduler + warmup
* AdamW + weight decay
* Gradient clipping
* Checkpointing + resume



## 🎯 Generation

* Temperature scaling
* Top-K / Top-P sampling
* Repetition penalty
* KV cache acceleration



## 📊 Evaluation & Benchmarking

Includes a **custom evaluation suite**:

* Loss & perplexity tracking
* Sampling strategy comparison
* Prompt sensitivity analysis
* Context length testing
* Failure case analysis
* Inference speed benchmarking



# ⚡ Performance Benchmarks

## Inference Speed (RTX 3050, 4GB)

| Method      | Time/Token | Tokens/sec |
| ----------- | ---------- | ---------- |
| No KV Cache | 4.2 ms     | 238        |
| KV Cache    | 4.0 ms     | 252        |

> KV cache improves efficiency (~1.05×) — impact limited due to short context length.



## Sampling Strategy Comparison

| Method        | Diversity | Coherence | Use Case      |
| ------------- | --------- | --------- | ------------- |
| Greedy        | Low       | High      | Deterministic |
| Top-K         | Medium    | Medium    | Balanced      |
| Top-P         | High      | High      | Creative      |
| Top-K + Top-P | High      | High      | Best overall  |



# 🧠 Key Insights (IMPORTANT — Differentiator)

### 1. Model Scaling vs Data

* 12.6M model performs best on WikiText-2
* 32M model begins **overfitting**
  👉 Confirms **model-data scaling tradeoff**



### 2. Tokenization Matters

* BPE significantly improves coherence
* Vocabulary size impacts generation quality



### 3. GPU Constraints

* RTX 3050 (4GB) limits model scaling
* Mixed precision + gradient accumulation are essential



### 4. Inference Optimization

* KV cache improves speed
* Gains limited due to small sequence length



# ⚠️ Limitations

* Small-scale dataset limits generalization
* Short context window (128 tokens)
* No distributed training / multi-GPU scaling
* Not comparable to large-scale LLMs (GPT, LLaMA)



# 📦 Project Structure


mini-gpt/
├── tokenizer.py
├── transformer.py
├── data_loader.py
├── train_*.py
├── inference.py
├── app_*.py
├── benchmark.py
├── checkpoints/
├── runs/




# 🖥️ Demo

### Run Web App


python app_wikitext_fixed.py


### CLI Generation


python inference.py --model wikitext --prompt "The future of AI"




# 🛠 Tech Stack

* PyTorch
* Gradio
* TensorBoard
* HuggingFace Datasets



# 🎯 What This Project Demonstrates

* Deep understanding of **LLM architecture**
* Ability to **train models from scratch**
* Strong **ML engineering + optimization skills**
* Experience with **evaluation, benchmarking, and deployment**



# ⭐ Acknowledgements

* Andrej Karpathy (NanoGPT inspiration)
* PyTorch team
* Project Gutenberg





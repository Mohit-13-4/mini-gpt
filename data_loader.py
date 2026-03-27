"""
data_loader.py - Data loading utilities for GPT training
Supports Tiny Shakespeare and WikiText-103 datasets
"""

import torch
import os
import urllib.request
import zipfile
import gzip
import shutil
from typing import Optional, Tuple

class DataLoader:
    """Flexible data loader for different datasets."""
    
    def __init__(self, dataset_name: str = 'shakespeare', config=None, tokenizer=None):
        """
        Args:
            dataset_name: 'shakespeare' or 'wikitext'
            config: Training configuration
            tokenizer: BPE tokenizer instance
        """
        self.dataset_name = dataset_name
        self.config = config
        self.tokenizer = tokenizer
        
        # Create data directory
        os.makedirs('data', exist_ok=True)
        
        # Load and process dataset
        text = self._load_dataset()
        
        # Tokenize if tokenizer provided
        if tokenizer is not None:
            if len(tokenizer.vocab) == 0:
                print("   Training tokenizer on dataset...")
                tokenizer.train(text, verbose=False)
            
            print("   Tokenizing text...")
            # Use smaller subset for faster training (optional)
            # text = text[:1000000]  # Uncomment to use only first 1M chars
            self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
            print(f"   Created {len(self.tokens):,} tokens")
        
        self.n_tokens = len(self.tokens) if hasattr(self, 'tokens') else len(text)
    
    def _load_dataset(self) -> str:
        """Load dataset based on name."""
        if self.dataset_name == 'shakespeare':
            return self._load_shakespeare()
        elif self.dataset_name == 'wikitext':
            return self._load_wikitext()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_shakespeare(self) -> str:
        """Load Tiny Shakespeare dataset."""
        print("\n📖 Loading Tiny Shakespeare dataset...")
        
        filepath = 'data/tiny_shakespeare.txt'
        if not os.path.exists(filepath):
            print("   Downloading Tiny Shakespeare...")
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            urllib.request.urlretrieve(url, filepath)
            print(f"   Downloaded to {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"   Loaded {len(text):,} characters")
        return text
    
    def _load_wikitext(self) -> str:
        """Load WikiText-103 dataset."""
        print("\n📖 Loading WikiText-103 dataset...")
        
        # Option 1: Use HuggingFace datasets (recommended)
        try:
            from datasets import load_dataset
            print("   Using HuggingFace datasets...")
            
            # CORRECTED: Use 'wikitext-103-raw-v1' instead of 'wikitext-103-raw'
            dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
            
            # Combine all text (filter out empty lines)
            texts = [text for text in dataset['text'] if text.strip()]
            
            # Use first 50k examples for faster training (adjust as needed)
            # For full dataset, remove [:50000]
            texts = texts[:50000]
            text = '\n'.join(texts)
            
            print(f"   Loaded {len(text):,} characters from {len(texts)} examples")
            print(f"   (Using subset for faster training - remove [:50000] for full dataset)")
            
            return text
            
        except ImportError:
            print("   Installing datasets library...")
            os.system("pip install datasets")
            from datasets import load_dataset
            dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
            texts = [text for text in dataset['text'] if text.strip()]
            texts = texts[:50000]
            text = '\n'.join(texts)
            return text
            
        except Exception as e:
            print(f"   HuggingFace failed: {e}")
            print("   Falling back to direct download of smaller version...")
            return self._load_wikitext_fallback()
    
    def _load_wikitext_fallback(self) -> str:
        """Fallback: Download a smaller version of WikiText."""
        print("   Using fallback: WikiText-2 (smaller version)...")
        
        try:
            from datasets import load_dataset
            # Use WikiText-2 which is smaller and always available
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
            texts = [text for text in dataset['text'] if text.strip()]
            text = '\n'.join(texts[:20000])  # Use 20k examples
            
            print(f"   Loaded {len(text):,} characters from WikiText-2")
            return text
            
        except Exception as e:
            print(f"   WikiText-2 also failed: {e}")
            print("   Final fallback: Using Tiny Shakespeare")
            return self._load_shakespeare()
    
    def get_batch(self, split: str = 'train') -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of data for training or validation."""
        # Use 90% for training
        n_train = int(0.9 * self.n_tokens)
        
        if split == 'train':
            # Random starting indices in training portion
            start_indices = torch.randint(0, n_train - self.config.block_size, 
                                         (self.config.batch_size,))
        else:
            # Random starting indices in validation portion
            start_indices = torch.randint(n_train, self.n_tokens - self.config.block_size,
                                         (self.config.batch_size,))
        
        # Create batches
        x = torch.stack([self.tokens[i:i+self.config.block_size] for i in start_indices])
        y = torch.stack([self.tokens[i+1:i+self.config.block_size+1] for i in start_indices])
        
        # Move to GPU
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        
        return x, y
    
    def get_stats(self) -> dict:
        """Return dataset statistics."""
        return {
            'dataset': self.dataset_name,
            'total_tokens': self.n_tokens,
            'vocab_size': len(self.tokenizer) if self.tokenizer else 0,
        }


# Quick test
if __name__ == "__main__":
    from tokenizer import BPETokenizer
    from transformer import GPTConfig
    
    class Config:
        block_size = 128
        batch_size = 64
    
    config = Config()
    tokenizer = BPETokenizer(vocab_size=1000)
    
    # Test Shakespeare
    print("\n" + "="*60)
    print("Testing Shakespeare dataset...")
    print("="*60)
    shakespeare_data = DataLoader('shakespeare', config, tokenizer)
    x, y = shakespeare_data.get_batch('train')
    print(f"Batch shape: {x.shape}")
    
    # Test WikiText
    print("\n" + "="*60)
    print("Testing WikiText-103 dataset...")
    print("="*60)
    wikitext_data = DataLoader('wikitext', config, tokenizer)
    x, y = wikitext_data.get_batch('train')
    print(f"Batch shape: {x.shape}")
    print(f"Stats: {wikitext_data.get_stats()}")
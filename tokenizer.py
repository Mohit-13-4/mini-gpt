"""
tokenizer.py - A complete tokenizer implementation for learning
Author: You
Date: 2026

This file contains two tokenizer implementations:
1. CharTokenizer - Simple character-level (perfect for learning)
2. BPETokenizer - Byte-Pair Encoding (what real GPT uses)
"""

import json
from typing import List, Dict, Optional, Union
from collections import defaultdict
import re

# ============================================================================
# PART 1: CHARACTER-LEVEL TOKENIZER (The Foundation)
# ============================================================================

class CharTokenizer:
    """
    A character-level tokenizer that treats each character as a token.
    
    Think of it as creating a mapping like:
    'a' -> 0
    'b' -> 1
    'c' -> 2
    ...
    """
    
    def __init__(self):
        """Initialize an empty tokenizer."""
        self.vocab = {}          # character -> ID
        self.reverse_vocab = {}  # ID -> character
        self.vocab_size = 0
        print("🆕 CharTokenizer initialized (empty)")
    
    def train(self, text: str) -> None:
        """Build vocabulary from input text."""
        print("\n🔤 Training character tokenizer...")
        
        # Find all unique characters
        unique_chars = set(text)
        print(f"   Found {len(unique_chars)} unique characters")
        
        # Sort for consistent ordering
        sorted_chars = sorted(unique_chars)
        
        # Build vocabulary mapping
        self.vocab = {char: idx for idx, char in enumerate(sorted_chars)}
        self.reverse_vocab = {idx: char for char, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        print(f"   ✅ Vocabulary size: {self.vocab_size}")
    
    def encode(self, text: str) -> List[int]:
        """Convert text string to list of token IDs."""
        tokens = [self.vocab[char] for char in text if char in self.vocab]
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Convert list of token IDs back to text string."""
        chars = [self.reverse_vocab[token] for token in tokens]
        return ''.join(chars)
    
    def save(self, filepath: str) -> None:
        """Save tokenizer vocabulary to a JSON file."""
        data = {
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'type': 'character'
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Tokenizer saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load tokenizer vocabulary from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.vocab_size = data['vocab_size']
        self.reverse_vocab = {int(idx): char for char, idx in self.vocab.items()}
        print(f"📂 Tokenizer loaded from {filepath}")
    
    def __len__(self) -> int:
        return self.vocab_size


# ============================================================================
# PART 2: BYTE-PAIR ENCODING (BPE) TOKENIZER
# ============================================================================

class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer - the algorithm behind GPT, Llama, and most modern LLMs.
    """
    
    def __init__(self, vocab_size: int = 1000):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab_size: Target vocabulary size (including special tokens)
        """
        self.vocab_size = vocab_size
        self.vocab = {}          # token -> id
        self.reverse_vocab = {}   # id -> token
        self.merges = []          # list of (token1, token2) merges in order
        
        # Special tokens - essential for any real tokenizer
        self.special_tokens = {
            '<PAD>': 0,    # Padding token
            '<BOS>': 1,    # Beginning of sequence
            '<EOS>': 2,    # End of sequence
            '<UNK>': 3,    # Unknown token
        }
        
        self.n_special = len(self.special_tokens)
        print(f"🆕 BPETokenizer initialized (target vocab: {vocab_size})")
        print(f"   Special tokens: {self.n_special} reserved")
    
    def _get_stats(self, word_freqs: Dict[str, int]) -> Dict[tuple, int]:
        """Count frequency of adjacent token pairs across all words."""
        pair_freqs = defaultdict(int)
        
        for word, freq in word_freqs.items():
            tokens = word.split()
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freqs[pair] += freq
        
        return pair_freqs
    
    def _merge_pair(self, word_freqs: Dict[str, int], pair: tuple, new_token: str) -> Dict[str, int]:
        """Merge a specific pair of tokens throughout all words."""
        new_word_freqs = {}
        
        for word, freq in word_freqs.items():
            tokens = word.split()
            i = 0
            new_tokens = []
            
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            new_word = ' '.join(new_tokens)
            new_word_freqs[new_word] = freq
        
        return new_word_freqs
    
    def train(self, text: str, verbose: bool = True) -> None:
        """Train BPE tokenizer on input text."""
        print("\n🔤 Training BPE tokenizer...")
        print(f"   Target vocabulary size: {self.vocab_size}")
        
        # Step 1: Initialize base vocabulary with all characters
        chars = sorted(list(set(text)))
        print(f"   Found {len(chars)} unique characters")
        
        # Create initial vocabulary (start after special tokens)
        current_vocab_size = len(chars) + self.n_special
        print(f"   Initial vocabulary size (chars + special): {current_vocab_size}")
        
        # If target vocab size is smaller than initial, use initial size
        if self.vocab_size <= current_vocab_size:
            print(f"   ⚠️ Target vocab size {self.vocab_size} is smaller than initial size {current_vocab_size}")
            print(f"   Using initial vocabulary only (no merges performed)")
            
            # Create vocabulary with just characters and special tokens
            self.vocab = {char: idx + self.n_special for idx, char in enumerate(chars)}
            for token, idx in self.special_tokens.items():
                self.vocab[token] = idx
            
            self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
            self.merges = []  # No merges
            print(f"\n✅ BPE training complete!")
            print(f"   Final vocabulary size: {len(self.vocab)}")
            print(f"   Number of merges performed: {len(self.merges)}")
            return
        
        # Create initial vocabulary
        self.vocab = {char: idx + self.n_special for idx, char in enumerate(chars)}
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
        
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        print(f"   Initial vocabulary: {len(self.vocab)} tokens")
        
        # Step 2: Prepare training data
        # Split into words (keeping whitespace as separate tokens)
        words = re.findall(r'\S+|\s+', text)
        word_freqs = {}
        for word in words:
            word_freqs[word] = word_freqs.get(word, 0) + 1
        
        print(f"   Found {len(word_freqs)} unique word tokens")
        
        # Step 3: Represent words as space-separated characters
        # Only process actual words (non-whitespace)
        char_word_freqs = {}
        for word, freq in word_freqs.items():
            if word.strip():  # Only process non-whitespace words
                char_word = ' '.join(list(word))
                char_word_freqs[char_word] = freq
        
        print(f"   Processing {len(char_word_freqs)} words for BPE training")
        
        # Step 4: Main BPE training loop
        num_merges = self.vocab_size - len(self.vocab)
        print(f"   Will perform up to {num_merges} merges to reach target vocab size")
        
        merge_count = 0
        for i in range(num_merges):
            # Count frequency of every adjacent pair
            pair_freqs = self._get_stats(char_word_freqs)
            
            if not pair_freqs:
                print("   No more pairs to merge! Stopping early.")
                break
            
            # Find the most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            best_freq = pair_freqs[best_pair]
            
            # Create new token by concatenating the pair
            new_token = best_pair[0] + best_pair[1]
            
            # Record this merge
            self.merges.append(best_pair)
            merge_count += 1
            
            # Add new token to vocabulary
            next_id = len(self.vocab)
            self.vocab[new_token] = next_id
            self.reverse_vocab[next_id] = new_token
            
            # Apply this merge to all words
            char_word_freqs = self._merge_pair(char_word_freqs, best_pair, new_token)
            
            if verbose:
                if i < 5 or (i + 1) % 100 == 0:
                    print(f"      Merge #{i+1}: {best_pair} -> '{new_token}' (freq: {best_freq})")
        
        print(f"\n✅ BPE training complete!")
        print(f"   Final vocabulary size: {len(self.vocab)}")
        print(f"   Number of merges performed: {merge_count}")
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs using learned BPE merges."""
        if not self.vocab:
            raise ValueError("Tokenizer hasn't been trained yet! Call train() first.")
        
        words = re.findall(r'\S+|\s+', text)
        tokens = []
        
        for word in words:
        
            if word.isspace():  # If it's a space, newline, etc.
                # Add space token (ID for space character)
                if ' ' in self.vocab:
                    tokens.append(self.vocab[' '])
                continue
    # ... rest of word processing
            if word.strip() and self.merges:  # Process words with merges
                # Start with characters (space-separated format)
                tokenized = ' '.join(list(word))
                
                # Apply each merge in the same order as training
                for pair in self.merges:
                    new_token = pair[0] + pair[1]
                    # Replace all occurrences of the pair with new token
                    # Need to handle cases where tokens might be part of larger tokens
                    pattern = ' '.join(pair)
                    tokenized = tokenized.replace(pattern, new_token)
                
                # Split into final tokens
                final_tokens = tokenized.split()
            else:
                # For whitespace or if no merges, just use characters
                final_tokens = list(word)
            
            # Convert tokens to IDs
            for token in final_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    # Unknown token - use <UNK>
                    tokens.append(self.special_tokens['<UNK>'])
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Convert token IDs back to text."""
        token_strings = []
        for token_id in tokens:
            if token_id in self.reverse_vocab:
                token = self.reverse_vocab[token_id]
                # Skip special tokens for decoding
                if token not in self.special_tokens:
                    token_strings.append(token)
            else:
                token_strings.append('<UNK>')
        
        return ''.join(token_strings)
    
    def save(self, filepath: str) -> None:
        """Save trained tokenizer to file."""
        data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'special_tokens': self.special_tokens,
            'vocab_size': self.vocab_size,
            'type': 'bpe'
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 BPE Tokenizer saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load trained tokenizer from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.merges = [tuple(pair) for pair in data['merges']]
        self.special_tokens = data['special_tokens']
        self.vocab_size = data['vocab_size']
        
        self.reverse_vocab = {int(idx): token for token, idx in self.vocab.items()}
        self.n_special = len(self.special_tokens)
        
        print(f"📂 BPE Tokenizer loaded from {filepath}")
        print(f"   Vocabulary size: {len(self.vocab)}")
        print(f"   Number of merges: {len(self.merges)}")
    
    def __len__(self) -> int:
        return len(self.vocab)


# ============================================================================
# TEST FUNCTIONS (defined outside classes, with correct indentation)
# ============================================================================

def test_char_tokenizer():
    """Test function to verify character tokenizer works correctly."""
    print("\n" + "="*60)
    print("🧪 TESTING CHARACTER TOKENIZER")
    print("="*60)
    
    sample_text = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You"""
    
    print("\n📝 Step 1: Creating tokenizer...")
    tokenizer = CharTokenizer()
    
    print("\n📝 Step 2: Training on sample text...")
    tokenizer.train(sample_text)
    
    print("\n📝 Step 3: Testing encode/decode...")
    test_text = "hello world"
    print(f"\nOriginal text: '{test_text}'")
    
    encoded = tokenizer.encode(test_text)
    print(f"Encoded: {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: '{decoded}'")
    
    assert test_text == decoded, "❌ Encode/decode failed!"
    print("✅ Encode/decode successful!")
    
    print("\n📝 Step 4: Testing save/load...")
    tokenizer.save("test_char_tokenizer.json")
    
    new_tokenizer = CharTokenizer()
    new_tokenizer.load("test_char_tokenizer.json")
    
    test_encoded = new_tokenizer.encode("hello")
    print(f"Loaded tokenizer encodes 'hello' as: {test_encoded}")
    
    print("\n" + "="*60)
    print("✅ CHARACTER TOKENIZER TESTS PASSED!")
    print("="*60)


def test_bpe_tokenizer():
    """Test function for BPE tokenizer."""
    print("\n" + "="*60)
    print("🧪 TESTING BPE TOKENIZER")
    print("="*60)
    
    sample_text = """
low low low low low low low low low low low low low low low
lower lower lower lower lower lower lower lower lower lower
lowest lowest lowest lowest lowest lowest lowest lowest
"""
    
    print("\n📝 Step 1: Creating BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=50)
    
    print("\n📝 Step 2: Training on sample text...")
    tokenizer.train(sample_text, verbose=True)
    
    print("\n📝 Step 3: Testing encode/decode...")
    test_words = ["low", "lower", "lowest", "new_word"]
    
    for word in test_words:
        encoded = tokenizer.encode(word)
        decoded = tokenizer.decode(encoded)
        print(f"\n   '{word}':")
        print(f"      Encoded: {encoded}")
        print(f"      Decoded: '{decoded}'")
        print(f"      Matches: {word == decoded}")
    
    print("\n📝 Step 4: Testing save/load...")
    tokenizer.save("test_bpe_tokenizer.json")
    
    new_tokenizer = BPETokenizer()
    new_tokenizer.load("test_bpe_tokenizer.json")
    
    test_encoded = new_tokenizer.encode("low")
    print(f"\nLoaded tokenizer encodes 'low' as: {test_encoded}")
    
    print("\n" + "="*60)
    print("✅ BPE TESTS COMPLETE!")
    print("="*60)


def compare_tokenizers():
    """Compare character vs BPE tokenizers on real text."""
    print("\n" + "="*60)
    print("📊 COMPARING TOKENIZERS")
    print("="*60)
    
    text = "The quick brown fox jumps over the lazy dog."
    
    # Character tokenizer
    print("\n🔤 Character Tokenizer:")
    char_tokenizer = CharTokenizer()
    char_tokenizer.train(text)
    char_tokens = char_tokenizer.encode(text)
    print(f"   First 20 tokens: {char_tokens[:20]}...")
    print(f"   Length: {len(char_tokens)} tokens")
    print(f"   Vocab size: {len(char_tokenizer)}")
    
    # BPE tokenizer - use larger vocab size to allow merges
    print("\n🔠 BPE Tokenizer:")
    # The text has about 26 unique characters (letters + space + punctuation)
    # So we need vocab_size > 26 + special tokens (4) = 30 to see merges
    bpe_tokenizer = BPETokenizer(vocab_size=50)  # Increase to allow merges
    bpe_tokenizer.train(text, verbose=True)
    bpe_tokens = bpe_tokenizer.encode(text)
    print(f"   Tokens: {bpe_tokens}")
    print(f"   Length: {len(bpe_tokens)} tokens")
    print(f"   Vocab size: {len(bpe_tokenizer)}")
    
    # Show BPE merges learned
    if bpe_tokenizer.merges:
        print("\n   BPE Merges learned:")
        for i, merge in enumerate(bpe_tokenizer.merges[:5]):
            print(f"      Merge {i+1}: {merge[0]} + {merge[1]} -> {merge[0]+merge[1]}")
    else:
        print("\n   No merges learned (vocab size too small)")
    
    if len(bpe_tokens) > 0:
        compression_ratio = len(char_tokens) / len(bpe_tokens)
        print(f"\n📈 Compression ratio: {compression_ratio:.2f}x")
        print(f"   (BPE uses {len(bpe_tokens)} tokens vs {len(char_tokens)} for characters)")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 TOKENIZER LEARNING MODULE")
    print("="*60)
    print("\nThis file contains two tokenizer implementations:")
    print("1. CharTokenizer - Simple character-level (foundation)")
    print("2. BPETokenizer - Byte-Pair Encoding (used by GPT)")
    
    # Run character tokenizer tests
    test_char_tokenizer()
    
    # Run BPE tokenizer tests
    test_bpe_tokenizer()
    
    # Compare them
    compare_tokenizers()
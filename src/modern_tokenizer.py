#!/usr/bin/env python3
"""
Modern BPE Tokenizer Implementation
Based on state-of-the-art GPT-4 style tokenization using HuggingFace tokenizers library
"""

import os
import sys
from typing import List, Optional
import time

try:
    from tokenizers import Tokenizer, pre_tokenizers, decoders, Regex
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    print("Warning: 'tokenizers' library not installed. Install with: pip install tokenizers")


class ModernBPETokenizer:
    """
    Modern BPE tokenizer using GPT-4 style preprocessing
    Superior to both simple character-level and basic BPE approaches
    """
    
    def __init__(self, vocab_size: int = 8192):
        if not TOKENIZERS_AVAILABLE:
            raise ImportError("tokenizers library required. Install with: pip install tokenizers")
            
        self.vocab_size = vocab_size
        self.tokenizer = None
        self._setup_tokenizer()
    
    def _setup_tokenizer(self):
        """Configure the tokenizer with BPE and byte-level processing"""
        
        # Create BPE tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
        
        # Use byte-level pre-tokenization (simpler and more reliable)
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        # Byte-level decoder
        self.tokenizer.decoder = decoders.ByteLevel()
    
    def train_from_file(self, text_file: str, show_progress: bool = True):
        """Train the tokenizer on a text file"""
        if not os.path.exists(text_file):
            raise FileNotFoundError(f"Training file not found: {text_file}")
        
        print(f"Training modern BPE tokenizer on: {text_file}")
        print(f"Target vocabulary size: {self.vocab_size}")
        
        # Configure trainer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            show_progress=show_progress,
            min_frequency=0,  # No minimum frequency requirement
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=[
                "<|endoftext|>",
                "<|pad|>",
                "<|unk|>",
            ]
        )
        
        # Train on the file
        start_time = time.time()
        self.tokenizer.train([text_file], trainer)
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.2f}s")
        print(f"Vocabulary size: {self.tokenizer.get_vocab_size()}")
        
        # Test tokenization
        self._test_tokenization()
    
    def train_from_iterator(self, text_iterator, show_progress: bool = True):
        """Train from an iterator of text strings"""
        print(f"Training modern BPE tokenizer from iterator")
        print(f"Target vocabulary size: {self.vocab_size}")
        
        # Configure trainer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            show_progress=show_progress,
            min_frequency=0,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=[
                "<|endoftext|>",
                "<|pad|>", 
                "<|unk|>",
            ]
        )
        
        # Train from iterator
        start_time = time.time()
        self.tokenizer.train_from_iterator(text_iterator, trainer)
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.2f}s")
        print(f"Vocabulary size: {self.tokenizer.get_vocab_size()}")
        
        self._test_tokenization()
    
    def _test_tokenization(self):
        """Test tokenization with sample text"""
        test_texts = [
            "Hello, how are you today?",
            "The quick brown fox jumps over the lazy dog.",
            "I'm learning about AI and machine learning!",
            "Human: What's 2+2?\nAssistant: 2+2 equals 4.",
        ]
        
        print("\nTokenization test:")
        for text in test_texts:
            tokens = self.encode(text)
            decoded = self.decode(tokens)
            compression = len(text) / len(tokens)
            print(f"  Text: {text}")
            print(f"  Tokens: {len(tokens)} (compression: {compression:.2f}x)")
            print(f"  Roundtrip: {'OK' if decoded == text else 'FAIL'}")
            print()
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not trained yet")
        
        encoding = self.tokenizer.encode(text)
        return encoding.ids
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not trained yet")
        
        return self.tokenizer.decode(tokens)
    
    def get_vocab_size(self) -> int:
        """Get actual vocabulary size"""
        if self.tokenizer is None:
            return 0
        return self.tokenizer.get_vocab_size()
    
    def save(self, filepath: str):
        """Save tokenizer to file"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not trained yet")
        
        self.tokenizer.save(filepath)
        print(f"Tokenizer saved to: {filepath}")
    
    def load(self, filepath: str):
        """Load tokenizer from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Tokenizer file not found: {filepath}")
        
        self.tokenizer = Tokenizer.from_file(filepath)
        self.vocab_size = self.tokenizer.get_vocab_size()
        print(f"Tokenizer loaded from: {filepath}")
        print(f"Vocabulary size: {self.vocab_size}")
    
    def get_vocab(self):
        """Get vocabulary dictionary"""
        if self.tokenizer is None:
            return {}
        return self.tokenizer.get_vocab()


def create_modern_tokenizer_from_data(data_file: str, vocab_size: int = 8192, save_path: str = None):
    """
    Factory function to create and train a modern tokenizer
    """
    print("Creating Modern BPE Tokenizer")
    print("=" * 40)
    
    tokenizer = ModernBPETokenizer(vocab_size=vocab_size)
    tokenizer.train_from_file(data_file)
    
    if save_path:
        tokenizer.save(save_path)
    
    return tokenizer


def benchmark_tokenizer(tokenizer, test_file: str = None):
    """Benchmark tokenizer performance"""
    if test_file and os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            test_text = f.read()[:10000]  # First 10k chars
    else:
        test_text = """
        Human: Hello, how are you today? I hope you're having a wonderful day!
        Assistant: Hello! I'm doing well, thank you for asking. How can I assist you today?
        Human: Can you tell me about artificial intelligence?
        Assistant: Artificial intelligence (AI) refers to the simulation of human intelligence in machines.
        """ * 10
    
    print(f"\nBenchmarking tokenizer performance...")
    print(f"Test text length: {len(test_text)} characters")
    
    # Encoding benchmark
    start_time = time.time()
    tokens = tokenizer.encode(test_text)
    encode_time = time.time() - start_time
    
    # Decoding benchmark
    start_time = time.time()
    decoded = tokenizer.decode(tokens)
    decode_time = time.time() - start_time
    
    print(f"Tokens: {len(tokens)}")
    print(f"Compression ratio: {len(test_text)/len(tokens):.2f}x")
    print(f"Encoding time: {encode_time*1000:.2f}ms")
    print(f"Decoding time: {decode_time*1000:.2f}ms")
    print(f"Roundtrip success: {'OK' if decoded == test_text else 'FAIL'}")


if __name__ == "__main__":
    if not TOKENIZERS_AVAILABLE:
        print("Please install tokenizers: pip install tokenizers")
        sys.exit(1)
    
    # Example usage
    data_file = "../datasets/large_conversation_training.txt"
    
    if os.path.exists(data_file):
        print("Training modern BPE tokenizer...")
        tokenizer = create_modern_tokenizer_from_data(
            data_file=data_file,
            vocab_size=8192,
            save_path="modern_tokenizer.json"
        )
        
        benchmark_tokenizer(tokenizer, data_file)
    else:
        print(f"Training data not found: {data_file}")
        print("Available dataset files:")
        datasets_dir = "../datasets"
        if os.path.exists(datasets_dir):
            for f in os.listdir(datasets_dir):
                if f.endswith('.txt'):
                    size = os.path.getsize(os.path.join(datasets_dir, f))
                    print(f"  {f} ({size:,} bytes)")
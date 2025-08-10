#!/usr/bin/env python3
"""
Training script for reasoning-capable GPT model
Implements Chain-of-Thought training with larger architecture
"""

import torch
import torch.nn as nn
import numpy as np
import time
import pickle
import os
import math
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.pico_gpt import GPT, GPTConfig
from src.modern_tokenizer import ModernBPETokenizer

def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters, batch_size, block_size, device):
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def train():
    print("Reasoning-Capable GPT Model Training")
    print("=" * 50)
    
    # Training hyperparameters
    batch_size = 8  # Smaller batch due to larger model
    learning_rate = 1e-4  # Lower LR for larger model
    max_iters = 5000  # More training for reasoning
    eval_interval = 500
    eval_iters = 100
    warmup_iters = 200
    lr_decay_iters = max_iters
    min_lr = learning_rate / 10
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Check GPU memory
    if device == 'cuda':
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_memory:.1f} GB")
    
    # Load reasoning dataset
    reasoning_file = os.path.join('datasets', 'reasoning_training_data.txt')
    if not os.path.exists(reasoning_file):
        print(f"Reasoning dataset not found: {reasoning_file}")
        print("Please add the file at 'datasets/reasoning_training_data.txt' or update the path in this script.")
        return
    
    print(f"Loading reasoning data from: {reasoning_file}")
    with open(reasoning_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Reasoning data loaded: {len(text):,} characters")
    
    # Load or create modern tokenizer
    tokenizer_path = 'models/modern_tokenizer.json'
    if os.path.exists(tokenizer_path):
        print("Loading pre-trained modern tokenizer...")
        tokenizer = ModernBPETokenizer()
        tokenizer.load(tokenizer_path)
    else:
        print("Training new tokenizer on reasoning data...")
        tokenizer = ModernBPETokenizer(vocab_size=4096)  # Smaller vocab for reasoning
        tokenizer.train_from_file(reasoning_file, show_progress=True)
        tokenizer.save(tokenizer_path)
    
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer vocabulary: {actual_vocab_size:,} tokens")
    
    # Tokenize reasoning data
    print("Tokenizing reasoning data...")
    start_time = time.time()
    tokens = tokenizer.encode(text)
    data = torch.tensor(tokens, dtype=torch.long)
    tokenize_time = time.time() - start_time
    
    print(f"Tokenized in {tokenize_time:.2f}s")
    print(f"Reasoning tokens: {len(data):,}")
    print(f"Compression ratio: {len(text)/len(data):.2f}x")
    
    # Split data
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Train tokens: {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")
    
    # Reasoning-optimized model configuration
    config = GPTConfig()
    config.block_size = 512  # Longer context for multi-step reasoning
    config.vocab_size = actual_vocab_size
    config.n_layer = 12  # More layers for reasoning capability
    config.n_head = 12   # More attention heads
    config.n_embd = 768  # Larger embedding dimension
    config.dropout = 0.1
    
    print(f"\nReasoning Model Configuration:")
    print(f"  Layers: {config.n_layer}")
    print(f"  Heads: {config.n_head}")
    print(f"  Embedding dim: {config.n_embd}")
    print(f"  Context length: {config.block_size}")
    print(f"  Vocabulary size: {config.vocab_size}")
    
    # Create reasoning model
    model = GPT(config)
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Estimated size: {n_params * 4 / 1024 / 1024:.1f} MB")
    
    # Check if model fits in memory
    if device == 'cuda':
        model_memory_gb = n_params * 4 / 1024**3
        if model_memory_gb > gpu_memory * 0.8:
            print(f"Warning: Model ({model_memory_gb:.1f}GB) may not fit in GPU memory ({gpu_memory:.1f}GB)")
            print("Consider reducing model size or using CPU")
    
    # Optimizer with weight decay for large model
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Training loop
    print("\nStarting reasoning model training...")
    print("-" * 50)
    
    start_time = time.time()
    best_val_loss = float('inf')
    
    for iter_num in range(max_iters):
        # Learning rate schedule
        lr = get_lr(iter_num, warmup_iters, learning_rate, lr_decay_iters, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluation
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, eval_iters, batch_size, config.block_size, device)
            elapsed = time.time() - start_time
            print(f"Step {iter_num:4d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e}, time {elapsed:.1f}s")
            
            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'tokenizer': tokenizer,
                    'tokenizer_path': tokenizer_path,
                    'train_loss': losses['train'],
                    'val_loss': losses['val'],
                    'iter_num': iter_num,
                    'reasoning_capable': True,
                    'training_type': 'chain_of_thought'
                }
                
                model_path = 'models/pico_gpt_reasoning.pt'
                os.makedirs('models', exist_ok=True)
                torch.save(checkpoint, model_path)
                print(f"Saved best reasoning model: {model_path}")
        
        # Forward pass
        try:
            X, Y = get_batch(train_data, batch_size, config.block_size, device)
            logits, loss = model(X, Y)
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # Gradient clipping for stable training
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU out of memory at iteration {iter_num}")
                print("Try reducing batch_size or model size")
                torch.cuda.empty_cache()
                break
            else:
                raise e
    
    total_time = time.time() - start_time
    print(f"\nReasoning model training completed in {total_time:.1f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved: models/pico_gpt_reasoning.pt")
    
    # Test reasoning capabilities
    print("\nTesting reasoning capabilities...")
    model.eval()
    
    reasoning_tests = [
        "Human: What is 25 × 17?\nAssistant:",
        "Human: If I have 12 apples and eat 3, then buy 8 more, how many do I have?\nAssistant:",
        "Human: What comes next: 2, 4, 8, 16?\nAssistant:",
    ]
    
    for i, test_prompt in enumerate(reasoning_tests, 1):
        print(f"\nReasoning Test {i}:")
        print(f"Prompt: {test_prompt}")
        
        tokens = tokenizer.encode(test_prompt)
        context = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            generated = model.generate(context, max_new_tokens=100, temperature=0.7, top_k=30)
        
        result = tokenizer.decode(generated[0].tolist())
        print(f"Response: {result[len(test_prompt):]}")
    
    # Model summary
    print(f"\n" + "=" * 50)
    print("REASONING MODEL SUMMARY")
    print("=" * 50)
    print(f"Model type: Chain-of-Thought Reasoning GPT")
    print(f"Parameters: {n_params:,}")
    print(f"Context length: {config.block_size}")
    print(f"Vocabulary: {actual_vocab_size:,}")
    print(f"Training time: {total_time:.1f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Training examples: Mathematical reasoning, logical deduction, problem solving")

if __name__ == "__main__":
    train()
#!/usr/bin/env python3
"""
Clean up datasets by removing duplicates and keeping only useful ones
"""

import os
import shutil
from datetime import datetime

def cleanup_datasets():
    """Clean up the datasets directory"""
    datasets_dir = 'datasets'
    if not os.path.exists(datasets_dir):
        print("Datasets directory not found")
        return
    
    # Files to keep (based on analysis)
    keep_files = [
        'large_conversation_training.txt',    # Primary conversation data (990KB)
        'reasoning_training_data.txt',        # Reasoning training (39KB) 
        'simple_conversation_data.txt',       # Clean conversation data (154KB)
        'combined_literature.txt'             # Literature dataset (7.6MB)
    ]
    
    # Files to remove
    remove_files = [
        'large_conversation.txt',
        'conversation_training.txt',
        'conversation_training.txt.backup',  # Also remove backup
        'combined_enhanced_data.txt',
        'conversation_data.txt', 
        'enhanced_conversations.txt',
        'smart_reasoning_data.txt',
        'comprehensive_conversations.txt',
        'clean_conversation_data.txt',
        'simple_conversation.txt',
        'smart_conversation_data.txt'
    ]
    
    print("DATASET CLEANUP")
    print("=" * 40)
    
    # Create backup directory
    backup_dir = f"datasets_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        print(f"Created backup directory: {backup_dir}")
    
    # Move files to backup before removing
    removed_count = 0
    saved_space = 0
    
    for filename in remove_files:
        filepath = os.path.join(datasets_dir, filename)
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            backup_path = os.path.join(backup_dir, filename)
            
            # Move to backup
            shutil.move(filepath, backup_path)
            
            removed_count += 1
            saved_space += file_size
            print(f"[OK] Moved to backup: {filename} ({file_size} bytes)")
    
    print(f"\nKEPT FILES:")
    total_kept_size = 0
    for filename in keep_files:
        filepath = os.path.join(datasets_dir, filename)
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            total_kept_size += file_size
            print(f"[KEEP] Kept: {filename} ({file_size:,} bytes)")
        else:
            print(f"[MISSING] Missing: {filename}")
    
    print(f"\nCLEANUP SUMMARY:")
    print(f"Files removed: {removed_count}")
    print(f"Files kept: {len([f for f in keep_files if os.path.exists(os.path.join(datasets_dir, f))])}")
    print(f"Space saved: {saved_space:,} bytes ({saved_space/1024:.1f} KB)")
    print(f"Total kept: {total_kept_size:,} bytes ({total_kept_size/1024:.1f} KB)")
    print(f"Backup created: {backup_dir}")
    
    return keep_files

def create_dataset_info():
    """Create information file about the cleaned datasets"""
    info_content = """# PICO-GPT DATASETS

## Active Datasets (Cleaned)

### 1. large_conversation_training.txt (990 KB)
**Purpose**: Primary conversation training data
**Content**: 6,641 Human-Assistant conversation pairs
**Use**: Main training for conversation models
**Format**: Human/Assistant dialogue format

### 2. reasoning_training_data.txt (39 KB) 
**Purpose**: Mathematical and logical reasoning training
**Content**: 159 step-by-step reasoning examples
**Use**: Training reasoning-capable models
**Format**: Chain-of-thought problem solving

### 3. simple_conversation_data.txt (154 KB)
**Purpose**: Clean, high-quality conversation data  
**Content**: 1,900 conversation pairs
**Use**: Fine-tuning and testing conversation models
**Format**: Human/Assistant dialogue format

### 4. combined_literature.txt (7.6 MB)
**Purpose**: General language modeling and knowledge
**Content**: Literature and text corpus
**Use**: Pre-training and language understanding
**Format**: Plain text

## Usage Recommendations

### For Conversation Models:
- Primary: `large_conversation_training.txt`
- Testing: `simple_conversation_data.txt`

### For Reasoning Models: 
- Primary: `reasoning_training_data.txt`
- Background: `large_conversation_training.txt`

### For General Language Models:
- Primary: `combined_literature.txt`
- Conversation: `large_conversation_training.txt`

## Training Scripts Updated:
- `training/train_conversation.py` → uses large_conversation_training.txt
- `training/train_reasoning_model.py` → uses reasoning_training_data.txt
- `training/train_modern_tokenizer.py` → uses large_conversation_training.txt

## Removed Files (Backed Up):
- Duplicate conversation datasets
- Small test datasets
- Redundant files
- Backup files

Total space saved: ~225 KB
Backup location: datasets_backup_[timestamp]/
"""
    
    with open('datasets/README.md', 'w', encoding='utf-8') as f:
        f.write(info_content)
    
    print("[OK] Created datasets/README.md with dataset information")

if __name__ == "__main__":
    keep_files = cleanup_datasets()
    create_dataset_info()
    
    print(f"\n" + "="*50)
    print("DATASET CLEANUP COMPLETE!")
    print("="*50)
    print("Your datasets directory now contains only:")
    for f in keep_files:
        if os.path.exists(os.path.join('datasets', f)):
            print(f"  [OK] {f}")
    print(f"\nRemoved files are backed up in datasets_backup_[timestamp]/")
    print(f"See datasets/README.md for detailed information.")
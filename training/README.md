# PICO-GPT TRAINING SCRIPTS

## Active Training Scripts

### 1. **train_conversation.py** - Conversation Model Training
- **Purpose**: Train conversational AI models
- **Dataset**: `datasets/large_conversation_training.txt` (990KB)
- **Features**: Large architecture, optimized for dialogue
- **Usage**: `python training/train_conversation.py`

### 2. **train_reasoning_model.py** - Reasoning Model Training  
- **Purpose**: Train reasoning-capable models
- **Dataset**: `datasets/reasoning_training_data.txt` (39KB)
- **Features**: Chain-of-thought training, larger context
- **Usage**: `python training/train_reasoning_model.py`

## Removed Scripts (Backed Up)
- `train_conversation_hf.py` - HuggingFace version
- `train_modern_tokenizer.py` - Modern tokenizer experiment
- `train_reasoning_demo.py` - Smaller demo version

Backup location: `training_backup_20250809/`

## Quick Start
```bash
# Train conversation model
python training/train_conversation.py

# Train reasoning model  
python training/train_reasoning_model.py
```
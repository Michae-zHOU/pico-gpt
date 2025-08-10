#!/usr/bin/env python3
"""
Interactive Model Capability Verification
Test your models with different types of questions to verify their capabilities
"""

import torch
import sys
import os
import time
import re

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.pico_gpt import GPT
from src.modern_tokenizer import ModernBPETokenizer

class ModelTester:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model()
        
    def load_model(self):
        """Load the model for testing"""
        print(f"Loading model: {os.path.basename(self.model_path)}")
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            self.config = checkpoint['config']
            self.tokenizer = checkpoint['tokenizer']
            
            self.model = GPT(self.config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"✓ Model loaded: {n_params:,} parameters")
            print(f"✓ Context length: {self.config.block_size}")
            print(f"✓ Vocabulary: {self.config.vocab_size:,}")
            
            # Check if it's a reasoning model
            self.is_reasoning = checkpoint.get('reasoning_capable', False)
            if self.is_reasoning:
                print("✓ Reasoning-capable model detected")
            print()
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def ask_question(self, question, max_tokens=150, temperature=0.7, show_full=True):
        """Ask the model a question and get response"""
        # Format as conversation
        if not question.startswith("Human:"):
            prompt = f"Human: {question}\nAssistant:"
        else:
            prompt = f"{question}\nAssistant:"
        
        try:
            # Tokenize
            tokens = self.tokenizer.encode(prompt)
            if len(tokens) > self.config.block_size - max_tokens:
                print(f"⚠ Warning: Question too long ({len(tokens)} tokens)")
                tokens = tokens[-(self.config.block_size - max_tokens):]
            
            context = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
            
            # Generate response
            start_time = time.time()
            with torch.no_grad():
                generated = self.model.generate(
                    context, 
                    max_new_tokens=max_tokens, 
                    temperature=temperature, 
                    top_k=20
                )
            generation_time = time.time() - start_time
            
            # Decode result
            full_response = self.tokenizer.decode(generated[0].tolist())
            response = full_response[len(prompt):].strip()
            
            if show_full:
                print(f"Q: {question}")
                print(f"A: {response}")
                print(f"⏱ Generated in {generation_time*1000:.0f}ms ({len(response.split())} words)")
                print()
            
            return response, generation_time
            
        except Exception as e:
            error_msg = f"Error generating response: {e}"
            if show_full:
                print(f"Q: {question}")
                print(f"A: {error_msg}")
                print()
            return error_msg, 0

def run_basic_tests(tester):
    """Run basic capability tests"""
    print("=" * 60)
    print("BASIC CAPABILITY TESTS")
    print("=" * 60)
    
    basic_questions = [
        "Hello, how are you?",
        "What is your name?",
        "Can you help me with something?",
        "Tell me about yourself",
        "What can you do?"
    ]
    
    print("Testing basic conversation...")
    for q in basic_questions:
        tester.ask_question(q, max_tokens=100)

def run_math_tests(tester):
    """Run mathematical reasoning tests"""
    print("=" * 60)
    print("MATHEMATICAL REASONING TESTS")
    print("=" * 60)
    
    math_questions = [
        "What is 15 + 27?",
        "What is 23 × 17?",
        "What is 144 ÷ 12?",
        "If I have 25 apples and eat 8, how many do I have left?",
        "A rectangle is 6 meters long and 4 meters wide. What is its area?",
        "What comes next in the sequence: 2, 4, 6, 8, ?",
        "What comes next in the sequence: 1, 4, 9, 16, ?",
    ]
    
    print("Testing mathematical reasoning...")
    correct_answers = [42, 391, 12, 17, 24, 10, 25]
    
    for i, q in enumerate(math_questions):
        response, _ = tester.ask_question(q, max_tokens=200)
        
        # Try to extract numerical answer
        numbers = re.findall(r'\b\d+\b', response)
        if numbers and i < len(correct_answers):
            predicted = int(numbers[-1])
            expected = correct_answers[i]
            if predicted == expected:
                print(f"✓ Correct answer: {expected}")
            else:
                print(f"✗ Expected {expected}, got {predicted}")
        print()

def run_reasoning_tests(tester):
    """Run logical reasoning tests"""
    print("=" * 60)
    print("LOGICAL REASONING TESTS")
    print("=" * 60)
    
    reasoning_questions = [
        "If all birds can fly, and a robin is a bird, can a robin fly?",
        "If it's raining and rain makes things wet, is the ground wet?",
        "All roses are flowers. All flowers need water. Do roses need water?",
        "I'm thinking of a number. It's even, greater than 10, and less than 20. It's divisible by 4. What number am I thinking of?",
        "Tom is taller than Jerry. Jerry is taller than Mike. Who is the tallest?",
    ]
    
    print("Testing logical reasoning...")
    for q in reasoning_questions:
        tester.ask_question(q, max_tokens=200)

def run_creativity_tests(tester):
    """Run creative capability tests"""
    print("=" * 60)
    print("CREATIVE CAPABILITY TESTS")
    print("=" * 60)
    
    creative_questions = [
        "Write a short poem about the ocean",
        "Tell me a joke",
        "Create a short story about a robot",
        "Describe a perfect day",
        "What would you do if you could fly?",
    ]
    
    print("Testing creativity...")
    for q in creative_questions:
        tester.ask_question(q, max_tokens=200)

def run_knowledge_tests(tester):
    """Run general knowledge tests"""
    print("=" * 60)
    print("KNOWLEDGE & COMPREHENSION TESTS")
    print("=" * 60)
    
    knowledge_questions = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is photosynthesis?",
        "Name three planets in our solar system",
        "What is the difference between HTML and CSS?",
        "Explain what gravity is",
    ]
    
    print("Testing general knowledge...")
    for q in knowledge_questions:
        tester.ask_question(q, max_tokens=150)

def run_problem_solving_tests(tester):
    """Run problem-solving tests"""
    print("=" * 60)
    print("PROBLEM SOLVING TESTS")
    print("=" * 60)
    
    problem_questions = [
        "How would you organize a birthday party?",
        "You have a 3-gallon jug and a 5-gallon jug. How can you measure exactly 4 gallons?",
        "If you were stranded on an island, what are the first three things you would do?",
        "How would you explain the internet to someone from 100 years ago?",
        "A farmer needs to cross a river with a fox, chicken, and bag of corn. The boat only holds the farmer and one item. How does he get everything across?",
    ]
    
    print("Testing problem solving...")
    for q in problem_questions:
        tester.ask_question(q, max_tokens=300)

def run_interactive_mode(tester):
    """Run interactive questioning mode"""
    print("=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Ask your own questions! Type 'quit' to exit.")
    print("Commands:")
    print("  /temp <value>  - Set temperature (0.1-2.0)")
    print("  /tokens <n>    - Set max tokens (50-500)")
    print("  /info          - Show model info")
    print("  /help          - Show this help")
    print()
    
    temperature = 0.7
    max_tokens = 150
    
    while True:
        try:
            question = input("You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif question == '/help':
                print("Commands: /temp <value>, /tokens <n>, /info, /help, quit")
                continue
            elif question == '/info':
                n_params = sum(p.numel() for p in tester.model.parameters())
                print(f"Model: {os.path.basename(tester.model_path)}")
                print(f"Parameters: {n_params:,}")
                print(f"Context: {tester.config.block_size}")
                print(f"Vocabulary: {tester.config.vocab_size:,}")
                print(f"Temperature: {temperature}")
                print(f"Max tokens: {max_tokens}")
                print(f"Reasoning capable: {tester.is_reasoning}")
                continue
            elif question.startswith('/temp '):
                try:
                    temp_val = float(question.split()[1])
                    if 0.1 <= temp_val <= 2.0:
                        temperature = temp_val
                        print(f"Temperature set to {temperature}")
                    else:
                        print("Temperature must be between 0.1 and 2.0")
                except:
                    print("Usage: /temp <value> (e.g., /temp 0.8)")
                continue
            elif question.startswith('/tokens '):
                try:
                    token_val = int(question.split()[1])
                    if 50 <= token_val <= 500:
                        max_tokens = token_val
                        print(f"Max tokens set to {max_tokens}")
                    else:
                        print("Max tokens must be between 50 and 500")
                except:
                    print("Usage: /tokens <number> (e.g., /tokens 200)")
                continue
            elif not question:
                continue
            
            # Ask the question
            print("Assistant: ", end="", flush=True)
            response, gen_time = tester.ask_question(
                question, 
                max_tokens=max_tokens, 
                temperature=temperature, 
                show_full=False
            )
            print(response)
            print(f"⏱ {gen_time*1000:.0f}ms")
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main verification program"""
    print("🔍 MODEL CAPABILITY VERIFICATION TOOL")
    print("=" * 60)
    
    # Find available models
    models_dir = "models"
    available_models = []
    
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith('.pt'):
                available_models.append(f)
    
    if not available_models:
        print("❌ No trained models found in models/ directory")
        print("Train a model first!")
        return
    
    # Show available models
    print("Available models:")
    for i, model in enumerate(available_models, 1):
        size = os.path.getsize(os.path.join(models_dir, model)) / 1024 / 1024
        print(f"{i}. {model} ({size:.1f} MB)")
    
    # Select model
    try:
        choice = input(f"\nSelect model (1-{len(available_models)}) or press Enter for reasoning model: ").strip()
        
        if not choice:
            # Auto-select reasoning model
            reasoning_models = [m for m in available_models if 'reasoning' in m.lower()]
            if reasoning_models:
                selected_model = reasoning_models[0]
                print(f"Auto-selected: {selected_model}")
            else:
                selected_model = available_models[0]
                print(f"No reasoning model found. Using: {selected_model}")
        else:
            idx = int(choice) - 1
            if 0 <= idx < len(available_models):
                selected_model = available_models[idx]
            else:
                print("Invalid selection")
                return
    except (ValueError, KeyboardInterrupt):
        print("Invalid input or cancelled")
        return
    
    # Load and test model
    model_path = os.path.join(models_dir, selected_model)
    tester = ModelTester(model_path)
    
    # Show test menu
    while True:
        print("=" * 60)
        print("MODEL VERIFICATION MENU")
        print("=" * 60)
        print("1. Basic Conversation Tests")
        print("2. Mathematical Reasoning Tests")
        print("3. Logical Reasoning Tests")
        print("4. Creative Capability Tests")
        print("5. Knowledge & Comprehension Tests")
        print("6. Problem Solving Tests")
        print("7. Interactive Mode (Ask Custom Questions)")
        print("8. Run All Tests")
        print("9. Switch Model")
        print("0. Exit")
        
        try:
            choice = input("\nSelect test (0-9): ").strip()
            
            if choice == '0':
                print("Goodbye!")
                break
            elif choice == '1':
                run_basic_tests(tester)
            elif choice == '2':
                run_math_tests(tester)
            elif choice == '3':
                run_reasoning_tests(tester)
            elif choice == '4':
                run_creativity_tests(tester)
            elif choice == '5':
                run_knowledge_tests(tester)
            elif choice == '6':
                run_problem_solving_tests(tester)
            elif choice == '7':
                run_interactive_mode(tester)
            elif choice == '8':
                run_basic_tests(tester)
                run_math_tests(tester)
                run_reasoning_tests(tester)
                run_creativity_tests(tester)
                run_knowledge_tests(tester)
                run_problem_solving_tests(tester)
            elif choice == '9':
                main()  # Restart to select new model
                break
            else:
                print("Invalid choice. Please select 0-9.")
            
            if choice != '7':  # Don't pause after interactive mode
                input("\nPress Enter to continue...")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()
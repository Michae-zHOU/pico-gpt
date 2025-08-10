#!/usr/bin/env python3
"""
Quick Model Capability Test
Fast way to verify what your model can and cannot do
"""

import torch
import sys
import os
import time
import re

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.pico_gpt import GPT

class QuickTester:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model()
        
    def load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']
        self.tokenizer = checkpoint['tokenizer']
        
        self.model = GPT(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        n_params = sum(p.numel() for p in self.model.parameters())
        model_name = os.path.basename(self.model_path)
        print(f"Testing: {model_name} ({n_params:,} parameters)")
        
    def test(self, question, expected_type="text", expected_value=None):
        """Test a single question"""
        prompt = f"Human: {question}\nAssistant:"
        
        try:
            tokens = self.tokenizer.encode(prompt)
            context = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                generated = self.model.generate(context, max_new_tokens=100, temperature=0.5, top_k=10)
            
            response = self.tokenizer.decode(generated[0].tolist())[len(prompt):].strip()
            
            # Evaluate response based on type
            success = False
            if expected_type == "number" and expected_value is not None:
                numbers = re.findall(r'\b\d+\b', response)
                if numbers:
                    predicted = int(numbers[-1])
                    success = predicted == expected_value
            elif expected_type == "contains":
                success = any(word.lower() in response.lower() for word in expected_value)
            elif expected_type == "text":
                success = len(response.strip()) > 10  # Has meaningful response
            
            # Show result
            status = "✅" if success else "❌"
            print(f"{status} Q: {question}")
            print(f"    A: {response[:80]}{'...' if len(response) > 80 else ''}")
            if expected_type == "number" and expected_value is not None:
                if numbers:
                    print(f"    Expected: {expected_value}, Got: {numbers[-1]}")
                else:
                    print(f"    Expected: {expected_value}, Got: No number found")
            print()
            
            return success, response
            
        except Exception as e:
            print(f"❌ Q: {question}")
            print(f"    Error: {e}")
            print()
            return False, str(e)

def run_quick_test(model_path):
    """Run quick capability assessment"""
    print("🚀 QUICK MODEL CAPABILITY TEST")
    print("=" * 50)
    
    tester = QuickTester(model_path)
    
    # Test categories with scoring
    categories = {
        "Basic Conversation": [],
        "Simple Math": [],
        "Reasoning": [],
        "Knowledge": []
    }
    
    print("\n📋 BASIC CONVERSATION")
    print("-" * 30)
    success, _ = tester.test("Hello, how are you?", "text")
    categories["Basic Conversation"].append(success)
    
    success, _ = tester.test("What is your name?", "text")
    categories["Basic Conversation"].append(success)
    
    success, _ = tester.test("Can you help me?", "contains", ["yes", "help", "sure", "of course"])
    categories["Basic Conversation"].append(success)
    
    print("\n🧮 SIMPLE MATH")
    print("-" * 30)
    success, _ = tester.test("What is 5 + 3?", "number", 8)
    categories["Simple Math"].append(success)
    
    success, _ = tester.test("What is 12 × 4?", "number", 48)
    categories["Simple Math"].append(success)
    
    success, _ = tester.test("What is 20 ÷ 4?", "number", 5)
    categories["Simple Math"].append(success)
    
    print("\n🧠 REASONING")
    print("-" * 30)
    success, _ = tester.test("If I have 10 apples and eat 3, how many are left?", "number", 7)
    categories["Reasoning"].append(success)
    
    success, _ = tester.test("What comes next: 2, 4, 6, 8, ?", "number", 10)
    categories["Reasoning"].append(success)
    
    success, _ = tester.test("If all birds fly and a robin is a bird, can a robin fly?", "contains", ["yes", "can", "fly"])
    categories["Reasoning"].append(success)
    
    print("\n📚 KNOWLEDGE")
    print("-" * 30)
    success, _ = tester.test("What is the capital of France?", "contains", ["paris"])
    categories["Knowledge"].append(success)
    
    success, _ = tester.test("How many days are in a week?", "number", 7)
    categories["Knowledge"].append(success)
    
    success, _ = tester.test("What color do you get when you mix red and blue?", "contains", ["purple", "violet"])
    categories["Knowledge"].append(success)
    
    # Calculate scores
    print("\n📊 CAPABILITY SCORES")
    print("=" * 30)
    total_score = 0
    total_tests = 0
    
    for category, results in categories.items():
        score = sum(results)
        total = len(results)
        percentage = (score / total) * 100 if total > 0 else 0
        total_score += score
        total_tests += total
        
        # Visual progress bar
        bar_length = 20
        filled = int(bar_length * percentage / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        print(f"{category:20}: {bar} {percentage:5.1f}% ({score}/{total})")
    
    overall_percentage = (total_score / total_tests) * 100 if total_tests > 0 else 0
    
    print("-" * 50)
    print(f"{'OVERALL CAPABILITY':20}: {overall_percentage:5.1f}% ({total_score}/{total_tests})")
    
    # Give assessment
    print("\n🎯 ASSESSMENT")
    print("-" * 20)
    if overall_percentage >= 90:
        assessment = "🌟 EXCELLENT - Model shows strong capabilities across all areas"
    elif overall_percentage >= 75:
        assessment = "🎉 VERY GOOD - Model performs well with minor weaknesses"
    elif overall_percentage >= 60:
        assessment = "👍 GOOD - Model shows decent capabilities, some areas need work"
    elif overall_percentage >= 40:
        assessment = "⚠️  FAIR - Model has basic functionality but needs improvement"
    elif overall_percentage >= 20:
        assessment = "📝 WEAK - Model struggles with many tasks, needs more training"
    else:
        assessment = "🔧 VERY WEAK - Model needs significant improvement"
    
    print(assessment)
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 30)
    
    weakest_category = min(categories.keys(), key=lambda k: sum(categories[k])/len(categories[k]))
    weakest_score = sum(categories[weakest_category])/len(categories[weakest_category]) * 100
    
    if weakest_score < 50:
        print(f"• Focus on improving: {weakest_category}")
    
    if categories["Simple Math"][0] + categories["Simple Math"][1] + categories["Simple Math"][2] < 2:
        print("• Consider training with more mathematical examples")
    
    if categories["Reasoning"][0] + categories["Reasoning"][1] + categories["Reasoning"][2] < 2:
        print("• Use chain-of-thought training for better reasoning")
    
    if categories["Basic Conversation"][0] + categories["Basic Conversation"][1] + categories["Basic Conversation"][2] < 2:
        print("• Train with more conversational data")
    
    if overall_percentage < 60:
        print("• Consider larger model architecture")
        print("• Increase training time/iterations")
        print("• Use higher quality training data")

def main():
    """Main program"""
    # Find models
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("❌ No models directory found")
        return
    
    models = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
    if not models:
        print("❌ No trained models found")
        return
    
    print("Available models:")
    for i, model in enumerate(models, 1):
        size = os.path.getsize(os.path.join(models_dir, model)) / 1024 / 1024
        print(f"{i}. {model} ({size:.1f} MB)")
    
    # Select model
    try:
        choice = input(f"\nSelect model (1-{len(models)}) or Enter for first: ").strip()
        if not choice:
            selected = models[0]
        else:
            idx = int(choice) - 1
            selected = models[idx]
        
        model_path = os.path.join(models_dir, selected)
        run_quick_test(model_path)
        
    except (ValueError, IndexError, KeyboardInterrupt):
        print("Invalid selection or cancelled")

if __name__ == "__main__":
    main()
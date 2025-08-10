#!/usr/bin/env python3
"""
Reasoning evaluation script for GPT models
Tests mathematical reasoning, logical deduction, and problem-solving capabilities
"""

import torch
import sys
import os
import time
import re
import math

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.pico_gpt import GPT
from src.modern_tokenizer import ModernBPETokenizer

class ReasoningEvaluator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model()
        
    def load_model(self):
        """Load the trained reasoning model"""
        print(f"Loading model: {self.model_path}")
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            self.config = checkpoint['config']
            self.tokenizer = checkpoint['tokenizer']
            
            self.model = GPT(self.config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded: {sum(p.numel() for p in self.model.parameters()):,} parameters")
            print(f"Device: {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate_response(self, prompt, max_tokens=150, temperature=0.3, top_k=10):
        """Generate response to a reasoning prompt"""
        try:
            tokens = self.tokenizer.encode(prompt)
            context = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                generated = self.model.generate(
                    context, 
                    max_new_tokens=max_tokens, 
                    temperature=temperature, 
                    top_k=top_k
                )
            
            full_response = self.tokenizer.decode(generated[0].tolist())
            # Extract just the assistant's response
            response = full_response[len(prompt):].strip()
            return response
            
        except Exception as e:
            return f"Error generating response: {e}"
    
    def evaluate_arithmetic(self):
        """Test basic arithmetic reasoning"""
        print("\n" + "="*50)
        print("ARITHMETIC REASONING EVALUATION")
        print("="*50)
        
        test_cases = [
            {"problem": "23 × 45", "answer": 23 * 45},
            {"problem": "67 × 34", "answer": 67 * 34},
            {"problem": "89 × 12", "answer": 89 * 12},
            {"problem": "156 ÷ 12", "answer": 156 // 12},
            {"problem": "144 ÷ 16", "answer": 144 // 16},
        ]
        
        correct = 0
        total = len(test_cases)
        
        for i, case in enumerate(test_cases, 1):
            prompt = f"Human: What is {case['problem']}?\nAssistant:"
            response = self.generate_response(prompt)
            
            # Extract numerical answer from response
            numbers = re.findall(r'\d+', response)
            predicted = int(numbers[-1]) if numbers else None
            
            is_correct = predicted == case['answer']
            correct += is_correct
            
            print(f"\nTest {i}: {case['problem']}")
            print(f"Expected: {case['answer']}")
            print(f"Predicted: {predicted}")
            print(f"Response: {response[:100]}...")
            print(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")
        
        accuracy = correct / total * 100
        print(f"\nArithmetic Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        return accuracy
    
    def evaluate_word_problems(self):
        """Test word problem solving"""
        print("\n" + "="*50)
        print("WORD PROBLEM EVALUATION")
        print("="*50)
        
        test_cases = [
            {
                "problem": "If each apple costs $3 and I buy 7 apples, how much do I spend?",
                "answer": 21
            },
            {
                "problem": "A farmer has 50 chickens and 30 cows. How many legs do all animals have in total?",
                "answer": 50 * 2 + 30 * 4  # 220
            },
            {
                "problem": "I have 15 books. I give away 4 and buy 8 more. How many books do I have now?",
                "answer": 15 - 4 + 8  # 19
            },
        ]
        
        correct = 0
        total = len(test_cases)
        
        for i, case in enumerate(test_cases, 1):
            prompt = f"Human: {case['problem']}\nAssistant:"
            response = self.generate_response(prompt, max_tokens=200)
            
            # Extract numerical answer
            numbers = re.findall(r'\d+', response)
            predicted = int(numbers[-1]) if numbers else None
            
            is_correct = predicted == case['answer']
            correct += is_correct
            
            print(f"\nTest {i}: {case['problem']}")
            print(f"Expected: {case['answer']}")
            print(f"Predicted: {predicted}")
            print(f"Response: {response[:150]}...")
            print(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")
        
        accuracy = correct / total * 100
        print(f"\nWord Problem Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        return accuracy
    
    def evaluate_sequences(self):
        """Test sequence pattern recognition"""
        print("\n" + "="*50)
        print("SEQUENCE PATTERN EVALUATION")
        print("="*50)
        
        test_cases = [
            {"sequence": "2, 4, 6, 8", "answer": 10},
            {"sequence": "1, 4, 9, 16", "answer": 25},
            {"sequence": "1, 1, 2, 3, 5", "answer": 8},
            {"sequence": "5, 10, 15, 20", "answer": 25},
        ]
        
        correct = 0
        total = len(test_cases)
        
        for i, case in enumerate(test_cases, 1):
            prompt = f"Human: What comes next in this sequence: {case['sequence']}?\nAssistant:"
            response = self.generate_response(prompt)
            
            # Extract numerical answer
            numbers = re.findall(r'\d+', response)
            predicted = int(numbers[-1]) if numbers else None
            
            is_correct = predicted == case['answer']
            correct += is_correct
            
            print(f"\nTest {i}: {case['sequence']}")
            print(f"Expected: {case['answer']}")
            print(f"Predicted: {predicted}")
            print(f"Response: {response[:100]}...")
            print(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")
        
        accuracy = correct / total * 100
        print(f"\nSequence Pattern Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        return accuracy
    
    def evaluate_logical_reasoning(self):
        """Test logical reasoning capabilities"""
        print("\n" + "="*50)
        print("LOGICAL REASONING EVALUATION")
        print("="*50)
        
        test_cases = [
            {
                "premise": 'Given: "All birds can fly" and "A robin is a bird"',
                "question": "Can a robin fly?",
                "expected_keywords": ["yes", "can", "fly", "robin"]
            },
            {
                "premise": 'Given: "If it rains, the ground gets wet" and "It is raining"',
                "question": "Is the ground wet?",
                "expected_keywords": ["yes", "wet", "ground"]
            },
            {
                "premise": 'Given: "All roses are flowers" and "All flowers need water"',
                "question": "Do roses need water?",
                "expected_keywords": ["yes", "need", "water", "roses"]
            },
        ]
        
        correct = 0
        total = len(test_cases)
        
        for i, case in enumerate(test_cases, 1):
            prompt = f"Human: {case['premise']}. {case['question']}\nAssistant:"
            response = self.generate_response(prompt, max_tokens=100)
            
            # Check if response contains expected reasoning
            response_lower = response.lower()
            has_keywords = any(keyword in response_lower for keyword in case['expected_keywords'])
            
            is_correct = has_keywords
            correct += is_correct
            
            print(f"\nTest {i}: {case['question']}")
            print(f"Premise: {case['premise']}")
            print(f"Response: {response[:150]}...")
            print(f"Keywords found: {[kw for kw in case['expected_keywords'] if kw in response_lower]}")
            print(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")
        
        accuracy = correct / total * 100
        print(f"\nLogical Reasoning Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        return accuracy
    
    def evaluate_chain_of_thought(self):
        """Test chain-of-thought reasoning"""
        print("\n" + "="*50)
        print("CHAIN-OF-THOUGHT EVALUATION")
        print("="*50)
        
        test_cases = [
            {
                "problem": "A rectangle is 8 meters long and 5 meters wide. What is its area?",
                "answer": 40,
                "reasoning_keywords": ["multiply", "length", "width", "8", "5"]
            },
            {
                "problem": "Tom has twice as many marbles as Jerry. If Jerry has 12 marbles, how many marbles do they have together?",
                "answer": 36,  # Jerry: 12, Tom: 24, Total: 36
                "reasoning_keywords": ["twice", "12", "24", "together"]
            },
        ]
        
        correct = 0
        reasoning_score = 0
        total = len(test_cases)
        
        for i, case in enumerate(test_cases, 1):
            prompt = f"Human: {case['problem']}\nAssistant:"
            response = self.generate_response(prompt, max_tokens=250)
            
            # Check numerical answer
            numbers = re.findall(r'\d+', response)
            predicted = int(numbers[-1]) if numbers else None
            answer_correct = predicted == case['answer']
            
            # Check reasoning process
            response_lower = response.lower()
            reasoning_keywords_found = sum(1 for kw in case['reasoning_keywords'] if kw in response_lower)
            reasoning_quality = reasoning_keywords_found / len(case['reasoning_keywords'])
            
            correct += answer_correct
            reasoning_score += reasoning_quality
            
            print(f"\nTest {i}: {case['problem']}")
            print(f"Expected Answer: {case['answer']}")
            print(f"Predicted Answer: {predicted}")
            print(f"Response: {response}")
            print(f"Answer Correct: {'YES' if answer_correct else 'NO'}")
            print(f"Reasoning Quality: {reasoning_quality:.2f}")
        
        accuracy = correct / total * 100
        reasoning_avg = reasoning_score / total * 100
        print(f"\nChain-of-Thought Results:")
        print(f"Answer Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        print(f"Reasoning Quality: {reasoning_avg:.1f}%")
        return accuracy, reasoning_avg

    def run_full_evaluation(self):
        """Run complete reasoning evaluation"""
        print("REASONING MODEL EVALUATION")
        print("="*60)
        
        start_time = time.time()
        
        # Run all evaluations
        arithmetic_acc = self.evaluate_arithmetic()
        word_problem_acc = self.evaluate_word_problems()
        sequence_acc = self.evaluate_sequences()
        logical_acc = self.evaluate_logical_reasoning()
        cot_acc, cot_reasoning = self.evaluate_chain_of_thought()
        
        # Calculate overall score
        overall_score = (arithmetic_acc + word_problem_acc + sequence_acc + logical_acc + cot_acc) / 5
        
        # Final summary
        elapsed = time.time() - start_time
        print("\n" + "="*60)
        print("FINAL REASONING EVALUATION SUMMARY")
        print("="*60)
        print(f"Model: {self.model_path}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Evaluation time: {elapsed:.1f}s")
        print()
        print(f"Arithmetic Reasoning:     {arithmetic_acc:.1f}%")
        print(f"Word Problems:           {word_problem_acc:.1f}%")
        print(f"Sequence Patterns:       {sequence_acc:.1f}%")
        print(f"Logical Reasoning:       {logical_acc:.1f}%")
        print(f"Chain-of-Thought:        {cot_acc:.1f}%")
        print(f"Reasoning Quality:       {cot_reasoning:.1f}%")
        print("-" * 30)
        print(f"OVERALL REASONING SCORE: {overall_score:.1f}%")
        
        # Grade the model
        if overall_score >= 90:
            grade = "A+ (Excellent)"
        elif overall_score >= 80:
            grade = "A (Very Good)"
        elif overall_score >= 70:
            grade = "B (Good)"
        elif overall_score >= 60:
            grade = "C (Fair)"
        else:
            grade = "D (Needs Improvement)"
            
        print(f"Reasoning Grade:         {grade}")
        
        return {
            'overall_score': overall_score,
            'arithmetic': arithmetic_acc,
            'word_problems': word_problem_acc,
            'sequences': sequence_acc,
            'logical': logical_acc,
            'chain_of_thought': cot_acc,
            'reasoning_quality': cot_reasoning,
            'grade': grade
        }

def main():
    # Check for available models
    models_dir = "models"
    available_models = []
    
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith('.pt'):
                available_models.append(f)
    
    if not available_models:
        print("No trained models found in models/ directory")
        print("Train a reasoning model first: python training/train_reasoning_model.py")
        return
    
    print("Available models:")
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model}")
    
    # Use reasoning model if available, otherwise use best available
    reasoning_model = None
    for model in available_models:
        if 'reasoning' in model:
            reasoning_model = model
            break
    
    if reasoning_model:
        model_path = os.path.join(models_dir, reasoning_model)
        print(f"\nUsing reasoning model: {reasoning_model}")
    else:
        model_path = os.path.join(models_dir, available_models[0])
        print(f"\nNo reasoning model found. Using: {available_models[0]}")
        print("For best results, train reasoning model: python training/train_reasoning_model.py")
    
    # Run evaluation
    try:
        evaluator = ReasoningEvaluator(model_path)
        results = evaluator.run_full_evaluation()
        
        # Save results
        results_file = f"reasoning_evaluation_results.txt"
        with open(results_file, 'w') as f:
            f.write(f"Reasoning Evaluation Results\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Overall Score: {results['overall_score']:.1f}%\n")
            f.write(f"Grade: {results['grade']}\n")
            f.write(f"Arithmetic: {results['arithmetic']:.1f}%\n")
            f.write(f"Word Problems: {results['word_problems']:.1f}%\n")
            f.write(f"Sequences: {results['sequences']:.1f}%\n")
            f.write(f"Logical: {results['logical']:.1f}%\n")
            f.write(f"Chain-of-Thought: {results['chain_of_thought']:.1f}%\n")
        
        print(f"\nResults saved to: {results_file}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")

if __name__ == "__main__":
    main()
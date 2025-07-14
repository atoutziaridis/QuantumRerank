#!/usr/bin/env python3
"""
Test 1 (Optimized): Few-Shot Biomedical Classification
Optimized version that batches operations for faster testing.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import random

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum_rerank.core.rag_reranker import QuantumRAGReranker
from quantum_rerank.core.quantum_similarity_engine import SimilarityEngineConfig, SimilarityMethod
from quantum_rerank.training.qpmel_trainer import QPMeLTrainer, QPMeLTrainingConfig
from quantum_rerank.ml.qpmel_circuits import QPMeLConfig

class OptimizedFewShotDataset:
    """Creates simplified few-shot scenarios for faster testing."""
    
    def __init__(self):
        # Smaller, focused biomedical dataset for speed
        self.data = {
            "cardiovascular": [
                "Heart attack causes cardiac muscle death from blocked coronary arteries.",
                "High blood pressure damages arteries and increases heart disease risk.",
                "Cholesterol buildup narrows arteries causing reduced blood flow.",
                "Irregular heartbeat can cause dangerous arrhythmias and stroke.",
                "Heart failure occurs when heart cannot pump blood effectively."
            ],
            
            "neurological": [
                "Alzheimer's disease causes progressive memory loss and dementia.",
                "Stroke results from blocked blood vessels in the brain.",
                "Parkinson's disease affects movement with tremors and rigidity.",
                "Epilepsy involves abnormal brain electrical activity causing seizures.",
                "Multiple sclerosis damages nerve coverings affecting coordination."
            ],
            
            "infectious": [
                "Pneumonia is lung infection causing fever and breathing difficulty.",
                "Tuberculosis is bacterial lung disease with chronic cough.",
                "HIV weakens immune system making infections more dangerous.",
                "Influenza virus causes seasonal respiratory illness outbreaks.",
                "Sepsis is life-threatening infection spreading through bloodstream."
            ]
        }
    
    def sample_few_shot(self, shot_count: int, test_size: int = 3) -> Dict:
        """Sample few-shot training and test sets."""
        train_data = {}
        test_data = {}
        
        for category, documents in self.data.items():
            # Shuffle documents
            shuffled = documents.copy()
            random.shuffle(shuffled)
            
            # Split into train and test
            train_data[category] = shuffled[:shot_count]
            test_data[category] = shuffled[shot_count:shot_count + test_size]
        
        return {
            "train": train_data,
            "test": test_data,
            "categories": list(self.data.keys())
        }

class OptimizedFewShotEvaluator:
    """Optimized evaluator for faster few-shot testing."""
    
    def __init__(self):
        self.dataset = OptimizedFewShotDataset()
    
    def create_quantum_reranker(self, model_path: str = None):
        """Create quantum reranker."""
        if model_path and Path(model_path).exists():
            config = QPMeLTrainingConfig(
                qpmel_config=QPMeLConfig(n_qubits=2, n_layers=1),
                batch_size=8
            )
            trainer = QPMeLTrainer(config=config)
            trainer.load_model(model_path)
            return trainer.get_trained_reranker()
        else:
            config = SimilarityEngineConfig(
                n_qubits=2,
                n_layers=1,
                similarity_method=SimilarityMethod.QUANTUM_FIDELITY,
                enable_caching=True
            )
            return QuantumRAGReranker(config=config)
    
    def create_classical_baseline(self):
        """Create classical baseline reranker."""
        config = SimilarityEngineConfig(
            n_qubits=2,
            n_layers=1,
            similarity_method=SimilarityMethod.CLASSICAL_COSINE,
            enable_caching=True
        )
        return QuantumRAGReranker(config=config)
    
    def evaluate_classification_batch(self, reranker, train_data, test_data, categories, method="quantum"):
        """Optimized batch evaluation using reranking."""
        correct_predictions = 0
        total_predictions = 0
        
        # Create flat list of all training examples with labels
        all_train_docs = []
        train_labels = []
        
        for category in categories:
            for doc in train_data[category]:
                all_train_docs.append(doc)
                train_labels.append(category)
        
        # Test each document against all training examples
        for true_category in categories:
            for test_doc in test_data[true_category]:
                try:
                    # Rerank all training documents for this test query
                    results = reranker.rerank(test_doc, all_train_docs, method=method, top_k=len(all_train_docs))
                    
                    if results:
                        # Get the top-ranked training document
                        top_result = results[0]
                        top_doc = top_result.get('text', top_result.get('content', ''))
                        
                        # Find which category this top document belongs to
                        predicted_category = None
                        for i, train_doc in enumerate(all_train_docs):
                            if train_doc == top_doc or top_doc in train_doc:
                                predicted_category = train_labels[i]
                                break
                        
                        if predicted_category == true_category:
                            correct_predictions += 1
                    
                    total_predictions += 1
                    
                except Exception as e:
                    print(f"Error in reranking: {e}")
                    total_predictions += 1
                    continue
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def run_fast_comparison(self, shot_counts: List[int], model_path: str = None):
        """Run optimized few-shot comparison."""
        print("🔬 Optimized Few-Shot Biomedical Classification Test")
        print("=" * 60)
        
        # Create rerankers
        print("\n🤖 Initializing Rerankers...")
        quantum_reranker = self.create_quantum_reranker(model_path)
        classical_reranker = self.create_classical_baseline()
        print("✅ Quantum reranker created")
        print("✅ Classical baseline created")
        
        results = {}
        
        for shot_count in shot_counts:
            print(f"\n📊 Testing {shot_count}-Shot Learning...")
            
            # Sample data once for this shot count
            data_split = self.dataset.sample_few_shot(shot_count, test_size=2)  # Smaller test size for speed
            
            print(f"  Training: {shot_count} examples per category")
            print(f"  Testing: 2 examples per category")
            
            # Test quantum reranker
            print("  🔬 Testing Quantum...")
            start_time = time.time()
            quantum_acc = self.evaluate_classification_batch(
                quantum_reranker, 
                data_split["train"], 
                data_split["test"], 
                data_split["categories"],
                method="quantum"
            )
            quantum_time = time.time() - start_time
            
            # Test classical reranker  
            print("  📊 Testing Classical...")
            start_time = time.time()
            classical_acc = self.evaluate_classification_batch(
                classical_reranker,
                data_split["train"],
                data_split["test"],
                data_split["categories"],
                method="classical"
            )
            classical_time = time.time() - start_time
            
            # Calculate improvement
            improvement = ((quantum_acc - classical_acc) / classical_acc * 100) if classical_acc > 0 else 0
            
            results[shot_count] = {
                "quantum_accuracy": quantum_acc,
                "classical_accuracy": classical_acc,
                "improvement_pct": improvement,
                "quantum_time": quantum_time,
                "classical_time": classical_time
            }
            
            print(f"  📈 Quantum: {quantum_acc:.3f} ({quantum_time:.1f}s)")
            print(f"  📈 Classical: {classical_acc:.3f} ({classical_time:.1f}s)")
            print(f"  🎯 Improvement: {improvement:+.1f}%")
        
        return results
    
    def analyze_fast_results(self, results):
        """Analyze optimized results."""
        print(f"\n📊 FAST FEW-SHOT ANALYSIS")
        print("=" * 50)
        
        print(f"\n{'Shots':<8} {'Quantum':<10} {'Classical':<10} {'Improvement':<12}")
        print("-" * 50)
        
        best_improvement = -float('inf')
        best_shot_count = 0
        significant_wins = 0
        
        for shot_count, data in results.items():
            quantum_acc = data["quantum_accuracy"]
            classical_acc = data["classical_accuracy"]
            improvement = data["improvement_pct"]
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_shot_count = shot_count
            
            if improvement > 5:  # >5% improvement considered significant
                significant_wins += 1
            
            print(f"{shot_count:<8} {quantum_acc:<10.3f} {classical_acc:<10.3f} {improvement:+.1f}%")
        
        print(f"\n💡 QUICK ASSESSMENT:")
        print("-" * 25)
        
        if significant_wins > 0:
            print(f"✅ Found {significant_wins} scenarios with >5% quantum advantage")
            print(f"🏆 Best: {best_shot_count}-shot with {best_improvement:+.1f}% improvement")
            
            if best_shot_count <= 5:
                print("🎯 Quantum excels in ultra-low data (≤5 examples)")
                recommendation = "PROCEED: Focus on few-shot applications"
            else:
                print("⚠️  Advantage requires more data than expected")
                recommendation = "CAUTIOUS: Investigate why more data needed"
        else:
            print("❌ No significant quantum advantage found")
            print("🔄 RECOMMENDATION: Move to Test 2 (Adversarial)")
            recommendation = "MOVE_ON: Try adversarial robustness test"
        
        return {
            "significant_wins": significant_wins,
            "best_improvement": best_improvement,
            "best_shot_count": best_shot_count,
            "recommendation": recommendation,
            "quantum_advantage_found": significant_wins > 0
        }

def main():
    """Main function for optimized few-shot test."""
    print("🚀 Starting Optimized Few-Shot Test (Test 1)")
    print("Fast version for initial quantum advantage detection")
    
    # Smaller test parameters for speed
    shot_counts = [1, 2, 3, 5]  # Reduced shot counts
    
    # Check for trained model
    model_path = "models/qpmel_extended.pt"
    if not Path(model_path).exists():
        print(f"⚠️  Using default quantum parameters")
        model_path = None
    else:
        print(f"✅ Using trained QPMeL model")
    
    # Run evaluation
    evaluator = OptimizedFewShotEvaluator()
    
    print("\n⏱️  Starting fast evaluation...")
    start_total = time.time()
    
    results = evaluator.run_fast_comparison(
        shot_counts=shot_counts,
        model_path=model_path
    )
    
    total_time = time.time() - start_total
    
    # Analyze results
    analysis = evaluator.analyze_fast_results(results)
    
    # Save results
    output_data = {
        "test_name": "optimized_few_shot_biomedical",
        "total_time_seconds": total_time,
        "results": results,
        "analysis": analysis
    }
    
    with open("optimized_few_shot_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to optimized_few_shot_results.json")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    
    # Decision point
    if analysis["quantum_advantage_found"]:
        print(f"\n🎉 QUANTUM ADVANTAGE DETECTED!")
        print(f"🎯 Best scenario: {analysis['best_shot_count']}-shot")
        print(f"📈 Improvement: {analysis['best_improvement']:+.1f}%")
        print(f"🚀 NEXT: Optimize this scenario further")
    else:
        print(f"\n🔄 NO CLEAR ADVANTAGE IN FEW-SHOT")
        print(f"📝 Moving to Test 2: Adversarial Robustness")
        print(f"💡 Few-shot negative result guides next test design")
    
    print(f"\n✨ Optimized Test 1 Complete!")
    return analysis["quantum_advantage_found"]

if __name__ == "__main__":
    main()
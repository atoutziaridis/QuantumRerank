#!/usr/bin/env python3
"""
Test 2: Adversarial Robustness
Test if quantum reranker is more robust to adversarial text perturbations.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import random
import re

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum_rerank.core.rag_reranker import QuantumRAGReranker
from quantum_rerank.core.quantum_similarity_engine import SimilarityEngineConfig, SimilarityMethod
from quantum_rerank.training.qpmel_trainer import QPMeLTrainer, QPMeLTrainingConfig
from quantum_rerank.ml.qpmel_circuits import QPMeLConfig

class AdversarialTextGenerator:
    """Creates adversarial text examples that challenge classical similarity."""
    
    def __init__(self):
        self.adversarial_types = [
            "negation",
            "contradiction", 
            "word_order",
            "synonym_swap",
            "noise_injection"
        ]
    
    def create_negation_pairs(self) -> List[Dict]:
        """Create pairs where negation changes meaning but keeps similar words."""
        return [
            {
                "query": "The treatment is effective for reducing symptoms",
                "correct": "The medication shows effectiveness in symptom reduction",
                "adversarial": "The treatment is NOT effective for reducing symptoms",
                "type": "negation"
            },
            {
                "query": "Patients experienced significant improvement with therapy",
                "correct": "Therapy led to notable patient improvements", 
                "adversarial": "Patients experienced no significant improvement with therapy",
                "type": "negation"
            },
            {
                "query": "The drug increases survival rates in clinical trials",
                "correct": "Clinical trials show improved survival with the medication",
                "adversarial": "The drug does not increase survival rates in clinical trials",
                "type": "negation"
            }
        ]
    
    def create_word_order_pairs(self) -> List[Dict]:
        """Create pairs where word order changes meaning."""
        return [
            {
                "query": "Apply treatment A before treatment B for best results",
                "correct": "Treatment A should precede treatment B for optimal outcomes",
                "adversarial": "Apply treatment B before treatment A for best results",
                "type": "word_order"
            },
            {
                "query": "Symptoms appeared after medication was discontinued",
                "correct": "Discontinuing medication led to symptom appearance",
                "adversarial": "Medication was discontinued after symptoms appeared",
                "type": "word_order"
            },
            {
                "query": "Fever resolved then rash appeared in patient",
                "correct": "Patient's fever subsided followed by rash development",
                "adversarial": "Rash appeared then fever resolved in patient",
                "type": "word_order"
            }
        ]
    
    def create_contradiction_pairs(self) -> List[Dict]:
        """Create subtle contradictions with high word overlap."""
        return [
            {
                "query": "Study shows vaccine prevents disease transmission",
                "correct": "Vaccination effectively blocks disease spread according to research",
                "adversarial": "Study shows vaccine prevents disease symptoms but not transmission",
                "type": "contradiction"
            },
            {
                "query": "Surgery is the primary treatment option",
                "correct": "Surgical intervention represents the main therapeutic approach",
                "adversarial": "Surgery is considered after other primary treatment options fail",
                "type": "contradiction"
            },
            {
                "query": "Early diagnosis improves patient outcomes significantly",
                "correct": "Patient prognosis benefits greatly from early detection",
                "adversarial": "Early diagnosis shows minimal impact on patient outcomes",
                "type": "contradiction"
            }
        ]
    
    def inject_noise(self, text: str, noise_level: float = 0.1) -> str:
        """Inject random noise into text."""
        words = text.split()
        num_words_to_modify = int(len(words) * noise_level)
        
        for _ in range(num_words_to_modify):
            idx = random.randint(0, len(words) - 1)
            noise_type = random.choice(['typo', 'insert', 'swap'])
            
            if noise_type == 'typo' and len(words[idx]) > 3:
                # Random character change
                char_idx = random.randint(1, len(words[idx]) - 2)
                words[idx] = words[idx][:char_idx] + random.choice('abcdefghijklmnopqrstuvwxyz') + words[idx][char_idx+1:]
            elif noise_type == 'insert':
                # Insert random word
                words.insert(idx, random.choice(['the', 'and', 'or', 'with', 'for']))
            elif noise_type == 'swap' and idx < len(words) - 1:
                # Swap adjacent words
                words[idx], words[idx + 1] = words[idx + 1], words[idx]
        
        return ' '.join(words)
    
    def create_all_adversarial_scenarios(self) -> List[Dict]:
        """Create comprehensive adversarial test set."""
        scenarios = []
        scenarios.extend(self.create_negation_pairs())
        scenarios.extend(self.create_word_order_pairs())
        scenarios.extend(self.create_contradiction_pairs())
        return scenarios

class AdversarialRobustnessEvaluator:
    """Evaluates robustness of rerankers to adversarial examples."""
    
    def __init__(self):
        self.generator = AdversarialTextGenerator()
        
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
    
    def evaluate_adversarial_robustness(self, reranker, scenarios, method="quantum"):
        """Evaluate how well reranker handles adversarial examples."""
        correct_rankings = 0
        total_tests = 0
        robustness_scores = []
        
        for scenario in scenarios:
            query = scenario["query"]
            correct_doc = scenario["correct"]
            adversarial_doc = scenario["adversarial"]
            
            # Test if reranker ranks correct document higher than adversarial
            candidates = [correct_doc, adversarial_doc]
            
            try:
                results = reranker.rerank(query, candidates, method=method, top_k=2)
                
                if results and len(results) >= 2:
                    top_doc = results[0].get('text', results[0].get('content', ''))
                    
                    # Check if correct document is ranked first
                    if top_doc == correct_doc or correct_doc in top_doc:
                        correct_rankings += 1
                        robustness_scores.append(1.0)
                    else:
                        robustness_scores.append(0.0)
                    
                total_tests += 1
                
            except Exception as e:
                print(f"Error in reranking: {e}")
                robustness_scores.append(0.0)
                total_tests += 1
        
        accuracy = correct_rankings / total_tests if total_tests > 0 else 0.0
        return {
            "accuracy": accuracy,
            "correct_rankings": correct_rankings,
            "total_tests": total_tests,
            "robustness_scores": robustness_scores
        }
    
    def evaluate_noise_robustness(self, reranker, clean_scenarios, noise_levels, method="quantum"):
        """Test robustness to increasing noise levels."""
        results_by_noise = {}
        
        for noise_level in noise_levels:
            # Create noisy versions of scenarios
            noisy_scenarios = []
            
            for scenario in clean_scenarios:
                noisy_scenario = scenario.copy()
                if noise_level > 0:
                    noisy_scenario["query"] = self.generator.inject_noise(scenario["query"], noise_level)
                    noisy_scenario["correct"] = self.generator.inject_noise(scenario["correct"], noise_level)
                noisy_scenarios.append(noisy_scenario)
            
            # Evaluate on noisy data
            results = self.evaluate_adversarial_robustness(reranker, noisy_scenarios, method)
            results_by_noise[noise_level] = results
        
        return results_by_noise
    
    def run_comprehensive_robustness_test(self, model_path: str = None):
        """Run complete adversarial robustness evaluation."""
        print("🛡️ Adversarial Robustness Test (Test 2)")
        print("=" * 60)
        print("Testing if quantum methods are more robust to adversarial text")
        
        # Create rerankers
        print("\n🤖 Initializing Rerankers...")
        quantum_reranker = self.create_quantum_reranker(model_path)
        classical_reranker = self.create_classical_baseline()
        print("✅ Quantum reranker created")
        print("✅ Classical baseline created")
        
        # Generate adversarial scenarios
        print("\n🎯 Generating Adversarial Scenarios...")
        scenarios = self.generator.create_all_adversarial_scenarios()
        print(f"Created {len(scenarios)} adversarial test cases")
        
        # Test 1: Direct adversarial robustness
        print("\n📊 Test 1: Direct Adversarial Examples")
        print("-" * 40)
        
        print("  🔬 Testing Quantum...")
        quantum_results = self.evaluate_adversarial_robustness(quantum_reranker, scenarios, "quantum")
        print(f"  Quantum accuracy: {quantum_results['accuracy']:.3f}")
        
        print("  📊 Testing Classical...")
        classical_results = self.evaluate_adversarial_robustness(classical_reranker, scenarios, "classical")
        print(f"  Classical accuracy: {classical_results['accuracy']:.3f}")
        
        improvement = ((quantum_results['accuracy'] - classical_results['accuracy']) / 
                      classical_results['accuracy'] * 100) if classical_results['accuracy'] > 0 else 0
        
        print(f"  🎯 Improvement: {improvement:+.1f}%")
        
        # Test 2: Noise robustness
        print("\n📊 Test 2: Robustness to Increasing Noise")
        print("-" * 40)
        
        noise_levels = [0.0, 0.1, 0.2, 0.3]
        print(f"  Testing noise levels: {noise_levels}")
        
        quantum_noise_results = self.evaluate_noise_robustness(
            quantum_reranker, scenarios[:3], noise_levels, "quantum"
        )
        
        classical_noise_results = self.evaluate_noise_robustness(
            classical_reranker, scenarios[:3], noise_levels, "classical"
        )
        
        print(f"\n  {'Noise':<8} {'Quantum':<10} {'Classical':<10} {'Q-Advantage'}")
        print("  " + "-" * 40)
        
        for noise_level in noise_levels:
            q_acc = quantum_noise_results[noise_level]['accuracy']
            c_acc = classical_noise_results[noise_level]['accuracy']
            advantage = q_acc - c_acc
            print(f"  {noise_level:<8.1f} {q_acc:<10.3f} {c_acc:<10.3f} {advantage:+.3f}")
        
        # Calculate robustness metrics
        quantum_degradation = (quantum_noise_results[0.0]['accuracy'] - 
                             quantum_noise_results[0.3]['accuracy'])
        classical_degradation = (classical_noise_results[0.0]['accuracy'] - 
                               classical_noise_results[0.3]['accuracy'])
        
        return {
            "adversarial_test": {
                "quantum": quantum_results,
                "classical": classical_results,
                "improvement_pct": improvement
            },
            "noise_robustness": {
                "quantum": quantum_noise_results,
                "classical": classical_noise_results,
                "quantum_degradation": quantum_degradation,
                "classical_degradation": classical_degradation
            },
            "scenarios": len(scenarios)
        }
    
    def analyze_results(self, results):
        """Analyze robustness test results."""
        print(f"\n🔍 ROBUSTNESS ANALYSIS")
        print("=" * 50)
        
        # Adversarial test analysis
        adv_results = results["adversarial_test"]
        q_acc = adv_results["quantum"]["accuracy"]
        c_acc = adv_results["classical"]["accuracy"]
        improvement = adv_results["improvement_pct"]
        
        print(f"\n📊 Adversarial Example Performance:")
        print(f"  Quantum: {q_acc:.3f}")
        print(f"  Classical: {c_acc:.3f}")
        print(f"  Improvement: {improvement:+.1f}%")
        
        # Noise robustness analysis
        noise_results = results["noise_robustness"]
        q_deg = noise_results["quantum_degradation"]
        c_deg = noise_results["classical_degradation"]
        
        print(f"\n📊 Noise Degradation (0% → 30% noise):")
        print(f"  Quantum degradation: {q_deg:.3f}")
        print(f"  Classical degradation: {c_deg:.3f}")
        
        # Determine if quantum shows robustness advantage
        robustness_advantage = False
        reasons = []
        
        if improvement > 5:
            robustness_advantage = True
            reasons.append(f"Better adversarial accuracy (+{improvement:.1f}%)")
        
        if q_deg < c_deg - 0.1:
            robustness_advantage = True
            reasons.append(f"More noise resistant ({q_deg:.3f} vs {c_deg:.3f} degradation)")
        
        print(f"\n💡 VERDICT:")
        print("-" * 25)
        
        if robustness_advantage:
            print("✅ Quantum shows robustness advantage!")
            for reason in reasons:
                print(f"  • {reason}")
            print("🎯 RECOMMENDATION: Explore robustness applications")
            verdict = "QUANTUM_ROBUST"
        else:
            print("❌ No significant robustness advantage")
            print("🔄 RECOMMENDATION: Move to Test 3 (Molecular)")
            verdict = "NO_ADVANTAGE"
        
        return {
            "robustness_advantage": robustness_advantage,
            "verdict": verdict,
            "reasons": reasons
        }

def main():
    """Main function for adversarial robustness test."""
    print("🚀 Starting Adversarial Robustness Test (Test 2)")
    print("Testing if quantum methods resist adversarial perturbations better")
    
    # Check for trained model
    model_path = "models/qpmel_extended.pt"
    if not Path(model_path).exists():
        print(f"⚠️  Using default quantum parameters")
        model_path = None
    else:
        print(f"✅ Using trained QPMeL model")
    
    # Run evaluation
    evaluator = AdversarialRobustnessEvaluator()
    
    print("\n⏱️  Starting robustness evaluation...")
    start_time = time.time()
    
    results = evaluator.run_comprehensive_robustness_test(model_path)
    
    total_time = time.time() - start_time
    
    # Analyze results
    analysis = evaluator.analyze_results(results)
    
    # Save results
    output_data = {
        "test_name": "adversarial_robustness",
        "total_time_seconds": total_time,
        "results": results,
        "analysis": analysis,
        "timestamp": time.time()
    }
    
    with open("adversarial_robustness_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to adversarial_robustness_results.json")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    
    # Decision point
    if analysis["robustness_advantage"]:
        print(f"\n🎉 QUANTUM ROBUSTNESS ADVANTAGE FOUND!")
        print(f"🛡️  Quantum methods more resistant to adversarial text")
        print(f"🚀 NEXT: Explore security/robustness applications")
    else:
        print(f"\n🔄 NO ROBUSTNESS ADVANTAGE")
        print(f"📝 Moving to Test 3: Molecular Similarity")
        print(f"🧪 Molecular domain may show quantum advantages")
    
    print(f"\n✨ Test 2 Complete!")
    return analysis["robustness_advantage"]

if __name__ == "__main__":
    main()
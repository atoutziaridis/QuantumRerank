#!/usr/bin/env python3
"""
Adversarial Edge Cases Test
Testing quantum robustness on challenging edge cases and extreme perturbations.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
import re
from scipy import stats
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum_rerank.core.rag_reranker import QuantumRAGReranker
from quantum_rerank.core.quantum_similarity_engine import SimilarityEngineConfig, SimilarityMethod
from quantum_rerank.training.qpmel_trainer import QPMeLTrainer, QPMeLTrainingConfig
from quantum_rerank.ml.qpmel_circuits import QPMeLConfig

class EdgeCaseAdversarialGenerator:
    """Generates extreme and edge case adversarial examples."""
    
    def __init__(self):
        self.edge_case_types = [
            "minimal_difference", "maximal_perturbation", "semantic_paradox",
            "contextual_ambiguity", "implicit_negation", "compound_logic",
            "temporal_paradox", "causality_reversal", "scope_trap",
            "double_negation", "conditional_logic", "quantifier_shift"
        ]
    
    def create_minimal_difference_cases(self) -> List[Dict]:
        """Cases where tiny changes completely alter meaning."""
        return [
            {
                "query": "Patient condition is stable and improving",
                "correct": "Medical status shows stability with positive progress",
                "adversarial": "Patient condition is stable and not improving",
                "type": "minimal_difference",
                "challenge": "Single word negation with high lexical overlap",
                "difficulty": "extreme"
            },
            {
                "query": "Treatment works in 90% of cases",
                "correct": "Therapy demonstrates efficacy in ninety percent of patients",
                "adversarial": "Treatment works in 9% of cases",
                "type": "minimal_difference", 
                "challenge": "Single digit change with massive semantic impact",
                "difficulty": "extreme"
            },
            {
                "query": "Drug is safe for pregnant women",
                "correct": "Medication poses no risk to expectant mothers",
                "adversarial": "Drug is unsafe for pregnant women",
                "type": "minimal_difference",
                "challenge": "Single prefix changes safety profile completely",
                "difficulty": "extreme"
            },
            {
                "query": "Surgery is necessary before radiation therapy",
                "correct": "Surgical intervention required prior to radiological treatment",
                "adversarial": "Surgery is necessary after radiation therapy",
                "type": "minimal_difference",
                "challenge": "Critical temporal ordering with life-threatening implications",
                "difficulty": "extreme"
            },
            {
                "query": "Virus spreads through airborne transmission",
                "correct": "Pathogen transmits via respiratory droplets in air",
                "adversarial": "Virus spreads through waterborne transmission",
                "type": "minimal_difference",
                "challenge": "Single word changes transmission vector",
                "difficulty": "extreme"
            }
        ]
    
    def create_maximal_perturbation_cases(self) -> List[Dict]:
        """Cases with maximum possible perturbation while maintaining surface similarity."""
        return [
            {
                "query": "Increase medication dosage gradually over time for safety",
                "correct": "Slowly escalate drug amounts progressively to ensure patient wellbeing",
                "adversarial": "Decrease medication dosage rapidly at once for danger",
                "type": "maximal_perturbation",
                "challenge": "Every key concept inverted while maintaining structure",
                "difficulty": "extreme"
            },
            {
                "query": "Early detection significantly improves survival rates",
                "correct": "Prompt identification substantially enhances patient longevity outcomes",
                "adversarial": "Late detection significantly worsens death rates",
                "type": "maximal_perturbation",
                "challenge": "All positive concepts flipped to negative",
                "difficulty": "extreme"
            },
            {
                "query": "New drug prevents disease progression in all patients",
                "correct": "Novel medication halts illness advancement across entire patient population",
                "adversarial": "Old drug causes disease progression in no patients",
                "type": "maximal_perturbation",
                "challenge": "Multiple concept inversions with logical complexity",
                "difficulty": "extreme"
            },
            {
                "query": "High-risk patients require immediate intensive care",
                "correct": "Critical condition individuals need urgent specialized treatment",
                "adversarial": "Low-risk patients avoid delayed minimal care",
                "type": "maximal_perturbation",
                "challenge": "Risk level, timing, and care intensity all inverted",
                "difficulty": "extreme"
            }
        ]
    
    def create_semantic_paradox_cases(self) -> List[Dict]:
        """Cases that create semantic paradoxes or contradictions."""
        return [
            {
                "query": "This statement is always true in medical practice",
                "correct": "The assertion consistently applies to healthcare situations",
                "adversarial": "This statement is always false in medical practice",
                "type": "semantic_paradox",
                "challenge": "Self-referential contradiction creates logical paradox",
                "difficulty": "extreme"
            },
            {
                "query": "All rules have exceptions except this rule",
                "correct": "Every guideline permits deviation excluding this principle",
                "adversarial": "All rules have exceptions including this rule",
                "type": "semantic_paradox",
                "challenge": "Self-referential rule about rules creates paradox",
                "difficulty": "extreme"
            },
            {
                "query": "Nobody knows everything about this disease",
                "correct": "No individual possesses complete knowledge of this condition",
                "adversarial": "Everybody knows nothing about this disease",
                "type": "semantic_paradox",
                "challenge": "Universal quantifier paradox with knowledge claims",
                "difficulty": "extreme"
            }
        ]
    
    def create_contextual_ambiguity_cases(self) -> List[Dict]:
        """Cases where context determines meaning dramatically."""
        return [
            {
                "query": "Patient temperature reading is critical",
                "correct": "Body heat measurement indicates severe medical concern",
                "adversarial": "Patient temperature reading is critical for diagnosis",
                "type": "contextual_ambiguity",
                "challenge": "Same words with completely different contextual implications",
                "difficulty": "extreme"
            },
            {
                "query": "Treatment should be aggressive in this case",
                "correct": "Therapy must be intensive and forceful for this patient",
                "adversarial": "Treatment should be aggressive in this legal case",
                "type": "contextual_ambiguity",
                "challenge": "Word 'case' shifts medical to legal context",
                "difficulty": "extreme"
            },
            {
                "query": "Discharge patient when symptoms resolve completely",
                "correct": "Release individual after full symptom resolution",
                "adversarial": "Discharge patient when symptoms resolve the conflict",
                "type": "contextual_ambiguity",
                "challenge": "Homonymous 'resolve' changes medical to interpersonal context",
                "difficulty": "extreme"
            }
        ]
    
    def create_implicit_negation_cases(self) -> List[Dict]:
        """Cases with hidden or implicit negations."""
        return [
            {
                "query": "Treatment rarely fails to improve symptoms",
                "correct": "Therapy typically succeeds in enhancing patient condition",
                "adversarial": "Treatment rarely succeeds to improve symptoms",
                "type": "implicit_negation",
                "challenge": "Double negation creates hidden positive meaning",
                "difficulty": "extreme"
            },
            {
                "query": "Few patients lack improvement with this drug",
                "correct": "Most individuals experience positive response to medication",
                "adversarial": "Few patients show improvement with this drug",
                "type": "implicit_negation",
                "challenge": "Implicit negation through 'lack' reverses meaning",
                "difficulty": "extreme"
            },
            {
                "query": "It is impossible that this treatment is ineffective",
                "correct": "The therapy cannot fail to provide beneficial results",
                "adversarial": "It is impossible that this treatment is effective",
                "type": "implicit_negation",
                "challenge": "Nested impossibility creates complex logical structure",
                "difficulty": "extreme"
            }
        ]
    
    def create_compound_logic_cases(self) -> List[Dict]:
        """Cases with complex logical connectives."""
        return [
            {
                "query": "If and only if symptoms persist, then increase dosage",
                "correct": "Escalate medication amount precisely when signs continue",
                "adversarial": "If and only if symptoms persist, then decrease dosage",
                "type": "compound_logic",
                "challenge": "Biconditional logic with critical medical decision",
                "difficulty": "extreme"
            },
            {
                "query": "Either surgery or radiation, but not both, is recommended",
                "correct": "Exclusive choice between surgical or radiological intervention",
                "adversarial": "Either surgery or radiation, and both, is recommended",
                "type": "compound_logic",
                "challenge": "Exclusive OR vs inclusive OR logical difference",
                "difficulty": "extreme"
            },
            {
                "query": "Treatment works unless patient has specific allergy",
                "correct": "Therapy effective except in cases of particular hypersensitivity",
                "adversarial": "Treatment works because patient has specific allergy",
                "type": "compound_logic",
                "challenge": "Exception condition vs causal condition",
                "difficulty": "extreme"
            }
        ]
    
    def create_temporal_paradox_cases(self) -> List[Dict]:
        """Cases with temporal logic paradoxes."""
        return [
            {
                "query": "Future test results will confirm past diagnosis",
                "correct": "Upcoming examination findings will validate previous medical assessment",
                "adversarial": "Past test results will confirm future diagnosis",
                "type": "temporal_paradox",
                "challenge": "Temporal causality reversal creates logical impossibility",
                "difficulty": "extreme"
            },
            {
                "query": "Prevention is better than cure after disease onset",
                "correct": "Prophylactic measures exceed therapeutic value post-illness",
                "adversarial": "Prevention is better than cure before disease onset",
                "type": "temporal_paradox",
                "challenge": "Temporal logic of prevention vs treatment",
                "difficulty": "extreme"
            }
        ]
    
    def create_quantifier_shift_cases(self) -> List[Dict]:
        """Cases where quantifier scope changes meaning dramatically."""
        return [
            {
                "query": "Every patient needs some medication for recovery",
                "correct": "All individuals require certain drugs for healing",
                "adversarial": "Some patient needs every medication for recovery",
                "type": "quantifier_shift",
                "challenge": "Quantifier scope shift changes feasibility completely",
                "difficulty": "extreme"
            },
            {
                "query": "All doctors know some treatment for most diseases",
                "correct": "Every physician understands certain therapies for majority of conditions",
                "adversarial": "Some doctors know all treatment for most diseases",
                "type": "quantifier_shift",
                "challenge": "Multiple quantifier interaction creates different claims",
                "difficulty": "extreme"
            },
            {
                "query": "Most symptoms appear before all complications develop",
                "correct": "Majority of signs emerge prior to complete complication onset",
                "adversarial": "All symptoms appear before most complications develop",
                "type": "quantifier_shift", 
                "challenge": "Temporal quantifier scope affects medical prediction",
                "difficulty": "extreme"
            }
        ]
    
    def create_comprehensive_edge_cases(self) -> List[Dict]:
        """Create comprehensive edge case dataset."""
        dataset = []
        dataset.extend(self.create_minimal_difference_cases())
        dataset.extend(self.create_maximal_perturbation_cases())
        dataset.extend(self.create_semantic_paradox_cases())
        dataset.extend(self.create_contextual_ambiguity_cases())
        dataset.extend(self.create_implicit_negation_cases())
        dataset.extend(self.create_compound_logic_cases())
        dataset.extend(self.create_temporal_paradox_cases())
        dataset.extend(self.create_quantifier_shift_cases())
        return dataset

class EdgeCaseEvaluator:
    """Evaluates performance on adversarial edge cases."""
    
    def __init__(self):
        self.generator = EdgeCaseAdversarialGenerator()
    
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
    
    def evaluate_edge_cases(self, reranker, scenarios: List[Dict], method: str) -> Dict:
        """Evaluate performance on edge cases."""
        results_by_type = defaultdict(list)
        all_results = []
        
        for scenario in scenarios:
            query = scenario["query"]
            correct_doc = scenario["correct"]
            adversarial_doc = scenario["adversarial"]
            edge_type = scenario["type"]
            challenge = scenario["challenge"]
            
            candidates = [correct_doc, adversarial_doc]
            
            try:
                results = reranker.rerank(query, candidates, method=method, top_k=2)
                
                if results and len(results) >= 2:
                    top_doc = results[0].get('text', results[0].get('content', ''))
                    success = (top_doc == correct_doc or correct_doc in top_doc)
                else:
                    success = False
                    
                results_by_type[edge_type].append(success)
                all_results.append(success)
                
            except Exception as e:
                print(f"Error in edge case reranking: {e}")
                results_by_type[edge_type].append(False)
                all_results.append(False)
        
        # Calculate accuracies by edge case type
        type_accuracies = {edge_type: np.mean(scores) for edge_type, scores in results_by_type.items()}
        overall_accuracy = np.mean(all_results)
        
        return {
            "overall_accuracy": overall_accuracy,
            "type_accuracies": type_accuracies,
            "total_scenarios": len(scenarios),
            "success_rate_by_type": {edge_type: f"{np.mean(scores):.1%}" for edge_type, scores in results_by_type.items()}
        }
    
    def run_edge_case_analysis(self, scenarios: List[Dict], n_trials: int = 3, model_path: str = None) -> Dict:
        """Run edge case analysis with multiple trials."""
        print(f"🎯 Running Edge Case Analysis ({n_trials} trials)")
        
        quantum_reranker = self.create_quantum_reranker(model_path)
        classical_reranker = self.create_classical_baseline()
        
        quantum_results = []
        classical_results = []
        
        for trial in range(n_trials):
            print(f"  Edge Case Trial {trial + 1}/{n_trials}")
            
            q_result = self.evaluate_edge_cases(quantum_reranker, scenarios, "quantum")
            c_result = self.evaluate_edge_cases(classical_reranker, scenarios, "classical")
            
            quantum_results.append(q_result)
            classical_results.append(c_result)
        
        return {
            "quantum_results": quantum_results,
            "classical_results": classical_results,
            "n_trials": n_trials
        }
    
    def analyze_edge_case_performance(self, quantum_results: List[Dict], classical_results: List[Dict]) -> Dict:
        """Analyze edge case performance differences."""
        # Overall performance
        q_overall = [result["overall_accuracy"] for result in quantum_results]
        c_overall = [result["overall_accuracy"] for result in classical_results]
        
        overall_stats = {
            "quantum_mean": np.mean(q_overall),
            "quantum_std": np.std(q_overall, ddof=1),
            "classical_mean": np.mean(c_overall),
            "classical_std": np.std(c_overall, ddof=1),
            "advantage": np.mean(q_overall) - np.mean(c_overall)
        }
        
        # Performance by edge case type
        edge_types = quantum_results[0]["type_accuracies"].keys()
        type_performance = {}
        
        for edge_type in edge_types:
            q_type_scores = [result["type_accuracies"][edge_type] for result in quantum_results]
            c_type_scores = [result["type_accuracies"][edge_type] for result in classical_results]
            
            type_performance[edge_type] = {
                "quantum_mean": np.mean(q_type_scores),
                "classical_mean": np.mean(c_type_scores),
                "advantage": np.mean(q_type_scores) - np.mean(c_type_scores),
                "quantum_wins": sum(1 for q, c in zip(q_type_scores, c_type_scores) if q > c),
                "total_trials": len(q_type_scores)
            }
        
        return {
            "overall": overall_stats,
            "by_edge_type": type_performance
        }
    
    def run_comprehensive_edge_case_evaluation(self, model_path: str = None) -> Dict:
        """Run comprehensive edge case evaluation."""
        print("🎯 COMPREHENSIVE EDGE CASE EVALUATION")
        print("=" * 60)
        
        # Create edge case dataset
        scenarios = self.generator.create_comprehensive_edge_cases()
        print(f"📊 Generated {len(scenarios)} edge case scenarios")
        
        edge_types = set(s['type'] for s in scenarios)
        print(f"🏷️  Edge case types: {', '.join(sorted(edge_types))}")
        
        # Show example challenges
        print(f"\n🔍 EXAMPLE CHALLENGES:")
        for scenario in scenarios[:3]:
            print(f"  • {scenario['type']}: {scenario['challenge']}")
        
        # Run evaluation
        print(f"\n🧪 RUNNING EDGE CASE TESTS")
        print("-" * 40)
        start_time = time.time()
        
        results = self.run_edge_case_analysis(scenarios, n_trials=3, model_path=model_path)
        analysis = self.analyze_edge_case_performance(
            results["quantum_results"], 
            results["classical_results"]
        )
        
        eval_time = time.time() - start_time
        
        return {
            "edge_case_evaluation": results,
            "performance_analysis": analysis,
            "scenarios": scenarios,
            "metadata": {
                "total_scenarios": len(scenarios),
                "edge_types": list(edge_types),
                "evaluation_time": eval_time
            }
        }

def main():
    """Main function for edge case adversarial testing."""
    print("🎯 Starting Edge Case Adversarial Robustness Test")
    print("Testing quantum performance on extreme adversarial cases")
    
    # Check for trained model
    model_path = "models/qpmel_extended.pt"
    if not Path(model_path).exists():
        print(f"⚠️  Using default quantum parameters")
        model_path = None
    else:
        print(f"✅ Using trained QPMeL model")
    
    # Run evaluation
    evaluator = EdgeCaseEvaluator()
    
    print(f"\n⏱️  Starting edge case evaluation...")
    total_start_time = time.time()
    
    results = evaluator.run_comprehensive_edge_case_evaluation(model_path)
    
    total_time = time.time() - total_start_time
    
    # Print results
    print(f"\n📊 EDGE CASE RESULTS")
    print("=" * 50)
    
    overall = results["performance_analysis"]["overall"]
    print(f"\n🎯 OVERALL EXTREME CASE PERFORMANCE:")
    print(f"  Quantum:  {overall['quantum_mean']:.1%} ± {overall['quantum_std']:.1%}")
    print(f"  Classical: {overall['classical_mean']:.1%} ± {overall['classical_std']:.1%}")
    print(f"  Advantage: {overall['advantage']:+.1%}")
    
    print(f"\n🔍 BY EDGE CASE TYPE:")
    type_perf = results["performance_analysis"]["by_edge_type"]
    print(f"{'Type':<20} {'Quantum':<10} {'Classical':<10} {'Advantage':<10} {'Q Wins'}")
    print("-" * 65)
    for edge_type, stats in type_perf.items():
        wins = f"{stats['quantum_wins']}/{stats['total_trials']}"
        print(f"{edge_type:<20} {stats['quantum_mean']:<10.1%} {stats['classical_mean']:<10.1%} "
              f"{stats['advantage']:+.1%}{'':>4} {wins}")
    
    # Save results
    output_data = {
        "test_name": "edge_case_adversarial",
        "total_time_seconds": total_time,
        "results": results,
        "timestamp": time.time()
    }
    
    output_file = "edge_case_adversarial_results.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to {output_file}")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    
    # Verdict
    advantage = overall['advantage']
    if advantage > 0.1:
        print(f"\n🏆 EXCEPTIONAL QUANTUM PERFORMANCE ON EDGE CASES!")
        print(f"🎯 {advantage:+.1%} advantage on most challenging scenarios")
        verdict = "EXCEPTIONAL_EDGE_PERFORMANCE"
    elif advantage > 0.05:
        print(f"\n✅ GOOD QUANTUM PERFORMANCE ON EDGE CASES")
        print(f"🎯 {advantage:+.1%} advantage on challenging scenarios")
        verdict = "GOOD_EDGE_PERFORMANCE"
    else:
        print(f"\n📊 MIXED PERFORMANCE ON EDGE CASES")
        print(f"🎯 {advantage:+.1%} advantage on challenging scenarios")
        verdict = "MIXED_EDGE_PERFORMANCE"
    
    print(f"\n✨ Edge Case Test Complete!")
    return verdict

if __name__ == "__main__":
    main()
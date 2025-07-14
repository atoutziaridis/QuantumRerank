#!/usr/bin/env python3
"""
Ultimate Adversarial Robustness Test
The definitive test combining the most challenging adversarial cases
to determine quantum advantage in semantic robustness.
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

class UltimateAdversarialDataset:
    """The ultimate collection of adversarial test cases."""
    
    def __init__(self):
        self.challenge_levels = ["expert", "extreme", "impossible"]
        
    def create_expert_level_cases(self) -> List[Dict]:
        """Expert-level adversarial cases."""
        return [
            {
                "query": "Patient shows 95% improvement with new treatment protocol",
                "correct": "Individual demonstrates near-complete recovery using novel therapeutic approach",
                "adversarial": "Patient shows 5% improvement with new treatment protocol",
                "type": "precision_critical",
                "domain": "medical",
                "challenge": "Single digit determines treatment success vs failure",
                "level": "expert"
            },
            {
                "query": "All clinical trials must be completed before drug approval",
                "correct": "Every research study requires completion prior to medication authorization",
                "adversarial": "Some clinical trials must be completed before drug approval",
                "type": "regulatory_critical",
                "domain": "medical",
                "challenge": "Quantifier determines regulatory compliance",
                "level": "expert"
            },
            {
                "query": "Evidence proves defendant innocent beyond reasonable doubt",
                "correct": "Proof establishes accused person's innocence with certainty",
                "adversarial": "Evidence proves defendant guilty beyond reasonable doubt",
                "type": "verdict_critical",
                "domain": "legal",
                "challenge": "Single word determines innocence vs guilt",
                "level": "expert"
            },
            {
                "query": "Investment guarantees 15% annual returns with zero risk",
                "correct": "Financial opportunity ensures fifteen percent yearly profits without danger",
                "adversarial": "Investment guarantees 15% annual losses with zero risk",
                "type": "financial_critical",
                "domain": "financial",
                "challenge": "Returns vs losses distinction critical for investment",
                "level": "expert"
            },
            {
                "query": "Nuclear reactor operates safely at current temperature levels",
                "correct": "Atomic facility functions securely at present thermal conditions",
                "adversarial": "Nuclear reactor operates unsafely at current temperature levels",
                "type": "safety_critical",
                "domain": "technical",
                "challenge": "Safety status critical for nuclear operations",
                "level": "expert"
            }
        ]
    
    def create_extreme_level_cases(self) -> List[Dict]:
        """Extreme-level adversarial cases."""
        return [
            {
                "query": "Treatment is contraindicated in pregnant women but recommended for everyone else",
                "correct": "Therapy prohibited for expectant mothers while advised for all other patients",
                "adversarial": "Treatment is recommended in pregnant women but contraindicated for everyone else",
                "type": "exception_reversal",
                "domain": "medical",
                "challenge": "Complete exception reversal with high-risk population",
                "level": "extreme"
            },
            {
                "query": "Either plead guilty and receive lenient sentence or fight charges and face maximum penalty",
                "correct": "Choose confession with reduced punishment or contest accusations risking severe consequences",
                "adversarial": "Either plead innocent and receive lenient sentence or fight charges and face maximum penalty",
                "type": "legal_logic_trap",
                "domain": "legal", 
                "challenge": "Logical structure completely inverted with legal implications",
                "level": "extreme"
            },
            {
                "query": "Stock price increases if and only if quarterly earnings exceed analyst projections",
                "correct": "Equity value rises precisely when financial results surpass expert forecasts",
                "adversarial": "Stock price decreases if and only if quarterly earnings exceed analyst projections",
                "type": "biconditional_inversion",
                "domain": "financial",
                "challenge": "Biconditional logic with market prediction reversal",
                "level": "extreme"
            },
            {
                "query": "System fails catastrophically unless all backup systems remain operational",
                "correct": "Infrastructure collapses disastrously except when every redundant system functions",
                "adversarial": "System fails catastrophically because all backup systems remain operational",
                "type": "conditional_causality",
                "domain": "technical",
                "challenge": "Unless vs because changes system reliability logic",
                "level": "extreme"
            },
            {
                "query": "Research confirms hypothesis was wrong about drug effectiveness in specific population",
                "correct": "Study validates that theory was incorrect regarding medication efficacy in particular group",
                "adversarial": "Research confirms hypothesis was right about drug effectiveness in specific population",
                "type": "nested_negation",
                "domain": "scientific",
                "challenge": "Nested negation about wrongness vs rightness of hypothesis",
                "level": "extreme"
            }
        ]
    
    def create_impossible_level_cases(self) -> List[Dict]:
        """Near-impossible adversarial cases."""
        return [
            {
                "query": "Nobody can prove that this treatment never fails to help some patients sometimes",
                "correct": "No individual can demonstrate this therapy consistently fails to assist certain individuals occasionally",
                "adversarial": "Everybody can prove that this treatment always fails to help all patients never",
                "type": "logical_nightmare",
                "domain": "medical",
                "challenge": "Multiple nested quantifiers and negations creating logical maze",
                "level": "impossible"
            },
            {
                "query": "Contract becomes invalid if either party fails to not violate non-disclosure terms unless waived",
                "correct": "Agreement turns void when any participant doesn't avoid breaching confidentiality unless exempted",
                "adversarial": "Contract becomes valid if either party fails to not violate non-disclosure terms unless waived",
                "type": "triple_negative_conditional",
                "domain": "legal",
                "challenge": "Triple negation with conditional exception",
                "level": "impossible"
            },
            {
                "query": "Portfolio returns are not guaranteed to never sometimes fail to exceed benchmarks",
                "correct": "Investment performance lacks assurance of consistently avoiding occasional underperformance relative to standards",
                "adversarial": "Portfolio returns are guaranteed to always sometimes succeed to exceed benchmarks",
                "type": "temporal_quantifier_paradox",
                "domain": "financial",
                "challenge": "Multiple temporal and negative quantifiers",
                "level": "impossible"
            },
            {
                "query": "Algorithm may not be unable to occasionally succeed in never failing completely",
                "correct": "Computational method might retain ability to sometimes achieve total success",
                "adversarial": "Algorithm may be able to occasionally succeed in always failing completely",
                "type": "modal_negative_cascade",
                "domain": "technical",
                "challenge": "Modal verbs with cascading negations",
                "level": "impossible"
            },
            {
                "query": "No experiment can fail to prove that some hypotheses aren't always incorrect about everything",
                "correct": "Every test must demonstrate certain theories sometimes correctly explain phenomena",
                "adversarial": "All experiments must fail to prove that no hypotheses are never correct about nothing",
                "type": "universal_existential_negation",
                "domain": "scientific",
                "challenge": "Universal and existential quantifiers with negation maze",
                "level": "impossible"
            }
        ]
    
    def create_semantic_traps(self) -> List[Dict]:
        """Semantic traps that fool classical similarity."""
        return [
            {
                "query": "Patient experienced rapid clinical improvement",
                "correct": "Individual showed swift medical progress and recovery",
                "adversarial": "Patient experienced rapid clinical deterioration",
                "type": "semantic_trap",
                "domain": "medical",
                "challenge": "High lexical overlap, opposite medical outcomes",
                "level": "expert"
            },
            {
                "query": "Evidence strongly supports defendant's alibi claim",
                "correct": "Proof robustly validates accused person's location defense",
                "adversarial": "Evidence strongly contradicts defendant's alibi claim",
                "type": "semantic_trap",
                "domain": "legal",
                "challenge": "Same evidence structure, opposite legal conclusion",
                "level": "expert"
            },
            {
                "query": "Market trends indicate bullish investor sentiment",
                "correct": "Trading patterns suggest optimistic market participant attitudes",
                "adversarial": "Market trends indicate bearish investor sentiment",
                "type": "semantic_trap",
                "domain": "financial",
                "challenge": "Identical structure, opposite market direction",
                "level": "expert"
            },
            {
                "query": "Server response time significantly improved",
                "correct": "System latency performance substantially enhanced",
                "adversarial": "Server response time significantly degraded",
                "type": "semantic_trap",
                "domain": "technical",
                "challenge": "Performance metric with opposite direction",
                "level": "expert"
            },
            {
                "query": "Experimental results strongly confirm theoretical predictions",
                "correct": "Research findings robustly validate scientific forecasts",
                "adversarial": "Experimental results strongly refute theoretical predictions",
                "type": "semantic_trap",
                "domain": "scientific",
                "challenge": "Scientific validation vs refutation with high overlap",
                "level": "expert"
            }
        ]
    
    def create_ultimate_dataset(self) -> List[Dict]:
        """Create the ultimate adversarial dataset."""
        dataset = []
        dataset.extend(self.create_expert_level_cases())
        dataset.extend(self.create_extreme_level_cases())
        dataset.extend(self.create_impossible_level_cases())
        dataset.extend(self.create_semantic_traps())
        return dataset

class UltimateAdversarialEvaluator:
    """Ultimate adversarial evaluator with comprehensive metrics."""
    
    def __init__(self):
        self.dataset_creator = UltimateAdversarialDataset()
        
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
    
    def evaluate_ultimate_trial(self, reranker, scenarios: List[Dict], method: str) -> Dict:
        """Evaluate single trial on ultimate adversarial dataset."""
        results_by_level = defaultdict(list)
        results_by_domain = defaultdict(list)
        results_by_type = defaultdict(list)
        all_results = []
        scenario_details = []
        
        for i, scenario in enumerate(scenarios):
            query = scenario["query"]
            correct_doc = scenario["correct"]
            adversarial_doc = scenario["adversarial"]
            level = scenario["level"]
            domain = scenario["domain"]
            challenge_type = scenario["type"]
            challenge = scenario["challenge"]
            
            candidates = [correct_doc, adversarial_doc]
            
            try:
                results = reranker.rerank(query, candidates, method=method, top_k=2)
                
                if results and len(results) >= 2:
                    top_doc = results[0].get('text', results[0].get('content', ''))
                    success = (top_doc == correct_doc or correct_doc in top_doc)
                else:
                    success = False
                
                results_by_level[level].append(success)
                results_by_domain[domain].append(success)
                results_by_type[challenge_type].append(success)
                all_results.append(success)
                
                scenario_details.append({
                    "scenario_id": i,
                    "level": level,
                    "domain": domain,
                    "type": challenge_type,
                    "challenge": challenge,
                    "success": success,
                    "query": query[:50] + "..." if len(query) > 50 else query
                })
                
            except Exception as e:
                print(f"Error in ultimate reranking: {e}")
                results_by_level[level].append(False)
                results_by_domain[domain].append(False)
                results_by_type[challenge_type].append(False)
                all_results.append(False)
                
                scenario_details.append({
                    "scenario_id": i,
                    "level": level,
                    "domain": domain,
                    "type": challenge_type,
                    "challenge": challenge,
                    "success": False,
                    "error": str(e),
                    "query": query[:50] + "..." if len(query) > 50 else query
                })
        
        # Calculate accuracies
        level_accuracies = {level: np.mean(scores) for level, scores in results_by_level.items()}
        domain_accuracies = {domain: np.mean(scores) for domain, scores in results_by_domain.items()}
        type_accuracies = {challenge_type: np.mean(scores) for challenge_type, scores in results_by_type.items()}
        overall_accuracy = np.mean(all_results)
        
        return {
            "overall_accuracy": overall_accuracy,
            "level_accuracies": level_accuracies,
            "domain_accuracies": domain_accuracies,
            "type_accuracies": type_accuracies,
            "scenario_details": scenario_details,
            "total_scenarios": len(scenarios)
        }
    
    def run_ultimate_evaluation(self, scenarios: List[Dict], n_trials: int = 7, model_path: str = None) -> Dict:
        """Run ultimate evaluation with multiple trials."""
        print(f"🎯 Running Ultimate Evaluation ({n_trials} trials)")
        
        quantum_reranker = self.create_quantum_reranker(model_path)
        classical_reranker = self.create_classical_baseline()
        
        quantum_results = []
        classical_results = []
        
        for trial in range(n_trials):
            print(f"  🚀 Ultimate Trial {trial + 1}/{n_trials}")
            
            q_result = self.evaluate_ultimate_trial(quantum_reranker, scenarios, "quantum")
            c_result = self.evaluate_ultimate_trial(classical_reranker, scenarios, "classical")
            
            quantum_results.append(q_result)
            classical_results.append(c_result)
            
            # Print brief progress
            q_acc = q_result["overall_accuracy"]
            c_acc = c_result["overall_accuracy"]
            advantage = q_acc - c_acc
            print(f"    Q: {q_acc:.1%}, C: {c_acc:.1%}, Adv: {advantage:+.1%}")
        
        return {
            "quantum_results": quantum_results,
            "classical_results": classical_results,
            "n_trials": n_trials
        }
    
    def analyze_ultimate_performance(self, quantum_results: List[Dict], classical_results: List[Dict]) -> Dict:
        """Analyze ultimate adversarial performance."""
        # Overall statistics
        q_overall = [result["overall_accuracy"] for result in quantum_results]
        c_overall = [result["overall_accuracy"] for result in classical_results]
        
        # Statistical tests
        t_stat, p_value = stats.ttest_ind(q_overall, c_overall)
        
        # Effect size
        pooled_std = np.sqrt(((np.std(q_overall, ddof=1) ** 2 + np.std(c_overall, ddof=1) ** 2) / 2))
        cohens_d = (np.mean(q_overall) - np.mean(c_overall)) / pooled_std if pooled_std > 0 else 0
        
        overall_stats = {
            "quantum_mean": np.mean(q_overall),
            "quantum_std": np.std(q_overall, ddof=1),
            "classical_mean": np.mean(c_overall),
            "classical_std": np.std(c_overall, ddof=1),
            "advantage": np.mean(q_overall) - np.mean(c_overall),
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohens_d": cohens_d
        }
        
        # Level-wise analysis
        levels = quantum_results[0]["level_accuracies"].keys()
        level_stats = {}
        
        for level in levels:
            q_level_scores = [result["level_accuracies"][level] for result in quantum_results]
            c_level_scores = [result["level_accuracies"][level] for result in classical_results]
            
            level_t_stat, level_p_value = stats.ttest_ind(q_level_scores, c_level_scores)
            
            level_stats[level] = {
                "quantum_mean": np.mean(q_level_scores),
                "classical_mean": np.mean(c_level_scores),
                "advantage": np.mean(q_level_scores) - np.mean(c_level_scores),
                "quantum_wins": sum(1 for q, c in zip(q_level_scores, c_level_scores) if q > c),
                "total_trials": len(q_level_scores),
                "p_value": level_p_value
            }
        
        # Domain analysis
        domains = quantum_results[0]["domain_accuracies"].keys()
        domain_stats = {}
        
        for domain in domains:
            q_domain_scores = [result["domain_accuracies"][domain] for result in quantum_results]
            c_domain_scores = [result["domain_accuracies"][domain] for result in classical_results]
            
            domain_stats[domain] = {
                "quantum_mean": np.mean(q_domain_scores),
                "classical_mean": np.mean(c_domain_scores),
                "advantage": np.mean(q_domain_scores) - np.mean(c_domain_scores)
            }
        
        return {
            "overall": overall_stats,
            "by_level": level_stats,
            "by_domain": domain_stats
        }
    
    def run_comprehensive_ultimate_evaluation(self, model_path: str = None) -> Dict:
        """Run the ultimate comprehensive evaluation."""
        print("🏆 ULTIMATE ADVERSARIAL ROBUSTNESS EVALUATION")
        print("=" * 60)
        print("The definitive test of quantum vs classical semantic robustness")
        
        # Create ultimate dataset
        scenarios = self.dataset_creator.create_ultimate_dataset()
        print(f"📊 Generated {len(scenarios)} ultimate adversarial scenarios")
        
        levels = set(s['level'] for s in scenarios)
        domains = set(s['domain'] for s in scenarios)
        types = set(s['type'] for s in scenarios)
        
        print(f"🏷️  Challenge levels: {', '.join(sorted(levels))}")
        print(f"🏢 Domains: {', '.join(sorted(domains))}")
        print(f"🔄 Challenge types: {', '.join(sorted(types))}")
        
        # Show example challenges
        print(f"\n🔍 EXAMPLE ULTIMATE CHALLENGES:")
        for level in ["expert", "extreme", "impossible"]:
            example = next((s for s in scenarios if s['level'] == level), None)
            if example:
                print(f"  • {level.upper()}: {example['challenge']}")
        
        # Run evaluation
        print(f"\n🚀 RUNNING ULTIMATE EVALUATION")
        print("-" * 40)
        start_time = time.time()
        
        results = self.run_ultimate_evaluation(scenarios, n_trials=7, model_path=model_path)
        analysis = self.analyze_ultimate_performance(
            results["quantum_results"], 
            results["classical_results"]
        )
        
        eval_time = time.time() - start_time
        
        return {
            "ultimate_evaluation": results,
            "performance_analysis": analysis,
            "scenarios": scenarios,
            "metadata": {
                "total_scenarios": len(scenarios),
                "challenge_levels": list(levels),
                "domains": list(domains),
                "challenge_types": list(types),
                "evaluation_time": eval_time
            }
        }

def main():
    """Main function for ultimate adversarial testing."""
    print("🏆 Starting Ultimate Adversarial Robustness Test")
    print("The definitive evaluation of quantum semantic understanding")
    
    # Check for trained model
    model_path = "models/qpmel_extended.pt"
    if not Path(model_path).exists():
        print(f"⚠️  Using default quantum parameters")
        model_path = None
    else:
        print(f"✅ Using trained QPMeL model")
    
    # Run ultimate evaluation
    evaluator = UltimateAdversarialEvaluator()
    
    print(f"\n⏱️  Starting ultimate evaluation...")
    total_start_time = time.time()
    
    results = evaluator.run_comprehensive_ultimate_evaluation(model_path)
    
    total_time = time.time() - total_start_time
    
    # Print ultimate results
    print(f"\n🏆 ULTIMATE RESULTS")
    print("=" * 60)
    
    overall = results["performance_analysis"]["overall"]
    print(f"\n🎯 ULTIMATE PERFORMANCE:")
    print(f"  Quantum:   {overall['quantum_mean']:.1%} ± {overall['quantum_std']:.1%}")
    print(f"  Classical: {overall['classical_mean']:.1%} ± {overall['classical_std']:.1%}")
    print(f"  Advantage: {overall['advantage']:+.1%}")
    print(f"  P-value:   {overall['p_value']:.2e}")
    print(f"  Effect:    {overall['cohens_d']:.2f} (Cohen's d)")
    
    print(f"\n🏷️  BY CHALLENGE LEVEL:")
    level_stats = results["performance_analysis"]["by_level"]
    print(f"{'Level':<12} {'Quantum':<10} {'Classical':<10} {'Advantage':<10} {'Q Wins'}")
    print("-" * 55)
    for level, stats in level_stats.items():
        wins = f"{stats['quantum_wins']}/{stats['total_trials']}"
        print(f"{level:<12} {stats['quantum_mean']:<10.1%} {stats['classical_mean']:<10.1%} "
              f"{stats['advantage']:+.1%}{'':>4} {wins}")
    
    print(f"\n🏢 BY DOMAIN:")
    domain_stats = results["performance_analysis"]["by_domain"]
    for domain, stats in domain_stats.items():
        print(f"  {domain:<12}: Q={stats['quantum_mean']:.1%}, C={stats['classical_mean']:.1%}, "
              f"Adv={stats['advantage']:+.1%}")
    
    # Save results
    output_data = {
        "test_name": "ultimate_adversarial_robustness",
        "total_time_seconds": total_time,
        "results": results,
        "timestamp": time.time()
    }
    
    output_file = "ultimate_adversarial_results.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to {output_file}")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    
    # Ultimate verdict
    advantage = overall['advantage']
    p_value = overall['p_value']
    effect_size = abs(overall['cohens_d'])
    
    print(f"\n🏆 ULTIMATE VERDICT:")
    print("=" * 40)
    
    if advantage > 0.15 and p_value < 0.001 and effect_size > 2.0:
        print(f"🚀 QUANTUM SUPREMACY IN ADVERSARIAL ROBUSTNESS!")
        print(f"🎯 {advantage:+.1%} advantage with p={p_value:.2e}")
        print(f"⚡ Effect size: {effect_size:.2f} (extremely large)")
        print(f"🏆 CONCLUSION: Quantum methods demonstrate superior semantic understanding")
        verdict = "QUANTUM_SUPREMACY"
    elif advantage > 0.10 and p_value < 0.01 and effect_size > 1.0:
        print(f"✅ STRONG QUANTUM ADVANTAGE CONFIRMED!")
        print(f"🎯 {advantage:+.1%} advantage with p={p_value:.2e}")
        print(f"📈 Effect size: {effect_size:.2f} (large)")
        print(f"🎖️  CONCLUSION: Quantum shows significant robustness benefits")
        verdict = "STRONG_QUANTUM_ADVANTAGE"
    elif advantage > 0.05 and p_value < 0.05:
        print(f"📈 MODERATE QUANTUM ADVANTAGE FOUND")
        print(f"🎯 {advantage:+.1%} advantage with p={p_value:.2e}")
        print(f"📊 CONCLUSION: Quantum provides meaningful improvements")
        verdict = "MODERATE_QUANTUM_ADVANTAGE"
    else:
        print(f"📊 MIXED OR NO QUANTUM ADVANTAGE")
        print(f"🎯 {advantage:+.1%} advantage with p={p_value:.2e}")
        print(f"🔍 CONCLUSION: Requires further investigation")
        verdict = "INCONCLUSIVE"
    
    print(f"\n✨ Ultimate Adversarial Test Complete!")
    return verdict

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Expanded Adversarial Robustness Test Suite
Testing quantum advantage across multiple domains and perturbation types.
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
import itertools

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum_rerank.core.rag_reranker import QuantumRAGReranker
from quantum_rerank.core.quantum_similarity_engine import SimilarityEngineConfig, SimilarityMethod
from quantum_rerank.training.qpmel_trainer import QPMeLTrainer, QPMeLTrainingConfig
from quantum_rerank.ml.qpmel_circuits import QPMeLConfig

class ExpandedAdversarialGenerator:
    """Advanced adversarial text generator with multiple domains and perturbation types."""
    
    def __init__(self):
        self.domains = ["medical", "legal", "financial", "technical", "news", "scientific"]
        self.perturbation_types = [
            "negation", "antonym_swap", "entity_swap", "temporal_shift", 
            "causal_reversal", "scope_modification", "intensity_change",
            "modal_flip", "context_shift", "logical_connective_swap"
        ]
        
    def create_medical_adversarial_dataset(self) -> List[Dict]:
        """Medical domain adversarial examples."""
        return [
            {
                "query": "Patient presents with acute chest pain and shortness of breath",
                "correct": "Emergency evaluation for myocardial infarction and pulmonary embolism indicated",
                "adversarial": "Patient presents with chronic chest pain and shortness of breath",
                "type": "temporal_shift",
                "domain": "medical",
                "difficulty": "medium"
            },
            {
                "query": "Medication reduces inflammation in rheumatoid arthritis patients",
                "correct": "Anti-inflammatory therapy effective for rheumatoid arthritis management",
                "adversarial": "Medication increases inflammation in rheumatoid arthritis patients",
                "type": "antonym_swap",
                "domain": "medical", 
                "difficulty": "easy"
            },
            {
                "query": "Surgery must be performed before chemotherapy for optimal outcomes",
                "correct": "Surgical intervention should precede chemotherapy treatment",
                "adversarial": "Surgery must be performed after chemotherapy for optimal outcomes",
                "type": "causal_reversal",
                "domain": "medical",
                "difficulty": "hard"
            },
            {
                "query": "High doses of the drug cause severe side effects",
                "correct": "Elevated medication dosages result in significant adverse reactions",
                "adversarial": "Low doses of the drug cause severe side effects",
                "type": "intensity_change",
                "domain": "medical",
                "difficulty": "medium"
            },
            {
                "query": "All patients in the study showed improvement",
                "correct": "Every participant demonstrated clinical progress in the trial",
                "adversarial": "Some patients in the study showed improvement",
                "type": "scope_modification",
                "domain": "medical",
                "difficulty": "hard"
            }
        ]
    
    def create_legal_adversarial_dataset(self) -> List[Dict]:
        """Legal domain adversarial examples."""
        return [
            {
                "query": "Contract is valid only if signed by both parties",
                "correct": "Agreement requires signatures from all contracting parties for validity",
                "adversarial": "Contract is valid even if signed by one party",
                "type": "modal_flip",
                "domain": "legal",
                "difficulty": "hard"
            },
            {
                "query": "Evidence was obtained legally through proper warrant",
                "correct": "Lawful evidence collection via authorized search warrant",
                "adversarial": "Evidence was obtained illegally through proper warrant",
                "type": "antonym_swap",
                "domain": "legal",
                "difficulty": "easy"
            },
            {
                "query": "Defendant must pay damages before appeal can proceed",
                "correct": "Monetary compensation required prior to appellate process",
                "adversarial": "Defendant must pay damages after appeal can proceed",
                "type": "temporal_shift",
                "domain": "legal",
                "difficulty": "medium"
            },
            {
                "query": "Plaintiff has strong evidence supporting their claim",
                "correct": "Claimant possesses compelling proof for their case",
                "adversarial": "Plaintiff has weak evidence supporting their claim",
                "type": "intensity_change",
                "domain": "legal",
                "difficulty": "medium"
            },
            {
                "query": "Judge ruled in favor of the plaintiff because evidence was convincing",
                "correct": "Court decided for claimant due to persuasive evidence",
                "adversarial": "Judge ruled in favor of the plaintiff although evidence was convincing",
                "type": "logical_connective_swap",
                "domain": "legal",
                "difficulty": "hard"
            }
        ]
    
    def create_financial_adversarial_dataset(self) -> List[Dict]:
        """Financial domain adversarial examples."""
        return [
            {
                "query": "Stock prices will increase due to positive earnings report",
                "correct": "Equity values expected to rise following favorable financial results",
                "adversarial": "Stock prices will decrease due to positive earnings report",
                "type": "antonym_swap",
                "domain": "financial",
                "difficulty": "easy"
            },
            {
                "query": "Investment carries high risk but offers substantial returns",
                "correct": "High-risk investment opportunity with significant profit potential",
                "adversarial": "Investment carries low risk but offers substantial returns",
                "type": "intensity_change",
                "domain": "financial",
                "difficulty": "medium"
            },
            {
                "query": "Market crashed before the federal announcement",
                "correct": "Financial markets declined prior to government statement",
                "adversarial": "Market crashed after the federal announcement",
                "type": "temporal_shift",
                "domain": "financial",
                "difficulty": "medium"
            },
            {
                "query": "All investors in the fund lost money during the crisis",
                "correct": "Every fund participant experienced losses throughout the downturn",
                "adversarial": "Most investors in the fund lost money during the crisis",
                "type": "scope_modification",
                "domain": "financial",
                "difficulty": "hard"
            },
            {
                "query": "Company profits increased because of new product launch",
                "correct": "Corporate earnings rose due to product introduction",
                "adversarial": "Company profits increased despite new product launch",
                "type": "logical_connective_swap",
                "domain": "financial",
                "difficulty": "hard"
            }
        ]
    
    def create_technical_adversarial_dataset(self) -> List[Dict]:
        """Technical/Engineering domain adversarial examples."""
        return [
            {
                "query": "System performance improves with increased memory allocation",
                "correct": "Enhanced memory resources boost system efficiency",
                "adversarial": "System performance degrades with increased memory allocation",
                "type": "antonym_swap",
                "domain": "technical",
                "difficulty": "easy"
            },
            {
                "query": "Algorithm must process data before encryption can occur",
                "correct": "Data processing required prior to cryptographic operations",
                "adversarial": "Algorithm must process data after encryption can occur",
                "type": "temporal_shift",
                "domain": "technical",
                "difficulty": "medium"
            },
            {
                "query": "Network latency is extremely high during peak usage",
                "correct": "Connection delays are severely elevated at maximum load",
                "adversarial": "Network latency is moderately high during peak usage",
                "type": "intensity_change",
                "domain": "technical",
                "difficulty": "medium"
            },
            {
                "query": "All servers in the cluster failed simultaneously",
                "correct": "Every machine in the computing cluster experienced concurrent failure",
                "adversarial": "Some servers in the cluster failed simultaneously",
                "type": "scope_modification",
                "domain": "technical",
                "difficulty": "hard"
            },
            {
                "query": "Database backup succeeded because storage space was available",
                "correct": "Data backup completed due to sufficient storage capacity",
                "adversarial": "Database backup succeeded although storage space was unavailable",
                "type": "logical_connective_swap",
                "domain": "technical",
                "difficulty": "hard"
            }
        ]
    
    def create_news_adversarial_dataset(self) -> List[Dict]:
        """News/Current events domain adversarial examples."""
        return [
            {
                "query": "Election results show incumbent candidate winning decisively",
                "correct": "Voting outcomes demonstrate current officeholder's clear victory",
                "adversarial": "Election results show challenger candidate winning decisively",
                "type": "entity_swap",
                "domain": "news",
                "difficulty": "medium"
            },
            {
                "query": "Unemployment rates decreased significantly this quarter",
                "correct": "Jobless statistics dropped substantially in recent months",
                "adversarial": "Unemployment rates increased significantly this quarter",
                "type": "antonym_swap",
                "domain": "news",
                "difficulty": "easy"
            },
            {
                "query": "Peace talks began after the ceasefire was declared",
                "correct": "Diplomatic negotiations started following armistice announcement",
                "adversarial": "Peace talks began before the ceasefire was declared",
                "type": "temporal_shift",
                "domain": "news",
                "difficulty": "medium"
            },
            {
                "query": "Weather conditions are severely dangerous for travel",
                "correct": "Meteorological circumstances pose extreme transportation hazards",
                "adversarial": "Weather conditions are mildly dangerous for travel",
                "type": "intensity_change",
                "domain": "news",
                "difficulty": "medium"
            },
            {
                "query": "Every region reported flooding after the storm",
                "correct": "All areas experienced inundation following severe weather",
                "adversarial": "Several regions reported flooding after the storm",
                "type": "scope_modification",
                "domain": "news",
                "difficulty": "hard"
            }
        ]
    
    def create_scientific_adversarial_dataset(self) -> List[Dict]:
        """Scientific research domain adversarial examples."""
        return [
            {
                "query": "Experiment results confirm the hypothesis with high confidence",
                "correct": "Research findings strongly support the theoretical prediction",
                "adversarial": "Experiment results refute the hypothesis with high confidence",
                "type": "antonym_swap",
                "domain": "scientific",
                "difficulty": "easy"
            },
            {
                "query": "Temperature must be increased before chemical reaction occurs",
                "correct": "Thermal elevation required prior to chemical process initiation",
                "adversarial": "Temperature must be increased after chemical reaction occurs",
                "type": "temporal_shift",
                "domain": "scientific",
                "difficulty": "medium"
            },
            {
                "query": "Sample concentration is extremely high in the solution",
                "correct": "Analyte levels are significantly elevated in the mixture",
                "adversarial": "Sample concentration is slightly high in the solution",
                "type": "intensity_change",
                "domain": "scientific",
                "difficulty": "medium"
            },
            {
                "query": "All specimens showed positive results for the marker",
                "correct": "Every sample demonstrated marker presence in testing",
                "adversarial": "Most specimens showed positive results for the marker",
                "type": "scope_modification",
                "domain": "scientific",
                "difficulty": "hard"
            },
            {
                "query": "Protein folded correctly because environment was optimal",
                "correct": "Molecular structure formed properly due to ideal conditions",
                "adversarial": "Protein folded correctly despite environment being suboptimal",
                "type": "logical_connective_swap",
                "domain": "scientific",
                "difficulty": "hard"
            }
        ]
    
    def inject_advanced_noise(self, text: str, noise_type: str, intensity: float = 0.1) -> str:
        """Inject various types of advanced noise."""
        words = text.split()
        if not words:
            return text
            
        if noise_type == "typo":
            return self._inject_typos(words, intensity)
        elif noise_type == "homophone":
            return self._inject_homophones(words, intensity)
        elif noise_type == "semantic_drift":
            return self._inject_semantic_drift(words, intensity)
        elif noise_type == "syntactic_error":
            return self._inject_syntactic_errors(words, intensity)
        elif noise_type == "word_reorder":
            return self._inject_word_reordering(words, intensity)
        else:
            return text
    
    def _inject_typos(self, words: List[str], intensity: float) -> str:
        """Inject realistic typos."""
        typo_patterns = {
            'a': ['e', 's'], 'e': ['a', 'r'], 'i': ['o', 'u'], 'o': ['i', 'p'],
            'u': ['i', 'y'], 's': ['a', 'd'], 'd': ['s', 'f'], 'f': ['d', 'g'],
            'g': ['f', 'h'], 'h': ['g', 'j'], 'j': ['h', 'k'], 'k': ['j', 'l'],
            'l': ['k', ';'], 'r': ['e', 't'], 't': ['r', 'y'], 'y': ['t', 'u']
        }
        
        num_changes = max(1, int(len(words) * intensity))
        for _ in range(num_changes):
            word_idx = random.randint(0, len(words) - 1)
            word = words[word_idx]
            if len(word) > 2:
                char_idx = random.randint(0, len(word) - 1)
                char = word[char_idx].lower()
                if char in typo_patterns:
                    replacement = random.choice(typo_patterns[char])
                    words[word_idx] = word[:char_idx] + replacement + word[char_idx+1:]
        
        return ' '.join(words)
    
    def _inject_homophones(self, words: List[str], intensity: float) -> str:
        """Replace words with homophones."""
        homophones = {
            'there': ['their', 'they\'re'], 'to': ['too', 'two'],
            'your': ['you\'re'], 'its': ['it\'s'], 'than': ['then'],
            'accept': ['except'], 'effect': ['affect'], 'lose': ['loose'],
            'principal': ['principle'], 'stationary': ['stationery']
        }
        
        num_changes = max(1, int(len(words) * intensity))
        for _ in range(num_changes):
            word_idx = random.randint(0, len(words) - 1)
            word = words[word_idx].lower()
            if word in homophones:
                replacement = random.choice(homophones[word])
                words[word_idx] = replacement
        
        return ' '.join(words)
    
    def _inject_semantic_drift(self, words: List[str], intensity: float) -> str:
        """Replace words with semantically similar but contextually wrong words."""
        semantic_pairs = {
            'increase': ['decrease', 'change'], 'improve': ['worsen', 'modify'],
            'effective': ['ineffective', 'different'], 'high': ['low', 'varying'],
            'before': ['after', 'during'], 'because': ['although', 'while'],
            'all': ['some', 'many'], 'must': ['should', 'might']
        }
        
        num_changes = max(1, int(len(words) * intensity))
        for _ in range(num_changes):
            word_idx = random.randint(0, len(words) - 1)
            word = words[word_idx].lower()
            if word in semantic_pairs:
                replacement = random.choice(semantic_pairs[word])
                words[word_idx] = replacement
        
        return ' '.join(words)
    
    def _inject_syntactic_errors(self, words: List[str], intensity: float) -> str:
        """Inject syntactic errors."""
        if len(words) < 3:
            return ' '.join(words)
            
        num_changes = max(1, int(len(words) * intensity))
        for _ in range(num_changes):
            error_type = random.choice(['subject_verb', 'article', 'preposition'])
            
            if error_type == 'subject_verb':
                # Find and modify verb forms
                for i, word in enumerate(words):
                    if word.endswith('s') and len(word) > 3:
                        words[i] = word[:-1]  # Remove 's' from verb
                        break
            elif error_type == 'article':
                # Add/remove articles
                if random.random() < 0.5:
                    words.insert(random.randint(0, len(words)), random.choice(['the', 'a', 'an']))
                else:
                    articles = ['the', 'a', 'an']
                    for i, word in enumerate(words):
                        if word.lower() in articles:
                            words.pop(i)
                            break
            elif error_type == 'preposition':
                # Replace prepositions
                prepositions = ['in', 'on', 'at', 'by', 'for', 'with', 'from', 'to']
                for i, word in enumerate(words):
                    if word.lower() in prepositions:
                        words[i] = random.choice(prepositions)
                        break
        
        return ' '.join(words)
    
    def _inject_word_reordering(self, words: List[str], intensity: float) -> str:
        """Reorder words to create syntactic confusion."""
        if len(words) < 4:
            return ' '.join(words)
            
        num_swaps = max(1, int(len(words) * intensity / 2))
        for _ in range(num_swaps):
            idx1 = random.randint(0, len(words) - 1)
            idx2 = random.randint(0, len(words) - 1)
            if idx1 != idx2:
                words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def create_comprehensive_dataset(self) -> List[Dict]:
        """Create comprehensive multi-domain adversarial dataset."""
        dataset = []
        dataset.extend(self.create_medical_adversarial_dataset())
        dataset.extend(self.create_legal_adversarial_dataset())
        dataset.extend(self.create_financial_adversarial_dataset())
        dataset.extend(self.create_technical_adversarial_dataset())
        dataset.extend(self.create_news_adversarial_dataset())
        dataset.extend(self.create_scientific_adversarial_dataset())
        return dataset

class ExpandedAdversarialEvaluator:
    """Comprehensive adversarial robustness evaluator."""
    
    def __init__(self):
        self.generator = ExpandedAdversarialGenerator()
        
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
    
    def evaluate_single_trial(self, reranker, scenarios: List[Dict], method: str) -> Dict:
        """Evaluate single trial of adversarial robustness."""
        results_by_domain = defaultdict(list)
        results_by_type = defaultdict(list)
        results_by_difficulty = defaultdict(list)
        
        for scenario in scenarios:
            query = scenario["query"]
            correct_doc = scenario["correct"]
            adversarial_doc = scenario["adversarial"]
            domain = scenario["domain"]
            pert_type = scenario["type"]
            difficulty = scenario["difficulty"]
            
            candidates = [correct_doc, adversarial_doc]
            
            try:
                results = reranker.rerank(query, candidates, method=method, top_k=2)
                
                if results and len(results) >= 2:
                    top_doc = results[0].get('text', results[0].get('content', ''))
                    success = (top_doc == correct_doc or correct_doc in top_doc)
                else:
                    success = False
                    
                results_by_domain[domain].append(success)
                results_by_type[pert_type].append(success)
                results_by_difficulty[difficulty].append(success)
                
            except Exception as e:
                print(f"Error in reranking: {e}")
                results_by_domain[domain].append(False)
                results_by_type[pert_type].append(False)
                results_by_difficulty[difficulty].append(False)
        
        # Calculate accuracies
        domain_accuracies = {domain: np.mean(scores) for domain, scores in results_by_domain.items()}
        type_accuracies = {pert_type: np.mean(scores) for pert_type, scores in results_by_type.items()}
        difficulty_accuracies = {difficulty: np.mean(scores) for difficulty, scores in results_by_difficulty.items()}
        overall_accuracy = np.mean([score for scores in results_by_domain.values() for score in scores])
        
        return {
            "overall_accuracy": overall_accuracy,
            "domain_accuracies": domain_accuracies,
            "type_accuracies": type_accuracies,
            "difficulty_accuracies": difficulty_accuracies,
            "total_scenarios": len(scenarios)
        }
    
    def run_multi_trial_evaluation(self, scenarios: List[Dict], n_trials: int = 5, model_path: str = None) -> Dict:
        """Run multiple trials for statistical significance."""
        print(f"🔬 Running {n_trials} trials on {len(scenarios)} scenarios")
        
        quantum_reranker = self.create_quantum_reranker(model_path)
        classical_reranker = self.create_classical_baseline()
        
        quantum_results = []
        classical_results = []
        
        for trial in range(n_trials):
            print(f"  Trial {trial + 1}/{n_trials}")
            
            # Quantum trial
            q_result = self.evaluate_single_trial(quantum_reranker, scenarios, "quantum")
            quantum_results.append(q_result)
            
            # Classical trial
            c_result = self.evaluate_single_trial(classical_reranker, scenarios, "classical")
            classical_results.append(c_result)
        
        return {
            "quantum_results": quantum_results,
            "classical_results": classical_results,
            "n_trials": n_trials
        }
    
    def test_noise_robustness_expanded(self, clean_scenarios: List[Dict], model_path: str = None) -> Dict:
        """Test robustness to various noise types."""
        print("🔊 Testing Advanced Noise Robustness")
        
        quantum_reranker = self.create_quantum_reranker(model_path)
        classical_reranker = self.create_classical_baseline()
        
        noise_types = ["typo", "homophone", "semantic_drift", "syntactic_error", "word_reorder"]
        intensity_levels = [0.1, 0.2, 0.3]
        
        results = {}
        
        for noise_type in noise_types:
            print(f"  Testing {noise_type} noise...")
            results[noise_type] = {}
            
            for intensity in intensity_levels:
                # Create noisy scenarios
                noisy_scenarios = []
                for scenario in clean_scenarios[:10]:  # Use subset for speed
                    noisy_scenario = scenario.copy()
                    noisy_scenario["query"] = self.generator.inject_advanced_noise(
                        scenario["query"], noise_type, intensity
                    )
                    noisy_scenario["correct"] = self.generator.inject_advanced_noise(
                        scenario["correct"], noise_type, intensity
                    )
                    noisy_scenarios.append(noisy_scenario)
                
                # Evaluate both methods
                q_result = self.evaluate_single_trial(quantum_reranker, noisy_scenarios, "quantum")
                c_result = self.evaluate_single_trial(classical_reranker, noisy_scenarios, "classical")
                
                results[noise_type][intensity] = {
                    "quantum_accuracy": q_result["overall_accuracy"],
                    "classical_accuracy": c_result["overall_accuracy"],
                    "quantum_advantage": q_result["overall_accuracy"] - c_result["overall_accuracy"]
                }
        
        return results
    
    def analyze_statistical_significance(self, quantum_results: List[Dict], classical_results: List[Dict]) -> Dict:
        """Perform comprehensive statistical analysis."""
        # Extract overall accuracies
        q_accuracies = [result["overall_accuracy"] for result in quantum_results]
        c_accuracies = [result["overall_accuracy"] for result in classical_results]
        
        # Statistical tests
        t_stat, p_value = stats.ttest_ind(q_accuracies, c_accuracies)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((np.std(q_accuracies, ddof=1) ** 2 + np.std(c_accuracies, ddof=1) ** 2) / 2))
        cohens_d = (np.mean(q_accuracies) - np.mean(c_accuracies)) / pooled_std if pooled_std > 0 else 0
        
        # Domain-wise analysis
        domains = quantum_results[0]["domain_accuracies"].keys()
        domain_stats = {}
        
        for domain in domains:
            q_domain_accs = [result["domain_accuracies"][domain] for result in quantum_results]
            c_domain_accs = [result["domain_accuracies"][domain] for result in classical_results]
            
            domain_t_stat, domain_p_value = stats.ttest_ind(q_domain_accs, c_domain_accs)
            domain_effect_size = (np.mean(q_domain_accs) - np.mean(c_domain_accs)) / np.sqrt(
                (np.std(q_domain_accs, ddof=1) ** 2 + np.std(c_domain_accs, ddof=1) ** 2) / 2
            ) if np.std(q_domain_accs, ddof=1) > 0 or np.std(c_domain_accs, ddof=1) > 0 else 0
            
            domain_stats[domain] = {
                "quantum_mean": np.mean(q_domain_accs),
                "classical_mean": np.mean(c_domain_accs),
                "quantum_std": np.std(q_domain_accs, ddof=1),
                "classical_std": np.std(c_domain_accs, ddof=1),
                "p_value": domain_p_value,
                "effect_size": domain_effect_size,
                "advantage": np.mean(q_domain_accs) - np.mean(c_domain_accs)
            }
        
        return {
            "overall": {
                "quantum_mean": np.mean(q_accuracies),
                "quantum_std": np.std(q_accuracies, ddof=1),
                "classical_mean": np.mean(c_accuracies),
                "classical_std": np.std(c_accuracies, ddof=1),
                "t_statistic": t_stat,
                "p_value": p_value,
                "cohens_d": cohens_d,
                "advantage": np.mean(q_accuracies) - np.mean(c_accuracies)
            },
            "by_domain": domain_stats
        }
    
    def run_comprehensive_evaluation(self, model_path: str = None) -> Dict:
        """Run comprehensive expanded adversarial evaluation."""
        print("🚀 EXPANDED ADVERSARIAL ROBUSTNESS EVALUATION")
        print("=" * 60)
        
        # Create comprehensive dataset
        scenarios = self.generator.create_comprehensive_dataset()
        print(f"📊 Generated {len(scenarios)} scenarios across {len(set(s['domain'] for s in scenarios))} domains")
        
        domains = set(s['domain'] for s in scenarios)
        types = set(s['type'] for s in scenarios)
        print(f"🏷️  Domains: {', '.join(sorted(domains))}")
        print(f"🔄 Perturbation types: {', '.join(sorted(types))}")
        
        # Run multi-trial evaluation
        print(f"\n🔬 MAIN EVALUATION")
        print("-" * 40)
        start_time = time.time()
        
        multi_trial_results = self.run_multi_trial_evaluation(scenarios, n_trials=5, model_path=model_path)
        
        main_eval_time = time.time() - start_time
        
        # Statistical analysis
        print(f"\n📈 STATISTICAL ANALYSIS")
        print("-" * 40)
        
        stats_analysis = self.analyze_statistical_significance(
            multi_trial_results["quantum_results"],
            multi_trial_results["classical_results"]
        )
        
        # Noise robustness testing
        print(f"\n🔊 NOISE ROBUSTNESS TESTING")
        print("-" * 40)
        start_time = time.time()
        
        noise_results = self.test_noise_robustness_expanded(scenarios, model_path)
        
        noise_eval_time = time.time() - start_time
        
        return {
            "main_evaluation": multi_trial_results,
            "statistical_analysis": stats_analysis,
            "noise_robustness": noise_results,
            "metadata": {
                "total_scenarios": len(scenarios),
                "domains": list(domains),
                "perturbation_types": list(types),
                "main_eval_time": main_eval_time,
                "noise_eval_time": noise_eval_time
            }
        }

def main():
    """Main function for expanded adversarial testing."""
    print("🚀 Starting Expanded Adversarial Robustness Test Suite")
    print("Testing quantum advantage across multiple domains and perturbation types")
    
    # Check for trained model
    model_path = "models/qpmel_extended.pt"
    if not Path(model_path).exists():
        print(f"⚠️  Using default quantum parameters")
        model_path = None
    else:
        print(f"✅ Using trained QPMeL model")
    
    # Run comprehensive evaluation
    evaluator = ExpandedAdversarialEvaluator()
    
    print(f"\n⏱️  Starting comprehensive evaluation...")
    total_start_time = time.time()
    
    results = evaluator.run_comprehensive_evaluation(model_path)
    
    total_time = time.time() - total_start_time
    
    # Print comprehensive results
    print(f"\n📊 COMPREHENSIVE RESULTS")
    print("=" * 60)
    
    stats = results["statistical_analysis"]["overall"]
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"  Quantum:  {stats['quantum_mean']:.1%} ± {stats['quantum_std']:.1%}")
    print(f"  Classical: {stats['classical_mean']:.1%} ± {stats['classical_std']:.1%}")
    print(f"  Advantage: {stats['advantage']:+.1%}")
    print(f"  P-value:   {stats['p_value']:.2e}")
    print(f"  Effect size: {stats['cohens_d']:.2f}")
    
    print(f"\n🏷️  BY DOMAIN:")
    domain_stats = results["statistical_analysis"]["by_domain"]
    print(f"{'Domain':<12} {'Quantum':<10} {'Classical':<10} {'Advantage':<10} {'P-value'}")
    print("-" * 55)
    for domain, stats in domain_stats.items():
        print(f"{domain:<12} {stats['quantum_mean']:<10.1%} {stats['classical_mean']:<10.1%} "
              f"{stats['advantage']:+.1%}{'':>4} {stats['p_value']:.2e}")
    
    print(f"\n🔊 NOISE ROBUSTNESS SUMMARY:")
    noise_results = results["noise_robustness"]
    for noise_type, intensities in noise_results.items():
        avg_advantage = np.mean([data["quantum_advantage"] for data in intensities.values()])
        print(f"  {noise_type:<15}: {avg_advantage:+.1%} average advantage")
    
    # Save results
    output_data = {
        "test_name": "expanded_adversarial_robustness",
        "total_time_seconds": total_time,
        "results": results,
        "timestamp": time.time()
    }
    
    output_file = "expanded_adversarial_results.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to {output_file}")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    
    # Final verdict
    overall_advantage = stats['advantage']
    significant = stats['p_value'] < 0.05
    large_effect = abs(stats.get('cohens_d', 0)) > 0.8
    
    print(f"\n🏆 FINAL VERDICT:")
    print("=" * 30)
    
    if overall_advantage > 0.05 and significant and large_effect:
        print(f"✅ STRONG QUANTUM ADVANTAGE CONFIRMED")
        print(f"🎯 {overall_advantage:+.1%} advantage with p={stats['p_value']:.2e}")
        print(f"🔬 Effect size: {stats['cohens_d']:.2f} (large)")
        verdict = "STRONG_QUANTUM_ADVANTAGE"
    elif overall_advantage > 0.02 and significant:
        print(f"✅ MODERATE QUANTUM ADVANTAGE FOUND")
        print(f"🎯 {overall_advantage:+.1%} advantage with p={stats['p_value']:.2e}")
        verdict = "MODERATE_QUANTUM_ADVANTAGE"
    else:
        print(f"❌ NO SIGNIFICANT QUANTUM ADVANTAGE")
        print(f"📊 {overall_advantage:+.1%} advantage with p={stats['p_value']:.2e}")
        verdict = "NO_ADVANTAGE"
    
    print(f"\n✨ Expanded Adversarial Test Complete!")
    return verdict

if __name__ == "__main__":
    main()
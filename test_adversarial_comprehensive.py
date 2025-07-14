#!/usr/bin/env python3
"""
Comprehensive Adversarial Robustness Test
Rigorous, objective evaluation with larger datasets and statistical analysis.
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

class ComprehensiveAdversarialGenerator:
    """Generates comprehensive adversarial test datasets."""
    
    def __init__(self):
        self.perturbation_types = [
            "negation", "word_order", "contradiction", "synonym_swap", 
            "typo_injection", "word_insertion", "word_deletion", "paraphrase"
        ]
        
    def create_large_adversarial_dataset(self) -> List[Dict]:
        """Create large-scale adversarial dataset with 50+ examples."""
        dataset = []
        
        # Negation examples (10 pairs)
        dataset.extend(self._create_negation_examples())
        
        # Word order examples (10 pairs) 
        dataset.extend(self._create_word_order_examples())
        
        # Contradiction examples (10 pairs)
        dataset.extend(self._create_contradiction_examples())
        
        # Synonym/paraphrase examples (10 pairs)
        dataset.extend(self._create_synonym_examples())
        
        # Typo/noise examples (10 pairs)
        dataset.extend(self._create_typo_examples())
        
        return dataset
    
    def _create_negation_examples(self) -> List[Dict]:
        """Create negation-based adversarial pairs."""
        return [
            {
                "query": "The medication is effective for treating depression",
                "correct": "This drug shows efficacy in depression treatment",
                "adversarial": "The medication is not effective for treating depression",
                "type": "negation",
                "difficulty": "easy"
            },
            {
                "query": "Patients showed significant improvement with therapy",
                "correct": "Therapy resulted in notable patient progress",
                "adversarial": "Patients showed no significant improvement with therapy",
                "type": "negation", 
                "difficulty": "easy"
            },
            {
                "query": "The vaccine prevents disease transmission",
                "correct": "Vaccination blocks disease spread effectively",
                "adversarial": "The vaccine does not prevent disease transmission",
                "type": "negation",
                "difficulty": "medium"
            },
            {
                "query": "Surgery is recommended for this condition",
                "correct": "Surgical intervention is advised for this medical issue",
                "adversarial": "Surgery is not recommended for this condition",
                "type": "negation",
                "difficulty": "medium"
            },
            {
                "query": "Exercise reduces cardiovascular disease risk",
                "correct": "Physical activity lowers heart disease probability",
                "adversarial": "Exercise does not reduce cardiovascular disease risk",
                "type": "negation",
                "difficulty": "hard"
            },
            {
                "query": "Early diagnosis improves patient outcomes significantly",
                "correct": "Timely detection enhances patient prognosis substantially",
                "adversarial": "Early diagnosis does not improve patient outcomes significantly",
                "type": "negation",
                "difficulty": "hard"
            },
            {
                "query": "The drug increases survival rates in clinical trials",
                "correct": "Clinical studies show improved survival with this medication",
                "adversarial": "The drug does not increase survival rates in clinical trials",
                "type": "negation",
                "difficulty": "easy"
            },
            {
                "query": "Antibiotics are effective against bacterial infections",
                "correct": "Antimicrobial agents successfully treat bacterial diseases",
                "adversarial": "Antibiotics are not effective against bacterial infections",
                "type": "negation",
                "difficulty": "medium"
            },
            {
                "query": "Smoking increases lung cancer risk substantially",
                "correct": "Tobacco use significantly raises lung cancer probability",
                "adversarial": "Smoking does not increase lung cancer risk substantially",
                "type": "negation",
                "difficulty": "hard"
            },
            {
                "query": "The treatment reduces symptom severity",
                "correct": "This intervention decreases symptom intensity",
                "adversarial": "The treatment does not reduce symptom severity",
                "type": "negation",
                "difficulty": "medium"
            }
        ]
    
    def _create_word_order_examples(self) -> List[Dict]:
        """Create word order sensitivity examples."""
        return [
            {
                "query": "Administer drug A before drug B for optimal effect",
                "correct": "Drug A should precede drug B for best results",
                "adversarial": "Administer drug B before drug A for optimal effect",
                "type": "word_order",
                "difficulty": "easy"
            },
            {
                "query": "Symptoms appeared after medication was stopped",
                "correct": "Discontinuing medication led to symptom emergence",
                "adversarial": "Medication was stopped after symptoms appeared",
                "type": "word_order",
                "difficulty": "medium"
            },
            {
                "query": "Fever subsided then rash developed in patient",
                "correct": "Patient's fever resolved followed by rash appearance",
                "adversarial": "Rash developed then fever subsided in patient",
                "type": "word_order",
                "difficulty": "hard"
            },
            {
                "query": "Surgery followed by chemotherapy improves survival",
                "correct": "Post-surgical chemotherapy enhances patient outcomes",
                "adversarial": "Chemotherapy followed by surgery improves survival",
                "type": "word_order",
                "difficulty": "medium"
            },
            {
                "query": "Diagnosis confirmed after test results reviewed",
                "correct": "Test result analysis led to diagnostic confirmation",
                "adversarial": "Test results reviewed after diagnosis confirmed",
                "type": "word_order",
                "difficulty": "hard"
            },
            {
                "query": "Pain decreased then mobility improved gradually",
                "correct": "Gradual mobility improvement followed pain reduction",
                "adversarial": "Mobility improved then pain decreased gradually",
                "type": "word_order",
                "difficulty": "medium"
            },
            {
                "query": "Treatment initiated before symptoms worsened",
                "correct": "Therapy began prior to symptom deterioration",
                "adversarial": "Symptoms worsened before treatment initiated",
                "type": "word_order",
                "difficulty": "easy"
            },
            {
                "query": "Blood pressure checked then medication adjusted",
                "correct": "Medication modification followed blood pressure assessment",
                "adversarial": "Medication adjusted then blood pressure checked",
                "type": "word_order",
                "difficulty": "medium"
            },
            {
                "query": "Infection cleared after antibiotic course completed",
                "correct": "Antibiotic completion resulted in infection resolution",
                "adversarial": "Antibiotic course completed after infection cleared",
                "type": "word_order",
                "difficulty": "hard"
            },
            {
                "query": "X-ray taken before surgery scheduled",
                "correct": "Pre-surgical X-ray imaging was performed",
                "adversarial": "Surgery scheduled before X-ray taken",
                "type": "word_order",
                "difficulty": "easy"
            }
        ]
    
    def _create_contradiction_examples(self) -> List[Dict]:
        """Create subtle contradiction examples."""
        return [
            {
                "query": "Study shows vaccine prevents disease transmission",
                "correct": "Research demonstrates vaccination blocks disease spread",
                "adversarial": "Study shows vaccine prevents symptoms but not transmission",
                "type": "contradiction",
                "difficulty": "hard"
            },
            {
                "query": "Drug reduces both pain and inflammation",
                "correct": "Medication alleviates pain and inflammatory responses",
                "adversarial": "Drug reduces pain but increases inflammation",
                "type": "contradiction",
                "difficulty": "medium"
            },
            {
                "query": "Exercise improves both strength and endurance",
                "correct": "Physical training enhances muscular strength and cardiovascular endurance",
                "adversarial": "Exercise improves strength but decreases endurance",
                "type": "contradiction",
                "difficulty": "medium"
            },
            {
                "query": "Treatment is safe and effective for all patients",
                "correct": "Therapy demonstrates safety and efficacy across patient populations",
                "adversarial": "Treatment is effective but not safe for all patients",
                "type": "contradiction",
                "difficulty": "hard"
            },
            {
                "query": "Diet change reduces weight and cholesterol levels",
                "correct": "Dietary modification decreases body weight and cholesterol",
                "adversarial": "Diet change reduces weight but increases cholesterol levels",
                "type": "contradiction",
                "difficulty": "medium"
            },
            {
                "query": "Medication improves sleep quality and duration",
                "correct": "Drug enhances both sleep quality and total sleep time",
                "adversarial": "Medication improves sleep quality but reduces duration",
                "type": "contradiction",
                "difficulty": "hard"
            },
            {
                "query": "Therapy helps with anxiety and depression symptoms",
                "correct": "Treatment alleviates both anxiety and depressive symptoms",
                "adversarial": "Therapy helps anxiety but worsens depression symptoms",
                "type": "contradiction",
                "difficulty": "medium"
            },
            {
                "query": "Surgery reduces tumor size and spread",
                "correct": "Surgical intervention decreases tumor size and metastasis",
                "adversarial": "Surgery reduces tumor size but increases spread",
                "type": "contradiction",
                "difficulty": "hard"
            },
            {
                "query": "Device monitors heart rate and blood pressure accurately",
                "correct": "Equipment precisely tracks cardiac rate and blood pressure",
                "adversarial": "Device monitors heart rate accurately but not blood pressure",
                "type": "contradiction",
                "difficulty": "medium"
            },
            {
                "query": "Program increases patient compliance and satisfaction",
                "correct": "Initiative enhances patient adherence and contentment",
                "adversarial": "Program increases compliance but decreases satisfaction",
                "type": "contradiction",
                "difficulty": "hard"
            }
        ]
    
    def _create_synonym_examples(self) -> List[Dict]:
        """Create synonym/paraphrase examples."""
        return [
            {
                "query": "The treatment alleviates chronic pain symptoms",
                "correct": "This therapy reduces persistent pain manifestations",
                "adversarial": "The treatment exacerbates chronic pain symptoms",
                "type": "synonym",
                "difficulty": "easy"
            },
            {
                "query": "Medication enhances cognitive function in elderly",
                "correct": "Drug improves mental performance in older adults",
                "adversarial": "Medication impairs cognitive function in elderly",
                "type": "synonym",
                "difficulty": "medium"
            },
            {
                "query": "Protocol minimizes surgical complications significantly",
                "correct": "Procedure substantially reduces operative adverse events",
                "adversarial": "Protocol maximizes surgical complications significantly",
                "type": "synonym",
                "difficulty": "medium"
            },
            {
                "query": "Intervention accelerates patient recovery time",
                "correct": "Treatment expedites patient healing duration",
                "adversarial": "Intervention decelerates patient recovery time",
                "type": "synonym",
                "difficulty": "easy"
            },
            {
                "query": "Approach optimizes treatment outcomes for patients",
                "correct": "Method maximizes therapeutic results for individuals",
                "adversarial": "Approach compromises treatment outcomes for patients",
                "type": "synonym",
                "difficulty": "hard"
            },
            {
                "query": "System facilitates accurate diagnostic assessment",
                "correct": "Technology enables precise diagnostic evaluation",
                "adversarial": "System hinders accurate diagnostic assessment",
                "type": "synonym",
                "difficulty": "medium"
            },
            {
                "query": "Strategy diminishes healthcare costs substantially",
                "correct": "Plan significantly reduces medical expenses",
                "adversarial": "Strategy amplifies healthcare costs substantially",
                "type": "synonym",
                "difficulty": "easy"
            },
            {
                "query": "Program strengthens patient-provider relationships",
                "correct": "Initiative reinforces patient-clinician connections",
                "adversarial": "Program weakens patient-provider relationships",
                "type": "synonym",
                "difficulty": "medium"
            },
            {
                "query": "Method consolidates multiple treatment approaches",
                "correct": "Technique integrates various therapeutic strategies",
                "adversarial": "Method fragments multiple treatment approaches",
                "type": "synonym",
                "difficulty": "hard"
            },
            {
                "query": "Innovation transforms healthcare delivery models",
                "correct": "Advancement revolutionizes medical care provision systems",
                "adversarial": "Innovation stagnates healthcare delivery models",
                "type": "synonym",
                "difficulty": "hard"
            }
        ]
    
    def _create_typo_examples(self) -> List[Dict]:
        """Create typo/noise injection examples."""
        return [
            {
                "query": "The antibiotic treats bacterial infections effectively",
                "correct": "Antimicrobial medication successfully manages bacterial diseases",
                "adversarial": "The antbiotic trets bactrial infctions effctively",
                "type": "typo",
                "difficulty": "easy"
            },
            {
                "query": "Physical therapy improves mobility and strength",
                "correct": "Rehabilitation exercises enhance movement and muscular power",
                "adversarial": "Physcal theraphy impoves moblity and stregth",
                "type": "typo",
                "difficulty": "medium"
            },
            {
                "query": "Vaccination prevents disease transmission in populations",
                "correct": "Immunization stops disease spread among communities",
                "adversarial": "Vaccintion prevnts disese transmisson in populatons",
                "type": "typo",
                "difficulty": "medium"
            },
            {
                "query": "Surgery requires careful patient selection and preparation",
                "correct": "Operative procedures need thorough patient screening and readiness",
                "adversarial": "Surgry requres carefl patint selecton and preparaton",
                "type": "typo",
                "difficulty": "hard"
            },
            {
                "query": "Diagnosis depends on comprehensive clinical assessment",
                "correct": "Medical identification relies on thorough clinical evaluation",
                "adversarial": "Diagosis depnds on comprehensve clincal assesment",
                "type": "typo",
                "difficulty": "medium"
            },
            {
                "query": "Chemotherapy targets malignant cells specifically",
                "correct": "Cancer treatment selectively affects cancerous cells",
                "adversarial": "Chemothrapy targts malignnt cels specifcally",
                "type": "typo",
                "difficulty": "hard"
            },
            {
                "query": "Rehabilitation improves functional independence gradually",
                "correct": "Recovery therapy progressively enhances self-sufficiency",
                "adversarial": "Rehabiltation improes functonal independnce gradualy",
                "type": "typo",
                "difficulty": "medium"
            },
            {
                "query": "Prevention strategies reduce healthcare costs significantly",
                "correct": "Preventive measures substantially decrease medical expenses",
                "adversarial": "Preventon strateges redce healthcre csts significatly",
                "type": "typo",
                "difficulty": "easy"
            },
            {
                "query": "Monitoring enables early detection of complications",
                "correct": "Surveillance allows prompt identification of adverse events",
                "adversarial": "Monitrng enales erly detecton of complicatons",
                "type": "typo",
                "difficulty": "medium"
            },
            {
                "query": "Technology enhances diagnostic accuracy and speed",
                "correct": "Innovation improves diagnostic precision and efficiency",
                "adversarial": "Technolgy enhaces diagostc accurcy and sped",
                "type": "typo",
                "difficulty": "hard"
            }
        ]
    
    def apply_noise_perturbations(self, text: str, noise_level: float) -> str:
        """Apply systematic noise perturbations."""
        if noise_level <= 0:
            return text
            
        words = text.split()
        num_words_to_modify = max(1, int(len(words) * noise_level))
        
        for _ in range(num_words_to_modify):
            if not words:
                break
                
            idx = random.randint(0, len(words) - 1)
            word = words[idx]
            
            if len(word) < 3:
                continue
                
            perturbation_type = random.choice(['character_swap', 'character_delete', 'character_insert'])
            
            if perturbation_type == 'character_swap' and len(word) > 1:
                pos = random.randint(0, len(word) - 2)
                word_list = list(word)
                word_list[pos], word_list[pos + 1] = word_list[pos + 1], word_list[pos]
                words[idx] = ''.join(word_list)
            elif perturbation_type == 'character_delete' and len(word) > 2:
                pos = random.randint(1, len(word) - 2)
                words[idx] = word[:pos] + word[pos + 1:]
            elif perturbation_type == 'character_insert':
                pos = random.randint(1, len(word) - 1)
                char = random.choice('abcdefghijklmnopqrstuvwxyz')
                words[idx] = word[:pos] + char + word[pos:]
        
        return ' '.join(words)

class ComprehensiveRobustnessEvaluator:
    """Comprehensive evaluator with statistical analysis."""
    
    def __init__(self):
        self.generator = ComprehensiveAdversarialGenerator()
        
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
        """Create classical baseline."""
        config = SimilarityEngineConfig(
            n_qubits=2,
            n_layers=1,
            similarity_method=SimilarityMethod.CLASSICAL_COSINE,
            enable_caching=True
        )
        return QuantumRAGReranker(config=config)
    
    def evaluate_single_scenario(self, reranker, scenario, method="quantum"):
        """Evaluate single adversarial scenario."""
        query = scenario["query"]
        correct_doc = scenario["correct"]
        adversarial_doc = scenario["adversarial"]
        
        try:
            candidates = [correct_doc, adversarial_doc]
            results = reranker.rerank(query, candidates, method=method, top_k=2)
            
            if results and len(results) >= 2:
                top_doc = results[0].get('text', results[0].get('content', ''))
                top_score = results[0].get('score', results[0].get('similarity', 0))
                second_score = results[1].get('score', results[1].get('similarity', 0))
                
                # Check if correct document ranked first
                is_correct = (top_doc == correct_doc or correct_doc in top_doc)
                
                # Calculate ranking confidence (score difference)
                confidence = abs(top_score - second_score) if top_score and second_score else 0
                
                return {
                    "correct": is_correct,
                    "confidence": confidence,
                    "top_score": top_score,
                    "second_score": second_score,
                    "success": True
                }
            else:
                return {"correct": False, "confidence": 0, "success": False}
                
        except Exception as e:
            return {"correct": False, "confidence": 0, "success": False, "error": str(e)}
    
    def evaluate_noise_robustness_detailed(self, reranker, clean_scenarios, noise_levels, method="quantum", trials=3):
        """Detailed noise robustness evaluation with multiple trials."""
        results_by_noise = {}
        
        for noise_level in noise_levels:
            trial_results = []
            
            for trial in range(trials):
                # Create noisy scenarios for this trial
                noisy_scenarios = []
                for scenario in clean_scenarios:
                    noisy_scenario = scenario.copy()
                    if noise_level > 0:
                        # Apply noise to query and correct document
                        noisy_scenario["query"] = self.generator.apply_noise_perturbations(
                            scenario["query"], noise_level
                        )
                        noisy_scenario["correct"] = self.generator.apply_noise_perturbations(
                            scenario["correct"], noise_level
                        )
                    noisy_scenarios.append(noisy_scenario)
                
                # Evaluate on noisy scenarios
                trial_accuracy = 0
                trial_confidence = []
                
                for scenario in noisy_scenarios:
                    result = self.evaluate_single_scenario(reranker, scenario, method)
                    if result["success"]:
                        if result["correct"]:
                            trial_accuracy += 1
                        trial_confidence.append(result["confidence"])
                
                trial_results.append({
                    "accuracy": trial_accuracy / len(noisy_scenarios) if noisy_scenarios else 0,
                    "avg_confidence": np.mean(trial_confidence) if trial_confidence else 0,
                    "num_scenarios": len(noisy_scenarios)
                })
            
            # Aggregate trial results
            accuracies = [r["accuracy"] for r in trial_results]
            confidences = [r["avg_confidence"] for r in trial_results]
            
            results_by_noise[noise_level] = {
                "mean_accuracy": np.mean(accuracies),
                "std_accuracy": np.std(accuracies),
                "mean_confidence": np.mean(confidences),
                "std_confidence": np.std(confidences),
                "trials": trial_results
            }
        
        return results_by_noise
    
    def run_comprehensive_evaluation(self, model_path: str = None, num_trials: int = 5):
        """Run comprehensive adversarial robustness evaluation."""
        print("🛡️ COMPREHENSIVE Adversarial Robustness Test")
        print("=" * 70)
        print(f"Large-scale objective evaluation with {num_trials} trials per test")
        
        # Create rerankers
        print("\n🤖 Initializing Systems...")
        quantum_reranker = self.create_quantum_reranker(model_path)
        classical_reranker = self.create_classical_baseline()
        print("✅ Quantum reranker created")
        print("✅ Classical baseline created")
        
        # Generate comprehensive dataset
        print("\n📊 Generating Comprehensive Dataset...")
        adversarial_dataset = self.generator.create_large_adversarial_dataset()
        print(f"Created {len(adversarial_dataset)} adversarial test pairs")
        
        # Group by type for analysis
        by_type = defaultdict(list)
        for scenario in adversarial_dataset:
            by_type[scenario["type"]].append(scenario)
        
        for adv_type, scenarios in by_type.items():
            print(f"  • {adv_type}: {len(scenarios)} pairs")
        
        # Test 1: Large-scale adversarial evaluation
        print(f"\n📊 Test 1: Large-Scale Adversarial Evaluation ({num_trials} trials)")
        print("-" * 50)
        
        quantum_results = {"accuracies": [], "confidences": [], "by_type": {}}
        classical_results = {"accuracies": [], "confidences": [], "by_type": {}}
        
        for trial in range(num_trials):
            print(f"  Trial {trial + 1}/{num_trials}")
            
            # Quantum evaluation
            q_correct = 0
            q_confidences = []
            q_by_type = defaultdict(list)
            
            for scenario in adversarial_dataset:
                result = self.evaluate_single_scenario(quantum_reranker, scenario, "quantum")
                if result["success"]:
                    if result["correct"]:
                        q_correct += 1
                    q_confidences.append(result["confidence"])
                    q_by_type[scenario["type"]].append(result["correct"])
            
            # Classical evaluation
            c_correct = 0
            c_confidences = []
            c_by_type = defaultdict(list)
            
            for scenario in adversarial_dataset:
                result = self.evaluate_single_scenario(classical_reranker, scenario, "classical")
                if result["success"]:
                    if result["correct"]:
                        c_correct += 1
                    c_confidences.append(result["confidence"])
                    c_by_type[scenario["type"]].append(result["correct"])
            
            # Store trial results
            q_accuracy = q_correct / len(adversarial_dataset)
            c_accuracy = c_correct / len(adversarial_dataset)
            
            quantum_results["accuracies"].append(q_accuracy)
            classical_results["accuracies"].append(c_accuracy)
            quantum_results["confidences"].extend(q_confidences)
            classical_results["confidences"].extend(c_confidences)
            
            # Store by-type results
            for adv_type in q_by_type:
                if adv_type not in quantum_results["by_type"]:
                    quantum_results["by_type"][adv_type] = []
                    classical_results["by_type"][adv_type] = []
                quantum_results["by_type"][adv_type].extend(q_by_type[adv_type])
                classical_results["by_type"][adv_type].extend(c_by_type[adv_type])
            
            print(f"    Quantum: {q_accuracy:.3f}, Classical: {c_accuracy:.3f}")
        
        # Test 2: Noise robustness with multiple trials
        print(f"\n📊 Test 2: Noise Robustness Analysis")
        print("-" * 50)
        
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
        subset_scenarios = adversarial_dataset[:20]  # Use subset for noise testing
        
        print(f"Testing noise levels: {noise_levels}")
        print(f"Using {len(subset_scenarios)} scenarios with {num_trials} trials each")
        
        quantum_noise_results = self.evaluate_noise_robustness_detailed(
            quantum_reranker, subset_scenarios, noise_levels, "quantum", trials=num_trials
        )
        
        classical_noise_results = self.evaluate_noise_robustness_detailed(
            classical_reranker, subset_scenarios, noise_levels, "classical", trials=num_trials
        )
        
        return {
            "adversarial_evaluation": {
                "quantum": quantum_results,
                "classical": classical_results,
                "dataset_size": len(adversarial_dataset),
                "num_trials": num_trials
            },
            "noise_robustness": {
                "quantum": quantum_noise_results,
                "classical": classical_noise_results,
                "noise_levels": noise_levels,
                "num_scenarios": len(subset_scenarios),
                "num_trials": num_trials
            }
        }
    
    def analyze_comprehensive_results(self, results):
        """Comprehensive statistical analysis."""
        print(f"\n📈 COMPREHENSIVE STATISTICAL ANALYSIS")
        print("=" * 70)
        
        adv_results = results["adversarial_evaluation"]
        noise_results = results["noise_robustness"]
        
        # Adversarial evaluation analysis
        q_accuracies = adv_results["quantum"]["accuracies"]
        c_accuracies = adv_results["classical"]["accuracies"]
        
        q_mean = np.mean(q_accuracies)
        q_std = np.std(q_accuracies)
        c_mean = np.mean(c_accuracies)
        c_std = np.std(c_accuracies)
        
        print(f"\n📊 Adversarial Evaluation Results:")
        print(f"  Dataset size: {adv_results['dataset_size']} pairs")
        print(f"  Number of trials: {adv_results['num_trials']}")
        print(f"  Quantum accuracy: {q_mean:.3f} ± {q_std:.3f}")
        print(f"  Classical accuracy: {c_mean:.3f} ± {c_std:.3f}")
        
        # Statistical significance test
        if len(q_accuracies) > 1 and len(c_accuracies) > 1:
            t_stat, p_value = stats.ttest_ind(q_accuracies, c_accuracies)
            print(f"  T-test p-value: {p_value:.4f}")
            is_significant = p_value < 0.05
            print(f"  Statistically significant: {'Yes' if is_significant else 'No'}")
        else:
            is_significant = False
            p_value = 1.0
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(q_accuracies) - 1) * q_std**2 + (len(c_accuracies) - 1) * c_std**2) / 
                           (len(q_accuracies) + len(c_accuracies) - 2))
        cohens_d = (q_mean - c_mean) / pooled_std if pooled_std > 0 else 0
        print(f"  Effect size (Cohen's d): {cohens_d:.3f}")
        
        # By-type analysis
        print(f"\n📊 Performance by Adversarial Type:")
        print(f"  {'Type':<15} {'Quantum':<10} {'Classical':<10} {'Difference'}")
        print("  " + "-" * 50)
        
        type_advantages = []
        for adv_type in adv_results["quantum"]["by_type"]:
            q_type_acc = np.mean(adv_results["quantum"]["by_type"][adv_type])
            c_type_acc = np.mean(adv_results["classical"]["by_type"][adv_type])
            difference = q_type_acc - c_type_acc
            type_advantages.append(difference)
            print(f"  {adv_type:<15} {q_type_acc:<10.3f} {c_type_acc:<10.3f} {difference:+.3f}")
        
        # Noise robustness analysis
        print(f"\n📊 Noise Robustness Analysis:")
        print(f"  {'Noise':<8} {'Q-Accuracy':<12} {'C-Accuracy':<12} {'Q-Advantage'}")
        print("  " + "-" * 45)
        
        noise_advantages = []
        for noise_level in noise_results["noise_levels"]:
            q_noise = noise_results["quantum"][noise_level]
            c_noise = noise_results["classical"][noise_level]
            
            q_acc = q_noise["mean_accuracy"]
            c_acc = c_noise["mean_accuracy"]
            advantage = q_acc - c_acc
            noise_advantages.append(advantage)
            
            print(f"  {noise_level:<8.1f} {q_acc:<12.3f} {c_acc:<12.3f} {advantage:+.3f}")
        
        # Degradation analysis
        q_clean = noise_results["quantum"][0.0]["mean_accuracy"]
        q_noisy = noise_results["quantum"][0.4]["mean_accuracy"]
        c_clean = noise_results["classical"][0.0]["mean_accuracy"]
        c_noisy = noise_results["classical"][0.4]["mean_accuracy"]
        
        q_degradation = q_clean - q_noisy
        c_degradation = c_clean - c_noisy
        
        print(f"\n📊 Performance Degradation (0% → 40% noise):")
        print(f"  Quantum degradation: {q_degradation:.3f}")
        print(f"  Classical degradation: {c_degradation:.3f}")
        print(f"  Robustness advantage: {c_degradation - q_degradation:+.3f}")
        
        # Overall verdict
        print(f"\n💡 OBJECTIVE ASSESSMENT:")
        print("-" * 35)
        
        # Criteria for quantum advantage
        criteria_met = []
        
        if q_mean > c_mean and is_significant:
            criteria_met.append("Significantly better adversarial accuracy")
        
        if abs(cohens_d) > 0.2:  # Small effect size threshold
            criteria_met.append(f"Meaningful effect size ({cohens_d:.2f})")
        
        if sum(1 for adv in type_advantages if adv > 0.05) >= 2:
            criteria_met.append("Better performance on multiple adversarial types")
        
        if c_degradation > q_degradation + 0.05:
            criteria_met.append("Superior noise robustness")
        
        if len(criteria_met) > 0:
            print("✅ Quantum robustness advantages found:")
            for criterion in criteria_met:
                print(f"  • {criterion}")
            overall_verdict = "QUANTUM_ROBUST"
        else:
            print("❌ No significant quantum robustness advantages")
            print(f"  • Mean accuracy difference: {q_mean - c_mean:+.3f}")
            print(f"  • Statistical significance: {'No' if p_value >= 0.05 else 'Yes'}")
            print(f"  • Effect size: {cohens_d:.3f} (threshold: 0.2)")
            overall_verdict = "NO_ADVANTAGE"
        
        return {
            "overall_verdict": overall_verdict,
            "adversarial_accuracy_quantum": q_mean,
            "adversarial_accuracy_classical": c_mean,
            "statistical_significance": is_significant,
            "p_value": p_value,
            "effect_size": cohens_d,
            "noise_robustness_advantage": c_degradation - q_degradation,
            "criteria_met": criteria_met
        }

def main():
    """Main comprehensive evaluation function."""
    print("🚀 COMPREHENSIVE Adversarial Robustness Evaluation")
    print("Objective, large-scale testing with statistical analysis")
    
    # Parameters
    num_trials = 5  # Number of independent trials
    
    # Check for trained model
    model_path = "models/qpmel_extended.pt"
    if not Path(model_path).exists():
        print(f"⚠️  Using default quantum parameters")
        model_path = None
    else:
        print(f"✅ Using trained QPMeL model")
    
    # Run comprehensive evaluation
    evaluator = ComprehensiveRobustnessEvaluator()
    
    print(f"\n⏱️  Starting comprehensive evaluation with {num_trials} trials...")
    start_time = time.time()
    
    results = evaluator.run_comprehensive_evaluation(model_path, num_trials)
    
    total_time = time.time() - start_time
    
    # Statistical analysis
    analysis = evaluator.analyze_comprehensive_results(results)
    
    # Save comprehensive results
    output_data = {
        "test_name": "comprehensive_adversarial_robustness",
        "total_time_seconds": total_time,
        "results": results,
        "analysis": analysis,
        "timestamp": time.time()
    }
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    output_data = convert_numpy_types(output_data)
    
    with open("comprehensive_adversarial_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to comprehensive_adversarial_results.json")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    
    # Final comprehensive verdict
    print(f"\n🎯 FINAL COMPREHENSIVE VERDICT:")
    print("=" * 50)
    
    if analysis["overall_verdict"] == "QUANTUM_ROBUST":
        print("✅ QUANTUM ROBUSTNESS ADVANTAGES CONFIRMED")
        print(f"📊 Statistical evidence:")
        print(f"  • P-value: {analysis['p_value']:.4f}")
        print(f"  • Effect size: {analysis['effect_size']:.3f}")
        print(f"  • Noise robustness advantage: {analysis['noise_robustness_advantage']:+.3f}")
        print(f"🎯 RECOMMENDATION: Deploy for robustness-critical applications")
    else:
        print("❌ NO SIGNIFICANT QUANTUM ADVANTAGES")
        print(f"📊 Evidence against quantum advantage:")
        print(f"  • Accuracy difference: {analysis['adversarial_accuracy_quantum'] - analysis['adversarial_accuracy_classical']:+.3f}")
        print(f"  • Statistical significance: {analysis['statistical_significance']}")
        print(f"🔄 RECOMMENDATION: Classical methods preferred")
    
    print(f"\n✨ Comprehensive Evaluation Complete!")
    return analysis["overall_verdict"] == "QUANTUM_ROBUST"

if __name__ == "__main__":
    main()
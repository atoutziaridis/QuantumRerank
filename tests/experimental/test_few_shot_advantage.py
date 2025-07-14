#!/usr/bin/env python3
"""
Test 1: Few-Shot Biomedical Classification
Test if quantum reranker shows advantage with very limited training data.
This leverages existing biomedical QPMeL training.
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

class FewShotDataset:
    """Creates few-shot scenarios from biomedical/scientific data."""
    
    def __init__(self):
        self.data = self._create_biomedical_classification_data()
    
    def _create_biomedical_classification_data(self):
        """Create a biomedical classification dataset with multiple classes."""
        return {
            "cardiovascular_disease": [
                "Myocardial infarction results from coronary artery occlusion leading to cardiac muscle necrosis and potential heart failure.",
                "Atherosclerosis involves lipid plaque formation in arterial walls causing vessel narrowing and increased cardiovascular risk.",
                "Hypertension increases cardiac workload and vascular stress, contributing to stroke, heart disease, and kidney damage.",
                "Arrhythmias disrupt normal cardiac electrical conduction, potentially causing palpitations, syncope, or sudden cardiac death.",
                "Heart failure occurs when cardiac output is insufficient to meet metabolic demands, causing fluid retention and fatigue.",
                "Peripheral arterial disease reduces blood flow to extremities, causing claudication, ulcers, and potential limb amputation.",
                "Aortic stenosis restricts blood flow from left ventricle, causing chest pain, syncope, and progressive heart failure.",
                "Endocarditis involves bacterial infection of heart valves, potentially causing valve destruction and systemic emboli.",
                "Coronary artery disease narrows cardiac vessels, reducing myocardial perfusion and increasing infarction risk.",
                "Venous thromboembolism forms blood clots in deep veins, with potential pulmonary embolism and life-threatening complications."
            ],
            
            "neurological_disorders": [
                "Alzheimer's disease involves amyloid plaques and tau tangles causing progressive memory loss and cognitive decline.",
                "Parkinson's disease affects dopaminergic neurons, causing tremor, bradykinesia, rigidity, and postural instability.",
                "Multiple sclerosis causes autoimmune demyelination of central nervous system, resulting in diverse neurological symptoms.",
                "Epilepsy involves recurrent seizures due to abnormal neuronal electrical activity in the brain.",
                "Stroke occurs from cerebrovascular occlusion or hemorrhage, causing acute focal neurological deficits.",
                "Migraine headaches involve vascular and neurogenic mechanisms causing severe throbbing pain and sensory sensitivity.",
                "Peripheral neuropathy damages peripheral nerves, causing numbness, tingling, weakness, and neuropathic pain.",
                "Huntington's disease is a genetic disorder causing progressive motor, cognitive, and psychiatric deterioration.",
                "Amyotrophic lateral sclerosis affects motor neurons, causing progressive muscle weakness and eventual respiratory failure.",
                "Brain tumors can be primary or metastatic, causing focal deficits, seizures, and increased intracranial pressure."
            ],
            
            "infectious_diseases": [
                "Pneumonia involves lung parenchyma infection causing fever, cough, dyspnea, and potential respiratory failure.",
                "Tuberculosis is caused by Mycobacterium tuberculosis, causing chronic cough, weight loss, and lung cavitation.",
                "HIV infection attacks CD4+ T cells, causing immunodeficiency and opportunistic infections without treatment.",
                "Hepatitis B virus causes liver inflammation, potentially leading to cirrhosis and hepatocellular carcinoma.",
                "Influenza is a respiratory viral infection causing fever, myalgia, cough, and potential pandemic spread.",
                "Malaria is transmitted by Anopheles mosquitoes, causing cyclical fever, anemia, and potential cerebral complications.",
                "Sepsis involves systemic inflammatory response to infection, potentially causing multi-organ failure and death.",
                "Urinary tract infections affect kidneys, bladder, or urethra, causing dysuria, frequency, and potential pyelonephritis.",
                "Meningitis involves central nervous system membrane inflammation, causing headache, neck stiffness, and altered consciousness.",
                "Cellulitis is bacterial skin and soft tissue infection causing erythema, warmth, swelling, and systemic symptoms."
            ],
            
            "endocrine_disorders": [
                "Diabetes mellitus involves insulin deficiency or resistance, causing hyperglycemia and long-term vascular complications.",
                "Hypothyroidism results from thyroid hormone deficiency, causing fatigue, weight gain, cold intolerance, and bradycardia.",
                "Hyperthyroidism involves excess thyroid hormones, causing weight loss, tachycardia, heat intolerance, and anxiety.",
                "Cushing's syndrome results from excess cortisol, causing central obesity, hypertension, and glucose intolerance.",
                "Addison's disease involves adrenal insufficiency, causing fatigue, hypotension, skin pigmentation, and electrolyte imbalance.",
                "Osteoporosis involves bone density reduction, increasing fracture risk, particularly in spine and hip.",
                "Polycystic ovary syndrome affects reproductive hormones, causing irregular menses, hirsutism, and metabolic dysfunction.",
                "Hyperparathyroidism causes elevated calcium levels, leading to kidney stones, bone disease, and neuropsychiatric symptoms.",
                "Growth hormone deficiency in children causes short stature and delayed puberty requiring hormone replacement therapy.",
                "Pheochromocytoma is a catecholamine-secreting tumor causing episodic hypertension, headaches, and diaphoresis."
            ],
            
            "cancer_types": [
                "Lung cancer is often associated with smoking, causing cough, dyspnea, weight loss, and potential metastasis.",
                "Breast cancer affects mammary tissue, potentially causing lumps, skin changes, and lymphatic spread.",
                "Colorectal cancer involves large intestine tumors, causing bleeding, obstruction, and potential liver metastasis.",
                "Prostate cancer affects male reproductive glands, potentially causing urinary symptoms and bone metastases.",
                "Pancreatic cancer has poor prognosis, causing abdominal pain, jaundice, weight loss, and rapid progression.",
                "Melanoma is aggressive skin cancer from melanocytes, with potential for early metastasis and poor outcomes.",
                "Leukemia involves blood cell malignancies, causing anemia, bleeding, infections, and bone marrow failure.",
                "Lymphoma affects lymphatic system, causing lymphadenopathy, fever, night sweats, and weight loss.",
                "Brain tumors can cause seizures, focal deficits, personality changes, and increased intracranial pressure.",
                "Ovarian cancer often presents late with abdominal distension, pelvic pain, and peritoneal spread."
            ]
        }
    
    def sample_few_shot(self, shot_count: int, test_size: int = 20) -> Dict:
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

class FewShotEvaluator:
    """Evaluates few-shot performance of quantum vs classical rerankers."""
    
    def __init__(self):
        self.dataset = FewShotDataset()
    
    def create_quantum_reranker(self, model_path: str = None):
        """Create quantum reranker with trained or default parameters."""
        if model_path and Path(model_path).exists():
            # Use trained QPMeL model
            config = QPMeLTrainingConfig(
                qpmel_config=QPMeLConfig(n_qubits=2, n_layers=1),
                batch_size=8
            )
            trainer = QPMeLTrainer(config=config)
            trainer.load_model(model_path)
            return trainer.get_trained_reranker()
        else:
            # Use default quantum reranker
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
    
    def evaluate_classification_accuracy(self, reranker, train_data, test_data, categories):
        """Evaluate classification accuracy using reranker for similarity-based classification."""
        correct_predictions = 0
        total_predictions = 0
        
        # For each test document, find most similar training category
        for true_category in categories:
            for test_doc in test_data[true_category]:
                # Find most similar training document across all categories
                best_similarity = -1
                predicted_category = None
                
                for train_category in categories:
                    for train_doc in train_data[train_category]:
                        # Rerank this single document
                        try:
                            results = reranker.rerank(test_doc, [train_doc], method="quantum" if "quantum" in str(type(reranker)).lower() else "classical", top_k=1)
                            if results:
                                similarity = results[0].get('score', results[0].get('similarity', 0))
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    predicted_category = train_category
                        except Exception as e:
                            print(f"Error in reranking: {e}")
                            continue
                
                if predicted_category == true_category:
                    correct_predictions += 1
                total_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def run_few_shot_comparison(self, shot_counts: List[int], model_path: str = None, num_trials: int = 3):
        """Run few-shot comparison across different shot counts."""
        print("🔬 Few-Shot Biomedical Classification Test")
        print("=" * 60)
        print(f"Testing quantum vs classical with {num_trials} trials per shot count")
        
        # Create rerankers
        print("\n🤖 Initializing Rerankers...")
        quantum_reranker = self.create_quantum_reranker(model_path)
        classical_reranker = self.create_classical_baseline()
        print("✅ Quantum reranker created")
        print("✅ Classical baseline created")
        
        results = {}
        
        for shot_count in shot_counts:
            print(f"\n📊 Testing {shot_count}-Shot Learning...")
            
            quantum_accuracies = []
            classical_accuracies = []
            
            for trial in range(num_trials):
                print(f"  Trial {trial + 1}/{num_trials}")
                
                # Sample few-shot data
                data_split = self.dataset.sample_few_shot(shot_count)
                
                # Test quantum reranker
                start_time = time.time()
                quantum_acc = self.evaluate_classification_accuracy(
                    quantum_reranker, 
                    data_split["train"], 
                    data_split["test"], 
                    data_split["categories"]
                )
                quantum_time = time.time() - start_time
                
                # Test classical reranker
                start_time = time.time()
                classical_acc = self.evaluate_classification_accuracy(
                    classical_reranker,
                    data_split["train"],
                    data_split["test"],
                    data_split["categories"]
                )
                classical_time = time.time() - start_time
                
                quantum_accuracies.append(quantum_acc)
                classical_accuracies.append(classical_acc)
                
                print(f"    Quantum: {quantum_acc:.3f} ({quantum_time:.1f}s)")
                print(f"    Classical: {classical_acc:.3f} ({classical_time:.1f}s)")
            
            # Calculate statistics
            quantum_mean = np.mean(quantum_accuracies)
            quantum_std = np.std(quantum_accuracies)
            classical_mean = np.mean(classical_accuracies)
            classical_std = np.std(classical_accuracies)
            
            improvement = ((quantum_mean - classical_mean) / classical_mean * 100) if classical_mean > 0 else 0
            
            results[shot_count] = {
                "quantum_mean": quantum_mean,
                "quantum_std": quantum_std,
                "classical_mean": classical_mean,
                "classical_std": classical_std,
                "improvement_pct": improvement,
                "quantum_accuracies": quantum_accuracies,
                "classical_accuracies": classical_accuracies
            }
            
            print(f"  📈 Results: Quantum {quantum_mean:.3f}±{quantum_std:.3f}, Classical {classical_mean:.3f}±{classical_std:.3f}")
            print(f"  🎯 Improvement: {improvement:+.1f}%")
        
        return results
    
    def analyze_results(self, results):
        """Analyze and report few-shot learning results."""
        print(f"\n📊 FEW-SHOT LEARNING ANALYSIS")
        print("=" * 60)
        
        print(f"\n{'Shot Count':<12} {'Quantum':<12} {'Classical':<12} {'Improvement':<12} {'Significant?'}")
        print("-" * 60)
        
        significant_improvements = 0
        best_improvement = -float('inf')
        best_shot_count = 0
        
        for shot_count, data in results.items():
            quantum_mean = data["quantum_mean"]
            classical_mean = data["classical_mean"]
            improvement = data["improvement_pct"]
            
            # Simple significance test (improvement > 2 standard errors)
            quantum_se = data["quantum_std"] / np.sqrt(len(data["quantum_accuracies"]))
            classical_se = data["classical_std"] / np.sqrt(len(data["classical_accuracies"]))
            combined_se = np.sqrt(quantum_se**2 + classical_se**2)
            
            is_significant = abs(quantum_mean - classical_mean) > 2 * combined_se and improvement > 0
            
            if is_significant and improvement > 0:
                significant_improvements += 1
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_shot_count = shot_count
            
            sig_marker = "✅ YES" if is_significant else "❌ No"
            
            print(f"{shot_count:<12} {quantum_mean:<12.3f} {classical_mean:<12.3f} {improvement:+.1f}%{'':<8} {sig_marker}")
        
        # Overall assessment
        print(f"\n💡 KEY INSIGHTS:")
        print("-" * 35)
        
        if significant_improvements > 0:
            print(f"✅ Found quantum advantage in {significant_improvements}/{len(results)} scenarios")
            print(f"🏆 Best performance: {best_shot_count}-shot with {best_improvement:+.1f}% improvement")
            
            if best_shot_count <= 5:
                print("🎯 Quantum shows advantage in ultra-low data regime (≤5 examples)")
                print("📈 Recommendation: Focus on few-shot biomedical applications")
            else:
                print("⚠️  Quantum advantage requires more training data than expected")
        else:
            print("❌ No significant quantum advantage found in few-shot learning")
            print("🔄 Recommendation: Try different quantum parameters or domains")
        
        # Performance analysis
        quantum_values = [data["quantum_mean"] for data in results.values()]
        classical_values = [data["classical_mean"] for data in results.values()]
        
        if max(quantum_values) > max(classical_values):
            print(f"📊 Quantum peak performance: {max(quantum_values):.3f}")
            print(f"📊 Classical peak performance: {max(classical_values):.3f}")
        
        return {
            "significant_improvements": significant_improvements,
            "best_improvement": best_improvement,
            "best_shot_count": best_shot_count,
            "quantum_advantage": significant_improvements > 0
        }

def main():
    """Main function to run few-shot advantage test."""
    print("🚀 Starting Few-Shot Advantage Test (Test 1)")
    print("This test evaluates quantum vs classical in low-data regimes")
    
    # Test parameters
    shot_counts = [1, 3, 5, 10, 20]  # Number of examples per class
    num_trials = 3  # Trials per shot count for statistical significance
    
    # Try to use trained model if available
    model_path = "models/qpmel_extended.pt"
    if not Path(model_path).exists():
        print(f"⚠️  Trained model not found at {model_path}")
        print("Using default quantum parameters")
        model_path = None
    else:
        print(f"✅ Using trained QPMeL model: {model_path}")
    
    # Run evaluation
    evaluator = FewShotEvaluator()
    results = evaluator.run_few_shot_comparison(
        shot_counts=shot_counts,
        model_path=model_path,
        num_trials=num_trials
    )
    
    # Analyze results
    analysis = evaluator.analyze_results(results)
    
    # Save results
    output_data = {
        "test_name": "few_shot_biomedical_classification",
        "parameters": {
            "shot_counts": shot_counts,
            "num_trials": num_trials,
            "model_path": model_path
        },
        "results": results,
        "analysis": analysis,
        "timestamp": time.time()
    }
    
    with open("few_shot_advantage_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to few_shot_advantage_results.json")
    
    # Final recommendation
    if analysis["quantum_advantage"]:
        print(f"\n🎉 SUCCESS: Quantum advantage found!")
        print(f"🎯 Best scenario: {analysis['best_shot_count']}-shot learning")
        print(f"📈 Next steps: Optimize for few-shot applications")
    else:
        print(f"\n🔄 NO ADVANTAGE: Moving to Test 2 (Adversarial Robustness)")
        print(f"📝 This negative result is still valuable data")
    
    print(f"\n✨ Test 1 Complete!")

if __name__ == "__main__":
    main()
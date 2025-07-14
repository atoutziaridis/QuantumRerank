#!/usr/bin/env python3
"""
Test 3: Molecular Similarity Search
Test quantum reranker on quantum-native molecular data.
This is the most promising domain for quantum advantage.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum_rerank.core.rag_reranker import QuantumRAGReranker
from quantum_rerank.core.quantum_similarity_engine import SimilarityEngineConfig, SimilarityMethod
from quantum_rerank.training.qpmel_trainer import QPMeLTrainer, QPMeLTrainingConfig
from quantum_rerank.ml.qpmel_circuits import QPMeLConfig

class MolecularDataset:
    """Creates molecular similarity dataset with known relationships."""
    
    def __init__(self):
        # Simplified molecular representations and known relationships
        self.molecules = self._create_molecular_database()
    
    def _create_molecular_database(self) -> Dict:
        """Create molecular database with known similarity relationships."""
        return {
            # Similar molecules (same therapeutic class)
            "cardiovascular_drugs": {
                "query": "ACE inhibitor for hypertension treatment with enalapril-like structure",
                "molecules": [
                    # Highly similar (other ACE inhibitors)
                    "Lisinopril ACE inhibitor reduces blood pressure by blocking angiotensin converting enzyme activity",
                    "Captopril ACE inhibitor treats hypertension through angiotensin converting enzyme blockade mechanism",
                    "Ramipril ACE inhibitor provides cardiovascular protection via angiotensin converting enzyme inhibition",
                    
                    # Moderately similar (other cardiovascular drugs)
                    "Amlodipine calcium channel blocker reduces blood pressure by preventing calcium influx in vascular smooth muscle",
                    "Metoprolol beta blocker treats hypertension by blocking beta-1 adrenergic receptors in heart",
                    "Losartan angiotensin receptor blocker lowers blood pressure by blocking AT1 receptor activation",
                    
                    # Different class (non-cardiovascular)
                    "Metformin diabetes medication improves glucose control by decreasing hepatic glucose production",
                    "Omeprazole proton pump inhibitor treats gastric acid disorders by blocking H+/K+ ATPase enzyme",
                    "Amoxicillin antibiotic treats bacterial infections by inhibiting bacterial cell wall synthesis"
                ],
                "expected_ranking": [0, 1, 2, 3, 4, 5, 6, 7, 8]  # ACE inhibitors should rank highest
            },
            
            "antibiotics": {
                "query": "Beta-lactam antibiotic for bacterial infection with penicillin-like mechanism",
                "molecules": [
                    # Highly similar (other beta-lactams)
                    "Ampicillin beta-lactam antibiotic inhibits bacterial cell wall synthesis via penicillin binding proteins",
                    "Cephalexin cephalosporin antibiotic disrupts bacterial cell wall formation through beta-lactam ring mechanism",
                    "Cefuroxime second-generation cephalosporin provides broad-spectrum antibacterial activity against gram-positive bacteria",
                    
                    # Moderately similar (other antibiotics)
                    "Azithromycin macrolide antibiotic inhibits protein synthesis by binding to 50S ribosomal subunit",
                    "Ciprofloxacin fluoroquinolone antibiotic disrupts DNA replication by inhibiting DNA gyrase enzyme",
                    "Vancomycin glycopeptide antibiotic treats gram-positive infections by binding to cell wall precursors",
                    
                    # Different class (non-antibiotics)
                    "Insulin hormone regulates glucose metabolism by facilitating cellular glucose uptake",
                    "Warfarin anticoagulant prevents blood clots by inhibiting vitamin K-dependent clotting factors",
                    "Albuterol bronchodilator treats asthma by activating beta-2 adrenergic receptors in airways"
                ],
                "expected_ranking": [0, 1, 2, 3, 4, 5, 6, 7, 8]
            },
            
            "neurology_drugs": {
                "query": "GABA receptor modulator for seizure control with benzodiazepine-like structure",
                "molecules": [
                    # Highly similar (GABA modulators)
                    "Lorazepam benzodiazepine enhances GABA neurotransmission by increasing chloride channel opening frequency",
                    "Clonazepam benzodiazepine provides anticonvulsant effects through GABA-A receptor positive allosteric modulation",
                    "Phenytoin antiepileptic drug stabilizes neuronal membranes by blocking voltage-gated sodium channels",
                    
                    # Moderately similar (other neuro drugs)
                    "Levodopa Parkinson's medication provides dopamine precursor to treat motor symptoms of dopamine deficiency",
                    "Sertraline SSRI antidepressant increases serotonin levels by blocking serotonin reuptake transporters",
                    "Donepezil cholinesterase inhibitor treats Alzheimer's disease by preventing acetylcholine breakdown",
                    
                    # Different class
                    "Atorvastatin statin lowers cholesterol by inhibiting HMG-CoA reductase enzyme in liver",
                    "Prednisone corticosteroid reduces inflammation through glucocorticoid receptor activation",
                    "Furosemide loop diuretic treats edema by blocking sodium-potassium-chloride cotransporter"
                ],
                "expected_ranking": [0, 1, 2, 3, 4, 5, 6, 7, 8]
            }
        }
    
    def get_scenarios(self) -> List[Dict]:
        """Get all molecular similarity scenarios."""
        scenarios = []
        for category, data in self.molecules.items():
            scenarios.append({
                "name": category,
                "query": data["query"],
                "molecules": data["molecules"],
                "expected_ranking": data["expected_ranking"]
            })
        return scenarios

class MolecularAdapter:
    """Adapts molecular data for quantum reranking."""
    
    def __init__(self):
        pass
    
    def molecular_to_text(self, molecular_description: str) -> str:
        """Convert molecular description to text (already text in our case)."""
        return molecular_description
    
    def extract_molecular_features(self, molecule_text: str) -> Dict:
        """Extract key molecular features from text description."""
        features = {
            "mechanism": self._extract_mechanism(molecule_text),
            "target": self._extract_target(molecule_text),
            "therapeutic_class": self._extract_class(molecule_text),
            "chemical_class": self._extract_chemical_class(molecule_text)
        }
        return features
    
    def _extract_mechanism(self, text: str) -> str:
        """Extract mechanism of action."""
        mechanisms = ["inhibitor", "blocker", "agonist", "antagonist", "modulator"]
        for mechanism in mechanisms:
            if mechanism in text.lower():
                return mechanism
        return "unknown"
    
    def _extract_target(self, text: str) -> str:
        """Extract biological target."""
        targets = ["enzyme", "receptor", "channel", "transporter", "ribosome"]
        for target in targets:
            if target in text.lower():
                return target
        return "unknown"
    
    def _extract_class(self, text: str) -> str:
        """Extract therapeutic class."""
        classes = ["antibiotic", "antihypertensive", "anticonvulsant", "analgesic"]
        for cls in classes:
            if cls in text.lower():
                return cls
        return "unknown"
    
    def _extract_chemical_class(self, text: str) -> str:
        """Extract chemical class."""
        chemical_classes = ["beta-lactam", "benzodiazepine", "statin", "fluoroquinolone"]
        for cls in chemical_classes:
            if cls in text.lower():
                return cls
        return "unknown"

class MolecularSimilarityEvaluator:
    """Evaluates quantum vs classical methods on molecular similarity."""
    
    def __init__(self):
        self.dataset = MolecularDataset()
        self.adapter = MolecularAdapter()
    
    def create_quantum_reranker(self, model_path: str = None):
        """Create quantum reranker for molecular data."""
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
        """Create classical baseline for molecular similarity."""
        config = SimilarityEngineConfig(
            n_qubits=2,
            n_layers=1,
            similarity_method=SimilarityMethod.CLASSICAL_COSINE,
            enable_caching=True
        )
        return QuantumRAGReranker(config=config)
    
    def evaluate_molecular_similarity(self, reranker, scenario, method="quantum"):
        """Evaluate molecular similarity ranking."""
        query = scenario["query"]
        molecules = scenario["molecules"]
        expected_ranking = scenario["expected_ranking"]
        
        try:
            # Rerank molecules based on similarity to query
            results = reranker.rerank(query, molecules, method=method, top_k=len(molecules))
            
            if not results:
                return {"error": "No results returned"}
            
            # Calculate ranking metrics
            actual_ranking = []
            for result in results:
                molecule_text = result.get('text', result.get('content', ''))
                # Find original index
                for i, original_molecule in enumerate(molecules):
                    if original_molecule == molecule_text or molecule_text in original_molecule:
                        actual_ranking.append(i)
                        break
            
            # Calculate NDCG and other metrics
            ndcg = self._calculate_ndcg(actual_ranking, expected_ranking)
            kendall_tau = self._calculate_kendall_tau(actual_ranking, expected_ranking)
            precision_3 = self._calculate_precision_at_k(actual_ranking, expected_ranking, k=3)
            
            # Check if similar molecules (same class) are ranked highly
            similar_molecules_in_top3 = sum(1 for idx in actual_ranking[:3] if idx < 3)
            
            return {
                "ndcg": ndcg,
                "kendall_tau": kendall_tau,
                "precision_3": precision_3,
                "similar_in_top3": similar_molecules_in_top3,
                "actual_ranking": actual_ranking,
                "success": True
            }
            
        except Exception as e:
            print(f"Error in molecular reranking: {e}")
            return {"error": str(e), "success": False}
    
    def _calculate_ndcg(self, actual_ranking: List[int], expected_ranking: List[int]) -> float:
        """Calculate NDCG for molecular ranking."""
        if not actual_ranking:
            return 0.0
        
        # Create relevance scores (higher for more similar molecules)
        relevance_scores = []
        for actual_idx in actual_ranking:
            # Higher relevance for molecules that should be ranked higher
            relevance = max(0, len(expected_ranking) - expected_ranking.index(actual_idx) 
                          if actual_idx < len(expected_ranking) else 0)
            relevance_scores.append(relevance)
        
        # DCG calculation
        dcg = relevance_scores[0] if relevance_scores else 0
        for i in range(1, len(relevance_scores)):
            dcg += relevance_scores[i] / np.log2(i + 2)
        
        # IDCG calculation
        ideal_scores = sorted([len(expected_ranking) - i for i in range(len(expected_ranking))], reverse=True)
        idcg = ideal_scores[0] if ideal_scores else 0
        for i in range(1, min(len(ideal_scores), len(relevance_scores))):
            idcg += ideal_scores[i] / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_kendall_tau(self, actual_ranking: List[int], expected_ranking: List[int]) -> float:
        """Calculate Kendall's tau correlation."""
        if len(actual_ranking) != len(expected_ranking):
            return 0.0
        
        n = len(actual_ranking)
        concordant = 0
        discordant = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                actual_order = actual_ranking[i] < actual_ranking[j]
                expected_order = expected_ranking[i] < expected_ranking[j]
                
                if actual_order == expected_order:
                    concordant += 1
                else:
                    discordant += 1
        
        total_pairs = n * (n - 1) // 2
        return (concordant - discordant) / total_pairs if total_pairs > 0 else 0.0
    
    def _calculate_precision_at_k(self, actual_ranking: List[int], expected_ranking: List[int], k: int) -> float:
        """Calculate precision@k for molecular similarity."""
        if not actual_ranking or k <= 0:
            return 0.0
        
        top_k_actual = actual_ranking[:k]
        top_k_expected = expected_ranking[:k]
        
        relevant_in_top_k = len(set(top_k_actual) & set(top_k_expected))
        return relevant_in_top_k / k
    
    def run_molecular_comparison(self, model_path: str = None):
        """Run comprehensive molecular similarity comparison."""
        print("🧪 Molecular Similarity Test (Test 3)")
        print("=" * 60)
        print("Testing quantum vs classical on quantum-native molecular data")
        
        # Create rerankers
        print("\n🤖 Initializing Rerankers...")
        quantum_reranker = self.create_quantum_reranker(model_path)
        classical_reranker = self.create_classical_baseline()
        print("✅ Quantum reranker created")
        print("✅ Classical baseline created")
        
        # Get molecular scenarios
        scenarios = self.dataset.get_scenarios()
        print(f"\n🧬 Testing {len(scenarios)} molecular similarity scenarios")
        
        all_results = {}
        
        for scenario in scenarios:
            scenario_name = scenario["name"]
            print(f"\n📊 Testing {scenario_name}")
            print(f"  Query: {scenario['query'][:80]}...")
            print(f"  Molecules: {len(scenario['molecules'])}")
            
            # Test quantum method
            print("  🔬 Testing Quantum...")
            start_time = time.time()
            quantum_results = self.evaluate_molecular_similarity(quantum_reranker, scenario, "quantum")
            quantum_time = time.time() - start_time
            
            # Test classical method
            print("  📊 Testing Classical...")
            start_time = time.time()
            classical_results = self.evaluate_molecular_similarity(classical_reranker, scenario, "classical")
            classical_time = time.time() - start_time
            
            # Compare results
            if quantum_results.get("success") and classical_results.get("success"):
                q_ndcg = quantum_results["ndcg"]
                c_ndcg = classical_results["ndcg"]
                q_similar_top3 = quantum_results["similar_in_top3"]
                c_similar_top3 = classical_results["similar_in_top3"]
                
                improvement_ndcg = ((q_ndcg - c_ndcg) / c_ndcg * 100) if c_ndcg > 0 else 0
                
                print(f"  📈 Quantum: NDCG={q_ndcg:.3f}, Similar-in-Top3={q_similar_top3} ({quantum_time:.1f}s)")
                print(f"  📈 Classical: NDCG={c_ndcg:.3f}, Similar-in-Top3={c_similar_top3} ({classical_time:.1f}s)")
                print(f"  🎯 NDCG Improvement: {improvement_ndcg:+.1f}%")
                
                all_results[scenario_name] = {
                    "quantum": quantum_results,
                    "classical": classical_results,
                    "quantum_time": quantum_time,
                    "classical_time": classical_time,
                    "improvement_ndcg": improvement_ndcg
                }
            else:
                print(f"  ❌ Failed to evaluate {scenario_name}")
                all_results[scenario_name] = {"error": "Evaluation failed"}
        
        return all_results
    
    def analyze_molecular_results(self, results):
        """Analyze molecular similarity results."""
        print(f"\n🔬 MOLECULAR SIMILARITY ANALYSIS")
        print("=" * 60)
        
        # Collect metrics
        improvements = []
        quantum_ndcgs = []
        classical_ndcgs = []
        quantum_similar_scores = []
        classical_similar_scores = []
        
        print(f"\n{'Scenario':<25} {'Q-NDCG':<8} {'C-NDCG':<8} {'Q-Sim':<6} {'C-Sim':<6} {'Improve'}")
        print("-" * 70)
        
        for scenario_name, data in results.items():
            if "error" not in data:
                q_ndcg = data["quantum"]["ndcg"]
                c_ndcg = data["classical"]["ndcg"]
                q_sim = data["quantum"]["similar_in_top3"]
                c_sim = data["classical"]["similar_in_top3"]
                improvement = data["improvement_ndcg"]
                
                improvements.append(improvement)
                quantum_ndcgs.append(q_ndcg)
                classical_ndcgs.append(c_ndcg)
                quantum_similar_scores.append(q_sim)
                classical_similar_scores.append(c_sim)
                
                print(f"{scenario_name:<25} {q_ndcg:<8.3f} {c_ndcg:<8.3f} {q_sim:<6} {c_sim:<6} {improvement:+.1f}%")
        
        if improvements:
            avg_improvement = np.mean(improvements)
            avg_q_ndcg = np.mean(quantum_ndcgs)
            avg_c_ndcg = np.mean(classical_ndcgs)
            avg_q_similar = np.mean(quantum_similar_scores)
            avg_c_similar = np.mean(classical_similar_scores)
            
            print(f"\n📊 SUMMARY STATISTICS:")
            print(f"  Average NDCG Improvement: {avg_improvement:+.1f}%")
            print(f"  Quantum Average NDCG: {avg_q_ndcg:.3f}")
            print(f"  Classical Average NDCG: {avg_c_ndcg:.3f}")
            print(f"  Quantum Similar-in-Top3: {avg_q_similar:.1f}")
            print(f"  Classical Similar-in-Top3: {avg_c_similar:.1f}")
            
            # Determine verdict
            significant_improvements = sum(1 for imp in improvements if imp > 5)
            molecular_advantage = significant_improvements > 0 or avg_improvement > 3
            
            print(f"\n💡 VERDICT:")
            print("-" * 25)
            
            if molecular_advantage:
                print("✅ Quantum advantage found in molecular similarity!")
                print(f"🧬 Scenarios with >5% improvement: {significant_improvements}/{len(improvements)}")
                print(f"🎯 RECOMMENDATION: Focus on molecular/chemical applications")
                verdict = "MOLECULAR_ADVANTAGE"
            else:
                print("❌ No significant quantum advantage in molecular domain")
                print("🔄 RECOMMENDATION: Investigate other quantum-native domains")
                verdict = "NO_MOLECULAR_ADVANTAGE"
            
            return {
                "molecular_advantage": molecular_advantage,
                "avg_improvement": avg_improvement,
                "significant_improvements": significant_improvements,
                "verdict": verdict
            }
        
        return {"molecular_advantage": False, "verdict": "INSUFFICIENT_DATA"}

def main():
    """Main function for molecular similarity test."""
    print("🚀 Starting Molecular Similarity Test (Test 3)")
    print("Testing quantum advantage on quantum-native molecular data")
    
    # Check for trained model
    model_path = "models/qpmel_extended.pt"
    if not Path(model_path).exists():
        print(f"⚠️  Using default quantum parameters")
        model_path = None
    else:
        print(f"✅ Using trained QPMeL model")
    
    # Run evaluation
    evaluator = MolecularSimilarityEvaluator()
    
    print("\n⏱️  Starting molecular similarity evaluation...")
    start_time = time.time()
    
    results = evaluator.run_molecular_comparison(model_path)
    
    total_time = time.time() - start_time
    
    # Analyze results
    analysis = evaluator.analyze_molecular_results(results)
    
    # Save results
    output_data = {
        "test_name": "molecular_similarity",
        "total_time_seconds": total_time,
        "results": results,
        "analysis": analysis,
        "timestamp": time.time()
    }
    
    with open("molecular_similarity_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to molecular_similarity_results.json")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    
    # Final decision
    if analysis.get("molecular_advantage"):
        print(f"\n🎉 QUANTUM ADVANTAGE IN MOLECULAR DOMAIN!")
        print(f"🧬 Average improvement: {analysis['avg_improvement']:+.1f}%")
        print(f"🚀 CONCLUSION: Quantum reranker has found its niche!")
    else:
        print(f"\n🔄 NO MOLECULAR ADVANTAGE")
        print(f"📝 But we found robustness advantages in Test 2")
        print(f"🎯 CONCLUSION: Quantum benefits in adversarial/noise scenarios")
    
    print(f"\n✨ Test 3 Complete!")
    return analysis.get("molecular_advantage", False)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test quantum reranker as a secondary reranker on top of classical methods.
Focus on biomedical/scientific domain where QPMeL was trained (NFCorpus-like data).
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum_rerank.core.rag_reranker import QuantumRAGReranker
from quantum_rerank.core.quantum_similarity_engine import SimilarityEngineConfig, SimilarityMethod
from quantum_rerank.training.qpmel_trainer import QPMeLTrainer, QPMeLTrainingConfig
from quantum_rerank.ml.qpmel_circuits import QPMeLConfig

def create_biomedical_nfcorpus_scenarios() -> List[Dict]:
    """Create biomedical/scientific scenarios similar to NFCorpus training data."""
    return [
        {
            "name": "Biomedical Research Query",
            "query": "What are the molecular mechanisms of Alzheimer's disease pathogenesis and potential therapeutic targets?",
            "documents": [
                # Highly relevant - specific to query
                "Alzheimer's disease pathogenesis involves amyloid-beta plaque formation and tau protein hyperphosphorylation, leading to neuronal death. Key therapeutic targets include beta-secretase inhibitors and gamma-secretase modulators.",
                "The molecular mechanisms of Alzheimer's disease include aberrant protein aggregation, oxidative stress, and neuroinflammation. Novel therapeutic approaches target amyloid clearance and tau stabilization.",
                
                # Moderately relevant - related but broader
                "Neurodegenerative diseases involve protein misfolding and aggregation, including alpha-synuclein in Parkinson's disease and huntingtin in Huntington's disease, sharing common pathological pathways.",
                "Therapeutic strategies for neurodegenerative disorders focus on neuroprotection, protein aggregation inhibition, and inflammation modulation through various pharmacological interventions.",
                
                # Lower relevance - general medical
                "Cardiovascular disease prevention involves lifestyle modifications including diet, exercise, and medication management to reduce risk factors such as hypertension and hyperlipidemia.",
                "Cancer immunotherapy harnesses the immune system to target malignant cells through checkpoint inhibitors, CAR-T cell therapy, and monoclonal antibodies.",
                
                # Irrelevant
                "Machine learning algorithms in computer vision use convolutional neural networks for image classification and object detection in autonomous vehicle systems.",
                "Renewable energy technologies including solar panels and wind turbines are essential for sustainable power generation and reducing carbon emissions."
            ],
            "expected_ranking": [0, 1, 2, 3, 4, 5, 6, 7]
        },
        
        {
            "name": "Clinical Research Query", 
            "query": "What are the efficacy and safety profiles of immunomodulatory therapies for multiple sclerosis treatment?",
            "documents": [
                # Highly relevant
                "Immunomodulatory therapies for multiple sclerosis include interferon-beta, glatiramer acetate, and newer agents like natalizumab and fingolimod, showing significant efficacy in reducing relapse rates with manageable safety profiles.",
                "Clinical trials of multiple sclerosis treatments demonstrate that disease-modifying therapies reduce inflammatory lesions and disability progression, though monitoring for potential adverse effects including liver toxicity and infections is essential.",
                
                # Moderately relevant
                "Autoimmune diseases require immunosuppressive treatments that balance therapeutic efficacy with safety concerns, including increased infection risk and potential malignancy development.",
                "Neurological disorders affecting the central nervous system often require long-term management strategies involving both pharmacological and non-pharmacological interventions.",
                
                # Lower relevance
                "Rheumatoid arthritis treatment involves disease-modifying antirheumatic drugs (DMARDs) and biologics that target inflammatory pathways to prevent joint destruction.",
                "Clinical trial design for rare diseases requires adaptive methodologies and biomarker-driven approaches to demonstrate therapeutic efficacy in small patient populations.",
                
                # Irrelevant
                "Agricultural biotechnology uses genetic engineering techniques to develop crop varieties with enhanced yield, disease resistance, and nutritional content.",
                "Blockchain technology applications in supply chain management provide transparency and traceability for goods from manufacturer to consumer."
            ],
            "expected_ranking": [0, 1, 2, 3, 4, 5, 6, 7]
        },
        
        {
            "name": "Pharmaceutical Research Query",
            "query": "How do CRISPR-Cas9 gene editing applications advance precision medicine and what are the current limitations?",
            "documents": [
                # Highly relevant
                "CRISPR-Cas9 gene editing enables precise modification of disease-causing mutations, advancing precision medicine through personalized therapeutic approaches, though delivery challenges and off-target effects remain significant limitations.",
                "Precision medicine applications of CRISPR include treating sickle cell disease, beta-thalassemia, and inherited blindness, with clinical trials demonstrating therapeutic potential while addressing safety and ethical considerations.",
                
                # Moderately relevant
                "Gene therapy approaches using viral vectors and novel delivery systems show promise for treating genetic disorders, though manufacturing challenges and immune responses present ongoing obstacles.",
                "Personalized medicine relies on genomic profiling and biomarker identification to tailor treatments to individual patients, improving therapeutic outcomes and reducing adverse effects.",
                
                # Lower relevance
                "Stem cell research investigates regenerative medicine applications for treating degenerative diseases, though ethical considerations and differentiation control remain challenging.",
                "Pharmacogenomics studies how genetic variations affect drug metabolism and response, enabling optimized dosing and drug selection for individual patients.",
                
                # Irrelevant
                "Quantum computing algorithms may eventually accelerate drug discovery through molecular simulation and optimization of pharmaceutical compound design.",
                "Sustainable manufacturing processes in the textile industry focus on reducing water consumption and chemical waste through innovative production technologies."
            ],
            "expected_ranking": [0, 1, 2, 3, 4, 5, 6, 7]
        }
    ]

def create_classical_baseline(documents: List[str], query: str) -> List[Tuple[str, float]]:
    """Create classical baseline ranking using cosine similarity."""
    # Create classical reranker
    classical_config = SimilarityEngineConfig(
        n_qubits=2,
        n_layers=1,
        similarity_method=SimilarityMethod.CLASSICAL_COSINE,
        enable_caching=True
    )
    classical_reranker = QuantumRAGReranker(config=classical_config)
    
    # Get classical ranking
    classical_results = classical_reranker.rerank(query, documents, method="classical", top_k=len(documents))
    
    # Return document text and scores
    ranked_docs = []
    for result in classical_results:
        doc_text = result.get('text', result.get('content', ''))
        score = result.get('score', result.get('similarity', 0.0))
        ranked_docs.append((doc_text, score))
    
    return ranked_docs

def test_quantum_as_secondary_reranker(scenario: Dict, quantum_reranker: QuantumRAGReranker) -> Dict:
    """Test quantum reranker as secondary reranker on top of classical baseline."""
    query = scenario["query"]
    documents = scenario["documents"]
    expected_ranking = scenario["expected_ranking"]
    
    print(f"  📋 Testing: {scenario['name']}")
    print(f"    Query: {query[:80]}...")
    
    # Step 1: Get classical baseline ranking
    print("    🔍 Step 1: Classical baseline ranking...")
    start_time = time.time()
    classical_ranked = create_classical_baseline(documents, query)
    classical_time = (time.time() - start_time) * 1000
    
    # Extract top-k documents from classical ranking for quantum reranking
    top_k_for_reranking = 6  # Rerank top 6 from classical
    classical_top_docs = [doc for doc, score in classical_ranked[:top_k_for_reranking]]
    classical_remaining = [doc for doc, score in classical_ranked[top_k_for_reranking:]]
    
    # Step 2: Apply quantum reranking to top-k classical results
    print(f"    ⚛️  Step 2: Quantum reranking top {top_k_for_reranking} classical results...")
    start_time = time.time()
    quantum_results = quantum_reranker.rerank(query, classical_top_docs, method="quantum", top_k=len(classical_top_docs))
    quantum_time = (time.time() - start_time) * 1000
    
    # Step 3: Combine quantum reranked top-k with remaining classical results
    final_ranking = []
    
    # Add quantum reranked results
    for result in quantum_results:
        doc_text = result.get('text', result.get('content', ''))
        final_ranking.append(doc_text)
    
    # Add remaining classical results
    final_ranking.extend(classical_remaining)
    
    # Calculate metrics
    def find_original_index(doc_text, original_docs):
        for i, original_doc in enumerate(original_docs):
            if original_doc == doc_text or doc_text in original_doc or original_doc in doc_text:
                return i
        return -1
    
    # Get rankings for comparison
    classical_only_ranking = []
    hybrid_ranking = []
    
    for doc, score in classical_ranked:
        idx = find_original_index(doc, documents)
        if idx != -1:
            classical_only_ranking.append(idx)
    
    for doc in final_ranking:
        idx = find_original_index(doc, documents)
        if idx != -1:
            hybrid_ranking.append(idx)
    
    # Calculate metrics
    def calculate_metrics(ranking, expected):
        if not ranking:
            return {"ndcg": 0.0, "kendall_tau": 0.0, "precision_3": 0.0}
        
        # NDCG calculation
        relevance_scores = []
        for i, actual_idx in enumerate(ranking):
            # Higher relevance for items that should be ranked higher
            if actual_idx < len(expected):
                relevance = max(0, len(expected) - expected[actual_idx])
            else:
                relevance = 0
            relevance_scores.append(relevance)
        
        # DCG
        dcg = relevance_scores[0] if relevance_scores else 0
        for i in range(1, len(relevance_scores)):
            dcg += relevance_scores[i] / np.log2(i + 2)
        
        # IDCG
        ideal_scores = sorted([len(expected) - i for i in range(len(expected))], reverse=True)
        idcg = ideal_scores[0] if ideal_scores else 0
        for i in range(1, min(len(ideal_scores), len(relevance_scores))):
            idcg += ideal_scores[i] / np.log2(i + 2)
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        # Kendall tau
        n = min(len(ranking), len(expected))
        concordant = 0
        discordant = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if i < len(ranking) and j < len(ranking):
                    actual_order = ranking[i] < ranking[j]
                    expected_order = expected[i] < expected[j]
                    
                    if actual_order == expected_order:
                        concordant += 1
                    else:
                        discordant += 1
        
        total_pairs = n * (n - 1) // 2
        kendall_tau = (concordant - discordant) / total_pairs if total_pairs > 0 else 0.0
        
        # Precision@3
        top_3_expected = set(expected[:3])
        top_3_actual = set(ranking[:3]) if len(ranking) >= 3 else set(ranking)
        precision_3 = len(top_3_expected & top_3_actual) / 3.0
        
        return {
            "ndcg": ndcg,
            "kendall_tau": kendall_tau, 
            "precision_3": precision_3
        }
    
    classical_metrics = calculate_metrics(classical_only_ranking, expected_ranking)
    hybrid_metrics = calculate_metrics(hybrid_ranking, expected_ranking)
    
    print(f"    📊 Classical only: NDCG={classical_metrics['ndcg']:.3f}, τ={classical_metrics['kendall_tau']:.3f}, P@3={classical_metrics['precision_3']:.3f}")
    print(f"    📊 Classical+Quantum: NDCG={hybrid_metrics['ndcg']:.3f}, τ={hybrid_metrics['kendall_tau']:.3f}, P@3={hybrid_metrics['precision_3']:.3f}")
    
    # Calculate improvement
    improvements = {}
    for metric in ['ndcg', 'kendall_tau', 'precision_3']:
        if classical_metrics[metric] > 0:
            improvement = ((hybrid_metrics[metric] - classical_metrics[metric]) / classical_metrics[metric]) * 100
        else:
            improvement = 0.0
        improvements[metric] = improvement
        
    if any(imp > 0 for imp in improvements.values()):
        print(f"    ✅ Improvements: NDCG={improvements['ndcg']:+.1f}%, τ={improvements['kendall_tau']:+.1f}%, P@3={improvements['precision_3']:+.1f}%")
    else:
        print(f"    ❌ No improvement over classical baseline")
    
    return {
        "classical_metrics": classical_metrics,
        "hybrid_metrics": hybrid_metrics,
        "improvements": improvements,
        "classical_time_ms": classical_time,
        "quantum_time_ms": quantum_time,
        "total_time_ms": classical_time + quantum_time,
        "classical_ranking": classical_only_ranking,
        "hybrid_ranking": hybrid_ranking,
        "reranked_docs": top_k_for_reranking
    }

def main():
    """Main evaluation function."""
    print("🔬 Quantum as Secondary Reranker Evaluation")
    print("Testing on NFCorpus-like biomedical/scientific data")
    print("=" * 65)
    
    # Create biomedical test scenarios
    scenarios = create_biomedical_nfcorpus_scenarios()
    print(f"\n📝 Created {len(scenarios)} biomedical test scenarios")
    for scenario in scenarios:
        print(f"   • {scenario['name']}: {len(scenario['documents'])} documents")
    
    # Load trained quantum reranker (extended model)
    print(f"\n🧠 Loading Extended QPMeL Model...")
    try:
        config = QPMeLTrainingConfig(
            qpmel_config=QPMeLConfig(n_qubits=2, n_layers=1),
            batch_size=8
        )
        trainer = QPMeLTrainer(config=config)
        trainer.load_model("models/qpmel_extended.pt")
        quantum_reranker = trainer.get_trained_reranker()
        print("✅ Extended QPMeL model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load extended model: {e}")
        return
    
    # Run evaluation
    all_results = {}
    total_improvements = {"ndcg": [], "kendall_tau": [], "precision_3": []}
    
    print(f"\n🧪 Testing Quantum as Secondary Reranker")
    print("-" * 50)
    
    for scenario in scenarios:
        print(f"\n🎯 Scenario: {scenario['name']}")
        results = test_quantum_as_secondary_reranker(scenario, quantum_reranker)
        all_results[scenario['name']] = results
        
        # Collect improvements
        for metric in total_improvements:
            total_improvements[metric].append(results['improvements'][metric])
    
    # Generate final report
    print(f"\n📊 FINAL EVALUATION: Quantum as Secondary Reranker")
    print("=" * 65)
    
    # Calculate average improvements
    avg_improvements = {}
    for metric in total_improvements:
        if total_improvements[metric]:
            avg_improvements[metric] = np.mean(total_improvements[metric])
        else:
            avg_improvements[metric] = 0.0
    
    print(f"\n🎯 Average Performance Improvements:")
    print(f"   NDCG: {avg_improvements['ndcg']:+.1f}%")
    print(f"   Kendall-τ: {avg_improvements['kendall_tau']:+.1f}%") 
    print(f"   Precision@3: {avg_improvements['precision_3']:+.1f}%")
    
    # Determine if quantum reranking is beneficial
    significant_improvement_threshold = 1.0  # 1% improvement
    beneficial_metrics = sum(1 for imp in avg_improvements.values() if imp > significant_improvement_threshold)
    
    print(f"\n💡 Key Insights:")
    print("-" * 35)
    
    if beneficial_metrics >= 2:
        print("✅ Quantum reranking provides significant benefits as secondary reranker")
        print("   Recommendation: Deploy quantum reranker on top of classical baseline")
    elif beneficial_metrics == 1:
        print("⚠️  Quantum reranking provides modest benefits")
        print("   Recommendation: Consider deployment if computational cost is acceptable")
    else:
        print("❌ Quantum reranking does not provide significant benefits")
        print("   Recommendation: Stick with classical baseline for now")
    
    # Performance analysis
    avg_classical_time = np.mean([results['classical_time_ms'] for results in all_results.values()])
    avg_quantum_time = np.mean([results['quantum_time_ms'] for results in all_results.values()])
    avg_total_time = np.mean([results['total_time_ms'] for results in all_results.values()])
    
    print(f"\n⚡ Performance Analysis:")
    print(f"   Classical baseline: {avg_classical_time:.1f}ms")
    print(f"   Quantum reranking: {avg_quantum_time:.1f}ms")
    print(f"   Total pipeline: {avg_total_time:.1f}ms")
    print(f"   Overhead: {((avg_total_time - avg_classical_time) / avg_classical_time * 100):+.1f}%")
    
    # Scenario breakdown
    print(f"\n📋 Detailed Results by Scenario:")
    print("-" * 50)
    
    for scenario_name, results in all_results.items():
        print(f"\n🎯 {scenario_name}")
        classical = results['classical_metrics']
        hybrid = results['hybrid_metrics']
        improvements = results['improvements']
        
        print(f"   Classical:     NDCG={classical['ndcg']:.3f}, τ={classical['kendall_tau']:.3f}, P@3={classical['precision_3']:.3f}")
        print(f"   Classical+QM:  NDCG={hybrid['ndcg']:.3f}, τ={hybrid['kendall_tau']:.3f}, P@3={hybrid['precision_3']:.3f}")
        print(f"   Improvement:   NDCG={improvements['ndcg']:+.1f}%, τ={improvements['kendall_tau']:+.1f}%, P@3={improvements['precision_3']:+.1f}%")
    
    # Save results
    with open('quantum_secondary_reranker_results.json', 'w') as f:
        json.dump({
            'summary': {
                'average_improvements': avg_improvements,
                'beneficial_metrics': beneficial_metrics,
                'avg_classical_time_ms': avg_classical_time,
                'avg_quantum_time_ms': avg_quantum_time,
                'avg_total_time_ms': avg_total_time
            },
            'scenarios': all_results
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to quantum_secondary_reranker_results.json")
    print(f"\n✨ Evaluation Complete!")

if __name__ == "__main__":
    main()
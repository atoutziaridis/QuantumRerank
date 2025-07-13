#!/usr/bin/env python3
"""
Final test: Do our trained quantum models beat classical methods?
This is the ultimate test to see if quantum reranking provides real advantages.
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

def create_challenging_reranking_scenarios() -> List[Dict]:
    """Create scenarios where reranking should show clear differences between methods."""
    return [
        {
            "name": "AI Research Paper Ranking",
            "query": "transformer attention mechanisms for natural language processing",
            "documents": [
                # Highly relevant - should be top ranked
                "Attention mechanisms in transformer architectures enable models to focus on relevant parts of input sequences, revolutionizing natural language processing with self-attention and multi-head attention layers.",
                "The transformer model introduces scaled dot-product attention and multi-head attention mechanisms that allow parallel processing and better handling of long-range dependencies in NLP tasks.",
                
                # Moderately relevant - should be middle ranked
                "Natural language processing techniques including tokenization, parsing, and semantic analysis are fundamental for understanding and generating human language in AI systems.",
                "Deep learning models for NLP use various architectures including recurrent neural networks, convolutional networks, and more recently transformer-based approaches for language understanding.",
                
                # Less relevant - should be lower ranked
                "Computer vision applications use convolutional neural networks for image classification, object detection, and image segmentation tasks in various domains.",
                "Machine learning optimization algorithms like gradient descent, Adam, and RMSprop are essential for training neural networks effectively on large datasets.",
                
                # Irrelevant - should be bottom ranked
                "Database management systems provide ACID properties and efficient query processing for storing and retrieving structured data in enterprise applications.",
                "Web development frameworks like React, Angular, and Vue.js enable building interactive user interfaces for modern web applications with component-based architectures."
            ],
            "expected_ranking": [0, 1, 2, 3, 4, 5, 6, 7]  # Perfect relevance order
        },
        
        {
            "name": "Quantum Computing Literature",
            "query": "quantum error correction codes for fault-tolerant quantum computing",
            "documents": [
                # Highly relevant
                "Quantum error correction codes protect quantum information from decoherence and gate errors, with surface codes being the most promising approach for fault-tolerant quantum computing implementations.",
                "Fault-tolerant quantum computing requires quantum error correction protocols that can detect and correct errors below the fault-tolerance threshold, enabling reliable quantum algorithm execution.",
                
                # Moderately relevant
                "Quantum computing algorithms leverage quantum superposition and entanglement to potentially achieve exponential speedups for specific computational problems like factoring and simulation.",
                "Decoherence in quantum systems poses significant challenges for quantum computing, requiring sophisticated error mitigation and correction strategies to maintain quantum coherence.",
                
                # Less relevant
                "Classical error correction techniques like Reed-Solomon codes and LDPC codes are widely used in telecommunications and data storage systems for reliable information transmission.",
                "Quantum simulation applications use quantum computers to model complex quantum systems that are intractable for classical computers, including molecular dynamics and materials science.",
                
                # Irrelevant
                "Blockchain technology uses cryptographic hash functions and distributed consensus mechanisms to ensure secure and immutable transaction records in decentralized networks.",
                "Artificial intelligence ethics considers the societal implications of AI systems, including bias, fairness, transparency, and accountability in algorithmic decision-making processes."
            ],
            "expected_ranking": [0, 1, 2, 3, 4, 5, 6, 7]
        },
        
        {
            "name": "Software Engineering Concepts",
            "query": "microservices architecture patterns for scalable distributed systems",
            "documents": [
                # Highly relevant
                "Microservices architecture patterns decompose large applications into small, independent services that communicate through well-defined APIs, enabling scalability and maintainability in distributed systems.",
                "Scalable distributed systems design requires careful consideration of service boundaries, data consistency, communication patterns, and fault tolerance in microservices architectures.",
                
                # Moderately relevant
                "Distributed systems engineering involves managing complexity across multiple nodes, handling network partitions, ensuring data consistency, and implementing reliable communication protocols.",
                "Software architecture patterns like event-driven architecture, CQRS, and saga patterns help design robust and maintainable systems that can scale with business requirements.",
                
                # Less relevant
                "Cloud computing platforms provide infrastructure services, container orchestration, and managed databases that support modern application deployment and scaling strategies.",
                "DevOps practices integrate development and operations teams through automation, continuous integration, and infrastructure as code to improve software delivery efficiency.",
                
                # Irrelevant
                "Mobile app development frameworks like React Native and Flutter enable cross-platform development with shared codebases for iOS and Android applications.",
                "Data visualization tools help analysts and scientists create interactive charts, graphs, and dashboards to communicate insights from complex datasets effectively."
            ],
            "expected_ranking": [0, 1, 2, 3, 4, 5, 6, 7]
        }
    ]

def test_reranking_method(reranker: QuantumRAGReranker, scenarios: List[Dict], method: str, method_name: str) -> Dict:
    """Test a specific reranking method on all scenarios."""
    print(f"\n🧪 Testing {method_name}")
    print("-" * 50)
    
    all_metrics = {"ndcg": [], "kendall_tau": [], "precision_3": [], "latency": []}
    scenario_results = {}
    
    for scenario in scenarios:
        query = scenario["query"]
        documents = scenario["documents"]
        expected_ranking = scenario["expected_ranking"]
        
        print(f"  📋 {scenario['name']}")
        
        start_time = time.time()
        try:
            # Rerank documents
            results = reranker.rerank(query, documents, method=method, top_k=len(documents))
            latency = (time.time() - start_time) * 1000
            
            # Calculate ranking quality metrics
            actual_ranking = []
            for result in results:
                doc_text = result.get('text', result.get('content', ''))
                # Find original index
                for i, original_doc in enumerate(documents):
                    if original_doc == doc_text or doc_text in original_doc:
                        actual_ranking.append(i)
                        break
            
            # Calculate metrics
            ndcg = calculate_ndcg_at_k(actual_ranking, expected_ranking, k=len(documents))
            kendall_tau = calculate_kendall_tau(actual_ranking, expected_ranking)
            precision_3 = calculate_precision_at_k(actual_ranking, expected_ranking, k=3)
            
            scenario_results[scenario['name']] = {
                "ndcg": ndcg,
                "kendall_tau": kendall_tau,
                "precision_3": precision_3,
                "latency_ms": latency,
                "actual_ranking": actual_ranking,
                "expected_ranking": expected_ranking
            }
            
            all_metrics["ndcg"].append(ndcg)
            all_metrics["kendall_tau"].append(kendall_tau)
            all_metrics["precision_3"].append(precision_3)
            all_metrics["latency"].append(latency)
            
            print(f"    NDCG: {ndcg:.3f}, Kendall-τ: {kendall_tau:.3f}, P@3: {precision_3:.3f}, Latency: {latency:.1f}ms")
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            scenario_results[scenario['name']] = None
    
    # Calculate averages
    if all_metrics["ndcg"]:
        summary = {
            "avg_ndcg": np.mean(all_metrics["ndcg"]),
            "avg_kendall_tau": np.mean(all_metrics["kendall_tau"]),
            "avg_precision_3": np.mean(all_metrics["precision_3"]),
            "avg_latency_ms": np.mean(all_metrics["latency"]),
            "num_scenarios": len(all_metrics["ndcg"])
        }
        
        print(f"  📊 Average: NDCG={summary['avg_ndcg']:.3f}, τ={summary['avg_kendall_tau']:.3f}, P@3={summary['avg_precision_3']:.3f}")
        
        return {
            "summary": summary,
            "scenarios": scenario_results
        }
    
    return {"summary": None, "scenarios": scenario_results}

def calculate_ndcg_at_k(actual_ranking: List[int], expected_ranking: List[int], k: int) -> float:
    """Calculate NDCG@k for ranking quality."""
    if not actual_ranking or k <= 0:
        return 0.0
    
    # Create relevance scores based on expected ranking
    relevance_scores = []
    for i in range(min(k, len(actual_ranking))):
        actual_pos = actual_ranking[i]
        # Higher score for items that should be ranked higher
        relevance = max(0, len(expected_ranking) - expected_ranking.index(actual_pos) if actual_pos in expected_ranking else 0)
        relevance_scores.append(relevance)
    
    # Calculate DCG
    dcg = relevance_scores[0] if relevance_scores else 0
    for i in range(1, len(relevance_scores)):
        dcg += relevance_scores[i] / np.log2(i + 2)
    
    # Calculate IDCG (perfect ranking)
    ideal_scores = sorted([len(expected_ranking) - i for i in range(len(expected_ranking))], reverse=True)[:k]
    idcg = ideal_scores[0] if ideal_scores else 0
    for i in range(1, len(ideal_scores)):
        idcg += ideal_scores[i] / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0

def calculate_kendall_tau(actual_ranking: List[int], expected_ranking: List[int]) -> float:
    """Calculate Kendall's tau rank correlation."""
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

def calculate_precision_at_k(actual_ranking: List[int], expected_ranking: List[int], k: int) -> float:
    """Calculate precision@k - how many of top-k are actually relevant."""
    if not actual_ranking or k <= 0:
        return 0.0
    
    top_k_actual = actual_ranking[:k]
    top_k_expected = expected_ranking[:k]
    
    relevant_in_top_k = len(set(top_k_actual) & set(top_k_expected))
    return relevant_in_top_k / k

def main():
    """Main comparison function."""
    print("🎯 Final Test: Quantum vs Classical Reranking Performance")
    print("=" * 65)
    print("Testing if trained quantum models beat classical methods on challenging scenarios")
    
    # Create challenging scenarios
    scenarios = create_challenging_reranking_scenarios()
    print(f"\n📝 Created {len(scenarios)} challenging reranking scenarios")
    for scenario in scenarios:
        print(f"   • {scenario['name']}: {len(scenario['documents'])} documents")
    
    # Test methods
    methods_to_test = [
        {
            "reranker_type": "classical",
            "name": "Classical Cosine Similarity",
            "method": "classical",
            "config": SimilarityEngineConfig(
                n_qubits=2,
                n_layers=1,
                similarity_method=SimilarityMethod.CLASSICAL_COSINE,
                enable_caching=True
            )
        },
        {
            "reranker_type": "quantum_untrained", 
            "name": "Quantum Untrained (Random)",
            "method": "quantum",
            "config": SimilarityEngineConfig(
                n_qubits=2,
                n_layers=1,
                similarity_method=SimilarityMethod.QUANTUM_FIDELITY,
                enable_caching=True
            )
        },
        {
            "reranker_type": "quantum_trained_mvp",
            "name": "Quantum Trained (MVP)",
            "method": "quantum",
            "model_path": "models/qpmel_mvp.pt"
        },
        {
            "reranker_type": "quantum_trained_extended",
            "name": "Quantum Trained (Extended)",
            "method": "quantum", 
            "model_path": "models/qpmel_extended.pt"
        }
    ]
    
    all_results = {}
    
    for method_config in methods_to_test:
        method_name = method_config["name"]
        print(f"\n🔄 Initializing {method_name}...")
        
        try:
            if method_config["reranker_type"].startswith("quantum_trained"):
                # Create trained quantum reranker
                config = QPMeLTrainingConfig(
                    qpmel_config=QPMeLConfig(n_qubits=2, n_layers=1),
                    batch_size=8
                )
                trainer = QPMeLTrainer(config=config)
                trainer.load_model(method_config["model_path"])
                reranker = trainer.get_trained_reranker()
            else:
                # Create classical or untrained quantum reranker
                reranker = QuantumRAGReranker(config=method_config["config"])
            
            # Test the method
            results = test_reranking_method(reranker, scenarios, method_config["method"], method_name)
            all_results[method_name] = results
            
        except Exception as e:
            print(f"❌ Failed to test {method_name}: {e}")
            all_results[method_name] = {"summary": None, "scenarios": {}}
    
    # Generate final comparison report
    print(f"\n📊 FINAL RESULTS: Quantum vs Classical Reranking")
    print("=" * 65)
    
    # Summary table
    print(f"\n{'Method':<30} {'NDCG':<8} {'Kendall-τ':<10} {'P@3':<8} {'Latency':<10}")
    print("-" * 65)
    
    method_scores = {}
    for method_name, results in all_results.items():
        if results["summary"]:
            summary = results["summary"]
            ndcg = summary["avg_ndcg"]
            tau = summary["avg_kendall_tau"]
            p3 = summary["avg_precision_3"]
            latency = summary["avg_latency_ms"]
            
            method_scores[method_name] = {
                "ndcg": ndcg,
                "tau": tau,
                "p3": p3,
                "latency": latency
            }
            
            print(f"{method_name:<30} {ndcg:<8.3f} {tau:<10.3f} {p3:<8.3f} {latency:<10.1f}ms")
        else:
            print(f"{method_name:<30} {'FAILED':<8} {'FAILED':<10} {'FAILED':<8} {'FAILED':<10}")
    
    # Find winners
    if method_scores:
        print(f"\n🏆 Performance Rankings:")
        print("-" * 35)
        
        # Best by NDCG
        best_ndcg = max(method_scores.keys(), key=lambda x: method_scores[x]["ndcg"])
        best_ndcg_score = method_scores[best_ndcg]["ndcg"]
        print(f"🥇 Best NDCG: {best_ndcg} ({best_ndcg_score:.3f})")
        
        # Best by Kendall-tau
        best_tau = max(method_scores.keys(), key=lambda x: method_scores[x]["tau"])
        best_tau_score = method_scores[best_tau]["tau"]
        print(f"🥇 Best Kendall-τ: {best_tau} ({best_tau_score:.3f})")
        
        # Best by Precision@3
        best_p3 = max(method_scores.keys(), key=lambda x: method_scores[x]["p3"])
        best_p3_score = method_scores[best_p3]["p3"]
        print(f"🥇 Best P@3: {best_p3} ({best_p3_score:.3f})")
        
        # Compare quantum vs classical
        classical_methods = [k for k in method_scores.keys() if "Classical" in k]
        quantum_methods = [k for k in method_scores.keys() if "Quantum" in k]
        
        if classical_methods and quantum_methods:
            classical_best_ndcg = max([method_scores[m]["ndcg"] for m in classical_methods])
            quantum_best_ndcg = max([method_scores[m]["ndcg"] for m in quantum_methods])
            
            print(f"\n🔬 Quantum vs Classical Analysis:")
            print(f"   Best Classical NDCG: {classical_best_ndcg:.3f}")
            print(f"   Best Quantum NDCG: {quantum_best_ndcg:.3f}")
            
            if quantum_best_ndcg > classical_best_ndcg:
                improvement = ((quantum_best_ndcg - classical_best_ndcg) / classical_best_ndcg * 100)
                print(f"   ✅ Quantum wins by {improvement:.1f}%!")
            elif quantum_best_ndcg < classical_best_ndcg:
                decline = ((classical_best_ndcg - quantum_best_ndcg) / classical_best_ndcg * 100)
                print(f"   ❌ Classical wins by {decline:.1f}%")
            else:
                print(f"   ➖ Tie between quantum and classical")
    
    # Save results
    with open('final_quantum_vs_classical_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to final_quantum_vs_classical_results.json")
    print(f"\n✨ Final Evaluation Complete!")

if __name__ == "__main__":
    main()
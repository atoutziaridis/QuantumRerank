#!/usr/bin/env python3
"""
Test specific benefits of trained QPMeL models vs random parameters.
This test replaces the parameter predictor with trained QPMeL models.
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
from quantum_rerank.core.quantum_similarity_engine import QuantumSimilarityEngine, SimilarityEngineConfig, SimilarityMethod
from quantum_rerank.training.qpmel_trainer import QPMeLTrainer, QPMeLTrainingConfig
from quantum_rerank.ml.qpmel_circuits import QPMeLConfig

def create_challenging_scenarios() -> List[Dict]:
    """Create scenarios where quantum advantages should be most apparent."""
    return [
        {
            "name": "Semantic Similarity Challenge",
            "query": "quantum machine learning algorithms for optimization",
            "documents": [
                # Highly relevant (quantum + ML + optimization)
                "Variational quantum algorithms optimize objective functions using parameterized quantum circuits combined with classical optimization techniques for machine learning applications.",
                "Quantum approximate optimization algorithm (QAOA) leverages quantum superposition and entanglement to solve combinatorial optimization problems with potential exponential speedup.",
                
                # Relevant (partial overlap)
                "Machine learning optimization techniques including gradient descent, Adam optimizer, and evolutionary algorithms for training neural networks and deep learning models.",
                "Quantum computing applications in cryptography and simulation use quantum superposition and entanglement to achieve computational advantages over classical methods.",
                
                # Less relevant 
                "Classical optimization algorithms like simulated annealing and genetic algorithms solve complex optimization problems in engineering and operations research.",
                "Deep learning frameworks such as TensorFlow and PyTorch provide tools for building and training machine learning models on large datasets.",
                
                # Irrelevant
                "Weather forecasting models use atmospheric data and computational fluid dynamics to predict temperature, precipitation, and storm patterns.",
                "Database optimization techniques improve query performance through indexing, partitioning, and storage engine tuning for large-scale data systems."
            ],
            "expected_relevant": [0, 1, 2, 3]  
        },
        
        {
            "name": "Technical Concept Matching",
            "query": "quantum error correction surface codes decoherence",
            "documents": [
                # Highly relevant
                "Surface codes are topological quantum error correction codes that protect quantum information from decoherence by encoding logical qubits in a 2D lattice of physical qubits.",
                "Quantum error correction protocols detect and correct errors caused by decoherence and gate imperfections using stabilizer measurements and syndrome decoding.",
                
                # Moderately relevant  
                "Decoherence in quantum systems causes loss of quantum coherence due to environmental interactions, limiting the fidelity of quantum computations and information storage.",
                "Fault-tolerant quantum computing requires quantum error correction codes with error rates below the fault-tolerance threshold to enable reliable quantum algorithms.",
                
                # Less relevant
                "Classical error correction codes like Reed-Solomon and LDPC codes protect data transmission and storage from noise and errors in communication systems.",
                "Quantum algorithms for factoring and search problems demonstrate potential quantum advantages but require error-corrected quantum computers for practical implementation.",
                
                # Irrelevant
                "Software debugging techniques help developers identify and fix coding errors in applications using debuggers, unit tests, and code review processes.",
                "Network error handling protocols ensure reliable data transmission over the internet using checksums, acknowledgments, and retransmission mechanisms."
            ],
            "expected_relevant": [0, 1, 2, 3]
        }
    ]

def create_reranker_with_trained_model(model_path: str, model_name: str) -> QuantumRAGReranker:
    """Create a reranker using a specific trained QPMeL model."""
    try:
        # Load trained QPMeL model
        config = QPMeLTrainingConfig(
            qpmel_config=QPMeLConfig(n_qubits=2, n_layers=1),
            batch_size=8
        )
        trainer = QPMeLTrainer(config=config)
        trainer.load_model(model_path)
        
        # Get the trained reranker
        reranker = trainer.get_trained_reranker()
        reranker.model_name = model_name
        
        return reranker
        
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        return None

def evaluate_reranker(reranker: QuantumRAGReranker, scenarios: List[Dict], model_name: str) -> Dict:
    """Evaluate a reranker on test scenarios."""
    print(f"\n🧪 Testing {model_name}")
    print("-" * 40)
    
    results = {}
    total_metrics = {"ndcg": [], "precision_3": [], "latency": []}
    
    for scenario in scenarios:
        query = scenario["query"]
        documents = scenario["documents"]
        expected_relevant = set(scenario["expected_relevant"])
        
        print(f"  📋 {scenario['name']}")
        
        # Test quantum method (primary focus)
        start_time = time.time()
        try:
            ranked_docs = reranker.rerank(query, documents, method="quantum", top_k=len(documents))
            latency = (time.time() - start_time) * 1000
            
            # Calculate metrics
            relevance_scores = []
            top_3_relevant = 0
            
            for i, result in enumerate(ranked_docs):
                # Find original document index
                doc_text = result.get('text', result.get('content', ''))
                original_idx = None
                for j, original_doc in enumerate(documents):
                    if original_doc == doc_text or doc_text in original_doc:
                        original_idx = j
                        break
                
                # Calculate relevance
                is_relevant = original_idx in expected_relevant if original_idx is not None else False
                relevance_scores.append(1.0 if is_relevant else 0.0)
                
                if i < 3 and is_relevant:
                    top_3_relevant += 1
            
            # Calculate NDCG@3
            ndcg_3 = calculate_ndcg(relevance_scores[:3])
            precision_3 = top_3_relevant / 3.0
            
            scenario_result = {
                "ndcg_3": ndcg_3,
                "precision_3": precision_3,
                "latency_ms": latency,
                "top_3_relevant": top_3_relevant,
                "total_relevant": len(expected_relevant)
            }
            
            results[scenario['name']] = scenario_result
            total_metrics["ndcg"].append(ndcg_3)
            total_metrics["precision_3"].append(precision_3)
            total_metrics["latency"].append(latency)
            
            print(f"    NDCG@3: {ndcg_3:.3f}, P@3: {precision_3:.3f}, Latency: {latency:.1f}ms")
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            results[scenario['name']] = None
    
    # Calculate averages
    if total_metrics["ndcg"]:
        avg_ndcg = np.mean(total_metrics["ndcg"])
        avg_precision = np.mean(total_metrics["precision_3"])
        avg_latency = np.mean(total_metrics["latency"])
        
        results["summary"] = {
            "avg_ndcg_3": avg_ndcg,
            "avg_precision_3": avg_precision,
            "avg_latency_ms": avg_latency,
            "num_scenarios": len(total_metrics["ndcg"])
        }
        
        print(f"  📊 Average: NDCG@3={avg_ndcg:.3f}, P@3={avg_precision:.3f}, Latency={avg_latency:.1f}ms")
    
    return results

def calculate_ndcg(relevance_scores: List[float]) -> float:
    """Calculate Normalized Discounted Cumulative Gain."""
    if not relevance_scores:
        return 0.0
    
    # DCG calculation
    dcg = relevance_scores[0]
    for i in range(1, len(relevance_scores)):
        dcg += relevance_scores[i] / np.log2(i + 2)
    
    # IDCG calculation (perfect ranking)
    sorted_relevance = sorted(relevance_scores, reverse=True)
    idcg = sorted_relevance[0] if sorted_relevance else 0
    for i in range(1, len(sorted_relevance)):
        idcg += sorted_relevance[i] / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0

def main():
    """Main evaluation function."""
    print("🔬 QPMeL Trained Models vs Random Parameters Evaluation")
    print("=" * 65)
    
    # Create challenging test scenarios
    scenarios = create_challenging_scenarios()
    print(f"\n📝 Created {len(scenarios)} challenging test scenarios")
    for scenario in scenarios:
        print(f"   • {scenario['name']}: {len(scenario['documents'])} documents ({len(scenario['expected_relevant'])} relevant)")
    
    # Test models
    models_to_test = [
        ("untrained", "Untrained (Random Parameters)", None),  # Default reranker
        ("models/qpmel_mvp.pt", "MVP (500 triplets, 5 epochs)", None),
        ("models/qpmel_extended.pt", "Extended (2000 triplets, 6+ epochs)", None)
    ]
    
    all_results = {}
    
    for model_path, model_name, _ in models_to_test:
        if model_path == "untrained":
            # Create untrained reranker with random parameters
            from quantum_rerank.core.quantum_similarity_engine import SimilarityEngineConfig, SimilarityMethod
            
            engine_config = SimilarityEngineConfig(
                n_qubits=2,
                n_layers=1,
                similarity_method=SimilarityMethod.QUANTUM_FIDELITY,
                enable_caching=True,
                performance_monitoring=True
            )
            reranker = QuantumRAGReranker(config=engine_config)
            reranker.model_name = model_name
        else:
            # Create trained reranker
            reranker = create_reranker_with_trained_model(model_path, model_name)
        
        if reranker:
            results = evaluate_reranker(reranker, scenarios, model_name)
            all_results[model_name] = results
    
    # Generate comparison report
    print(f"\n📊 Quantum Reranking Performance Comparison")
    print("=" * 65)
    
    # Summary table
    print(f"\n{'Model':<35} {'Avg NDCG@3':<12} {'Avg P@3':<10} {'Avg Latency':<12}")
    print("-" * 65)
    
    baseline_ndcg = None
    for model_name, results in all_results.items():
        if results and "summary" in results:
            summary = results["summary"]
            ndcg = summary["avg_ndcg_3"]
            precision = summary["avg_precision_3"]
            latency = summary["avg_latency_ms"]
            
            if "Untrained" in model_name:
                baseline_ndcg = ndcg
            
            print(f"{model_name:<35} {ndcg:<12.3f} {precision:<10.3f} {latency:<12.1f}ms")
    
    # Calculate improvements
    if baseline_ndcg:
        print(f"\n🎯 Performance Improvements vs Untrained:")
        print("-" * 45)
        
        for model_name, results in all_results.items():
            if results and "summary" in results and "Untrained" not in model_name:
                summary = results["summary"]
                ndcg = summary["avg_ndcg_3"]
                improvement = ((ndcg - baseline_ndcg) / baseline_ndcg * 100) if baseline_ndcg > 0 else 0
                
                if improvement > 0:
                    print(f"✅ {model_name}: +{improvement:.1f}% improvement in NDCG@3")
                elif improvement < 0:
                    print(f"❌ {model_name}: {improvement:.1f}% decrease in NDCG@3")
                else:
                    print(f"➖ {model_name}: No change in NDCG@3")
    
    # Detailed scenario breakdown
    print(f"\n📋 Detailed Results by Scenario:")
    print("-" * 65)
    
    for scenario in scenarios:
        scenario_name = scenario["name"]
        print(f"\n🎯 {scenario_name}")
        
        for model_name, results in all_results.items():
            if results and scenario_name in results and results[scenario_name]:
                metrics = results[scenario_name]
                ndcg = metrics["ndcg_3"]
                precision = metrics["precision_3"]
                latency = metrics["latency_ms"]
                print(f"  {model_name:<30}: NDCG={ndcg:.3f}, P@3={precision:.3f}, {latency:.0f}ms")
    
    # Key insights
    print(f"\n💡 Key Insights:")
    print("-" * 35)
    
    # Find best performing model
    best_model = None
    best_ndcg = 0
    
    for model_name, results in all_results.items():
        if results and "summary" in results:
            ndcg = results["summary"]["avg_ndcg_3"]
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_model = model_name
    
    if best_model:
        print(f"🏆 Best performing model: {best_model} (NDCG@3: {best_ndcg:.3f})")
    
    # Check if training helped
    trained_models = [name for name in all_results.keys() if "triplets" in name]
    if trained_models and baseline_ndcg:
        trained_ndcgs = [all_results[name]["summary"]["avg_ndcg_3"] 
                        for name in trained_models 
                        if "summary" in all_results[name]]
        
        if trained_ndcgs:
            best_trained_ndcg = max(trained_ndcgs)
            improvement = ((best_trained_ndcg - baseline_ndcg) / baseline_ndcg * 100) if baseline_ndcg > 0 else 0
            
            if improvement > 5:
                print(f"✅ QPMeL training provides significant improvement: +{improvement:.1f}%")
            elif improvement > 0:
                print(f"✓ QPMeL training provides modest improvement: +{improvement:.1f}%")
            else:
                print(f"⚠️  QPMeL training did not improve performance significantly")
    
    # Save results
    with open('qpmel_evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Detailed results saved to qpmel_evaluation_results.json")
    
    print(f"\n✨ QPMeL Evaluation Complete!")

if __name__ == "__main__":
    main()
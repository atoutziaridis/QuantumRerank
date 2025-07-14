#!/usr/bin/env python3
"""
Real-world testing of QuantumRerank benefits.
Tests on actual use cases where reranking should provide clear benefits.
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def create_test_scenarios() -> List[Dict]:
    """
    Create realistic test scenarios where reranking should provide clear benefits.
    Each scenario has a query and documents with known relevance rankings.
    """
    
    scenarios = [
        {
            "name": "Technical Documentation Search",
            "query": "How to implement quantum error correction",
            "documents": [
                # Highly relevant
                "Quantum error correction is implemented using stabilizer codes like the surface code. The process involves encoding logical qubits into multiple physical qubits and performing syndrome measurements to detect and correct errors.",
                "Implementation of quantum error correction requires careful design of encoding circuits, syndrome extraction, and classical processing for error correction. Surface codes are the most promising approach for fault-tolerant quantum computing.",
                
                # Moderately relevant  
                "Error correction in classical computing uses parity bits and checksums. While different from quantum error correction, some principles like redundancy and syndrome detection are similar.",
                "Quantum computing faces challenges from decoherence and gate errors. Various approaches including error correction, error mitigation, and decoherence-free subspaces are being researched.",
                
                # Less relevant
                "Machine learning algorithms can be used to optimize quantum circuits and reduce noise. Neural networks have shown promise in learning error patterns in quantum devices.",
                "Classical error correction codes like Hamming codes and Reed-Solomon codes are widely used in telecommunications and data storage systems.",
                
                # Irrelevant
                "Python programming tutorial for beginners covers variables, functions, and basic data structures. This is essential for anyone starting to learn programming.",
                "Climate change affects global weather patterns and requires urgent action from governments worldwide. Temperature increases are causing sea level rise.",
                "Recipe for chocolate chip cookies: mix flour, sugar, eggs, and chocolate chips. Bake at 350°F for 12-15 minutes until golden brown.",
                "Stock market analysis shows volatility in tech stocks. Investors should diversify their portfolios to minimize risk during uncertain economic times."
            ],
            "expected_ranking": [0, 1, 3, 2, 4, 5, 6, 7, 8, 9]  # Indices of documents in relevance order
        },
        
        {
            "name": "Scientific Literature Search", 
            "query": "machine learning applications in drug discovery",
            "documents": [
                # Highly relevant
                "Machine learning accelerates drug discovery by predicting molecular properties, identifying drug targets, and optimizing lead compounds. Deep learning models analyze chemical structures to predict bioactivity and toxicity.",
                "AI-driven drug discovery platforms use machine learning to screen millions of compounds, predict ADMET properties, and identify novel drug-target interactions. This approach reduces development time and costs significantly.",
                "Deep learning applications in pharmaceutical research include protein structure prediction, molecular generation, and clinical trial optimization. These methods are transforming how new drugs are discovered and developed.",
                
                # Moderately relevant
                "Artificial intelligence is revolutionizing healthcare through applications in medical imaging, diagnosis, and personalized treatment. Machine learning algorithms analyze patient data to improve outcomes.",
                "Computational biology uses algorithms to analyze biological data, including genomics, proteomics, and metabolomics. These approaches help understand disease mechanisms and identify therapeutic targets.",
                
                # Less relevant
                "Machine learning finds applications in finance for fraud detection, algorithmic trading, and risk assessment. Financial institutions use AI to analyze market patterns and customer behavior.",
                "Computer vision techniques using deep learning have improved image recognition, object detection, and autonomous vehicle navigation. These systems process visual data to make intelligent decisions.",
                
                # Irrelevant
                "Traditional medicine practices from ancient cultures include herbal remedies and acupuncture. These approaches focus on holistic healing and natural treatments.",
                "Agricultural techniques for sustainable farming include crop rotation, organic fertilizers, and precision agriculture. These methods improve yield while protecting the environment.",
                "Space exploration missions to Mars require advanced propulsion systems, life support technology, and radiation shielding. NASA and private companies are developing new spacecraft designs."
            ],
            "expected_ranking": [0, 1, 2, 4, 3, 5, 6, 7, 8, 9]
        },
        
        {
            "name": "Code Documentation Search",
            "query": "async await python error handling",
            "documents": [
                # Highly relevant
                "Async/await error handling in Python uses try-except blocks with asyncio. You can catch exceptions in async functions and handle them appropriately. Use asyncio.gather() with return_exceptions=True for multiple tasks.",
                "Python asyncio error handling requires understanding how exceptions propagate in async contexts. Use try-except around await statements and consider using asyncio.create_task() with exception handling.",
                "Best practices for async error handling include using context managers, proper exception propagation, and logging. Always handle exceptions in async functions to prevent silent failures.",
                
                # Moderately relevant
                "Python async programming with asyncio enables concurrent execution of I/O-bound tasks. Async functions are defined with 'async def' and called with 'await' keywords.",
                "Error handling in Python uses try-except-finally blocks to catch and manage exceptions. Different exception types can be handled specifically using multiple except clauses.",
                
                # Less relevant  
                "JavaScript async/await syntax provides a cleaner way to handle promises compared to callback functions. Error handling uses try-catch blocks around await expressions.",
                "Python decorators can be used to add functionality to functions, including error handling, logging, and timing. They provide a clean way to modify function behavior.",
                
                # Irrelevant
                "SQL database queries use SELECT, INSERT, UPDATE, and DELETE statements. Proper indexing and query optimization improve database performance significantly.",
                "React components manage state and props to create dynamic user interfaces. JSX syntax combines JavaScript and HTML-like markup for component development.",
                "Docker containers provide isolated environments for applications. Dockerfile defines the container image with dependencies and configuration settings."
            ],
            "expected_ranking": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        }
    ]
    
    return scenarios

def evaluate_reranking_methods(scenarios: List[Dict]) -> Dict[str, Dict]:
    """
    Evaluate different reranking methods on the test scenarios.
    Returns performance metrics for each method.
    """
    
    print("🧪 Testing Reranking Methods on Real-World Scenarios")
    print("="*60)
    
    # Import reranking methods
    try:
        from quantum_rerank.core.rag_reranker import QuantumRAGReranker
        quantum_reranker = QuantumRAGReranker()
        print("✅ QuantumRerank system loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load QuantumRerank: {e}")
        return {}
    
    # Define reranking methods to test
    methods = {
        "QuantumRerank-Classical": lambda q, docs: quantum_reranker.rerank(q, docs, method="classical"),
        "QuantumRerank-Quantum": lambda q, docs: quantum_reranker.rerank(q, docs, method="quantum"), 
        "QuantumRerank-Hybrid": lambda q, docs: quantum_reranker.rerank(q, docs, method="hybrid"),
        "Random-Baseline": lambda q, docs: random_rerank(q, docs)
    }
    
    results = {}
    
    for method_name, rerank_func in methods.items():
        print(f"\n📊 Evaluating {method_name}...")
        method_results = {
            "ndcg_scores": [],
            "kendall_tau_scores": [],
            "precision_at_3": [],
            "avg_time_ms": []
        }
        
        for scenario in scenarios:
            print(f"  Testing scenario: {scenario['name']}")
            
            query = scenario["query"]
            documents = scenario["documents"]
            expected_ranking = scenario["expected_ranking"]
            
            # Time the reranking
            start_time = time.time()
            try:
                if method_name == "Random-Baseline":
                    ranked_results = rerank_func(query, documents)
                else:
                    ranked_results = rerank_func(query, documents)
                
                rerank_time = (time.time() - start_time) * 1000
                method_results["avg_time_ms"].append(rerank_time)
                
                # Calculate metrics
                if ranked_results:
                    # Get the actual ranking produced by the method
                    actual_ranking = get_actual_ranking(ranked_results, documents)
                    
                    # Calculate NDCG
                    ndcg = calculate_ndcg(actual_ranking, expected_ranking)
                    method_results["ndcg_scores"].append(ndcg)
                    
                    # Calculate Kendall's Tau (rank correlation)
                    tau = calculate_kendall_tau(actual_ranking, expected_ranking)
                    method_results["kendall_tau_scores"].append(tau)
                    
                    # Calculate Precision@3
                    p_at_3 = calculate_precision_at_k(actual_ranking, expected_ranking, k=3)
                    method_results["precision_at_3"].append(p_at_3)
                    
                    print(f"    NDCG: {ndcg:.3f}, Kendall-τ: {tau:.3f}, P@3: {p_at_3:.3f}, Time: {rerank_time:.1f}ms")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                continue
        
        # Calculate averages
        if method_results["ndcg_scores"]:
            results[method_name] = {
                "avg_ndcg": sum(method_results["ndcg_scores"]) / len(method_results["ndcg_scores"]),
                "avg_kendall_tau": sum(method_results["kendall_tau_scores"]) / len(method_results["kendall_tau_scores"]),
                "avg_precision_at_3": sum(method_results["precision_at_3"]) / len(method_results["precision_at_3"]),
                "avg_time_ms": sum(method_results["avg_time_ms"]) / len(method_results["avg_time_ms"]),
                "num_scenarios": len(method_results["ndcg_scores"])
            }
        
        print(f"✅ {method_name} evaluation complete")
    
    return results

def random_rerank(query: str, documents: List[str]) -> List[Dict]:
    """Random baseline for comparison."""
    import random
    docs_with_scores = [(doc, random.random()) for doc in documents]
    docs_with_scores.sort(key=lambda x: x[1], reverse=True)
    return [{"text": doc, "score": score, "similarity_score": score} for doc, score in docs_with_scores]

def get_actual_ranking(ranked_results: List[Dict], original_documents: List[str]) -> List[int]:
    """Get the ranking indices produced by a reranking method."""
    ranking = []
    for result in ranked_results:
        result_text = result.get("text", result.get("document", ""))
        # Find the index in original documents
        for i, doc in enumerate(original_documents):
            if doc == result_text:
                ranking.append(i)
                break
    return ranking

def calculate_ndcg(actual_ranking: List[int], expected_ranking: List[int], k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain."""
    import math
    
    # Create relevance scores based on expected ranking (higher rank = higher relevance)
    relevance_scores = {}
    for pos, doc_idx in enumerate(expected_ranking):
        relevance_scores[doc_idx] = len(expected_ranking) - pos
    
    # Calculate DCG for actual ranking
    dcg = 0.0
    for pos, doc_idx in enumerate(actual_ranking[:k]):
        relevance = relevance_scores.get(doc_idx, 0)
        dcg += relevance / math.log2(pos + 2)
    
    # Calculate IDCG (ideal DCG)
    idcg = 0.0
    for pos, doc_idx in enumerate(expected_ranking[:k]):
        relevance = relevance_scores.get(doc_idx, 0)
        idcg += relevance / math.log2(pos + 2)
    
    return dcg / idcg if idcg > 0 else 0.0

def calculate_kendall_tau(actual_ranking: List[int], expected_ranking: List[int]) -> float:
    """Calculate Kendall's Tau rank correlation coefficient."""
    
    # Create position mappings
    actual_positions = {doc_idx: pos for pos, doc_idx in enumerate(actual_ranking)}
    expected_positions = {doc_idx: pos for pos, doc_idx in enumerate(expected_ranking)}
    
    # Count concordant and discordant pairs
    concordant = 0
    discordant = 0
    
    common_docs = set(actual_positions.keys()) & set(expected_positions.keys())
    
    for i, doc1 in enumerate(common_docs):
        for doc2 in list(common_docs)[i+1:]:
            actual_order = actual_positions[doc1] < actual_positions[doc2]
            expected_order = expected_positions[doc1] < expected_positions[doc2]
            
            if actual_order == expected_order:
                concordant += 1
            else:
                discordant += 1
    
    total_pairs = concordant + discordant
    return (concordant - discordant) / total_pairs if total_pairs > 0 else 0.0

def calculate_precision_at_k(actual_ranking: List[int], expected_ranking: List[int], k: int) -> float:
    """Calculate Precision at k."""
    if not actual_ranking or k == 0:
        return 0.0
    
    # Top k from expected ranking are considered relevant
    relevant_docs = set(expected_ranking[:k])
    
    # Check how many of top k actual results are relevant
    top_k_actual = actual_ranking[:k]
    relevant_in_top_k = sum(1 for doc_idx in top_k_actual if doc_idx in relevant_docs)
    
    return relevant_in_top_k / min(k, len(top_k_actual))

def print_results(results: Dict[str, Dict]):
    """Print evaluation results in a formatted table."""
    
    print("\n" + "="*80)
    print("📊 REAL-WORLD RERANKING EVALUATION RESULTS")
    print("="*80)
    
    if not results:
        print("No results to display")
        return
    
    # Table header
    print(f"{'Method':<25} {'NDCG':<8} {'Kendall-τ':<10} {'P@3':<8} {'Time(ms)':<10} {'Scenarios':<10}")
    print("-" * 80)
    
    # Sort by NDCG score
    sorted_methods = sorted(results.items(), key=lambda x: x[1]["avg_ndcg"], reverse=True)
    
    for method_name, metrics in sorted_methods:
        print(f"{method_name:<25} "
              f"{metrics['avg_ndcg']:<8.4f} "
              f"{metrics['avg_kendall_tau']:<10.4f} "
              f"{metrics['avg_precision_at_3']:<8.4f} "
              f"{metrics['avg_time_ms']:<10.1f} "
              f"{metrics['num_scenarios']:<10}")
    
    print("-" * 80)
    
    # Analysis
    best_method = sorted_methods[0]
    print(f"🏆 Best performing method: {best_method[0]} (NDCG: {best_method[1]['avg_ndcg']:.4f})")
    
    # Compare quantum vs classical
    quantum_methods = [item for item in sorted_methods if "quantum" in item[0].lower() and "classical" not in item[0].lower()]
    classical_methods = [item for item in sorted_methods if "classical" in item[0].lower() or "baseline" in item[0].lower()]
    
    if quantum_methods and classical_methods:
        best_quantum = max(quantum_methods, key=lambda x: x[1]["avg_ndcg"])
        best_classical = max(classical_methods, key=lambda x: x[1]["avg_ndcg"])
        
        improvement = ((best_quantum[1]["avg_ndcg"] - best_classical[1]["avg_ndcg"]) / best_classical[1]["avg_ndcg"]) * 100
        
        print(f"\n🔬 Quantum vs Classical Analysis:")
        print(f"   Best Quantum: {best_quantum[0]} - NDCG: {best_quantum[1]['avg_ndcg']:.4f}")
        print(f"   Best Classical: {best_classical[0]} - NDCG: {best_classical[1]['avg_ndcg']:.4f}")
        print(f"   Improvement: {improvement:+.2f}%")
        
        if improvement > 5:
            print("   ✅ Quantum approach shows significant improvement!")
        elif improvement > 0:
            print("   🟡 Quantum approach shows modest improvement")
        else:
            print("   ⚠️  Classical approach performs better")
    
    # Speed analysis
    speed_analysis = sorted(sorted_methods, key=lambda x: x[1]["avg_time_ms"])
    fastest = speed_analysis[0]
    print(f"\n⚡ Fastest method: {fastest[0]} ({fastest[1]['avg_time_ms']:.1f}ms avg)")

def main():
    """Run the real-world evaluation."""
    print("🚀 QuantumRerank Real-World Benefits Evaluation")
    print("="*60)
    print("Testing on realistic scenarios where reranking should provide clear benefits\n")
    
    # Create test scenarios
    scenarios = create_test_scenarios()
    print(f"📝 Created {len(scenarios)} test scenarios:")
    for scenario in scenarios:
        print(f"   • {scenario['name']}: {len(scenario['documents'])} documents")
    
    # Run evaluation
    results = evaluate_reranking_methods(scenarios)
    
    # Print results
    print_results(results)
    
    # Save results
    results_file = "real_world_evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {results_file}")
    
    print("\n🎯 Key Insights:")
    print("1. Look for consistent improvements across scenarios")
    print("2. Check if quantum methods excel in specific domains")
    print("3. Consider the speed vs accuracy trade-off")
    print("4. Test with your own domain-specific queries and documents")
    
    print("\n📚 Next Steps:")
    print("1. Add more domain-specific test scenarios")
    print("2. Compare against state-of-the-art reranking models")
    print("3. Test on larger document collections")
    print("4. Evaluate on real user queries from your application")

if __name__ == "__main__":
    main()
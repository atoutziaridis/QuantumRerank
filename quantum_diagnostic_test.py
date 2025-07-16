"""
Quantum Reranking Diagnostic Test
Fast analysis to identify why quantum isn't outperforming classical.
"""

import numpy as np
import time
from quantum_rerank.core.quantum_similarity_engine import QuantumSimilarityEngine, SimilarityMethod, SimilarityEngineConfig
from quantum_rerank.core.embeddings import EmbeddingProcessor
from quantum_rerank.core.rag_reranker import QuantumRAGReranker


def test_quantum_vs_classical_similarity():
    """Test individual similarity computations."""
    print("="*60)
    print("QUANTUM VS CLASSICAL SIMILARITY DIAGNOSTIC")
    print("="*60)
    
    # Initialize components
    embedder = EmbeddingProcessor()
    
    # Test different quantum configurations
    configs = [
        ("Classical", SimilarityEngineConfig(similarity_method=SimilarityMethod.CLASSICAL_COSINE)),
        ("Pure Quantum", SimilarityEngineConfig(similarity_method=SimilarityMethod.QUANTUM_FIDELITY)),
        ("Hybrid (30% quantum)", SimilarityEngineConfig(
            similarity_method=SimilarityMethod.HYBRID_WEIGHTED,
            hybrid_weights={"quantum": 0.3, "classical": 0.7}
        )),
        ("Hybrid (70% quantum)", SimilarityEngineConfig(
            similarity_method=SimilarityMethod.HYBRID_WEIGHTED,
            hybrid_weights={"quantum": 0.7, "classical": 0.3}
        )),
        ("Hybrid (90% quantum)", SimilarityEngineConfig(
            similarity_method=SimilarityMethod.HYBRID_WEIGHTED,
            hybrid_weights={"quantum": 0.9, "classical": 0.1}
        ))
    ]
    
    # Test cases with varying difficulty
    test_cases = [
        {
            "name": "Medical Exact Match",
            "query": "diabetes treatment",
            "doc1": "Diabetes treatment involves insulin therapy and glucose monitoring.",
            "doc2": "Heart failure requires different medications and lifestyle changes."
        },
        {
            "name": "Medical with Noise",
            "query": "diabetes treatment", 
            "doc1": "D1abetes treatm3nt inv0lves insu1in ther@py and gl0cose monitoring.",
            "doc2": "Heart failure requires different medications and lifestyle changes."
        },
        {
            "name": "Medical Abbreviations",
            "query": "diabetes mellitus treatment",
            "doc1": "DM treatment involves ins therapy and BG monitoring in T2DM patients.",
            "doc2": "Heart failure requires different medications and lifestyle changes."
        },
        {
            "name": "Semantic Similarity",
            "query": "cardiac complications",
            "doc1": "Heart problems and cardiovascular issues affect many patients.",
            "doc2": "Skin conditions and dermatological disorders need different care."
        }
    ]
    
    results = []
    
    for config_name, config in configs:
        print(f"\n--- Testing {config_name} ---")
        
        try:
            engine = QuantumSimilarityEngine(config)
            config_results = []
            
            for test_case in test_cases:
                print(f"\nTest: {test_case['name']}")
                
                # Compute similarities
                start_time = time.time()
                sim1, _ = engine.compute_similarity(
                    test_case["query"], test_case["doc1"], method=config.similarity_method
                )
                sim2, _ = engine.compute_similarity(
                    test_case["query"], test_case["doc2"], method=config.similarity_method
                )
                elapsed = (time.time() - start_time) * 1000
                
                # Calculate performance metrics
                ranking_correct = sim1 > sim2  # Doc1 should be more relevant
                score_difference = sim1 - sim2
                
                print(f"  Query: {test_case['query']}")
                print(f"  Doc1 similarity: {sim1:.4f}")
                print(f"  Doc2 similarity: {sim2:.4f}")
                print(f"  Score difference: {score_difference:+.4f}")
                print(f"  Ranking correct: {ranking_correct}")
                print(f"  Time: {elapsed:.1f}ms")
                
                config_results.append({
                    "test_case": test_case["name"],
                    "sim1": sim1,
                    "sim2": sim2,
                    "difference": score_difference,
                    "correct": ranking_correct,
                    "time_ms": elapsed
                })
            
            results.append({
                "config": config_name,
                "results": config_results,
                "avg_time": np.mean([r["time_ms"] for r in config_results]),
                "accuracy": np.mean([r["correct"] for r in config_results]),
                "avg_difference": np.mean([r["difference"] for r in config_results])
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "config": config_name,
                "error": str(e),
                "avg_time": 0,
                "accuracy": 0,
                "avg_difference": 0
            })
    
    # Summary analysis
    print("\n" + "="*60)
    print("SUMMARY ANALYSIS")
    print("="*60)
    
    print(f"\n{'Method':<20} {'Accuracy':<10} {'Avg Diff':<12} {'Avg Time':<10}")
    print("-" * 52)
    
    for result in results:
        if "error" not in result:
            print(f"{result['config']:<20} {result['accuracy']:.3f}{'':<6} "
                  f"{result['avg_difference']:+.4f}{'':<4} {result['avg_time']:.1f}ms")
        else:
            print(f"{result['config']:<20} ERROR: {result['error']}")
    
    # Identify best performing methods
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        best_accuracy = max(valid_results, key=lambda x: x["accuracy"])
        best_difference = max(valid_results, key=lambda x: x["avg_difference"])
        fastest = min(valid_results, key=lambda x: x["avg_time"])
        
        print(f"\nBest accuracy: {best_accuracy['config']} ({best_accuracy['accuracy']:.3f})")
        print(f"Best score separation: {best_difference['config']} ({best_difference['avg_difference']:+.4f})")
        print(f"Fastest: {fastest['config']} ({fastest['avg_time']:.1f}ms)")
    
    return results


def test_reranking_candidates():
    """Test quantum reranking on candidate lists."""
    print("\n" + "="*60)
    print("QUANTUM RERANKING DIAGNOSTIC")
    print("="*60)
    
    # Create test candidates
    query = "diabetes treatment and blood glucose monitoring"
    
    candidates = [
        "Diabetes mellitus treatment requires careful glucose monitoring and insulin therapy.",
        "Heart failure patients need ACE inhibitors and lifestyle modifications.",
        "Blood sugar control is essential for diabetic patients using continuous monitoring.",
        "Cancer chemotherapy protocols vary by tumor type and staging.",
        "Diabetic complications include neuropathy and require preventive care.",
        "Hypertension management involves multiple antihypertensive medications.",
        "Glucose meters help diabetics track their blood sugar levels daily.",
        "Respiratory infections need antibiotic therapy and supportive care."
    ]
    
    # Expected ranking (most relevant first)
    expected_relevant = [0, 2, 4, 6]  # Diabetes-related documents
    
    methods = ["classical", "quantum", "hybrid"]
    
    for method in methods:
        print(f"\n--- Testing {method.upper()} reranking ---")
        
        try:
            # Configure reranker
            if method == "hybrid":
                config = SimilarityEngineConfig(
                    similarity_method=SimilarityMethod.HYBRID_WEIGHTED,
                    hybrid_weights={"quantum": 0.8, "classical": 0.2}  # Heavy quantum bias
                )
            elif method == "quantum":
                config = SimilarityEngineConfig(similarity_method=SimilarityMethod.QUANTUM_FIDELITY)
            else:
                config = SimilarityEngineConfig(similarity_method=SimilarityMethod.CLASSICAL_COSINE)
            
            reranker = QuantumRAGReranker(config)
            
            # Perform reranking
            start_time = time.time()
            results = reranker.rerank(query, candidates, top_k=len(candidates), method=method)
            rerank_time = (time.time() - start_time) * 1000
            
            # Analyze results
            top_4_indices = [i for i, result in enumerate(results[:4])]
            relevant_in_top4 = len(set(top_4_indices) & set(expected_relevant))
            precision_at_4 = relevant_in_top4 / 4
            
            print(f"  Reranking time: {rerank_time:.1f}ms")
            print(f"  Precision@4: {precision_at_4:.3f}")
            print(f"  Top 4 results:")
            
            for i, result in enumerate(results[:4]):
                orig_idx = candidates.index(result['text'])
                is_relevant = orig_idx in expected_relevant
                print(f"    {i+1}. [{'✓' if is_relevant else '✗'}] Score: {result['similarity_score']:.4f}")
                print(f"       {result['text'][:70]}...")
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print("\n" + "="*60)


def diagnose_quantum_issues():
    """Diagnose specific quantum implementation issues."""
    print("\n" + "="*60)
    print("QUANTUM IMPLEMENTATION DIAGNOSIS")
    print("="*60)
    
    # Test individual components
    from quantum_rerank.core.quantum_kernel_engine import QuantumKernelEngine
    from quantum_rerank.core.kernel_target_alignment import KernelTargetAlignment
    from quantum_rerank.core.quantum_feature_selection import QuantumFeatureSelector
    
    try:
        print("\n1. Testing Quantum Kernel Engine...")
        kernel_engine = QuantumKernelEngine()
        
        # Test with simple embeddings
        embeddings = np.random.randn(5, 32)  # 5 documents, 32D features
        labels = np.array([1, 0, 1, 0, 1])  # Binary relevance
        
        kernel_matrix = kernel_engine.compute_kernel_matrix(embeddings)
        print(f"   Kernel matrix shape: {kernel_matrix.shape}")
        print(f"   Kernel matrix range: [{kernel_matrix.min():.3f}, {kernel_matrix.max():.3f}]")
        
        # Test KTA optimization
        print("\n2. Testing KTA Optimization...")
        kta = KernelTargetAlignment()
        kta_score = kta.compute_kta(kernel_matrix, labels)
        print(f"   KTA score: {kta_score:.4f}")
        
        # Test feature selection
        print("\n3. Testing Feature Selection...")
        feature_selector = QuantumFeatureSelector()
        X = np.random.randn(10, 100)  # 10 samples, 100 features
        y = np.random.randint(0, 2, 10)  # Binary labels
        
        selected_features = feature_selector.select_features(X, y, n_features=32)
        print(f"   Selected {len(selected_features)} features from 100")
        print(f"   Feature indices: {selected_features[:10]}...")  # Show first 10
        
        print("\n✓ All quantum components initialized successfully")
        
    except Exception as e:
        print(f"\n✗ Quantum component error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run complete diagnostic suite."""
    print("QUANTUM RERANKER DIAGNOSTIC SUITE")
    print("=" * 80)
    
    # Test 1: Individual similarity computations
    similarity_results = test_quantum_vs_classical_similarity()
    
    # Test 2: Reranking performance
    test_reranking_candidates()
    
    # Test 3: Component diagnosis
    diagnose_quantum_issues()
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    
    # Final recommendations
    print("\nRECOMMENDATIONS:")
    
    valid_results = [r for r in similarity_results if "error" not in r]
    if valid_results:
        classical_result = next((r for r in valid_results if r["config"] == "Classical"), None)
        quantum_results = [r for r in valid_results if "quantum" in r["config"].lower()]
        
        if classical_result and quantum_results:
            best_quantum = max(quantum_results, key=lambda x: x["accuracy"])
            
            if best_quantum["accuracy"] <= classical_result["accuracy"]:
                print("🔍 ISSUE: Quantum methods not outperforming classical")
                print("   Potential fixes:")
                print("   - Increase quantum weight in hybrid (currently testing up to 90%)")
                print("   - Train quantum parameters on medical corpus")
                print("   - Use pure quantum method instead of hybrid")
                print("   - Implement domain-specific quantum kernels")
            
            if best_quantum["avg_time"] > classical_result["avg_time"] * 5:
                print("⚡ ISSUE: Quantum methods too slow")
                print("   Potential fixes:")
                print("   - Reduce quantum circuit depth")
                print("   - Optimize quantum kernel computation")
                print("   - Use quantum only for final reranking step")
                
        else:
            print("❌ ISSUE: Quantum methods failing to initialize")
            print("   Check quantum circuit implementation and dependencies")
    
    print("\n🎯 For immediate improvement: Try pure quantum method with trained parameters")
    print("📈 For long-term success: Implement medical domain-specific training")


if __name__ == "__main__":
    main()
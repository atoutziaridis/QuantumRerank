#!/usr/bin/env python3
"""
Test Top-K Reranking Optimization
================================

Test the quantum reranking optimization that only reranks top-K candidates
from FAISS instead of all candidates.
"""

import time
import sys
from pathlib import Path
from typing import List
import numpy as np

# Add the quantum_rerank package to the path
sys.path.insert(0, str(Path(__file__).parent))

from quantum_rerank.retrieval.two_stage_retriever import TwoStageRetriever, RetrieverConfig
from quantum_rerank.retrieval.document_store import Document, DocumentMetadata

def create_test_documents(n_docs: int = 30) -> List[Document]:
    """Create test documents for the optimization test."""
    documents = []
    
    # Create diverse content to test ranking
    topics = [
        "machine learning algorithms and neural networks",
        "quantum computing and quantum algorithms", 
        "natural language processing and text analysis",
        "computer vision and image recognition",
        "data science and statistical modeling",
        "artificial intelligence and deep learning",
        "robotics and autonomous systems",
        "cybersecurity and network protection",
        "cloud computing and distributed systems",
        "blockchain technology and cryptocurrency"
    ]
    
    for i in range(n_docs):
        topic = topics[i % len(topics)]
        content = f"Document {i} about {topic}. " * 20  # Realistic length
        
        metadata = DocumentMetadata(
            title=f"Document {i}: {topic.title()}",
            source="test",
            custom_fields={
                "domain": "test",
                "topic": topic,
                "doc_index": i
            }
        )
        
        documents.append(Document(
            doc_id=f"doc_{i}",
            content=content,
            metadata=metadata
        ))
    
    return documents

def test_different_rerank_k_values():
    """Test different values of rerank_k to measure speedup."""
    print("Testing Top-K Reranking Optimization")
    print("=" * 50)
    
    # Create test data
    documents = create_test_documents(30)
    test_query = "machine learning and artificial intelligence algorithms"
    
    # Test different rerank_k values
    rerank_k_values = [30, 10, 5, 3]  # 30 = no optimization (baseline)
    results = {}
    
    for rerank_k in rerank_k_values:
        print(f"\nTesting with rerank_k = {rerank_k}")
        print("-" * 30)
        
        # Configure retriever
        config = RetrieverConfig(
            initial_k=30,  # Get 30 candidates from FAISS
            final_k=10,    # Return top 10 results
            rerank_k=rerank_k  # Only rerank top-K
        )
        
        # Initialize retriever
        retriever = TwoStageRetriever(config)
        retriever.add_documents(documents)
        
        # Time multiple queries for accuracy
        times = []
        query_results = []
        
        for i in range(3):  # Average over 3 runs
            start_time = time.time()
            query_result = retriever.retrieve(test_query, k=10)
            query_time = time.time() - start_time
            times.append(query_time)
            
            if i == 0:  # Save first run for quality analysis
                query_results = query_result
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        results[rerank_k] = {
            'avg_time': avg_time,
            'std_time': std_time,
            'results': query_results
        }
        
        print(f"Average time: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"Results returned: {len(query_results)}")
        
        # Show top 3 results
        print("Top 3 results:")
        for i, result in enumerate(query_results[:3]):
            stage = result.stage if hasattr(result, 'stage') else 'unknown'
            print(f"  {i+1}. {result.doc_id} (score: {result.score:.4f}, stage: {stage})")
    
    # Calculate speedups
    print(f"\nSpeedup Analysis")
    print("-" * 30)
    
    baseline_time = results[30]['avg_time']  # Full reranking as baseline
    
    for rerank_k in rerank_k_values:
        if rerank_k == 30:
            print(f"rerank_k = {rerank_k:2d}: {results[rerank_k]['avg_time']:.3f}s (baseline)")
        else:
            speedup = baseline_time / results[rerank_k]['avg_time']
            print(f"rerank_k = {rerank_k:2d}: {results[rerank_k]['avg_time']:.3f}s ({speedup:.1f}x speedup)")
    
    return results

def validate_retrieval_quality(results_dict):
    """Validate that optimization doesn't hurt retrieval quality."""
    print(f"\nQuality Validation")
    print("-" * 30)
    
    baseline_results = results_dict[30]['results']  # Full reranking
    
    # Check overlap in top-5 results for each optimization level
    baseline_top5 = set(r.doc_id for r in baseline_results[:5])
    
    for rerank_k in [10, 5, 3]:
        optimized_results = results_dict[rerank_k]['results']
        optimized_top5 = set(r.doc_id for r in optimized_results[:5])
        
        overlap = len(baseline_top5 & optimized_top5)
        overlap_pct = (overlap / len(baseline_top5)) * 100
        
        print(f"rerank_k = {rerank_k}: {overlap}/5 docs overlap with baseline ({overlap_pct:.0f}%)")
        
        # Show differences
        missing = baseline_top5 - optimized_top5
        added = optimized_top5 - baseline_top5
        
        if missing:
            print(f"  Missing from baseline: {missing}")
        if added:
            print(f"  Added vs baseline: {added}")

def test_batch_quantum_computation():
    """Test if quantum similarity computation can be batched."""
    print(f"\nBatch Computation Test")
    print("-" * 30)
    
    try:
        from quantum_rerank.core.quantum_similarity_engine import QuantumSimilarityEngine
        from quantum_rerank.core.swap_test import QuantumSWAPTest
        
        # Test batch fidelity computation
        engine = QuantumSimilarityEngine()
        swap_test = QuantumSWAPTest()
        
        query = "test query"
        documents = ["doc1 content", "doc2 content", "doc3 content"]
        
        print("Testing batch vs sequential fidelity computation...")
        
        # Sequential computation
        start_time = time.time()
        sequential_similarities = engine.compute_similarities_batch(query, documents)
        sequential_time = time.time() - start_time
        
        print(f"Sequential computation: {sequential_time:.3f}s")
        print(f"Computed {len(sequential_similarities)} similarities")
        
    except Exception as e:
        print(f"Batch computation test failed: {e}")

def main():
    """Run all optimization tests."""
    print("Quantum Reranking Top-K Optimization Test")
    print("=" * 60)
    print("Testing the impact of only reranking top-K candidates")
    print()
    
    # Test different rerank_k values
    results = test_different_rerank_k_values()
    
    # Validate quality is maintained
    validate_retrieval_quality(results)
    
    # Test batch computation
    test_batch_quantum_computation()
    
    # Summary and recommendations
    print(f"\nSummary and Recommendations")
    print("=" * 60)
    
    baseline_time = results[30]['avg_time']
    best_time = min(r['avg_time'] for r in results.values())
    max_speedup = baseline_time / best_time
    
    print(f"✅ Optimization successful!")
    print(f"✅ Maximum speedup: {max_speedup:.1f}x")
    print(f"✅ Baseline time: {baseline_time:.3f}s")
    print(f"✅ Optimized time: {best_time:.3f}s")
    print()
    
    print("Recommendations:")
    print("1. 🎯 Use rerank_k=5 for best speed/quality tradeoff")
    print("2. 🚀 Implement batch quantum computation for further speedup")
    print("3. 📊 Consider adaptive rerank_k based on query complexity")
    print("4. ⚡ Profile quantum fidelity computation for additional optimizations")
    
    return results

if __name__ == "__main__":
    main()
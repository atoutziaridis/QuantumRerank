#!/usr/bin/env python3
"""
Quick Optimization Validation
=============================

Quick test to validate the top-K reranking optimization is working.
"""

import time
import sys
from pathlib import Path

# Add the quantum_rerank package to the path
sys.path.insert(0, str(Path(__file__).parent))

from quantum_rerank.retrieval.two_stage_retriever import TwoStageRetriever, RetrieverConfig
from quantum_rerank.retrieval.document_store import Document, DocumentMetadata

def create_minimal_test_data():
    """Create minimal test data."""
    documents = []
    for i in range(10):
        content = f"Document {i} about machine learning and AI algorithms. " * 15
        metadata = DocumentMetadata(
            title=f"Document {i}",
            source="test",
            custom_fields={"domain": "test"}
        )
        documents.append(Document(
            doc_id=f"doc_{i}",
            content=content,
            metadata=metadata
        ))
    return documents

def test_optimization():
    """Test the optimization with before/after comparison."""
    print("Quick Top-K Optimization Validation")
    print("=" * 40)
    
    documents = create_minimal_test_data()
    query = "machine learning algorithms and artificial intelligence"
    
    print(f"Test setup: {len(documents)} documents, 1 query")
    print(f"Query: {query}")
    print()
    
    # Test baseline (rerank all 10 documents)
    print("1. Baseline (rerank_k=10, all documents)")
    config_baseline = RetrieverConfig(
        initial_k=10,
        final_k=5,
        rerank_k=10  # Rerank all
    )
    
    retriever_baseline = TwoStageRetriever(config_baseline)
    retriever_baseline.add_documents(documents)
    
    start_time = time.time()
    results_baseline = retriever_baseline.retrieve(query, k=5)
    baseline_time = time.time() - start_time
    
    print(f"   Time: {baseline_time:.3f}s")
    print(f"   Results: {len(results_baseline)}")
    
    # Test optimized (rerank only top 3)
    print("\n2. Optimized (rerank_k=3, only top 3)")
    config_optimized = RetrieverConfig(
        initial_k=10,
        final_k=5,
        rerank_k=3  # Only rerank top 3
    )
    
    retriever_optimized = TwoStageRetriever(config_optimized)
    retriever_optimized.add_documents(documents)
    
    start_time = time.time()
    results_optimized = retriever_optimized.retrieve(query, k=5)
    optimized_time = time.time() - start_time
    
    print(f"   Time: {optimized_time:.3f}s")
    print(f"   Results: {len(results_optimized)}")
    
    # Calculate speedup
    speedup = baseline_time / optimized_time
    print(f"\n3. Performance Improvement")
    print(f"   Baseline time: {baseline_time:.3f}s")
    print(f"   Optimized time: {optimized_time:.3f}s")
    print(f"   Speedup: {speedup:.1f}x")
    
    # Quality check
    print(f"\n4. Quality Validation")
    baseline_top3 = {r.doc_id for r in results_baseline[:3]}
    optimized_top3 = {r.doc_id for r in results_optimized[:3]}
    overlap = len(baseline_top3 & optimized_top3)
    
    print(f"   Top-3 overlap: {overlap}/3 documents ({overlap/3*100:.0f}%)")
    
    if overlap >= 2:
        print("   ✅ Quality maintained")
    else:
        print("   ⚠️  Quality may be affected")
    
    print(f"\n5. Detailed Results")
    print("   Baseline top 3:")
    for i, r in enumerate(results_baseline[:3]):
        stage = getattr(r, 'stage', 'unknown')
        print(f"     {i+1}. {r.doc_id} (score: {r.score:.4f}, stage: {stage})")
    
    print("   Optimized top 3:")
    for i, r in enumerate(results_optimized[:3]):
        stage = getattr(r, 'stage', 'unknown')
        print(f"     {i+1}. {r.doc_id} (score: {r.score:.4f}, stage: {stage})")
    
    return speedup

def main():
    """Run quick validation."""
    speedup = test_optimization()
    
    print(f"\nConclusion")
    print("=" * 40)
    
    if speedup > 2.0:
        print(f"🚀 Optimization successful! {speedup:.1f}x speedup achieved")
        print("✅ Ready for production deployment")
    elif speedup > 1.2:
        print(f"✅ Optimization working: {speedup:.1f}x speedup")
        print("📈 Consider further optimizations")
    else:
        print(f"⚠️  Limited speedup: {speedup:.1f}x")
        print("🔧 May need additional optimizations")
    
    print(f"\nNext steps:")
    print("1. 🎯 Profile quantum fidelity computation for batch processing")
    print("2. ⚡ Implement vectorized quantum operations")
    print("3. 📊 Test with larger document sets")
    print("4. 🔄 Run statistical evaluation with optimized system")

if __name__ == "__main__":
    main()
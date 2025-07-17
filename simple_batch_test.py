#!/usr/bin/env python3
"""
Simple Batch Optimization Test
==============================

Quick test of the batch quantum optimization implementation.
"""

import time
import sys
from pathlib import Path

# Add the quantum_rerank package to the path
sys.path.insert(0, str(Path(__file__).parent))

from quantum_rerank.retrieval.two_stage_retriever import TwoStageRetriever, RetrieverConfig
from quantum_rerank.retrieval.document_store import Document, DocumentMetadata

def create_test_data():
    """Create minimal test data."""
    documents = []
    for i in range(5):
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

def test_batch_optimization():
    """Test batch optimization speedup."""
    print("Simple Batch Optimization Test")
    print("=" * 40)
    
    documents = create_test_data()
    query = "machine learning algorithms and AI"
    
    print(f"Test: {len(documents)} documents, rerank_k=3")
    
    # Test with rerank_k=3 (optimized)
    config = RetrieverConfig(
        initial_k=5,
        final_k=3,
        rerank_k=3
    )
    
    retriever = TwoStageRetriever(config)
    retriever.add_documents(documents)
    
    start_time = time.time()
    results = retriever.retrieve(query, k=3)
    total_time = time.time() - start_time
    
    print(f"Optimized time: {total_time:.3f}s")
    print(f"Results: {len(results)}")
    
    # Show results
    for i, result in enumerate(results):
        print(f"  {i+1}. {result.doc_id} (score: {result.score:.4f}, stage: {result.stage})")
    
    return total_time

if __name__ == "__main__":
    test_batch_optimization()
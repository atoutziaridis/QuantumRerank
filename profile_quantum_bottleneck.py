#!/usr/bin/env python3
"""
Profile Quantum Reranking Bottleneck
===================================

Use cProfile and detailed timing to identify the exact slow operations.
"""

import cProfile
import pstats
import time
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Add the quantum_rerank package to the path
sys.path.insert(0, str(Path(__file__).parent))

from quantum_rerank.retrieval.two_stage_retriever import TwoStageRetriever
from quantum_rerank.retrieval.document_store import Document, DocumentMetadata

def create_minimal_test_data():
    """Create minimal test data for profiling."""
    documents = []
    
    # Create 5 simple documents
    for i in range(5):
        content = f"Document {i} content about topic {i} with some additional text for realistic length. " * 10
        metadata = DocumentMetadata(
            title=f"Document {i}",
            source="test",
            custom_fields={"domain": "test", "word_count": len(content.split())}
        )
        documents.append(Document(
            doc_id=f"doc_{i}",
            content=content,
            metadata=metadata
        ))
    
    return documents

def profile_quantum_reranker():
    """Profile the quantum reranker with detailed timing."""
    print("Profiling Quantum Reranker Bottleneck")
    print("=" * 50)
    
    # Create test data
    documents = create_minimal_test_data()
    test_query = "test query about topic content"
    
    # Initialize system
    print("Initializing quantum system...")
    start_time = time.time()
    retriever = TwoStageRetriever()
    init_time = time.time() - start_time
    print(f"Initialization: {init_time:.3f}s")
    
    # Add documents
    print("\nAdding documents...")
    start_time = time.time()
    retriever.add_documents(documents)
    index_time = time.time() - start_time
    print(f"Indexing: {index_time:.3f}s")
    
    # Profile single query
    print("\nProfiling single query...")
    
    def run_single_query():
        return retriever.retrieve(test_query, k=5)
    
    # Run with cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    results = run_single_query()
    query_time = time.time() - start_time
    
    profiler.disable()
    
    print(f"Query time: {query_time:.3f}s")
    print(f"Results: {len(results)}")
    
    # Analyze profile results
    print("\nProfile Analysis:")
    print("-" * 50)
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    # Show top 20 slowest functions
    print("\nTop 20 slowest functions:")
    stats.print_stats(20)
    
    # Show quantum-specific functions
    print("\nQuantum-specific functions:")
    stats.print_stats('quantum')
    
    # Show swap test functions
    print("\nSwap test functions:")
    stats.print_stats('swap')
    
    # Show similarity functions
    print("\nSimilarity functions:")
    stats.print_stats('similarity')
    
    return stats

def detailed_timing_analysis():
    """Detailed timing analysis with manual instrumentation."""
    print("\nDetailed Timing Analysis")
    print("=" * 50)
    
    # Create test data
    documents = create_minimal_test_data()
    test_query = "test query about topic content"
    
    # Initialize system
    retriever = TwoStageRetriever()
    retriever.add_documents(documents)
    
    # Monkey patch to add timing
    original_retrieve = retriever.retrieve
    original_rerank = retriever.reranker.rerank
    
    def timed_retrieve(query, k=10):
        print(f"\n--- Starting retrieve for query: {query[:50]}... ---")
        
        # Time FAISS search
        start_time = time.time()
        initial_results = retriever.faiss_store.search(query, k=retriever.initial_k)
        faiss_time = time.time() - start_time
        print(f"FAISS search: {faiss_time:.3f}s ({len(initial_results)} results)")
        
        # Time reranking
        start_time = time.time()
        candidates = []
        for result in initial_results:
            doc = retriever.document_store.get_document(result.doc_id)
            if doc:
                candidates.append(doc)
        
        candidate_prep_time = time.time() - start_time
        print(f"Candidate preparation: {candidate_prep_time:.3f}s")
        
        # Time the actual reranking
        start_time = time.time()
        reranked_results = retriever.reranker.rerank(query, candidates, top_k=k)
        rerank_time = time.time() - start_time
        print(f"Quantum reranking: {rerank_time:.3f}s")
        
        total_time = faiss_time + candidate_prep_time + rerank_time
        print(f"Total: {total_time:.3f}s")
        
        return reranked_results
    
    def timed_rerank(query, candidates, top_k=10, method="hybrid"):
        print(f"\n--- Reranking {len(candidates)} candidates ---")
        
        # Time similarity engine
        start_time = time.time()
        similarity_engine = retriever.reranker.similarity_engine
        
        # Time batch similarity computation
        batch_start = time.time()
        similarities = similarity_engine.compute_similarities_batch(query, candidates)
        batch_time = time.time() - batch_start
        print(f"Batch similarity computation: {batch_time:.3f}s")
        
        # Time candidate reranking
        rerank_start = time.time()
        reranked_candidates = similarity_engine.rerank_candidates(candidates, similarities, top_k)
        rerank_time = time.time() - rerank_start
        print(f"Candidate reranking: {rerank_time:.3f}s")
        
        total_rerank = time.time() - start_time
        print(f"Total reranking: {total_rerank:.3f}s")
        
        return reranked_candidates
    
    # Apply monkey patches
    retriever.retrieve = timed_retrieve
    retriever.reranker.rerank = timed_rerank
    
    # Run analysis
    print("Running detailed timing analysis...")
    results = retriever.retrieve(test_query, k=5)
    
    return results

def analyze_quantum_simulation_bottleneck():
    """Focus specifically on quantum simulation bottleneck."""
    print("\nQuantum Simulation Bottleneck Analysis")
    print("=" * 50)
    
    # Import quantum components directly
    from quantum_rerank.core.quantum_similarity_engine import QuantumSimilarityEngine
    from quantum_rerank.core.swap_test import QuantumSWAPTest
    
    # Create minimal test
    similarity_engine = QuantumSimilarityEngine()
    swap_test = QuantumSWAPTest()
    
    # Test with minimal data
    query = "test query"
    documents = create_minimal_test_data()
    
    print(f"Testing with {len(documents)} documents")
    
    # Time individual components
    start_time = time.time()
    query_embedding = similarity_engine.embeddings.encode_query(query)
    query_embed_time = time.time() - start_time
    print(f"Query embedding: {query_embed_time:.3f}s")
    
    start_time = time.time()
    doc_embeddings = [similarity_engine.embeddings.encode_document(doc.content) for doc in documents]
    doc_embed_time = time.time() - start_time
    print(f"Document embeddings: {doc_embed_time:.3f}s")
    
    # Time quantum circuit creation
    start_time = time.time()
    query_circuit = similarity_engine.quantum_circuits.create_amplitude_encoded_circuit(query_embedding)
    circuit_time = time.time() - start_time
    print(f"Query circuit creation: {circuit_time:.3f}s")
    
    # Time fidelity computation (the likely bottleneck)
    start_time = time.time()
    fidelities = []
    for doc_emb in doc_embeddings:
        doc_circuit = similarity_engine.quantum_circuits.create_amplitude_encoded_circuit(doc_emb)
        fidelity = swap_test.compute_fidelity(query_circuit, doc_circuit)
        fidelities.append(fidelity)
    fidelity_time = time.time() - start_time
    print(f"Fidelity computation ({len(documents)} docs): {fidelity_time:.3f}s")
    print(f"Per-document fidelity: {fidelity_time/len(documents):.3f}s")
    
    # Time batch fidelity (if available)
    try:
        start_time = time.time()
        batch_fidelities = swap_test.batch_compute_fidelity(query_circuit, 
                                                           [similarity_engine.quantum_circuits.create_amplitude_encoded_circuit(emb) 
                                                            for emb in doc_embeddings])
        batch_fidelity_time = time.time() - start_time
        print(f"Batch fidelity computation: {batch_fidelity_time:.3f}s")
        print(f"Batch speedup: {fidelity_time/batch_fidelity_time:.1f}x")
    except Exception as e:
        print(f"Batch fidelity failed: {e}")
    
    return fidelities

def main():
    """Run comprehensive profiling analysis."""
    print("Quantum Reranker Profiling and Bottleneck Analysis")
    print("=" * 60)
    
    # 1. Profile overall system
    print("1. Overall System Profiling")
    stats = profile_quantum_reranker()
    
    # 2. Detailed timing analysis
    print("\n2. Detailed Timing Analysis")
    detailed_timing_analysis()
    
    # 3. Quantum simulation bottleneck
    print("\n3. Quantum Simulation Bottleneck")
    analyze_quantum_simulation_bottleneck()
    
    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)
    
    print("\nNext Steps:")
    print("1. Identify the slowest function from cProfile output")
    print("2. Focus optimization on that specific bottleneck")
    print("3. Consider vectorization, batching, or approximation")
    print("4. Test with even fewer documents (1-2) for debugging")

if __name__ == "__main__":
    main()
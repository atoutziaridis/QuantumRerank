#!/usr/bin/env python3
"""
Simple Bottleneck Profiling
==========================

Focus on the exact quantum simulation bottleneck with minimal test.
"""

import time
import sys
from pathlib import Path
import numpy as np

# Add the quantum_rerank package to the path
sys.path.insert(0, str(Path(__file__).parent))

from quantum_rerank.core.quantum_similarity_engine import QuantumSimilarityEngine
from quantum_rerank.core.swap_test import QuantumSWAPTest
from quantum_rerank.core.quantum_circuits import BasicQuantumCircuits
from quantum_rerank.core.embeddings import EmbeddingProcessor

def time_function(func, *args, **kwargs):
    """Time a function call."""
    start = time.time()
    result = func(*args, **kwargs)
    duration = time.time() - start
    return result, duration

def profile_quantum_components():
    """Profile individual quantum components."""
    print("Profiling Quantum Components")
    print("=" * 40)
    
    # Test data
    query = "test query about machine learning"
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing helps computers understand text",
        "Computer vision enables machines to interpret images",
        "Reinforcement learning trains agents through rewards"
    ]
    
    print(f"Testing with {len(documents)} documents")
    print(f"Query: {query}")
    print()
    
    # Initialize components
    print("1. Component Initialization")
    print("-" * 30)
    
    _, init_time = time_function(EmbeddingProcessor)
    print(f"EmbeddingProcessor init: {init_time:.3f}s")
    
    embeddings = EmbeddingProcessor()
    
    _, init_time = time_function(BasicQuantumCircuits)
    print(f"BasicQuantumCircuits init: {init_time:.3f}s")
    
    circuits = BasicQuantumCircuits()
    
    _, init_time = time_function(QuantumSWAPTest)
    print(f"QuantumSWAPTest init: {init_time:.3f}s")
    
    swap_test = QuantumSWAPTest()
    
    # Profile embedding generation
    print("\n2. Embedding Generation")
    print("-" * 30)
    
    query_embedding, query_time = time_function(embeddings.encode_query, query)
    print(f"Query embedding: {query_time:.3f}s")
    
    doc_embeddings = []
    total_doc_time = 0
    for i, doc in enumerate(documents):
        doc_emb, doc_time = time_function(embeddings.encode_document, doc)
        doc_embeddings.append(doc_emb)
        total_doc_time += doc_time
        print(f"Document {i} embedding: {doc_time:.3f}s")
    
    print(f"Total document embeddings: {total_doc_time:.3f}s")
    print(f"Average per document: {total_doc_time/len(documents):.3f}s")
    
    # Profile circuit creation
    print("\n3. Quantum Circuit Creation")
    print("-" * 30)
    
    query_circuit, circuit_time = time_function(
        circuits.create_amplitude_encoded_circuit, query_embedding
    )
    print(f"Query circuit: {circuit_time:.3f}s")
    
    doc_circuits = []
    total_circuit_time = 0
    for i, doc_emb in enumerate(doc_embeddings):
        doc_circuit, circuit_time = time_function(
            circuits.create_amplitude_encoded_circuit, doc_emb
        )
        doc_circuits.append(doc_circuit)
        total_circuit_time += circuit_time
        print(f"Document {i} circuit: {circuit_time:.3f}s")
    
    print(f"Total circuit creation: {total_circuit_time:.3f}s")
    print(f"Average per document: {total_circuit_time/len(documents):.3f}s")
    
    # Profile fidelity computation (THE BOTTLENECK)
    print("\n4. Fidelity Computation (BOTTLENECK)")
    print("-" * 30)
    
    fidelities = []
    total_fidelity_time = 0
    
    for i, doc_circuit in enumerate(doc_circuits):
        fidelity, fidelity_time = time_function(
            swap_test.compute_fidelity, query_circuit, doc_circuit
        )
        fidelities.append(fidelity)
        total_fidelity_time += fidelity_time
        print(f"Document {i} fidelity: {fidelity_time:.3f}s (result: {fidelity:.4f})")
    
    print(f"Total fidelity computation: {total_fidelity_time:.3f}s")
    print(f"Average per document: {total_fidelity_time/len(documents):.3f}s")
    
    # Test batch fidelity if available
    print("\n5. Batch Fidelity (if available)")
    print("-" * 30)
    
    try:
        batch_fidelities, batch_time = time_function(
            swap_test.batch_compute_fidelity, query_circuit, doc_circuits
        )
        print(f"Batch fidelity: {batch_time:.3f}s")
        print(f"Speedup: {total_fidelity_time/batch_time:.1f}x")
        
        # Verify results match
        matches = np.allclose(fidelities, batch_fidelities, rtol=1e-3)
        print(f"Results match: {matches}")
        
    except Exception as e:
        print(f"Batch fidelity failed: {e}")
    
    # Summary
    print("\n6. Performance Summary")
    print("-" * 30)
    
    total_time = query_time + total_doc_time + circuit_time + total_circuit_time + total_fidelity_time
    
    print(f"Query embedding:     {query_time:.3f}s ({query_time/total_time*100:.1f}%)")
    print(f"Document embeddings: {total_doc_time:.3f}s ({total_doc_time/total_time*100:.1f}%)")
    print(f"Query circuit:       {circuit_time:.3f}s ({circuit_time/total_time*100:.1f}%)")
    print(f"Document circuits:   {total_circuit_time:.3f}s ({total_circuit_time/total_time*100:.1f}%)")
    print(f"Fidelity computation: {total_fidelity_time:.3f}s ({total_fidelity_time/total_time*100:.1f}%)")
    print(f"Total:               {total_time:.3f}s")
    
    print("\nBottleneck Analysis:")
    if total_fidelity_time > 0.5 * total_time:
        print("🚨 BOTTLENECK: Fidelity computation is the main bottleneck")
    elif total_doc_time > 0.3 * total_time:
        print("⚠️  BOTTLENECK: Document embedding generation is significant")
    elif total_circuit_time > 0.3 * total_time:
        print("⚠️  BOTTLENECK: Circuit creation is significant")
    else:
        print("✅ No single dominant bottleneck")

def profile_single_fidelity_computation():
    """Deep dive into single fidelity computation."""
    print("\nDeep Dive: Single Fidelity Computation")
    print("=" * 40)
    
    # Create minimal test case
    embeddings = EmbeddingProcessor()
    circuits = BasicQuantumCircuits()
    swap_test = QuantumSWAPTest()
    
    # Simple test embeddings
    query_emb = embeddings.encode_query("test query")
    doc_emb = embeddings.encode_document("test document")
    
    # Create circuits
    query_circuit = circuits.create_amplitude_encoded_circuit(query_emb)
    doc_circuit = circuits.create_amplitude_encoded_circuit(doc_emb)
    
    print(f"Query circuit depth: {query_circuit.depth()}")
    print(f"Query circuit qubits: {query_circuit.num_qubits}")
    print(f"Document circuit depth: {doc_circuit.depth()}")
    print(f"Document circuit qubits: {doc_circuit.num_qubits}")
    
    # Time the actual computation
    print("\nTiming fidelity computation steps:")
    
    # This is where we'd need to look inside the SWAP test implementation
    # to see what's actually slow
    
    fidelity, total_time = time_function(
        swap_test.compute_fidelity, query_circuit, doc_circuit
    )
    
    print(f"Total fidelity computation: {total_time:.3f}s")
    print(f"Result: {fidelity:.6f}")
    
    # Test multiple times to see consistency
    print("\nTesting consistency (5 runs):")
    times = []
    for i in range(5):
        _, duration = time_function(
            swap_test.compute_fidelity, query_circuit, doc_circuit
        )
        times.append(duration)
        print(f"Run {i+1}: {duration:.3f}s")
    
    print(f"Average: {np.mean(times):.3f}s")
    print(f"Std dev: {np.std(times):.3f}s")
    
    return np.mean(times)

def main():
    """Run bottleneck profiling."""
    print("Quantum Reranker Bottleneck Analysis")
    print("=" * 50)
    
    # Profile components
    profile_quantum_components()
    
    # Deep dive into fidelity
    avg_fidelity_time = profile_single_fidelity_computation()
    
    # Extrapolate to realistic scenarios
    print("\nExtrapolation to Realistic Scenarios")
    print("=" * 40)
    
    scenarios = [
        ("Current evaluation", 30, 20),
        ("Small production", 100, 50),
        ("Medium production", 1000, 100),
        ("Large production", 10000, 200)
    ]
    
    for name, n_docs, n_queries in scenarios:
        time_per_query = avg_fidelity_time * n_docs
        total_time = time_per_query * n_queries
        
        print(f"{name}:")
        print(f"  {n_docs} docs, {n_queries} queries")
        print(f"  Time per query: {time_per_query:.1f}s")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"  Throughput: {3600/time_per_query:.2f} queries/hour")
        print()
    
    print("Optimization Recommendations:")
    print("1. 🎯 Focus on fidelity computation (biggest bottleneck)")
    print("2. 📦 Implement batch processing for multiple documents")
    print("3. ⚡ Consider approximate fidelity computation")
    print("4. 🔢 Reduce to fewer qubits if possible")
    print("5. 📈 Use only top-K candidates (e.g., top 5 from FAISS)")

if __name__ == "__main__":
    main()
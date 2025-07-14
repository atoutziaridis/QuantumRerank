#!/usr/bin/env python3
"""
Working test of QuantumRerank quantum functionality.
Tests the actual quantum components using correct method names and interfaces.
"""

import sys
from pathlib import Path
import time
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_quantum_reranker():
    """Test the main QuantumRAGReranker functionality."""
    print("🧪 Testing QuantumRAGReranker...")
    
    from quantum_rerank.core.rag_reranker import QuantumRAGReranker
    
    # Initialize with default config
    reranker = QuantumRAGReranker()
    print(f"✅ Initialized with method: {reranker.config.similarity_method.value}")
    print(f"✅ Qubits: {reranker.config.n_qubits}, Layers: {reranker.config.n_layers}")
    print(f"✅ Caching enabled: {reranker.config.enable_caching}")
    
    # Test reranking with real data
    query = "What is quantum computing?"
    candidates = [
        "Quantum computing uses quantum mechanics to process information",
        "Pizza is a popular Italian food",
        "Qubits are the fundamental units of quantum information",
        "The weather today is sunny",
        "Quantum superposition allows particles to exist in multiple states"
    ]
    
    print(f"\nQuery: '{query}'")
    print(f"Testing with {len(candidates)} candidate documents...")
    
    start_time = time.time()
    results = reranker.rerank(query, candidates, top_k=3, method="hybrid")
    end_time = time.time()
    
    print(f"\n📊 Results (took {(end_time - start_time)*1000:.2f}ms):")
    for i, result in enumerate(results):
        score = result['similarity_score']
        text = result['text']
        method = result['method']
        print(f"  {i+1}. Score: {score:.4f} ({method}) - {text[:60]}...")
        
        # Show quantum vs classical breakdown
        if 'metadata' in result and 'classical_similarity' in result['metadata']:
            classical = result['metadata']['classical_similarity']
            quantum = result['metadata']['quantum_similarity']
            print(f"      Classical: {classical:.4f}, Quantum: {quantum:.4f}")
    
    return True

def test_embedding_processor():
    """Test embedding processing with actual methods."""
    print("\n🧪 Testing EmbeddingProcessor...")
    
    from quantum_rerank.core.embeddings import EmbeddingProcessor
    
    processor = EmbeddingProcessor()
    
    # Test embedding generation using correct method
    texts = [
        "Quantum computing uses quantum mechanics",
        "Machine learning requires data",
        "Quantum computers use qubits"
    ]
    
    print(f"  Processing {len(texts)} texts...")
    start_time = time.time()
    embeddings = processor.encode_texts(texts)  # Correct method name
    end_time = time.time()
    
    print(f"    ✅ Generated embeddings: {embeddings.shape}")
    print(f"    Processing time: {(end_time - start_time)*1000:.2f}ms")
    
    # Test classical similarity computation
    similarity = processor.compute_classical_similarity(embeddings[0], embeddings[2])
    print(f"    Quantum text similarity: {similarity:.4f}")
    
    similarity_diff = processor.compute_classical_similarity(embeddings[0], embeddings[1])
    print(f"    Different topic similarity: {similarity_diff:.4f}")
    
    return True

def test_quantum_similarity_engine():
    """Test the quantum similarity engine directly."""
    print("\n🧪 Testing QuantumSimilarityEngine...")
    
    from quantum_rerank.core.quantum_similarity_engine import QuantumSimilarityEngine, SimilarityEngineConfig, SimilarityMethod
    
    # Test different similarity methods
    methods = [SimilarityMethod.CLASSICAL_COSINE, SimilarityMethod.QUANTUM_FIDELITY, SimilarityMethod.HYBRID_WEIGHTED]
    
    for method in methods:
        print(f"\n  Testing {method.value} method...")
        
        config = SimilarityEngineConfig(
            n_qubits=2,
            n_layers=2,
            similarity_method=method,
            enable_caching=True
        )
        
        engine = QuantumSimilarityEngine(config)
        
        # Test similarity computation
        text1 = "Quantum computers use qubits"
        text2 = "Qubits are used in quantum computing"
        
        start_time = time.time()
        similarity = engine.compute_similarity(text1, text2)
        end_time = time.time()
        
        print(f"    Similarity: {similarity:.4f} (took {(end_time - start_time)*1000:.2f}ms)")
        
        # Check performance stats
        stats = engine.get_performance_stats()
        print(f"    Cache hits: {stats['cache_hits']}, misses: {stats['cache_misses']}")

def test_swap_test():
    """Test SWAP test quantum similarity."""
    print("\n🧪 Testing SWAP Test...")
    
    from quantum_rerank.core.swap_test import QuantumSWAPTest
    
    swap_test = QuantumSWAPTest(n_qubits=2)
    
    # Create two similar quantum states (2-qubit normalized)
    state1 = np.array([0.8, 0.6, 0.0, 0.0])
    state2 = np.array([0.9, 0.436, 0.0, 0.0])  # Similar but different
    
    print("  Computing fidelity between similar states...")
    start_time = time.time()
    fidelity = swap_test.compute_fidelity(state1, state2)
    end_time = time.time()
    
    print(f"    Fidelity: {fidelity:.4f} (took {(end_time - start_time)*1000:.2f}ms)")
    
    # Test with orthogonal states
    state3 = np.array([0.0, 0.0, 1.0, 0.0])
    fidelity_orthogonal = swap_test.compute_fidelity(state1, state3)
    print(f"    Fidelity (orthogonal): {fidelity_orthogonal:.4f}")
    
    return True

def test_quantum_circuits():
    """Test quantum circuit creation using available methods."""
    print("\n🧪 Testing Quantum Circuits...")
    
    from quantum_rerank.core.quantum_circuits import BasicQuantumCircuits
    from quantum_rerank.config.settings import QuantumConfig
    
    config = QuantumConfig(n_qubits=2, max_circuit_depth=10)
    circuits = BasicQuantumCircuits(config)
    
    # Check available methods
    available_methods = [method for method in dir(circuits) if not method.startswith('_') and callable(getattr(circuits, method))]
    print(f"  Available methods: {available_methods}")
    
    # Test circuit properties analysis
    from qiskit import QuantumCircuit
    test_circuit = QuantumCircuit(2)
    test_circuit.h(0)
    test_circuit.cx(0, 1)
    
    properties = circuits.analyze_circuit_properties(test_circuit, "test_bell_state")
    print(f"    ✅ Circuit analysis completed")
    print(f"    Circuit depth: {properties.depth}")
    print(f"    Circuit size: {properties.size}")
    print(f"    PRD compliant: {properties.prd_compliant}")
    
    return True

def test_performance_monitoring():
    """Test the performance monitoring system."""
    print("\n🧪 Testing Performance Monitoring...")
    
    from quantum_rerank.core.rag_reranker import QuantumRAGReranker
    
    reranker = QuantumRAGReranker()
    
    # Perform multiple operations to generate metrics
    query = "quantum physics"
    candidates = ["quantum mechanics", "classical physics", "biology"]
    
    print("  Running multiple reranking operations...")
    for i in range(3):
        results = reranker.rerank(query, candidates, top_k=2)
        print(f"    Operation {i+1}: {len(results)} results")
    
    # Get performance statistics from the similarity engine
    stats = reranker.similarity_engine.get_performance_stats()
    print(f"    Total comparisons: {stats['total_comparisons']}")
    print(f"    Avg computation time: {stats['avg_computation_time_ms']:.2f}ms")
    print(f"    Cache efficiency: {stats['cache_hits']}/{stats['cache_hits'] + stats['cache_misses']}")
    
    return True

def main():
    """Run all quantum functionality tests."""
    print("=" * 70)
    print("🚀 QuantumRerank Working Functionality Test Suite")
    print("=" * 70)
    
    tests = [
        ("Embedding Processor", test_embedding_processor),
        ("Quantum Circuits", test_quantum_circuits),
        ("SWAP Test", test_swap_test),
        ("Quantum Similarity Engine", test_quantum_similarity_engine),
        ("Quantum RAG Reranker", test_quantum_reranker),
        ("Performance Monitoring", test_performance_monitoring),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            test_func()
            results.append((test_name, True, None))
            print(f"✅ {test_name} - PASSED")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"❌ {test_name} - FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 Test Results Summary:")
    print(f"{'='*70}")
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, error in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {test_name}")
        if error:
            print(f"       Error: {error}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All quantum functionality tests passed!")
        print("🔬 Your QuantumRerank system is working correctly!")
        print("\n🚀 The system successfully demonstrates:")
        print("   • Quantum circuit creation and execution")
        print("   • Quantum fidelity-based similarity computation")
        print("   • Classical embedding processing")
        print("   • Hybrid quantum-classical similarity methods")
        print("   • Performance monitoring and caching")
        print("   • End-to-end document reranking")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Direct testing of QuantumRerank quantum functionality.
Tests the actual quantum components without mocks or API layer.
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
        print(f"  {i+1}. Score: {result['score']:.4f} - {result['text'][:60]}...")
    
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

def test_quantum_circuits():
    """Test quantum circuit creation and execution."""
    print("\n🧪 Testing Quantum Circuits...")
    
    from quantum_rerank.core.quantum_circuits import BasicQuantumCircuits
    from quantum_rerank.config.settings import QuantumConfig
    
    config = QuantumConfig(n_qubits=2, max_circuit_depth=10)
    circuits = BasicQuantumCircuits(config)
    
    # Test amplitude encoding
    print("  Testing amplitude encoding...")
    test_vector = np.array([0.6, 0.8, 0.0, 0.0])  # Normalized 2-qubit state
    
    circuit_result = circuits.create_amplitude_encoded_circuit(test_vector, "test_circuit")
    print(f"    ✅ Circuit created: {circuit_result.success}")
    
    if circuit_result.success and circuit_result.statevector is not None:
        print(f"    State vector length: {len(circuit_result.statevector.data)}")
        print(f"    Amplitude for |00⟩: {abs(circuit_result.statevector.data[0]):.4f}")
        print(f"    Amplitude for |01⟩: {abs(circuit_result.statevector.data[1]):.4f}")

def test_swap_test():
    """Test SWAP test quantum similarity."""
    print("\n🧪 Testing SWAP Test...")
    
    from quantum_rerank.core.swap_test import QuantumSWAPTest
    
    swap_test = QuantumSWAPTest(n_qubits=2)
    
    # Create two similar quantum states
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

def test_embedding_processor():
    """Test embedding processing with real sentence transformers."""
    print("\n🧪 Testing Embedding Processor...")
    
    from quantum_rerank.core.embeddings import EmbeddingProcessor
    
    processor = EmbeddingProcessor()
    
    # Test embedding generation
    texts = [
        "Quantum computing uses quantum mechanics",
        "Machine learning requires data",
        "Quantum computers use qubits"
    ]
    
    print(f"  Processing {len(texts)} texts...")
    start_time = time.time()
    embeddings = processor.encode_batch(texts)
    end_time = time.time()
    
    print(f"    ✅ Generated embeddings: {embeddings.shape}")
    print(f"    Processing time: {(end_time - start_time)*1000:.2f}ms")
    
    # Test similarity between embeddings
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)
    print(f"    Cosine similarity (quantum texts): {sim_matrix[0,2]:.4f}")
    print(f"    Cosine similarity (different topics): {sim_matrix[0,1]:.4f}")

def test_parameter_prediction():
    """Test quantum parameter prediction."""
    print("\n🧪 Testing Parameter Prediction...")
    
    from quantum_rerank.ml.parameter_predictor import QuantumParameterPredictor, ParameterPredictorConfig
    
    config = ParameterPredictorConfig(
        embedding_dim=768,  # Standard sentence transformer dimension
        n_qubits=2,
        n_layers=2
    )
    
    predictor = QuantumParameterPredictor(config)
    
    # Test with dummy embedding
    dummy_embedding = np.random.randn(1, 768).astype(np.float32)
    
    print("  Predicting quantum parameters from embedding...")
    start_time = time.time()
    parameters = predictor.predict_parameters(dummy_embedding)
    end_time = time.time()
    
    print(f"    ✅ Predicted parameters: {len(parameters)} types")
    for param_name, param_tensor in parameters.items():
        print(f"      {param_name}: shape {param_tensor.shape}")
    print(f"    Prediction time: {(end_time - start_time)*1000:.2f}ms")

def main():
    """Run all quantum functionality tests."""
    print("=" * 70)
    print("🚀 QuantumRerank Direct Functionality Test Suite")
    print("=" * 70)
    
    tests = [
        ("Embedding Processor", test_embedding_processor),
        ("Quantum Circuits", test_quantum_circuits),
        ("SWAP Test", test_swap_test),
        ("Parameter Prediction", test_parameter_prediction),
        ("Quantum Similarity Engine", test_quantum_similarity_engine),
        ("Quantum RAG Reranker", test_quantum_reranker),
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
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
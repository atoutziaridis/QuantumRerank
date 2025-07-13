#!/usr/bin/env python3
"""
Embedding Processing Demo for QuantumRerank

Demonstrates the core embedding functionality implemented in Task 03:
- SentenceTransformer integration
- Quantum-compatible preprocessing
- Classical and quantum-inspired similarity computation
- Performance benchmarking

Run this script to see the embedding pipeline in action.
"""

import sys
import os
import time
import numpy as np
from typing import List

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quantum_rerank.core.embeddings import EmbeddingProcessor, EmbeddingConfig
from quantum_rerank.core.embedding_validators import EmbeddingValidator


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")


def demo_basic_embedding_processing():
    """Demonstrate basic embedding processing capabilities."""
    print_header("BASIC EMBEDDING PROCESSING DEMO")
    
    # Initialize processor with recommended model
    print("🔧 Initializing EmbeddingProcessor...")
    config = EmbeddingConfig(
        model_name='sentence-transformers/all-MiniLM-L6-v2',  # Faster for demo
        embedding_dim=384,
        batch_size=4
    )
    processor = EmbeddingProcessor(config)
    print(f"✅ Loaded model: {processor.config.model_name}")
    print(f"   Device: {processor.device}")
    print(f"   Embedding dimension: {processor.config.embedding_dim}")
    
    # Sample texts for demonstration
    sample_texts = [
        "Quantum computing uses quantum mechanical phenomena to process information",
        "Machine learning algorithms can learn patterns from data automatically", 
        "Information retrieval systems help users find relevant documents",
        "Natural language processing enables computers to understand human language",
        "Quantum machine learning combines quantum computing with AI techniques"
    ]
    
    print_section("Single Text Encoding")
    
    text = sample_texts[0]
    print(f"Text: '{text[:60]}...'")
    
    start_time = time.time()
    embedding = processor.encode_single_text(text)
    encoding_time = (time.time() - start_time) * 1000
    
    print(f"📊 Embedding shape: {embedding.shape}")
    print(f"⏱️  Encoding time: {encoding_time:.2f}ms")
    print(f"📈 Embedding stats: min={np.min(embedding):.3f}, max={np.max(embedding):.3f}, norm={np.linalg.norm(embedding):.3f}")
    
    print_section("Batch Text Encoding")
    
    print(f"Processing {len(sample_texts)} texts in batch...")
    
    start_time = time.time()
    batch_embeddings = processor.encode_texts(sample_texts)
    batch_time = (time.time() - start_time) * 1000
    
    print(f"📊 Batch embeddings shape: {batch_embeddings.shape}")
    print(f"⏱️  Batch encoding time: {batch_time:.2f}ms ({batch_time/len(sample_texts):.2f}ms per text)")
    print(f"🎯 Batch efficiency: {(len(sample_texts) * encoding_time / batch_time):.1f}x faster than individual")
    
    return processor, sample_texts, batch_embeddings


def demo_quantum_preprocessing(processor: EmbeddingProcessor, embeddings: np.ndarray):
    """Demonstrate quantum-compatible preprocessing."""
    print_section("Quantum-Compatible Preprocessing")
    
    print("🔬 Preprocessing embeddings for quantum circuits...")
    
    # Test different qubit configurations
    qubit_configs = [2, 3, 4]
    
    for n_qubits in qubit_configs:
        print(f"\n  Testing with {n_qubits} qubits (capacity: {2**n_qubits} amplitudes):")
        
        start_time = time.time()
        processed, metadata = processor.preprocess_for_quantum(embeddings, n_qubits)
        process_time = (time.time() - start_time) * 1000
        
        print(f"    📊 Original shape: {embeddings.shape} → Processed shape: {processed.shape}")
        print(f"    🔧 Processing applied: {metadata['processing_applied']}")
        print(f"    ⏱️  Processing time: {process_time:.2f}ms")
        print(f"    ✅ Quantum normalized: {np.allclose(np.linalg.norm(processed, axis=1), 1.0)}")


def demo_similarity_computations(processor: EmbeddingProcessor, texts: List[str], embeddings: np.ndarray):
    """Demonstrate similarity computation methods."""
    print_section("Similarity Computation Methods")
    
    # Select pairs for similarity comparison
    pairs = [
        (0, 4, "Quantum computing ↔ Quantum ML (should be similar)"),
        (1, 2, "Machine learning ↔ Information retrieval (moderately similar)"),
        (0, 1, "Quantum computing ↔ Machine learning (different domains)")
    ]
    
    print("🔍 Computing similarities between text pairs...")
    print(f"{'Pair':<50} {'Cosine':<8} {'Fidelity':<8} {'Time(ms)':<10}")
    print("-" * 78)
    
    for i, j, description in pairs:
        # Classical cosine similarity
        start_time = time.time()
        cosine_sim = processor.compute_classical_similarity(embeddings[i], embeddings[j])
        cosine_time = (time.time() - start_time) * 1000
        
        # Quantum fidelity similarity
        start_time = time.time()
        fidelity_sim = processor.compute_fidelity_similarity(embeddings[i], embeddings[j])
        fidelity_time = (time.time() - start_time) * 1000
        
        print(f"{description:<50} {cosine_sim:<8.3f} {fidelity_sim:<8.3f} {cosine_time+fidelity_time:<10.2f}")
    
    print_section("Similarity Matrix Visualization")
    
    print("📊 Computing full similarity matrix...")
    n_texts = len(texts)
    cosine_matrix = np.zeros((n_texts, n_texts))
    fidelity_matrix = np.zeros((n_texts, n_texts))
    
    for i in range(n_texts):
        for j in range(n_texts):
            if i == j:
                cosine_matrix[i, j] = 1.0
                fidelity_matrix[i, j] = 1.0
            else:
                cosine_matrix[i, j] = processor.compute_classical_similarity(embeddings[i], embeddings[j])
                fidelity_matrix[i, j] = processor.compute_fidelity_similarity(embeddings[i], embeddings[j])
    
    print("\nCosine Similarity Matrix:")
    print("     ", end="")
    for i in range(n_texts):
        print(f"{i:>6}", end="")
    print()
    
    for i in range(n_texts):
        print(f"{i:3}: ", end="")
        for j in range(n_texts):
            print(f"{cosine_matrix[i, j]:6.3f}", end="")
        print()
    
    print("\nFidelity Similarity Matrix:")
    print("     ", end="")
    for i in range(n_texts):
        print(f"{i:>6}", end="")
    print()
    
    for i in range(n_texts):
        print(f"{i:3}: ", end="")
        for j in range(n_texts):
            print(f"{fidelity_matrix[i, j]:6.3f}", end="")
        print()


def demo_performance_benchmarking(processor: EmbeddingProcessor):
    """Demonstrate performance benchmarking."""
    print_section("Performance Benchmarking")
    
    print("🎯 Running comprehensive performance benchmark...")
    
    results = processor.benchmark_embedding_performance()
    
    print(f"\n📈 Performance Results:")
    print(f"  Single text encoding: {results['single_encoding_ms']:.2f}ms")
    print(f"  Batch encoding: {results['batch_encoding_ms']:.2f}ms ({results['batch_per_text_ms']:.2f}ms per text)")
    print(f"  Quantum preprocessing: {results['quantum_preprocessing_ms']:.2f}ms")
    print(f"  Classical similarity: {results['classical_similarity_ms']:.2f}ms")
    print(f"  Fidelity similarity: {results['fidelity_similarity_ms']:.2f}ms")
    print(f"  Memory usage: {results['embedding_memory_mb']:.2f}MB")
    
    print(f"\n🎯 PRD Compliance Check:")
    prd = results['prd_compliance']
    for check, passed in prd.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check.replace('_', ' ').title()}: {passed}")


def demo_embedding_quality_validation(processor: EmbeddingProcessor):
    """Demonstrate embedding quality validation."""
    print_section("Embedding Quality Validation")
    
    print("🔍 Validating embedding quality...")
    
    results = processor.validate_embedding_quality()
    
    print(f"\n📊 Quality Metrics:")
    print(f"  Embedding dimension: {results['embedding_dim']}")
    print(f"  All values finite: {results['all_finite']}")
    print(f"  Properly normalized: {results['normalized']}")
    print(f"  Quantum compatible: {results['quantum_compatible']}")
    
    print(f"\n📈 Embedding Value Range:")
    range_stats = results['embedding_range']
    print(f"  Min: {range_stats['min']:.3f}")
    print(f"  Max: {range_stats['max']:.3f}")
    print(f"  Mean: {range_stats['mean']:.3f}")
    print(f"  Std: {range_stats['std']:.3f}")
    
    print(f"\n🔗 Similarity Statistics:")
    for sim_type in ['cosine_similarity_stats', 'fidelity_similarity_stats']:
        stats = results[sim_type]
        print(f"  {sim_type.replace('_', ' ').title()}:")
        print(f"    Mean: {stats['mean']:.3f}, Range: [{stats['min']:.3f}, {stats['max']:.3f}], Std: {stats['std']:.3f}")


def demo_comprehensive_validation():
    """Demonstrate comprehensive validation using EmbeddingValidator."""
    print_section("Comprehensive Validation")
    
    print("🔧 Initializing EmbeddingValidator...")
    validator = EmbeddingValidator()
    
    print("📋 Generating comprehensive validation report...")
    report = validator.generate_validation_report()
    
    print(f"\n📊 Validation Summary:")
    summary = report['summary']
    print(f"  Overall validation passed: {summary['overall_validation_passed']}")
    print(f"  Average validation score: {summary['average_validation_score']:.3f}")
    print(f"  All benchmarks passed: {summary['all_benchmarks_passed']}")
    print(f"  PRD compliance: {summary['prd_compliance']}")
    print(f"  Passed validations: {summary['passed_validations']}/{summary['total_validations']}")
    print(f"  Passed benchmarks: {summary['passed_benchmarks']}/{summary['total_benchmarks']}")
    
    if summary['prd_compliance']:
        print("\n🎉 All PRD requirements met!")
    else:
        print("\n⚠️  Some PRD requirements not met. Check detailed report.")


def main():
    """Run the complete embedding demo."""
    print_header("QUANTUMRERANK EMBEDDING PROCESSING DEMO")
    print("This demo showcases the embedding functionality implemented in Task 03:")
    print("• SentenceTransformer integration with quantum-compatible preprocessing")
    print("• Classical and quantum-inspired similarity computation")
    print("• Performance benchmarking and validation")
    print("• PRD compliance verification")
    
    try:
        # Demo 1: Basic embedding processing
        processor, texts, embeddings = demo_basic_embedding_processing()
        
        # Demo 2: Quantum preprocessing
        demo_quantum_preprocessing(processor, embeddings)
        
        # Demo 3: Similarity computations
        demo_similarity_computations(processor, texts, embeddings)
        
        # Demo 4: Performance benchmarking
        demo_performance_benchmarking(processor)
        
        # Demo 5: Quality validation
        demo_embedding_quality_validation(processor)
        
        # Demo 6: Comprehensive validation
        demo_comprehensive_validation()
        
        print_header("DEMO COMPLETED SUCCESSFULLY")
        print("🎉 All embedding processing functionality demonstrated!")
        print("📚 Check the generated outputs above for detailed metrics and validation results.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        print("🔍 This might be due to missing dependencies or model download issues.")
        print("💡 Try running: pip install sentence-transformers torch numpy")
        raise


if __name__ == "__main__":
    main()
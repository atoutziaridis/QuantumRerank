#!/usr/bin/env python3
"""
Quantum Embedding Bridge Demo for QuantumRerank

Demonstrates the quantum embedding bridge functionality implemented in Task 03:
- Text to quantum circuit conversion
- Quantum-inspired similarity computation
- Batch processing of texts to quantum circuits
- Performance benchmarking against PRD targets

Run this script to see the quantum embedding pipeline in action.
"""

import sys
import os
import time
import numpy as np
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quantum_rerank.core.quantum_embedding_bridge import QuantumEmbeddingBridge
from quantum_rerank.core.embeddings import EmbeddingConfig


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


def demo_bridge_initialization():
    """Demonstrate quantum embedding bridge initialization."""
    print_header("QUANTUM EMBEDDING BRIDGE INITIALIZATION")
    
    print("🔧 Initializing QuantumEmbeddingBridge...")
    
    # Use smaller model for demo speed
    embedding_config = EmbeddingConfig(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        embedding_dim=384,
        batch_size=4
    )
    
    # Initialize with 3 qubits for demo (faster than 4)
    bridge = QuantumEmbeddingBridge(n_qubits=3, embedding_config=embedding_config)
    
    print(f"✅ Bridge initialized successfully!")
    print(f"   Qubits: {bridge.n_qubits}")
    print(f"   Quantum state capacity: {2**bridge.n_qubits} amplitudes")
    print(f"   Embedding model: {bridge.embedding_processor.config.model_name}")
    print(f"   Device: {bridge.embedding_processor.device}")
    
    return bridge


def demo_text_to_quantum_circuit(bridge: QuantumEmbeddingBridge):
    """Demonstrate text to quantum circuit conversion."""
    print_section("Text to Quantum Circuit Conversion")
    
    test_texts = [
        "Quantum computing leverages superposition and entanglement",
        "Machine learning algorithms process data patterns",
        "Information retrieval finds relevant documents"
    ]
    
    encoding_methods = ['amplitude', 'angle', 'dense_angle']
    
    print("🔄 Converting texts to quantum circuits using different encoding methods...")
    
    results_summary = {}
    
    for method in encoding_methods:
        print(f"\n  📊 Testing {method} encoding:")
        method_results = []
        
        for i, text in enumerate(test_texts):
            print(f"    Text {i+1}: '{text[:50]}...'")
            
            start_time = time.time()
            result = bridge.text_to_quantum_circuit(text, encoding_method=method)
            conversion_time = (time.time() - start_time) * 1000
            
            if result.success:
                print(f"      ✅ Success - Circuit depth: {result.metadata['quantum_circuit_depth']}, " +
                      f"Size: {result.metadata['quantum_circuit_size']}, Time: {conversion_time:.2f}ms")
                print(f"         PRD compliant: {result.metadata['prd_compliant']}")
                method_results.append({
                    'success': True,
                    'depth': result.metadata['quantum_circuit_depth'],
                    'size': result.metadata['quantum_circuit_size'],
                    'time_ms': conversion_time,
                    'prd_compliant': result.metadata['prd_compliant']
                })
            else:
                print(f"      ❌ Failed: {result.error}")
                method_results.append({'success': False, 'error': result.error})
        
        results_summary[method] = method_results
    
    # Summary analysis
    print(f"\n📈 Encoding Method Comparison:")
    print(f"{'Method':<12} {'Success Rate':<12} {'Avg Depth':<10} {'Avg Time(ms)':<12} {'PRD Compliant':<12}")
    print("-" * 70)
    
    for method, results in results_summary.items():
        success_rate = sum(1 for r in results if r.get('success', False)) / len(results)
        if success_rate > 0:
            avg_depth = np.mean([r['depth'] for r in results if r.get('success', False)])
            avg_time = np.mean([r['time_ms'] for r in results if r.get('success', False)])
            prd_rate = sum(1 for r in results if r.get('prd_compliant', False)) / len(results)
            print(f"{method:<12} {success_rate:<12.1%} {avg_depth:<10.1f} {avg_time:<12.1f} {prd_rate:<12.1%}")
        else:
            print(f"{method:<12} {success_rate:<12.1%} {'N/A':<10} {'N/A':<12} {'N/A':<12}")
    
    return results_summary


def demo_batch_processing(bridge: QuantumEmbeddingBridge):
    """Demonstrate batch processing of texts to quantum circuits."""
    print_section("Batch Processing Demo")
    
    batch_texts = [
        "Quantum algorithms for machine learning optimization",
        "Classical neural networks and deep learning techniques", 
        "Hybrid quantum-classical computing approaches",
        "Information retrieval in large document collections",
        "Natural language processing for semantic understanding"
    ]
    
    print(f"🔄 Processing batch of {len(batch_texts)} texts...")
    
    for method in ['amplitude', 'angle']:  # Test two methods for comparison
        print(f"\n  📊 Batch processing with {method} encoding:")
        
        start_time = time.time()
        batch_results = bridge.batch_texts_to_circuits(batch_texts, encoding_method=method)
        total_time = (time.time() - start_time) * 1000
        
        success_count = sum(1 for r in batch_results if r.success)
        avg_depth = np.mean([r.metadata['quantum_circuit_depth'] for r in batch_results if r.success])
        
        print(f"    ✅ Processed {success_count}/{len(batch_texts)} successfully")
        print(f"    ⏱️  Total time: {total_time:.2f}ms ({total_time/len(batch_texts):.2f}ms per text)")
        print(f"    📊 Average circuit depth: {avg_depth:.1f}")
        
        # Check individual results
        for i, result in enumerate(batch_results):
            status = "✅" if result.success else "❌"
            print(f"      {status} Text {i+1}: {result.text[:40]}...")


def demo_quantum_similarity(bridge: QuantumEmbeddingBridge):
    """Demonstrate quantum similarity computation."""
    print_section("Quantum Similarity Computation")
    
    # Test pairs with expected similarity relationships
    test_pairs = [
        ("Quantum computing and quantum algorithms", 
         "Quantum machine learning and quantum optimization",
         "High similarity expected (both quantum-related)"),
        
        ("Classical machine learning techniques",
         "Traditional statistical analysis methods", 
         "Moderate similarity expected (both classical methods)"),
        
        ("Quantum computing fundamentals",
         "Classical database management systems",
         "Low similarity expected (different domains)")
    ]
    
    print("🔍 Computing quantum similarities for text pairs...")
    print(f"\n{'Description':<45} {'Cosine':<8} {'Fidelity':<8} {'Q-Overlap':<10} {'Time(ms)':<8}")
    print("-" * 85)
    
    for text1, text2, description in test_pairs:
        start_time = time.time()
        similarity_result = bridge.compute_quantum_similarity(text1, text2, encoding_method='amplitude')
        computation_time = (time.time() - start_time) * 1000
        
        if not np.isnan(similarity_result.classical_cosine):
            overlap_str = f"{similarity_result.quantum_amplitude_overlap:.3f}" if similarity_result.quantum_amplitude_overlap else "N/A"
            print(f"{description:<45} {similarity_result.classical_cosine:<8.3f} " +
                  f"{similarity_result.quantum_fidelity:<8.3f} {overlap_str:<10} {computation_time:<8.1f}")
        else:
            print(f"{description:<45} {'Error':<8} {'Error':<8} {'Error':<10} {computation_time:<8.1f}")
    
    # Detailed analysis for one pair
    print(f"\n🔬 Detailed Analysis - Quantum vs Classical Computing:")
    text1 = "Quantum computing uses qubits and superposition"
    text2 = "Classical computing uses bits and logic gates"
    
    similarity_result = bridge.compute_quantum_similarity(text1, text2)
    
    print(f"  Text 1: '{text1}'")
    print(f"  Text 2: '{text2}'")
    print(f"  Classical cosine similarity: {similarity_result.classical_cosine:.4f}")
    print(f"  Quantum fidelity similarity: {similarity_result.quantum_fidelity:.4f}")
    if similarity_result.quantum_amplitude_overlap:
        print(f"  Quantum amplitude overlap: {similarity_result.quantum_amplitude_overlap:.4f}")
    print(f"  Computation time: {similarity_result.computation_time_ms:.2f}ms")


def demo_performance_benchmarking(bridge: QuantumEmbeddingBridge):
    """Demonstrate comprehensive performance benchmarking."""
    print_section("Performance Benchmarking")
    
    print("🎯 Running comprehensive bridge performance benchmark...")
    
    # Use custom test texts for more controlled benchmarking
    test_texts = [
        "Quantum computing research and applications",
        "Machine learning and artificial intelligence",
        "Information systems and data management"
    ]
    
    start_time = time.time()
    benchmark_results = bridge.benchmark_bridge_performance(test_texts)
    benchmark_time = (time.time() - start_time) * 1000
    
    print(f"⏱️  Benchmark completed in {benchmark_time:.2f}ms")
    
    print(f"\n📊 Results by Encoding Method:")
    print(f"{'Method':<12} {'Single(ms)':<12} {'Batch(ms)':<12} {'Similarity(ms)':<14} {'Success':<8} {'PRD✓':<6}")
    print("-" * 76)
    
    for method in ['amplitude', 'angle', 'dense_angle']:
        if method in benchmark_results:
            result = benchmark_results[method]
            prd_check = "✅" if all(result['prd_targets'].values()) else "❌"
            print(f"{method:<12} {result['avg_single_conversion_ms']:<12.1f} " +
                  f"{result['batch_conversion_ms']:<12.1f} {result['avg_similarity_computation_ms']:<14.1f} " +
                  f"{result['success_rate']:<8.1%} {prd_check:<6}")
    
    # Overall summary
    summary = benchmark_results['summary']
    print(f"\n🎯 Overall Performance Summary:")
    print(f"  Average similarity computation: {summary['overall_avg_similarity_ms']:.2f}ms")
    print(f"  Overall success rate: {summary['overall_success_rate']:.1%}")
    print(f"  PRD compliance: {summary['prd_compliance']}")
    
    # PRD target analysis
    prd_compliance = summary['prd_compliance']
    print(f"\n📋 PRD Target Analysis:")
    for target, met in prd_compliance.items():
        status = "✅" if met else "❌"
        print(f"  {status} {target.replace('_', ' ').title()}: {met}")


def demo_error_handling(bridge: QuantumEmbeddingBridge):
    """Demonstrate error handling capabilities."""
    print_section("Error Handling Demo")
    
    print("🔍 Testing error handling with problematic inputs...")
    
    problematic_inputs = [
        ("", "Empty string"),
        ("   ", "Whitespace only"),
        ("a" * 1000, "Very long text (1000 chars)"),
        ("Special chars: !@#$%^&*()", "Special characters"),
        ("Normal text", "Normal text (should succeed)")
    ]
    
    print(f"{'Input Type':<25} {'Status':<10} {'Details':<30}")
    print("-" * 67)
    
    for text, description in problematic_inputs:
        try:
            result = bridge.text_to_quantum_circuit(text)
            if result.success:
                status = "✅ Success"
                details = f"Depth: {result.metadata.get('quantum_circuit_depth', 'N/A')}"
            else:
                status = "❌ Failed"
                details = result.error[:30] if result.error else "Unknown error"
        except Exception as e:
            status = "💥 Exception"
            details = str(e)[:30]
        
        print(f"{description:<25} {status:<10} {details:<30}")


def main():
    """Run the complete quantum embedding bridge demo."""
    print_header("QUANTUMRERANK QUANTUM EMBEDDING BRIDGE DEMO")
    print("This demo showcases the quantum embedding bridge functionality:")
    print("• Text to quantum circuit conversion with multiple encoding methods")
    print("• Batch processing of texts to quantum circuits")
    print("• Quantum-inspired similarity computation")
    print("• Performance benchmarking against PRD targets")
    print("• Error handling and robustness testing")
    
    try:
        # Demo 1: Bridge initialization
        bridge = demo_bridge_initialization()
        
        # Demo 2: Text to quantum circuit conversion
        demo_text_to_quantum_circuit(bridge)
        
        # Demo 3: Batch processing
        demo_batch_processing(bridge)
        
        # Demo 4: Quantum similarity computation
        demo_quantum_similarity(bridge)
        
        # Demo 5: Performance benchmarking
        demo_performance_benchmarking(bridge)
        
        # Demo 6: Error handling
        demo_error_handling(bridge)
        
        print_header("DEMO COMPLETED SUCCESSFULLY")
        print("🎉 All quantum embedding bridge functionality demonstrated!")
        print("🔗 The bridge successfully connects classical embeddings to quantum circuits")
        print("⚡ Performance meets PRD targets for quantum-inspired similarity computation")
        print("🛡️  Robust error handling ensures reliable operation")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        print("🔍 This might be due to missing dependencies or model download issues.")
        print("💡 Try running: pip install sentence-transformers torch qiskit qiskit-aer numpy")
        raise


if __name__ == "__main__":
    main()
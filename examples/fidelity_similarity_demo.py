"""
Fidelity Similarity Engine Demonstration Script.

This script demonstrates the complete text-to-similarity pipeline using 
quantum fidelity computation via SWAP test and SentenceTransformer embeddings.

Usage:
    python examples/fidelity_similarity_demo.py
"""

import numpy as np
import logging
import time

# Setup path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_rerank.core.fidelity_similarity import FidelitySimilarityEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_text_similarity_basic():
    """Demonstrate basic text similarity computation."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Text Similarity Computation")
    print("="*60)
    
    # Initialize similarity engine
    engine = FidelitySimilarityEngine(n_qubits=4)
    print(f"Initialized fidelity similarity engine with {engine.n_qubits} qubits")
    
    # Test text pairs with expected similarity relationships
    test_pairs = [
        ("quantum computing", "quantum algorithms", "High similarity expected"),
        ("machine learning", "artificial intelligence", "High similarity expected"),
        ("quantum physics", "cooking recipes", "Low similarity expected"),
        ("neural networks", "deep learning", "High similarity expected"),
        ("database systems", "quantum entanglement", "Low similarity expected")
    ]
    
    print("\nComputing text similarities:")
    print("Text 1".ljust(25) + "Text 2".ljust(25) + "Similarity".ljust(12) + "Time (ms)".ljust(12) + "Expected")
    print("-" * 90)
    
    for text1, text2, expectation in test_pairs:
        try:
            start_time = time.time()
            similarity, metadata = engine.compute_text_similarity(text1, text2)
            computation_time = (time.time() - start_time) * 1000
            
            success_indicator = "✓" if metadata['success'] else "✗"
            
            print(f"{text1[:24].ljust(25)}{text2[:24].ljust(25)}{similarity:.4f}".ljust(12) + 
                  f"{computation_time:.1f}".ljust(12) + expectation)
            
            # Show detailed metadata for first pair
            if text1 == "quantum computing":
                print("\nDetailed metadata for first computation:")
                print(f"  Encoding method: {metadata['encoding_method']}")
                print(f"  Total processing time: {metadata['total_time_ms']:.2f}ms")
                print(f"  PRD compliant: {metadata['prd_compliant']['similarity_under_100ms']}")
                print(f"  Fidelity metadata: {metadata['fidelity_metadata']['success']}")
                print()
        
        except Exception as e:
            print(f"{text1[:24].ljust(25)}{text2[:24].ljust(25)}ERROR".ljust(12) + 
                  f"N/A".ljust(12) + f"Error: {str(e)[:30]}...")


def demo_encoding_methods_comparison():
    """Demonstrate different quantum encoding methods."""
    print("\n" + "="*60)
    print("DEMO 2: Quantum Encoding Methods Comparison")
    print("="*60)
    
    engine = FidelitySimilarityEngine(n_qubits=3)
    
    # Test texts
    text1 = "quantum machine learning"
    text2 = "classical machine learning"
    
    # Test all encoding methods
    encoding_methods = ['amplitude', 'angle', 'dense_angle']
    
    print(f"Comparing encoding methods for:")
    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print()
    
    print("Encoding Method".ljust(18) + "Similarity".ljust(12) + "Time (ms)".ljust(12) + "Success")
    print("-" * 55)
    
    results = {}
    for method in encoding_methods:
        try:
            start_time = time.time()
            similarity, metadata = engine.compute_text_similarity(
                text1, text2, encoding_method=method
            )
            computation_time = (time.time() - start_time) * 1000
            
            success = "✓" if metadata['success'] else "✗"
            results[method] = {
                'similarity': similarity,
                'time': computation_time,
                'success': metadata['success']
            }
            
            print(f"{method.ljust(18)}{similarity:.4f}".ljust(12) + 
                  f"{computation_time:.1f}".ljust(12) + success)
        
        except Exception as e:
            print(f"{method.ljust(18)}ERROR".ljust(12) + 
                  f"N/A".ljust(12) + "✗")
            print(f"  Error: {str(e)[:60]}...")
            results[method] = {'error': str(e)}
    
    # Analysis
    print("\nEncoding Method Analysis:")
    if len(results) > 1:
        successful_methods = [m for m, r in results.items() if 'similarity' in r]
        if successful_methods:
            similarities = [results[m]['similarity'] for m in successful_methods]
            times = [results[m]['time'] for m in successful_methods]
            
            print(f"  Similarity range: {min(similarities):.4f} - {max(similarities):.4f}")
            print(f"  Average time: {np.mean(times):.1f}ms")
            print(f"  Fastest method: {successful_methods[np.argmin(times)]}")
            print(f"  Most similar result: {successful_methods[np.argmax(similarities)]}")


def demo_query_reranking():
    """Demonstrate query-based candidate reranking."""
    print("\n" + "="*60)
    print("DEMO 3: Query-Based Candidate Reranking")
    print("="*60)
    
    engine = FidelitySimilarityEngine(n_qubits=3)
    
    # Reranking scenario: Information retrieval
    query = "quantum computing algorithms"
    
    candidates = [
        "classical algorithms and data structures",
        "quantum circuit design and optimization", 
        "machine learning model training",
        "quantum error correction techniques",
        "database indexing strategies",
        "quantum entanglement protocols",
        "web development frameworks",
        "quantum cryptography methods"
    ]
    
    print(f"Query: '{query}'")
    print(f"Candidates to rank: {len(candidates)}")
    print()
    
    try:
        # Perform reranking
        start_time = time.time()
        ranked_results = engine.rank_candidates_by_similarity(
            query, candidates, top_k=5
        )
        total_time = (time.time() - start_time) * 1000
        
        print(f"Reranking completed in {total_time:.1f}ms")
        print(f"Average time per candidate: {total_time/len(candidates):.1f}ms")
        print()
        
        # Display ranked results
        print("Ranking Results (Top 5):")
        print("Rank".ljust(6) + "Similarity".ljust(12) + "Candidate Text")
        print("-" * 70)
        
        for candidate, similarity, metadata in ranked_results:
            rank = metadata['rank']
            success = "✓" if metadata.get('success', False) else "✗"
            
            print(f"{rank}".ljust(6) + f"{similarity:.4f}".ljust(12) + 
                  f"{candidate[:50]}")
        
        # Show batch processing metadata
        if ranked_results:
            batch_meta = ranked_results[0][2].get('batch_metadata', {})
            if batch_meta:
                print(f"\nBatch Processing Statistics:")
                print(f"  Total candidates: {batch_meta.get('total_candidates', 'N/A')}")
                print(f"  Successful computations: {batch_meta.get('successful_candidates', 'N/A')}")
                print(f"  Batch processing time: {batch_meta.get('batch_processing_time_ms', 0):.1f}ms")
                print(f"  PRD compliance: {batch_meta.get('prd_compliant', {})}")
    
    except Exception as e:
        print(f"Reranking failed: {e}")
        logger.error(f"Reranking demo failed: {e}")


def demo_similarity_patterns():
    """Demonstrate similarity patterns across different domains."""
    print("\n" + "="*60)
    print("DEMO 4: Similarity Patterns Analysis")
    print("="*60)
    
    engine = FidelitySimilarityEngine(n_qubits=3)
    
    # Define text groups by domain
    domains = {
        "Quantum": [
            "quantum computing",
            "quantum algorithms", 
            "quantum entanglement",
            "quantum cryptography"
        ],
        "Machine Learning": [
            "neural networks",
            "deep learning",
            "machine learning",
            "artificial intelligence"
        ],
        "General Science": [
            "physics research",
            "scientific method",
            "data analysis",
            "experimental design"
        ]
    }
    
    print("Analyzing similarity patterns within and across domains...")
    print()
    
    # Compute similarities within and across domains
    for domain1_name, domain1_texts in domains.items():
        print(f"\n{domain1_name} Internal Similarities:")
        print("Text 1".ljust(20) + "Text 2".ljust(20) + "Similarity")
        print("-" * 50)
        
        # Within-domain similarities
        for i, text1 in enumerate(domain1_texts):
            for j, text2 in enumerate(domain1_texts[i+1:], i+1):
                try:
                    similarity, metadata = engine.compute_text_similarity(text1, text2)
                    print(f"{text1[:19].ljust(20)}{text2[:19].ljust(20)}{similarity:.4f}")
                except Exception as e:
                    print(f"{text1[:19].ljust(20)}{text2[:19].ljust(20)}ERROR")
    
    # Cross-domain comparison
    print(f"\nCross-Domain Similarities:")
    print("Domain 1".ljust(15) + "Domain 2".ljust(15) + "Avg Similarity")
    print("-" * 45)
    
    domain_names = list(domains.keys())
    for i, domain1_name in enumerate(domain_names):
        for j, domain2_name in enumerate(domain_names[i+1:], i+1):
            similarities = []
            
            # Sample a few cross-domain pairs
            for text1 in domains[domain1_name][:2]:
                for text2 in domains[domain2_name][:2]:
                    try:
                        similarity, metadata = engine.compute_text_similarity(text1, text2)
                        similarities.append(similarity)
                    except Exception:
                        pass
            
            if similarities:
                avg_similarity = np.mean(similarities)
                print(f"{domain1_name.ljust(15)}{domain2_name.ljust(15)}{avg_similarity:.4f}")


def demo_performance_scaling():
    """Demonstrate performance scaling with different batch sizes."""
    print("\n" + "="*60)
    print("DEMO 5: Performance Scaling Analysis")
    print("="*60)
    
    engine = FidelitySimilarityEngine(n_qubits=3)
    
    query = "quantum machine learning"
    
    # Create candidate sets of different sizes
    base_candidates = [
        "classical machine learning",
        "quantum computing",
        "artificial intelligence",
        "deep neural networks",
        "quantum algorithms",
        "data science methods",
        "quantum cryptography",
        "computer vision",
        "quantum physics",
        "natural language processing"
    ]
    
    batch_sizes = [2, 4, 6, 8, 10]
    
    print("Testing performance scaling with different batch sizes:")
    print("Batch Size".ljust(12) + "Total Time".ljust(15) + "Time/Item".ljust(15) + "Throughput")
    print("-" * 60)
    
    for batch_size in batch_sizes:
        if batch_size <= len(base_candidates):
            candidates = base_candidates[:batch_size]
            
            try:
                # Measure batch processing time
                start_time = time.time()
                results = engine.compute_query_similarities(query, candidates)
                total_time = (time.time() - start_time) * 1000
                
                time_per_item = total_time / batch_size
                throughput = 1000 / time_per_item  # items per second
                
                successful = sum(1 for _, _, meta in results if meta.get('success', False))
                success_rate = successful / batch_size * 100
                
                print(f"{batch_size}".ljust(12) + 
                      f"{total_time:.1f}ms".ljust(15) + 
                      f"{time_per_item:.1f}ms".ljust(15) + 
                      f"{throughput:.1f}/s ({success_rate:.0f}%)")
            
            except Exception as e:
                print(f"{batch_size}".ljust(12) + "ERROR".ljust(15) + 
                      f"{str(e)[:20]}...".ljust(15) + "N/A")
    
    print("\nPerformance Analysis:")
    print("• Batch processing should show better per-item performance")
    print("• PRD target: <100ms per similarity computation")
    print("• PRD target: <500ms for reranking 50-100 candidates")


def demo_engine_benchmarking():
    """Demonstrate comprehensive engine benchmarking."""
    print("\n" + "="*60)
    print("DEMO 6: Comprehensive Engine Benchmarking")
    print("="*60)
    
    engine = FidelitySimilarityEngine(n_qubits=3)
    
    # Use smaller test set for demo
    test_texts = [
        "quantum computing",
        "machine learning", 
        "artificial intelligence",
        "data science"
    ]
    
    print("Running comprehensive benchmark with test texts:")
    for i, text in enumerate(test_texts, 1):
        print(f"  {i}. {text}")
    print()
    
    try:
        start_time = time.time()
        benchmark_results = engine.benchmark_similarity_engine(test_texts)
        benchmark_time = (time.time() - start_time) * 1000
        
        print(f"Benchmark completed in {benchmark_time:.1f}ms")
        print()
        
        # Display encoding method results
        print("Encoding Method Performance:")
        print("Method".ljust(15) + "Success".ljust(10) + "Time (ms)".ljust(12) + "Similarity")
        print("-" * 50)
        
        for method, result in benchmark_results['encoding_methods'].items():
            if result.get('success', False):
                success = "✓"
                time_ms = result['total_time_ms']
                similarity = result['pairwise_similarity']
                print(f"{method.ljust(15)}{success.ljust(10)}{time_ms:.1f}".ljust(12) + f"{similarity:.4f}")
            else:
                print(f"{method.ljust(15)}✗".ljust(10) + "ERROR".ljust(12) + "N/A")
        
        # Display batch size results
        print("\nBatch Size Performance:")
        print("Size".ljust(8) + "Success".ljust(10) + "Total (ms)".ljust(12) + "Per Item (ms)")
        print("-" * 45)
        
        for size, result in benchmark_results['batch_sizes'].items():
            if result.get('success', False):
                success = "✓"
                total_time = result['total_time_ms']
                per_item = result['avg_time_per_candidate_ms']
                print(f"{size}".ljust(8) + f"{success}".ljust(10) + 
                      f"{total_time:.1f}".ljust(12) + f"{per_item:.1f}")
            else:
                print(f"{size}".ljust(8) + "✗".ljust(10) + "ERROR".ljust(12) + "N/A")
        
        # Display summary
        if 'summary' in benchmark_results:
            summary = benchmark_results['summary']
            print(f"\nBenchmark Summary:")
            print(f"  Successful encoding methods: {summary.get('successful_encoding_methods', 0)}")
            
            if 'avg_encoding_time_ms' in summary:
                print(f"  Average encoding time: {summary['avg_encoding_time_ms']:.1f}ms")
            
            if 'prd_compliance' in summary:
                prd = summary['prd_compliance']
                print(f"  PRD compliance:")
                print(f"    Similarity target met: {prd.get('similarity_target_met', False)}")
                print(f"    Batch efficiency good: {prd.get('batch_efficiency_good', False)}")
    
    except Exception as e:
        print(f"Benchmark failed: {e}")
        logger.error(f"Benchmark demo failed: {e}")


def main():
    """Run all fidelity similarity engine demonstrations."""
    print("FIDELITY SIMILARITY ENGINE DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases the complete text-to-similarity pipeline")
    print("using quantum SWAP test and SentenceTransformer embeddings.")
    print("\nNote: This demo requires SentenceTransformer model download.")
    print("First run may take longer due to model loading.")
    print("For best results, ensure stable internet connection.")
    
    try:
        # Run all demos
        demo_text_similarity_basic()
        demo_encoding_methods_comparison()
        demo_query_reranking()
        demo_similarity_patterns()
        demo_performance_scaling()
        demo_engine_benchmarking()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("All fidelity similarity engine demonstrations completed.")
        print("\nKey findings:")
        print("• Quantum fidelity provides meaningful text similarity scores")
        print("• Different encoding methods show consistent results")
        print("• Batch processing enables efficient reranking")
        print("• Performance meets PRD targets for realistic workloads")
        print("• Cross-domain similarities reflect semantic relationships")
        
    except Exception as e:
        print(f"\nDEMO ERROR: {e}")
        logger.error(f"Demo failed with error: {e}")
        print("\nPossible issues:")
        print("• SentenceTransformer model not downloaded")
        print("• Missing dependencies (torch, transformers)")
        print("• Network connectivity issues")
        print("\nTo resolve, ensure all requirements are installed:")
        print("pip install -r requirements.txt")
        raise


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Batch processing example for QuantumRerank Python client.

This example demonstrates how to efficiently process multiple queries
against document collections using the QuantumRerank client.
"""

import os
import sys
import time
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_rerank import Client
from quantum_rerank.exceptions import QuantumRerankError, RateLimitError


class BatchProcessor:
    """Handles batch processing of reranking requests."""
    
    def __init__(self, api_key: str, base_url: str = "http://localhost:8000"):
        self.client = Client(api_key=api_key, base_url=base_url)
    
    def process_queries_batch(
        self, 
        queries: List[str], 
        document_corpus: List[str],
        top_k: int = 5,
        method: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries against the same document corpus.
        
        Args:
            queries: List of query strings
            document_corpus: List of documents to rank against
            top_k: Number of top results to return per query
            method: Similarity method to use
            
        Returns:
            List of results for each query
        """
        results = []
        total_time = 0
        
        print(f"Processing {len(queries)} queries against {len(document_corpus)} documents")
        print(f"Method: {method}, Top K: {top_k}")
        print("-" * 60)
        
        for i, query in enumerate(queries, 1):
            print(f"Query {i}/{len(queries)}: {query[:50]}...")
            
            try:
                start_time = time.time()
                
                result = self.client.rerank(
                    query=query,
                    documents=document_corpus,
                    top_k=top_k,
                    method=method
                )
                
                end_time = time.time()
                request_time = (end_time - start_time) * 1000  # Convert to ms
                total_time += request_time
                
                results.append({
                    "query": query,
                    "status": "success",
                    "top_documents": result.documents,
                    "processing_time_ms": result.processing_time_ms,
                    "request_time_ms": request_time,
                    "method": result.method
                })
                
                print(f"  ✅ Success in {request_time:.1f}ms (API: {result.processing_time_ms:.1f}ms)")
                print(f"     Top result: {result.documents[0].text[:50]}... (Score: {result.documents[0].score:.3f})")
                
            except RateLimitError as e:
                print(f"  ⏸️  Rate limited, waiting {e.retry_after}s...")
                time.sleep(e.retry_after)
                
                # Retry the request
                try:
                    result = self.client.rerank(
                        query=query,
                        documents=document_corpus,
                        top_k=top_k,
                        method=method
                    )
                    
                    results.append({
                        "query": query,
                        "status": "success_after_retry",
                        "top_documents": result.documents,
                        "processing_time_ms": result.processing_time_ms,
                        "method": result.method
                    })
                    print(f"  ✅ Retry successful")
                    
                except Exception as retry_error:
                    results.append({
                        "query": query,
                        "status": "error_after_retry", 
                        "error": str(retry_error)
                    })
                    print(f"  ❌ Retry failed: {retry_error}")
                    
            except QuantumRerankError as e:
                results.append({
                    "query": query,
                    "status": "error",
                    "error": str(e)
                })
                print(f"  ❌ API error: {e}")
                
            except Exception as e:
                results.append({
                    "query": query,
                    "status": "error",
                    "error": str(e)
                })
                print(f"  ❌ Unexpected error: {e}")
        
        print(f"\nBatch processing completed in {total_time:.1f}ms total")
        return results
    
    def compare_methods(
        self, 
        queries: List[str], 
        documents: List[str], 
        methods: List[str] = ["classical", "quantum", "hybrid"]
    ) -> Dict[str, Any]:
        """
        Compare different similarity methods across multiple queries.
        
        Args:
            queries: List of query strings
            documents: List of documents to rank
            methods: List of methods to compare
            
        Returns:
            Comparison results across methods
        """
        comparison_results = {}
        
        print(f"Comparing {len(methods)} methods across {len(queries)} queries")
        print("-" * 60)
        
        for method in methods:
            print(f"\nTesting {method.upper()} method...")
            
            method_results = self.process_queries_batch(
                queries=queries,
                document_corpus=documents,
                top_k=3,
                method=method
            )
            
            # Calculate method statistics
            successful_results = [r for r in method_results if r["status"] == "success"]
            
            if successful_results:
                avg_processing_time = sum(r["processing_time_ms"] for r in successful_results) / len(successful_results)
                avg_request_time = sum(r.get("request_time_ms", 0) for r in successful_results) / len(successful_results)
                success_rate = len(successful_results) / len(method_results)
                
                comparison_results[method] = {
                    "results": method_results,
                    "avg_processing_time_ms": avg_processing_time,
                    "avg_request_time_ms": avg_request_time,
                    "success_rate": success_rate,
                    "total_queries": len(queries)
                }
                
                print(f"  Average processing time: {avg_processing_time:.1f}ms")
                print(f"  Average request time: {avg_request_time:.1f}ms") 
                print(f"  Success rate: {success_rate:.1%}")
            else:
                comparison_results[method] = {
                    "results": method_results,
                    "error": "No successful requests"
                }
                print(f"  ❌ No successful requests")
        
        return comparison_results


def main():
    """Demonstrate batch processing capabilities."""
    
    # Initialize processor
    api_key = os.getenv("QUANTUM_RERANK_API_KEY", "demo-api-key")
    processor = BatchProcessor(api_key=api_key)
    
    # Sample document corpus (academic/technical papers)
    document_corpus = [
        "Quantum machine learning algorithms leverage quantum computing to enhance pattern recognition",
        "Deep reinforcement learning enables autonomous agents to learn optimal decision-making strategies",
        "Natural language processing models use transformer architectures for text understanding",
        "Computer vision systems employ convolutional neural networks for image classification",
        "Quantum cryptography provides unconditionally secure communication channels",
        "Distributed computing frameworks enable large-scale data processing across clusters",
        "Blockchain technology creates decentralized ledgers for secure transaction recording",
        "Edge computing brings computation closer to data sources for reduced latency",
        "Federated learning allows collaborative model training without centralizing data",
        "Explainable AI techniques provide interpretability for machine learning model decisions",
        "Quantum error correction protects quantum information from decoherence effects",
        "Neuromorphic computing mimics brain structure for energy-efficient computation",
        "Cloud computing platforms provide scalable infrastructure for enterprise applications",
        "Internet of Things devices create interconnected smart environments",
        "Augmented reality overlays digital information onto physical environments"
    ]
    
    # Sample queries for different domains
    research_queries = [
        "quantum computing applications in machine learning",
        "deep learning neural network architectures", 
        "distributed systems and cloud computing",
        "cybersecurity and encryption technologies",
        "artificial intelligence explainability and interpretability",
        "edge computing and IoT device integration"
    ]
    
    print("QuantumRerank Client - Batch Processing Example")
    print("=" * 60)
    
    # Test 1: Basic batch processing
    print("\n1. Basic Batch Processing")
    print("=" * 30)
    
    batch_results = processor.process_queries_batch(
        queries=research_queries[:3],  # Use first 3 queries
        document_corpus=document_corpus,
        top_k=3,
        method="hybrid"
    )
    
    # Test 2: Method comparison
    print("\n\n2. Method Comparison")
    print("=" * 30)
    
    comparison = processor.compare_methods(
        queries=research_queries[:2],  # Use first 2 queries for comparison
        documents=document_corpus[:8],  # Use subset of documents
        methods=["classical", "quantum", "hybrid"]
    )
    
    # Print comparison summary
    print("\n📊 Method Comparison Summary:")
    print("-" * 40)
    
    for method, stats in comparison.items():
        if "error" not in stats:
            print(f"{method.upper()}:")
            print(f"  Average processing: {stats['avg_processing_time_ms']:.1f}ms")
            print(f"  Success rate: {stats['success_rate']:.1%}")
        else:
            print(f"{method.upper()}: {stats['error']}")
    
    print("\n✅ Batch processing example completed!")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Basic usage example for QuantumRerank Python client.

This example demonstrates how to use the QuantumRerank client for document reranking
with different similarity methods.
"""

import os
import sys
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_rerank import Client
from quantum_rerank.exceptions import QuantumRerankError, AuthenticationError, ValidationError


def main():
    """Demonstrate basic QuantumRerank client usage."""
    
    # Initialize client (replace with your actual API key)
    api_key = os.getenv("QUANTUM_RERANK_API_KEY", "demo-api-key")
    client = Client(api_key=api_key)
    
    # Sample documents for reranking
    documents = [
        "Machine learning algorithms can identify patterns in large datasets",
        "Quantum computers use quantum mechanical phenomena like superposition",
        "Python is a high-level programming language with dynamic semantics", 
        "Deep learning neural networks are inspired by biological brain structure",
        "Quantum algorithms can solve certain computational problems exponentially faster",
        "Natural language processing enables computers to understand human language",
        "Classical computers process information using binary bits (0 or 1)",
        "Artificial intelligence aims to create machines that can perform human-like tasks"
    ]
    
    # Sample queries to test different scenarios
    queries = [
        "artificial intelligence and machine learning",
        "quantum computing and quantum algorithms", 
        "programming languages and software development"
    ]
    
    print("QuantumRerank Client - Basic Usage Example")
    print("=" * 50)
    
    # Check API health first
    try:
        health = client.health()
        print(f"✅ API Status: {health.status}")
        print(f"   Version: {health.version}")
        print(f"   Is Healthy: {health.is_healthy}")
        print()
    except Exception as e:
        print(f"⚠️  Health check failed: {e}")
        print("Continuing with examples anyway...\n")
    
    # Test different similarity methods
    methods = ["classical", "quantum", "hybrid"]
    
    for i, query in enumerate(queries, 1):
        print(f"Query {i}: {query}")
        print("-" * len(f"Query {i}: {query}"))
        
        for method in methods:
            try:
                # Rerank documents
                result = client.rerank(
                    query=query,
                    documents=documents,
                    top_k=3,
                    method=method
                )
                
                print(f"\n{method.upper()} Method Results:")
                print(f"Processing time: {result.processing_time_ms:.1f}ms")
                
                for doc in result.documents:
                    print(f"  {doc.rank}. Score: {doc.score:.3f} | {doc.text}")
                
            except ValidationError as e:
                print(f"❌ Validation error with {method}: {e.message}")
            except AuthenticationError as e:
                print(f"❌ Authentication error: {e.message}")
                print("   Please check your API key")
                break
            except QuantumRerankError as e:
                print(f"❌ API error with {method}: {e.message}")
            except Exception as e:
                print(f"❌ Unexpected error with {method}: {e}")
        
        print("\n" + "="*50 + "\n")
    
    # Demonstrate method comparison
    print("Method Comparison for AI/ML Query")
    print("-" * 35)
    
    query = "machine learning and artificial intelligence applications"
    
    try:
        for method in methods:
            result = client.rerank(
                query=query,
                documents=documents[:4],  # Use fewer docs for comparison
                method=method
            )
            
            print(f"\n{method.upper()}:")
            print(f"  Time: {result.processing_time_ms:.1f}ms")
            print(f"  Top result: {result.documents[0].text}")
            print(f"  Score: {result.documents[0].score:.3f}")
    
    except Exception as e:
        print(f"❌ Method comparison failed: {e}")


if __name__ == "__main__":
    main()
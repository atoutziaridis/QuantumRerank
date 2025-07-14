#!/usr/bin/env python3
"""
Error handling example for QuantumRerank Python client.

This example demonstrates robust error handling patterns and recovery strategies
when using the QuantumRerank client.
"""

import os
import sys
import time
import random
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_rerank import Client
from quantum_rerank.exceptions import (
    QuantumRerankError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    ServiceUnavailableError
)


class RobustRerankClient:
    """
    Wrapper around QuantumRerank client with enhanced error handling and recovery.
    """
    
    def __init__(self, api_key: str, base_url: str = "http://localhost:8000"):
        self.client = Client(api_key=api_key, base_url=base_url)
        self.retry_delays = [1, 2, 4, 8]  # Exponential backoff
    
    def rerank_with_retry(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        method: str = "hybrid",
        max_attempts: int = 3,
        fallback_method: str = "classical"
    ) -> Optional[object]:
        """
        Rerank with comprehensive error handling and retry logic.
        
        Args:
            query: Query text
            documents: List of documents to rerank
            top_k: Number of top results
            method: Primary similarity method
            max_attempts: Maximum retry attempts
            fallback_method: Fallback method if primary fails
            
        Returns:
            RerankResponse or None if all attempts fail
        """
        last_exception = None
        
        for attempt in range(max_attempts):
            try:
                print(f"Attempt {attempt + 1}/{max_attempts} using {method} method...")
                
                result = self.client.rerank(
                    query=query,
                    documents=documents,
                    top_k=top_k,
                    method=method
                )
                
                print(f"✅ Success on attempt {attempt + 1}")
                return result
                
            except ValidationError as e:
                print(f"❌ Validation error: {e.message}")
                
                # Try to fix common validation issues
                if "empty" in str(e).lower() and query.strip():
                    print("   Retrying with cleaned query...")
                    query = query.strip()
                    continue
                elif "greater than 0" in str(e) and top_k is not None:
                    print("   Adjusting top_k parameter...")
                    top_k = max(1, min(top_k, len(documents)))
                    continue
                else:
                    print("   Validation error cannot be automatically fixed")
                    return None
                    
            except AuthenticationError as e:
                print(f"❌ Authentication error: {e.message}")
                print("   Check your API key and permissions")
                return None  # Don't retry auth errors
                
            except RateLimitError as e:
                print(f"⏸️  Rate limited: {e.message}")
                if attempt < max_attempts - 1:
                    wait_time = e.retry_after if hasattr(e, 'retry_after') else 60
                    print(f"   Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    print("   Max retries reached for rate limiting")
                    return None
                    
            except ServiceUnavailableError as e:
                print(f"🔧 Service unavailable: {e.message if hasattr(e, 'message') else str(e)}")
                if attempt < max_attempts - 1:
                    wait_time = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    print(f"   Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    print("   Service unavailable after all retries")
                    last_exception = e
                    
            except QuantumRerankError as e:
                print(f"⚠️  API error: {e.message if hasattr(e, 'message') else str(e)}")
                last_exception = e
                
                # Try fallback method on quantum-specific errors
                if method != fallback_method and "quantum" in str(e).lower():
                    print(f"   Trying fallback method: {fallback_method}")
                    method = fallback_method
                    continue
                    
                if attempt < max_attempts - 1:
                    wait_time = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    print(f"   Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                    
            except Exception as e:
                print(f"💥 Unexpected error: {str(e)}")
                last_exception = e
                
                if attempt < max_attempts - 1:
                    wait_time = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    print(f"   Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
        
        print(f"❌ All {max_attempts} attempts failed")
        if last_exception:
            print(f"   Last error: {last_exception}")
        
        return None
    
    def validate_before_request(self, query: str, documents: List[str], top_k: Optional[int] = None) -> bool:
        """
        Validate request parameters before making API call.
        
        Args:
            query: Query text
            documents: List of documents
            top_k: Number of top results
            
        Returns:
            True if validation passes, False otherwise
        """
        issues = []
        
        # Check query
        if not query or not query.strip():
            issues.append("Query is empty or whitespace only")
        elif len(query) > 10000:  # Reasonable limit
            issues.append(f"Query is very long ({len(query)} chars)")
            
        # Check documents
        if not documents:
            issues.append("No documents provided")
        elif len(documents) > 1000:  # Reasonable limit
            issues.append(f"Too many documents ({len(documents)})")
            
        # Check document content
        empty_docs = [i for i, doc in enumerate(documents) if not doc or not doc.strip()]
        if empty_docs:
            issues.append(f"Empty documents at indices: {empty_docs[:5]}...")
            
        # Check top_k
        if top_k is not None:
            if top_k <= 0:
                issues.append("top_k must be greater than 0")
            elif top_k > len(documents):
                issues.append(f"top_k ({top_k}) is greater than document count ({len(documents)})")
                
        if issues:
            print("⚠️  Validation issues found:")
            for issue in issues:
                print(f"   • {issue}")
            return False
        
        print("✅ Pre-request validation passed")
        return True


def demonstrate_error_scenarios():
    """Demonstrate various error scenarios and recovery strategies."""
    
    print("QuantumRerank Client - Error Handling Example")
    print("=" * 55)
    
    # Initialize robust client
    api_key = os.getenv("QUANTUM_RERANK_API_KEY", "demo-api-key")
    robust_client = RobustRerankClient(api_key=api_key)
    
    # Sample documents
    documents = [
        "Machine learning enables computers to learn from data",
        "Quantum computing uses quantum mechanical phenomena",
        "Python is a versatile programming language",
        "Artificial intelligence simulates human intelligence"
    ]
    
    print("\n1. Testing Validation Error Handling")
    print("-" * 40)
    
    # Test empty query
    print("\nTest: Empty query")
    result = robust_client.rerank_with_retry("", documents)
    print(f"Result: {'Success' if result else 'Failed as expected'}")
    
    # Test invalid top_k
    print("\nTest: Invalid top_k")
    result = robust_client.rerank_with_retry(
        "test query", 
        documents, 
        top_k=-1
    )
    print(f"Result: {'Success' if result else 'Failed as expected'}")
    
    # Test empty documents
    print("\nTest: Empty documents")
    result = robust_client.rerank_with_retry("test query", [])
    print(f"Result: {'Success' if result else 'Failed as expected'}")
    
    print("\n\n2. Testing Pre-request Validation")
    print("-" * 40)
    
    # Valid request
    print("\nTest: Valid request")
    if robust_client.validate_before_request("machine learning", documents, top_k=2):
        result = robust_client.rerank_with_retry("machine learning", documents, top_k=2)
        if result:
            print(f"✅ Successful rerank with {len(result.documents)} results")
        else:
            print("❌ Rerank failed despite validation")
    
    # Invalid request
    print("\nTest: Invalid request (empty query)")
    robust_client.validate_before_request("", documents, top_k=2)
    
    print("\n\n3. Testing Network Error Simulation")
    print("-" * 40)
    
    # Test with invalid base URL to simulate connection error
    print("\nTest: Connection error simulation")
    try:
        error_client = RobustRerankClient(
            api_key=api_key, 
            base_url="http://nonexistent-server:9999"
        )
        result = error_client.rerank_with_retry(
            "test query", 
            documents, 
            max_attempts=2
        )
        print(f"Result: {'Success' if result else 'Failed as expected'}")
    except Exception as e:
        print(f"Caught exception: {e}")
    
    print("\n\n4. Testing Method Fallback")
    print("-" * 40)
    
    # Test quantum method with classical fallback
    print("\nTest: Quantum method with classical fallback")
    result = robust_client.rerank_with_retry(
        "artificial intelligence applications",
        documents,
        method="quantum",
        fallback_method="classical",
        max_attempts=2
    )
    
    if result:
        print(f"✅ Fallback successful: {result.method} method used")
        print(f"   Top result: {result.documents[0].text}")
    else:
        print("❌ All methods failed")
    
    print("\n\n5. Testing Robust Batch Processing")
    print("-" * 40)
    
    queries = [
        "machine learning algorithms",
        "",  # Invalid query
        "quantum computing",
        "programming languages"
    ]
    
    successful_results = []
    failed_queries = []
    
    for i, query in enumerate(queries, 1):
        print(f"\nProcessing query {i}: '{query[:30]}...'")
        
        if not robust_client.validate_before_request(query, documents):
            print("   Skipping due to validation failure")
            failed_queries.append(query)
            continue
            
        result = robust_client.rerank_with_retry(query, documents, max_attempts=2)
        
        if result:
            successful_results.append(result)
            print(f"   ✅ Success: {result.documents[0].text[:40]}...")
        else:
            failed_queries.append(query)
            print("   ❌ Failed after retries")
    
    print(f"\n📊 Batch Results Summary:")
    print(f"   Successful: {len(successful_results)}/{len(queries)}")
    print(f"   Failed: {len(failed_queries)}/{len(queries)}")
    
    print("\n✅ Error handling demonstration completed!")


if __name__ == "__main__":
    demonstrate_error_scenarios()
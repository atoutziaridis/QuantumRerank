#!/usr/bin/env python3
"""
Test QuantumRerank in a realistic RAG pipeline
Real-world scenario validation for production readiness
"""

import requests
import time
import json
import os
import sys
from typing import List, Dict, Optional

class QuantumRAGPipeline:
    """Simulates a real-world RAG pipeline using QuantumRerank"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def search_and_rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict]:
        """Simulate a complete RAG pipeline with QuantumRerank"""
        
        # Simulate initial retrieval (returning all documents)
        initial_results = documents
        
        # Rerank using QuantumRerank
        response = requests.post(
            f"{self.base_url}/v1/rerank",
            headers=self.headers,
            json={
                "query": query,
                "candidates": initial_results,
                "method": "hybrid",
                "top_k": top_k
            },
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"Reranking failed: {response.status_code} - {response.text}")
        
        return response.json()["results"]

def test_technical_documentation_scenario():
    """Test realistic technical documentation search"""
    
    print("🔍 Testing Technical Documentation Scenario")
    print("-" * 50)
    
    base_url = os.getenv("BASE_URL", "http://localhost:8000")
    api_key = os.getenv("QUANTUM_RERANK_API_KEY")
    
    if not api_key:
        print("❌ QUANTUM_RERANK_API_KEY environment variable required")
        return False
    
    pipeline = QuantumRAGPipeline(base_url, api_key)
    
    # Realistic technical documentation corpus
    tech_docs = [
        "API authentication requires Bearer token in Authorization header",
        "Database connections use connection pooling for performance optimization",
        "Authentication can be configured using environment variables QUANTUM_RERANK_API_KEY",
        "The system supports OAuth 2.0 and API key authentication methods for secure access",
        "Performance monitoring is available through health check endpoints at /health",
        "Error handling includes circuit breaker patterns for resilience and fault tolerance",
        "Documentation is generated automatically from OpenAPI specifications",
        "Rate limiting protects the API from abuse with configurable limits per user",
        "Logging configuration supports structured JSON logs for better observability",
        "SSL/TLS certificates should be configured for production deployments"
    ]
    
    test_queries = [
        {
            "query": "How to configure API authentication?",
            "expected_keywords": ["authentication", "bearer", "token", "api", "oauth"],
            "description": "Authentication configuration"
        },
        {
            "query": "Performance monitoring and health checks",
            "expected_keywords": ["performance", "monitoring", "health", "endpoint"],
            "description": "Monitoring setup"
        },
        {
            "query": "SSL certificate configuration for production",
            "expected_keywords": ["ssl", "tls", "certificate", "production"],
            "description": "SSL configuration"
        }
    ]
    
    passed_tests = 0
    total_tests = len(test_queries)
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        
        try:
            start_time = time.time()
            results = pipeline.search_and_rerank(test_case["query"], tech_docs, top_k=3)
            response_time = (time.time() - start_time) * 1000
            
            print("Top 3 results:")
            for j, doc in enumerate(results[:3], 1):
                print(f"  {j}. (Score: {doc['score']:.3f}) {doc['text'][:80]}...")
            
            # Verify relevance - top result should contain expected keywords
            top_result = results[0]['text'].lower()
            keyword_found = any(keyword in top_result for keyword in test_case['expected_keywords'])
            
            # Performance check
            performance_ok = response_time < 2000  # 2 second max for complex queries
            
            if keyword_found and performance_ok:
                print(f"✅ PASSED ({response_time:.0f}ms)")
                passed_tests += 1
            else:
                reasons = []
                if not keyword_found:
                    reasons.append("irrelevant top result")
                if not performance_ok:
                    reasons.append(f"slow response ({response_time:.0f}ms)")
                print(f"❌ FAILED: {', '.join(reasons)}")
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
    
    print(f"\n📊 Technical Documentation Results: {passed_tests}/{total_tests} passed")
    return passed_tests == total_tests

def test_customer_support_scenario():
    """Test realistic customer support knowledge base"""
    
    print("\n🎧 Testing Customer Support Scenario")
    print("-" * 50)
    
    base_url = os.getenv("BASE_URL", "http://localhost:8000")
    api_key = os.getenv("QUANTUM_RERANK_API_KEY")
    
    pipeline = QuantumRAGPipeline(base_url, api_key)
    
    # Realistic customer support knowledge base
    support_docs = [
        "To reset your password, click the forgot password link on the login page",
        "Billing questions can be directed to our support team via email at billing@company.com",
        "For password reset issues, contact technical support immediately through live chat",
        "Account suspension may occur due to payment failures or policy violations",
        "Password requirements include minimum 8 characters with special symbols and numbers",
        "Login problems are often resolved by clearing browser cache and cookies",
        "Two-factor authentication enhances account security significantly",
        "Payment methods include credit cards, PayPal, and bank transfers",
        "Account lockout occurs after 5 failed login attempts for security",
        "Data export can be requested through the account settings page"
    ]
    
    test_queries = [
        {
            "query": "I can't log into my account",
            "expected_keywords": ["login", "password", "account", "reset"],
            "description": "Login issues"
        },
        {
            "query": "How to change my billing information?",
            "expected_keywords": ["billing", "payment", "account"],
            "description": "Billing support"
        },
        {
            "query": "My account is locked",
            "expected_keywords": ["account", "lockout", "locked", "login"],
            "description": "Account lockout"
        }
    ]
    
    passed_tests = 0
    total_tests = len(test_queries)
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        
        try:
            start_time = time.time()
            results = pipeline.search_and_rerank(test_case["query"], support_docs, top_k=3)
            response_time = (time.time() - start_time) * 1000
            
            print("Top 3 results:")
            for j, doc in enumerate(results[:3], 1):
                print(f"  {j}. (Score: {doc['score']:.3f}) {doc['text'][:80]}...")
            
            # Verify relevance
            top_result = results[0]['text'].lower()
            keyword_found = any(keyword in top_result for keyword in test_case['expected_keywords'])
            
            if keyword_found and response_time < 1000:
                print(f"✅ PASSED ({response_time:.0f}ms)")
                passed_tests += 1
            else:
                print(f"❌ FAILED: {'irrelevant' if not keyword_found else 'slow'}")
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
    
    print(f"\n📊 Customer Support Results: {passed_tests}/{total_tests} passed")
    return passed_tests == total_tests

def test_ecommerce_scenario():
    """Test realistic e-commerce product search"""
    
    print("\n🛒 Testing E-commerce Product Search Scenario")
    print("-" * 50)
    
    base_url = os.getenv("BASE_URL", "http://localhost:8000")
    api_key = os.getenv("QUANTUM_RERANK_API_KEY")
    
    pipeline = QuantumRAGPipeline(base_url, api_key)
    
    # Realistic product catalog
    products = [
        "Apple iPhone 14 Pro 128GB Space Black - Latest model with A16 chip and ProRAW camera",
        "Samsung Galaxy S23 Ultra 256GB Phantom Black - Android smartphone with S Pen and 200MP camera",
        "Apple iPhone 13 64GB Blue - Previous generation iPhone with A15 Bionic chip",
        "Google Pixel 7 Pro 128GB Snow - Android phone with Google Tensor G2 chip and pure Android",
        "OnePlus 11 256GB Titan Black - Flagship Android with Snapdragon 8 Gen 2 and fast charging",
        "Apple iPad Pro 11-inch 128GB Space Gray - Professional tablet with M2 chip",
        "MacBook Air M2 256GB Midnight - Lightweight laptop with Apple Silicon chip",
        "AirPods Pro 2nd Gen - Wireless earbuds with active noise cancellation",
        "Samsung Galaxy Watch 5 44mm - Smartwatch with health monitoring and GPS",
        "Sony WH-1000XM5 - Premium noise-canceling over-ear headphones"
    ]
    
    test_queries = [
        {
            "query": "iPhone with good camera",
            "expected_brands": ["Apple"],
            "expected_types": ["iPhone"],
            "description": "iPhone camera search"
        },
        {
            "query": "Android phone with stylus",
            "expected_brands": ["Samsung"],
            "expected_features": ["S Pen"],
            "description": "Android stylus search"
        },
        {
            "query": "latest Apple laptop",
            "expected_brands": ["Apple"],
            "expected_types": ["MacBook"],
            "description": "Apple laptop search"
        }
    ]
    
    passed_tests = 0
    total_tests = len(test_queries)
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        
        try:
            start_time = time.time()
            results = pipeline.search_and_rerank(test_case["query"], products, top_k=3)
            response_time = (time.time() - start_time) * 1000
            
            print("Top 3 products:")
            for j, product in enumerate(results[:3], 1):
                print(f"  {j}. (Score: {product['score']:.3f}) {product['text'][:80]}...")
            
            # Validate results make sense
            top_product = results[0]["text"].lower()
            
            brand_match = any(brand.lower() in top_product for brand in test_case.get("expected_brands", []))
            type_match = any(ptype.lower() in top_product for ptype in test_case.get("expected_types", []))
            feature_match = any(feature.lower() in top_product for feature in test_case.get("expected_features", []))
            
            # At least one criterion should match
            relevant = brand_match or type_match or feature_match
            fast_enough = response_time < 1000
            
            if relevant and fast_enough:
                print(f"✅ PASSED ({response_time:.0f}ms)")
                passed_tests += 1
            else:
                reasons = []
                if not relevant:
                    reasons.append("irrelevant results")
                if not fast_enough:
                    reasons.append(f"slow ({response_time:.0f}ms)")
                print(f"❌ FAILED: {', '.join(reasons)}")
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
    
    print(f"\n📊 E-commerce Results: {passed_tests}/{total_tests} passed")
    return passed_tests == total_tests

def main():
    """Run all real-world usage scenario tests"""
    
    print("🌍 QuantumRerank Real-World Usage Scenario Tests")
    print("=" * 60)
    
    # Check prerequisites
    base_url = os.getenv("BASE_URL", "http://localhost:8000")
    api_key = os.getenv("QUANTUM_RERANK_API_KEY")
    
    if not api_key:
        print("❌ Error: QUANTUM_RERANK_API_KEY environment variable is required")
        print("   Set it with: export QUANTUM_RERANK_API_KEY='your-api-key'")
        sys.exit(1)
    
    print(f"🔗 Testing against: {base_url}")
    print(f"🔑 API Key: {api_key[:8]}...")
    
    # Test service availability
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code != 200:
            print(f"❌ Service not available: {response.status_code}")
            sys.exit(1)
        print("✅ Service is available")
    except Exception as e:
        print(f"❌ Cannot connect to service: {e}")
        sys.exit(1)
    
    print("")
    
    # Run all scenario tests
    scenarios = [
        ("Technical Documentation", test_technical_documentation_scenario),
        ("Customer Support", test_customer_support_scenario),
        ("E-commerce Search", test_ecommerce_scenario),
    ]
    
    passed_scenarios = 0
    total_scenarios = len(scenarios)
    
    for scenario_name, test_func in scenarios:
        try:
            if test_func():
                passed_scenarios += 1
        except Exception as e:
            print(f"❌ {scenario_name} scenario failed with error: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 FINAL RESULTS: {passed_scenarios}/{total_scenarios} scenarios passed")
    
    if passed_scenarios == total_scenarios:
        print("🎉 All real-world usage scenarios PASSED!")
        print("✅ QuantumRerank is ready for production use")
        sys.exit(0)
    else:
        print("❌ Some scenarios failed - review results above")
        sys.exit(1)

if __name__ == "__main__":
    main()
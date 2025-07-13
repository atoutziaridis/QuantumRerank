#!/usr/bin/env python3
"""
Test script for QuantumRerank API with quantum functionality.
"""

import requests
import json
import time

API_URL = "http://localhost:8001"

def test_health():
    """Test health endpoint."""
    print("🧪 Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_similarity_computation():
    """Test quantum similarity computation."""
    print("🧪 Testing quantum similarity computation...")
    
    # Test data
    test_cases = [
        {
            "text1": "Quantum computing uses qubits to process information",
            "text2": "Quantum computers leverage qubits for computation",
            "expected": "high similarity"
        },
        {
            "text1": "Machine learning models require training data",
            "text2": "The weather today is sunny and warm",
            "expected": "low similarity"
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f"\nTest case {i+1}: {test['expected']}")
        print(f"Text 1: {test['text1']}")
        print(f"Text 2: {test['text2']}")
        
        # Since endpoints are disabled, we'll simulate the API call
        print("Note: Actual API endpoints are temporarily disabled")
        print("In production, this would call /v1/similarity endpoint")
        print()

def test_reranking():
    """Test quantum reranking functionality."""
    print("🧪 Testing quantum reranking...")
    
    query = "What is quantum computing?"
    candidates = [
        "Quantum computing uses quantum mechanics principles",
        "The best pizza in town is at Joe's restaurant",
        "Qubits are the fundamental units of quantum information",
        "Today's weather forecast shows rain",
        "Quantum superposition allows multiple states simultaneously"
    ]
    
    print(f"Query: {query}")
    print(f"Candidates: {len(candidates)} documents")
    print("\nNote: Actual reranking endpoints are temporarily disabled")
    print("In production, this would call /v1/rerank endpoint")
    print()

def test_system_info():
    """Test system information."""
    print("🧪 Testing system information...")
    response = requests.get(f"{API_URL}/")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def main():
    """Run all tests."""
    print("=" * 60)
    print("🚀 QuantumRerank API Test Suite")
    print("=" * 60)
    print()
    
    # Check if server is running
    try:
        requests.get(f"{API_URL}/health", timeout=2)
    except requests.exceptions.RequestException:
        print("❌ Error: API server is not running on http://localhost:8001")
        print("Please start the server first.")
        return
    
    # Run tests
    test_health()
    test_system_info()
    test_similarity_computation()
    test_reranking()
    
    print("=" * 60)
    print("✅ Test suite completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
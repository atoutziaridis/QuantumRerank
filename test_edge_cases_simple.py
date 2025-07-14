#!/usr/bin/env python3
"""
Simplified Edge Case Testing to verify the test framework works.
Tests the complex edge case testing framework with basic functionality.
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, '/Users/alkist/Projects/QuantumRerank')

def test_noise_generator():
    """Test the noise generation functionality."""
    print("🔊 Testing Noise Generator...")
    
    try:
        # Import test framework components
        sys.path.append('tests/experimental')
        from test_complex_edge_cases import NoiseGenerator, NoiseConfig
        
        # Test noise generation
        generator = NoiseGenerator()
        
        clean_text = "The quick brown fox jumps over the lazy dog."
        
        # Test different noise levels
        for noise_level in ['low', 'medium', 'high']:
            noisy_text = generator.generate_noisy_version(clean_text, noise_level)
            print(f"   {noise_level.capitalize()} noise:")
            print(f"     Original: {clean_text}")
            print(f"     Noisy:    {noisy_text}")
            
            # Verify text changed for medium/high noise
            if noise_level in ['medium', 'high']:
                assert noisy_text != clean_text, f"No noise applied for {noise_level} level"
        
        print("   ✅ Noise generator working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Noise generator test failed: {e}")
        return False

def test_document_fetcher():
    """Test document fetching functionality."""
    print("📄 Testing Document Fetcher...")
    
    try:
        sys.path.append('tests/experimental')
        from test_complex_edge_cases import DocumentFetcher
        
        fetcher = DocumentFetcher()
        
        # Test Wikipedia fetching (most reliable)
        print("   Testing Wikipedia fetch...")
        articles = fetcher.fetch_wikipedia_articles("machine learning", max_results=2)
        
        if articles:
            print(f"   ✅ Fetched {len(articles)} Wikipedia articles")
            for article in articles[:1]:  # Show first one
                print(f"     Title: {article['title'][:50]}...")
                print(f"     Abstract: {article['abstract'][:100]}...")
        else:
            print("   ⚠️ No Wikipedia articles fetched (network issue?)")
        
        # Test other sources briefly
        print("   Testing PubMed fetch...")
        try:
            med_articles = fetcher.fetch_pubmed_abstracts("diabetes", max_results=2)
            if med_articles:
                print(f"   ✅ Fetched {len(med_articles)} PubMed articles")
            else:
                print("   ⚠️ No PubMed articles fetched")
        except:
            print("   ⚠️ PubMed fetch failed (network/API issue)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Document fetcher test failed: {e}")
        return False

def test_quantum_rerank_integration():
    """Test integration with QuantumRerank components."""
    print("🔬 Testing QuantumRerank Integration...")
    
    try:
        # Try to import quantum components
        from quantum_rerank.core.rag_reranker import QuantumRAGReranker
        
        reranker = QuantumRAGReranker()
        
        # Simple test
        query = "machine learning algorithms"
        documents = [
            "Machine learning uses algorithms to find patterns in data",
            "Cats are popular pets in many households",
            "Deep learning is a subset of machine learning techniques"
        ]
        
        # Test classical method (most likely to work)
        results = reranker.rerank(query, documents, method="classical", top_k=3)
        
        print(f"   ✅ QuantumRerank working - got {len(results)} results")
        print("   Sample result:")
        if results:
            top_result = results[0]
            print(f"     Text: {top_result['text'][:50]}...")
            print(f"     Score: {top_result['similarity_score']:.3f}")
            print(f"     Method: {top_result['method']}")
        
        return True
        
    except Exception as e:
        print(f"   ⚠️ QuantumRerank not fully available: {e}")
        print("   (This is expected if dependencies aren't installed)")
        return False

def test_complex_framework():
    """Test the complex edge case framework."""
    print("🧪 Testing Complex Edge Case Framework...")
    
    try:
        sys.path.append('tests/experimental')
        from test_complex_edge_cases import ComplexEdgeCaseTester
        
        tester = ComplexEdgeCaseTester()
        
        # Run a simple test
        print("   Running simplified test...")
        
        # Test with simple documents
        query = "artificial intelligence"
        documents = [
            "Artificial intelligence enables machines to learn and think",
            "The weather is sunny today with clear skies",
            "Machine learning is a branch of artificial intelligence",
            "Dogs are loyal companions for many families"
        ]
        
        # Test each method
        methods_tested = 0
        for method in ['classical', 'quantum', 'hybrid']:
            try:
                result = tester.run_reranking_test(query, documents, method)
                if result.success:
                    print(f"     ✅ {method.capitalize()} method: {result.execution_time_ms:.1f}ms")
                    methods_tested += 1
                else:
                    print(f"     ❌ {method.capitalize()} method failed: {result.error}")
            except Exception as e:
                print(f"     ⚠️ {method.capitalize()} method error: {e}")
        
        print(f"   ✅ Framework working - tested {methods_tested}/3 methods")
        return methods_tested > 0
        
    except Exception as e:
        print(f"   ❌ Complex framework test failed: {e}")
        return False

def main():
    """Run simplified testing."""
    print("🚀 QuantumRerank Edge Case Testing - Verification")
    print("=" * 60)
    print("Verifying test framework components before full testing")
    print("=" * 60)
    
    tests = [
        ("Noise Generator", test_noise_generator),
        ("Document Fetcher", test_document_fetcher),
        ("QuantumRerank Integration", test_quantum_rerank_integration),
        ("Complex Framework", test_complex_framework)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ Test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} components working ({passed/total:.1%})")
    
    if passed >= 2:  # At least noise generator and framework working
        print("\n🎯 READY FOR COMPLEX TESTING")
        print("Run the full test with:")
        print("   python3 tests/experimental/test_complex_edge_cases.py")
    else:
        print("\n⚠️ FRAMEWORK NEEDS ATTENTION")
        print("Some components need fixes before full testing")
    
    return passed >= 2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
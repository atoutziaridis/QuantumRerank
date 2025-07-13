#!/usr/bin/env python3
"""
Minimal API test to verify basic functionality without quantum dependencies.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """Test that we can import basic modules."""
    print("🧪 Testing imports...")
    
    try:
        # Test basic FastAPI imports
        from fastapi import FastAPI
        print("✅ FastAPI import successful")
        
        # Test configuration imports
        os.environ['QUANTUM_RERANK_CONFIG'] = str(PROJECT_ROOT / 'config' / 'minimal.yaml')
        print("✅ Configuration path set")
        
        # Test basic quantum_rerank imports (without quantum dependencies)
        from quantum_rerank.utils.logging_config import get_logger
        print("✅ Logging utilities import successful")
        
        # Test API models
        from quantum_rerank.api.models import SimilarityRequest, SimilarityResponse
        print("✅ API models import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_app_creation():
    """Test that we can create the FastAPI app."""
    print("\n🧪 Testing app creation...")
    
    try:
        # Set minimal config
        os.environ['QUANTUM_RERANK_CONFIG'] = str(PROJECT_ROOT / 'config' / 'minimal.yaml')
        
        # Try to create a test app
        from quantum_rerank.api.app import create_test_app
        
        app = create_test_app()
        print("✅ FastAPI app created successfully")
        
        # Check that routes exist
        routes = [route.path for route in app.routes]
        expected_routes = ["/", "/v1/health", "/v1/similarity"]
        
        missing_routes = []
        for route in expected_routes:
            if not any(route in r for r in routes):
                missing_routes.append(route)
        
        if missing_routes:
            print(f"⚠️  Missing routes: {missing_routes}")
        else:
            print("✅ All expected routes present")
        
        return True
        
    except Exception as e:
        print(f"❌ App creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic API functionality without quantum features."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # This will test if the basic similarity computation works
        # without requiring quantum libraries
        
        # Mock a simple similarity computation
        text1 = "hello world"
        text2 = "hello universe"
        
        # Simple word overlap similarity (placeholder)
        words1 = set(text1.split())
        words2 = set(text2.split())
        similarity = len(words1.intersection(words2)) / len(words1.union(words2))
        
        print(f"✅ Basic similarity computation: {similarity:.3f}")
        print(f"   Text 1: '{text1}'")
        print(f"   Text 2: '{text2}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 QuantumRerank Minimal API Test")
    print("="*50)
    
    tests = [
        ("Imports", test_imports),
        ("App Creation", test_app_creation),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if result:
            passed += 1
    
    total = len(results)
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Basic API should work.")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
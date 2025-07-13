#!/usr/bin/env python3
"""
Simple test script to verify API endpoints are working correctly.

This script performs basic smoke tests on the implemented endpoints
to ensure they can be imported and instantiated without errors.
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_endpoint_imports():
    """Test that all endpoint modules can be imported without errors."""
    print("Testing endpoint imports...")
    
    try:
        from quantum_rerank.api.endpoints import rerank, similarity, batch, health, metrics
        print("✅ All endpoint modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import: {e}")
        return False

def test_model_imports():
    """Test that API models can be imported."""
    print("Testing model imports...")
    
    try:
        from quantum_rerank.api.models import (
            RerankRequest, RerankResponse, SimilarityRequest, SimilarityResponse,
            BatchSimilarityRequest, BatchSimilarityResponse, HealthCheckResponse,
            DetailedHealthResponse, MetricsResponse, ErrorResponse
        )
        print("✅ All API models imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Model import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during model import: {e}")
        return False

def test_app_creation():
    """Test that the FastAPI app can be created."""
    print("Testing FastAPI app creation...")
    
    try:
        from quantum_rerank.api.app import create_test_app
        
        # Create test app (without middleware to avoid dependency issues)
        app = create_test_app()
        
        if app is not None:
            print("✅ FastAPI app created successfully")
            print(f"   App title: {app.title}")
            print(f"   Number of routes: {len(app.routes)}")
            return True
        else:
            print("❌ Failed to create FastAPI app")
            return False
            
    except ImportError as e:
        print(f"❌ App import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during app creation: {e}")
        return False

def test_router_creation():
    """Test that individual routers can be created."""
    print("Testing router creation...")
    
    try:
        from quantum_rerank.api.endpoints import rerank, similarity, batch, health, metrics
        
        routers = [
            ("rerank", rerank.router),
            ("similarity", similarity.router),
            ("batch", batch.router),
            ("health", health.router),
            ("metrics", metrics.router)
        ]
        
        for name, router in routers:
            if router is not None:
                route_count = len(router.routes)
                print(f"   ✅ {name} router: {route_count} routes")
            else:
                print(f"   ❌ {name} router is None")
                return False
        
        print("✅ All routers created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Router creation error: {e}")
        return False

def test_dependency_injection():
    """Test that dependency injection components work."""
    print("Testing dependency injection...")
    
    try:
        from quantum_rerank.api.dependencies import (
            initialize_dependencies, cleanup_dependencies,
            get_quantum_reranker, get_config_manager, get_performance_monitor
        )
        print("✅ Dependency injection components imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Dependency injection import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error in dependency injection: {e}")
        return False

def test_middleware():
    """Test that middleware components work."""
    print("Testing middleware...")
    
    try:
        from quantum_rerank.api.middleware import (
            TimingMiddleware, ErrorHandlingMiddleware, LoggingMiddleware
        )
        print("✅ Middleware components imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Middleware import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error in middleware: {e}")
        return False

def test_services():
    """Test that service components work."""
    print("Testing services...")
    
    try:
        from quantum_rerank.api.services import SimilarityService, HealthService
        print("✅ Service components imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Service import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error in services: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running API Endpoint Tests")
    print("=" * 50)
    
    tests = [
        test_model_imports,
        test_endpoint_imports,
        test_dependency_injection,
        test_middleware,
        test_services,
        test_router_creation,
        test_app_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print()
        if test():
            passed += 1
        else:
            print()
    
    print()
    print("=" * 50)
    print(f"🧪 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API endpoints are ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
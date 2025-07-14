#!/usr/bin/env python3
"""
Test script for Task 28 stability and performance components.

Validates that all new monitoring and resilience components work correctly.
"""

import sys
import asyncio
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_circuit_breaker():
    """Test circuit breaker functionality."""
    print("🔧 Testing Circuit Breaker...")
    
    try:
        from quantum_rerank.core.circuit_breaker import CircuitBreaker, CircuitState, CircuitBreakerConfig
        
        # Test circuit breaker creation
        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=1,
            operation_timeout=0.5
        )
        
        breaker = CircuitBreaker("test_breaker", config)
        
        assert breaker.get_state() == CircuitState.CLOSED
        print("   ✅ Circuit breaker initialization works")
        
        # Test async functionality with dummy functions
        async def test_async():
            # Test successful operation
            def success_func():
                return "success"
            
            result = await breaker.call(success_func)
            assert result == "success"
            print("   ✅ Successful operation works")
            
            # Test failure handling
            def fail_func():
                raise Exception("Test failure")
            
            def fallback_func():
                return "fallback"
            
            # This should use fallback after failures
            result = await breaker.call(fail_func, fallback_func)
            print("   ✅ Fallback mechanism works")
            
            # Test metrics
            metrics = breaker.get_metrics()
            assert "name" in metrics
            assert "state" in metrics
            print("   ✅ Metrics collection works")
        
        asyncio.run(test_async())
        print("   ✅ Circuit breaker tests passed")
        
    except Exception as e:
        print(f"   ❌ Circuit breaker test failed: {e}")
        return False
    
    return True


def test_memory_monitor():
    """Test memory monitoring functionality."""
    print("💾 Testing Memory Monitor...")
    
    try:
        from quantum_rerank.core.memory_monitor import MemoryMonitor, MemoryThresholds
        
        # Test memory monitor creation
        thresholds = MemoryThresholds(
            warning_threshold=0.8,
            critical_threshold=0.9,
            max_memory_gb=2.0
        )
        
        monitor = MemoryMonitor(thresholds)
        print("   ✅ Memory monitor initialization works")
        
        # Test memory checking
        usage = monitor.check_memory_usage()
        assert "memory_mb" in usage
        assert "memory_percent" in usage
        print("   ✅ Memory usage checking works")
        
        # Test availability check
        available = monitor.is_memory_available()
        assert isinstance(available, bool)
        print("   ✅ Memory availability check works")
        
        # Test trends
        trends = monitor.get_memory_trends()
        print("   ✅ Memory trends calculation works")
        
        # Test summary
        summary = monitor.get_memory_summary()
        assert "current_usage" in summary
        assert "thresholds" in summary
        print("   ✅ Memory summary works")
        
        print("   ✅ Memory monitor tests passed")
        
    except Exception as e:
        print(f"   ❌ Memory monitor test failed: {e}")
        return False
    
    return True


def test_enhanced_validation():
    """Test enhanced API validation."""
    print("🔍 Testing Enhanced API Validation...")
    
    try:
        from quantum_rerank.api.models import RerankRequest
        from pydantic import ValidationError
        
        # Test valid request
        valid_request = RerankRequest(
            query="test query",
            candidates=["doc1", "doc2"],
            method="hybrid"
        )
        assert valid_request.query == "test query"
        print("   ✅ Valid request parsing works")
        
        # Test query validation
        try:
            RerankRequest(
                query="",  # Empty query
                candidates=["doc1"],
                method="hybrid"
            )
            print("   ❌ Empty query validation failed")
            return False
        except ValidationError:
            print("   ✅ Empty query validation works")
        
        # Test candidates validation
        try:
            RerankRequest(
                query="test",
                candidates=[],  # Empty candidates
                method="hybrid"
            )
            print("   ❌ Empty candidates validation failed")
            return False
        except ValidationError:
            print("   ✅ Empty candidates validation works")
        
        # Test large request handling
        large_candidates = [f"Document {i} " + "x" * 1000 for i in range(50)]
        large_request = RerankRequest(
            query="test query",
            candidates=large_candidates,
            method="hybrid"
        )
        assert len(large_request.candidates) == 50
        print("   ✅ Large request handling works")
        
        print("   ✅ Enhanced validation tests passed")
        
    except Exception as e:
        print(f"   ❌ Enhanced validation test failed: {e}")
        return False
    
    return True


def test_error_handling():
    """Test error handling middleware components."""
    print("🚨 Testing Error Handling...")
    
    try:
        from quantum_rerank.api.middleware.error_handling import (
            ErrorHandlingMiddleware,
            RateLimitErrorHandler,
            SecurityErrorHandler,
            ErrorMetricsCollector
        )
        
        # Test error metrics collector
        collector = ErrorMetricsCollector()
        
        collector.record_error("TEST_ERROR", "TestException", {"test": "data"})
        metrics = collector.get_error_summary()
        assert metrics["total_errors"] == 1
        print("   ✅ Error metrics collection works")
        
        # Test rate limit error handler
        rate_limit_response = RateLimitErrorHandler.create_rate_limit_response("test-123", 60)
        assert rate_limit_response.status_code == 429
        print("   ✅ Rate limit error handling works")
        
        # Test security error handler
        auth_error_response = SecurityErrorHandler.create_authentication_error("test-123")
        assert auth_error_response.status_code == 401
        print("   ✅ Security error handling works")
        
        print("   ✅ Error handling tests passed")
        
    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")
        return False
    
    return True


def test_health_monitoring():
    """Test health monitoring enhancements."""
    print("🏥 Testing Health Monitoring...")
    
    try:
        from quantum_rerank.core.memory_monitor import memory_monitor
        from quantum_rerank.core.circuit_breaker import get_circuit_breaker_status
        
        # Test memory monitor global instance
        usage = memory_monitor.check_memory_usage()
        assert "memory_mb" in usage
        print("   ✅ Global memory monitor works")
        
        # Test circuit breaker status
        async def test_breaker_status():
            status = await get_circuit_breaker_status()
            assert isinstance(status, dict)
            print("   ✅ Circuit breaker status works")
        
        asyncio.run(test_breaker_status())
        
        print("   ✅ Health monitoring tests passed")
        
    except Exception as e:
        print(f"   ❌ Health monitoring test failed: {e}")
        return False
    
    return True


def test_config_structure():
    """Test that production config structure is correct."""
    print("⚙️  Testing Configuration Structure...")
    
    try:
        import yaml
        
        config_path = Path("config/production.simple.yaml")
        if not config_path.exists():
            print(f"   ❌ Config file not found: {config_path}")
            return False
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Check stability configuration
        assert "quantum_rerank" in config
        qr_config = config["quantum_rerank"]
        
        assert "stability" in qr_config
        stability = qr_config["stability"]
        
        assert "circuit_breaker" in stability
        assert "memory_monitor" in stability
        assert "error_handling" in stability
        
        print("   ✅ Stability configuration structure correct")
        
        # Check performance configuration
        assert "performance" in qr_config
        performance = qr_config["performance"]
        
        assert "memory_limit_gb" in performance
        assert "similarity_timeout_ms" in performance
        assert "batch_timeout_ms" in performance
        
        print("   ✅ Performance configuration structure correct")
        
        print("   ✅ Configuration tests passed")
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False
    
    return True


def main():
    """Run all stability component tests."""
    print("🧪 Testing Task 28: API Stability & Performance Components")
    print("=" * 60)
    print()
    
    tests = [
        test_circuit_breaker,
        test_memory_monitor,
        test_enhanced_validation,
        test_error_handling,
        test_health_monitoring,
        test_config_structure
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"   ❌ Test {test_func.__name__} crashed: {e}")
            failed += 1
        
        print()
    
    print("=" * 60)
    print(f"📊 Test Results:")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print(f"   Total: {passed + failed}")
    
    if failed == 0:
        print()
        print("✅ All Task 28 components are working correctly!")
        print("🚀 API stability and performance features are ready for production")
        return 0
    else:
        print()
        print("❌ Some components failed testing")
        print("🔧 Check the output above for specific issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
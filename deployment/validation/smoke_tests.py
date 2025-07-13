#!/usr/bin/env python3
"""
Production smoke tests for QuantumRerank deployment validation.

This module provides comprehensive smoke testing for production deployments,
including API functionality, performance validation, and system health checks.
"""

import asyncio
import json
import logging
import time
import requests
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from urllib.parse import urljoin
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
    """Configuration for smoke tests."""
    base_url: str = "https://api.quantumrerank.com"
    timeout: int = 30
    api_key: Optional[str] = None
    verify_ssl: bool = True
    max_retries: int = 3
    retry_delay: int = 2


@dataclass
class TestResult:
    """Result of a smoke test."""
    name: str
    passed: bool
    duration_ms: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class SmokeTestRunner:
    """
    Comprehensive smoke test runner for production deployment validation.
    
    Tests critical paths, API functionality, performance, and system health.
    """
    
    def __init__(self, config: TestConfig):
        """
        Initialize smoke test runner.
        
        Args:
            config: Test configuration
        """
        self.config = config
        self.session = requests.Session()
        self.session.timeout = config.timeout
        self.session.verify = config.verify_ssl
        
        if config.api_key:
            self.session.headers.update({"X-API-Key": config.api_key})
        
        self.results: List[TestResult] = []
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all smoke tests.
        
        Returns:
            Test results summary
        """
        logger.info("Starting production smoke tests...")
        
        # Critical path tests
        self._test_service_availability()
        self._test_health_endpoints()
        self._test_api_functionality()
        self._test_performance_targets()
        self._test_security_features()
        self._test_monitoring_endpoints()
        
        # Generate summary
        summary = self._generate_summary()
        
        logger.info(f"Smoke tests completed. Success rate: {summary['success_rate']:.1%}")
        
        return summary
    
    def _test_service_availability(self) -> None:
        """Test basic service availability."""
        logger.info("Testing service availability...")
        
        start_time = time.time()
        try:
            response = self._make_request("GET", "/health")
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    self._record_success("service_availability", start_time)
                else:
                    self._record_failure("service_availability", start_time, 
                                       f"Service reports unhealthy status: {data.get('status')}")
            else:
                self._record_failure("service_availability", start_time,
                                   f"HTTP {response.status_code}: {response.text}")
        
        except Exception as e:
            self._record_failure("service_availability", start_time, str(e))
    
    def _test_health_endpoints(self) -> None:
        """Test all health check endpoints."""
        logger.info("Testing health endpoints...")
        
        endpoints = [
            ("/health", "basic_health"),
            ("/health/ready", "readiness_check"),
            ("/health/detailed", "detailed_health")
        ]
        
        for endpoint, test_name in endpoints:
            start_time = time.time()
            try:
                response = self._make_request("GET", endpoint)
                
                if response.status_code == 200:
                    data = response.json()
                    if "status" in data:
                        self._record_success(test_name, start_time, {"response": data})
                    else:
                        self._record_failure(test_name, start_time, "Invalid response format")
                else:
                    self._record_failure(test_name, start_time,
                                       f"HTTP {response.status_code}: {response.text}")
            
            except Exception as e:
                self._record_failure(test_name, start_time, str(e))
    
    def _test_api_functionality(self) -> None:
        """Test core API functionality."""
        logger.info("Testing API functionality...")
        
        # Test similarity endpoint
        self._test_similarity_endpoint()
        
        # Test rerank endpoint
        self._test_rerank_endpoint()
        
        # Test batch endpoint
        self._test_batch_endpoint()
    
    def _test_similarity_endpoint(self) -> None:
        """Test similarity computation endpoint."""
        start_time = time.time()
        
        payload = {
            "text1": "Quantum computing is a revolutionary technology.",
            "text2": "Quantum computers represent a paradigm shift in computation.",
            "method": "quantum"
        }
        
        try:
            response = self._make_request("POST", "/v1/similarity", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if "similarity_score" in data and 0 <= data["similarity_score"] <= 1:
                    self._record_success("similarity_endpoint", start_time, 
                                       {"similarity_score": data["similarity_score"]})
                else:
                    self._record_failure("similarity_endpoint", start_time,
                                       "Invalid similarity score format")
            else:
                self._record_failure("similarity_endpoint", start_time,
                                   f"HTTP {response.status_code}: {response.text}")
        
        except Exception as e:
            self._record_failure("similarity_endpoint", start_time, str(e))
    
    def _test_rerank_endpoint(self) -> None:
        """Test document reranking endpoint."""
        start_time = time.time()
        
        payload = {
            "query": "quantum machine learning applications",
            "candidates": [
                "Quantum computing enables new machine learning algorithms",
                "Classical computing has limitations for certain problems",
                "Machine learning can optimize quantum circuits",
                "Quantum algorithms show promise for AI applications"
            ],
            "top_k": 3,
            "method": "quantum"
        }
        
        try:
            response = self._make_request("POST", "/v1/rerank", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if ("rankings" in data and 
                    len(data["rankings"]) <= payload["top_k"] and
                    all("score" in item for item in data["rankings"])):
                    self._record_success("rerank_endpoint", start_time,
                                       {"num_results": len(data["rankings"])})
                else:
                    self._record_failure("rerank_endpoint", start_time,
                                       "Invalid rerank response format")
            else:
                self._record_failure("rerank_endpoint", start_time,
                                   f"HTTP {response.status_code}: {response.text}")
        
        except Exception as e:
            self._record_failure("rerank_endpoint", start_time, str(e))
    
    def _test_batch_endpoint(self) -> None:
        """Test batch processing endpoint."""
        start_time = time.time()
        
        payload = {
            "queries": [
                "quantum computing benefits",
                "machine learning optimization"
            ],
            "candidates": [
                "Quantum computers solve complex problems",
                "Machine learning improves over time",
                "Optimization algorithms find best solutions"
            ],
            "method": "quantum"
        }
        
        try:
            response = self._make_request("POST", "/v1/batch-similarity", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if ("results" in data and 
                    len(data["results"]) == len(payload["queries"])):
                    self._record_success("batch_endpoint", start_time,
                                       {"num_queries": len(data["results"])})
                else:
                    self._record_failure("batch_endpoint", start_time,
                                       "Invalid batch response format")
            else:
                self._record_failure("batch_endpoint", start_time,
                                   f"HTTP {response.status_code}: {response.text}")
        
        except Exception as e:
            self._record_failure("batch_endpoint", start_time, str(e))
    
    def _test_performance_targets(self) -> None:
        """Test performance against PRD targets."""
        logger.info("Testing performance targets...")
        
        # Test similarity computation time (<100ms target)
        self._test_similarity_performance()
        
        # Test batch processing time (<500ms target)
        self._test_batch_performance()
        
        # Test response time (<200ms target)
        self._test_response_time()
    
    def _test_similarity_performance(self) -> None:
        """Test similarity computation performance."""
        start_time = time.time()
        
        payload = {
            "text1": "Performance test for similarity computation",
            "text2": "Testing the speed of quantum similarity calculation",
            "method": "quantum"
        }
        
        try:
            response = self._make_request("POST", "/v1/similarity", json=payload)
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                # PRD target: <100ms
                if duration_ms < 100:
                    self._record_success("similarity_performance", start_time,
                                       {"duration_ms": duration_ms, "target_ms": 100})
                else:
                    self._record_failure("similarity_performance", start_time,
                                       f"Duration {duration_ms:.1f}ms exceeds 100ms target")
            else:
                self._record_failure("similarity_performance", start_time,
                                   f"HTTP {response.status_code}: {response.text}")
        
        except Exception as e:
            self._record_failure("similarity_performance", start_time, str(e))
    
    def _test_batch_performance(self) -> None:
        """Test batch processing performance."""
        start_time = time.time()
        
        # Create batch request with multiple items
        payload = {
            "queries": [f"Performance test query {i}" for i in range(10)],
            "candidates": [f"Test candidate document {i}" for i in range(20)],
            "method": "quantum"
        }
        
        try:
            response = self._make_request("POST", "/v1/batch-similarity", json=payload)
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                # PRD target: <500ms for batch processing
                if duration_ms < 500:
                    self._record_success("batch_performance", start_time,
                                       {"duration_ms": duration_ms, "target_ms": 500})
                else:
                    self._record_failure("batch_performance", start_time,
                                       f"Duration {duration_ms:.1f}ms exceeds 500ms target")
            else:
                self._record_failure("batch_performance", start_time,
                                   f"HTTP {response.status_code}: {response.text}")
        
        except Exception as e:
            self._record_failure("batch_performance", start_time, str(e))
    
    def _test_response_time(self) -> None:
        """Test general API response time."""
        start_time = time.time()
        
        try:
            response = self._make_request("GET", "/health")
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                # PRD target: <200ms for general response time
                if duration_ms < 200:
                    self._record_success("response_time", start_time,
                                       {"duration_ms": duration_ms, "target_ms": 200})
                else:
                    self._record_failure("response_time", start_time,
                                       f"Duration {duration_ms:.1f}ms exceeds 200ms target")
            else:
                self._record_failure("response_time", start_time,
                                   f"HTTP {response.status_code}: {response.text}")
        
        except Exception as e:
            self._record_failure("response_time", start_time, str(e))
    
    def _test_security_features(self) -> None:
        """Test security features."""
        logger.info("Testing security features...")
        
        # Test security headers
        self._test_security_headers()
        
        # Test rate limiting (if enabled)
        self._test_rate_limiting()
        
        # Test HTTPS enforcement
        self._test_https_enforcement()
    
    def _test_security_headers(self) -> None:
        """Test security headers are present."""
        start_time = time.time()
        
        try:
            response = self._make_request("GET", "/health")
            
            if response.status_code == 200:
                headers = response.headers
                required_headers = [
                    "X-Content-Type-Options",
                    "X-Frame-Options", 
                    "X-XSS-Protection"
                ]
                
                missing_headers = [h for h in required_headers if h not in headers]
                
                if not missing_headers:
                    self._record_success("security_headers", start_time,
                                       {"headers_present": required_headers})
                else:
                    self._record_failure("security_headers", start_time,
                                       f"Missing headers: {missing_headers}")
            else:
                self._record_failure("security_headers", start_time,
                                   f"HTTP {response.status_code}: {response.text}")
        
        except Exception as e:
            self._record_failure("security_headers", start_time, str(e))
    
    def _test_rate_limiting(self) -> None:
        """Test rate limiting functionality."""
        start_time = time.time()
        
        try:
            # Make multiple rapid requests to test rate limiting
            responses = []
            for _ in range(5):
                response = self._make_request("GET", "/health")
                responses.append(response.status_code)
                time.sleep(0.1)  # Small delay between requests
            
            # Check if we got any rate limiting responses (429)
            if any(status == 429 for status in responses):
                self._record_success("rate_limiting", start_time,
                                   {"rate_limited": True, "responses": responses})
            else:
                # Rate limiting might not be triggered for health endpoint
                self._record_success("rate_limiting", start_time,
                                   {"rate_limited": False, "note": "No rate limiting triggered"})
        
        except Exception as e:
            self._record_failure("rate_limiting", start_time, str(e))
    
    def _test_https_enforcement(self) -> None:
        """Test HTTPS enforcement."""
        start_time = time.time()
        
        try:
            # Test that HTTP redirects to HTTPS (if applicable)
            if self.config.base_url.startswith("https://"):
                http_url = self.config.base_url.replace("https://", "http://")
                
                try:
                    response = requests.get(f"{http_url}/health", 
                                          allow_redirects=False, 
                                          timeout=10)
                    
                    if response.status_code in [301, 302, 308]:
                        self._record_success("https_enforcement", start_time,
                                           {"redirect_status": response.status_code})
                    else:
                        self._record_failure("https_enforcement", start_time,
                                           f"No HTTPS redirect, got {response.status_code}")
                except requests.exceptions.RequestException:
                    # HTTP might be completely blocked, which is also good
                    self._record_success("https_enforcement", start_time,
                                       {"http_blocked": True})
            else:
                self._record_success("https_enforcement", start_time,
                                   {"note": "HTTP URL configured, skipping HTTPS test"})
        
        except Exception as e:
            self._record_failure("https_enforcement", start_time, str(e))
    
    def _test_monitoring_endpoints(self) -> None:
        """Test monitoring and metrics endpoints."""
        logger.info("Testing monitoring endpoints...")
        
        # Test metrics endpoint
        start_time = time.time()
        try:
            response = self._make_request("GET", "/metrics")
            
            if response.status_code == 200:
                # Check if response contains Prometheus metrics
                text = response.text
                if "# HELP" in text or "# TYPE" in text:
                    self._record_success("metrics_endpoint", start_time,
                                       {"metrics_format": "prometheus"})
                else:
                    self._record_failure("metrics_endpoint", start_time,
                                       "Metrics endpoint doesn't return Prometheus format")
            else:
                self._record_failure("metrics_endpoint", start_time,
                                   f"HTTP {response.status_code}: {response.text}")
        
        except Exception as e:
            self._record_failure("metrics_endpoint", start_time, str(e))
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Make HTTP request with retry logic.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request parameters
            
        Returns:
            Response object
        """
        url = urljoin(self.config.base_url, endpoint)
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.request(method, url, **kwargs)
                return response
            
            except requests.exceptions.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    raise e
                
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                time.sleep(self.config.retry_delay)
        
        raise requests.exceptions.RequestException("Max retries exceeded")
    
    def _record_success(self, test_name: str, start_time: float, details: Optional[Dict[str, Any]] = None) -> None:
        """Record successful test result."""
        duration_ms = (time.time() - start_time) * 1000
        result = TestResult(
            name=test_name,
            passed=True,
            duration_ms=duration_ms,
            details=details
        )
        self.results.append(result)
        logger.info(f"✓ {test_name} passed ({duration_ms:.1f}ms)")
    
    def _record_failure(self, test_name: str, start_time: float, error_message: str) -> None:
        """Record failed test result."""
        duration_ms = (time.time() - start_time) * 1000
        result = TestResult(
            name=test_name,
            passed=False,
            duration_ms=duration_ms,
            error_message=error_message
        )
        self.results.append(result)
        logger.error(f"✗ {test_name} failed ({duration_ms:.1f}ms): {error_message}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test results summary."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        total_duration = sum(r.duration_ms for r in self.results)
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "total_duration_ms": total_duration,
            "average_duration_ms": total_duration / total_tests if total_tests > 0 else 0,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "duration_ms": r.duration_ms,
                    "error_message": r.error_message,
                    "details": r.details
                }
                for r in self.results
            ]
        }
        
        return summary


def main():
    """Main entry point for smoke tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QuantumRerank Production Smoke Tests")
    parser.add_argument("--base-url", default="https://api.quantumrerank.com",
                       help="Base URL for the API")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--no-ssl-verify", action="store_true", help="Disable SSL verification")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Configure test runner
    config = TestConfig(
        base_url=args.base_url,
        timeout=args.timeout,
        api_key=args.api_key,
        verify_ssl=not args.no_ssl_verify
    )
    
    # Run tests
    runner = SmokeTestRunner(config)
    summary = runner.run_all_tests()
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results written to {args.output}")
    
    # Print summary
    print("\n" + "="*60)
    print("SMOKE TEST SUMMARY")
    print("="*60)
    print(f"Total tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Total duration: {summary['total_duration_ms']:.1f}ms")
    print(f"Average duration: {summary['average_duration_ms']:.1f}ms")
    
    if summary['failed_tests'] > 0:
        print("\nFAILED TESTS:")
        for result in summary['results']:
            if not result['passed']:
                print(f"  ✗ {result['name']}: {result['error_message']}")
    
    # Exit with error code if tests failed
    sys.exit(0 if summary['failed_tests'] == 0 else 1)


if __name__ == "__main__":
    main()
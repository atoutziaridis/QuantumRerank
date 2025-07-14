#!/usr/bin/env python3
"""
Load testing for QuantumRerank API.

This script performs comprehensive load testing to validate performance
targets and stability under production load patterns.
"""

import asyncio
import aiohttp
import time
import statistics
import json
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    base_url: str = "http://localhost:8000"
    api_key: str = "demo-api-key"
    concurrent_users: int = 10
    requests_per_user: int = 10
    request_timeout: float = 30.0
    ramp_up_seconds: int = 0
    test_duration_seconds: Optional[int] = None
    

@dataclass
class LoadTestResult:
    """Results from a load test run."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time_seconds: float
    requests_per_second: float
    avg_response_time_ms: float
    median_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    max_response_time_ms: float
    min_response_time_ms: float
    error_rate_percent: float
    errors: List[Dict[str, Any]]


class LoadTester:
    """
    Comprehensive load tester for QuantumRerank API.
    
    Tests performance against PRD requirements:
    - <100ms per similarity computation
    - <500ms for batch reranking (50-100 documents)
    - Stable under 100+ requests/minute
    """
    
    def __init__(self, config: LoadTestConfig):
        """
        Initialize load tester.
        
        Args:
            config: Load test configuration
        """
        self.config = config
        self.session_timeout = aiohttp.ClientTimeout(total=config.request_timeout)
        
        # Test scenarios
        self.test_scenarios = {
            "single_similarity": self._create_single_similarity_request,
            "small_batch": self._create_small_batch_request,
            "large_batch": self._create_large_batch_request,
            "mixed_methods": self._create_mixed_method_request
        }
    
    def _create_single_similarity_request(self) -> Dict[str, Any]:
        """Create a single similarity request payload."""
        return {
            "query": "What is machine learning?",
            "candidates": [
                "Machine learning is a subset of artificial intelligence",
                "Python is a programming language",
                "Deep learning uses neural networks"
            ],
            "method": "hybrid",
            "top_k": 3
        }
    
    def _create_small_batch_request(self) -> Dict[str, Any]:
        """Create a small batch request (PRD: 50-100 docs)."""
        candidates = [
            f"Document {i}: This is about machine learning and AI applications."
            for i in range(25)
        ]
        
        return {
            "query": "artificial intelligence and machine learning applications",
            "candidates": candidates,
            "method": "hybrid",
            "top_k": 10
        }
    
    def _create_large_batch_request(self) -> Dict[str, Any]:
        """Create a large batch request (PRD: 50-100 docs)."""
        candidates = [
            f"Document {i}: This discusses quantum computing, machine learning, "
            f"artificial intelligence, and their practical applications in various domains."
            for i in range(75)
        ]
        
        return {
            "query": "quantum computing applications in machine learning",
            "candidates": candidates,
            "method": "hybrid",
            "top_k": 20
        }
    
    def _create_mixed_method_request(self) -> Dict[str, Any]:
        """Create request with random method selection."""
        methods = ["classical", "quantum", "hybrid"]
        import random
        
        return {
            "query": "semantic similarity and text understanding",
            "candidates": [
                "Natural language processing techniques",
                "Machine learning algorithms",
                "Quantum computing applications",
                "Classical information retrieval methods"
            ],
            "method": random.choice(methods),
            "top_k": 4
        }
    
    async def single_request(
        self, 
        session: aiohttp.ClientSession, 
        scenario: str = "single_similarity"
    ) -> Dict[str, Any]:
        """
        Make a single request and return timing/result information.
        
        Args:
            session: HTTP client session
            scenario: Test scenario to use
            
        Returns:
            Request result with timing information
        """
        start_time = time.perf_counter()
        
        # Get request payload for scenario
        payload = self.test_scenarios[scenario]()
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "X-Request-ID": f"load-test-{int(time.time() * 1000000)}"
        }
        
        try:
            async with session.post(
                f"{self.config.base_url}/v1/rerank",
                json=payload,
                headers=headers
            ) as response:
                response_time = time.perf_counter() - start_time
                
                # Get response data
                if response.status == 200:
                    response_data = await response.json()
                    num_results = len(response_data.get("results", []))
                    processing_time = response_data.get("computation_time_ms", 0)
                    
                    return {
                        "success": True,
                        "response_time_ms": response_time * 1000,
                        "status_code": response.status,
                        "scenario": scenario,
                        "num_results": num_results,
                        "api_processing_time_ms": processing_time,
                        "num_candidates": len(payload["candidates"]),
                        "method": payload["method"]
                    }
                else:
                    error_data = await response.text()
                    return {
                        "success": False,
                        "response_time_ms": response_time * 1000,
                        "status_code": response.status,
                        "scenario": scenario,
                        "error": error_data,
                        "method": payload["method"]
                    }
                    
        except Exception as e:
            response_time = time.perf_counter() - start_time
            return {
                "success": False,
                "response_time_ms": response_time * 1000,
                "status_code": 0,
                "scenario": scenario,
                "error": str(e),
                "method": payload.get("method", "unknown")
            }
    
    async def run_concurrent_test(
        self, 
        scenario: str = "single_similarity",
        progress_callback: Optional[callable] = None
    ) -> LoadTestResult:
        """
        Run concurrent load test with specified scenario.
        
        Args:
            scenario: Test scenario to use
            progress_callback: Optional callback for progress updates
            
        Returns:
            Load test results
        """
        print(f"Running load test: {self.config.concurrent_users} users, "
              f"{self.config.requests_per_user} requests each")
        print(f"Scenario: {scenario}")
        print(f"Target: {self.config.base_url}")
        
        # Create connector with proper settings
        connector = aiohttp.TCPConnector(
            limit=self.config.concurrent_users * 2,
            limit_per_host=self.config.concurrent_users * 2
        )
        
        async with aiohttp.ClientSession(
            timeout=self.session_timeout,
            connector=connector
        ) as session:
            
            # Create tasks for concurrent users
            tasks = []
            total_requests = self.config.concurrent_users * self.config.requests_per_user
            
            for user_id in range(self.config.concurrent_users):
                for req_id in range(self.config.requests_per_user):
                    # Add ramp-up delay
                    if self.config.ramp_up_seconds > 0:
                        delay = (user_id * self.config.ramp_up_seconds) / self.config.concurrent_users
                        tasks.append(self._delayed_request(session, scenario, delay))
                    else:
                        tasks.append(self.single_request(session, scenario))
            
            # Execute all requests
            start_time = time.perf_counter()
            print(f"Starting {total_requests} requests...")
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_time = time.perf_counter() - start_time
            
            # Process results
            return self._analyze_results(results, total_time, scenario)
    
    async def _delayed_request(self, session, scenario: str, delay: float):
        """Execute request with initial delay for ramp-up."""
        await asyncio.sleep(delay)
        return await self.single_request(session, scenario)
    
    def _analyze_results(self, results: List, total_time: float, scenario: str) -> LoadTestResult:
        """
        Analyze load test results and calculate statistics.
        
        Args:
            results: List of request results
            total_time: Total test execution time
            scenario: Test scenario name
            
        Returns:
            Analyzed load test results
        """
        # Separate successful and failed requests
        successful_results = []
        failed_results = []
        errors = []
        
        for result in results:
            if isinstance(result, Exception):
                errors.append({
                    "type": "exception",
                    "error": str(result),
                    "timestamp": time.time()
                })
                failed_results.append(result)
            elif isinstance(result, dict):
                if result.get("success", False):
                    successful_results.append(result)
                else:
                    failed_results.append(result)
                    errors.append({
                        "type": "api_error",
                        "status_code": result.get("status_code", 0),
                        "error": result.get("error", "Unknown error"),
                        "scenario": result.get("scenario", scenario),
                        "timestamp": time.time()
                    })
        
        # Calculate timing statistics
        response_times = [r["response_time_ms"] for r in successful_results]
        
        if response_times:
            # Sort for percentile calculations
            sorted_times = sorted(response_times)
            n = len(sorted_times)
            
            stats = LoadTestResult(
                total_requests=len(results),
                successful_requests=len(successful_results),
                failed_requests=len(failed_results),
                total_time_seconds=total_time,
                requests_per_second=len(successful_results) / total_time,
                avg_response_time_ms=statistics.mean(response_times),
                median_response_time_ms=statistics.median(response_times),
                p95_response_time_ms=sorted_times[int(n * 0.95)] if n > 0 else 0,
                p99_response_time_ms=sorted_times[int(n * 0.99)] if n > 0 else 0,
                max_response_time_ms=max(response_times),
                min_response_time_ms=min(response_times),
                error_rate_percent=(len(failed_results) / len(results)) * 100,
                errors=errors
            )
        else:
            # All requests failed
            stats = LoadTestResult(
                total_requests=len(results),
                successful_requests=0,
                failed_requests=len(failed_results),
                total_time_seconds=total_time,
                requests_per_second=0,
                avg_response_time_ms=0,
                median_response_time_ms=0,
                p95_response_time_ms=0,
                p99_response_time_ms=0,
                max_response_time_ms=0,
                min_response_time_ms=0,
                error_rate_percent=100.0,
                errors=errors
            )
        
        return stats
    
    def print_results(self, results: LoadTestResult, scenario: str):
        """Print formatted test results."""
        print(f"\n{'='*60}")
        print(f"Load Test Results - {scenario}")
        print(f"{'='*60}")
        
        print(f"📊 Overview:")
        print(f"   Total requests: {results.total_requests}")
        print(f"   Successful: {results.successful_requests}")
        print(f"   Failed: {results.failed_requests}")
        print(f"   Error rate: {results.error_rate_percent:.1f}%")
        
        print(f"\n⚡ Performance:")
        print(f"   Requests/sec: {results.requests_per_second:.1f}")
        print(f"   Avg response: {results.avg_response_time_ms:.1f}ms")
        print(f"   Median response: {results.median_response_time_ms:.1f}ms")
        print(f"   95th percentile: {results.p95_response_time_ms:.1f}ms")
        print(f"   99th percentile: {results.p99_response_time_ms:.1f}ms")
        print(f"   Max response: {results.max_response_time_ms:.1f}ms")
        
        print(f"\n🎯 PRD Compliance:")
        prd_similarity_ok = results.avg_response_time_ms < 100
        prd_batch_ok = results.p95_response_time_ms < 500
        prd_throughput_ok = results.requests_per_second >= 1.67  # 100 req/min
        
        print(f"   Similarity <100ms: {'✅' if prd_similarity_ok else '❌'} "
              f"({results.avg_response_time_ms:.1f}ms)")
        print(f"   P95 <500ms: {'✅' if prd_batch_ok else '❌'} "
              f"({results.p95_response_time_ms:.1f}ms)")
        print(f"   Throughput >100/min: {'✅' if prd_throughput_ok else '❌'} "
              f"({results.requests_per_second * 60:.1f}/min)")
        
        if results.errors:
            print(f"\n❌ Errors ({len(results.errors)} total):")
            error_summary = {}
            for error in results.errors[:10]:  # Show first 10
                error_type = error.get("type", "unknown")
                error_summary[error_type] = error_summary.get(error_type, 0) + 1
            
            for error_type, count in error_summary.items():
                print(f"   {error_type}: {count}")
        
        print(f"\n")
    
    async def run_comprehensive_test(self):
        """Run comprehensive test across all scenarios."""
        print("🚀 Starting Comprehensive Load Test")
        print(f"Base URL: {self.config.base_url}")
        print(f"API Key: {self.config.api_key[:8]}...")
        
        # Test scenarios in order of complexity
        scenarios = [
            "single_similarity",
            "small_batch", 
            "large_batch",
            "mixed_methods"
        ]
        
        all_results = {}
        
        for scenario in scenarios:
            print(f"\n🧪 Running {scenario} test...")
            results = await self.run_concurrent_test(scenario)
            all_results[scenario] = results
            self.print_results(results, scenario)
            
            # Brief pause between tests
            await asyncio.sleep(2)
        
        # Overall summary
        self._print_overall_summary(all_results)
        
        return all_results
    
    def _print_overall_summary(self, all_results: Dict[str, LoadTestResult]):
        """Print overall test summary."""
        print(f"\n{'='*60}")
        print("📋 Overall Test Summary")
        print(f"{'='*60}")
        
        total_requests = sum(r.total_requests for r in all_results.values())
        total_successful = sum(r.successful_requests for r in all_results.values())
        total_failed = sum(r.failed_requests for r in all_results.values())
        
        avg_throughput = statistics.mean([r.requests_per_second for r in all_results.values()])
        avg_response_time = statistics.mean([r.avg_response_time_ms for r in all_results.values()])
        max_p95 = max([r.p95_response_time_ms for r in all_results.values()])
        
        print(f"Total requests: {total_requests}")
        print(f"Success rate: {(total_successful/total_requests)*100:.1f}%")
        print(f"Average throughput: {avg_throughput:.1f} req/s")
        print(f"Average response time: {avg_response_time:.1f}ms")
        print(f"Worst P95 response time: {max_p95:.1f}ms")
        
        # Overall PRD compliance
        overall_compliance = (
            avg_response_time < 100 and 
            max_p95 < 500 and 
            avg_throughput >= 1.67
        )
        
        print(f"\n🎯 Overall PRD Compliance: {'✅ PASS' if overall_compliance else '❌ FAIL'}")


async def main():
    """Main load testing function."""
    parser = argparse.ArgumentParser(description="QuantumRerank Load Testing")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL")
    parser.add_argument("--api-key", default="demo-api-key", help="API key")
    parser.add_argument("--users", type=int, default=10, help="Concurrent users")
    parser.add_argument("--requests", type=int, default=10, help="Requests per user")
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout")
    parser.add_argument("--scenario", choices=["single_similarity", "small_batch", "large_batch", "mixed_methods", "all"], 
                       default="all", help="Test scenario")
    parser.add_argument("--ramp-up", type=int, default=0, help="Ramp-up time in seconds")
    
    args = parser.parse_args()
    
    config = LoadTestConfig(
        base_url=args.url,
        api_key=args.api_key,
        concurrent_users=args.users,
        requests_per_user=args.requests,
        request_timeout=args.timeout,
        ramp_up_seconds=args.ramp_up
    )
    
    tester = LoadTester(config)
    
    if args.scenario == "all":
        await tester.run_comprehensive_test()
    else:
        results = await tester.run_concurrent_test(args.scenario)
        tester.print_results(results, args.scenario)


if __name__ == "__main__":
    asyncio.run(main())
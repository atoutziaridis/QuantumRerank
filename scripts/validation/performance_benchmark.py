#!/usr/bin/env python3
"""
Comprehensive performance benchmarking for QuantumRerank production validation
Tests all performance requirements from the PRD
"""

import requests
import time
import statistics
import concurrent.futures
import threading
import json
import os
import sys
from typing import List, Dict, Tuple, Optional

class PerformanceBenchmark:
    """Comprehensive performance testing suite"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.test_documents = [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms",
            "Deep learning uses neural networks with multiple layers to process data",
            "Natural language processing enables computers to understand human language",
            "Computer vision allows machines to interpret and analyze visual information",
            "Reinforcement learning trains agents through interaction with environments"
        ]
    
    def single_request_benchmark(self, num_requests: int = 100) -> Dict:
        """Benchmark single request performance"""
        print(f"📊 Running single request benchmark ({num_requests} requests)...")
        
        response_times = []
        errors = 0
        methods_tested = ["classical", "quantum", "hybrid"]
        
        for i in range(num_requests):
            method = methods_tested[i % len(methods_tested)]
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{self.base_url}/v1/rerank",
                    headers=self.headers,
                    json={
                        "query": f"test query {i}",
                        "candidates": self.test_documents,
                        "method": method,
                        "top_k": 3
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    response_time = (time.time() - start_time) * 1000  # ms
                    response_times.append(response_time)
                    
                    # Validate response structure
                    result = response.json()
                    if "results" not in result or len(result["results"]) == 0:
                        errors += 1
                else:
                    errors += 1
                    
            except Exception as e:
                errors += 1
                print(f"  Request {i} failed: {e}")
        
        if response_times:
            sorted_times = sorted(response_times)
            return {
                "total_requests": num_requests,
                "successful_requests": len(response_times),
                "error_count": errors,
                "success_rate": len(response_times) / num_requests,
                "avg_response_time": statistics.mean(response_times),
                "median_response_time": statistics.median(response_times),
                "p95_response_time": sorted_times[int(len(sorted_times) * 0.95)],
                "p99_response_time": sorted_times[int(len(sorted_times) * 0.99)],
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0
            }
        else:
            return {"error": "All requests failed", "error_count": errors}
    
    def concurrent_load_test(self, concurrent_users: int = 20, requests_per_user: int = 10) -> Dict:
        """Test performance under concurrent load"""
        print(f"📊 Running concurrent load test ({concurrent_users} users, {requests_per_user} requests each)...")
        
        def user_simulation(user_id: int):
            user_response_times = []
            user_errors = 0
            
            for request_id in range(requests_per_user):
                start_time = time.time()
                
                try:
                    response = requests.post(
                        f"{self.base_url}/v1/rerank",
                        headers=self.headers,
                        json={
                            "query": f"concurrent test query from user {user_id} request {request_id}",
                            "candidates": self.test_documents,
                            "method": "hybrid",
                            "top_k": 3
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        response_time = (time.time() - start_time) * 1000
                        user_response_times.append(response_time)
                    else:
                        user_errors += 1
                        
                except Exception:
                    user_errors += 1
            
            return user_response_times, user_errors
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(user_simulation, i) for i in range(concurrent_users)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Aggregate results
        all_response_times = []
        total_errors = 0
        
        for response_times, errors in results:
            all_response_times.extend(response_times)
            total_errors += errors
        
        total_requests = concurrent_users * requests_per_user
        successful_requests = len(all_response_times)
        
        if all_response_times:
            sorted_times = sorted(all_response_times)
            return {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "total_errors": total_errors,
                "success_rate": successful_requests / total_requests,
                "total_time": total_time,
                "requests_per_second": successful_requests / total_time,
                "avg_response_time": statistics.mean(all_response_times),
                "p95_response_time": sorted_times[int(len(sorted_times) * 0.95)],
                "p99_response_time": sorted_times[int(len(sorted_times) * 0.99)],
                "max_response_time": max(all_response_times),
                "concurrent_users": concurrent_users
            }
        else:
            return {"error": "All requests failed", "total_errors": total_errors}
    
    def method_comparison_test(self) -> Dict:
        """Compare performance across different reranking methods"""
        print("📊 Running method comparison test...")
        
        methods = ["classical", "quantum", "hybrid"]
        results = {}
        
        for method in methods:
            print(f"  Testing {method} method...")
            method_times = []
            
            for i in range(20):  # 20 requests per method
                start_time = time.time()
                
                try:
                    response = requests.post(
                        f"{self.base_url}/v1/rerank",
                        headers=self.headers,
                        json={
                            "query": f"method comparison test {i}",
                            "candidates": self.test_documents,
                            "method": method,
                            "top_k": 5
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        response_time = (time.time() - start_time) * 1000
                        method_times.append(response_time)
                        
                except Exception as e:
                    print(f"    Request failed: {e}")
            
            if method_times:
                results[method] = {
                    "avg_time": statistics.mean(method_times),
                    "median_time": statistics.median(method_times),
                    "max_time": max(method_times),
                    "min_time": min(method_times),
                    "success_count": len(method_times)
                }
            else:
                results[method] = {"error": "All requests failed"}
        
        return results
    
    def scalability_test(self) -> Dict:
        """Test how performance scales with number of documents"""
        print("📊 Running scalability test...")
        
        # Test with different document set sizes
        doc_counts = [5, 10, 25, 50, 100]
        results = {}
        
        for count in doc_counts:
            print(f"  Testing with {count} documents...")
            
            # Create document set of specified size
            test_docs = [f"Document {i}: This is test content for scalability testing" for i in range(count)]
            
            times = []
            for i in range(10):  # 10 requests per size
                start_time = time.time()
                
                try:
                    response = requests.post(
                        f"{self.base_url}/v1/rerank",
                        headers=self.headers,
                        json={
                            "query": "scalability test query",
                            "candidates": test_docs,
                            "method": "classical",  # Use fastest method
                            "top_k": min(10, count)
                        },
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        response_time = (time.time() - start_time) * 1000
                        times.append(response_time)
                        
                except Exception as e:
                    print(f"    Request failed: {e}")
            
            if times:
                results[count] = {
                    "avg_time": statistics.mean(times),
                    "max_time": max(times),
                    "success_count": len(times)
                }
            else:
                results[count] = {"error": "All requests failed"}
        
        return results
    
    def stress_test(self, duration_seconds: int = 60) -> Dict:
        """Run sustained load for specified duration"""
        print(f"📊 Running stress test for {duration_seconds} seconds...")
        
        results = {
            "duration": duration_seconds,
            "response_times": [],
            "errors": 0,
            "requests_completed": 0
        }
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        def worker():
            while time.time() < end_time:
                request_start = time.time()
                
                try:
                    response = requests.post(
                        f"{self.base_url}/v1/rerank",
                        headers=self.headers,
                        json={
                            "query": "stress test query",
                            "candidates": self.test_documents,
                            "method": "classical",
                            "top_k": 3
                        },
                        timeout=30
                    )
                    
                    request_time = (time.time() - request_start) * 1000
                    
                    if response.status_code == 200:
                        results["response_times"].append(request_time)
                        results["requests_completed"] += 1
                    else:
                        results["errors"] += 1
                        
                except Exception:
                    results["errors"] += 1
                
                # Small delay to avoid overwhelming
                time.sleep(0.1)
        
        # Run 5 concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker) for _ in range(5)]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        
        actual_duration = time.time() - start_time
        
        if results["response_times"]:
            results.update({
                "actual_duration": actual_duration,
                "requests_per_second": results["requests_completed"] / actual_duration,
                "avg_response_time": statistics.mean(results["response_times"]),
                "max_response_time": max(results["response_times"]),
                "error_rate": results["errors"] / (results["requests_completed"] + results["errors"])
            })
        
        return results

def validate_performance_requirements(results: Dict) -> bool:
    """Validate results against PRD requirements"""
    
    print("\n🎯 Validating against PRD requirements...")
    print("-" * 50)
    
    passed = True
    
    # Requirement 1: <100ms per similarity computation (single requests)
    if "single_request" in results:
        avg_time = results["single_request"].get("avg_response_time", float('inf'))
        if avg_time < 100:
            print(f"✅ Single request avg time: {avg_time:.1f}ms (<100ms target)")
        else:
            print(f"❌ Single request avg time: {avg_time:.1f}ms (exceeds 100ms target)")
            passed = False
    
    # Requirement 2: <500ms for batch reranking (concurrent load)
    if "concurrent_load" in results:
        p95_time = results["concurrent_load"].get("p95_response_time", float('inf'))
        if p95_time < 500:
            print(f"✅ Concurrent p95 time: {p95_time:.1f}ms (<500ms target)")
        else:
            print(f"❌ Concurrent p95 time: {p95_time:.1f}ms (exceeds 500ms target)")
            passed = False
    
    # Requirement 3: Success rate >95%
    if "concurrent_load" in results:
        success_rate = results["concurrent_load"].get("success_rate", 0)
        if success_rate > 0.95:
            print(f"✅ Success rate: {success_rate:.1%} (>95% target)")
        else:
            print(f"❌ Success rate: {success_rate:.1%} (below 95% target)")
            passed = False
    
    # Requirement 4: Handle concurrent users
    if "concurrent_load" in results:
        rps = results["concurrent_load"].get("requests_per_second", 0)
        if rps > 10:
            print(f"✅ Requests per second: {rps:.1f} (>10 target)")
        else:
            print(f"❌ Requests per second: {rps:.1f} (below 10 target)")
            passed = False
    
    return passed

def main():
    """Run comprehensive performance benchmark"""
    
    print("🚀 QuantumRerank Comprehensive Performance Benchmark")
    print("=" * 70)
    
    # Configuration
    base_url = os.getenv("BASE_URL", "http://localhost:8000")
    api_key = os.getenv("QUANTUM_RERANK_API_KEY")
    
    if not api_key:
        print("❌ Error: QUANTUM_RERANK_API_KEY environment variable is required")
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
    
    print("\n" + "=" * 70)
    
    # Initialize benchmarker
    benchmarker = PerformanceBenchmark(base_url, api_key)
    all_results = {}
    
    try:
        # Test 1: Single request performance
        all_results["single_request"] = benchmarker.single_request_benchmark(100)
        
        if "error" not in all_results["single_request"]:
            print(f"📊 Single Request Results:")
            print(f"  Average: {all_results['single_request']['avg_response_time']:.1f}ms")
            print(f"  P95: {all_results['single_request']['p95_response_time']:.1f}ms")
            print(f"  P99: {all_results['single_request']['p99_response_time']:.1f}ms")
            print(f"  Success rate: {all_results['single_request']['success_rate']:.1%}")
        else:
            print("❌ Single request test failed")
        
        # Test 2: Concurrent load test
        all_results["concurrent_load"] = benchmarker.concurrent_load_test(20, 10)
        
        if "error" not in all_results["concurrent_load"]:
            print(f"\n📊 Concurrent Load Results:")
            print(f"  RPS: {all_results['concurrent_load']['requests_per_second']:.1f}")
            print(f"  Average: {all_results['concurrent_load']['avg_response_time']:.1f}ms")
            print(f"  P95: {all_results['concurrent_load']['p95_response_time']:.1f}ms")
            print(f"  Success rate: {all_results['concurrent_load']['success_rate']:.1%}")
        else:
            print("❌ Concurrent load test failed")
        
        # Test 3: Method comparison
        all_results["method_comparison"] = benchmarker.method_comparison_test()
        
        print(f"\n📊 Method Comparison Results:")
        for method, data in all_results["method_comparison"].items():
            if "error" not in data:
                print(f"  {method}: {data['avg_time']:.1f}ms avg")
            else:
                print(f"  {method}: failed")
        
        # Test 4: Scalability test
        all_results["scalability"] = benchmarker.scalability_test()
        
        print(f"\n📊 Scalability Results:")
        for doc_count, data in all_results["scalability"].items():
            if "error" not in data:
                print(f"  {doc_count} docs: {data['avg_time']:.1f}ms avg")
            else:
                print(f"  {doc_count} docs: failed")
        
        # Test 5: Stress test
        all_results["stress_test"] = benchmarker.stress_test(30)
        
        if "error_rate" in all_results["stress_test"]:
            print(f"\n📊 Stress Test Results (30s):")
            print(f"  RPS: {all_results['stress_test']['requests_per_second']:.1f}")
            print(f"  Average: {all_results['stress_test']['avg_response_time']:.1f}ms")
            print(f"  Error rate: {all_results['stress_test']['error_rate']:.1%}")
        
        # Validate against requirements
        requirements_met = validate_performance_requirements(all_results)
        
        print("\n" + "=" * 70)
        if requirements_met:
            print("🎉 ALL PERFORMANCE REQUIREMENTS MET!")
            print("✅ QuantumRerank is ready for production deployment")
            sys.exit(0)
        else:
            print("❌ Some performance requirements not met")
            print("🔧 Review results and optimize before production deployment")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Benchmark failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
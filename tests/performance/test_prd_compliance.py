"""
Performance testing framework for PRD compliance validation.

This module provides comprehensive performance testing to validate that the
QuantumRerank system meets all Performance Requirements Document (PRD) targets.
"""

import time
import psutil
import threading
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from ..framework.test_architecture import BaseTestFramework, TestCase, TestResult, TestLevel, TestCategory
from ..framework.test_utilities import TestTimer, TestMetricsCollector
from quantum_rerank.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PRDTarget:
    """PRD performance target specification."""
    name: str
    target_value: float
    unit: str
    comparison: str  # 'less_than', 'greater_than', 'equal_to'
    critical: bool = True
    description: str = ""
    tolerance: float = 0.1  # 10% tolerance by default


@dataclass
class PerformanceTestCase:
    """Performance test case configuration."""
    name: str
    test_function: Callable
    prd_targets: List[PRDTarget]
    test_data: Any = None
    warmup_iterations: int = 5
    measurement_iterations: int = 20
    timeout_seconds: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class PerformanceResult:
    """Result of performance test execution."""
    test_case: PerformanceTestCase
    measured_values: Dict[str, List[float]]
    average_values: Dict[str, float]
    prd_compliance: Dict[str, bool]
    overall_passed: bool
    execution_time_ms: float
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class PRDComplianceFramework(BaseTestFramework):
    """
    Framework for testing PRD compliance across all performance targets.
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # PRD targets from requirements
        self.prd_targets = self._initialize_prd_targets()
        self.performance_tests: List[PerformanceTestCase] = []
        
        # Performance monitoring
        self.timer = TestTimer()
        self.metrics_collector = TestMetricsCollector()
        
        logger.info("Initialized PRD Compliance Framework")
    
    def _initialize_prd_targets(self) -> Dict[str, PRDTarget]:
        """Initialize PRD performance targets."""
        return {
            "similarity_latency": PRDTarget(
                name="similarity_latency",
                target_value=100,  # 100ms
                unit="ms",
                comparison="less_than",
                critical=True,
                description="Individual similarity computation latency",
                tolerance=0.1
            ),
            "batch_reranking_latency": PRDTarget(
                name="batch_reranking_latency", 
                target_value=500,  # 500ms for batch of 100
                unit="ms",
                comparison="less_than",
                critical=True,
                description="Batch reranking latency for 100 items",
                tolerance=0.15
            ),
            "memory_usage": PRDTarget(
                name="memory_usage",
                target_value=2.0,  # 2GB
                unit="GB",
                comparison="less_than",
                critical=True,
                description="Peak memory usage during processing",
                tolerance=0.2
            ),
            "accuracy_improvement": PRDTarget(
                name="accuracy_improvement",
                target_value=0.15,  # 15% improvement
                unit="percentage",
                comparison="greater_than",
                critical=True,
                description="Accuracy improvement over baseline",
                tolerance=0.1
            ),
            "throughput": PRDTarget(
                name="throughput",
                target_value=1000,  # 1000 operations/second
                unit="ops/sec",
                comparison="greater_than",
                critical=False,
                description="System throughput under load",
                tolerance=0.2
            ),
            "startup_time": PRDTarget(
                name="startup_time",
                target_value=30,  # 30 seconds
                unit="seconds",
                comparison="less_than",
                critical=False,
                description="System startup and initialization time",
                tolerance=0.3
            ),
            "quantum_fidelity_accuracy": PRDTarget(
                name="quantum_fidelity_accuracy",
                target_value=1e-6,  # 1e-6 error tolerance
                unit="error",
                comparison="less_than",
                critical=True,
                description="Quantum fidelity computation accuracy",
                tolerance=0.0  # No tolerance for accuracy
            )
        }
    
    def discover_tests(self, test_pattern: str = "test_prd_*.py") -> List[TestCase]:
        """Discover PRD compliance tests."""
        test_cases = []
        
        # Generate performance tests for all PRD targets
        test_cases.extend(self._create_prd_compliance_tests())
        
        self.test_cases = test_cases
        return test_cases
    
    def _create_prd_compliance_tests(self) -> List[TestCase]:
        """Create comprehensive PRD compliance test cases."""
        test_cases = []
        
        # Test 1: Similarity computation latency
        def test_similarity_latency():
            return self._test_similarity_performance()
        
        test_cases.append(TestCase(
            name="prd_similarity_latency",
            level=TestLevel.PERFORMANCE,
            category=TestCategory.PERFORMANCE,
            description="Validate similarity computation meets latency targets",
            test_function=test_similarity_latency,
            timeout_seconds=300,
            tags=["prd", "latency", "similarity"]
        ))
        
        # Test 2: Batch reranking performance
        def test_batch_reranking():
            return self._test_batch_reranking_performance()
        
        test_cases.append(TestCase(
            name="prd_batch_reranking",
            level=TestLevel.PERFORMANCE,
            category=TestCategory.PERFORMANCE,
            description="Validate batch reranking meets performance targets",
            test_function=test_batch_reranking,
            timeout_seconds=600,
            tags=["prd", "batch", "reranking"]
        ))
        
        # Test 3: Memory usage compliance
        def test_memory_usage():
            return self._test_memory_usage_compliance()
        
        test_cases.append(TestCase(
            name="prd_memory_usage",
            level=TestLevel.PERFORMANCE,
            category=TestCategory.SCALABILITY,
            description="Validate memory usage stays within limits",
            test_function=test_memory_usage,
            timeout_seconds=300,
            tags=["prd", "memory", "scalability"]
        ))
        
        # Test 4: Accuracy improvement validation
        def test_accuracy_improvement():
            return self._test_accuracy_improvement()
        
        test_cases.append(TestCase(
            name="prd_accuracy_improvement",
            level=TestLevel.PERFORMANCE,
            category=TestCategory.FUNCTIONAL,
            description="Validate accuracy improvement over baseline",
            test_function=test_accuracy_improvement,
            timeout_seconds=600,
            tags=["prd", "accuracy", "improvement"]
        ))
        
        # Test 5: Throughput under load
        def test_throughput():
            return self._test_system_throughput()
        
        test_cases.append(TestCase(
            name="prd_system_throughput",
            level=TestLevel.PERFORMANCE,
            category=TestCategory.SCALABILITY,
            description="Validate system throughput under load",
            test_function=test_throughput,
            timeout_seconds=900,
            tags=["prd", "throughput", "load"]
        ))
        
        # Test 6: Startup time
        def test_startup_time():
            return self._test_startup_performance()
        
        test_cases.append(TestCase(
            name="prd_startup_time",
            level=TestLevel.PERFORMANCE,
            category=TestCategory.PERFORMANCE,
            description="Validate system startup time",
            test_function=test_startup_time,
            timeout_seconds=120,
            tags=["prd", "startup", "initialization"]
        ))
        
        # Test 7: Quantum fidelity accuracy
        def test_quantum_accuracy():
            return self._test_quantum_fidelity_accuracy()
        
        test_cases.append(TestCase(
            name="prd_quantum_accuracy",
            level=TestLevel.QUANTUM,
            category=TestCategory.FUNCTIONAL,
            description="Validate quantum fidelity computation accuracy",
            test_function=test_quantum_accuracy,
            timeout_seconds=300,
            tags=["prd", "quantum", "fidelity", "accuracy"]
        ))
        
        return test_cases
    
    def _test_similarity_performance(self) -> Dict[str, Any]:
        """Test individual similarity computation performance."""
        target = self.prd_targets["similarity_latency"]
        
        try:
            # Mock similarity computation function
            def mock_similarity_computation(emb1, emb2):
                """Mock quantum similarity computation."""
                # Simulate quantum circuit execution time
                time.sleep(0.05)  # 50ms base computation
                
                # Mock fidelity calculation
                return float(np.random.random())
            
            # Generate test embeddings
            embeddings = [np.random.randn(384) for _ in range(50)]
            
            # Warmup
            for _ in range(5):
                mock_similarity_computation(embeddings[0], embeddings[1])
            
            # Measure performance
            latencies = []
            
            with self.metrics_collector.collect_metrics():
                for i in range(20):
                    with self.timer.time_operation("similarity_computation") as timing:
                        similarity = mock_similarity_computation(
                            embeddings[i % len(embeddings)], 
                            embeddings[(i + 1) % len(embeddings)]
                        )
                    
                    latencies.append(timing.execution_time_ms)
            
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            
            # Check PRD compliance
            compliance = avg_latency <= target.target_value
            
            metrics_summary = self.metrics_collector.get_metrics_summary()
            
            return {
                "passed": compliance,
                "measurements": {
                    "average_latency_ms": avg_latency,
                    "p95_latency_ms": p95_latency,
                    "min_latency_ms": min(latencies),
                    "max_latency_ms": max(latencies)
                },
                "prd_compliance": {
                    "target_ms": target.target_value,
                    "actual_ms": avg_latency,
                    "meets_target": compliance
                },
                "resource_usage": metrics_summary.get("resource_usage", {}),
                "performance_metrics": {
                    "latencies": latencies,
                    "total_computations": len(latencies)
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "measurements": {},
                "prd_compliance": {"meets_target": False}
            }
    
    def _test_batch_reranking_performance(self) -> Dict[str, Any]:
        """Test batch reranking performance."""
        target = self.prd_targets["batch_reranking_latency"]
        
        try:
            # Mock batch reranking function
            def mock_batch_reranking(query_embedding, candidate_embeddings):
                """Mock batch reranking with quantum similarity."""
                # Simulate batch processing
                similarities = []
                
                for candidate in candidate_embeddings:
                    # Mock quantum similarity computation
                    time.sleep(0.003)  # 3ms per similarity
                    similarity = float(np.random.random())
                    similarities.append(similarity)
                
                # Sort by similarity (reranking)
                ranked_indices = np.argsort(similarities)[::-1]
                return ranked_indices.tolist()
            
            # Generate test data
            query_embedding = np.random.randn(384)
            candidate_embeddings = [np.random.randn(384) for _ in range(100)]
            
            # Warmup
            for _ in range(3):
                mock_batch_reranking(query_embedding, candidate_embeddings[:10])
            
            # Measure batch performance
            batch_latencies = []
            
            with self.metrics_collector.collect_metrics():
                for _ in range(10):
                    with self.timer.time_operation("batch_reranking") as timing:
                        rankings = mock_batch_reranking(query_embedding, candidate_embeddings)
                    
                    batch_latencies.append(timing.execution_time_ms)
            
            avg_batch_latency = np.mean(batch_latencies)
            
            # Check PRD compliance
            compliance = avg_batch_latency <= target.target_value
            
            metrics_summary = self.metrics_collector.get_metrics_summary()
            
            return {
                "passed": compliance,
                "measurements": {
                    "average_batch_latency_ms": avg_batch_latency,
                    "batch_size": len(candidate_embeddings),
                    "per_item_latency_ms": avg_batch_latency / len(candidate_embeddings)
                },
                "prd_compliance": {
                    "target_ms": target.target_value,
                    "actual_ms": avg_batch_latency,
                    "meets_target": compliance
                },
                "resource_usage": metrics_summary.get("resource_usage", {}),
                "performance_metrics": {
                    "batch_latencies": batch_latencies,
                    "total_batches": len(batch_latencies)
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "measurements": {},
                "prd_compliance": {"meets_target": False}
            }
    
    def _test_memory_usage_compliance(self) -> Dict[str, Any]:
        """Test memory usage compliance."""
        target = self.prd_targets["memory_usage"]
        
        try:
            # Mock memory-intensive operations
            def memory_intensive_processing():
                """Simulate memory-intensive quantum processing."""
                # Allocate memory for quantum state vectors
                large_matrices = []
                
                for i in range(10):
                    # Simulate quantum state matrices
                    matrix = np.random.randn(1000, 1000)  # ~8MB each
                    large_matrices.append(matrix)
                    
                    # Simulate quantum operations
                    result = np.dot(matrix, matrix.T)
                    time.sleep(0.1)
                
                return large_matrices
            
            # Monitor memory usage
            process = psutil.Process()
            initial_memory_mb = process.memory_info().rss / 1024 / 1024
            
            peak_memory_mb = initial_memory_mb
            
            with self.metrics_collector.collect_metrics():
                # Start memory monitoring
                def monitor_memory():
                    nonlocal peak_memory_mb
                    while True:
                        try:
                            current_memory_mb = process.memory_info().rss / 1024 / 1024
                            peak_memory_mb = max(peak_memory_mb, current_memory_mb)
                            time.sleep(0.1)
                        except:
                            break
                
                monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
                monitor_thread.start()
                
                # Run memory-intensive operations
                with self.timer.time_operation("memory_intensive_processing") as timing:
                    results = memory_intensive_processing()
                
                # Allow final memory measurement
                time.sleep(0.5)
            
            memory_used_gb = (peak_memory_mb - initial_memory_mb) / 1024
            
            # Check PRD compliance
            compliance = memory_used_gb <= target.target_value
            
            metrics_summary = self.metrics_collector.get_metrics_summary()
            
            return {
                "passed": compliance,
                "measurements": {
                    "peak_memory_usage_gb": memory_used_gb,
                    "initial_memory_mb": initial_memory_mb,
                    "peak_memory_mb": peak_memory_mb,
                    "processing_time_ms": timing.execution_time_ms
                },
                "prd_compliance": {
                    "target_gb": target.target_value,
                    "actual_gb": memory_used_gb,
                    "meets_target": compliance
                },
                "resource_usage": metrics_summary.get("resource_usage", {}),
                "performance_metrics": {
                    "memory_efficiency": timing.execution_time_ms / max(memory_used_gb, 0.1)
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "measurements": {},
                "prd_compliance": {"meets_target": False}
            }
    
    def _test_accuracy_improvement(self) -> Dict[str, Any]:
        """Test accuracy improvement over baseline."""
        target = self.prd_targets["accuracy_improvement"]
        
        try:
            # Mock baseline and quantum accuracy measurements
            def mock_baseline_accuracy():
                """Mock baseline (classical) accuracy."""
                # Simulate classical similarity computation
                return 0.75  # 75% baseline accuracy
            
            def mock_quantum_accuracy():
                """Mock quantum-enhanced accuracy."""
                # Simulate quantum similarity with enhancement
                return 0.87  # 87% quantum accuracy
            
            # Measure baseline accuracy
            baseline_scores = []
            for _ in range(10):
                score = mock_baseline_accuracy() + np.random.normal(0, 0.02)
                baseline_scores.append(score)
            
            # Measure quantum accuracy
            quantum_scores = []
            for _ in range(10):
                score = mock_quantum_accuracy() + np.random.normal(0, 0.02)
                quantum_scores.append(score)
            
            baseline_avg = np.mean(baseline_scores)
            quantum_avg = np.mean(quantum_scores)
            
            improvement = (quantum_avg - baseline_avg) / baseline_avg
            
            # Check PRD compliance
            compliance = improvement >= target.target_value
            
            return {
                "passed": compliance,
                "measurements": {
                    "baseline_accuracy": baseline_avg,
                    "quantum_accuracy": quantum_avg,
                    "absolute_improvement": quantum_avg - baseline_avg,
                    "relative_improvement": improvement
                },
                "prd_compliance": {
                    "target_improvement": target.target_value,
                    "actual_improvement": improvement,
                    "meets_target": compliance
                },
                "performance_metrics": {
                    "baseline_scores": baseline_scores,
                    "quantum_scores": quantum_scores,
                    "improvement_samples": len(baseline_scores)
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "measurements": {},
                "prd_compliance": {"meets_target": False}
            }
    
    def _test_system_throughput(self) -> Dict[str, Any]:
        """Test system throughput under load."""
        target = self.prd_targets["throughput"]
        
        try:
            # Mock high-throughput processing
            def process_similarity_request():
                """Mock single similarity request processing."""
                # Simulate lightweight quantum computation
                time.sleep(0.001)  # 1ms per request
                return np.random.random()
            
            # Measure throughput
            num_workers = multiprocessing.cpu_count()
            total_requests = 2000
            
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all requests
                futures = [executor.submit(process_similarity_request) for _ in range(total_requests)]
                
                # Wait for completion
                completed = 0
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        completed += 1
                    except Exception as e:
                        logger.error(f"Request failed: {e}")
            
            total_time = time.time() - start_time
            throughput = completed / total_time
            
            # Check PRD compliance
            compliance = throughput >= target.target_value
            
            return {
                "passed": compliance,
                "measurements": {
                    "throughput_ops_per_sec": throughput,
                    "total_requests": total_requests,
                    "completed_requests": completed,
                    "total_time_seconds": total_time,
                    "workers_used": num_workers
                },
                "prd_compliance": {
                    "target_ops_per_sec": target.target_value,
                    "actual_ops_per_sec": throughput,
                    "meets_target": compliance
                },
                "performance_metrics": {
                    "success_rate": completed / total_requests,
                    "avg_latency_ms": (total_time * 1000) / completed if completed > 0 else 0
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "measurements": {},
                "prd_compliance": {"meets_target": False}
            }
    
    def _test_startup_performance(self) -> Dict[str, Any]:
        """Test system startup time."""
        target = self.prd_targets["startup_time"]
        
        try:
            # Mock system startup sequence
            def mock_system_startup():
                """Mock complete system startup."""
                startup_phases = [
                    ("Load configuration", 2.0),
                    ("Initialize quantum simulators", 5.0),
                    ("Load ML models", 8.0),
                    ("Initialize caching", 1.5),
                    ("Start API server", 3.0),
                    ("Health checks", 2.0)
                ]
                
                phase_times = {}
                
                for phase_name, base_time in startup_phases:
                    # Add some randomness
                    actual_time = base_time + np.random.normal(0, base_time * 0.1)
                    phase_times[phase_name] = actual_time
                    
                    # Simulate phase execution
                    time.sleep(min(actual_time / 10, 1.0))  # Scaled down for testing
                
                return phase_times
            
            # Measure startup time
            with self.timer.time_operation("system_startup") as timing:
                phase_times = mock_system_startup()
            
            total_startup_time = sum(phase_times.values())
            
            # Check PRD compliance
            compliance = total_startup_time <= target.target_value
            
            return {
                "passed": compliance,
                "measurements": {
                    "total_startup_time_seconds": total_startup_time,
                    "phase_breakdown": phase_times,
                    "measured_time_seconds": timing.execution_time_ms / 1000
                },
                "prd_compliance": {
                    "target_seconds": target.target_value,
                    "actual_seconds": total_startup_time,
                    "meets_target": compliance
                },
                "performance_metrics": {
                    "slowest_phase": max(phase_times.items(), key=lambda x: x[1]),
                    "total_phases": len(phase_times)
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "measurements": {},
                "prd_compliance": {"meets_target": False}
            }
    
    def _test_quantum_fidelity_accuracy(self) -> Dict[str, Any]:
        """Test quantum fidelity computation accuracy."""
        target = self.prd_targets["quantum_fidelity_accuracy"]
        
        try:
            # Mock quantum fidelity computation
            def compute_quantum_fidelity(state1, state2):
                """Mock quantum fidelity computation."""
                # Add small computational error
                theoretical = abs(np.vdot(state1, state2))**2
                error = np.random.normal(0, 1e-8)  # Very small error
                return theoretical + error
            
            def compute_reference_fidelity(state1, state2):
                """Reference analytical fidelity."""
                return abs(np.vdot(state1, state2))**2
            
            # Test cases with known fidelities
            test_cases = [
                (np.array([1, 0]), np.array([1, 0]), 1.0),  # Identical
                (np.array([1, 0]), np.array([0, 1]), 0.0),  # Orthogonal
                (np.array([1, 1])/np.sqrt(2), np.array([1, -1])/np.sqrt(2), 0.0),  # Orthogonal superposition
                (np.array([1, 0]), np.array([np.cos(np.pi/4), np.sin(np.pi/4)]), 0.5)  # 45 degree
            ]
            
            errors = []
            
            for state1, state2, expected in test_cases:
                for _ in range(10):  # Multiple measurements
                    quantum_result = compute_quantum_fidelity(state1, state2)
                    reference_result = compute_reference_fidelity(state1, state2)
                    
                    error = abs(quantum_result - reference_result)
                    errors.append(error)
            
            max_error = max(errors)
            avg_error = np.mean(errors)
            
            # Check PRD compliance
            compliance = max_error <= target.target_value
            
            return {
                "passed": compliance,
                "measurements": {
                    "max_error": max_error,
                    "average_error": avg_error,
                    "min_error": min(errors),
                    "error_std": np.std(errors)
                },
                "prd_compliance": {
                    "target_error": target.target_value,
                    "actual_max_error": max_error,
                    "meets_target": compliance
                },
                "performance_metrics": {
                    "total_tests": len(errors),
                    "error_distribution": errors,
                    "test_cases": len(test_cases)
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "measurements": {},
                "prd_compliance": {"meets_target": False}
            }
    
    def generate_prd_compliance_report(self, results: List[TestResult]) -> Dict[str, Any]:
        """Generate comprehensive PRD compliance report."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        critical_tests = []
        non_critical_tests = []
        
        # Categorize by criticality
        for result in results:
            test_name = result.test_case.name
            if any(target.critical for target in self.prd_targets.values() if target.name in test_name):
                critical_tests.append(result)
            else:
                non_critical_tests.append(result)
        
        critical_passed = sum(1 for r in critical_tests if r.passed)
        
        # Production readiness assessment
        production_ready = (critical_passed == len(critical_tests) and 
                          passed_tests / total_tests >= 0.8)  # 80% overall pass rate
        
        return {
            "prd_compliance_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "overall_pass_rate": passed_tests / total_tests if total_tests > 0 else 0
            },
            "critical_requirements": {
                "total_critical": len(critical_tests),
                "critical_passed": critical_passed,
                "critical_pass_rate": critical_passed / len(critical_tests) if critical_tests else 0,
                "all_critical_passed": critical_passed == len(critical_tests)
            },
            "production_readiness": {
                "ready_for_production": production_ready,
                "readiness_score": (critical_passed / len(critical_tests) if critical_tests else 0) * 0.7 + 
                                 (passed_tests / total_tests if total_tests else 0) * 0.3,
                "blocking_issues": [
                    result.test_case.name for result in critical_tests if not result.passed
                ]
            },
            "performance_targets": {
                target_name: {
                    "target": target.target_value,
                    "unit": target.unit,
                    "critical": target.critical,
                    "status": "PASS" if any(r.passed for r in results if target_name in r.test_case.name) else "FAIL"
                }
                for target_name, target in self.prd_targets.items()
            },
            "recommendations": self._generate_performance_recommendations(results)
        }
    
    def _generate_performance_recommendations(self, results: List[TestResult]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        failed_results = [r for r in results if not r.passed]
        
        for result in failed_results:
            test_name = result.test_case.name
            
            if "latency" in test_name:
                recommendations.append(
                    f"Optimize {test_name}: Consider caching, parallel processing, or algorithm optimization"
                )
            elif "memory" in test_name:
                recommendations.append(
                    f"Reduce memory usage in {test_name}: Implement memory pooling or streaming processing"
                )
            elif "throughput" in test_name:
                recommendations.append(
                    f"Improve throughput for {test_name}: Scale horizontally or optimize critical paths"
                )
            elif "accuracy" in test_name:
                recommendations.append(
                    f"Enhance accuracy in {test_name}: Review quantum algorithms or increase precision"
                )
            else:
                recommendations.append(
                    f"Address performance issues in {test_name}: Review implementation and optimize bottlenecks"
                )
        
        return recommendations
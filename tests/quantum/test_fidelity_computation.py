"""
Specialized testing for quantum fidelity computation accuracy and performance.

This module provides comprehensive testing for fidelity accuracy, computation benchmarks,
and validation against theoretical values.
"""

import numpy as np
import torch
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from .quantum_test_framework import QuantumTestFramework, FidelityTestCase, QuantumTestResult
from quantum_rerank.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class FidelityBenchmark:
    """Benchmark configuration for fidelity computation."""
    name: str
    state_dimension: int
    num_iterations: int
    expected_accuracy: float
    max_time_ms: float
    description: str = ""
    test_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class FidelityTestResult:
    """Result of fidelity computation test."""
    benchmark: FidelityBenchmark
    computed_fidelity: float
    reference_fidelity: float
    accuracy_error: float
    computation_time_ms: float
    passed: bool
    error_message: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class FidelityAccuracyTester:
    """
    Tests quantum fidelity computation accuracy against analytical solutions.
    """
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.test_cases: List[FidelityTestCase] = []
        self.logger = logger
        
        # Initialize quantum framework for fidelity computation
        self.quantum_framework = QuantumTestFramework()
    
    def add_test_case(self, test_case: FidelityTestCase) -> None:
        """Add a fidelity test case."""
        self.test_cases.append(test_case)
    
    def generate_standard_test_cases(self) -> List[FidelityTestCase]:
        """Generate comprehensive standard test cases for fidelity computation."""
        test_cases = []
        
        # Test 1: Identical states (|ψ⟩ = |φ⟩)
        state = np.array([1.0, 0.0])
        test_cases.append(FidelityTestCase(
            name="identical_states_2d",
            state1=state,
            state2=state.copy(),
            expected_fidelity=1.0,
            tolerance=1e-12,
            description="Fidelity between identical 2D states should be exactly 1.0",
            tags=["basic", "analytical", "2d"]
        ))
        
        # Test 2: Orthogonal computational basis states
        state1 = np.array([1.0, 0.0])
        state2 = np.array([0.0, 1.0])
        test_cases.append(FidelityTestCase(
            name="orthogonal_computational_basis",
            state1=state1,
            state2=state2,
            expected_fidelity=0.0,
            tolerance=1e-12,
            description="Fidelity between orthogonal computational basis states",
            tags=["basic", "analytical", "orthogonal"]
        ))
        
        # Test 3: Equal superposition states
        state1 = np.array([1, 1]) / np.sqrt(2)
        state2 = np.array([1, -1]) / np.sqrt(2)
        test_cases.append(FidelityTestCase(
            name="orthogonal_superposition",
            state1=state1,
            state2=state2,
            expected_fidelity=0.0,
            tolerance=1e-10,
            description="Fidelity between orthogonal superposition states",
            tags=["superposition", "analytical", "orthogonal"]
        ))
        
        # Test 4: Partially overlapping states
        state1 = np.array([1, 0]) 
        state2 = np.array([np.cos(np.pi/4), np.sin(np.pi/4)])  # 45 degree rotation
        expected_fidelity = np.cos(np.pi/4)**2  # |⟨ψ|φ⟩|²
        test_cases.append(FidelityTestCase(
            name="partial_overlap_45deg",
            state1=state1,
            state2=state2,
            expected_fidelity=expected_fidelity,
            tolerance=1e-10,
            description="Fidelity between states with 45-degree angle",
            tags=["partial", "analytical", "rotation"]
        ))
        
        # Test 5: Complex states
        state1 = np.array([1, 1j]) / np.sqrt(2)
        state2 = np.array([1j, 1]) / np.sqrt(2)
        expected_fidelity = 0.0  # These are orthogonal
        test_cases.append(FidelityTestCase(
            name="complex_orthogonal",
            state1=state1,
            state2=state2,
            expected_fidelity=expected_fidelity,
            tolerance=1e-10,
            description="Fidelity between orthogonal complex states",
            tags=["complex", "analytical", "orthogonal"]
        ))
        
        # Test 6: Higher dimensional states
        dim = 4
        state1 = np.zeros(dim, dtype=complex)
        state1[0] = 1.0
        state2 = np.ones(dim, dtype=complex) / np.sqrt(dim)
        expected_fidelity = (1/dim)  # |⟨e₁|uniform⟩|²
        test_cases.append(FidelityTestCase(
            name="4d_basis_uniform",
            state1=state1,
            state2=state2,
            expected_fidelity=expected_fidelity,
            tolerance=1e-10,
            description="Fidelity between basis state and uniform superposition in 4D",
            tags=["high_dim", "analytical", "uniform"]
        ))
        
        # Test 7: Random normalized states (statistical test)
        np.random.seed(42)
        for i in range(5):
            state1 = np.random.randn(2) + 1j * np.random.randn(2)
            state1 = state1 / np.linalg.norm(state1)
            state2 = np.random.randn(2) + 1j * np.random.randn(2)
            state2 = state2 / np.linalg.norm(state2)
            
            # Analytical fidelity
            expected_fidelity = abs(np.vdot(state1, state2))**2
            
            test_cases.append(FidelityTestCase(
                name=f"random_normalized_{i}",
                state1=state1,
                state2=state2,
                expected_fidelity=expected_fidelity,
                tolerance=1e-8,
                description=f"Random normalized complex states {i}",
                tags=["random", "complex", "statistical"]
            ))
        
        return test_cases
    
    def run_accuracy_tests(self, test_cases: Optional[List[FidelityTestCase]] = None) -> List[FidelityTestResult]:
        """
        Run fidelity accuracy tests.
        
        Args:
            test_cases: Specific test cases to run (uses standard if None)
            
        Returns:
            List of test results
        """
        if test_cases is None:
            test_cases = self.generate_standard_test_cases()
        
        results = []
        
        for test_case in test_cases:
            self.logger.info(f"Running fidelity accuracy test: {test_case.name}")
            
            start_time = time.time()
            
            try:
                # Compute fidelity using quantum implementation
                computed_fidelity = self.quantum_framework._compute_quantum_fidelity(
                    test_case.state1, test_case.state2
                )
                
                # Compute reference fidelity analytically
                reference_fidelity = self.quantum_framework._compute_reference_fidelity(
                    test_case.state1, test_case.state2
                )
                
                computation_time_ms = (time.time() - start_time) * 1000
                
                # Check accuracy
                accuracy_error = abs(computed_fidelity - reference_fidelity)
                passed = accuracy_error <= test_case.tolerance
                
                # Performance metrics
                performance_metrics = {
                    "accuracy_error": accuracy_error,
                    "relative_error": accuracy_error / (reference_fidelity + 1e-12),
                    "computation_time_ms": computation_time_ms,
                    "state_dimension": len(test_case.state1)
                }
                
                result = FidelityTestResult(
                    benchmark=None,  # Not a benchmark test
                    computed_fidelity=computed_fidelity,
                    reference_fidelity=reference_fidelity,
                    accuracy_error=accuracy_error,
                    computation_time_ms=computation_time_ms,
                    passed=passed,
                    performance_metrics=performance_metrics
                )
                
                results.append(result)
                
                if passed:
                    self.logger.info(f"✓ {test_case.name}: accuracy_error={accuracy_error:.2e}")
                else:
                    self.logger.warning(f"✗ {test_case.name}: accuracy_error={accuracy_error:.2e} > tolerance={test_case.tolerance:.2e}")
                
            except Exception as e:
                computation_time_ms = (time.time() - start_time) * 1000
                
                result = FidelityTestResult(
                    benchmark=None,
                    computed_fidelity=0.0,
                    reference_fidelity=test_case.expected_fidelity or 0.0,
                    accuracy_error=float('inf'),
                    computation_time_ms=computation_time_ms,
                    passed=False,
                    error_message=str(e)
                )
                
                results.append(result)
                self.logger.error(f"✗ {test_case.name}: Exception - {e}")
        
        return results
    
    def generate_accuracy_report(self, results: List[FidelityTestResult]) -> Dict[str, Any]:
        """Generate comprehensive accuracy test report."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = total_tests - passed_tests
        
        if results:
            avg_error = np.mean([r.accuracy_error for r in results if r.accuracy_error != float('inf')])
            max_error = max([r.accuracy_error for r in results if r.accuracy_error != float('inf')])
            avg_time = np.mean([r.computation_time_ms for r in results])
        else:
            avg_error = max_error = avg_time = 0.0
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "accuracy_statistics": {
                "average_error": avg_error,
                "maximum_error": max_error,
                "tolerance_threshold": self.tolerance
            },
            "performance_statistics": {
                "average_time_ms": avg_time,
                "total_time_ms": sum(r.computation_time_ms for r in results)
            },
            "failed_tests": [
                {
                    "test_name": f"test_{i}",
                    "error": r.error_message or f"Accuracy error {r.accuracy_error:.2e}",
                    "accuracy_error": r.accuracy_error
                }
                for i, r in enumerate(results) if not r.passed
            ]
        }


class FidelityBenchmarkSuite:
    """
    Comprehensive benchmarking suite for fidelity computation performance.
    """
    
    def __init__(self):
        self.benchmarks: List[FidelityBenchmark] = []
        self.logger = logger
        self.quantum_framework = QuantumTestFramework()
    
    def add_benchmark(self, benchmark: FidelityBenchmark) -> None:
        """Add a benchmark to the suite."""
        self.benchmarks.append(benchmark)
    
    def generate_standard_benchmarks(self) -> List[FidelityBenchmark]:
        """Generate standard performance benchmarks."""
        benchmarks = []
        
        # Dimension scaling benchmarks
        for dim in [2, 4, 8, 16, 32]:
            benchmarks.append(FidelityBenchmark(
                name=f"dimension_scaling_{dim}d",
                state_dimension=dim,
                num_iterations=100,
                expected_accuracy=1e-6,
                max_time_ms=dim * 10,  # Scale with dimension
                description=f"Fidelity computation scaling for {dim}-dimensional states",
                test_parameters={"complexity": "linear"}
            ))
        
        # Iteration count benchmarks
        for iterations in [10, 100, 1000]:
            benchmarks.append(FidelityBenchmark(
                name=f"iteration_scaling_{iterations}",
                state_dimension=4,
                num_iterations=iterations,
                expected_accuracy=1e-6,
                max_time_ms=iterations * 2,
                description=f"Performance with {iterations} iterations",
                test_parameters={"focus": "iteration_scaling"}
            ))
        
        # Accuracy vs. performance trade-off
        for tolerance in [1e-4, 1e-6, 1e-8, 1e-10]:
            benchmarks.append(FidelityBenchmark(
                name=f"accuracy_tolerance_{tolerance:.0e}",
                state_dimension=8,
                num_iterations=50,
                expected_accuracy=tolerance,
                max_time_ms=200,
                description=f"Accuracy vs performance with tolerance {tolerance:.0e}",
                test_parameters={"tolerance": tolerance}
            ))
        
        return benchmarks
    
    def run_benchmarks(self, benchmarks: Optional[List[FidelityBenchmark]] = None) -> List[FidelityTestResult]:
        """
        Run performance benchmarks.
        
        Args:
            benchmarks: Specific benchmarks to run (uses standard if None)
            
        Returns:
            List of benchmark results
        """
        if benchmarks is None:
            benchmarks = self.generate_standard_benchmarks()
        
        results = []
        
        for benchmark in benchmarks:
            self.logger.info(f"Running benchmark: {benchmark.name}")
            
            # Generate test states for benchmark
            test_states = self._generate_benchmark_states(
                benchmark.state_dimension, 
                benchmark.num_iterations
            )
            
            # Run benchmark
            start_time = time.time()
            
            try:
                fidelities = []
                
                for state1, state2 in test_states:
                    fidelity = self.quantum_framework._compute_quantum_fidelity(state1, state2)
                    fidelities.append(fidelity)
                
                total_time_ms = (time.time() - start_time) * 1000
                avg_time_per_computation = total_time_ms / len(test_states)
                
                # Check performance criteria
                passed = (avg_time_per_computation <= benchmark.max_time_ms and 
                         all(0 <= f <= 1 for f in fidelities))
                
                # Performance metrics
                performance_metrics = {
                    "total_computations": len(test_states),
                    "avg_time_per_computation_ms": avg_time_per_computation,
                    "throughput_computations_per_second": 1000 / avg_time_per_computation if avg_time_per_computation > 0 else 0,
                    "state_dimension": benchmark.state_dimension,
                    "min_fidelity": min(fidelities),
                    "max_fidelity": max(fidelities),
                    "avg_fidelity": np.mean(fidelities)
                }
                
                result = FidelityTestResult(
                    benchmark=benchmark,
                    computed_fidelity=np.mean(fidelities),
                    reference_fidelity=0.5,  # Expected average for random states
                    accuracy_error=0.0,  # Not testing accuracy here
                    computation_time_ms=total_time_ms,
                    passed=passed,
                    performance_metrics=performance_metrics
                )
                
                results.append(result)
                
                if passed:
                    self.logger.info(f"✓ {benchmark.name}: {avg_time_per_computation:.2f}ms/computation")
                else:
                    self.logger.warning(f"✗ {benchmark.name}: {avg_time_per_computation:.2f}ms/computation > {benchmark.max_time_ms}ms")
                
            except Exception as e:
                total_time_ms = (time.time() - start_time) * 1000
                
                result = FidelityTestResult(
                    benchmark=benchmark,
                    computed_fidelity=0.0,
                    reference_fidelity=0.0,
                    accuracy_error=float('inf'),
                    computation_time_ms=total_time_ms,
                    passed=False,
                    error_message=str(e)
                )
                
                results.append(result)
                self.logger.error(f"✗ {benchmark.name}: Exception - {e}")
        
        return results
    
    def _generate_benchmark_states(self, dimension: int, num_pairs: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate random state pairs for benchmarking."""
        np.random.seed(42)  # Reproducible benchmarks
        
        pairs = []
        for _ in range(num_pairs):
            # Generate random normalized states
            state1 = np.random.randn(dimension) + 1j * np.random.randn(dimension)
            state1 = state1 / np.linalg.norm(state1)
            
            state2 = np.random.randn(dimension) + 1j * np.random.randn(dimension)
            state2 = state2 / np.linalg.norm(state2)
            
            pairs.append((state1, state2))
        
        return pairs
    
    def generate_performance_report(self, results: List[FidelityTestResult]) -> Dict[str, Any]:
        """Generate comprehensive performance benchmark report."""
        total_benchmarks = len(results)
        passed_benchmarks = sum(1 for r in results if r.passed)
        failed_benchmarks = total_benchmarks - passed_benchmarks
        
        # Performance statistics
        if results:
            all_times = [r.performance_metrics.get("avg_time_per_computation_ms", 0) for r in results if r.performance_metrics]
            avg_computation_time = np.mean(all_times) if all_times else 0
            max_computation_time = max(all_times) if all_times else 0
            
            all_throughputs = [r.performance_metrics.get("throughput_computations_per_second", 0) for r in results if r.performance_metrics]
            avg_throughput = np.mean(all_throughputs) if all_throughputs else 0
        else:
            avg_computation_time = max_computation_time = avg_throughput = 0
        
        # Dimension scaling analysis
        dimension_performance = {}
        for result in results:
            if result.benchmark and result.performance_metrics:
                dim = result.benchmark.state_dimension
                if dim not in dimension_performance:
                    dimension_performance[dim] = []
                dimension_performance[dim].append(result.performance_metrics.get("avg_time_per_computation_ms", 0))
        
        return {
            "total_benchmarks": total_benchmarks,
            "passed_benchmarks": passed_benchmarks,
            "failed_benchmarks": failed_benchmarks,
            "pass_rate": passed_benchmarks / total_benchmarks if total_benchmarks > 0 else 0,
            "performance_statistics": {
                "avg_computation_time_ms": avg_computation_time,
                "max_computation_time_ms": max_computation_time,
                "avg_throughput_computations_per_second": avg_throughput
            },
            "dimension_scaling": {
                str(dim): {
                    "avg_time_ms": np.mean(times),
                    "min_time_ms": min(times),
                    "max_time_ms": max(times)
                }
                for dim, times in dimension_performance.items()
            },
            "failed_benchmarks": [
                {
                    "benchmark_name": r.benchmark.name if r.benchmark else "unknown",
                    "error": r.error_message or "Performance threshold exceeded",
                    "actual_time_ms": r.performance_metrics.get("avg_time_per_computation_ms", 0) if r.performance_metrics else 0
                }
                for r in results if not r.passed
            ]
        }
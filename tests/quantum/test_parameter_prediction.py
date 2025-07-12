"""
Testing for quantum parameter prediction and optimization.

This module provides comprehensive testing for ML-based quantum parameter prediction,
parameter optimization algorithms, and parameter validation.
"""

import numpy as np
import torch
import time
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from .quantum_test_framework import QuantumTestFramework, QuantumTestCase, QuantumTestResult
from quantum_rerank.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ParameterPredictionTest:
    """Test case for parameter prediction."""
    name: str
    embeddings: List[np.ndarray]
    expected_parameters: Optional[List[np.ndarray]]
    predictor_function: Callable
    accuracy_threshold: float = 0.1
    latency_threshold_ms: float = 100
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class ParameterValidationTest:
    """Test case for parameter validation."""
    name: str
    parameters: List[np.ndarray]
    validation_criteria: Dict[str, Any]
    validation_functions: List[Callable]
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class PredictionResult:
    """Result of parameter prediction test."""
    test_case: ParameterPredictionTest
    predicted_parameters: List[np.ndarray]
    accuracy_scores: List[float]
    prediction_time_ms: float
    average_accuracy: float
    latency_passed: bool
    accuracy_passed: bool
    passed: bool = False
    error_message: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class ParameterPredictionTester:
    """
    Tests quantum parameter prediction algorithms and ML models.
    """
    
    def __init__(self, accuracy_threshold: float = 0.1):
        self.accuracy_threshold = accuracy_threshold
        self.prediction_tests: List[ParameterPredictionTest] = []
        self.logger = logger
        
        # Initialize quantum framework
        self.quantum_framework = QuantumTestFramework()
    
    def add_prediction_test(self, test: ParameterPredictionTest) -> None:
        """Add a parameter prediction test."""
        self.prediction_tests.append(test)
    
    def generate_standard_prediction_tests(self) -> List[ParameterPredictionTest]:
        """Generate standard parameter prediction test cases."""
        tests = []
        
        # Mock parameter prediction functions for testing
        def mock_simple_predictor(embedding):
            """Simple mock parameter predictor."""
            # Simulate parameter prediction based on embedding
            normalized_emb = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            # Simple heuristic: map embedding features to parameter ranges
            num_params = 8  # Standard number of parameters
            
            # Map embedding to parameter space [-π, π]
            if len(normalized_emb) >= num_params:
                params = normalized_emb[:num_params] * np.pi
            else:
                # Repeat pattern if embedding is smaller
                params = np.tile(normalized_emb, (num_params // len(normalized_emb) + 1))[:num_params] * np.pi
            
            return params
        
        def mock_ml_predictor(embedding):
            """Mock ML-based parameter predictor."""
            # Simulate learned mapping
            embedding_norm = np.linalg.norm(embedding)
            embedding_mean = np.mean(embedding)
            
            # Generate parameters based on embedding statistics
            params = np.array([
                embedding_mean * np.pi,
                embedding_norm * 0.5,
                np.std(embedding) * 2,
                np.max(embedding) * 0.8,
                np.min(embedding) * 1.2,
                np.sum(embedding[:4]) * 0.1 if len(embedding) >= 4 else 0,
                np.prod(embedding[:2]) * 0.5 if len(embedding) >= 2 else 0,
                np.mean(embedding**2) * 0.3
            ])
            
            # Clip to valid range
            return np.clip(params, -np.pi, np.pi)
        
        def mock_optimal_predictor(embedding):
            """Mock optimal parameter predictor (for testing accuracy)."""
            # This predictor "knows" the optimal parameters
            # In real scenarios, this would be the trained model
            optimal_base = np.array([0.5, 1.0, -0.5, 0.8, -0.3, 1.2, -0.7, 0.2])
            
            # Add small perturbation based on embedding
            perturbation = np.sin(embedding[:8] if len(embedding) >= 8 else np.tile(embedding, 8)[:8]) * 0.1
            
            return optimal_base + perturbation
        
        # Test 1: Basic parameter prediction accuracy
        test_embeddings = [
            np.random.randn(16) for _ in range(10)
        ]
        
        expected_params = [
            mock_optimal_predictor(emb) for emb in test_embeddings
        ]
        
        tests.append(ParameterPredictionTest(
            name="basic_prediction_accuracy",
            embeddings=test_embeddings,
            expected_parameters=expected_params,
            predictor_function=mock_simple_predictor,
            accuracy_threshold=0.5,  # Relaxed for mock predictor
            latency_threshold_ms=50,
            description="Test basic parameter prediction accuracy",
            tags=["basic", "accuracy", "mock"]
        ))
        
        # Test 2: ML model prediction performance
        tests.append(ParameterPredictionTest(
            name="ml_prediction_performance",
            embeddings=test_embeddings,
            expected_parameters=expected_params,
            predictor_function=mock_ml_predictor,
            accuracy_threshold=0.3,
            latency_threshold_ms=100,
            description="Test ML-based parameter prediction",
            tags=["ml", "performance", "latency"]
        ))
        
        # Test 3: High-dimensional embedding handling
        high_dim_embeddings = [
            np.random.randn(384) for _ in range(5)  # Typical BERT embedding size
        ]
        
        high_dim_expected = [
            mock_optimal_predictor(emb) for emb in high_dim_embeddings
        ]
        
        tests.append(ParameterPredictionTest(
            name="high_dimensional_embeddings",
            embeddings=high_dim_embeddings,
            expected_parameters=high_dim_expected,
            predictor_function=mock_ml_predictor,
            accuracy_threshold=0.4,
            latency_threshold_ms=200,
            description="Test parameter prediction for high-dimensional embeddings",
            tags=["high_dim", "bert", "384d"]
        ))
        
        # Test 4: Batch prediction efficiency
        batch_embeddings = [
            np.random.randn(64) for _ in range(50)  # Larger batch
        ]
        
        batch_expected = [
            mock_optimal_predictor(emb) for emb in batch_embeddings
        ]
        
        tests.append(ParameterPredictionTest(
            name="batch_prediction_efficiency",
            embeddings=batch_embeddings,
            expected_parameters=batch_expected,
            predictor_function=mock_ml_predictor,
            accuracy_threshold=0.4,
            latency_threshold_ms=500,  # Total time for batch
            description="Test batch parameter prediction efficiency",
            tags=["batch", "efficiency", "scalability"]
        ))
        
        # Test 5: Edge case handling
        edge_case_embeddings = [
            np.zeros(16),  # Zero embedding
            np.ones(16),   # Uniform embedding
            np.random.randn(16) * 100,  # Large magnitude
            np.random.randn(16) * 1e-6,  # Small magnitude
            np.array([1] + [0]*15),  # Sparse embedding
        ]
        
        edge_case_expected = [
            mock_optimal_predictor(emb) for emb in edge_case_embeddings
        ]
        
        tests.append(ParameterPredictionTest(
            name="edge_case_robustness",
            embeddings=edge_case_embeddings,
            expected_parameters=edge_case_expected,
            predictor_function=mock_ml_predictor,
            accuracy_threshold=0.6,  # More lenient for edge cases
            latency_threshold_ms=100,
            description="Test parameter prediction robustness to edge cases",
            tags=["edge_cases", "robustness", "special"]
        ))
        
        return tests
    
    def run_prediction_tests(self, tests: Optional[List[ParameterPredictionTest]] = None) -> List[PredictionResult]:
        """
        Run parameter prediction tests.
        
        Args:
            tests: Specific tests to run (uses standard if None)
            
        Returns:
            List of prediction results
        """
        if tests is None:
            tests = self.generate_standard_prediction_tests()
        
        results = []
        
        for test in tests:
            self.logger.info(f"Running parameter prediction test: {test.name}")
            
            start_time = time.time()
            
            try:
                predicted_params = []
                accuracy_scores = []
                
                # Run prediction for each embedding
                for i, embedding in enumerate(test.embeddings):
                    prediction_start = time.time()
                    predicted = test.predictor_function(embedding)
                    prediction_time = (time.time() - prediction_start) * 1000
                    
                    predicted_params.append(predicted)
                    
                    # Calculate accuracy if expected parameters provided
                    if test.expected_parameters and i < len(test.expected_parameters):
                        expected = test.expected_parameters[i]
                        accuracy = self._calculate_parameter_accuracy(predicted, expected)
                        accuracy_scores.append(accuracy)
                
                total_time_ms = (time.time() - start_time) * 1000
                
                # Calculate metrics
                average_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.0
                latency_passed = total_time_ms <= test.latency_threshold_ms
                accuracy_passed = average_accuracy >= test.accuracy_threshold
                
                # Performance metrics
                performance_metrics = {
                    "total_predictions": len(predicted_params),
                    "average_accuracy": average_accuracy,
                    "accuracy_std": np.std(accuracy_scores) if accuracy_scores else 0.0,
                    "total_time_ms": total_time_ms,
                    "avg_time_per_prediction_ms": total_time_ms / len(test.embeddings),
                    "throughput_predictions_per_second": len(test.embeddings) * 1000 / total_time_ms if total_time_ms > 0 else 0
                }
                
                result = PredictionResult(
                    test_case=test,
                    predicted_parameters=predicted_params,
                    accuracy_scores=accuracy_scores,
                    prediction_time_ms=total_time_ms,
                    average_accuracy=average_accuracy,
                    latency_passed=latency_passed,
                    accuracy_passed=accuracy_passed,
                    passed=latency_passed and accuracy_passed,
                    performance_metrics=performance_metrics
                )
                
                results.append(result)
                
                if result.passed:
                    self.logger.info(f"✓ {test.name}: accuracy={average_accuracy:.3f}, time={total_time_ms:.1f}ms")
                else:
                    self.logger.warning(f"✗ {test.name}: accuracy={average_accuracy:.3f} (req: {test.accuracy_threshold}), time={total_time_ms:.1f}ms (req: {test.latency_threshold_ms}ms)")
                
            except Exception as e:
                total_time_ms = (time.time() - start_time) * 1000
                
                result = PredictionResult(
                    test_case=test,
                    predicted_parameters=[],
                    accuracy_scores=[],
                    prediction_time_ms=total_time_ms,
                    average_accuracy=0.0,
                    latency_passed=False,
                    accuracy_passed=False,
                    passed=False,
                    error_message=str(e)
                )
                
                results.append(result)
                self.logger.error(f"✗ {test.name}: Exception - {e}")
        
        return results
    
    def _calculate_parameter_accuracy(self, predicted: np.ndarray, expected: np.ndarray) -> float:
        """Calculate accuracy between predicted and expected parameters."""
        if len(predicted) != len(expected):
            # Handle different lengths by padding or truncating
            min_len = min(len(predicted), len(expected))
            predicted = predicted[:min_len]
            expected = expected[:min_len]
        
        # Calculate normalized mean squared error
        mse = np.mean((predicted - expected)**2)
        max_possible_error = np.mean((np.pi * np.ones_like(expected))**2)  # Max parameter range
        
        # Convert to accuracy score (1 - normalized_error)
        normalized_error = mse / max_possible_error
        accuracy = max(0, 1 - normalized_error)
        
        return accuracy


class QuantumParameterValidator:
    """
    Validates quantum parameters for correctness and optimization.
    """
    
    def __init__(self):
        self.validation_tests: List[ParameterValidationTest] = []
        self.logger = logger
    
    def add_validation_test(self, test: ParameterValidationTest) -> None:
        """Add a parameter validation test."""
        self.validation_tests.append(test)
    
    def generate_standard_validation_tests(self) -> List[ParameterValidationTest]:
        """Generate standard parameter validation tests."""
        tests = []
        
        # Validation functions
        def check_parameter_range(params):
            """Check parameters are in valid range [-π, π]."""
            return np.all((-np.pi <= params) & (params <= np.pi))
        
        def check_no_nans(params):
            """Check parameters contain no NaN values."""
            return not np.any(np.isnan(params))
        
        def check_no_infs(params):
            """Check parameters contain no infinite values."""
            return not np.any(np.isinf(params))
        
        def check_expected_dimension(params, expected_dim=8):
            """Check parameters have expected dimension."""
            return len(params) == expected_dim
        
        def check_parameter_diversity(params, min_std=0.1):
            """Check parameters have sufficient diversity."""
            return np.std(params) >= min_std
        
        # Test 1: Valid parameter range
        valid_params = [
            np.array([0.5, 1.0, -0.5, 0.8, -0.3, 1.2, -0.7, 0.2]),
            np.array([np.pi, -np.pi, 0, np.pi/2, -np.pi/2, np.pi/4, -np.pi/4, np.pi/3]),
            np.random.uniform(-np.pi, np.pi, 8)
        ]
        
        tests.append(ParameterValidationTest(
            name="valid_parameter_range",
            parameters=valid_params,
            validation_criteria={"range": "[-π, π]", "finite": True},
            validation_functions=[
                check_parameter_range,
                check_no_nans,
                check_no_infs,
                lambda p: check_expected_dimension(p, 8)
            ],
            description="Validate parameters are in correct range and finite",
            tags=["range", "finite", "basic"]
        ))
        
        # Test 2: Invalid parameters (should fail validation)
        invalid_params = [
            np.array([0.5, 1.0, -0.5, 5.0, -0.3, 1.2, -0.7, 0.2]),  # Out of range
            np.array([0.5, np.nan, -0.5, 0.8, -0.3, 1.2, -0.7, 0.2]),  # Contains NaN
            np.array([0.5, 1.0, np.inf, 0.8, -0.3, 1.2, -0.7, 0.2]),  # Contains Inf
            np.array([0.5, 1.0, -0.5, 0.8, -0.3])  # Wrong dimension
        ]
        
        tests.append(ParameterValidationTest(
            name="invalid_parameter_detection",
            parameters=invalid_params,
            validation_criteria={"expect_failures": True},
            validation_functions=[
                check_parameter_range,
                check_no_nans,
                check_no_infs,
                lambda p: check_expected_dimension(p, 8)
            ],
            description="Detect invalid parameters (should fail validation)",
            tags=["invalid", "detection", "negative"]
        ))
        
        # Test 3: Parameter diversity
        diverse_params = [
            np.random.uniform(-np.pi, np.pi, 8),  # Random diverse
            np.linspace(-np.pi, np.pi, 8),  # Evenly distributed
            np.array([-np.pi, np.pi, -np.pi/2, np.pi/2, 0, np.pi/4, -np.pi/4, np.pi/3])  # Hand-crafted diverse
        ]
        
        uniform_params = [
            np.zeros(8),  # All zeros
            np.ones(8) * 0.5,  # All same value
            np.array([0.1] * 8)  # Low diversity
        ]
        
        tests.append(ParameterValidationTest(
            name="parameter_diversity_check",
            parameters=diverse_params + uniform_params,
            validation_criteria={"diversity": "high"},
            validation_functions=[
                lambda p: check_parameter_diversity(p, 0.5)  # Require std >= 0.5
            ],
            description="Check parameter diversity to avoid local optima",
            tags=["diversity", "optimization", "quality"]
        ))
        
        # Test 4: Optimization-ready parameters
        def check_gradient_friendly(params):
            """Check parameters are not at boundaries (good for gradient optimization)."""
            boundary_threshold = 0.1
            return np.all((params > -np.pi + boundary_threshold) & (params < np.pi - boundary_threshold))
        
        def check_symmetric_distribution(params):
            """Check parameters are symmetrically distributed around zero."""
            return abs(np.mean(params)) < 0.5
        
        optimization_ready_params = [
            np.random.uniform(-2.5, 2.5, 8),  # Away from boundaries
            np.array([0.1, -0.1, 0.3, -0.3, 0.5, -0.5, 0.7, -0.7]),  # Symmetric
            np.random.normal(0, 1, 8)  # Normal distribution around zero
        ]
        
        tests.append(ParameterValidationTest(
            name="optimization_ready_parameters",
            parameters=optimization_ready_params,
            validation_criteria={"optimization_friendly": True},
            validation_functions=[
                check_gradient_friendly,
                check_symmetric_distribution,
                lambda p: check_parameter_diversity(p, 0.3)
            ],
            description="Validate parameters are suitable for optimization",
            tags=["optimization", "gradients", "ready"]
        ))
        
        return tests
    
    def run_validation_tests(self, tests: Optional[List[ParameterValidationTest]] = None) -> List[QuantumTestResult]:
        """
        Run parameter validation tests.
        
        Args:
            tests: Specific tests to run (uses standard if None)
            
        Returns:
            List of validation results
        """
        if tests is None:
            tests = self.generate_standard_validation_tests()
        
        results = []
        
        for test in tests:
            self.logger.info(f"Running parameter validation test: {test.name}")
            
            start_time = time.time()
            
            try:
                validation_results = []
                all_passed = True
                
                # Test each parameter set
                for i, params in enumerate(test.parameters):
                    param_results = {}
                    param_passed = True
                    
                    # Run validation functions
                    for j, validation_func in enumerate(test.validation_functions):
                        try:
                            func_result = validation_func(params)
                            param_results[f"validation_{j}"] = func_result
                            
                            # For "invalid_parameter_detection" test, we expect some to fail
                            if test.name == "invalid_parameter_detection" and i > 0:
                                # We expect parameters 1-3 to fail validation
                                if not func_result:
                                    continue  # Failure is expected
                                else:
                                    param_passed = False  # Should have failed but didn't
                            else:
                                if not func_result:
                                    param_passed = False
                        
                        except Exception as e:
                            param_results[f"validation_{j}"] = False
                            param_results[f"validation_{j}_error"] = str(e)
                            param_passed = False
                    
                    validation_results.append({
                        "parameter_set": i,
                        "results": param_results,
                        "passed": param_passed
                    })
                    
                    if not param_passed and test.name != "invalid_parameter_detection":
                        all_passed = False
                
                # Special handling for negative test case
                if test.name == "invalid_parameter_detection":
                    # For this test, we expect the first param set to pass and others to fail
                    expected_pattern = [True, False, False, False]
                    actual_pattern = [result["passed"] for result in validation_results]
                    all_passed = (actual_pattern == expected_pattern)
                
                execution_time_ms = (time.time() - start_time) * 1000
                
                # Create quantum test case for result
                quantum_test_case = QuantumTestCase(
                    name=test.name,
                    test_type=None,  # Validation test
                    test_function=lambda: None,
                    description=test.description,
                    tags=test.tags
                )
                
                result = QuantumTestResult(
                    test_case=quantum_test_case,
                    passed=all_passed,
                    actual_result=validation_results,
                    expected_result=test.validation_criteria,
                    error_tolerance=0.0,
                    execution_time_ms=execution_time_ms,
                    quantum_metrics={
                        "validation_details": validation_results,
                        "total_parameter_sets": len(test.parameters),
                        "total_validations": len(test.validation_functions) * len(test.parameters),
                        "passed_parameter_sets": sum(1 for result in validation_results if result["passed"])
                    }
                )
                
                results.append(result)
                
                if all_passed:
                    self.logger.info(f"✓ {test.name}: All parameter validations passed")
                else:
                    failed_count = sum(1 for result in validation_results if not result["passed"])
                    self.logger.warning(f"✗ {test.name}: {failed_count} parameter sets failed validation")
                
            except Exception as e:
                execution_time_ms = (time.time() - start_time) * 1000
                
                quantum_test_case = QuantumTestCase(
                    name=test.name,
                    test_type=None,
                    test_function=lambda: None,
                    description=test.description
                )
                
                result = QuantumTestResult(
                    test_case=quantum_test_case,
                    passed=False,
                    actual_result=None,
                    expected_result=test.validation_criteria,
                    error_tolerance=float('inf'),
                    execution_time_ms=execution_time_ms,
                    error_message=str(e)
                )
                
                results.append(result)
                self.logger.error(f"✗ {test.name}: Exception - {e}")
        
        return results
    
    def generate_validation_report(self, results: List[QuantumTestResult]) -> Dict[str, Any]:
        """Generate comprehensive parameter validation report."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = total_tests - passed_tests
        
        # Calculate detailed statistics
        total_parameter_sets = sum(
            r.quantum_metrics.get("total_parameter_sets", 0)
            for r in results if r.quantum_metrics
        )
        
        total_passed_sets = sum(
            r.quantum_metrics.get("passed_parameter_sets", 0)
            for r in results if r.quantum_metrics
        )
        
        parameter_pass_rate = total_passed_sets / total_parameter_sets if total_parameter_sets > 0 else 0
        
        if results:
            avg_time = np.mean([r.execution_time_ms for r in results])
        else:
            avg_time = 0.0
        
        return {
            "total_validation_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "test_pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "parameter_statistics": {
                "total_parameter_sets_tested": total_parameter_sets,
                "total_parameter_sets_passed": total_passed_sets,
                "parameter_set_pass_rate": parameter_pass_rate
            },
            "performance_statistics": {
                "average_test_time_ms": avg_time,
                "total_validation_time_ms": sum(r.execution_time_ms for r in results)
            },
            "failed_tests": [
                {
                    "test_name": r.test_case.name,
                    "error": r.error_message or "Parameter validation failed",
                    "failed_parameter_sets": r.quantum_metrics.get("total_parameter_sets", 0) - r.quantum_metrics.get("passed_parameter_sets", 0) if r.quantum_metrics else 0
                }
                for r in results if not r.passed
            ]
        }
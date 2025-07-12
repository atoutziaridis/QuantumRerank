"""
Testing for quantum circuit optimization and validation.

This module provides comprehensive testing for quantum circuit performance,
optimization algorithms, and circuit validation.
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
class CircuitOptimizationTest:
    """Test case for circuit optimization."""
    name: str
    circuit_function: Callable
    initial_parameters: np.ndarray
    target_cost: float
    max_iterations: int
    optimization_method: str = "adam"
    tolerance: float = 1e-6
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class CircuitValidationTest:
    """Test case for circuit validation."""
    name: str
    circuit_function: Callable
    test_inputs: List[np.ndarray]
    expected_properties: Dict[str, Any]
    validation_rules: List[Callable]
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class OptimizationResult:
    """Result of circuit optimization test."""
    test_case: CircuitOptimizationTest
    final_cost: float
    initial_cost: float
    cost_improvement: float
    iterations_taken: int
    convergence_achieved: bool
    optimization_time_ms: float
    final_parameters: np.ndarray
    cost_history: List[float] = field(default_factory=list)
    passed: bool = False
    error_message: Optional[str] = None


class CircuitOptimizationTester:
    """
    Tests quantum circuit optimization algorithms and performance.
    """
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.optimization_tests: List[CircuitOptimizationTest] = []
        self.logger = logger
        
        # Quantum framework for circuit testing
        self.quantum_framework = QuantumTestFramework()
    
    def add_optimization_test(self, test: CircuitOptimizationTest) -> None:
        """Add a circuit optimization test."""
        self.optimization_tests.append(test)
    
    def generate_standard_optimization_tests(self) -> List[CircuitOptimizationTest]:
        """Generate standard circuit optimization test cases."""
        tests = []
        
        # Test 1: Parameter initialization optimization
        def cost_function_simple(params):
            """Simple quadratic cost function."""
            return np.sum((params - np.array([0.5, 1.0, -0.5]))**2)
        
        tests.append(CircuitOptimizationTest(
            name="quadratic_parameter_optimization",
            circuit_function=cost_function_simple,
            initial_parameters=np.array([0.0, 0.0, 0.0]),
            target_cost=0.01,
            max_iterations=100,
            optimization_method="adam",
            tolerance=1e-6,
            description="Optimize simple quadratic cost function",
            tags=["basic", "quadratic", "adam"]
        ))
        
        # Test 2: Quantum state preparation optimization
        def state_preparation_cost(params):
            """Cost for quantum state preparation."""
            try:
                # Mock quantum state preparation circuit
                # In real implementation, this would use actual quantum circuits
                target_state = np.array([1, 1, 1, 1]) / 2  # Equal superposition
                
                # Simulate parameterized circuit output
                theta, phi, gamma = params[:3] if len(params) >= 3 else (params[0], 0, 0)
                prepared_state = np.array([
                    np.cos(theta/2),
                    np.sin(theta/2) * np.exp(1j * phi),
                    np.sin(gamma/2),
                    np.cos(gamma/2) * np.exp(1j * phi)
                ])
                prepared_state = prepared_state / np.linalg.norm(prepared_state)
                
                # Fidelity-based cost (1 - fidelity)
                fidelity = abs(np.vdot(target_state, prepared_state))**2
                return 1 - fidelity
                
            except Exception as e:
                logger.warning(f"State preparation cost error: {e}")
                return 1.0  # Maximum cost on error
        
        tests.append(CircuitOptimizationTest(
            name="quantum_state_preparation",
            circuit_function=state_preparation_cost,
            initial_parameters=np.random.rand(3) * 2 * np.pi,
            target_cost=0.05,
            max_iterations=200,
            optimization_method="adam",
            tolerance=1e-4,
            description="Optimize quantum state preparation circuit",
            tags=["quantum", "state_preparation", "fidelity"]
        ))
        
        # Test 3: Variational quantum eigensolver (VQE) simulation
        def vqe_cost_function(params):
            """Mock VQE cost function for H2 molecule."""
            # Simplified H2 Hamiltonian expectation value
            # In real implementation, this would compute actual expectation values
            optimal_params = np.array([np.pi/4, np.pi/3])
            param_diff = params[:2] - optimal_params
            return -1.137 + 0.5 * np.sum(param_diff**2)  # Ground state energy ≈ -1.137
        
        tests.append(CircuitOptimizationTest(
            name="vqe_h2_simulation",
            circuit_function=vqe_cost_function,
            initial_parameters=np.random.rand(2) * 2 * np.pi,
            target_cost=-1.13,  # Close to H2 ground state
            max_iterations=300,
            optimization_method="adam",
            tolerance=1e-3,
            description="VQE optimization for H2 molecule simulation",
            tags=["vqe", "chemistry", "h2", "optimization"]
        ))
        
        # Test 4: Quantum approximate optimization algorithm (QAOA)
        def qaoa_cost_function(params):
            """Mock QAOA cost function for MaxCut problem."""
            # Simplified MaxCut objective
            beta, gamma = params[:2] if len(params) >= 2 else (params[0], 0)
            
            # Mock expectation value calculation
            # In reality, this would evaluate the QAOA circuit
            optimal_cut = 2.5  # Target for 3-node graph
            
            # Simplified approximation
            cut_value = optimal_cut * np.cos(beta) * np.sin(gamma)
            return -cut_value  # Minimize negative (maximize cut)
        
        tests.append(CircuitOptimizationTest(
            name="qaoa_maxcut_3node",
            circuit_function=qaoa_cost_function,
            initial_parameters=np.array([np.pi/4, np.pi/6]),
            target_cost=-2.0,
            max_iterations=150,
            optimization_method="adam",
            tolerance=1e-3,
            description="QAOA optimization for 3-node MaxCut problem",
            tags=["qaoa", "maxcut", "combinatorial"]
        ))
        
        # Test 5: Circuit depth optimization
        def depth_vs_accuracy_cost(params):
            """Cost function balancing circuit depth and accuracy."""
            # Mock implementation
            depth_penalty = len(params) * 0.1  # Penalty for circuit depth
            accuracy_term = np.sum(np.sin(params)**2)  # Accuracy term
            return depth_penalty + (1 - accuracy_term / len(params))
        
        tests.append(CircuitOptimizationTest(
            name="depth_accuracy_tradeoff",
            circuit_function=depth_vs_accuracy_cost,
            initial_parameters=np.random.rand(8) * 2 * np.pi,
            target_cost=0.2,
            max_iterations=100,
            optimization_method="adam",
            tolerance=1e-4,
            description="Optimize trade-off between circuit depth and accuracy",
            tags=["depth", "accuracy", "tradeoff"]
        ))
        
        return tests
    
    def run_optimization_tests(self, tests: Optional[List[CircuitOptimizationTest]] = None) -> List[OptimizationResult]:
        """
        Run circuit optimization tests.
        
        Args:
            tests: Specific tests to run (uses standard if None)
            
        Returns:
            List of optimization results
        """
        if tests is None:
            tests = self.generate_standard_optimization_tests()
        
        results = []
        
        for test in tests:
            self.logger.info(f"Running optimization test: {test.name}")
            
            start_time = time.time()
            
            try:
                # Run optimization
                result = self._run_optimization(test)
                result.optimization_time_ms = (time.time() - start_time) * 1000
                
                # Check if target achieved
                result.passed = (result.final_cost <= test.target_cost and 
                               result.convergence_achieved)
                
                results.append(result)
                
                if result.passed:
                    self.logger.info(f"✓ {test.name}: cost {result.final_cost:.4f} -> {test.target_cost:.4f}")
                else:
                    self.logger.warning(f"✗ {test.name}: cost {result.final_cost:.4f} (target: {test.target_cost:.4f})")
                
            except Exception as e:
                optimization_time_ms = (time.time() - start_time) * 1000
                
                result = OptimizationResult(
                    test_case=test,
                    final_cost=float('inf'),
                    initial_cost=float('inf'),
                    cost_improvement=0.0,
                    iterations_taken=0,
                    convergence_achieved=False,
                    optimization_time_ms=optimization_time_ms,
                    final_parameters=test.initial_parameters,
                    passed=False,
                    error_message=str(e)
                )
                
                results.append(result)
                self.logger.error(f"✗ {test.name}: Exception - {e}")
        
        return results
    
    def _run_optimization(self, test: CircuitOptimizationTest) -> OptimizationResult:
        """Run optimization algorithm for a test case."""
        params = test.initial_parameters.copy()
        initial_cost = test.circuit_function(params)
        
        cost_history = [initial_cost]
        learning_rate = 0.01
        
        # Simple gradient descent optimization
        for iteration in range(test.max_iterations):
            # Numerical gradient estimation
            gradient = self._estimate_gradient(test.circuit_function, params)
            
            # Update parameters
            params -= learning_rate * gradient
            
            # Evaluate cost
            current_cost = test.circuit_function(params)
            cost_history.append(current_cost)
            
            # Check convergence
            if len(cost_history) > 1:
                cost_change = abs(cost_history[-2] - cost_history[-1])
                if cost_change < test.tolerance:
                    convergence_achieved = True
                    break
        else:
            convergence_achieved = False
        
        final_cost = cost_history[-1]
        cost_improvement = initial_cost - final_cost
        
        return OptimizationResult(
            test_case=test,
            final_cost=final_cost,
            initial_cost=initial_cost,
            cost_improvement=cost_improvement,
            iterations_taken=len(cost_history) - 1,
            convergence_achieved=convergence_achieved,
            optimization_time_ms=0,  # Will be set by caller
            final_parameters=params,
            cost_history=cost_history
        )
    
    def _estimate_gradient(self, cost_function: Callable, params: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """Estimate gradient using finite differences."""
        gradient = np.zeros_like(params)
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            
            params_plus[i] += epsilon
            params_minus[i] -= epsilon
            
            cost_plus = cost_function(params_plus)
            cost_minus = cost_function(params_minus)
            
            gradient[i] = (cost_plus - cost_minus) / (2 * epsilon)
        
        return gradient


class QuantumCircuitValidator:
    """
    Validates quantum circuit properties and correctness.
    """
    
    def __init__(self):
        self.validation_tests: List[CircuitValidationTest] = []
        self.logger = logger
    
    def add_validation_test(self, test: CircuitValidationTest) -> None:
        """Add a circuit validation test."""
        self.validation_tests.append(test)
    
    def generate_standard_validation_tests(self) -> List[CircuitValidationTest]:
        """Generate standard circuit validation tests."""
        tests = []
        
        # Test 1: Unitarity check
        def mock_unitary_circuit(input_state):
            """Mock unitary circuit operation."""
            # Simple rotation circuit
            theta = np.pi / 4
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            return rotation_matrix @ input_state
        
        def check_unitarity(circuit_output, original_norm):
            """Check if circuit preserves norm (unitarity)."""
            output_norm = np.linalg.norm(circuit_output)
            return abs(output_norm - original_norm) < 1e-10
        
        test_inputs = [
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([1, 1]) / np.sqrt(2)
        ]
        
        tests.append(CircuitValidationTest(
            name="unitarity_preservation",
            circuit_function=mock_unitary_circuit,
            test_inputs=test_inputs,
            expected_properties={"preserves_norm": True},
            validation_rules=[
                lambda out, inp: check_unitarity(out, np.linalg.norm(inp))
            ],
            description="Validate that quantum circuits preserve state norm",
            tags=["unitarity", "norm", "basic"]
        ))
        
        # Test 2: Quantum gate fidelity
        def pauli_x_gate(input_state):
            """Pauli-X gate implementation."""
            pauli_x = np.array([[0, 1], [1, 0]])
            return pauli_x @ input_state
        
        def check_pauli_x_action(circuit_output, input_state):
            """Check correct Pauli-X action."""
            if np.allclose(input_state, [1, 0]):
                return np.allclose(circuit_output, [0, 1])
            elif np.allclose(input_state, [0, 1]):
                return np.allclose(circuit_output, [1, 0])
            else:
                # For superposition states, check correct transformation
                expected = np.array([input_state[1], input_state[0]])
                return np.allclose(circuit_output, expected)
        
        tests.append(CircuitValidationTest(
            name="pauli_x_gate_validation",
            circuit_function=pauli_x_gate,
            test_inputs=test_inputs,
            expected_properties={"correct_pauli_x": True},
            validation_rules=[
                lambda out, inp: check_pauli_x_action(out, inp)
            ],
            description="Validate Pauli-X gate implementation",
            tags=["pauli", "gates", "single_qubit"]
        ))
        
        # Test 3: Hadamard gate validation
        def hadamard_gate(input_state):
            """Hadamard gate implementation."""
            hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            return hadamard @ input_state
        
        def check_hadamard_superposition(circuit_output, input_state):
            """Check Hadamard creates correct superposition."""
            if np.allclose(input_state, [1, 0]):
                expected = np.array([1, 1]) / np.sqrt(2)
                return np.allclose(circuit_output, expected)
            elif np.allclose(input_state, [0, 1]):
                expected = np.array([1, -1]) / np.sqrt(2)
                return np.allclose(circuit_output, expected)
            return True  # Don't check complex superposition cases
        
        tests.append(CircuitValidationTest(
            name="hadamard_gate_validation",
            circuit_function=hadamard_gate,
            test_inputs=[np.array([1, 0]), np.array([0, 1])],
            expected_properties={"creates_superposition": True},
            validation_rules=[
                lambda out, inp: check_hadamard_superposition(out, inp)
            ],
            description="Validate Hadamard gate creates correct superposition",
            tags=["hadamard", "superposition", "gates"]
        ))
        
        # Test 4: Controlled gate validation
        def cnot_gate(input_state):
            """CNOT gate for 2-qubit system."""
            # Input should be 4-dimensional for 2 qubits
            if len(input_state) != 4:
                # Pad or truncate to 4 dimensions
                padded = np.zeros(4)
                padded[:min(len(input_state), 4)] = input_state[:4]
                input_state = padded
            
            cnot = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ])
            return cnot @ input_state
        
        def check_cnot_logic(circuit_output, input_state):
            """Check CNOT gate logic."""
            # Basic check for valid quantum state
            return abs(np.linalg.norm(circuit_output) - 1) < 1e-10
        
        two_qubit_inputs = [
            np.array([1, 0, 0, 0]),  # |00⟩
            np.array([0, 1, 0, 0]),  # |01⟩
            np.array([0, 0, 1, 0]),  # |10⟩
            np.array([0, 0, 0, 1])   # |11⟩
        ]
        
        tests.append(CircuitValidationTest(
            name="cnot_gate_validation",
            circuit_function=cnot_gate,
            test_inputs=two_qubit_inputs,
            expected_properties={"entangling": True},
            validation_rules=[
                lambda out, inp: check_cnot_logic(out, inp)
            ],
            description="Validate CNOT gate implementation",
            tags=["cnot", "entangling", "two_qubit"]
        ))
        
        return tests
    
    def run_validation_tests(self, tests: Optional[List[CircuitValidationTest]] = None) -> List[QuantumTestResult]:
        """
        Run circuit validation tests.
        
        Args:
            tests: Specific tests to run (uses standard if None)
            
        Returns:
            List of validation results
        """
        if tests is None:
            tests = self.generate_standard_validation_tests()
        
        results = []
        
        for test in tests:
            self.logger.info(f"Running validation test: {test.name}")
            
            start_time = time.time()
            
            try:
                all_passed = True
                validation_details = []
                
                # Test each input
                for i, test_input in enumerate(test.test_inputs):
                    input_copy = test_input.copy()
                    circuit_output = test.circuit_function(input_copy)
                    
                    # Run validation rules
                    for j, rule in enumerate(test.validation_rules):
                        rule_passed = rule(circuit_output, input_copy)
                        validation_details.append({
                            "input_index": i,
                            "rule_index": j,
                            "passed": rule_passed
                        })
                        
                        if not rule_passed:
                            all_passed = False
                
                execution_time_ms = (time.time() - start_time) * 1000
                
                # Create quantum test case for result
                quantum_test_case = QuantumTestCase(
                    name=test.name,
                    test_type=None,  # Validation test
                    test_function=test.circuit_function,
                    description=test.description,
                    tags=test.tags
                )
                
                result = QuantumTestResult(
                    test_case=quantum_test_case,
                    passed=all_passed,
                    actual_result=validation_details,
                    expected_result=test.expected_properties,
                    error_tolerance=0.0,
                    execution_time_ms=execution_time_ms,
                    quantum_metrics={
                        "validation_details": validation_details,
                        "total_inputs_tested": len(test.test_inputs),
                        "total_rules_tested": len(test.validation_rules) * len(test.test_inputs),
                        "passed_rules": sum(1 for detail in validation_details if detail["passed"])
                    }
                )
                
                results.append(result)
                
                if all_passed:
                    self.logger.info(f"✓ {test.name}: All validation rules passed")
                else:
                    failed_count = sum(1 for detail in validation_details if not detail["passed"])
                    self.logger.warning(f"✗ {test.name}: {failed_count} validation rules failed")
                
            except Exception as e:
                execution_time_ms = (time.time() - start_time) * 1000
                
                quantum_test_case = QuantumTestCase(
                    name=test.name,
                    test_type=None,
                    test_function=test.circuit_function,
                    description=test.description
                )
                
                result = QuantumTestResult(
                    test_case=quantum_test_case,
                    passed=False,
                    actual_result=None,
                    expected_result=test.expected_properties,
                    error_tolerance=float('inf'),
                    execution_time_ms=execution_time_ms,
                    error_message=str(e)
                )
                
                results.append(result)
                self.logger.error(f"✗ {test.name}: Exception - {e}")
        
        return results
    
    def generate_validation_report(self, results: List[QuantumTestResult]) -> Dict[str, Any]:
        """Generate comprehensive validation test report."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = total_tests - passed_tests
        
        # Calculate detailed statistics
        total_rules_tested = sum(
            r.quantum_metrics.get("total_rules_tested", 0) 
            for r in results if r.quantum_metrics
        )
        
        total_rules_passed = sum(
            r.quantum_metrics.get("passed_rules", 0)
            for r in results if r.quantum_metrics
        )
        
        rule_pass_rate = total_rules_passed / total_rules_tested if total_rules_tested > 0 else 0
        
        if results:
            avg_time = np.mean([r.execution_time_ms for r in results])
        else:
            avg_time = 0.0
        
        return {
            "total_validation_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "test_pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "rule_statistics": {
                "total_rules_tested": total_rules_tested,
                "total_rules_passed": total_rules_passed,
                "rule_pass_rate": rule_pass_rate
            },
            "performance_statistics": {
                "average_test_time_ms": avg_time,
                "total_validation_time_ms": sum(r.execution_time_ms for r in results)
            },
            "failed_tests": [
                {
                    "test_name": r.test_case.name,
                    "error": r.error_message or "Validation rules failed",
                    "failed_rules": len(r.quantum_metrics.get("validation_details", [])) - r.quantum_metrics.get("passed_rules", 0) if r.quantum_metrics else 0
                }
                for r in results if not r.passed
            ]
        }
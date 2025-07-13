"""
Quantum-specific testing framework for QuantumRerank.

This module provides specialized testing capabilities for quantum computations,
including fidelity accuracy testing, circuit consistency validation, and
quantum-classical integration testing.
"""

from .quantum_test_framework import (
    QuantumTestFramework,
    FidelityTestCase,
    QuantumTestCase,
    QuantumTestResult
)
from .test_fidelity_computation import (
    FidelityAccuracyTester,
    FidelityBenchmarkSuite
)
from .test_circuit_optimization import (
    CircuitOptimizationTester,
    QuantumCircuitValidator
)
from .test_parameter_prediction import (
    ParameterPredictionTester,
    QuantumParameterValidator
)

__all__ = [
    "QuantumTestFramework",
    "FidelityTestCase",
    "QuantumTestCase", 
    "QuantumTestResult",
    "FidelityAccuracyTester",
    "FidelityBenchmarkSuite",
    "CircuitOptimizationTester",
    "QuantumCircuitValidator",
    "ParameterPredictionTester",
    "QuantumParameterValidator"
]
"""
PyTorch Autograd Integration for Quantum Operations.

This module provides custom autograd functions that enable gradient computation
through quantum operations, implementing parameter-shift rules and quantum-classical
gradient bridges for hybrid quantum-classical training.

Implements critical autograd integration identified in Task Documentation Alignment Analysis.
"""

from .quantum_functions import (
    QuantumSimilarityFunction,
    QuantumParameterFunction,
    QuantumFidelityFunction
)
from .parameter_shift import (
    ParameterShiftGradient,
    QuantumGradientEstimator
)
from .gradient_bridge import (
    QuantumClassicalBridge,
    HybridGradientFlow
)

__all__ = [
    "QuantumSimilarityFunction",
    "QuantumParameterFunction", 
    "QuantumFidelityFunction",
    "ParameterShiftGradient",
    "QuantumGradientEstimator",
    "QuantumClassicalBridge",
    "HybridGradientFlow"
]
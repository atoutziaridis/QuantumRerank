"""
Hardware Acceleration Module for Quantum-Inspired RAG.

This module provides hardware acceleration capabilities for tensor network operations,
quantum-inspired computations, and production deployment optimization.

Components:
- FPGA/TPU optimization engines
- Custom tensor network kernels
- Performance profiling and monitoring
- Hardware-specific optimizations
"""

__version__ = "1.0.0"

from .tensor_acceleration import (
    TensorAccelerationEngine,
    AccelerationConfig,
    HardwareType
)

from .performance_profiler import (
    PerformanceProfiler,
    ProfilingConfig,
    ProfileMetrics
)

__all__ = [
    "TensorAccelerationEngine",
    "AccelerationConfig", 
    "HardwareType",
    "PerformanceProfiler",
    "ProfilingConfig",
    "ProfileMetrics"
]
"""
Adaptive Compression Module for Dynamic Resource Management.

This module provides adaptive compression capabilities that dynamically adjust
quality vs performance trade-offs based on real-time resource availability
and performance requirements.

Components:
- Resource-aware compression algorithms
- Dynamic quality adjustment
- Performance monitoring and feedback
- Adaptive optimization strategies
"""

__version__ = "1.0.0"

from .resource_aware_compressor import (
    ResourceAwareCompressor,
    CompressionConfig,
    CompressionLevel
)

from .dynamic_optimizer import (
    DynamicOptimizer,
    OptimizationConfig,
    OptimizationStrategy
)

from .resource_monitor import (
    ResourceMonitor,
    ResourceMetrics,
    ResourceThreshold
)

__all__ = [
    "ResourceAwareCompressor",
    "CompressionConfig",
    "CompressionLevel",
    "DynamicOptimizer",
    "OptimizationConfig", 
    "OptimizationStrategy",
    "ResourceMonitor",
    "ResourceMetrics",
    "ResourceThreshold"
]
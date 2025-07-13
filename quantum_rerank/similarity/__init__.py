"""
Multi-Method Similarity Engine for QuantumRerank.

This module provides a unified similarity engine supporting classical, quantum,
and hybrid similarity methods with intelligent method selection and performance optimization.
"""

from .multi_method_engine import (
    MultiMethodSimilarityEngine,
    SimilarityRequirements,
    SimilarityResult,
    ConsensusResult
)
from .method_selector import (
    MethodSelector,
    MethodSelectionContext,
    MethodProfile
)
from .optimizer import (
    PerformanceOptimizer,
    OptimizationStrategy,
    PerformanceBottleneck
)
from .aggregator import (
    ResultAggregator,
    AggregationStrategy,
    ConsensusAlgorithm
)
from .performance_monitor import (
    SimilarityPerformanceMonitor,
    MethodPerformanceStats,
    PerformanceAlert
)

__all__ = [
    # Main engine
    "MultiMethodSimilarityEngine",
    "SimilarityRequirements",
    "SimilarityResult",
    "ConsensusResult",
    
    # Method selection
    "MethodSelector",
    "MethodSelectionContext",
    "MethodProfile",
    
    # Optimization
    "PerformanceOptimizer",
    "OptimizationStrategy",
    "PerformanceBottleneck",
    
    # Aggregation
    "ResultAggregator",
    "AggregationStrategy",
    "ConsensusAlgorithm",
    
    # Monitoring
    "SimilarityPerformanceMonitor",
    "MethodPerformanceStats",
    "PerformanceAlert"
]
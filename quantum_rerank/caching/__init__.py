"""
Advanced Caching System for QuantumRerank.

This module provides intelligent multi-level caching for similarity computations,
quantum circuit results, and embedding processing to optimize performance.
"""

from .cache_manager import (
    AdvancedCacheManager,
    CacheLevel,
    CacheConfig
)
from .similarity_cache import (
    SimilarityCache,
    SimilarityResult,
    CacheKey
)
from .quantum_cache import (
    QuantumCircuitCache,
    QuantumResult,
    ParameterCluster
)
from .embedding_cache import (
    EmbeddingCache,
    EmbeddingResult,
    TextHash
)
from .cache_monitor import (
    CachePerformanceMonitor,
    CacheMetrics,
    CacheAlert
)
from .cache_optimizer import (
    CacheOptimizer,
    OptimizationStrategy,
    PerformanceAnalyzer
)
from .approximate_matcher import (
    ApproximateSimilarityMatcher,
    MatchingStrategy,
    SimilarityThreshold
)

__all__ = [
    # Main cache manager
    "AdvancedCacheManager",
    "CacheLevel",
    "CacheConfig",
    
    # Similarity caching
    "SimilarityCache",
    "SimilarityResult",
    "CacheKey",
    
    # Quantum caching
    "QuantumCircuitCache",
    "QuantumResult",
    "ParameterCluster",
    
    # Embedding caching
    "EmbeddingCache",
    "EmbeddingResult",
    "TextHash",
    
    # Monitoring
    "CachePerformanceMonitor",
    "CacheMetrics",
    "CacheAlert",
    
    # Optimization
    "CacheOptimizer",
    "OptimizationStrategy",
    "PerformanceAnalyzer",
    
    # Approximate matching
    "ApproximateSimilarityMatcher",
    "MatchingStrategy",
    "SimilarityThreshold"
]
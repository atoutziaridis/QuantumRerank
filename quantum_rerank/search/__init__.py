"""
Scalable vector search integration module.

This module provides high-performance vector search capabilities with
multiple backend support and quantum reranking integration.
"""

from .faiss_integration import AdvancedFAISSIntegration
from .multi_backend import MultiBackendSearchEngine
from .retrieval_pipeline import RetrievalPipeline, SearchResult
from .data_management import ScalableDataManager
from .optimization import PipelineOptimizer, SearchPerformanceTargets
from .performance_monitor import SearchPerformanceMonitor

__all__ = [
    "AdvancedFAISSIntegration",
    "MultiBackendSearchEngine", 
    "RetrievalPipeline",
    "SearchResult",
    "ScalableDataManager",
    "PipelineOptimizer",
    "SearchPerformanceTargets",
    "SearchPerformanceMonitor"
]
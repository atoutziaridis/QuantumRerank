"""
Multi-backend vector search engine with intelligent backend selection.

This module provides a unified interface for multiple vector search backends
with automatic selection based on dataset characteristics and performance requirements.
"""

import time
from typing import Dict, List, Optional, Any, Tuple, Type
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .backends.base_backend import BaseVectorSearchBackend, BackendCapabilities, SearchConfiguration
from .backends.faiss_backend import FAISSBackend
from ..utils import get_logger


class BackendType(Enum):
    """Supported vector search backends."""
    FAISS = "faiss"
    ANNOY = "annoy" 
    HNSWLIB = "hnswlib"
    ELASTICSEARCH = "elasticsearch"
    AUTO = "auto"


@dataclass
class BackendSelectionCriteria:
    """Criteria for backend selection."""
    collection_size: int
    dimension: int
    query_latency_target_ms: int = 100
    accuracy_requirement: float = 0.95
    memory_budget_mb: int = 1024
    concurrent_queries: int = 1
    use_gpu: bool = False


@dataclass
class BackendPerformanceScore:
    """Performance score for a backend."""
    backend_type: BackendType
    score: float
    latency_score: float
    accuracy_score: float
    memory_score: float
    compatibility_score: float
    reasons: List[str]


class MultiBackendSearchEngine:
    """
    Multi-backend vector search engine with intelligent selection.
    
    This engine automatically selects the optimal vector search backend
    based on dataset characteristics and performance requirements.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Backend registry
        self.backend_classes: Dict[BackendType, Type[BaseVectorSearchBackend]] = {
            BackendType.FAISS: FAISSBackend
        }
        
        # Active backends
        self.backends: Dict[BackendType, BaseVectorSearchBackend] = {}
        self.active_backend: Optional[BaseVectorSearchBackend] = None
        self.active_backend_type: Optional[BackendType] = None
        
        # Performance tracking
        self.backend_performance: Dict[BackendType, List[float]] = {}
        self.selection_history: List[Dict[str, Any]] = []
        
        # Backend configurations
        self.backend_configs = self._initialize_backend_configs()
        
        self.logger.info("Initialized MultiBackendSearchEngine")
    
    def select_optimal_backend(self, criteria: BackendSelectionCriteria) -> BackendType:
        """
        Select optimal backend based on selection criteria.
        
        Args:
            criteria: Selection criteria including size, latency, accuracy requirements
            
        Returns:
            Selected backend type
        """
        # Score all available backends
        backend_scores = []
        
        for backend_type in self.backend_classes.keys():
            score = self._score_backend(backend_type, criteria)
            backend_scores.append(score)
        
        # Select backend with highest score
        best_backend = max(backend_scores, key=lambda x: x.score)
        
        # Record selection decision
        selection_record = {
            "selected_backend": best_backend.backend_type,
            "selection_criteria": criteria.__dict__,
            "all_scores": [score.__dict__ for score in backend_scores],
            "timestamp": time.time()
        }
        self.selection_history.append(selection_record)
        
        self.logger.info(
            f"Selected backend: {best_backend.backend_type.value} "
            f"(score: {best_backend.score:.3f})"
        )
        
        return best_backend.backend_type
    
    def create_collection(self, collection_id: str,
                         embeddings: np.ndarray,
                         document_ids: List[str],
                         backend_type: Optional[BackendType] = None,
                         **kwargs) -> bool:
        """
        Create collection with specified or automatically selected backend.
        
        Args:
            collection_id: Unique collection identifier
            embeddings: Embedding vectors to index
            document_ids: Corresponding document identifiers  
            backend_type: Specific backend to use, or None for auto-selection
            **kwargs: Additional arguments for backend configuration
            
        Returns:
            Success status
        """
        try:
            # Auto-select backend if not specified
            if backend_type is None or backend_type == BackendType.AUTO:
                criteria = self._create_selection_criteria(embeddings, **kwargs)
                backend_type = self.select_optimal_backend(criteria)
            
            # Initialize backend if needed
            if backend_type not in self.backends:
                self._initialize_backend(backend_type)
            
            # Switch to selected backend
            self._switch_backend(backend_type)
            
            # Create collection
            start_time = time.time()
            success = self.active_backend.build_index(embeddings, document_ids)
            build_time_ms = (time.time() - start_time) * 1000
            
            if success:
                # Record performance
                self._record_backend_performance(backend_type, build_time_ms)
                
                self.logger.info(
                    f"Created collection {collection_id} with {backend_type.value} "
                    f"({len(document_ids)} documents, {build_time_ms:.1f}ms)"
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create collection {collection_id}: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray,
              top_k: int = 100,
              config: Optional[SearchConfiguration] = None) -> Optional[Tuple[List[str], List[float]]]:
        """
        Search using active backend.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            config: Search configuration
            
        Returns:
            Tuple of (document_ids, similarity_scores) or None if failed
        """
        if self.active_backend is None:
            self.logger.error("No active backend. Create collection first.")
            return None
        
        try:
            start_time = time.time()
            result = self.active_backend.search(query_embedding, top_k)
            search_time_ms = (time.time() - start_time) * 1000
            
            # Record performance
            self._record_backend_performance(self.active_backend_type, search_time_ms)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Search failed with {self.active_backend_type.value}: {e}")
            return None
    
    def batch_search(self, query_embeddings: np.ndarray,
                    top_k: int = 100) -> Optional[List[Tuple[List[str], List[float]]]]:
        """
        Perform batch search using active backend.
        
        Args:
            query_embeddings: Multiple query vectors
            top_k: Number of results per query
            
        Returns:
            List of (document_ids, similarity_scores) per query or None if failed
        """
        if self.active_backend is None:
            self.logger.error("No active backend. Create collection first.")
            return None
        
        try:
            start_time = time.time()
            results = self.active_backend.batch_search(query_embeddings, top_k)
            batch_time_ms = (time.time() - start_time) * 1000
            avg_time_ms = batch_time_ms / len(query_embeddings)
            
            # Record performance
            self._record_backend_performance(self.active_backend_type, avg_time_ms)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch search failed with {self.active_backend_type.value}: {e}")
            return None
    
    def add_embeddings(self, embeddings: np.ndarray,
                      document_ids: List[str]) -> bool:
        """Add embeddings to active backend."""
        if self.active_backend is None:
            self.logger.error("No active backend. Create collection first.")
            return False
        
        try:
            self.active_backend.add_embeddings(embeddings, document_ids)
            return True
        except Exception as e:
            self.logger.error(f"Failed to add embeddings: {e}")
            return False
    
    def optimize_backend(self) -> Dict[str, Any]:
        """Optimize active backend performance."""
        if self.active_backend is None:
            return {"optimized": False, "reason": "No active backend"}
        
        try:
            return self.active_backend.optimize_index()
        except Exception as e:
            self.logger.error(f"Backend optimization failed: {e}")
            return {"optimized": False, "error": str(e)}
    
    def switch_backend(self, backend_type: BackendType,
                      preserve_data: bool = True) -> bool:
        """
        Switch to different backend, optionally preserving data.
        
        Args:
            backend_type: Target backend type
            preserve_data: Whether to transfer existing data
            
        Returns:
            Success status
        """
        if backend_type == self.active_backend_type:
            return True  # Already using this backend
        
        try:
            # Initialize new backend if needed
            if backend_type not in self.backends:
                self._initialize_backend(backend_type)
            
            # TODO: Implement data migration if preserve_data is True
            # This would require extracting embeddings from current backend
            # and rebuilding index in new backend
            
            # Switch active backend
            self._switch_backend(backend_type)
            
            self.logger.info(f"Switched to backend: {backend_type.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to switch backend: {e}")
            return False
    
    def get_backend_comparison(self, criteria: BackendSelectionCriteria) -> List[BackendPerformanceScore]:
        """Get performance comparison for all backends."""
        scores = []
        for backend_type in self.backend_classes.keys():
            score = self._score_backend(backend_type, criteria)
            scores.append(score)
        
        # Sort by total score (descending)
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            "active_backend": self.active_backend_type.value if self.active_backend_type else None,
            "backend_performance": {},
            "selection_history": self.selection_history[-10:],  # Last 10 selections
            "backend_capabilities": {}
        }
        
        # Add performance data for each backend
        for backend_type, times in self.backend_performance.items():
            if times:
                stats["backend_performance"][backend_type.value] = {
                    "average_time_ms": sum(times) / len(times),
                    "min_time_ms": min(times),
                    "max_time_ms": max(times),
                    "query_count": len(times)
                }
        
        # Add capabilities for initialized backends
        for backend_type, backend in self.backends.items():
            capabilities = backend.get_capabilities()
            stats["backend_capabilities"][backend_type.value] = capabilities.__dict__
        
        return stats
    
    def _score_backend(self, backend_type: BackendType,
                      criteria: BackendSelectionCriteria) -> BackendPerformanceScore:
        """Score a backend based on selection criteria."""
        # Get backend configuration
        backend_config = self.backend_configs.get(backend_type, {})
        
        reasons = []
        
        # Score different aspects
        latency_score = self._score_latency(backend_type, criteria, reasons)
        accuracy_score = self._score_accuracy(backend_type, criteria, reasons)
        memory_score = self._score_memory(backend_type, criteria, reasons)
        compatibility_score = self._score_compatibility(backend_type, criteria, reasons)
        
        # Calculate weighted total score
        weights = {
            "latency": 0.3,
            "accuracy": 0.25,
            "memory": 0.25,
            "compatibility": 0.2
        }
        
        total_score = (
            weights["latency"] * latency_score +
            weights["accuracy"] * accuracy_score +
            weights["memory"] * memory_score +
            weights["compatibility"] * compatibility_score
        )
        
        return BackendPerformanceScore(
            backend_type=backend_type,
            score=total_score,
            latency_score=latency_score,
            accuracy_score=accuracy_score,
            memory_score=memory_score,
            compatibility_score=compatibility_score,
            reasons=reasons
        )
    
    def _score_latency(self, backend_type: BackendType,
                      criteria: BackendSelectionCriteria,
                      reasons: List[str]) -> float:
        """Score backend latency performance."""
        backend_config = self.backend_configs.get(backend_type, {})
        expected_latency = backend_config.get("expected_latency_ms", 100)
        
        # Score based on how well it meets target
        if expected_latency <= criteria.query_latency_target_ms:
            score = 1.0
            reasons.append(f"Meets latency target ({expected_latency}ms <= {criteria.query_latency_target_ms}ms)")
        else:
            # Penalize for exceeding target
            ratio = criteria.query_latency_target_ms / expected_latency
            score = max(0.0, ratio)
            reasons.append(f"Exceeds latency target ({expected_latency}ms > {criteria.query_latency_target_ms}ms)")
        
        return score
    
    def _score_accuracy(self, backend_type: BackendType,
                       criteria: BackendSelectionCriteria,
                       reasons: List[str]) -> float:
        """Score backend accuracy performance."""
        backend_config = self.backend_configs.get(backend_type, {})
        expected_accuracy = backend_config.get("expected_accuracy", 0.95)
        
        if expected_accuracy >= criteria.accuracy_requirement:
            score = 1.0
            reasons.append(f"Meets accuracy requirement ({expected_accuracy:.2f} >= {criteria.accuracy_requirement:.2f})")
        else:
            # Penalize for not meeting requirement
            score = expected_accuracy / criteria.accuracy_requirement
            reasons.append(f"Below accuracy requirement ({expected_accuracy:.2f} < {criteria.accuracy_requirement:.2f})")
        
        return score
    
    def _score_memory(self, backend_type: BackendType,
                     criteria: BackendSelectionCriteria,
                     reasons: List[str]) -> float:
        """Score backend memory usage."""
        backend_config = self.backend_configs.get(backend_type, {})
        memory_factor = backend_config.get("memory_factor", 1.0)
        
        # Estimate memory usage
        estimated_memory_mb = criteria.collection_size * criteria.dimension * 4 * memory_factor / (1024 * 1024)
        
        if estimated_memory_mb <= criteria.memory_budget_mb:
            score = 1.0
            reasons.append(f"Within memory budget ({estimated_memory_mb:.0f}MB <= {criteria.memory_budget_mb}MB)")
        else:
            # Penalize for exceeding budget
            score = max(0.0, criteria.memory_budget_mb / estimated_memory_mb)
            reasons.append(f"Exceeds memory budget ({estimated_memory_mb:.0f}MB > {criteria.memory_budget_mb}MB)")
        
        return score
    
    def _score_compatibility(self, backend_type: BackendType,
                            criteria: BackendSelectionCriteria,
                            reasons: List[str]) -> float:
        """Score backend compatibility with requirements."""
        backend_config = self.backend_configs.get(backend_type, {})
        
        score = 1.0
        
        # Check GPU requirement
        if criteria.use_gpu and not backend_config.get("supports_gpu", False):
            score *= 0.5
            reasons.append("GPU not supported")
        
        # Check collection size limits
        max_size = backend_config.get("max_collection_size", float('inf'))
        if criteria.collection_size > max_size:
            score *= 0.3
            reasons.append(f"Collection size exceeds limit ({criteria.collection_size} > {max_size})")
        
        # Check dimension limits  
        max_dimension = backend_config.get("max_dimension", float('inf'))
        if criteria.dimension > max_dimension:
            score *= 0.1
            reasons.append(f"Dimension exceeds limit ({criteria.dimension} > {max_dimension})")
        
        if score == 1.0:
            reasons.append("Fully compatible with requirements")
        
        return score
    
    def _initialize_backend(self, backend_type: BackendType) -> None:
        """Initialize a specific backend."""
        if backend_type not in self.backend_classes:
            raise ValueError(f"Unsupported backend type: {backend_type}")
        
        # Get backend configuration
        backend_config = self.config.get(backend_type.value, {})
        
        # Create backend instance
        backend_class = self.backend_classes[backend_type]
        backend = backend_class(backend_config)
        
        self.backends[backend_type] = backend
        self.backend_performance[backend_type] = []
        
        self.logger.info(f"Initialized backend: {backend_type.value}")
    
    def _switch_backend(self, backend_type: BackendType) -> None:
        """Switch to specified backend."""
        if backend_type not in self.backends:
            raise ValueError(f"Backend not initialized: {backend_type}")
        
        self.active_backend = self.backends[backend_type]
        self.active_backend_type = backend_type
    
    def _create_selection_criteria(self, embeddings: np.ndarray, **kwargs) -> BackendSelectionCriteria:
        """Create selection criteria from embeddings and arguments."""
        return BackendSelectionCriteria(
            collection_size=embeddings.shape[0],
            dimension=embeddings.shape[1],
            query_latency_target_ms=kwargs.get("query_latency_target_ms", 100),
            accuracy_requirement=kwargs.get("accuracy_requirement", 0.95),
            memory_budget_mb=kwargs.get("memory_budget_mb", 1024),
            concurrent_queries=kwargs.get("concurrent_queries", 1),
            use_gpu=kwargs.get("use_gpu", False)
        )
    
    def _record_backend_performance(self, backend_type: BackendType, time_ms: float) -> None:
        """Record performance measurement for backend."""
        if backend_type in self.backend_performance:
            self.backend_performance[backend_type].append(time_ms)
            
            # Keep only recent measurements
            if len(self.backend_performance[backend_type]) > 1000:
                self.backend_performance[backend_type] = self.backend_performance[backend_type][-1000:]
    
    def _initialize_backend_configs(self) -> Dict[BackendType, Dict[str, Any]]:
        """Initialize default backend configurations."""
        return {
            BackendType.FAISS: {
                "expected_latency_ms": 50,
                "expected_accuracy": 0.95,
                "memory_factor": 1.1,
                "supports_gpu": True,
                "max_collection_size": 1000000000,
                "max_dimension": 65536
            }
        }


__all__ = [
    "BackendType",
    "BackendSelectionCriteria",
    "BackendPerformanceScore",
    "MultiBackendSearchEngine"
]
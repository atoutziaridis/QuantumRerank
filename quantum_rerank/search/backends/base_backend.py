"""
Base class for vector search backends.

This module defines the interface that all vector search backends must implement
for integration with the QuantumRerank system.
"""

import time
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from ...utils import get_logger


@dataclass
class SearchConfiguration:
    """Configuration for vector search operations."""
    top_k: int = 100
    accuracy_requirement: float = 0.95
    latency_budget_ms: int = 500
    use_cache: bool = True
    batch_size: int = 32


@dataclass 
class BackendCapabilities:
    """Capabilities and limitations of a search backend."""
    max_dimension: int
    max_collection_size: int
    supports_gpu: bool = False
    supports_batch_search: bool = True
    supports_incremental_update: bool = True
    memory_efficiency_score: float = 0.8
    latency_score: float = 0.8
    accuracy_score: float = 0.9


@dataclass
class SearchMetrics:
    """Performance metrics for search operations."""
    query_time_ms: float
    index_memory_mb: float
    query_throughput_qps: float
    accuracy_score: float
    backend_name: str
    timestamp: float


class BaseVectorSearchBackend(ABC):
    """
    Abstract base class for vector search backends.
    
    All vector search implementations must inherit from this class
    and implement the required methods for consistency across backends.
    """
    
    def __init__(self, backend_name: str, config: Dict[str, Any]):
        self.backend_name = backend_name
        self.config = config
        self.logger = get_logger(f"{__name__}.{backend_name}")
        
        # Performance tracking
        self.query_count = 0
        self.total_query_time_ms = 0.0
        self.last_metrics: Optional[SearchMetrics] = None
        
        self.logger.info(f"Initialized {backend_name} backend")
    
    @abstractmethod
    def build_index(self, embeddings: np.ndarray, 
                   document_ids: List[str]) -> None:
        """
        Build search index from embeddings.
        
        Args:
            embeddings: Embedding vectors (n_docs, dimension)
            document_ids: Corresponding document identifiers
        """
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, 
              top_k: int = 100) -> Tuple[List[str], List[float]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            Tuple of (document_ids, similarity_scores)
        """
        pass
    
    @abstractmethod
    def add_embeddings(self, embeddings: np.ndarray, 
                      document_ids: List[str]) -> None:
        """
        Add new embeddings to existing index.
        
        Args:
            embeddings: New embedding vectors
            document_ids: Corresponding document identifiers
        """
        pass
    
    @abstractmethod
    def remove_embeddings(self, document_ids: List[str]) -> None:
        """
        Remove embeddings from index.
        
        Args:
            document_ids: Document identifiers to remove
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> BackendCapabilities:
        """Get backend capabilities and limitations."""
        pass
    
    @abstractmethod
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        pass
    
    def batch_search(self, query_embeddings: np.ndarray,
                    top_k: int = 100) -> List[Tuple[List[str], List[float]]]:
        """
        Perform batch search for multiple queries.
        
        Args:
            query_embeddings: Multiple query vectors (n_queries, dimension)
            top_k: Number of results per query
            
        Returns:
            List of (document_ids, similarity_scores) for each query
        """
        results = []
        for query_embedding in query_embeddings:
            result = self.search(query_embedding, top_k)
            results.append(result)
        return results
    
    def optimize_index(self) -> Dict[str, Any]:
        """
        Optimize index performance.
        
        Returns:
            Optimization results and statistics
        """
        # Default implementation - backends can override
        return {"optimized": False, "reason": "No optimization available"}
    
    def save_index(self, filepath: str) -> bool:
        """
        Save index to disk.
        
        Args:
            filepath: Path to save index
            
        Returns:
            Success status
        """
        # Default implementation - backends can override
        self.logger.warning(f"Save not implemented for {self.backend_name}")
        return False
    
    def load_index(self, filepath: str) -> bool:
        """
        Load index from disk.
        
        Args:
            filepath: Path to load index from
            
        Returns:
            Success status
        """
        # Default implementation - backends can override
        self.logger.warning(f"Load not implemented for {self.backend_name}")
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get backend performance statistics."""
        avg_query_time = (
            self.total_query_time_ms / max(self.query_count, 1)
        )
        
        return {
            "backend_name": self.backend_name,
            "query_count": self.query_count,
            "average_query_time_ms": avg_query_time,
            "memory_usage_mb": self.get_memory_usage_mb(),
            "capabilities": self.get_capabilities().__dict__,
            "last_metrics": self.last_metrics.__dict__ if self.last_metrics else None
        }
    
    def _record_query_metrics(self, query_time_ms: float, 
                             accuracy_score: float = 1.0) -> None:
        """Record performance metrics for a query."""
        self.query_count += 1
        self.total_query_time_ms += query_time_ms
        
        # Estimate throughput
        throughput_qps = 1000.0 / max(query_time_ms, 0.1)
        
        self.last_metrics = SearchMetrics(
            query_time_ms=query_time_ms,
            index_memory_mb=self.get_memory_usage_mb(),
            query_throughput_qps=throughput_qps,
            accuracy_score=accuracy_score,
            backend_name=self.backend_name,
            timestamp=time.time()
        )
    
    def validate_embeddings(self, embeddings: np.ndarray) -> bool:
        """Validate embedding format and dimensions."""
        if not isinstance(embeddings, np.ndarray):
            self.logger.error("Embeddings must be numpy array")
            return False
        
        if embeddings.ndim != 2:
            self.logger.error(f"Embeddings must be 2D, got {embeddings.ndim}D")
            return False
        
        capabilities = self.get_capabilities()
        if embeddings.shape[1] > capabilities.max_dimension:
            self.logger.error(
                f"Embedding dimension {embeddings.shape[1]} exceeds maximum {capabilities.max_dimension}"
            )
            return False
        
        if embeddings.shape[0] > capabilities.max_collection_size:
            self.logger.error(
                f"Collection size {embeddings.shape[0]} exceeds maximum {capabilities.max_collection_size}"
            )
            return False
        
        return True
    
    def __str__(self) -> str:
        return f"{self.backend_name}Backend(queries={self.query_count})"
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(backend_name='{self.backend_name}', "
                f"query_count={self.query_count})")


__all__ = [
    "SearchConfiguration",
    "BackendCapabilities", 
    "SearchMetrics",
    "BaseVectorSearchBackend"
]
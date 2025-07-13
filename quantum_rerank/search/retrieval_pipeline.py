"""
Retrieval-rerank pipeline for scalable two-stage search architecture.

This module implements the complete retrieval-then-rerank pipeline that combines
fast vector search with quantum reranking for optimal search quality.
"""

import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

from .multi_backend import MultiBackendSearchEngine, BackendType
from ..similarity.multi_method_engine import MultiMethodSimilarityEngine
from ..caching.cache_manager import AdvancedCacheManager
from ..utils import get_logger


@dataclass
class Document:
    """Document representation for search results."""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class SearchRequest:
    """Search request configuration."""
    query: str
    query_embedding: Optional[np.ndarray] = None
    collection_id: str = "default"
    top_k: int = 10
    retrieval_size: Optional[int] = None  # Auto-calculated if None
    rerank_method: str = "hybrid"
    accuracy_requirement: float = 0.95
    latency_budget_ms: int = 500
    use_cache: bool = True


@dataclass
class SearchResult:
    """Search result with comprehensive metadata."""
    query: str
    documents: List[Document]
    retrieval_time_ms: float
    rerank_time_ms: float
    total_time_ms: float
    retrieval_size: int
    cache_hits: Dict[str, int]
    performance_metrics: Dict[str, Any]
    backend_used: str
    rerank_method: str


@dataclass
class PipelineConfiguration:
    """Configuration for retrieval-rerank pipeline."""
    # Retrieval configuration
    default_retrieval_multiplier: float = 3.0
    max_retrieval_size: int = 1000
    min_retrieval_size: int = 50
    
    # Reranking configuration
    enable_reranking: bool = True
    batch_rerank_size: int = 100
    rerank_timeout_ms: int = 2000
    
    # Performance configuration
    enable_caching: bool = True
    cache_retrieval_results: bool = True
    cache_rerank_results: bool = True
    
    # Optimization configuration
    enable_adaptive_retrieval: bool = True
    optimize_retrieval_size: bool = True
    monitor_performance: bool = True


class RetrievalPipeline:
    """
    Complete retrieval-rerank pipeline for scalable search.
    
    This pipeline implements a two-stage architecture:
    1. Fast vector search for initial retrieval
    2. Quantum reranking for optimal result quality
    """
    
    def __init__(self, config: Optional[PipelineConfiguration] = None):
        self.config = config or PipelineConfiguration()
        self.logger = get_logger(__name__)
        
        # Core components
        self.search_engine = MultiBackendSearchEngine()
        self.similarity_engine = MultiMethodSimilarityEngine()
        self.cache_manager = AdvancedCacheManager() if self.config.enable_caching else None
        
        # Document storage
        self.document_stores: Dict[str, Dict[str, Document]] = {}
        
        # Performance tracking
        self.pipeline_metrics = {
            "total_queries": 0,
            "total_retrieval_time_ms": 0.0,
            "total_rerank_time_ms": 0.0,
            "cache_hit_rate": 0.0,
            "average_retrieval_size": 0.0
        }
        
        self.optimization_history: List[Dict[str, Any]] = []
        
        self.logger.info("Initialized RetrievalPipeline")
    
    def create_collection(self, collection_id: str,
                         documents: List[Document],
                         backend_type: Optional[BackendType] = None) -> bool:
        """
        Create searchable collection from documents.
        
        Args:
            collection_id: Unique collection identifier
            documents: List of documents to index
            backend_type: Specific backend to use, or None for auto-selection
            
        Returns:
            Success status
        """
        try:
            # Store documents
            self.document_stores[collection_id] = {doc.id: doc for doc in documents}
            
            # Extract embeddings and IDs
            embeddings = []
            document_ids = []
            
            for doc in documents:
                if doc.embedding is not None:
                    embeddings.append(doc.embedding)
                    document_ids.append(doc.id)
                else:
                    self.logger.warning(f"Document {doc.id} has no embedding, skipping")
            
            if not embeddings:
                raise ValueError("No documents with embeddings found")
            
            embeddings_array = np.array(embeddings)
            
            # Create search index
            success = self.search_engine.create_collection(
                collection_id, embeddings_array, document_ids, backend_type
            )
            
            if success:
                self.logger.info(f"Created collection {collection_id} with {len(documents)} documents")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to create collection {collection_id}: {e}")
            return False
    
    def search(self, request: SearchRequest) -> Optional[SearchResult]:
        """
        Perform complete search and rerank operation.
        
        Args:
            request: Search request configuration
            
        Returns:
            Search result with ranked documents or None if failed
        """
        start_time = time.time()
        
        try:
            # Update metrics
            self.pipeline_metrics["total_queries"] += 1
            
            # Prepare query embedding if not provided
            if request.query_embedding is None:
                # TODO: Implement query embedding generation
                self.logger.warning("Query embedding generation not implemented")
                return None
            
            # Stage 1: Initial retrieval
            retrieval_result = self._perform_retrieval(request)
            if retrieval_result is None:
                return None
            
            candidate_docs, retrieval_time_ms, cache_hits = retrieval_result
            
            # Stage 2: Quantum reranking (if enabled)
            if self.config.enable_reranking and len(candidate_docs) > 1:
                rerank_result = self._perform_reranking(request, candidate_docs)
                if rerank_result is not None:
                    ranked_docs, rerank_time_ms = rerank_result
                else:
                    ranked_docs, rerank_time_ms = candidate_docs, 0.0
            else:
                ranked_docs, rerank_time_ms = candidate_docs, 0.0
            
            # Select top-k results
            final_docs = ranked_docs[:request.top_k]
            
            # Calculate total time
            total_time_ms = (time.time() - start_time) * 1000
            
            # Update performance metrics
            self._update_pipeline_metrics(retrieval_time_ms, rerank_time_ms, len(candidate_docs))
            
            # Create result
            result = SearchResult(
                query=request.query,
                documents=final_docs,
                retrieval_time_ms=retrieval_time_ms,
                rerank_time_ms=rerank_time_ms,
                total_time_ms=total_time_ms,
                retrieval_size=len(candidate_docs),
                cache_hits=cache_hits,
                performance_metrics=self._get_current_metrics(),
                backend_used=str(self.search_engine.active_backend_type.value) if self.search_engine.active_backend_type else "unknown",
                rerank_method=request.rerank_method
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return None
    
    def batch_search(self, requests: List[SearchRequest]) -> List[Optional[SearchResult]]:
        """
        Perform batch search operations.
        
        Args:
            requests: List of search requests
            
        Returns:
            List of search results (may contain None for failed searches)
        """
        results = []
        
        # Group requests by collection for efficiency
        collection_groups: Dict[str, List[Tuple[int, SearchRequest]]] = {}
        for i, request in enumerate(requests):
            if request.collection_id not in collection_groups:
                collection_groups[request.collection_id] = []
            collection_groups[request.collection_id].append((i, request))
        
        # Process each collection group
        for collection_id, group_requests in collection_groups.items():
            collection_results = self._batch_search_collection(group_requests)
            results.extend(collection_results)
        
        # Sort results back to original order
        results.sort(key=lambda x: x[0] if x[0] is not None else float('inf'))
        return [result[1] for result in results]
    
    def optimize_pipeline(self, collection_id: str) -> Dict[str, Any]:
        """
        Optimize pipeline performance for a collection.
        
        Args:
            collection_id: Collection to optimize
            
        Returns:
            Optimization results
        """
        optimization_results = {}
        
        # Optimize backend
        backend_optimization = self.search_engine.optimize_backend()
        optimization_results["backend"] = backend_optimization
        
        # Optimize retrieval size if enabled
        if self.config.optimize_retrieval_size:
            retrieval_optimization = self._optimize_retrieval_size(collection_id)
            optimization_results["retrieval_size"] = retrieval_optimization
        
        # Optimize caching if enabled
        if self.cache_manager is not None:
            cache_optimization = self.cache_manager.optimize_performance()
            optimization_results["caching"] = cache_optimization
        
        # Record optimization
        optimization_record = {
            "collection_id": collection_id,
            "results": optimization_results,
            "timestamp": time.time()
        }
        self.optimization_history.append(optimization_record)
        
        return optimization_results
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        stats = {
            "pipeline_metrics": self.pipeline_metrics.copy(),
            "search_engine_stats": self.search_engine.get_performance_statistics(),
            "collections": list(self.document_stores.keys()),
            "recent_optimizations": self.optimization_history[-5:] if self.optimization_history else []
        }
        
        # Add cache statistics if available
        if self.cache_manager is not None:
            stats["cache_stats"] = self.cache_manager.get_performance_summary()
        
        # Add similarity engine stats
        stats["similarity_engine_stats"] = self.similarity_engine.get_performance_summary()
        
        return stats
    
    def _perform_retrieval(self, request: SearchRequest) -> Optional[Tuple[List[Document], float, Dict[str, int]]]:
        """Perform initial vector search retrieval."""
        start_time = time.time()
        cache_hits = {"retrieval": 0}
        
        # Calculate optimal retrieval size
        retrieval_size = self._calculate_retrieval_size(request)
        
        # Check cache first if enabled
        if self.cache_manager is not None and request.use_cache:
            cache_key = self._create_retrieval_cache_key(request, retrieval_size)
            cached_result = self.cache_manager.get_retrieval_cache(cache_key)
            
            if cached_result is not None:
                cache_hits["retrieval"] = 1
                retrieval_time_ms = (time.time() - start_time) * 1000
                return cached_result, retrieval_time_ms, cache_hits
        
        # Perform vector search
        search_result = self.search_engine.search(request.query_embedding, retrieval_size)
        
        if search_result is None:
            return None
        
        doc_ids, scores = search_result
        
        # Load documents
        candidate_docs = []
        if request.collection_id in self.document_stores:
            doc_store = self.document_stores[request.collection_id]
            
            for doc_id, score in zip(doc_ids, scores):
                if doc_id in doc_store:
                    doc = doc_store[doc_id]
                    doc.score = score
                    candidate_docs.append(doc)
        
        retrieval_time_ms = (time.time() - start_time) * 1000
        
        # Cache result if enabled
        if self.cache_manager is not None and request.use_cache:
            cache_key = self._create_retrieval_cache_key(request, retrieval_size)
            self.cache_manager.put_retrieval_cache(cache_key, candidate_docs)
        
        return candidate_docs, retrieval_time_ms, cache_hits
    
    def _perform_reranking(self, request: SearchRequest, 
                          candidates: List[Document]) -> Optional[Tuple[List[Document], float]]:
        """Perform quantum reranking of candidate documents."""
        start_time = time.time()
        
        try:
            # Prepare embeddings for reranking
            candidate_embeddings = []
            for doc in candidates:
                if doc.embedding is not None:
                    candidate_embeddings.append(doc.embedding)
                else:
                    self.logger.warning(f"Document {doc.id} missing embedding for reranking")
                    return None
            
            candidate_embeddings = np.array(candidate_embeddings)
            
            # Perform similarity computation with quantum reranking
            similarity_result = self.similarity_engine.compute_similarity(
                query_embedding=request.query_embedding,
                candidate_embeddings=candidate_embeddings,
                method=request.rerank_method
            )
            
            if similarity_result is None:
                return None
            
            # Update document scores with reranked similarities
            reranked_docs = []
            for i, doc in enumerate(candidates):
                if i < len(similarity_result.similarities):
                    doc.score = similarity_result.similarities[i]
                    reranked_docs.append(doc)
            
            # Sort by reranked scores
            reranked_docs.sort(key=lambda d: d.score, reverse=True)
            
            rerank_time_ms = (time.time() - start_time) * 1000
            return reranked_docs, rerank_time_ms
            
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            return None
    
    def _calculate_retrieval_size(self, request: SearchRequest) -> int:
        """Calculate optimal retrieval size for the request."""
        if request.retrieval_size is not None:
            return min(request.retrieval_size, self.config.max_retrieval_size)
        
        # Auto-calculate based on top_k and configuration
        base_size = max(
            request.top_k * self.config.default_retrieval_multiplier,
            self.config.min_retrieval_size
        )
        
        # Adjust based on accuracy requirement
        if request.accuracy_requirement > 0.98:
            multiplier = 1.5
        elif request.accuracy_requirement > 0.95:
            multiplier = 1.2
        else:
            multiplier = 1.0
        
        optimal_size = int(base_size * multiplier)
        
        return min(optimal_size, self.config.max_retrieval_size)
    
    def _optimize_retrieval_size(self, collection_id: str) -> Dict[str, Any]:
        """Optimize retrieval size based on performance history."""
        # Simple optimization: analyze recent performance
        if self.pipeline_metrics["total_queries"] < 10:
            return {"optimized": False, "reason": "Insufficient data"}
        
        # Calculate average metrics
        avg_retrieval_size = self.pipeline_metrics["average_retrieval_size"]
        avg_total_time = (
            self.pipeline_metrics["total_retrieval_time_ms"] + 
            self.pipeline_metrics["total_rerank_time_ms"]
        ) / self.pipeline_metrics["total_queries"]
        
        optimization_result = {
            "current_avg_retrieval_size": avg_retrieval_size,
            "current_avg_total_time_ms": avg_total_time,
            "optimized": False
        }
        
        # Suggest optimizations based on performance
        if avg_total_time > 1000:  # If average query time > 1 second
            # Suggest reducing retrieval size
            new_multiplier = max(2.0, self.config.default_retrieval_multiplier * 0.8)
            optimization_result["suggested_retrieval_multiplier"] = new_multiplier
            optimization_result["reason"] = "High latency detected"
            optimization_result["optimized"] = True
        
        return optimization_result
    
    def _batch_search_collection(self, requests: List[Tuple[int, SearchRequest]]) -> List[Tuple[int, Optional[SearchResult]]]:
        """Perform batch search for a single collection."""
        results = []
        
        # Extract query embeddings for batch retrieval
        query_embeddings = []
        for _, request in requests:
            if request.query_embedding is not None:
                query_embeddings.append(request.query_embedding)
            else:
                # Skip requests without embeddings
                results.append((_, None))
                continue
        
        if not query_embeddings:
            return results
        
        # Perform batch retrieval
        batch_retrieval_size = max(req.top_k * 3 for _, req in requests)
        batch_results = self.search_engine.batch_search(
            np.array(query_embeddings), batch_retrieval_size
        )
        
        if batch_results is None:
            return [(idx, None) for idx, _ in requests]
        
        # Process each result
        for i, (original_idx, request) in enumerate(requests):
            if i < len(batch_results):
                # Create individual search result from batch result
                doc_ids, scores = batch_results[i]
                
                # Load documents and create result
                # (Implementation similar to individual search)
                # ... (simplified for brevity)
                
                results.append((original_idx, None))  # Placeholder
            else:
                results.append((original_idx, None))
        
        return results
    
    def _update_pipeline_metrics(self, retrieval_time_ms: float, 
                                rerank_time_ms: float, retrieval_size: int) -> None:
        """Update pipeline performance metrics."""
        self.pipeline_metrics["total_retrieval_time_ms"] += retrieval_time_ms
        self.pipeline_metrics["total_rerank_time_ms"] += rerank_time_ms
        
        # Update average retrieval size (exponential moving average)
        current_avg = self.pipeline_metrics["average_retrieval_size"]
        if current_avg == 0:
            self.pipeline_metrics["average_retrieval_size"] = retrieval_size
        else:
            alpha = 0.1
            self.pipeline_metrics["average_retrieval_size"] = (
                alpha * retrieval_size + (1 - alpha) * current_avg
            )
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        total_queries = max(self.pipeline_metrics["total_queries"], 1)
        
        return {
            "avg_retrieval_time_ms": self.pipeline_metrics["total_retrieval_time_ms"] / total_queries,
            "avg_rerank_time_ms": self.pipeline_metrics["total_rerank_time_ms"] / total_queries,
            "avg_total_time_ms": (
                self.pipeline_metrics["total_retrieval_time_ms"] + 
                self.pipeline_metrics["total_rerank_time_ms"]
            ) / total_queries,
            "avg_retrieval_size": self.pipeline_metrics["average_retrieval_size"]
        }
    
    def _create_retrieval_cache_key(self, request: SearchRequest, retrieval_size: int) -> str:
        """Create cache key for retrieval results."""
        # Simple hash-based key
        key_components = [
            request.collection_id,
            str(retrieval_size),
            str(hash(request.query_embedding.tobytes()))[:16]
        ]
        return "|".join(key_components)


__all__ = [
    "Document",
    "SearchRequest", 
    "SearchResult",
    "PipelineConfiguration",
    "RetrievalPipeline"
]
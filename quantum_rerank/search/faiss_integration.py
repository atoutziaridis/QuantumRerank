"""
Advanced FAISS integration with enhanced features for scalable vector search.

This module provides comprehensive FAISS integration with automatic optimization,
multiple index types, and performance monitoring for large-scale search.
"""

import time
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

from .backends.faiss_backend import FAISSBackend
from .backends.base_backend import BackendCapabilities
from ..utils import get_logger


@dataclass
class FAISSConfiguration:
    """Configuration for FAISS integration."""
    index_type: str = "auto"  # auto, IndexFlatIP, IndexIVFFlat, IndexHNSWFlat, IndexIVFPQ
    use_gpu: bool = False
    nlist: int = 1024  # Number of clusters for IVF
    nprobe: int = 64   # Number of clusters to search
    ef_construction: int = 200  # HNSW construction parameter
    ef_search: int = 100        # HNSW search parameter
    m_hnsw: int = 32           # HNSW connectivity parameter
    enable_optimization: bool = True
    save_index: bool = True
    index_cache_dir: str = "./faiss_indexes"


@dataclass
class IndexPerformanceProfile:
    """Performance profile for FAISS index."""
    index_type: str
    build_time_ms: float
    search_time_ms: float
    memory_usage_mb: float
    accuracy_score: float
    collection_size: int
    dimension: int
    timestamp: float = field(default_factory=time.time)


class AdvancedFAISSIntegration:
    """
    Advanced FAISS integration with automatic optimization and monitoring.
    
    This class provides high-level FAISS operations with intelligent
    index selection, performance optimization, and comprehensive monitoring.
    """
    
    def __init__(self, config: Optional[FAISSConfiguration] = None):
        self.config = config or FAISSConfiguration()
        self.logger = get_logger(__name__)
        
        # Backend management
        self.backend: Optional[FAISSBackend] = None
        self.collection_metadata: Dict[str, Any] = {}
        
        # Performance tracking
        self.performance_profiles: List[IndexPerformanceProfile] = []
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Index management
        self.index_cache_enabled = self.config.save_index
        if self.index_cache_enabled:
            os.makedirs(self.config.index_cache_dir, exist_ok=True)
        
        self.logger.info("Initialized Advanced FAISS Integration")
    
    def create_collection(self, collection_id: str, 
                         embeddings: np.ndarray,
                         document_ids: List[str],
                         force_rebuild: bool = False) -> bool:
        """
        Create or load a FAISS collection.
        
        Args:
            collection_id: Unique identifier for the collection
            embeddings: Embedding vectors to index
            document_ids: Corresponding document identifiers
            force_rebuild: Force rebuild even if cached index exists
            
        Returns:
            Success status
        """
        try:
            # Check for cached index
            index_path = self._get_index_path(collection_id)
            
            if not force_rebuild and self._can_load_cached_index(index_path, embeddings.shape):
                return self._load_cached_collection(collection_id, index_path)
            
            # Create new collection
            return self._build_new_collection(collection_id, embeddings, document_ids)
        
        except Exception as e:
            self.logger.error(f"Failed to create collection {collection_id}: {e}")
            return False
    
    def search_collection(self, collection_id: str,
                         query_embedding: np.ndarray,
                         top_k: int = 100,
                         optimize_search: bool = True) -> Optional[Tuple[List[str], List[float]]]:
        """
        Search within a FAISS collection.
        
        Args:
            collection_id: Collection to search
            query_embedding: Query vector
            top_k: Number of results to return
            optimize_search: Apply search optimization
            
        Returns:
            Tuple of (document_ids, similarity_scores) or None if failed
        """
        if self.backend is None:
            self.logger.error("No collection loaded")
            return None
        
        try:
            # Apply search optimization if enabled
            if optimize_search and self.config.enable_optimization:
                self._optimize_search_parameters(top_k)
            
            # Perform search
            start_time = time.time()
            doc_ids, scores = self.backend.search(query_embedding, top_k)
            search_time_ms = (time.time() - start_time) * 1000
            
            # Update performance tracking
            self._update_search_performance(collection_id, search_time_ms, len(doc_ids))
            
            return doc_ids, scores
        
        except Exception as e:
            self.logger.error(f"Search failed for collection {collection_id}: {e}")
            return None
    
    def batch_search_collection(self, collection_id: str,
                               query_embeddings: np.ndarray,
                               top_k: int = 100) -> Optional[List[Tuple[List[str], List[float]]]]:
        """
        Perform batch search within a FAISS collection.
        
        Args:
            collection_id: Collection to search
            query_embeddings: Multiple query vectors
            top_k: Number of results per query
            
        Returns:
            List of (document_ids, similarity_scores) per query or None if failed
        """
        if self.backend is None:
            self.logger.error("No collection loaded")
            return None
        
        try:
            start_time = time.time()
            results = self.backend.batch_search(query_embeddings, top_k)
            batch_time_ms = (time.time() - start_time) * 1000
            
            # Update performance tracking
            avg_search_time = batch_time_ms / len(query_embeddings)
            self._update_search_performance(collection_id, avg_search_time, top_k)
            
            return results
        
        except Exception as e:
            self.logger.error(f"Batch search failed for collection {collection_id}: {e}")
            return None
    
    def add_to_collection(self, collection_id: str,
                         embeddings: np.ndarray,
                         document_ids: List[str]) -> bool:
        """
        Add new embeddings to existing collection.
        
        Args:
            collection_id: Target collection
            embeddings: New embedding vectors
            document_ids: Corresponding document identifiers
            
        Returns:
            Success status
        """
        if self.backend is None:
            self.logger.error("No collection loaded")
            return False
        
        try:
            self.backend.add_embeddings(embeddings, document_ids)
            
            # Update metadata
            if collection_id in self.collection_metadata:
                self.collection_metadata[collection_id]["size"] += len(document_ids)
                self.collection_metadata[collection_id]["last_updated"] = time.time()
            
            # Save updated index if caching enabled
            if self.index_cache_enabled:
                index_path = self._get_index_path(collection_id)
                self.backend.save_index(index_path)
            
            self.logger.info(f"Added {len(document_ids)} documents to collection {collection_id}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to add to collection {collection_id}: {e}")
            return False
    
    def optimize_collection(self, collection_id: str) -> Dict[str, Any]:
        """
        Optimize collection performance.
        
        Args:
            collection_id: Collection to optimize
            
        Returns:
            Optimization results
        """
        if self.backend is None:
            return {"optimized": False, "reason": "No collection loaded"}
        
        try:
            # Perform backend optimization
            backend_results = self.backend.optimize_index()
            
            # Additional high-level optimizations
            high_level_results = self._perform_high_level_optimization(collection_id)
            
            # Combine results
            optimization_results = {
                **backend_results,
                **high_level_results,
                "collection_id": collection_id,
                "timestamp": time.time()
            }
            
            # Record optimization history
            self.optimization_history.append(optimization_results)
            
            # Save optimized index
            if self.index_cache_enabled and optimization_results.get("optimized", False):
                index_path = self._get_index_path(collection_id)
                self.backend.save_index(index_path)
            
            return optimization_results
        
        except Exception as e:
            self.logger.error(f"Optimization failed for collection {collection_id}: {e}")
            return {"optimized": False, "error": str(e)}
    
    def get_collection_info(self, collection_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive collection information."""
        if collection_id not in self.collection_metadata:
            return None
        
        metadata = self.collection_metadata[collection_id].copy()
        
        # Add backend statistics if available
        if self.backend is not None:
            backend_stats = self.backend.get_statistics()
            metadata["backend_stats"] = backend_stats
            metadata["capabilities"] = self.backend.get_capabilities().__dict__
        
        # Add performance profile
        collection_profiles = [
            profile for profile in self.performance_profiles
            if profile.collection_size == metadata.get("size", 0)
        ]
        if collection_profiles:
            latest_profile = max(collection_profiles, key=lambda p: p.timestamp)
            metadata["performance_profile"] = latest_profile.__dict__
        
        return metadata
    
    def _build_new_collection(self, collection_id: str,
                             embeddings: np.ndarray,
                             document_ids: List[str]) -> bool:
        """Build new FAISS collection from scratch."""
        start_time = time.time()
        
        # Select optimal configuration
        optimal_config = self._select_optimal_configuration(embeddings.shape)
        
        # Create backend with optimized configuration
        self.backend = FAISSBackend(optimal_config)
        
        # Build index
        self.backend.build_index(embeddings, document_ids)
        
        build_time_ms = (time.time() - start_time) * 1000
        
        # Store metadata
        self.collection_metadata[collection_id] = {
            "size": len(document_ids),
            "dimension": embeddings.shape[1],
            "index_type": optimal_config["index_type"],
            "created_at": time.time(),
            "last_updated": time.time(),
            "build_time_ms": build_time_ms
        }
        
        # Create performance profile
        self._create_performance_profile(collection_id, embeddings.shape, build_time_ms)
        
        # Save index if caching enabled
        if self.index_cache_enabled:
            index_path = self._get_index_path(collection_id)
            self.backend.save_index(index_path)
        
        self.logger.info(
            f"Built collection {collection_id}: {len(document_ids)} documents, "
            f"{build_time_ms:.1f}ms"
        )
        
        return True
    
    def _load_cached_collection(self, collection_id: str, index_path: str) -> bool:
        """Load collection from cached index."""
        try:
            # Load existing configuration or use default
            config_dict = self.config.__dict__.copy()
            
            # Create backend
            self.backend = FAISSBackend(config_dict)
            
            # Load index
            success = self.backend.load_index(index_path)
            
            if success:
                # Load metadata if available
                metadata_path = index_path + ".collection_meta"
                if os.path.exists(metadata_path):
                    metadata = np.load(metadata_path, allow_pickle=True).item()
                    self.collection_metadata[collection_id] = metadata
                
                self.logger.info(f"Loaded cached collection {collection_id}")
                return True
            
            return False
        
        except Exception as e:
            self.logger.error(f"Failed to load cached collection: {e}")
            return False
    
    def _select_optimal_configuration(self, embeddings_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Select optimal FAISS configuration based on data characteristics."""
        num_vectors, dimension = embeddings_shape
        
        # Start with base configuration
        config = self.config.__dict__.copy()
        
        # Adjust index type if auto
        if config["index_type"] == "auto":
            if num_vectors < 10000:
                config["index_type"] = "IndexFlatIP"
            elif num_vectors < 100000:
                config["index_type"] = "IndexIVFFlat"
                config["nlist"] = min(1024, num_vectors // 50)
            elif num_vectors < 1000000:
                config["index_type"] = "IndexHNSWFlat"
            else:
                config["index_type"] = "IndexIVFPQ"
                config["nlist"] = min(4096, num_vectors // 100)
        
        # Adjust parameters based on dataset size
        if config["index_type"] == "IndexIVFFlat":
            config["nprobe"] = max(1, min(config["nlist"] // 8, 128))
        
        return config
    
    def _optimize_search_parameters(self, top_k: int) -> None:
        """Optimize search parameters based on query requirements."""
        if self.backend is None:
            return
        
        # Adjust nprobe for IVF indexes based on top_k
        if hasattr(self.backend.index, 'nprobe'):
            if top_k <= 10:
                # For small result sets, use fewer probes for speed
                self.backend.nprobe = max(8, self.backend.nprobe // 2)
            elif top_k >= 100:
                # For large result sets, use more probes for recall
                self.backend.nprobe = min(256, self.backend.nprobe * 2)
            
            self.backend.index.nprobe = self.backend.nprobe
        
        # Adjust ef_search for HNSW indexes
        if hasattr(self.backend.index, 'efSearch'):
            if top_k <= 10:
                self.backend.ef_search = max(50, top_k * 5)
            else:
                self.backend.ef_search = max(100, min(top_k * 2, 400))
            
            self.backend.index.efSearch = self.backend.ef_search
    
    def _perform_high_level_optimization(self, collection_id: str) -> Dict[str, Any]:
        """Perform high-level optimization strategies."""
        results = {}
        
        # Analyze query patterns
        if collection_id in self.collection_metadata:
            metadata = self.collection_metadata[collection_id]
            
            # Check if index rebuild would be beneficial
            if metadata.get("last_updated", 0) > metadata.get("last_optimized", 0):
                results["rebuild_recommended"] = True
                results["rebuild_reason"] = "Index has been updated since last optimization"
        
        return results
    
    def _create_performance_profile(self, collection_id: str,
                                   embeddings_shape: Tuple[int, int],
                                   build_time_ms: float) -> None:
        """Create performance profile for the collection."""
        num_vectors, dimension = embeddings_shape
        
        # Estimate performance metrics
        memory_usage_mb = self.backend.get_memory_usage_mb() if self.backend else 0.0
        
        profile = IndexPerformanceProfile(
            index_type=self.collection_metadata[collection_id]["index_type"],
            build_time_ms=build_time_ms,
            search_time_ms=0.0,  # Will be updated during searches
            memory_usage_mb=memory_usage_mb,
            accuracy_score=1.0,  # Will be updated based on benchmarks
            collection_size=num_vectors,
            dimension=dimension
        )
        
        self.performance_profiles.append(profile)
    
    def _update_search_performance(self, collection_id: str,
                                  search_time_ms: float,
                                  result_count: int) -> None:
        """Update search performance metrics."""
        # Find relevant performance profile
        for profile in self.performance_profiles:
            if (collection_id in self.collection_metadata and
                profile.collection_size == self.collection_metadata[collection_id]["size"]):
                
                # Update with exponential moving average
                if profile.search_time_ms == 0.0:
                    profile.search_time_ms = search_time_ms
                else:
                    alpha = 0.1
                    profile.search_time_ms = (
                        alpha * search_time_ms + (1 - alpha) * profile.search_time_ms
                    )
                break
    
    def _get_index_path(self, collection_id: str) -> str:
        """Get file path for cached index."""
        return os.path.join(self.config.index_cache_dir, f"{collection_id}.faiss")
    
    def _can_load_cached_index(self, index_path: str, 
                              embeddings_shape: Tuple[int, int]) -> bool:
        """Check if cached index can be loaded."""
        if not os.path.exists(index_path):
            return False
        
        # Check metadata compatibility
        metadata_path = index_path + ".meta"
        if os.path.exists(metadata_path):
            try:
                metadata = np.load(metadata_path, allow_pickle=True)
                cached_dimension = int(metadata['dimension'])
                
                # Check dimension compatibility
                if cached_dimension != embeddings_shape[1]:
                    return False
                
                return True
            except Exception:
                return False
        
        return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            "collections_count": len(self.collection_metadata),
            "optimization_count": len(self.optimization_history),
            "performance_profiles": [profile.__dict__ for profile in self.performance_profiles]
        }
        
        # Add backend statistics
        if self.backend is not None:
            summary["backend_stats"] = self.backend.get_statistics()
        
        # Add recent optimizations
        if self.optimization_history:
            summary["recent_optimizations"] = self.optimization_history[-5:]
        
        return summary


__all__ = [
    "FAISSConfiguration",
    "IndexPerformanceProfile", 
    "AdvancedFAISSIntegration"
]
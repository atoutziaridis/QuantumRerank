"""
Advanced FAISS backend implementation for scalable vector search.

This module provides a high-performance FAISS-based vector search backend
with support for multiple index types, GPU acceleration, and optimization.
"""

import time
import os
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from .base_backend import BaseVectorSearchBackend, BackendCapabilities, SearchConfiguration
from ...utils import get_logger

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FAISSBackend(BaseVectorSearchBackend):
    """
    Advanced FAISS backend for scalable vector search.
    
    Supports multiple index types, automatic optimization,
    and GPU acceleration for high-performance search.
    """
    
    def __init__(self, config: Dict[str, Any]):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        super().__init__("faiss", config)
        
        # FAISS configuration
        self.index_type = config.get("index_type", "auto")
        self.use_gpu = config.get("use_gpu", False)
        self.nlist = config.get("nlist", 1024)
        self.nprobe = config.get("nprobe", 64)
        self.ef_construction = config.get("ef_construction", 200)
        self.ef_search = config.get("ef_search", 100)
        self.m_hnsw = config.get("m_hnsw", 32)
        
        # State
        self.index: Optional[faiss.Index] = None
        self.document_ids: List[str] = []
        self.dimension = None
        self.gpu_resources = None
        
        # Initialize GPU if requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            self._initialize_gpu()
        
        self.logger.info(f"Initialized FAISS backend: type={self.index_type}, gpu={self.use_gpu}")
    
    def _initialize_gpu(self) -> None:
        """Initialize GPU resources for FAISS."""
        try:
            self.gpu_resources = faiss.StandardGpuResources()
            self.logger.info(f"GPU initialized: {faiss.get_num_gpus()} GPUs available")
        except Exception as e:
            self.logger.warning(f"GPU initialization failed: {e}")
            self.use_gpu = False
    
    def build_index(self, embeddings: np.ndarray, document_ids: List[str]) -> None:
        """Build FAISS index from embeddings."""
        if not self.validate_embeddings(embeddings):
            raise ValueError("Invalid embeddings format")
        
        if len(document_ids) != embeddings.shape[0]:
            raise ValueError("Number of document IDs must match number of embeddings")
        
        start_time = time.time()
        
        # Store metadata
        self.dimension = embeddings.shape[1]
        self.document_ids = document_ids.copy()
        
        # Select optimal index type
        index_type = self._select_optimal_index_type(embeddings.shape[0], self.dimension)
        
        # Create index
        self.index = self._create_index(index_type, self.dimension)
        
        # Train if necessary
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            self.logger.info("Training index...")
            self.index.train(embeddings.astype(np.float32))
        
        # Add vectors
        self.index.add(embeddings.astype(np.float32))
        
        build_time_ms = (time.time() - start_time) * 1000
        self.logger.info(
            f"Built FAISS index: {embeddings.shape[0]} vectors, "
            f"{build_time_ms:.1f}ms, type={index_type}"
        )
    
    def search(self, query_embedding: np.ndarray, top_k: int = 100) -> Tuple[List[str], List[float]]:
        """Search for similar embeddings using FAISS."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        start_time = time.time()
        
        # Prepare query
        query = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Set search parameters
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe
        if hasattr(self.index, 'efSearch'):
            self.index.efSearch = self.ef_search
        
        # Perform search
        distances, indices = self.index.search(query, top_k)
        
        # Convert results
        doc_ids = []
        scores = []
        
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.document_ids):
                doc_ids.append(self.document_ids[idx])
                # Convert FAISS distance to similarity score
                scores.append(self._distance_to_similarity(distances[0][i]))
        
        query_time_ms = (time.time() - start_time) * 1000
        self._record_query_metrics(query_time_ms)
        
        return doc_ids, scores
    
    def add_embeddings(self, embeddings: np.ndarray, document_ids: List[str]) -> None:
        """Add new embeddings to existing index."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        if not self.validate_embeddings(embeddings):
            raise ValueError("Invalid embeddings format")
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: {embeddings.shape[1]} != {self.dimension}")
        
        # Add to index
        self.index.add(embeddings.astype(np.float32))
        
        # Update document mapping
        self.document_ids.extend(document_ids)
        
        self.logger.info(f"Added {len(document_ids)} embeddings to index")
    
    def remove_embeddings(self, document_ids: List[str]) -> None:
        """Remove embeddings from index."""
        # FAISS doesn't support efficient removal, so we log a warning
        self.logger.warning(
            "FAISS doesn't support efficient removal. "
            "Consider rebuilding the index for best performance."
        )
        
        # Remove from document mapping (for consistency)
        for doc_id in document_ids:
            if doc_id in self.document_ids:
                idx = self.document_ids.index(doc_id)
                self.document_ids[idx] = None  # Mark as removed
    
    def get_capabilities(self) -> BackendCapabilities:
        """Get FAISS backend capabilities."""
        return BackendCapabilities(
            max_dimension=65536,
            max_collection_size=1000000000,  # 1B vectors
            supports_gpu=self.use_gpu and faiss.get_num_gpus() > 0,
            supports_batch_search=True,
            supports_incremental_update=True,
            memory_efficiency_score=0.85,
            latency_score=0.9,
            accuracy_score=0.95
        )
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        if self.index is None:
            return 0.0
        
        # Estimate memory usage
        # Base index memory + vectors
        vector_memory = len(self.document_ids) * self.dimension * 4  # float32
        index_overhead = vector_memory * 0.2  # Estimated 20% overhead
        
        return (vector_memory + index_overhead) / (1024 * 1024)
    
    def batch_search(self, query_embeddings: np.ndarray,
                    top_k: int = 100) -> List[Tuple[List[str], List[float]]]:
        """Perform efficient batch search using FAISS."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        start_time = time.time()
        
        # Prepare queries
        queries = query_embeddings.astype(np.float32)
        
        # Set search parameters
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe
        if hasattr(self.index, 'efSearch'):
            self.index.efSearch = self.ef_search
        
        # Perform batch search
        distances, indices = self.index.search(queries, top_k)
        
        # Convert results
        results = []
        for i in range(len(queries)):
            doc_ids = []
            scores = []
            
            for j, idx in enumerate(indices[i]):
                if idx >= 0 and idx < len(self.document_ids) and self.document_ids[idx] is not None:
                    doc_ids.append(self.document_ids[idx])
                    scores.append(self._distance_to_similarity(distances[i][j]))
            
            results.append((doc_ids, scores))
        
        batch_time_ms = (time.time() - start_time) * 1000
        avg_query_time = batch_time_ms / len(queries)
        self._record_query_metrics(avg_query_time)
        
        return results
    
    def optimize_index(self) -> Dict[str, Any]:
        """Optimize FAISS index performance."""
        if self.index is None:
            return {"optimized": False, "reason": "No index to optimize"}
        
        optimization_results = {}
        
        # Optimize nprobe for IVF indexes
        if hasattr(self.index, 'nprobe'):
            original_nprobe = self.index.nprobe
            optimal_nprobe = self._find_optimal_nprobe()
            
            if optimal_nprobe != original_nprobe:
                self.nprobe = optimal_nprobe
                optimization_results["nprobe_optimized"] = {
                    "old": original_nprobe,
                    "new": optimal_nprobe
                }
        
        # Optimize ef_search for HNSW indexes
        if hasattr(self.index, 'efSearch'):
            original_ef = self.index.efSearch
            optimal_ef = self._find_optimal_ef_search()
            
            if optimal_ef != original_ef:
                self.ef_search = optimal_ef
                optimization_results["ef_search_optimized"] = {
                    "old": original_ef,
                    "new": optimal_ef
                }
        
        optimization_results["optimized"] = len(optimization_results) > 0
        return optimization_results
    
    def save_index(self, filepath: str) -> bool:
        """Save FAISS index to disk."""
        if self.index is None:
            return False
        
        try:
            # Save index
            faiss.write_index(self.index, filepath)
            
            # Save metadata
            metadata_path = filepath + ".meta"
            np.savez(
                metadata_path,
                document_ids=self.document_ids,
                dimension=self.dimension,
                index_type=self.index_type,
                config=self.config
            )
            
            self.logger.info(f"Saved FAISS index to {filepath}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")
            return False
    
    def load_index(self, filepath: str) -> bool:
        """Load FAISS index from disk."""
        try:
            # Load index
            self.index = faiss.read_index(filepath)
            
            # Load metadata
            metadata_path = filepath + ".meta"
            if os.path.exists(metadata_path):
                metadata = np.load(metadata_path, allow_pickle=True)
                self.document_ids = metadata['document_ids'].tolist()
                self.dimension = int(metadata['dimension'])
                self.index_type = str(metadata['index_type'])
            
            # Move to GPU if requested
            if self.use_gpu and self.gpu_resources is not None:
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
            
            self.logger.info(f"Loaded FAISS index from {filepath}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to load index: {e}")
            return False
    
    def _select_optimal_index_type(self, num_vectors: int, dimension: int) -> str:
        """Select optimal FAISS index type based on data characteristics."""
        if self.index_type != "auto":
            return self.index_type
        
        # Selection logic based on dataset size and requirements
        if num_vectors < 10000:
            return "IndexFlatIP"  # Exact search for small datasets
        elif num_vectors < 100000:
            return "IndexIVFFlat"  # IVF for medium datasets
        elif num_vectors < 1000000:
            return "IndexHNSWFlat"  # HNSW for large datasets
        else:
            return "IndexIVFPQ"  # Product quantization for very large datasets
    
    def _create_index(self, index_type: str, dimension: int) -> faiss.Index:
        """Create FAISS index of specified type."""
        if index_type == "IndexFlatIP":
            index = faiss.IndexFlatIP(dimension)
        
        elif index_type == "IndexIVFFlat":
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist)
        
        elif index_type == "IndexHNSWFlat":
            index = faiss.IndexHNSWFlat(dimension, self.m_hnsw)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search
        
        elif index_type == "IndexIVFPQ":
            quantizer = faiss.IndexFlatIP(dimension)
            m = 8  # Number of subquantizers
            bits = 8  # Bits per subquantizer
            index = faiss.IndexIVFPQ(quantizer, dimension, self.nlist, m, bits)
        
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Move to GPU if requested
        if self.use_gpu and self.gpu_resources is not None:
            try:
                index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
                self.logger.info("Moved index to GPU")
            except Exception as e:
                self.logger.warning(f"Failed to move index to GPU: {e}")
        
        return index
    
    def _distance_to_similarity(self, distance: float) -> float:
        """Convert FAISS distance to similarity score."""
        # For IP (Inner Product), distance is already similarity
        # For L2, convert distance to similarity
        if hasattr(self.index, 'metric_type'):
            if self.index.metric_type == faiss.METRIC_L2:
                # Convert L2 distance to similarity (0-1 range)
                return 1.0 / (1.0 + distance)
            else:
                # IP distance is already similarity-like
                return max(0.0, distance)
        else:
            # Default to IP behavior
            return max(0.0, distance)
    
    def _find_optimal_nprobe(self) -> int:
        """Find optimal nprobe value for IVF indexes."""
        # Simple heuristic: balance speed vs accuracy
        if hasattr(self.index, 'nlist'):
            # Start with 1/16 of centroids
            optimal = max(1, self.index.nlist // 16)
            return min(optimal, 128)  # Cap at 128 for performance
        return self.nprobe
    
    def _find_optimal_ef_search(self) -> int:
        """Find optimal efSearch value for HNSW indexes."""
        # Balance between search quality and speed
        # Higher ef_search = better recall but slower
        if self.ef_search < 50:
            return 50  # Minimum for reasonable recall
        elif self.ef_search > 200:
            return 200  # Maximum for reasonable speed
        return self.ef_search


__all__ = ["FAISSBackend"]
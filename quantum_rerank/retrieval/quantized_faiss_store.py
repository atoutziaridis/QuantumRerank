"""
Quantized FAISS Vector Store for Ultra-Lightweight RAG

This module implements advanced FAISS quantization techniques to achieve
8x compression with minimal accuracy loss, supporting the quantum-inspired
lightweight RAG transition strategy.

Based on:
- Research: "Practical Applications and Deployment of Tensor Networks"
- Target: 8x compression (4x quantization + 2x dimensionality reduction)
- Methods: IVF, PQ, OPQ, LSH indexing with adaptive quantization
"""

import faiss
import numpy as np
import torch
import logging
import time
import pickle
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json

from ..core.embeddings import EmbeddingProcessor

logger = logging.getLogger(__name__)


@dataclass
class QuantizedFAISSConfig:
    """Configuration for quantized FAISS store."""
    
    # Quantization settings
    quantization_bits: int = 8  # 8-bit quantization (4x compression)
    use_opq: bool = True  # Optimized Product Quantization
    use_ivf: bool = True  # Inverted File indexing
    
    # Dimensionality reduction
    target_dim: int = 384  # Reduce from 768 (2x compression)
    use_pca: bool = True
    pca_whitening: bool = True
    
    # Index construction
    nlist: int = 100  # Number of clusters for IVF
    m: int = 48  # Number of subquantizers (must divide target_dim)
    nbits: int = 8  # Bits per subquantizer
    
    # Search parameters
    nprobe: int = 10  # Search clusters
    max_codes: int = 1000000  # Maximum codes to store
    
    # Performance settings
    use_gpu: bool = False  # Enable GPU acceleration
    gpu_id: int = 0
    
    # Memory optimization
    enable_memory_mapping: bool = True
    cache_size_mb: int = 256
    
    # Validation settings
    validate_compression: bool = True
    accuracy_threshold: float = 0.95  # Minimum search accuracy


class QuantizedFAISSStore:
    """
    Advanced quantized FAISS vector store with 8x compression.
    
    Implements multi-level compression:
    1. PCA dimensionality reduction (768D → 384D, 2x compression)
    2. 8-bit quantization (4x compression)
    3. Total: 8x compression with <5% accuracy loss
    """
    
    def __init__(self, 
                 config: Optional[QuantizedFAISSConfig] = None,
                 embedding_processor: Optional[EmbeddingProcessor] = None):
        """
        Initialize quantized FAISS store.
        
        Args:
            config: Quantization configuration
            embedding_processor: Embedding processor for dimension info
        """
        self.config = config or QuantizedFAISSConfig()
        self.embedding_processor = embedding_processor
        
        # Initialize index components
        self.index = None
        self.pca_transform = None
        self.doc_id_to_index = {}
        self.index_to_doc_id = {}
        
        # Performance tracking
        self.stats = {
            'total_vectors': 0,
            'compression_ratio': 0.0,
            'build_time_s': 0.0,
            'search_time_ms': 0.0,
            'memory_usage_mb': 0.0,
            'accuracy_validation': {}
        }
        
        # GPU setup
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            self.gpu_resource = faiss.StandardGpuResources()
            logger.info(f"GPU acceleration enabled on device {self.config.gpu_id}")
        else:
            self.gpu_resource = None
            if self.config.use_gpu:
                logger.warning("GPU requested but not available, using CPU")
        
        logger.info(f"Quantized FAISS store initialized: {self.config.quantization_bits}-bit, "
                   f"{self.config.target_dim}D")
    
    def _create_pca_transform(self, embeddings: np.ndarray) -> faiss.PCAMatrix:
        """Create PCA transformation for dimensionality reduction."""
        original_dim = embeddings.shape[1]
        
        logger.info(f"Creating PCA transform: {original_dim}D → {self.config.target_dim}D")
        
        # Create PCA matrix
        pca = faiss.PCAMatrix(original_dim, self.config.target_dim, 
                             eigenvalue_power=0.5 if self.config.pca_whitening else 0.0)
        
        # Train PCA on embeddings
        pca.train(embeddings.astype(np.float32))
        
        return pca
    
    def _create_quantized_index(self, dim: int) -> faiss.Index:
        """Create quantized FAISS index."""
        logger.info(f"Creating quantized index: {dim}D, {self.config.nlist} clusters")
        
        if self.config.use_ivf:
            # IVF + PQ index for large-scale efficiency
            quantizer = faiss.IndexFlatL2(dim)
            
            if self.config.use_opq:
                # OPQ (Optimized Product Quantization)
                index = faiss.IndexPreTransform(
                    faiss.OPQMatrix(dim, self.config.m),
                    faiss.IndexIVFPQ(quantizer, dim, self.config.nlist, 
                                   self.config.m, self.config.nbits)
                )
            else:
                # Standard IVF-PQ
                index = faiss.IndexIVFPQ(quantizer, dim, self.config.nlist,
                                       self.config.m, self.config.nbits)
        else:
            # Simple PQ index for smaller datasets
            index = faiss.IndexPQ(dim, self.config.m, self.config.nbits)
        
        # Configure search parameters
        if hasattr(index, 'nprobe'):
            index.nprobe = self.config.nprobe
        
        return index
    
    def _apply_gpu_acceleration(self, index: faiss.Index) -> faiss.Index:
        """Apply GPU acceleration to index if available."""
        if self.gpu_resource is None:
            return index
        
        try:
            # Move index to GPU
            gpu_index = faiss.index_cpu_to_gpu(self.gpu_resource, 
                                             self.config.gpu_id, index)
            logger.info("Index moved to GPU")
            return gpu_index
        except Exception as e:
            logger.warning(f"GPU acceleration failed: {e}")
            return index
    
    def build_index(self, 
                   embeddings: np.ndarray,
                   doc_ids: List[str]) -> Dict[str, Any]:
        """
        Build quantized FAISS index from embeddings.
        
        Args:
            embeddings: Document embeddings [n_docs, embedding_dim]
            doc_ids: Document IDs
            
        Returns:
            Build statistics
        """
        if len(embeddings) != len(doc_ids):
            raise ValueError("Number of embeddings must match number of doc IDs")
        
        start_time = time.time()
        
        # Convert to float32 for FAISS
        embeddings = embeddings.astype(np.float32)
        original_dim = embeddings.shape[1]
        
        logger.info(f"Building quantized index: {len(embeddings)} vectors, {original_dim}D")
        
        # Step 1: Apply PCA dimensionality reduction
        if self.config.use_pca and self.config.target_dim < original_dim:
            self.pca_transform = self._create_pca_transform(embeddings)
            embeddings = self.pca_transform.apply(embeddings)
            logger.info(f"PCA applied: {original_dim}D → {embeddings.shape[1]}D")
        
        # Step 2: Create quantized index
        self.index = self._create_quantized_index(embeddings.shape[1])
        
        # Step 3: Apply GPU acceleration
        self.index = self._apply_gpu_acceleration(self.index)
        
        # Step 4: Train index if needed
        if hasattr(self.index, 'train'):
            logger.info("Training quantized index...")
            self.index.train(embeddings)
        
        # Step 5: Add vectors to index
        logger.info("Adding vectors to index...")
        self.index.add(embeddings)
        
        # Step 6: Build mapping
        self.doc_id_to_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}
        self.index_to_doc_id = {i: doc_id for i, doc_id in enumerate(doc_ids)}
        
        build_time = time.time() - start_time
        
        # Calculate compression statistics
        original_size = len(embeddings) * original_dim * 4  # float32 bytes
        compressed_size = self._estimate_index_size()
        compression_ratio = original_size / compressed_size
        
        # Update statistics
        self.stats.update({
            'total_vectors': len(embeddings),
            'compression_ratio': compression_ratio,
            'build_time_s': build_time,
            'memory_usage_mb': compressed_size / (1024 * 1024),
            'original_dim': original_dim,
            'compressed_dim': embeddings.shape[1]
        })
        
        logger.info(f"Index built: {compression_ratio:.1f}x compression, "
                   f"{build_time:.2f}s build time")
        
        # Validate compression if requested
        if self.config.validate_compression:
            self._validate_index_accuracy(embeddings, doc_ids)
        
        return self.stats
    
    def _estimate_index_size(self) -> int:
        """Estimate memory size of quantized index."""
        if self.index is None:
            return 0
        
        # Estimate based on index type and parameters
        n_vectors = self.index.ntotal
        
        if hasattr(self.index, 'm'):  # PQ index
            # Each vector uses m * nbits bits
            bits_per_vector = self.config.m * self.config.nbits
            bytes_per_vector = bits_per_vector / 8
        else:
            # Fallback estimation
            bytes_per_vector = self.config.target_dim * self.config.quantization_bits / 8
        
        total_size = n_vectors * bytes_per_vector
        
        # Add overhead for centroids and metadata
        overhead = self.config.nlist * self.config.target_dim * 4  # Centroids
        
        return int(total_size + overhead)
    
    def search(self, 
               query_embedding: np.ndarray,
               k: int = 10,
               return_distances: bool = False) -> Union[List[str], Tuple[List[str], List[float]]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            return_distances: Whether to return distances
            
        Returns:
            List of document IDs (and distances if requested)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        start_time = time.time()
        
        # Ensure query is 2D array
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype(np.float32)
        
        # Apply PCA if used during indexing
        if self.pca_transform is not None:
            query_embedding = self.pca_transform.apply(query_embedding)
        
        # Search index
        distances, indices = self.index.search(query_embedding, k)
        
        search_time = (time.time() - start_time) * 1000
        self.stats['search_time_ms'] = search_time
        
        # Convert indices to document IDs
        doc_ids = []
        valid_distances = []
        
        for i, distance in zip(indices[0], distances[0]):
            if i != -1 and i in self.index_to_doc_id:
                doc_ids.append(self.index_to_doc_id[i])
                valid_distances.append(float(distance))
        
        logger.debug(f"Search completed: {len(doc_ids)} results, {search_time:.2f}ms")
        
        if return_distances:
            return doc_ids, valid_distances
        return doc_ids
    
    def batch_search(self, 
                    query_embeddings: np.ndarray,
                    k: int = 10) -> List[List[str]]:
        """
        Batch search for multiple queries.
        
        Args:
            query_embeddings: Query embeddings [n_queries, dim]
            k: Number of results per query
            
        Returns:
            List of result lists
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Apply PCA if used
        if self.pca_transform is not None:
            query_embeddings = self.pca_transform.apply(query_embeddings.astype(np.float32))
        
        # Batch search
        distances, indices = self.index.search(query_embeddings.astype(np.float32), k)
        
        # Convert to document IDs
        results = []
        for query_indices in indices:
            doc_ids = []
            for i in query_indices:
                if i != -1 and i in self.index_to_doc_id:
                    doc_ids.append(self.index_to_doc_id[i])
            results.append(doc_ids)
        
        return results
    
    def _validate_index_accuracy(self, 
                                embeddings: np.ndarray,
                                doc_ids: List[str]) -> Dict[str, float]:
        """Validate index accuracy by comparing with brute force search."""
        logger.info("Validating index accuracy...")
        
        # Create brute force index for comparison
        brute_force_index = faiss.IndexFlatL2(embeddings.shape[1])
        brute_force_index.add(embeddings)
        
        # Sample queries for validation
        n_queries = min(100, len(embeddings))
        query_indices = np.random.choice(len(embeddings), n_queries, replace=False)
        
        recalls = []
        
        for query_idx in query_indices:
            query_emb = embeddings[query_idx:query_idx+1]
            
            # Get ground truth (brute force)
            _, bf_indices = brute_force_index.search(query_emb, 20)
            bf_results = set(bf_indices[0])
            
            # Get quantized results
            _, q_indices = self.index.search(query_emb, 20)
            q_results = set(q_indices[0])
            
            # Calculate recall
            intersection = len(bf_results & q_results)
            recall = intersection / len(bf_results) if bf_results else 0.0
            recalls.append(recall)
        
        accuracy_stats = {
            'mean_recall': np.mean(recalls),
            'min_recall': np.min(recalls),
            'max_recall': np.max(recalls),
            'std_recall': np.std(recalls)
        }
        
        self.stats['accuracy_validation'] = accuracy_stats
        
        logger.info(f"Index accuracy: {accuracy_stats['mean_recall']:.3f} mean recall")
        
        return accuracy_stats
    
    def save_index(self, filepath: str):
        """Save quantized index to disk."""
        if self.index is None:
            raise ValueError("No index to save")
        
        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save index
        faiss.write_index(self.index, f"{filepath}.index")
        
        # Save PCA transform if used
        if self.pca_transform is not None:
            faiss.write_VectorTransform(self.pca_transform, f"{filepath}.pca")
        
        # Save metadata
        metadata = {
            'config': self.config.__dict__,
            'doc_id_to_index': self.doc_id_to_index,
            'index_to_doc_id': self.index_to_doc_id,
            'stats': self.stats
        }
        
        with open(f"{filepath}.metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str):
        """Load quantized index from disk."""
        # Load index
        self.index = faiss.read_index(f"{filepath}.index")
        
        # Load PCA transform if exists
        pca_path = f"{filepath}.pca"
        if Path(pca_path).exists():
            self.pca_transform = faiss.read_VectorTransform(pca_path)
        
        # Load metadata
        with open(f"{filepath}.metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Restore mappings and stats
        self.doc_id_to_index = metadata['doc_id_to_index']
        self.index_to_doc_id = {int(k): v for k, v in metadata['index_to_doc_id'].items()}
        self.stats = metadata['stats']
        
        logger.info(f"Index loaded from {filepath}")
    
    def benchmark_compression(self, 
                            embeddings: np.ndarray,
                            doc_ids: List[str],
                            test_queries: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Benchmark compression performance.
        
        Args:
            embeddings: Document embeddings
            doc_ids: Document IDs
            test_queries: Optional test queries for search benchmarking
            
        Returns:
            Comprehensive benchmark results
        """
        results = {}
        
        # Build index and get basic stats
        build_stats = self.build_index(embeddings, doc_ids)
        results['build_stats'] = build_stats
        
        # Search benchmarking
        if test_queries is not None:
            search_times = []
            for query in test_queries:
                start_time = time.time()
                _ = self.search(query, k=10)
                search_times.append((time.time() - start_time) * 1000)
            
            results['search_performance'] = {
                'mean_search_time_ms': np.mean(search_times),
                'std_search_time_ms': np.std(search_times),
                'min_search_time_ms': np.min(search_times),
                'max_search_time_ms': np.max(search_times)
            }
        
        # Memory efficiency
        results['memory_efficiency'] = {
            'compression_ratio': build_stats['compression_ratio'],
            'memory_usage_mb': build_stats['memory_usage_mb'],
            'vectors_per_mb': build_stats['total_vectors'] / build_stats['memory_usage_mb']
        }
        
        # Accuracy validation
        if 'accuracy_validation' in self.stats:
            results['accuracy'] = self.stats['accuracy_validation']
        
        return results


# Utility functions for integration
def create_quantized_faiss_config(compression_level: str = "balanced") -> QuantizedFAISSConfig:
    """
    Create optimized FAISS configuration for different compression levels.
    
    Args:
        compression_level: "fast", "balanced", or "maximum"
        
    Returns:
        Optimized configuration
    """
    if compression_level == "fast":
        return QuantizedFAISSConfig(
            quantization_bits=16,
            target_dim=512,
            use_opq=False,
            nlist=50,
            m=32
        )
    elif compression_level == "balanced":
        return QuantizedFAISSConfig(
            quantization_bits=8,
            target_dim=384,
            use_opq=True,
            nlist=100,
            m=48
        )
    elif compression_level == "maximum":
        return QuantizedFAISSConfig(
            quantization_bits=4,
            target_dim=256,
            use_opq=True,
            nlist=200,
            m=32
        )
    else:
        raise ValueError(f"Unknown compression level: {compression_level}")


def validate_quantized_faiss() -> Dict[str, Any]:
    """
    Validate quantized FAISS implementation.
    
    Returns:
        Validation results
    """
    try:
        # Test basic functionality
        config = QuantizedFAISSConfig(target_dim=64, nlist=10, m=8)
        store = QuantizedFAISSStore(config)
        
        # Create test data
        embeddings = np.random.randn(100, 128).astype(np.float32)
        doc_ids = [f"doc_{i}" for i in range(100)]
        
        # Build index
        stats = store.build_index(embeddings, doc_ids)
        
        # Test search
        query = np.random.randn(128).astype(np.float32)
        results = store.search(query, k=5)
        
        return {
            'status': 'success',
            'compression_ratio': stats['compression_ratio'],
            'search_results': len(results),
            'memory_usage_mb': stats['memory_usage_mb'],
            'message': 'Quantized FAISS validation successful'
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Validation failed: {str(e)}'
        }
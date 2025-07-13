"""
Index builder with optimized construction strategies for different data sizes.
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np

from .index_manager import IndexConfiguration
from ..backends.faiss_backend import FAISSBackend
from ...utils import get_logger


@dataclass
class BuildStrategy:
    """Index building strategy configuration."""
    name: str
    min_size: int
    max_size: int
    index_type: str
    parameters: Dict[str, Any]
    memory_efficient: bool = False
    parallel_build: bool = False


class IndexBuilder:
    """
    Optimized index builder for different dataset characteristics.
    
    This builder selects optimal construction strategies based on
    data size, dimension, and performance requirements.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.build_strategies = self._initialize_build_strategies()
    
    def build_optimized_index(self, embeddings: np.ndarray,
                             document_ids: List[str],
                             config: Optional[IndexConfiguration] = None) -> FAISSBackend:
        """
        Build optimized index using best strategy for data characteristics.
        
        Args:
            embeddings: Embedding vectors to index
            document_ids: Corresponding document identifiers
            config: Optional configuration (auto-selected if None)
            
        Returns:
            Built FAISS backend
        """
        start_time = time.time()
        
        # Auto-select configuration if not provided
        if config is None:
            config = self._select_optimal_config(embeddings.shape)
        
        # Select build strategy
        strategy = self._select_build_strategy(embeddings.shape, config)
        
        self.logger.info(
            f"Building index with strategy '{strategy.name}' for "
            f"{embeddings.shape[0]} vectors (dim={embeddings.shape[1]})"
        )
        
        # Create and configure backend
        backend_config = self._create_backend_config(config, strategy)
        backend = FAISSBackend(backend_config)
        
        # Build index using selected strategy
        if strategy.memory_efficient:
            self._build_memory_efficient(backend, embeddings, document_ids, strategy)
        elif strategy.parallel_build:
            self._build_parallel(backend, embeddings, document_ids, strategy)
        else:
            self._build_standard(backend, embeddings, document_ids, strategy)
        
        build_time = time.time() - start_time
        self.logger.info(f"Index build completed in {build_time:.2f}s")
        
        return backend
    
    def _select_optimal_config(self, embeddings_shape: Tuple[int, int]) -> IndexConfiguration:
        """Select optimal configuration based on data characteristics."""
        num_vectors, dimension = embeddings_shape
        
        if num_vectors < 1000:
            # Very small dataset - use exact search
            return IndexConfiguration(
                index_type="IndexFlatIP",
                dimension=dimension,
                backend_type="faiss"
            )
        elif num_vectors < 50000:
            # Small dataset - use IVF with small nlist
            return IndexConfiguration(
                index_type="IndexIVFFlat",
                dimension=dimension,
                parameters={"nlist": min(100, num_vectors // 10)},
                backend_type="faiss"
            )
        elif num_vectors < 500000:
            # Medium dataset - use HNSW
            return IndexConfiguration(
                index_type="IndexHNSWFlat",
                dimension=dimension,
                parameters={"M": 16, "efConstruction": 100},
                backend_type="faiss"
            )
        else:
            # Large dataset - use IVF with PQ
            return IndexConfiguration(
                index_type="IndexIVFPQ",
                dimension=dimension,
                parameters={
                    "nlist": min(4096, num_vectors // 100),
                    "m": min(64, dimension // 4),
                    "bits": 8
                },
                backend_type="faiss"
            )
    
    def _select_build_strategy(self, embeddings_shape: Tuple[int, int],
                              config: IndexConfiguration) -> BuildStrategy:
        """Select optimal build strategy."""
        num_vectors, dimension = embeddings_shape
        
        # Find matching strategy
        for strategy in self.build_strategies:
            if strategy.min_size <= num_vectors <= strategy.max_size:
                return strategy
        
        # Default to largest strategy
        return self.build_strategies[-1]
    
    def _build_standard(self, backend: FAISSBackend,
                       embeddings: np.ndarray,
                       document_ids: List[str],
                       strategy: BuildStrategy) -> None:
        """Standard index building approach."""
        backend.build_index(embeddings, document_ids)
    
    def _build_memory_efficient(self, backend: FAISSBackend,
                               embeddings: np.ndarray,
                               document_ids: List[str],
                               strategy: BuildStrategy) -> None:
        """Memory-efficient index building for large datasets."""
        batch_size = 10000  # Process in smaller batches
        
        # Build initial index with first batch
        first_batch_embeddings = embeddings[:batch_size]
        first_batch_ids = document_ids[:batch_size]
        
        backend.build_index(first_batch_embeddings, first_batch_ids)
        
        # Add remaining data in batches
        for i in range(batch_size, len(embeddings), batch_size):
            end_idx = min(i + batch_size, len(embeddings))
            batch_embeddings = embeddings[i:end_idx]
            batch_ids = document_ids[i:end_idx]
            
            backend.add_embeddings(batch_embeddings, batch_ids)
            
            self.logger.debug(f"Processed batch {i//batch_size + 1}")
    
    def _build_parallel(self, backend: FAISSBackend,
                       embeddings: np.ndarray,
                       document_ids: List[str],
                       strategy: BuildStrategy) -> None:
        """Parallel index building (placeholder for future implementation)."""
        # For now, fall back to standard building
        # Future implementation could use parallel training
        self._build_standard(backend, embeddings, document_ids, strategy)
    
    def _create_backend_config(self, config: IndexConfiguration,
                              strategy: BuildStrategy) -> Dict[str, Any]:
        """Create backend configuration from index config and strategy."""
        backend_config = {
            "index_type": config.index_type,
            "use_gpu": config.enable_gpu,
            **config.parameters
        }
        
        # Apply strategy-specific optimizations
        if strategy.name == "memory_efficient":
            # Reduce memory usage
            if "efConstruction" in backend_config:
                backend_config["efConstruction"] = min(backend_config["efConstruction"], 50)
        
        elif strategy.name == "speed_optimized":
            # Optimize for build speed
            if "nlist" in backend_config:
                backend_config["nlist"] = min(backend_config["nlist"], 1024)
        
        return backend_config
    
    def _initialize_build_strategies(self) -> List[BuildStrategy]:
        """Initialize available build strategies."""
        return [
            BuildStrategy(
                name="exact_small",
                min_size=0,
                max_size=10000,
                index_type="IndexFlatIP",
                parameters={}
            ),
            BuildStrategy(
                name="ivf_medium",
                min_size=10000,
                max_size=100000,
                index_type="IndexIVFFlat",
                parameters={"nlist": 1024}
            ),
            BuildStrategy(
                name="hnsw_large",
                min_size=100000,
                max_size=1000000,
                index_type="IndexHNSWFlat",
                parameters={"M": 32, "efConstruction": 200}
            ),
            BuildStrategy(
                name="memory_efficient",
                min_size=1000000,
                max_size=10000000,
                index_type="IndexIVFPQ",
                parameters={"nlist": 4096, "m": 8, "bits": 8},
                memory_efficient=True
            ),
            BuildStrategy(
                name="massive_scale",
                min_size=10000000,
                max_size=float('inf'),
                index_type="IndexIVFPQ",
                parameters={"nlist": 8192, "m": 16, "bits": 8},
                memory_efficient=True,
                parallel_build=True
            )
        ]


__all__ = ["BuildStrategy", "IndexBuilder"]
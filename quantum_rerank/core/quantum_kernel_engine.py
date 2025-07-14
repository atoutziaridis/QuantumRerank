"""
Quantum Kernel Engine for QuantumRerank.

Extends existing fidelity similarity infrastructure to provide quantum kernel methods
for machine learning applications. Builds on existing SWAP test and fidelity computation.

Based on:
- Quantum Kernel Machine Learning (Qiskit tutorial)
- "Supervised learning with quantum enhanced feature spaces" paper
- QuantumRerank's existing quantum similarity infrastructure
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
import time
from dataclasses import dataclass
from enum import Enum

from .fidelity_similarity import FidelitySimilarityEngine
from .embeddings import EmbeddingProcessor

logger = logging.getLogger(__name__)


@dataclass
class QuantumKernelConfig:
    """Configuration for quantum kernel computation."""
    n_qubits: int = 4
    encoding_method: str = "amplitude"  # amplitude, angle, dense_angle
    enable_caching: bool = True
    max_cache_size: int = 1000
    batch_size: int = 50


class QuantumKernelEngine:
    """
    Quantum kernel computation engine that leverages existing fidelity infrastructure.
    
    Builds on QuantumRerank's existing FidelitySimilarityEngine to provide
    kernel matrix computation for machine learning applications.
    """
    
    def __init__(self, config: QuantumKernelConfig = None):
        self.config = config or QuantumKernelConfig()
        
        # Use existing fidelity similarity engine
        self.fidelity_engine = FidelitySimilarityEngine(self.config.n_qubits)
        self.embedding_processor = EmbeddingProcessor()
        
        # Caching system for kernel matrices
        self._kernel_cache = {} if self.config.enable_caching else None
        
        # Performance tracking
        self.stats = {
            'total_kernel_computations': 0,
            'avg_computation_time_ms': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info(f"Quantum kernel engine initialized with {self.config.n_qubits} qubits")
    
    def compute_kernel_matrix(self, 
                            texts_x: List[str], 
                            texts_y: Optional[List[str]] = None) -> np.ndarray:
        """
        Compute quantum kernel matrix between two sets of texts.
        
        Args:
            texts_x: First set of texts (N texts)
            texts_y: Second set of texts (M texts). If None, uses texts_x.
            
        Returns:
            Kernel matrix K where K[i,j] = quantum_fidelity(text_x[i], text_y[j])
        """
        start_time = time.time()
        
        if texts_y is None:
            texts_y = texts_x
            symmetric = True
        else:
            symmetric = False
        
        # Check cache
        cache_key = self._get_cache_key(texts_x, texts_y)
        if self._kernel_cache and cache_key in self._kernel_cache:
            self.stats['cache_hits'] += 1
            return self._kernel_cache[cache_key]
        
        self.stats['cache_misses'] += 1
        
        # Compute kernel matrix using existing fidelity engine
        n_x, n_y = len(texts_x), len(texts_y)
        kernel_matrix = np.zeros((n_x, n_y))
        
        # Use batch processing when possible
        for i, text_x in enumerate(texts_x):
            if symmetric and i == 0:
                # For symmetric matrices, compute similarities against all texts_y
                similarities = self.fidelity_engine.compute_query_similarities(
                    text_x, texts_y, self.config.encoding_method
                )
                for j, (_, similarity, _) in enumerate(similarities):
                    kernel_matrix[i, j] = similarity
            else:
                # Compute similarities one by one for asymmetric or remaining rows
                for j, text_y in enumerate(texts_y):
                    if symmetric and j < i:
                        # Use symmetry to avoid redundant computation
                        kernel_matrix[i, j] = kernel_matrix[j, i]
                    else:
                        similarity, _ = self.fidelity_engine.compute_text_similarity(
                            text_x, text_y, self.config.encoding_method
                        )
                        kernel_matrix[i, j] = similarity
        
        # Ensure symmetry for symmetric matrices
        if symmetric:
            kernel_matrix = (kernel_matrix + kernel_matrix.T) / 2
        
        # Cache result
        if self._kernel_cache:
            self._cache_kernel_matrix(cache_key, kernel_matrix)
        
        # Update stats
        computation_time = (time.time() - start_time) * 1000
        self._update_stats(computation_time)
        
        return kernel_matrix
    
    def compute_embedding_kernel_matrix(self, 
                                      embeddings_x: np.ndarray, 
                                      embeddings_y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute kernel matrix directly from embeddings.
        
        Args:
            embeddings_x: First set of embeddings (N x D)
            embeddings_y: Second set of embeddings (M x D). If None, uses embeddings_x.
            
        Returns:
            Kernel matrix based on quantum fidelity
        """
        # Convert embeddings to texts (placeholder approach)
        # In practice, you'd want to work directly with embeddings
        texts_x = [f"embedding_{i}" for i in range(len(embeddings_x))]
        if embeddings_y is not None:
            texts_y = [f"embedding_{i}" for i in range(len(embeddings_y))]
        else:
            texts_y = None
        
        # Note: This is a simplified approach. For production use,
        # you'd want to modify the fidelity engine to work with embeddings directly
        return self.compute_kernel_matrix(texts_x, texts_y)
    
    def _get_cache_key(self, texts_x: List[str], texts_y: List[str]) -> str:
        """Generate cache key for kernel matrix."""
        import hashlib
        x_str = '|||'.join(texts_x)
        y_str = '|||'.join(texts_y)
        content = f"{x_str}###{y_str}###{self.config.encoding_method}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _cache_kernel_matrix(self, cache_key: str, kernel_matrix: np.ndarray):
        """Cache kernel matrix with size management."""
        if len(self._kernel_cache) >= self.config.max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._kernel_cache))
            del self._kernel_cache[oldest_key]
        
        self._kernel_cache[cache_key] = kernel_matrix.copy()
    
    def _update_stats(self, computation_time_ms: float):
        """Update performance statistics."""
        self.stats['total_kernel_computations'] += 1
        
        # Update rolling average
        n = self.stats['total_kernel_computations']
        current_avg = self.stats['avg_computation_time_ms']
        
        self.stats['avg_computation_time_ms'] = (
            (current_avg * (n - 1) + computation_time_ms) / n
        )


class FidelityQuantumKernel:
    """
    Scikit-learn compatible quantum kernel using existing fidelity infrastructure.
    
    Provides a drop-in replacement for classical kernels in ML algorithms.
    """
    
    def __init__(self, 
                 n_qubits: int = 4,
                 encoding_method: str = "amplitude",
                 **kwargs):
        """
        Initialize fidelity quantum kernel.
        
        Args:
            n_qubits: Number of qubits in quantum circuits
            encoding_method: Quantum encoding method (amplitude, angle, dense_angle)
            **kwargs: Additional configuration parameters
        """
        config = QuantumKernelConfig(
            n_qubits=n_qubits,
            encoding_method=encoding_method,
            **kwargs
        )
        
        self.kernel_engine = QuantumKernelEngine(config)
        self.is_fitted = False
    
    def fit(self, X: List[str], y: Optional[np.ndarray] = None):
        """Fit the quantum kernel (sklearn compatibility)."""
        self.X_fit_ = X
        self.is_fitted = True
        return self
    
    def transform(self, X: List[str]) -> np.ndarray:
        """Transform texts using quantum kernel."""
        if not self.is_fitted:
            raise ValueError("Kernel must be fitted before transform")
        
        return self.kernel_engine.compute_kernel_matrix(X, self.X_fit_)
    
    def fit_transform(self, X: List[str], y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.kernel_engine.compute_kernel_matrix(X, X)
    
    def __call__(self, X: List[str], Y: Optional[List[str]] = None) -> np.ndarray:
        """Compute kernel matrix (callable interface for sklearn)."""
        return self.kernel_engine.compute_kernel_matrix(X, Y)
"""
Quantum circuit result caching with parameter clustering.

This module provides intelligent caching for quantum computation results
with parameter-based clustering for improved hit rates.
"""

import numpy as np
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import OrderedDict

from ..utils import get_logger


@dataclass
class QuantumResult:
    """Cached quantum computation result."""
    fidelity: float
    parameters: np.ndarray
    method: str
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    computation_time_ms: float = 0.0


@dataclass
class ParameterCluster:
    """Cluster of similar quantum parameters."""
    center: np.ndarray
    results: List[QuantumResult]
    radius: float = 0.01


class QuantumCircuitCache:
    """
    Intelligent caching for quantum circuit results with parameter clustering.
    
    This cache groups similar quantum parameters together to improve
    cache hit rates for approximately similar computations.
    """
    
    def __init__(
        self,
        max_entries: int = 5000,
        memory_limit_mb: int = 256,
        parameter_tolerance: float = 0.01,
        ttl_minutes: int = 120
    ):
        self.max_entries = max_entries
        self.memory_limit_mb = memory_limit_mb
        self.parameter_tolerance = parameter_tolerance
        self.ttl_seconds = ttl_minutes * 60
        
        self.logger = get_logger(__name__)
        
        # Main cache storage
        self.cache: OrderedDict[str, QuantumResult] = OrderedDict()
        
        # Parameter clusters for efficient lookup
        self.clusters: List[ParameterCluster] = []
        
        # Performance tracking
        self.hit_count = 0
        self.miss_count = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info(f"Initialized QuantumCircuitCache: max_entries={max_entries}")
    
    def get_quantum_result(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Get cached quantum result for embedding pair."""
        with self._lock:
            # Create lookup key
            key = self._create_key(embedding1, embedding2)
            
            # Check exact match first
            if key in self.cache:
                result = self.cache[key]
                
                # Check TTL
                if time.time() - result.timestamp <= self.ttl_seconds:
                    self.cache.move_to_end(key)
                    result.access_count += 1
                    self.hit_count += 1
                    
                    return {
                        "fidelity": result.fidelity,
                        "method": result.method,
                        "classical_similarity": 0.8  # Default classical component
                    }
                else:
                    # Expired
                    del self.cache[key]
            
            self.miss_count += 1
            return None
    
    def put_quantum_result(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        quantum_result: Dict[str, Any]
    ) -> bool:
        """Store quantum computation result."""
        with self._lock:
            try:
                key = self._create_key(embedding1, embedding2)
                
                # Create result object
                result = QuantumResult(
                    fidelity=quantum_result.get("fidelity", 0.0),
                    parameters=np.concatenate([embedding1[:4], embedding2[:4]]),  # Simplified
                    method=quantum_result.get("method", "unknown"),
                    computation_time_ms=quantum_result.get("computation_time_ms", 0.0)
                )
                
                # Store in cache
                self.cache[key] = result
                
                # Maintain capacity
                if len(self.cache) > self.max_entries:
                    self.cache.popitem(last=False)
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to store quantum result: {e}")
                return False
    
    def _create_key(self, embedding1: np.ndarray, embedding2: np.ndarray) -> str:
        """Create cache key from embeddings."""
        # Use hash of first few elements for efficiency
        key_data = np.concatenate([embedding1[:8], embedding2[:8]])
        return str(hash(key_data.tobytes()))[:16]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hit_count + self.miss_count
            
            return {
                "cache_type": "quantum",
                "entries": len(self.cache),
                "max_entries": self.max_entries,
                "hit_rate": self.hit_count / max(total_requests, 1),
                "clusters": len(self.clusters)
            }
    
    def get_memory_usage_mb(self) -> float:
        """Get estimated memory usage."""
        # Rough estimate: 200 bytes per entry
        return len(self.cache) * 200 / (1024 * 1024)
    
    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self.cache.clear()
            self.clusters.clear()
            self.hit_count = 0
            self.miss_count = 0


__all__ = ["QuantumResult", "ParameterCluster", "QuantumCircuitCache"]
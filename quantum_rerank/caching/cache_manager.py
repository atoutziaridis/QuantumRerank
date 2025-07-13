"""
Advanced Cache Manager for coordinated multi-level caching.

This module provides the main caching coordination system that manages
similarity, quantum, and embedding caches with intelligent optimization.
"""

import threading
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from ..config import Configurable, QuantumRerankConfigSchema
from ..utils import get_logger, QuantumRerankException
from .similarity_cache import SimilarityCache
from .quantum_cache import QuantumCircuitCache
from .embedding_cache import EmbeddingCache
from .cache_monitor import CachePerformanceMonitor
from .cache_optimizer import CacheOptimizer


class CacheLevel(Enum):
    """Cache hierarchy levels."""
    L1_MEMORY = "l1_memory"      # Hot in-memory cache
    L2_WARM = "l2_warm"          # Warm memory cache
    L3_PERSISTENT = "l3_persistent"  # Persistent disk cache
    L4_DISTRIBUTED = "l4_distributed"  # Distributed cache (future)


@dataclass
class CacheConfig:
    """Configuration for cache system."""
    # Similarity cache config
    similarity_max_entries: int = 10000
    similarity_memory_limit_mb: int = 512
    similarity_approximate_threshold: float = 0.95
    similarity_ttl_minutes: int = 60
    
    # Quantum cache config
    quantum_max_entries: int = 5000
    quantum_memory_limit_mb: int = 256
    quantum_parameter_tolerance: float = 0.01
    quantum_ttl_minutes: int = 120
    
    # Embedding cache config
    embedding_max_entries: int = 50000
    embedding_memory_limit_mb: int = 1024
    embedding_disk_cache_gb: float = 5.0
    embedding_ttl_hours: int = 24
    
    # Performance targets
    target_hit_rate_similarity: float = 0.25
    target_hit_rate_quantum: float = 0.15
    target_hit_rate_embedding: float = 0.60
    target_lookup_latency_ms: float = 2.0
    target_memory_limit_gb: float = 1.0
    
    # Optimization settings
    enable_approximate_matching: bool = True
    enable_cache_optimization: bool = True
    enable_performance_monitoring: bool = True
    optimization_interval_minutes: int = 15


@dataclass
class CacheOperationResult:
    """Result of a cache operation."""
    success: bool
    cached_result: Any = None
    cache_hit: bool = False
    cache_level: Optional[str] = None
    latency_ms: float = 0.0
    error_message: Optional[str] = None


class AdvancedCacheManager(Configurable):
    """
    Coordinated multi-level caching system for QuantumRerank.
    
    This manager orchestrates similarity, quantum, and embedding caches
    with intelligent optimization and performance monitoring.
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.logger = get_logger(__name__)
        
        # Initialize cache components
        self.caches = self._initialize_caches()
        
        # Initialize monitoring and optimization
        self.monitor = CachePerformanceMonitor(
            target_hit_rates={
                "similarity": self.config.target_hit_rate_similarity,
                "quantum": self.config.target_hit_rate_quantum,
                "embedding": self.config.target_hit_rate_embedding
            }
        )
        
        self.optimizer = CacheOptimizer(
            target_memory_gb=self.config.target_memory_limit_gb,
            optimization_enabled=self.config.enable_cache_optimization
        )
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background optimization
        self._optimization_thread = None
        self._stop_optimization = threading.Event()
        
        if self.config.enable_cache_optimization:
            self._start_background_optimization()
        
        self.logger.info("Initialized AdvancedCacheManager with multi-level caching")
    
    def _initialize_caches(self) -> Dict[str, Any]:
        """Initialize all cache components."""
        caches = {}
        
        # Similarity cache
        caches["similarity"] = SimilarityCache(
            max_entries=self.config.similarity_max_entries,
            memory_limit_mb=self.config.similarity_memory_limit_mb,
            approximate_threshold=self.config.similarity_approximate_threshold,
            ttl_minutes=self.config.similarity_ttl_minutes,
            enable_approximate_matching=self.config.enable_approximate_matching
        )
        
        # Quantum circuit cache
        caches["quantum"] = QuantumCircuitCache(
            max_entries=self.config.quantum_max_entries,
            memory_limit_mb=self.config.quantum_memory_limit_mb,
            parameter_tolerance=self.config.quantum_parameter_tolerance,
            ttl_minutes=self.config.quantum_ttl_minutes
        )
        
        # Embedding cache
        caches["embedding"] = EmbeddingCache(
            max_entries=self.config.embedding_max_entries,
            memory_limit_mb=self.config.embedding_memory_limit_mb,
            disk_cache_gb=self.config.embedding_disk_cache_gb,
            ttl_hours=self.config.embedding_ttl_hours
        )
        
        return caches
    
    def get_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        method: str,
        accuracy_requirement: float = 0.9
    ) -> CacheOperationResult:
        """
        Get similarity with intelligent caching hierarchy.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            method: Similarity computation method
            accuracy_requirement: Required accuracy level
            
        Returns:
            CacheOperationResult with similarity and cache information
        """
        start_time = time.time()
        
        try:
            with self._lock:
                # Level 1: Check similarity cache (exact match)
                cached_similarity = self.caches["similarity"].get_exact_match(
                    embedding1, embedding2, method
                )
                
                if cached_similarity is not None:
                    latency_ms = (time.time() - start_time) * 1000
                    self.monitor.record_cache_hit("similarity", "exact", latency_ms)
                    
                    return CacheOperationResult(
                        success=True,
                        cached_result=cached_similarity,
                        cache_hit=True,
                        cache_level="similarity_exact",
                        latency_ms=latency_ms
                    )
                
                # Level 2: Check similarity cache (approximate match)
                if self.config.enable_approximate_matching:
                    approx_similarity = self.caches["similarity"].get_approximate_match(
                        embedding1, embedding2, method, accuracy_requirement
                    )
                    
                    if approx_similarity is not None:
                        latency_ms = (time.time() - start_time) * 1000
                        self.monitor.record_cache_hit("similarity", "approximate", latency_ms)
                        
                        return CacheOperationResult(
                            success=True,
                            cached_result=approx_similarity,
                            cache_hit=True,
                            cache_level="similarity_approximate",
                            latency_ms=latency_ms
                        )
                
                # Level 3: Check quantum cache for quantum/hybrid methods
                if method in ["quantum_precise", "quantum_approximate", "hybrid_balanced", "hybrid_batch"]:
                    quantum_result = self.caches["quantum"].get_quantum_result(
                        embedding1, embedding2
                    )
                    
                    if quantum_result is not None:
                        # Compute similarity from cached quantum result
                        similarity = self._compute_similarity_from_quantum_cache(
                            quantum_result, method
                        )
                        
                        if similarity is not None:
                            # Cache the computed similarity for future use
                            self.caches["similarity"].put(
                                embedding1, embedding2, method, similarity
                            )
                            
                            latency_ms = (time.time() - start_time) * 1000
                            self.monitor.record_cache_hit("quantum", "derived", latency_ms)
                            
                            return CacheOperationResult(
                                success=True,
                                cached_result=similarity,
                                cache_hit=True,
                                cache_level="quantum_derived",
                                latency_ms=latency_ms
                            )
                
                # Cache miss - record and return
                latency_ms = (time.time() - start_time) * 1000
                self.monitor.record_cache_miss("similarity", method, latency_ms)
                
                return CacheOperationResult(
                    success=True,
                    cached_result=None,
                    cache_hit=False,
                    latency_ms=latency_ms
                )
        
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Cache lookup failed: {e}")
            
            return CacheOperationResult(
                success=False,
                cache_hit=False,
                latency_ms=latency_ms,
                error_message=str(e)
            )
    
    def put_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        method: str,
        similarity: float,
        quantum_result: Optional[Dict[str, Any]] = None,
        computation_time_ms: Optional[float] = None
    ) -> CacheOperationResult:
        """
        Store similarity result with intelligent caching strategy.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            method: Similarity computation method
            similarity: Computed similarity value
            quantum_result: Optional quantum computation intermediate results
            computation_time_ms: Time taken for computation
            
        Returns:
            CacheOperationResult indicating success
        """
        start_time = time.time()
        
        try:
            with self._lock:
                # Store similarity result
                similarity_stored = self.caches["similarity"].put(
                    embedding1, embedding2, method, similarity
                )
                
                # Store quantum intermediate results if available
                quantum_stored = False
                if quantum_result and method in ["quantum_precise", "quantum_approximate", "hybrid_balanced", "hybrid_batch"]:
                    quantum_stored = self.caches["quantum"].put_quantum_result(
                        embedding1, embedding2, quantum_result
                    )
                
                # Update performance metrics
                store_latency_ms = (time.time() - start_time) * 1000
                self.monitor.record_cache_store(
                    method, store_latency_ms, computation_time_ms
                )
                
                # Trigger optimization if needed
                if self.optimizer.should_optimize():
                    self._trigger_optimization()
                
                return CacheOperationResult(
                    success=similarity_stored,
                    latency_ms=store_latency_ms
                )
        
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Cache store failed: {e}")
            
            return CacheOperationResult(
                success=False,
                latency_ms=latency_ms,
                error_message=str(e)
            )
    
    def get_embedding(
        self,
        text: str,
        model_name: str
    ) -> CacheOperationResult:
        """Get cached embedding for text."""
        start_time = time.time()
        
        try:
            cached_embedding = self.caches["embedding"].get(text, model_name)
            
            latency_ms = (time.time() - start_time) * 1000
            
            if cached_embedding is not None:
                self.monitor.record_cache_hit("embedding", "exact", latency_ms)
                return CacheOperationResult(
                    success=True,
                    cached_result=cached_embedding,
                    cache_hit=True,
                    cache_level="embedding",
                    latency_ms=latency_ms
                )
            else:
                self.monitor.record_cache_miss("embedding", model_name, latency_ms)
                return CacheOperationResult(
                    success=True,
                    cached_result=None,
                    cache_hit=False,
                    latency_ms=latency_ms
                )
        
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return CacheOperationResult(
                success=False,
                latency_ms=latency_ms,
                error_message=str(e)
            )
    
    def put_embedding(
        self,
        text: str,
        model_name: str,
        embedding: np.ndarray
    ) -> CacheOperationResult:
        """Store embedding in cache."""
        start_time = time.time()
        
        try:
            success = self.caches["embedding"].put(text, model_name, embedding)
            latency_ms = (time.time() - start_time) * 1000
            
            self.monitor.record_cache_store("embedding", latency_ms)
            
            return CacheOperationResult(
                success=success,
                latency_ms=latency_ms
            )
        
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return CacheOperationResult(
                success=False,
                latency_ms=latency_ms,
                error_message=str(e)
            )
    
    def _compute_similarity_from_quantum_cache(
        self,
        quantum_result: Dict[str, Any],
        method: str
    ) -> Optional[float]:
        """Compute similarity from cached quantum result."""
        try:
            if "fidelity" in quantum_result:
                fidelity = quantum_result["fidelity"]
                
                if method == "quantum_precise":
                    return fidelity
                elif method == "quantum_approximate":
                    return fidelity * 0.95  # Slight adjustment for approximation
                elif method.startswith("hybrid"):
                    # Hybrid methods combine quantum and classical
                    classical_component = quantum_result.get("classical_similarity", 0.8)
                    return 0.5 * fidelity + 0.5 * classical_component
                
            return None
        
        except Exception as e:
            self.logger.warning(f"Failed to compute similarity from quantum cache: {e}")
            return None
    
    def _start_background_optimization(self) -> None:
        """Start background cache optimization thread."""
        def optimization_loop():
            while not self._stop_optimization.wait(self.config.optimization_interval_minutes * 60):
                try:
                    self.optimize_caches()
                except Exception as e:
                    self.logger.error(f"Background cache optimization failed: {e}")
        
        self._optimization_thread = threading.Thread(
            target=optimization_loop,
            daemon=True,
            name="CacheOptimization"
        )
        self._optimization_thread.start()
        self.logger.info("Started background cache optimization")
    
    def _trigger_optimization(self) -> None:
        """Trigger immediate cache optimization."""
        try:
            self.optimize_caches()
        except Exception as e:
            self.logger.error(f"Cache optimization failed: {e}")
    
    def optimize_caches(self) -> Dict[str, Any]:
        """Optimize all caches for better performance."""
        optimization_results = {}
        
        try:
            with self._lock:
                # Get current performance metrics
                metrics = self.monitor.get_performance_summary()
                
                # Optimize each cache
                for cache_name, cache in self.caches.items():
                    if hasattr(cache, 'optimize'):
                        cache_metrics = metrics.get(cache_name, {})
                        optimization_result = self.optimizer.optimize_cache(
                            cache, cache_metrics
                        )
                        optimization_results[cache_name] = optimization_result
                
                # Global optimization strategies
                global_optimization = self.optimizer.optimize_global_strategy(
                    self.caches, metrics
                )
                optimization_results["global"] = global_optimization
                
                self.logger.debug(f"Cache optimization completed: {optimization_results}")
        
        except Exception as e:
            self.logger.error(f"Cache optimization failed: {e}")
            optimization_results["error"] = str(e)
        
        return optimization_results
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            "performance_summary": self.monitor.get_performance_summary(),
            "cache_details": {},
            "system_health": self.get_cache_health()
        }
        
        # Get detailed statistics for each cache
        for cache_name, cache in self.caches.items():
            if hasattr(cache, 'get_statistics'):
                stats["cache_details"][cache_name] = cache.get_statistics()
        
        return stats
    
    def get_cache_health(self) -> Dict[str, Any]:
        """Get overall cache system health."""
        health = {
            "overall_status": "healthy",
            "issues": [],
            "recommendations": []
        }
        
        try:
            # Check hit rates
            metrics = self.monitor.get_performance_summary()
            
            for cache_type in ["similarity", "quantum", "embedding"]:
                hit_rate = metrics.get(f"{cache_type}_hit_rate", 0)
                target_attr = f"target_hit_rate_{cache_type}"
                target = getattr(self.config, target_attr, 0.2)
                
                if hit_rate < target * 0.8:  # 80% of target
                    health["issues"].append(f"Low hit rate for {cache_type}: {hit_rate:.2%}")
                    health["recommendations"].append(f"Consider increasing {cache_type} cache size")
            
            # Check memory usage
            total_memory_mb = sum(
                cache.get_memory_usage_mb() 
                for cache in self.caches.values()
                if hasattr(cache, 'get_memory_usage_mb')
            )
            
            if total_memory_mb > self.config.target_memory_limit_gb * 1024:
                health["issues"].append(f"High memory usage: {total_memory_mb:.0f}MB")
                health["recommendations"].append("Consider reducing cache sizes or enabling compression")
            
            # Determine overall status
            if len(health["issues"]) > 2:
                health["overall_status"] = "degraded"
            elif len(health["issues"]) > 0:
                health["overall_status"] = "warning"
        
        except Exception as e:
            health["overall_status"] = "error"
            health["issues"].append(f"Health check failed: {str(e)}")
        
        return health
    
    def clear_caches(self, cache_types: Optional[List[str]] = None) -> Dict[str, bool]:
        """Clear specified caches or all caches."""
        results = {}
        
        with self._lock:
            target_caches = cache_types or list(self.caches.keys())
            
            for cache_name in target_caches:
                if cache_name in self.caches:
                    try:
                        self.caches[cache_name].clear()
                        results[cache_name] = True
                        self.logger.info(f"Cleared {cache_name} cache")
                    except Exception as e:
                        results[cache_name] = False
                        self.logger.error(f"Failed to clear {cache_name} cache: {e}")
                else:
                    results[cache_name] = False
                    self.logger.warning(f"Unknown cache type: {cache_name}")
        
        return results
    
    def configure(self, config: QuantumRerankConfigSchema) -> None:
        """Configure cache manager with new settings."""
        # Update cache configurations based on global config
        if hasattr(config, 'performance'):
            perf_config = config.performance
            
            # Update memory limits
            if hasattr(perf_config, 'max_memory_gb'):
                self.config.target_memory_limit_gb = perf_config.max_memory_gb * 0.3  # 30% for caching
            
            # Update performance targets
            if hasattr(perf_config, 'similarity_timeout_ms'):
                self.config.target_lookup_latency_ms = min(5.0, perf_config.similarity_timeout_ms * 0.05)
    
    def validate_config(self, config: QuantumRerankConfigSchema) -> bool:
        """Validate configuration for cache manager."""
        # Check if performance configuration exists
        if not hasattr(config, 'performance'):
            return False
        
        return True
    
    def get_config_requirements(self) -> List[str]:
        """Get required configuration sections."""
        return ["performance"]
    
    def __del__(self):
        """Cleanup cache manager resources."""
        if hasattr(self, '_stop_optimization'):
            self._stop_optimization.set()
        
        if hasattr(self, '_optimization_thread') and self._optimization_thread:
            self._optimization_thread.join(timeout=1.0)


__all__ = [
    "CacheLevel",
    "CacheConfig",
    "CacheOperationResult",
    "AdvancedCacheManager"
]
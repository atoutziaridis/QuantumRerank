"""
Similarity computation result caching with intelligent matching.

This module provides caching for similarity computations with exact and
approximate matching capabilities for improved performance.
"""

import hashlib
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import OrderedDict
import numpy as np

from ..utils import get_logger
from .approximate_matcher import ApproximateSimilarityMatcher


@dataclass
class SimilarityResult:
    """Cached similarity result with metadata."""
    similarity: float
    method: str
    embedding1_hash: str
    embedding2_hash: str
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    computation_time_ms: float = 0.0
    confidence: float = 1.0
    
    def update_access(self) -> None:
        """Update access information."""
        self.access_count += 1
        # Could update timestamp for LRU tracking


@dataclass
class CacheKey:
    """Cache key for similarity computations."""
    embedding1_hash: str
    embedding2_hash: str
    method: str
    
    def __str__(self) -> str:
        return f"{self.embedding1_hash}:{self.embedding2_hash}:{self.method}"
    
    def __hash__(self) -> int:
        return hash(str(self))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, CacheKey):
            return False
        return str(self) == str(other)


class SimilarityCache:
    """
    Intelligent caching system for similarity computation results.
    
    This cache provides both exact and approximate matching capabilities
    with LRU eviction and performance weighting.
    """
    
    def __init__(
        self,
        max_entries: int = 10000,
        memory_limit_mb: int = 512,
        approximate_threshold: float = 0.95,
        ttl_minutes: int = 60,
        enable_approximate_matching: bool = True
    ):
        self.max_entries = max_entries
        self.memory_limit_mb = memory_limit_mb
        self.approximate_threshold = approximate_threshold
        self.ttl_seconds = ttl_minutes * 60
        self.enable_approximate_matching = enable_approximate_matching
        
        self.logger = get_logger(__name__)
        
        # Cache storage - using OrderedDict for LRU behavior
        self.exact_cache: OrderedDict[CacheKey, SimilarityResult] = OrderedDict()
        
        # Approximate matching engine
        self.approximate_matcher = None
        if enable_approximate_matching:
            self.approximate_matcher = ApproximateSimilarityMatcher(
                similarity_threshold=approximate_threshold
            )
        
        # Performance tracking
        self.hit_count = 0
        self.miss_count = 0
        self.approximate_hit_count = 0
        self.total_lookups = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Memory usage tracking
        self._estimated_memory_bytes = 0
        
        self.logger.info(
            f"Initialized SimilarityCache: max_entries={max_entries}, "
            f"memory_limit={memory_limit_mb}MB, approximate_matching={enable_approximate_matching}"
        )
    
    def get_exact_match(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        method: str
    ) -> Optional[float]:
        """Get exact match from cache."""
        with self._lock:
            self.total_lookups += 1
            
            # Create cache key
            key = self._create_cache_key(embedding1, embedding2, method)
            
            # Check exact cache
            if key in self.exact_cache:
                result = self.exact_cache[key]
                
                # Check TTL
                if time.time() - result.timestamp <= self.ttl_seconds:
                    # Move to end for LRU
                    self.exact_cache.move_to_end(key)
                    result.update_access()
                    
                    self.hit_count += 1
                    self.logger.debug(f"Exact cache hit for {method}")
                    
                    return result.similarity
                else:
                    # Expired entry
                    self._remove_entry(key)
            
            return None
    
    def get_approximate_match(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        method: str,
        accuracy_requirement: float = 0.9
    ) -> Optional[float]:
        """Get approximate match from cache."""
        if not self.enable_approximate_matching or not self.approximate_matcher:
            return None
        
        with self._lock:
            # Try approximate matching
            approximate_result = self.approximate_matcher.find_approximate_match(
                embedding1, embedding2, method, self.exact_cache, accuracy_requirement
            )
            
            if approximate_result is not None:
                self.approximate_hit_count += 1
                self.logger.debug(f"Approximate cache hit for {method}")
                return approximate_result
            
            return None
    
    def put(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        method: str,
        similarity: float,
        computation_time_ms: float = 0.0
    ) -> bool:
        """Store similarity result in cache."""
        with self._lock:
            try:
                # Create cache key and result
                key = self._create_cache_key(embedding1, embedding2, method)
                result = SimilarityResult(
                    similarity=similarity,
                    method=method,
                    embedding1_hash=key.embedding1_hash,
                    embedding2_hash=key.embedding2_hash,
                    computation_time_ms=computation_time_ms
                )
                
                # Check if update or new entry
                if key in self.exact_cache:
                    # Update existing entry
                    old_result = self.exact_cache[key]
                    result.access_count = old_result.access_count
                    self.exact_cache[key] = result
                    self.exact_cache.move_to_end(key)
                else:
                    # New entry - check capacity
                    self._ensure_capacity()
                    self.exact_cache[key] = result
                    self._update_memory_estimate(key, result, added=True)
                
                # Update approximate matcher if enabled
                if self.approximate_matcher:
                    self.approximate_matcher.add_to_index(
                        embedding1, embedding2, method, similarity
                    )
                
                return True
            
            except Exception as e:
                self.logger.error(f"Failed to store similarity result: {e}")
                return False
    
    def _create_cache_key(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        method: str
    ) -> CacheKey:
        """Create cache key from embeddings and method."""
        # Normalize embedding order for consistent caching
        if self._embedding_hash(embedding1) > self._embedding_hash(embedding2):
            embedding1, embedding2 = embedding2, embedding1
        
        return CacheKey(
            embedding1_hash=self._embedding_hash(embedding1),
            embedding2_hash=self._embedding_hash(embedding2),
            method=method
        )
    
    def _embedding_hash(self, embedding: np.ndarray) -> str:
        """Create hash from embedding."""
        # Use a subset of embedding values for efficiency
        # Take every 8th value to create a representative hash
        subset = embedding[::8]
        
        # Round to reduce sensitivity to small numerical differences
        rounded = np.round(subset, decimals=4)
        
        # Create hash
        return hashlib.md5(rounded.tobytes()).hexdigest()[:16]
    
    def _ensure_capacity(self) -> None:
        """Ensure cache capacity constraints are met."""
        # Check entry count limit
        while len(self.exact_cache) >= self.max_entries:
            self._evict_entry()
        
        # Check memory limit
        while self._estimated_memory_bytes > self.memory_limit_mb * 1024 * 1024:
            self._evict_entry()
    
    def _evict_entry(self) -> None:
        """Evict least recently used entry."""
        if not self.exact_cache:
            return
        
        # Use weighted LRU - consider both recency and access frequency
        if len(self.exact_cache) > 100:
            # For large caches, use more sophisticated eviction
            self._evict_weighted_lru()
        else:
            # Simple LRU for small caches
            key, result = self.exact_cache.popitem(last=False)
            self._update_memory_estimate(key, result, added=False)
    
    def _evict_weighted_lru(self) -> None:
        """Evict entry using weighted LRU strategy."""
        current_time = time.time()
        min_score = float('inf')
        evict_key = None
        
        # Evaluate first 10 entries (oldest)
        items_to_check = list(self.exact_cache.items())[:10]
        
        for key, result in items_to_check:
            # Weighted score: recency + access frequency + computation cost
            age_hours = (current_time - result.timestamp) / 3600
            access_weight = min(result.access_count, 10)  # Cap at 10
            computation_weight = min(result.computation_time_ms / 100, 5)  # Cap at 5
            
            # Lower score = higher eviction priority
            score = access_weight + computation_weight - age_hours * 0.1
            
            if score < min_score:
                min_score = score
                evict_key = key
        
        if evict_key:
            result = self.exact_cache.pop(evict_key)
            self._update_memory_estimate(evict_key, result, added=False)
    
    def _remove_entry(self, key: CacheKey) -> None:
        """Remove specific entry from cache."""
        if key in self.exact_cache:
            result = self.exact_cache.pop(key)
            self._update_memory_estimate(key, result, added=False)
    
    def _update_memory_estimate(
        self,
        key: CacheKey,
        result: SimilarityResult,
        added: bool
    ) -> None:
        """Update memory usage estimate."""
        # Rough estimate: key (64 bytes) + result (128 bytes)
        entry_size = 192
        
        if added:
            self._estimated_memory_bytes += entry_size
        else:
            self._estimated_memory_bytes = max(0, self._estimated_memory_bytes - entry_size)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            total_requests = self.hit_count + self.miss_count
            
            stats = {
                "cache_type": "similarity",
                "capacity": {
                    "max_entries": self.max_entries,
                    "current_entries": len(self.exact_cache),
                    "memory_limit_mb": self.memory_limit_mb,
                    "estimated_memory_mb": self._estimated_memory_bytes / (1024 * 1024)
                },
                "performance": {
                    "total_lookups": self.total_lookups,
                    "exact_hits": self.hit_count,
                    "approximate_hits": self.approximate_hit_count,
                    "misses": self.miss_count,
                    "exact_hit_rate": self.hit_count / max(total_requests, 1),
                    "total_hit_rate": (self.hit_count + self.approximate_hit_count) / max(self.total_lookups, 1)
                },
                "configuration": {
                    "ttl_minutes": self.ttl_seconds / 60,
                    "approximate_threshold": self.approximate_threshold,
                    "approximate_matching_enabled": self.enable_approximate_matching
                }
            }
            
            # Add access pattern analysis
            if self.exact_cache:
                access_counts = [result.access_count for result in self.exact_cache.values()]
                stats["access_patterns"] = {
                    "avg_access_count": np.mean(access_counts),
                    "max_access_count": np.max(access_counts),
                    "highly_accessed_entries": sum(1 for count in access_counts if count > 5)
                }
        
        return stats
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        with self._lock:
            return self._estimated_memory_bytes / (1024 * 1024)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.exact_cache.clear()
            self._estimated_memory_bytes = 0
            
            if self.approximate_matcher:
                self.approximate_matcher.clear_index()
            
            # Reset statistics
            self.hit_count = 0
            self.miss_count = 0
            self.approximate_hit_count = 0
            self.total_lookups = 0
            
            self.logger.info("Similarity cache cleared")
    
    def optimize(self) -> Dict[str, Any]:
        """Optimize cache performance."""
        optimization_results = {}
        
        with self._lock:
            initial_size = len(self.exact_cache)
            
            # Remove expired entries
            current_time = time.time()
            expired_keys = [
                key for key, result in self.exact_cache.items()
                if current_time - result.timestamp > self.ttl_seconds
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
            
            expired_count = len(expired_keys)
            optimization_results["expired_entries_removed"] = expired_count
            
            # Optimize approximate matcher if enabled
            if self.approximate_matcher:
                matcher_optimization = self.approximate_matcher.optimize()
                optimization_results["approximate_matcher"] = matcher_optimization
            
            # Analyze access patterns and adjust if needed
            if self.exact_cache:
                access_counts = [result.access_count for result in self.exact_cache.values()]
                avg_access = np.mean(access_counts)
                
                # Remove rarely accessed entries if memory pressure is high
                if self._estimated_memory_bytes > self.memory_limit_mb * 1024 * 1024 * 0.9:
                    low_access_threshold = max(1, avg_access * 0.3)
                    low_access_keys = [
                        key for key, result in self.exact_cache.items()
                        if result.access_count < low_access_threshold
                    ]
                    
                    # Remove up to 10% of low-access entries
                    removal_count = min(len(low_access_keys), len(self.exact_cache) // 10)
                    for key in low_access_keys[:removal_count]:
                        self._remove_entry(key)
                    
                    optimization_results["low_access_entries_removed"] = removal_count
            
            final_size = len(self.exact_cache)
            optimization_results["size_reduction"] = initial_size - final_size
            optimization_results["final_cache_size"] = final_size
        
        return optimization_results


__all__ = [
    "SimilarityResult",
    "CacheKey", 
    "SimilarityCache"
]
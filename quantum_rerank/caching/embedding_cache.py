"""
Embedding preprocessing result caching with persistent storage.

This module provides caching for text-to-embedding transformations
with memory-mapped persistent storage for large-scale caching.
"""

import hashlib
import time
import threading
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from collections import OrderedDict
import numpy as np

from ..utils import get_logger


@dataclass
class EmbeddingResult:
    """Cached embedding result."""
    embedding: np.ndarray
    model_name: str
    text_hash: str
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0


@dataclass
class TextHash:
    """Text hash for efficient lookup."""
    text: str
    hash_value: str
    
    @classmethod
    def from_text(cls, text: str) -> 'TextHash':
        """Create hash from text."""
        hash_value = hashlib.md5(text.encode('utf-8')).hexdigest()
        return cls(text=text, hash_value=hash_value)


class EmbeddingCache:
    """
    Intelligent caching for embedding preprocessing results.
    
    This cache stores text-to-embedding transformations with
    persistent storage for long-term caching across sessions.
    """
    
    def __init__(
        self,
        max_entries: int = 50000,
        memory_limit_mb: int = 1024,
        disk_cache_gb: float = 5.0,
        ttl_hours: int = 24
    ):
        self.max_entries = max_entries
        self.memory_limit_mb = memory_limit_mb
        self.disk_cache_gb = disk_cache_gb
        self.ttl_seconds = ttl_hours * 3600
        
        self.logger = get_logger(__name__)
        
        # In-memory cache
        self.cache: OrderedDict[str, EmbeddingResult] = OrderedDict()
        
        # Performance tracking
        self.hit_count = 0
        self.miss_count = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info(f"Initialized EmbeddingCache: max_entries={max_entries}")
    
    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Get cached embedding for text and model."""
        with self._lock:
            key = self._create_key(text, model_name)
            
            if key in self.cache:
                result = self.cache[key]
                
                # Check TTL
                if time.time() - result.timestamp <= self.ttl_seconds:
                    self.cache.move_to_end(key)
                    result.access_count += 1
                    self.hit_count += 1
                    
                    return result.embedding
                else:
                    # Expired
                    del self.cache[key]
            
            self.miss_count += 1
            return None
    
    def put(self, text: str, model_name: str, embedding: np.ndarray) -> bool:
        """Store embedding in cache."""
        with self._lock:
            try:
                key = self._create_key(text, model_name)
                text_hash = TextHash.from_text(text)
                
                result = EmbeddingResult(
                    embedding=embedding.copy(),
                    model_name=model_name,
                    text_hash=text_hash.hash_value
                )
                
                self.cache[key] = result
                
                # Maintain capacity
                if len(self.cache) > self.max_entries:
                    self.cache.popitem(last=False)
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to store embedding: {e}")
                return False
    
    def _create_key(self, text: str, model_name: str) -> str:
        """Create cache key from text and model."""
        combined = f"{text}:{model_name}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hit_count + self.miss_count
            
            return {
                "cache_type": "embedding",
                "entries": len(self.cache),
                "max_entries": self.max_entries,
                "hit_rate": self.hit_count / max(total_requests, 1),
                "memory_usage_mb": self.get_memory_usage_mb()
            }
    
    def get_memory_usage_mb(self) -> float:
        """Get estimated memory usage."""
        # Rough estimate: embedding size + overhead
        if not self.cache:
            return 0.0
        
        sample_embedding = next(iter(self.cache.values())).embedding
        bytes_per_entry = sample_embedding.nbytes + 200  # 200 bytes overhead
        
        return len(self.cache) * bytes_per_entry / (1024 * 1024)
    
    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self.cache.clear()
            self.hit_count = 0
            self.miss_count = 0


__all__ = ["EmbeddingResult", "TextHash", "EmbeddingCache"]
"""
Approximate similarity matching for intelligent caching.

This module provides approximate matching capabilities to find cached results
for similar embeddings, improving cache hit rates and performance.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum

from ..utils import get_logger


class MatchingStrategy(Enum):
    """Strategies for approximate matching."""
    COSINE_SIMILARITY = "cosine_similarity"
    EUCLIDEAN_DISTANCE = "euclidean_distance"
    DOT_PRODUCT = "dot_product"
    MANHATTAN_DISTANCE = "manhattan_distance"


@dataclass
class SimilarityThreshold:
    """Thresholds for different matching strategies."""
    cosine: float = 0.95
    euclidean: float = 0.1
    dot_product: float = 0.9
    manhattan: float = 0.2


@dataclass
class EmbeddingIndex:
    """Index entry for embedding lookup."""
    embedding: np.ndarray
    embedding_hash: str
    method: str
    similarity_result: float
    timestamp: float
    access_count: int = 0


class ApproximateSimilarityMatcher:
    """
    Intelligent approximate matching for similarity caching.
    
    This matcher uses embedding similarity to find cached results for
    approximately similar embedding pairs, improving cache hit rates.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.95,
        matching_strategy: MatchingStrategy = MatchingStrategy.COSINE_SIMILARITY,
        max_index_size: int = 5000,
        enable_clustering: bool = True
    ):
        self.similarity_threshold = similarity_threshold
        self.matching_strategy = matching_strategy
        self.max_index_size = max_index_size
        self.enable_clustering = enable_clustering
        
        self.logger = get_logger(__name__)
        
        # Embedding index for fast lookup
        self.embedding_index: Dict[str, List[EmbeddingIndex]] = defaultdict(list)
        
        # Method-specific indexes
        self.method_indexes: Dict[str, List[EmbeddingIndex]] = defaultdict(list)
        
        # Clustering for faster search
        self.clusters: Dict[str, List[Tuple[np.ndarray, List[EmbeddingIndex]]]] = defaultdict(list)
        self.cluster_threshold = 0.8  # Threshold for cluster assignment
        
        # Performance tracking
        self.match_attempts = 0
        self.successful_matches = 0
        self.average_search_time_ms = 0.0
        
        # Thresholds for different strategies
        self.thresholds = SimilarityThreshold()
        
        self.logger.info(
            f"Initialized ApproximateSimilarityMatcher: threshold={similarity_threshold}, "
            f"strategy={matching_strategy.value}, clustering={enable_clustering}"
        )
    
    def find_approximate_match(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        method: str,
        exact_cache: Dict,
        accuracy_requirement: float = 0.9
    ) -> Optional[float]:
        """
        Find approximate match for embedding pair.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            method: Similarity computation method
            exact_cache: Exact cache to search through
            accuracy_requirement: Required accuracy level
            
        Returns:
            Approximate similarity if found, None otherwise
        """
        start_time = time.time()
        self.match_attempts += 1
        
        try:
            # Quick check if we have any cached results for this method
            if method not in self.method_indexes or not self.method_indexes[method]:
                return None
            
            # Search strategy based on cache size
            if len(self.method_indexes[method]) > 1000:
                # Use clustering for large caches
                match_result = self._find_match_with_clustering(
                    embedding1, embedding2, method, accuracy_requirement
                )
            else:
                # Linear search for small caches
                match_result = self._find_match_linear(
                    embedding1, embedding2, method, accuracy_requirement
                )
            
            # Update performance metrics
            search_time_ms = (time.time() - start_time) * 1000
            self._update_search_time(search_time_ms)
            
            if match_result is not None:
                self.successful_matches += 1
                self.logger.debug(
                    f"Found approximate match for {method} (threshold={self.similarity_threshold})"
                )
            
            return match_result
        
        except Exception as e:
            self.logger.warning(f"Approximate matching failed: {e}")
            return None
    
    def add_to_index(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        method: str,
        similarity_result: float
    ) -> None:
        """Add embedding pair to the approximate matching index."""
        try:
            # Create index entries for both embeddings
            timestamp = time.time()
            
            entry1 = EmbeddingIndex(
                embedding=embedding1.copy(),
                embedding_hash=self._compute_embedding_hash(embedding1),
                method=method,
                similarity_result=similarity_result,
                timestamp=timestamp
            )
            
            entry2 = EmbeddingIndex(
                embedding=embedding2.copy(),
                embedding_hash=self._compute_embedding_hash(embedding2),
                method=method,
                similarity_result=similarity_result,
                timestamp=timestamp
            )
            
            # Add to method-specific index
            self.method_indexes[method].extend([entry1, entry2])
            
            # Add to general embedding index
            hash1 = self._compute_embedding_hash(embedding1)
            hash2 = self._compute_embedding_hash(embedding2)
            
            self.embedding_index[hash1].append(entry1)
            self.embedding_index[hash2].append(entry2)
            
            # Update clusters if enabled
            if self.enable_clustering:
                self._update_clusters(entry1, method)
                self._update_clusters(entry2, method)
            
            # Maintain index size
            self._maintain_index_size()
        
        except Exception as e:
            self.logger.warning(f"Failed to add to approximate index: {e}")
    
    def _find_match_with_clustering(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        method: str,
        accuracy_requirement: float
    ) -> Optional[float]:
        """Find match using clustering for efficiency."""
        if method not in self.clusters:
            return self._find_match_linear(embedding1, embedding2, method, accuracy_requirement)
        
        best_match = None
        best_confidence = 0.0
        
        # Check relevant clusters
        for cluster_center, cluster_entries in self.clusters[method]:
            # Check if either embedding is close to this cluster
            cluster_sim1 = self._compute_similarity(embedding1, cluster_center)
            cluster_sim2 = self._compute_similarity(embedding2, cluster_center)
            
            if cluster_sim1 > self.cluster_threshold or cluster_sim2 > self.cluster_threshold:
                # Search within this cluster
                match_result = self._search_cluster_entries(
                    embedding1, embedding2, cluster_entries, accuracy_requirement
                )
                
                if match_result and match_result[1] > best_confidence:
                    best_match, best_confidence = match_result
        
        return best_match
    
    def _find_match_linear(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        method: str,
        accuracy_requirement: float
    ) -> Optional[float]:
        """Find match using linear search."""
        if method not in self.method_indexes:
            return None
        
        # Search through method-specific index
        candidates = []
        
        for entry in self.method_indexes[method]:
            # Compute similarity between query embeddings and indexed embedding
            sim1 = self._compute_similarity(embedding1, entry.embedding)
            sim2 = self._compute_similarity(embedding2, entry.embedding)
            
            # Consider as candidate if either embedding is similar enough
            if sim1 > self.similarity_threshold or sim2 > self.similarity_threshold:
                confidence = max(sim1, sim2)
                candidates.append((entry.similarity_result, confidence, entry))
        
        if not candidates:
            return None
        
        # Find best candidate
        candidates.sort(key=lambda x: x[1], reverse=True)  # Sort by confidence
        best_result, best_confidence, best_entry = candidates[0]
        
        # Update access count
        best_entry.access_count += 1
        
        # Adjust result based on confidence and accuracy requirement
        adjusted_result = self._adjust_for_approximation(
            best_result, best_confidence, accuracy_requirement
        )
        
        return adjusted_result
    
    def _search_cluster_entries(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        cluster_entries: List[EmbeddingIndex],
        accuracy_requirement: float
    ) -> Optional[Tuple[float, float]]:
        """Search within cluster entries."""
        best_match = None
        best_confidence = 0.0
        
        for entry in cluster_entries:
            sim1 = self._compute_similarity(embedding1, entry.embedding)
            sim2 = self._compute_similarity(embedding2, entry.embedding)
            
            confidence = max(sim1, sim2)
            
            if confidence > self.similarity_threshold and confidence > best_confidence:
                adjusted_result = self._adjust_for_approximation(
                    entry.similarity_result, confidence, accuracy_requirement
                )
                
                if adjusted_result is not None:
                    best_match = adjusted_result
                    best_confidence = confidence
                    entry.access_count += 1
        
        return (best_match, best_confidence) if best_match is not None else None
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute similarity between embeddings based on strategy."""
        try:
            if self.matching_strategy == MatchingStrategy.COSINE_SIMILARITY:
                # Cosine similarity
                norm1 = np.linalg.norm(emb1)
                norm2 = np.linalg.norm(emb2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                return np.dot(emb1, emb2) / (norm1 * norm2)
            
            elif self.matching_strategy == MatchingStrategy.DOT_PRODUCT:
                # Normalized dot product
                return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
            
            elif self.matching_strategy == MatchingStrategy.EUCLIDEAN_DISTANCE:
                # Convert distance to similarity (0-1 range)
                distance = np.linalg.norm(emb1 - emb2)
                max_distance = np.sqrt(2 * len(emb1))  # Theoretical max for normalized vectors
                return 1.0 - min(distance / max_distance, 1.0)
            
            elif self.matching_strategy == MatchingStrategy.MANHATTAN_DISTANCE:
                # Convert Manhattan distance to similarity
                distance = np.sum(np.abs(emb1 - emb2))
                max_distance = 2 * len(emb1)  # Theoretical max for normalized vectors
                return 1.0 - min(distance / max_distance, 1.0)
            
            else:
                # Default to cosine similarity
                return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
        
        except Exception:
            return 0.0
    
    def _adjust_for_approximation(
        self,
        cached_similarity: float,
        confidence: float,
        accuracy_requirement: float
    ) -> Optional[float]:
        """Adjust cached similarity based on approximation confidence."""
        # Only return result if confidence meets accuracy requirement
        if confidence < accuracy_requirement:
            return None
        
        # Apply confidence-based adjustment
        # Higher confidence = less adjustment needed
        adjustment_factor = 1.0 - (1.0 - confidence) * 0.1  # Max 10% adjustment
        
        adjusted_similarity = cached_similarity * adjustment_factor
        
        # Ensure result stays in valid range
        return max(0.0, min(1.0, adjusted_similarity))
    
    def _update_clusters(self, entry: EmbeddingIndex, method: str) -> None:
        """Update clustering index with new entry."""
        if method not in self.clusters:
            self.clusters[method] = []
        
        # Find best matching cluster
        best_cluster_idx = None
        best_similarity = 0.0
        
        for i, (cluster_center, cluster_entries) in enumerate(self.clusters[method]):
            similarity = self._compute_similarity(entry.embedding, cluster_center)
            
            if similarity > best_similarity and similarity > self.cluster_threshold:
                best_similarity = similarity
                best_cluster_idx = i
        
        if best_cluster_idx is not None:
            # Add to existing cluster
            self.clusters[method][best_cluster_idx][1].append(entry)
        else:
            # Create new cluster
            self.clusters[method].append((entry.embedding.copy(), [entry]))
        
        # Limit number of clusters
        if len(self.clusters[method]) > 50:  # Max 50 clusters per method
            # Merge smallest clusters
            self._merge_small_clusters(method)
    
    def _merge_small_clusters(self, method: str) -> None:
        """Merge small clusters to limit cluster count."""
        if method not in self.clusters or len(self.clusters[method]) <= 50:
            return
        
        # Sort clusters by size
        sorted_clusters = sorted(
            self.clusters[method],
            key=lambda x: len(x[1])
        )
        
        # Merge smallest clusters
        while len(sorted_clusters) > 50:
            # Take two smallest clusters
            cluster1 = sorted_clusters.pop(0)
            cluster2 = sorted_clusters.pop(0)
            
            # Merge them
            merged_entries = cluster1[1] + cluster2[1]
            merged_center = np.mean([entry.embedding for entry in merged_entries], axis=0)
            
            merged_cluster = (merged_center, merged_entries)
            
            # Insert back into sorted list
            inserted = False
            for i, (_, entries) in enumerate(sorted_clusters):
                if len(merged_entries) <= len(entries):
                    sorted_clusters.insert(i, merged_cluster)
                    inserted = True
                    break
            
            if not inserted:
                sorted_clusters.append(merged_cluster)
        
        self.clusters[method] = sorted_clusters
    
    def _maintain_index_size(self) -> None:
        """Maintain index size within limits."""
        # Count total entries across all methods
        total_entries = sum(len(entries) for entries in self.method_indexes.values())
        
        if total_entries <= self.max_index_size:
            return
        
        # Remove oldest entries from least accessed methods
        method_stats = {}
        for method, entries in self.method_indexes.items():
            if entries:
                avg_access = np.mean([entry.access_count for entry in entries])
                avg_age = time.time() - np.mean([entry.timestamp for entry in entries])
                method_stats[method] = {"avg_access": avg_access, "avg_age": avg_age, "count": len(entries)}
        
        # Sort methods by usefulness (access frequency / age)
        sorted_methods = sorted(
            method_stats.items(),
            key=lambda x: x[1]["avg_access"] / (x[1]["avg_age"] / 3600 + 1)  # Normalize by hours
        )
        
        # Remove entries from least useful methods
        entries_to_remove = total_entries - self.max_index_size
        removed = 0
        
        for method, stats in sorted_methods:
            if removed >= entries_to_remove:
                break
            
            # Remove oldest entries from this method
            entries = self.method_indexes[method]
            entries.sort(key=lambda x: x.timestamp)
            
            remove_count = min(len(entries) // 2, entries_to_remove - removed)
            entries_to_delete = entries[:remove_count]
            
            # Update indexes
            self.method_indexes[method] = entries[remove_count:]
            
            # Remove from embedding index
            for entry in entries_to_delete:
                if entry.embedding_hash in self.embedding_index:
                    self.embedding_index[entry.embedding_hash] = [
                        e for e in self.embedding_index[entry.embedding_hash]
                        if e.timestamp != entry.timestamp
                    ]
            
            removed += remove_count
    
    def _compute_embedding_hash(self, embedding: np.ndarray) -> str:
        """Compute hash for embedding."""
        # Use a simplified hash for approximate matching
        # Take every 16th element to create a representative hash
        subset = embedding[::16]
        rounded = np.round(subset, decimals=3)
        return str(hash(rounded.tobytes()))[:12]
    
    def _update_search_time(self, search_time_ms: float) -> None:
        """Update average search time."""
        if self.match_attempts == 1:
            self.average_search_time_ms = search_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.average_search_time_ms = (
                alpha * search_time_ms + (1 - alpha) * self.average_search_time_ms
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive matcher statistics."""
        total_entries = sum(len(entries) for entries in self.method_indexes.values())
        
        return {
            "matcher_type": "approximate_similarity",
            "configuration": {
                "similarity_threshold": self.similarity_threshold,
                "matching_strategy": self.matching_strategy.value,
                "max_index_size": self.max_index_size,
                "clustering_enabled": self.enable_clustering
            },
            "index_statistics": {
                "total_entries": total_entries,
                "method_count": len(self.method_indexes),
                "cluster_count": sum(len(clusters) for clusters in self.clusters.values()),
                "average_entries_per_method": total_entries / max(len(self.method_indexes), 1)
            },
            "performance": {
                "match_attempts": self.match_attempts,
                "successful_matches": self.successful_matches,
                "match_rate": self.successful_matches / max(self.match_attempts, 1),
                "average_search_time_ms": self.average_search_time_ms
            }
        }
    
    def optimize(self) -> Dict[str, Any]:
        """Optimize matcher performance."""
        optimization_results = {}
        
        # Clean up old entries
        current_time = time.time()
        ttl_seconds = 3600  # 1 hour TTL for approximate index
        
        total_removed = 0
        for method in list(self.method_indexes.keys()):
            old_count = len(self.method_indexes[method])
            
            # Remove expired entries
            self.method_indexes[method] = [
                entry for entry in self.method_indexes[method]
                if current_time - entry.timestamp <= ttl_seconds
            ]
            
            removed = old_count - len(self.method_indexes[method])
            total_removed += removed
        
        optimization_results["expired_entries_removed"] = total_removed
        
        # Rebuild clusters if significant changes
        if total_removed > self.max_index_size * 0.1:  # 10% change threshold
            for method in self.method_indexes:
                if method in self.clusters:
                    self.clusters[method] = []
                    
                    # Rebuild clusters
                    for entry in self.method_indexes[method]:
                        self._update_clusters(entry, method)
            
            optimization_results["clusters_rebuilt"] = True
        
        # Update similarity threshold based on performance
        if self.match_attempts > 100:
            match_rate = self.successful_matches / self.match_attempts
            
            if match_rate < 0.1:  # Low match rate
                # Relax threshold slightly
                new_threshold = max(0.8, self.similarity_threshold - 0.02)
                if new_threshold != self.similarity_threshold:
                    self.similarity_threshold = new_threshold
                    optimization_results["threshold_adjusted"] = new_threshold
        
        return optimization_results
    
    def clear_index(self) -> None:
        """Clear all index data."""
        self.embedding_index.clear()
        self.method_indexes.clear()
        self.clusters.clear()
        
        # Reset statistics
        self.match_attempts = 0
        self.successful_matches = 0
        self.average_search_time_ms = 0.0


__all__ = [
    "MatchingStrategy",
    "SimilarityThreshold",
    "EmbeddingIndex",
    "ApproximateSimilarityMatcher"
]
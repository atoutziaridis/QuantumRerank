"""
Vector search and retrieval metrics collector.

This module provides specialized metric collection for vector search operations,
including FAISS performance, embedding quality, and retrieval accuracy.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque, defaultdict
import numpy as np

from ..metrics_collector import MetricsCollector
from ...utils import get_logger


@dataclass
class SearchOperationMetrics:
    """Metrics for a vector search operation."""
    operation_id: str
    query_embedding_dim: int
    search_space_size: int
    k_retrieved: int
    search_time_ms: float
    embedding_time_ms: float
    total_time_ms: float
    accuracy_score: Optional[float] = None
    cache_hit: bool = False
    success: bool = True
    error_type: Optional[str] = None
    timestamp: float = 0.0


@dataclass
class IndexMetrics:
    """Vector index performance metrics."""
    index_size: int
    index_type: str
    build_time_ms: float
    memory_usage_mb: float
    search_throughput_qps: float
    index_quality_score: float
    timestamp: float = 0.0


class SearchMetricsCollector:
    """
    Specialized collector for vector search and retrieval metrics.
    
    Tracks FAISS index performance, search latency, retrieval accuracy,
    and embedding quality metrics.
    """
    
    def __init__(self, base_collector: Optional[MetricsCollector] = None):
        self.base_collector = base_collector or MetricsCollector()
        self.logger = get_logger(__name__)
        
        # Search operation tracking
        self.search_operations: deque = deque(maxlen=1000)
        self.index_metrics: Dict[str, IndexMetrics] = {}
        
        # Performance tracking
        self.search_latency_history: deque = deque(maxlen=500)
        self.accuracy_history: deque = deque(maxlen=200)
        self.throughput_measurements: deque = deque(maxlen=100)
        
        # Cache performance
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info("Initialized SearchMetricsCollector")
    
    def record_search_operation(self, metrics: SearchOperationMetrics) -> None:
        """Record comprehensive search operation metrics."""
        with self._lock:
            # Store detailed metrics
            metrics.timestamp = time.time()
            self.search_operations.append(metrics)
            
            # Update cache statistics
            if metrics.cache_hit:
                self.cache_stats["hits"] += 1
            else:
                self.cache_stats["misses"] += 1
            self.cache_stats["total_requests"] += 1
            
            # Record in base collector
            tags = {
                "operation_id": metrics.operation_id,
                "success": str(metrics.success),
                "cache_hit": str(metrics.cache_hit),
                "component": "search"
            }
            
            self.base_collector.record_timer("search.total_time",
                                            metrics.total_time_ms, tags)
            self.base_collector.record_timer("search.search_time",
                                            metrics.search_time_ms, tags)
            self.base_collector.record_timer("search.embedding_time",
                                            metrics.embedding_time_ms, tags)
            
            self.base_collector.record_gauge("search.query_dimension",
                                           metrics.query_embedding_dim, "", tags)
            self.base_collector.record_gauge("search.space_size",
                                           metrics.search_space_size, "", tags)
            self.base_collector.record_gauge("search.k_retrieved",
                                           metrics.k_retrieved, "", tags)
            
            if metrics.accuracy_score is not None:
                self.base_collector.record_gauge("search.accuracy",
                                                metrics.accuracy_score, "", tags)
                self.accuracy_history.append({
                    "score": metrics.accuracy_score,
                    "timestamp": metrics.timestamp,
                    "k": metrics.k_retrieved
                })
            
            # Count operations
            self.base_collector.record_counter("search.operations", 1, tags)
            
            if not metrics.success:
                error_tags = {**tags, "error_type": metrics.error_type or "unknown"}
                self.base_collector.record_counter("search.errors", 1, error_tags)
            
            # Track latency
            self.search_latency_history.append({
                "total_time_ms": metrics.total_time_ms,
                "search_time_ms": metrics.search_time_ms,
                "embedding_time_ms": metrics.embedding_time_ms,
                "timestamp": metrics.timestamp
            })
    
    def record_index_build(self, index_name: str, metrics: IndexMetrics) -> None:
        """Record vector index build metrics."""
        with self._lock:
            # Store index metrics
            metrics.timestamp = time.time()
            self.index_metrics[index_name] = metrics
            
            # Record in base collector
            tags = {
                "index_name": index_name,
                "index_type": metrics.index_type,
                "component": "search"
            }
            
            self.base_collector.record_timer("search.index.build_time",
                                            metrics.build_time_ms, tags)
            self.base_collector.record_gauge("search.index.size",
                                           metrics.index_size, "", tags)
            self.base_collector.record_gauge("search.index.memory_usage",
                                           metrics.memory_usage_mb, "MB", tags)
            self.base_collector.record_gauge("search.index.quality_score",
                                           metrics.index_quality_score, "", tags)
            self.base_collector.record_gauge("search.index.throughput",
                                           metrics.search_throughput_qps, "qps", tags)
            
            # Count index builds
            self.base_collector.record_counter("search.index.builds", 1, tags)
    
    def record_embedding_computation(self, text_length: int, embedding_dim: int,
                                   computation_time_ms: float, 
                                   embedding_quality: Optional[float] = None) -> None:
        """Record embedding computation metrics."""
        tags = {"component": "search", "operation": "embedding"}
        
        self.base_collector.record_timer("search.embedding.computation_time",
                                        computation_time_ms, tags)
        self.base_collector.record_gauge("search.embedding.text_length",
                                        text_length, "chars", tags)
        self.base_collector.record_gauge("search.embedding.dimension",
                                        embedding_dim, "", tags)
        
        if embedding_quality is not None:
            self.base_collector.record_gauge("search.embedding.quality",
                                            embedding_quality, "", tags)
        
        self.base_collector.record_counter("search.embeddings.computed", 1, tags)
    
    def record_cache_performance(self, cache_name: str, hit_rate: float,
                                eviction_rate: float, memory_usage_mb: float) -> None:
        """Record cache performance metrics."""
        tags = {"cache_name": cache_name, "component": "search"}
        
        self.base_collector.record_gauge("search.cache.hit_rate",
                                        hit_rate, "", tags)
        self.base_collector.record_gauge("search.cache.eviction_rate",
                                        eviction_rate, "", tags)
        self.base_collector.record_gauge("search.cache.memory_usage",
                                        memory_usage_mb, "MB", tags)
    
    def record_retrieval_accuracy(self, ground_truth_ids: List[str],
                                retrieved_ids: List[str], k: int) -> float:
        """Record retrieval accuracy metrics."""
        # Calculate accuracy metrics
        accuracy_at_k = self._calculate_accuracy_at_k(ground_truth_ids, retrieved_ids, k)
        precision_at_k = self._calculate_precision_at_k(ground_truth_ids, retrieved_ids, k)
        recall_at_k = self._calculate_recall_at_k(ground_truth_ids, retrieved_ids, k)
        
        tags = {"component": "search", "metric": "accuracy", "k": str(k)}
        
        self.base_collector.record_gauge("search.accuracy.accuracy_at_k",
                                        accuracy_at_k, "", tags)
        self.base_collector.record_gauge("search.accuracy.precision_at_k",
                                        precision_at_k, "", tags)
        self.base_collector.record_gauge("search.accuracy.recall_at_k",
                                        recall_at_k, "", tags)
        
        return accuracy_at_k
    
    def record_throughput_measurement(self, queries_per_second: float,
                                    concurrent_queries: int,
                                    measurement_duration_seconds: float) -> None:
        """Record search throughput measurements."""
        with self._lock:
            self.throughput_measurements.append({
                "qps": queries_per_second,
                "concurrent_queries": concurrent_queries,
                "duration_seconds": measurement_duration_seconds,
                "timestamp": time.time()
            })
        
        tags = {"component": "search", "measurement": "throughput"}
        
        self.base_collector.record_gauge("search.throughput.qps",
                                        queries_per_second, "qps", tags)
        self.base_collector.record_gauge("search.throughput.concurrent_queries",
                                        concurrent_queries, "", tags)
    
    def get_search_performance_summary(self, time_window_seconds: int = 300) -> Dict[str, Any]:
        """Get comprehensive search performance summary."""
        cutoff_time = time.time() - time_window_seconds
        
        with self._lock:
            # Recent search operations
            recent_operations = [
                op for op in self.search_operations
                if op.timestamp >= cutoff_time
            ]
            
            # Recent accuracy measurements
            recent_accuracy = [
                acc for acc in self.accuracy_history
                if acc["timestamp"] >= cutoff_time
            ]
            
            # Recent latency measurements
            recent_latency = [
                lat for lat in self.search_latency_history
                if lat["timestamp"] >= cutoff_time
            ]
            
            summary = {
                "time_window_seconds": time_window_seconds,
                "operation_analysis": self._analyze_search_operations(recent_operations),
                "latency_analysis": self._analyze_search_latency(recent_latency),
                "accuracy_analysis": self._analyze_search_accuracy(recent_accuracy),
                "cache_performance": self._analyze_cache_performance(),
                "index_performance": self._analyze_index_performance(),
                "throughput_analysis": self._analyze_throughput_performance()
            }
            
            return summary
    
    def get_search_health_indicators(self) -> Dict[str, Any]:
        """Get search system health indicators."""
        current_time = time.time()
        recent_time = current_time - 300  # Last 5 minutes
        
        with self._lock:
            recent_operations = [
                op for op in self.search_operations
                if op.timestamp >= recent_time
            ]
            
            indicators = {
                "search_success_rate": self._calculate_search_success_rate(recent_operations),
                "average_search_latency_ms": self._calculate_average_search_latency(recent_operations),
                "cache_hit_rate": self._calculate_current_cache_hit_rate(),
                "search_accuracy": self._calculate_average_accuracy(),
                "index_health": self._assess_index_health(),
                "throughput_health": self._assess_throughput_health(),
                "overall_search_health": "unknown"
            }
            
            # Calculate overall health
            indicators["overall_search_health"] = self._assess_overall_search_health(indicators)
            
            return indicators
    
    def _calculate_accuracy_at_k(self, ground_truth: List[str], 
                               retrieved: List[str], k: int) -> float:
        """Calculate accuracy@k metric."""
        if not ground_truth or not retrieved:
            return 0.0
        
        retrieved_k = retrieved[:k]
        relevant_retrieved = len(set(ground_truth) & set(retrieved_k))
        
        return relevant_retrieved / min(k, len(ground_truth))
    
    def _calculate_precision_at_k(self, ground_truth: List[str],
                                retrieved: List[str], k: int) -> float:
        """Calculate precision@k metric."""
        if not retrieved:
            return 0.0
        
        retrieved_k = retrieved[:k]
        relevant_retrieved = len(set(ground_truth) & set(retrieved_k))
        
        return relevant_retrieved / len(retrieved_k)
    
    def _calculate_recall_at_k(self, ground_truth: List[str],
                             retrieved: List[str], k: int) -> float:
        """Calculate recall@k metric."""
        if not ground_truth:
            return 0.0
        
        retrieved_k = retrieved[:k]
        relevant_retrieved = len(set(ground_truth) & set(retrieved_k))
        
        return relevant_retrieved / len(ground_truth)
    
    def _analyze_search_operations(self, operations: List[SearchOperationMetrics]) -> Dict[str, Any]:
        """Analyze search operation performance."""
        if not operations:
            return {"count": 0}
        
        successful_ops = [op for op in operations if op.success]
        
        analysis = {
            "total_operations": len(operations),
            "successful_operations": len(successful_ops),
            "success_rate": len(successful_ops) / len(operations),
            "cache_hit_rate": len([op for op in operations if op.cache_hit]) / len(operations)
        }
        
        if successful_ops:
            analysis.update({
                "avg_total_time_ms": np.mean([op.total_time_ms for op in successful_ops]),
                "p95_total_time_ms": np.percentile([op.total_time_ms for op in successful_ops], 95),
                "avg_search_space_size": np.mean([op.search_space_size for op in successful_ops]),
                "avg_k_retrieved": np.mean([op.k_retrieved for op in successful_ops])
            })
        
        return analysis
    
    def _analyze_search_latency(self, latency_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze search latency performance."""
        if not latency_data:
            return {"count": 0}
        
        total_times = [lat["total_time_ms"] for lat in latency_data]
        search_times = [lat["search_time_ms"] for lat in latency_data]
        embedding_times = [lat["embedding_time_ms"] for lat in latency_data]
        
        return {
            "measurement_count": len(latency_data),
            "avg_total_latency_ms": np.mean(total_times),
            "p95_total_latency_ms": np.percentile(total_times, 95),
            "p99_total_latency_ms": np.percentile(total_times, 99),
            "avg_search_latency_ms": np.mean(search_times),
            "avg_embedding_latency_ms": np.mean(embedding_times),
            "search_latency_ratio": np.mean(search_times) / np.mean(total_times) if np.mean(total_times) > 0 else 0
        }
    
    def _analyze_search_accuracy(self, accuracy_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze search accuracy performance."""
        if not accuracy_data:
            return {"count": 0}
        
        scores = [acc["score"] for acc in accuracy_data]
        
        return {
            "measurement_count": len(accuracy_data),
            "avg_accuracy": np.mean(scores),
            "min_accuracy": np.min(scores),
            "max_accuracy": np.max(scores),
            "accuracy_std": np.std(scores),
            "high_accuracy_rate": len([s for s in scores if s >= 0.9]) / len(scores)
        }
    
    def _analyze_cache_performance(self) -> Dict[str, Any]:
        """Analyze cache performance."""
        total_requests = self.cache_stats["total_requests"]
        
        if total_requests == 0:
            return {"hit_rate": 0.0, "total_requests": 0}
        
        return {
            "hit_rate": self.cache_stats["hits"] / total_requests,
            "miss_rate": self.cache_stats["misses"] / total_requests,
            "total_requests": total_requests,
            "cache_efficiency": "good" if self.cache_stats["hits"] / total_requests > 0.6 else "needs_improvement"
        }
    
    def _analyze_index_performance(self) -> Dict[str, Any]:
        """Analyze vector index performance."""
        if not self.index_metrics:
            return {"index_count": 0}
        
        latest_metrics = list(self.index_metrics.values())
        
        return {
            "index_count": len(self.index_metrics),
            "avg_quality_score": np.mean([m.index_quality_score for m in latest_metrics]),
            "avg_memory_usage_mb": np.mean([m.memory_usage_mb for m in latest_metrics]),
            "avg_throughput_qps": np.mean([m.search_throughput_qps for m in latest_metrics]),
            "index_types": list(set(m.index_type for m in latest_metrics))
        }
    
    def _analyze_throughput_performance(self) -> Dict[str, Any]:
        """Analyze search throughput performance."""
        if not self.throughput_measurements:
            return {"measurement_count": 0}
        
        recent_measurements = list(self.throughput_measurements)[-10:]  # Last 10 measurements
        qps_values = [m["qps"] for m in recent_measurements]
        
        return {
            "measurement_count": len(recent_measurements),
            "avg_qps": np.mean(qps_values),
            "max_qps": np.max(qps_values),
            "min_qps": np.min(qps_values),
            "qps_stability": 1.0 - (np.std(qps_values) / np.mean(qps_values)) if np.mean(qps_values) > 0 else 0.0
        }
    
    def _calculate_search_success_rate(self, operations: List[SearchOperationMetrics]) -> float:
        """Calculate search operation success rate."""
        if not operations:
            return 1.0
        
        return len([op for op in operations if op.success]) / len(operations)
    
    def _calculate_average_search_latency(self, operations: List[SearchOperationMetrics]) -> float:
        """Calculate average search latency."""
        successful_ops = [op for op in operations if op.success]
        if not successful_ops:
            return 0.0
        
        return np.mean([op.total_time_ms for op in successful_ops])
    
    def _calculate_current_cache_hit_rate(self) -> float:
        """Calculate current cache hit rate."""
        if self.cache_stats["total_requests"] == 0:
            return 0.0
        
        return self.cache_stats["hits"] / self.cache_stats["total_requests"]
    
    def _calculate_average_accuracy(self) -> float:
        """Calculate average search accuracy."""
        if not self.accuracy_history:
            return 0.0
        
        recent_accuracy = list(self.accuracy_history)[-20:]  # Last 20 measurements
        return np.mean([acc["score"] for acc in recent_accuracy])
    
    def _assess_index_health(self) -> str:
        """Assess vector index health."""
        if not self.index_metrics:
            return "no_indexes"
        
        avg_quality = np.mean([m.index_quality_score for m in self.index_metrics.values()])
        
        if avg_quality >= 0.9:
            return "excellent"
        elif avg_quality >= 0.8:
            return "good"
        elif avg_quality >= 0.7:
            return "fair"
        else:
            return "poor"
    
    def _assess_throughput_health(self) -> str:
        """Assess search throughput health."""
        if not self.throughput_measurements:
            return "unknown"
        
        recent_qps = [m["qps"] for m in list(self.throughput_measurements)[-5:]]
        avg_qps = np.mean(recent_qps)
        
        if avg_qps >= 100:
            return "excellent"
        elif avg_qps >= 50:
            return "good"
        elif avg_qps >= 20:
            return "fair"
        else:
            return "poor"
    
    def _assess_overall_search_health(self, indicators: Dict[str, Any]) -> str:
        """Assess overall search system health."""
        health_score = 0
        
        # Success rate contribution
        if indicators["search_success_rate"] >= 0.99:
            health_score += 25
        elif indicators["search_success_rate"] >= 0.95:
            health_score += 20
        elif indicators["search_success_rate"] >= 0.90:
            health_score += 15
        
        # Latency contribution
        if indicators["average_search_latency_ms"] <= 50:
            health_score += 25
        elif indicators["average_search_latency_ms"] <= 100:
            health_score += 20
        elif indicators["average_search_latency_ms"] <= 200:
            health_score += 15
        
        # Cache hit rate contribution
        if indicators["cache_hit_rate"] >= 0.8:
            health_score += 25
        elif indicators["cache_hit_rate"] >= 0.6:
            health_score += 20
        elif indicators["cache_hit_rate"] >= 0.4:
            health_score += 15
        
        # Accuracy contribution
        if indicators["search_accuracy"] >= 0.95:
            health_score += 25
        elif indicators["search_accuracy"] >= 0.90:
            health_score += 20
        elif indicators["search_accuracy"] >= 0.85:
            health_score += 15
        
        if health_score >= 85:
            return "excellent"
        elif health_score >= 70:
            return "good"
        elif health_score >= 50:
            return "fair"
        else:
            return "poor"


__all__ = [
    "SearchOperationMetrics",
    "IndexMetrics",
    "SearchMetricsCollector"
]
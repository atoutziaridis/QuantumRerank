"""
Performance monitoring for search and retrieval operations.

This module provides comprehensive performance tracking and analysis
for the complete search pipeline including retrieval and reranking.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import numpy as np

from ..utils import get_logger


class MetricType(Enum):
    """Types of performance metrics."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    RESOURCE_USAGE = "resource_usage"
    CACHE_PERFORMANCE = "cache_performance"


@dataclass
class PerformanceMetric:
    """Individual performance metric measurement."""
    metric_type: MetricType
    name: str
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchPerformanceTargets:
    """Performance targets for search operations."""
    # Retrieval targets
    retrieval_latency_ms: float = 50.0
    retrieval_throughput_qps: float = 1000.0
    retrieval_accuracy: float = 0.98
    
    # Reranking targets
    rerank_latency_ms: float = 500.0
    rerank_throughput_qps: float = 100.0
    rerank_accuracy: float = 0.99
    
    # End-to-end targets
    total_latency_ms: float = 600.0
    total_throughput_qps: float = 50.0
    
    # Resource targets
    memory_usage_mb: float = 2048.0
    cpu_utilization: float = 0.8
    
    # Cache targets
    cache_hit_rate: float = 0.6
    cache_latency_ms: float = 5.0


@dataclass
class PipelinePerformanceSnapshot:
    """Snapshot of pipeline performance at a point in time."""
    timestamp: float
    retrieval_latency_ms: float
    rerank_latency_ms: float
    total_latency_ms: float
    throughput_qps: float
    cache_hit_rate: float
    memory_usage_mb: float
    active_queries: int
    backend_type: str


class SearchPerformanceMonitor:
    """
    Comprehensive performance monitoring for search pipeline.
    
    This monitor tracks all aspects of search performance including
    latency, throughput, accuracy, and resource utilization.
    """
    
    def __init__(self, targets: Optional[SearchPerformanceTargets] = None):
        self.targets = targets or SearchPerformanceTargets()
        self.logger = get_logger(__name__)
        
        # Metric storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.metric_aggregates: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Performance snapshots
        self.snapshots: deque = deque(maxlen=1000)
        
        # Real-time tracking
        self.active_queries: Dict[str, float] = {}  # query_id -> start_time
        self.query_statistics = {
            "total_queries": 0,
            "completed_queries": 0,
            "failed_queries": 0,
            "active_count": 0
        }
        
        # Performance alerts
        self.performance_alerts: List[Dict[str, Any]] = []
        self.alert_thresholds = self._initialize_alert_thresholds()
        
        # Threading
        self._lock = threading.RLock()
        self.monitoring_enabled = True
        
        self.logger.info("Initialized SearchPerformanceMonitor")
    
    def start_query_tracking(self, query_id: str, 
                           query_type: str = "search") -> None:
        """Start tracking a new query."""
        if not self.monitoring_enabled:
            return
        
        with self._lock:
            self.active_queries[query_id] = time.time()
            self.query_statistics["total_queries"] += 1
            self.query_statistics["active_count"] = len(self.active_queries)
    
    def record_retrieval_performance(self, query_id: str,
                                   latency_ms: float,
                                   result_count: int,
                                   backend_type: str,
                                   cache_hit: bool = False) -> None:
        """Record retrieval stage performance."""
        if not self.monitoring_enabled:
            return
        
        with self._lock:
            # Record latency
            self._record_metric("retrieval_latency_ms", latency_ms)
            
            # Record throughput (queries per second)
            qps = 1000.0 / max(latency_ms, 0.1)
            self._record_metric("retrieval_throughput_qps", qps)
            
            # Record result count
            self._record_metric("retrieval_result_count", result_count)
            
            # Record backend performance
            self._record_metric(f"retrieval_latency_{backend_type}_ms", latency_ms)
            
            # Record cache performance
            if cache_hit:
                self._record_metric("retrieval_cache_hits", 1)
            else:
                self._record_metric("retrieval_cache_misses", 1)
            
            # Check performance alerts
            self._check_retrieval_alerts(latency_ms, qps)
    
    def record_rerank_performance(self, query_id: str,
                                latency_ms: float,
                                input_count: int,
                                output_count: int,
                                method: str,
                                accuracy_score: Optional[float] = None) -> None:
        """Record reranking stage performance."""
        if not self.monitoring_enabled:
            return
        
        with self._lock:
            # Record latency
            self._record_metric("rerank_latency_ms", latency_ms)
            
            # Record throughput
            qps = 1000.0 / max(latency_ms, 0.1)
            self._record_metric("rerank_throughput_qps", qps)
            
            # Record reranking efficiency
            efficiency = output_count / max(input_count, 1)
            self._record_metric("rerank_efficiency", efficiency)
            
            # Record method-specific performance
            self._record_metric(f"rerank_latency_{method}_ms", latency_ms)
            
            # Record accuracy if provided
            if accuracy_score is not None:
                self._record_metric("rerank_accuracy", accuracy_score)
            
            # Check performance alerts
            self._check_rerank_alerts(latency_ms, qps)
    
    def complete_query_tracking(self, query_id: str,
                              success: bool = True,
                              error_message: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Complete query tracking and return performance summary."""
        if not self.monitoring_enabled:
            return None
        
        with self._lock:
            if query_id not in self.active_queries:
                return None
            
            start_time = self.active_queries.pop(query_id)
            total_time_ms = (time.time() - start_time) * 1000
            
            # Update statistics
            if success:
                self.query_statistics["completed_queries"] += 1
            else:
                self.query_statistics["failed_queries"] += 1
            
            self.query_statistics["active_count"] = len(self.active_queries)
            
            # Record total latency
            self._record_metric("total_latency_ms", total_time_ms)
            
            # Record overall throughput
            total_qps = 1000.0 / max(total_time_ms, 0.1)
            self._record_metric("total_throughput_qps", total_qps)
            
            # Check end-to-end alerts
            self._check_total_performance_alerts(total_time_ms, total_qps)
            
            # Record error if occurred
            if not success and error_message:
                self._record_error(query_id, error_message)
            
            return self._get_query_performance_summary(query_id, total_time_ms)
    
    def record_resource_usage(self, memory_usage_mb: float,
                            cpu_utilization: float,
                            active_connections: int = 0) -> None:
        """Record system resource usage."""
        if not self.monitoring_enabled:
            return
        
        with self._lock:
            self._record_metric("memory_usage_mb", memory_usage_mb)
            self._record_metric("cpu_utilization", cpu_utilization)
            self._record_metric("active_connections", active_connections)
            
            # Check resource alerts
            self._check_resource_alerts(memory_usage_mb, cpu_utilization)
    
    def record_cache_performance(self, cache_type: str,
                               hit_rate: float,
                               latency_ms: float,
                               memory_usage_mb: float) -> None:
        """Record cache performance metrics."""
        if not self.monitoring_enabled:
            return
        
        with self._lock:
            self._record_metric(f"cache_{cache_type}_hit_rate", hit_rate)
            self._record_metric(f"cache_{cache_type}_latency_ms", latency_ms)
            self._record_metric(f"cache_{cache_type}_memory_mb", memory_usage_mb)
            
            # Overall cache metrics
            self._record_metric("cache_hit_rate", hit_rate)
            self._record_metric("cache_latency_ms", latency_ms)
    
    def create_performance_snapshot(self, backend_type: str = "unknown") -> PipelinePerformanceSnapshot:
        """Create a snapshot of current performance."""
        with self._lock:
            # Calculate current metrics
            retrieval_latency = self._get_recent_average("retrieval_latency_ms", 10)
            rerank_latency = self._get_recent_average("rerank_latency_ms", 10)
            total_latency = self._get_recent_average("total_latency_ms", 10)
            throughput = self._get_recent_average("total_throughput_qps", 10)
            cache_hit_rate = self._get_recent_average("cache_hit_rate", 10)
            memory_usage = self._get_recent_average("memory_usage_mb", 5)
            
            snapshot = PipelinePerformanceSnapshot(
                timestamp=time.time(),
                retrieval_latency_ms=retrieval_latency,
                rerank_latency_ms=rerank_latency,
                total_latency_ms=total_latency,
                throughput_qps=throughput,
                cache_hit_rate=cache_hit_rate,
                memory_usage_mb=memory_usage,
                active_queries=self.query_statistics["active_count"],
                backend_type=backend_type
            )
            
            self.snapshots.append(snapshot)
            return snapshot
    
    def get_performance_report(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        with self._lock:
            cutoff_time = time.time() - (time_window_minutes * 60)
            
            report = {
                "time_window_minutes": time_window_minutes,
                "query_statistics": self.query_statistics.copy(),
                "performance_metrics": {},
                "target_compliance": {},
                "alerts": self.performance_alerts[-20:] if self.performance_alerts else [],
                "trends": {},
                "recommendations": []
            }
            
            # Calculate metrics for time window
            for metric_name, values in self.metrics.items():
                recent_values = [
                    value for timestamp, value in values
                    if timestamp >= cutoff_time
                ]
                
                if recent_values:
                    report["performance_metrics"][metric_name] = {
                        "count": len(recent_values),
                        "average": np.mean(recent_values),
                        "min": np.min(recent_values),
                        "max": np.max(recent_values),
                        "p50": np.percentile(recent_values, 50),
                        "p95": np.percentile(recent_values, 95),
                        "p99": np.percentile(recent_values, 99)
                    }
            
            # Check target compliance
            report["target_compliance"] = self._calculate_target_compliance(report["performance_metrics"])
            
            # Calculate trends
            report["trends"] = self._calculate_performance_trends()
            
            # Generate recommendations
            report["recommendations"] = self._generate_performance_recommendations(report)
            
            return report
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics."""
        with self._lock:
            return {
                "active_queries": self.query_statistics["active_count"],
                "total_queries": self.query_statistics["total_queries"],
                "success_rate": (
                    self.query_statistics["completed_queries"] / 
                    max(self.query_statistics["total_queries"], 1)
                ),
                "current_retrieval_latency_ms": self._get_recent_average("retrieval_latency_ms", 5),
                "current_rerank_latency_ms": self._get_recent_average("rerank_latency_ms", 5),
                "current_total_latency_ms": self._get_recent_average("total_latency_ms", 5),
                "current_throughput_qps": self._get_recent_average("total_throughput_qps", 5),
                "current_cache_hit_rate": self._get_recent_average("cache_hit_rate", 10),
                "current_memory_usage_mb": self._get_recent_average("memory_usage_mb", 3),
                "alert_count": len([alert for alert in self.performance_alerts if not alert.get("acknowledged", False)])
            }
    
    def _record_metric(self, metric_name: str, value: float) -> None:
        """Record a metric measurement."""
        timestamp = time.time()
        self.metrics[metric_name].append((timestamp, value))
        
        # Update aggregates
        if metric_name not in self.metric_aggregates:
            self.metric_aggregates[metric_name] = {
                "count": 0,
                "sum": 0.0,
                "min": float('inf'),
                "max": float('-inf')
            }
        
        agg = self.metric_aggregates[metric_name]
        agg["count"] += 1
        agg["sum"] += value
        agg["min"] = min(agg["min"], value)
        agg["max"] = max(agg["max"], value)
    
    def _get_recent_average(self, metric_name: str, count: int = 10) -> float:
        """Get average of recent metric values."""
        if metric_name not in self.metrics:
            return 0.0
        
        recent_values = list(self.metrics[metric_name])[-count:]
        if not recent_values:
            return 0.0
        
        return sum(value for _, value in recent_values) / len(recent_values)
    
    def _check_retrieval_alerts(self, latency_ms: float, qps: float) -> None:
        """Check retrieval performance alerts."""
        if latency_ms > self.targets.retrieval_latency_ms * self.alert_thresholds["latency_warning"]:
            self._create_alert(
                "retrieval_latency_high",
                f"Retrieval latency {latency_ms:.1f}ms exceeds target {self.targets.retrieval_latency_ms:.1f}ms",
                "warning" if latency_ms < self.targets.retrieval_latency_ms * self.alert_thresholds["latency_critical"] else "critical"
            )
        
        if qps < self.targets.retrieval_throughput_qps * self.alert_thresholds["throughput_warning"]:
            self._create_alert(
                "retrieval_throughput_low", 
                f"Retrieval throughput {qps:.1f} QPS below target {self.targets.retrieval_throughput_qps:.1f} QPS",
                "warning" if qps > self.targets.retrieval_throughput_qps * self.alert_thresholds["throughput_critical"] else "critical"
            )
    
    def _check_rerank_alerts(self, latency_ms: float, qps: float) -> None:
        """Check reranking performance alerts."""
        if latency_ms > self.targets.rerank_latency_ms * self.alert_thresholds["latency_warning"]:
            self._create_alert(
                "rerank_latency_high",
                f"Rerank latency {latency_ms:.1f}ms exceeds target {self.targets.rerank_latency_ms:.1f}ms",
                "warning" if latency_ms < self.targets.rerank_latency_ms * self.alert_thresholds["latency_critical"] else "critical"
            )
    
    def _check_total_performance_alerts(self, latency_ms: float, qps: float) -> None:
        """Check end-to-end performance alerts."""
        if latency_ms > self.targets.total_latency_ms * self.alert_thresholds["latency_warning"]:
            self._create_alert(
                "total_latency_high",
                f"Total latency {latency_ms:.1f}ms exceeds target {self.targets.total_latency_ms:.1f}ms",
                "critical"
            )
    
    def _check_resource_alerts(self, memory_mb: float, cpu_util: float) -> None:
        """Check resource usage alerts."""
        if memory_mb > self.targets.memory_usage_mb * self.alert_thresholds["resource_warning"]:
            self._create_alert(
                "memory_usage_high",
                f"Memory usage {memory_mb:.0f}MB exceeds target {self.targets.memory_usage_mb:.0f}MB",
                "warning" if memory_mb < self.targets.memory_usage_mb * self.alert_thresholds["resource_critical"] else "critical"
            )
        
        if cpu_util > self.targets.cpu_utilization * self.alert_thresholds["resource_warning"]:
            self._create_alert(
                "cpu_utilization_high",
                f"CPU utilization {cpu_util:.1%} exceeds target {self.targets.cpu_utilization:.1%}",
                "warning" if cpu_util < self.targets.cpu_utilization * self.alert_thresholds["resource_critical"] else "critical"
            )
    
    def _create_alert(self, alert_type: str, message: str, severity: str) -> None:
        """Create performance alert."""
        alert = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": time.time(),
            "acknowledged": False
        }
        
        # Avoid duplicate alerts
        recent_alerts = [
            a for a in self.performance_alerts[-10:]
            if a["type"] == alert_type and (time.time() - a["timestamp"]) < 300  # 5 minutes
        ]
        
        if not recent_alerts:
            self.performance_alerts.append(alert)
            self.logger.warning(f"Performance alert: {message}")
    
    def _record_error(self, query_id: str, error_message: str) -> None:
        """Record query error."""
        self._record_metric("query_errors", 1)
        
        self._create_alert(
            "query_error",
            f"Query {query_id} failed: {error_message}",
            "warning"
        )
    
    def _get_query_performance_summary(self, query_id: str, total_time_ms: float) -> Dict[str, Any]:
        """Get performance summary for completed query."""
        return {
            "query_id": query_id,
            "total_time_ms": total_time_ms,
            "retrieval_time_ms": self._get_recent_average("retrieval_latency_ms", 1),
            "rerank_time_ms": self._get_recent_average("rerank_latency_ms", 1),
            "cache_hit": self._get_recent_average("retrieval_cache_hits", 1) > 0,
            "meets_latency_target": total_time_ms <= self.targets.total_latency_ms
        }
    
    def _calculate_target_compliance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate compliance with performance targets."""
        compliance = {}
        
        # Check each target
        target_checks = [
            ("retrieval_latency_ms", "average", self.targets.retrieval_latency_ms, "<="),
            ("rerank_latency_ms", "average", self.targets.rerank_latency_ms, "<="),
            ("total_latency_ms", "average", self.targets.total_latency_ms, "<="),
            ("total_throughput_qps", "average", self.targets.total_throughput_qps, ">="),
            ("cache_hit_rate", "average", self.targets.cache_hit_rate, ">="),
            ("memory_usage_mb", "average", self.targets.memory_usage_mb, "<=")
        ]
        
        for metric_name, stat_type, target_value, comparison in target_checks:
            if metric_name in metrics and stat_type in metrics[metric_name]:
                actual_value = metrics[metric_name][stat_type]
                
                if comparison == "<=":
                    meets_target = actual_value <= target_value
                else:  # ">="
                    meets_target = actual_value >= target_value
                
                compliance[metric_name] = {
                    "meets_target": meets_target,
                    "actual_value": actual_value,
                    "target_value": target_value,
                    "deviation_percent": abs(actual_value - target_value) / target_value * 100
                }
        
        return compliance
    
    def _calculate_performance_trends(self) -> Dict[str, str]:
        """Calculate performance trends over recent measurements."""
        trends = {}
        
        key_metrics = [
            "retrieval_latency_ms",
            "rerank_latency_ms", 
            "total_latency_ms",
            "total_throughput_qps",
            "cache_hit_rate"
        ]
        
        for metric_name in key_metrics:
            if metric_name in self.metrics and len(self.metrics[metric_name]) >= 20:
                recent_values = [value for _, value in list(self.metrics[metric_name])[-20:]]
                
                # Simple trend calculation
                first_half = np.mean(recent_values[:10])
                second_half = np.mean(recent_values[10:])
                
                if second_half > first_half * 1.05:
                    trend = "increasing"
                elif second_half < first_half * 0.95:
                    trend = "decreasing"
                else:
                    trend = "stable"
                
                trends[metric_name] = trend
        
        return trends
    
    def _generate_performance_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        metrics = report.get("performance_metrics", {})
        compliance = report.get("target_compliance", {})
        
        # Check for high latency
        if ("total_latency_ms" in compliance and 
            not compliance["total_latency_ms"]["meets_target"]):
            recommendations.append("Consider optimizing retrieval size or switching to faster backend")
        
        # Check for low throughput
        if ("total_throughput_qps" in compliance and
            not compliance["total_throughput_qps"]["meets_target"]):
            recommendations.append("Consider enabling batch processing or caching optimization")
        
        # Check for low cache hit rate
        if ("cache_hit_rate" in compliance and
            not compliance["cache_hit_rate"]["meets_target"]):
            recommendations.append("Consider adjusting cache size or improving cache key strategy")
        
        # Check for high memory usage
        if ("memory_usage_mb" in compliance and
            not compliance["memory_usage_mb"]["meets_target"]):
            recommendations.append("Consider reducing index size or enabling memory optimization")
        
        return recommendations
    
    def _initialize_alert_thresholds(self) -> Dict[str, float]:
        """Initialize alert threshold multipliers."""
        return {
            "latency_warning": 1.2,    # 120% of target
            "latency_critical": 2.0,   # 200% of target
            "throughput_warning": 0.8, # 80% of target
            "throughput_critical": 0.5, # 50% of target
            "resource_warning": 0.9,   # 90% of target
            "resource_critical": 0.95  # 95% of target
        }


__all__ = [
    "MetricType",
    "PerformanceMetric", 
    "SearchPerformanceTargets",
    "PipelinePerformanceSnapshot",
    "SearchPerformanceMonitor"
]
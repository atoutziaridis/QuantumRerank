"""
Performance monitoring for similarity computation methods.

This module provides comprehensive monitoring and alerting for similarity
method performance, tracking latency, accuracy, and resource usage.
"""

import time
import threading
from typing import Dict, List, Optional, Any, ContextManager
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from contextlib import contextmanager

from ..utils import get_logger


@dataclass
class MethodPerformanceStats:
    """Performance statistics for a similarity method."""
    method_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    p95_latency_ms: float = 0.0
    avg_accuracy: float = 0.0
    success_rate: float = 0.0
    last_error: Optional[str] = None
    performance_trend: str = "stable"  # "improving", "degrading", "stable"
    
    def update_latency(self, latency_ms: float) -> None:
        """Update latency statistics."""
        self.avg_latency_ms = (
            (self.avg_latency_ms * self.successful_calls + latency_ms) /
            (self.successful_calls + 1)
        )
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
    
    def update_success_rate(self) -> None:
        """Update success rate."""
        if self.total_calls > 0:
            self.success_rate = self.successful_calls / self.total_calls


@dataclass
class PerformanceAlert:
    """Performance alert for degraded method performance."""
    method_name: str
    alert_type: str  # "latency", "accuracy", "failure_rate"
    severity: str    # "warning", "critical"
    message: str
    metric_value: float
    threshold_value: float
    timestamp: float = field(default_factory=time.time)


class PerformanceMonitor:
    """Base performance monitoring with context manager support."""
    
    def __init__(self, method_name: str, monitor: 'SimilarityPerformanceMonitor'):
        self.method_name = method_name
        self.monitor = monitor
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            latency_ms = (time.time() - self.start_time) * 1000
            
            if exc_type is None:
                # Success
                self.monitor.record_success(self.method_name, latency_ms)
            else:
                # Failure
                self.monitor.record_failure(self.method_name, str(exc_val), latency_ms)


class SimilarityPerformanceMonitor:
    """
    Comprehensive performance monitor for similarity computation methods.
    
    This monitor tracks latency, accuracy, success rates, and resource usage
    across all similarity methods, providing alerts and optimization insights.
    """
    
    def __init__(
        self,
        latency_threshold_ms: float = 100.0,
        accuracy_threshold: float = 0.8,
        failure_rate_threshold: float = 0.05,
        history_size: int = 1000
    ):
        self.logger = get_logger(__name__)
        
        # Thresholds for alerts
        self.latency_threshold_ms = latency_threshold_ms
        self.accuracy_threshold = accuracy_threshold
        self.failure_rate_threshold = failure_rate_threshold
        
        # Performance data
        self.method_stats: Dict[str, MethodPerformanceStats] = {}
        self.latency_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.accuracy_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        
        # Alerts
        self.active_alerts: List[PerformanceAlert] = []
        self.alert_history: deque = deque(maxlen=1000)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Monitoring state
        self._monitoring_enabled = True
        
        self.logger.info("Initialized SimilarityPerformanceMonitor")
    
    @contextmanager
    def measure(self, method_name: str) -> ContextManager:
        """Context manager for measuring method performance."""
        yield PerformanceMonitor(method_name, self)
    
    def record_success(
        self,
        method_name: str,
        latency_ms: float,
        accuracy: Optional[float] = None
    ) -> None:
        """Record successful method execution."""
        if not self._monitoring_enabled:
            return
        
        with self._lock:
            # Initialize stats if needed
            if method_name not in self.method_stats:
                self.method_stats[method_name] = MethodPerformanceStats(method_name)
            
            stats = self.method_stats[method_name]
            
            # Update call counts
            stats.total_calls += 1
            stats.successful_calls += 1
            
            # Update latency statistics
            stats.update_latency(latency_ms)
            self.latency_history[method_name].append(latency_ms)
            
            # Update accuracy if provided
            if accuracy is not None:
                old_count = stats.successful_calls - 1
                if old_count > 0:
                    stats.avg_accuracy = (
                        (stats.avg_accuracy * old_count + accuracy) / stats.successful_calls
                    )
                else:
                    stats.avg_accuracy = accuracy
                
                self.accuracy_history[method_name].append(accuracy)
            
            # Update success rate
            stats.update_success_rate()
            
            # Update percentiles
            self._update_percentiles(method_name)
            
            # Check for alerts
            self._check_performance_alerts(method_name, latency_ms, accuracy)
            
            # Update performance trend
            self._update_performance_trend(method_name)
    
    def record_failure(
        self,
        method_name: str,
        error_message: str,
        latency_ms: Optional[float] = None
    ) -> None:
        """Record failed method execution."""
        if not self._monitoring_enabled:
            return
        
        with self._lock:
            # Initialize stats if needed
            if method_name not in self.method_stats:
                self.method_stats[method_name] = MethodPerformanceStats(method_name)
            
            stats = self.method_stats[method_name]
            
            # Update call counts
            stats.total_calls += 1
            stats.failed_calls += 1
            stats.last_error = error_message
            
            # Record latency even for failures
            if latency_ms is not None:
                self.latency_history[method_name].append(latency_ms)
            
            # Update success rate
            stats.update_success_rate()
            
            # Check for failure rate alerts
            self._check_failure_rate_alert(method_name)
            
            self.logger.debug(f"Recorded failure for {method_name}: {error_message}")
    
    def get_method_statistics(self, method_name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive statistics for a method."""
        with self._lock:
            if method_name not in self.method_stats:
                return None
            
            stats = self.method_stats[method_name]
            
            # Compute additional metrics
            recent_latencies = list(self.latency_history[method_name])[-100:]
            recent_accuracies = list(self.accuracy_history[method_name])[-100:]
            
            return {
                "method_name": stats.method_name,
                "total_calls": stats.total_calls,
                "successful_calls": stats.successful_calls,
                "failed_calls": stats.failed_calls,
                "success_rate": stats.success_rate,
                "latency": {
                    "avg_ms": stats.avg_latency_ms,
                    "min_ms": stats.min_latency_ms if stats.min_latency_ms != float('inf') else 0,
                    "max_ms": stats.max_latency_ms,
                    "p95_ms": stats.p95_latency_ms,
                    "recent_avg_ms": np.mean(recent_latencies) if recent_latencies else 0
                },
                "accuracy": {
                    "avg": stats.avg_accuracy,
                    "recent_avg": np.mean(recent_accuracies) if recent_accuracies else 0
                },
                "performance_trend": stats.performance_trend,
                "last_error": stats.last_error
            }
    
    def get_overall_statistics(self) -> Dict[str, Any]:
        """Get overall performance statistics across all methods."""
        with self._lock:
            if not self.method_stats:
                return {"message": "No performance data available"}
            
            total_calls = sum(stats.total_calls for stats in self.method_stats.values())
            total_successes = sum(stats.successful_calls for stats in self.method_stats.values())
            total_failures = sum(stats.failed_calls for stats in self.method_stats.values())
            
            # Aggregate latency statistics
            all_latencies = []
            for history in self.latency_history.values():
                all_latencies.extend(history)
            
            # Method rankings
            method_rankings = self._compute_method_rankings()
            
            return {
                "total_calls": total_calls,
                "total_successes": total_successes,
                "total_failures": total_failures,
                "overall_success_rate": total_successes / total_calls if total_calls > 0 else 0,
                "overall_latency": {
                    "avg_ms": np.mean(all_latencies) if all_latencies else 0,
                    "p95_ms": np.percentile(all_latencies, 95) if all_latencies else 0,
                    "p99_ms": np.percentile(all_latencies, 99) if all_latencies else 0
                },
                "method_count": len(self.method_stats),
                "active_alerts": len(self.active_alerts),
                "method_rankings": method_rankings,
                "monitoring_enabled": self._monitoring_enabled
            }
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get currently active performance alerts."""
        with self._lock:
            return self.active_alerts.copy()
    
    def get_alert_history(self, limit: int = 50) -> List[PerformanceAlert]:
        """Get recent alert history."""
        with self._lock:
            return list(self.alert_history)[-limit:]
    
    def clear_alerts(self, method_name: Optional[str] = None) -> int:
        """Clear alerts for a specific method or all alerts."""
        with self._lock:
            if method_name is None:
                count = len(self.active_alerts)
                self.active_alerts.clear()
                return count
            else:
                initial_count = len(self.active_alerts)
                self.active_alerts = [
                    alert for alert in self.active_alerts
                    if alert.method_name != method_name
                ]
                return initial_count - len(self.active_alerts)
    
    def set_thresholds(
        self,
        latency_ms: Optional[float] = None,
        accuracy: Optional[float] = None,
        failure_rate: Optional[float] = None
    ) -> None:
        """Update performance thresholds."""
        with self._lock:
            if latency_ms is not None:
                self.latency_threshold_ms = latency_ms
            
            if accuracy is not None:
                self.accuracy_threshold = accuracy
            
            if failure_rate is not None:
                self.failure_rate_threshold = failure_rate
            
            self.logger.info(
                f"Updated thresholds: latency={self.latency_threshold_ms}ms, "
                f"accuracy={self.accuracy_threshold}, failure_rate={self.failure_rate_threshold}"
            )
    
    def enable_monitoring(self) -> None:
        """Enable performance monitoring."""
        self._monitoring_enabled = True
        self.logger.info("Performance monitoring enabled")
    
    def disable_monitoring(self) -> None:
        """Disable performance monitoring."""
        self._monitoring_enabled = False
        self.logger.info("Performance monitoring disabled")
    
    def reset_statistics(self, method_name: Optional[str] = None) -> None:
        """Reset statistics for specific method or all methods."""
        with self._lock:
            if method_name is None:
                # Reset all
                self.method_stats.clear()
                self.latency_history.clear()
                self.accuracy_history.clear()
                self.active_alerts.clear()
                self.logger.info("Reset all performance statistics")
            else:
                # Reset specific method
                if method_name in self.method_stats:
                    del self.method_stats[method_name]
                
                if method_name in self.latency_history:
                    self.latency_history[method_name].clear()
                
                if method_name in self.accuracy_history:
                    self.accuracy_history[method_name].clear()
                
                # Remove alerts for this method
                self.clear_alerts(method_name)
                
                self.logger.info(f"Reset statistics for method: {method_name}")
    
    def _update_percentiles(self, method_name: str) -> None:
        """Update percentile statistics for a method."""
        latencies = list(self.latency_history[method_name])
        if len(latencies) >= 20:  # Need sufficient data
            self.method_stats[method_name].p95_latency_ms = float(np.percentile(latencies, 95))
    
    def _check_performance_alerts(
        self,
        method_name: str,
        latency_ms: float,
        accuracy: Optional[float]
    ) -> None:
        """Check for performance alerts."""
        # Latency alert
        if latency_ms > self.latency_threshold_ms:
            severity = "critical" if latency_ms > self.latency_threshold_ms * 2 else "warning"
            alert = PerformanceAlert(
                method_name=method_name,
                alert_type="latency",
                severity=severity,
                message=f"High latency detected: {latency_ms:.1f}ms",
                metric_value=latency_ms,
                threshold_value=self.latency_threshold_ms
            )
            self._add_alert(alert)
        
        # Accuracy alert
        if accuracy is not None and accuracy < self.accuracy_threshold:
            severity = "critical" if accuracy < self.accuracy_threshold * 0.8 else "warning"
            alert = PerformanceAlert(
                method_name=method_name,
                alert_type="accuracy",
                severity=severity,
                message=f"Low accuracy detected: {accuracy:.3f}",
                metric_value=accuracy,
                threshold_value=self.accuracy_threshold
            )
            self._add_alert(alert)
    
    def _check_failure_rate_alert(self, method_name: str) -> None:
        """Check for failure rate alerts."""
        stats = self.method_stats[method_name]
        
        if stats.total_calls >= 10:  # Need sufficient data
            failure_rate = stats.failed_calls / stats.total_calls
            
            if failure_rate > self.failure_rate_threshold:
                severity = "critical" if failure_rate > self.failure_rate_threshold * 2 else "warning"
                alert = PerformanceAlert(
                    method_name=method_name,
                    alert_type="failure_rate",
                    severity=severity,
                    message=f"High failure rate: {failure_rate:.1%}",
                    metric_value=failure_rate,
                    threshold_value=self.failure_rate_threshold
                )
                self._add_alert(alert)
    
    def _add_alert(self, alert: PerformanceAlert) -> None:
        """Add alert if not already active."""
        # Check if similar alert already exists
        existing = any(
            a.method_name == alert.method_name and
            a.alert_type == alert.alert_type and
            a.severity == alert.severity
            for a in self.active_alerts
        )
        
        if not existing:
            self.active_alerts.append(alert)
            self.alert_history.append(alert)
            
            self.logger.warning(
                f"Performance alert: {alert.method_name} - {alert.message}"
            )
    
    def _update_performance_trend(self, method_name: str) -> None:
        """Update performance trend for a method."""
        recent_latencies = list(self.latency_history[method_name])[-20:]
        
        if len(recent_latencies) >= 10:
            # Split into two halves and compare
            mid = len(recent_latencies) // 2
            first_half = recent_latencies[:mid]
            second_half = recent_latencies[mid:]
            
            avg_first = np.mean(first_half)
            avg_second = np.mean(second_half)
            
            # Determine trend
            if avg_second < avg_first * 0.9:
                trend = "improving"
            elif avg_second > avg_first * 1.1:
                trend = "degrading"
            else:
                trend = "stable"
            
            self.method_stats[method_name].performance_trend = trend
    
    def _compute_method_rankings(self) -> Dict[str, Any]:
        """Compute method rankings based on performance."""
        if not self.method_stats:
            return {}
        
        # Rank by latency (lower is better)
        latency_ranking = sorted(
            self.method_stats.items(),
            key=lambda x: x[1].avg_latency_ms
        )
        
        # Rank by accuracy (higher is better)
        accuracy_ranking = sorted(
            self.method_stats.items(),
            key=lambda x: x[1].avg_accuracy,
            reverse=True
        )
        
        # Rank by success rate (higher is better)
        reliability_ranking = sorted(
            self.method_stats.items(),
            key=lambda x: x[1].success_rate,
            reverse=True
        )
        
        return {
            "fastest_methods": [name for name, _ in latency_ranking[:3]],
            "most_accurate_methods": [name for name, _ in accuracy_ranking[:3]],
            "most_reliable_methods": [name for name, _ in reliability_ranking[:3]]
        }


__all__ = [
    "MethodPerformanceStats",
    "PerformanceAlert", 
    "SimilarityPerformanceMonitor"
]
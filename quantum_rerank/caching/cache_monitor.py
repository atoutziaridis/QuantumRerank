"""
Cache performance monitoring and alerting system.

This module provides comprehensive monitoring of cache performance across
all cache levels with alerting and optimization recommendations.
"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum

from ..utils import get_logger


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    cache_type: str
    hit_count: int = 0
    miss_count: int = 0
    total_lookups: int = 0
    hit_rate: float = 0.0
    average_lookup_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    eviction_count: int = 0
    error_count: int = 0
    last_updated: float = field(default_factory=time.time)
    
    def update_hit_rate(self) -> None:
        """Update hit rate calculation."""
        if self.total_lookups > 0:
            self.hit_rate = self.hit_count / self.total_lookups


@dataclass
class CacheAlert:
    """Cache performance alert."""
    cache_type: str
    severity: AlertSeverity
    metric: str
    message: str
    current_value: float
    threshold_value: float
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False


class CachePerformanceMonitor:
    """
    Comprehensive cache performance monitoring system.
    
    This monitor tracks performance across all cache types and provides
    alerting when performance degrades below acceptable thresholds.
    """
    
    def __init__(
        self,
        target_hit_rates: Optional[Dict[str, float]] = None,
        target_lookup_time_ms: float = 2.0,
        alert_threshold_factor: float = 0.8,
        history_size: int = 1000
    ):
        self.target_hit_rates = target_hit_rates or {
            "similarity": 0.25,
            "quantum": 0.15,
            "embedding": 0.60
        }
        self.target_lookup_time_ms = target_lookup_time_ms
        self.alert_threshold_factor = alert_threshold_factor
        
        self.logger = get_logger(__name__)
        
        # Current metrics for each cache type
        self.current_metrics: Dict[str, CacheMetrics] = {}
        
        # Historical data
        self.lookup_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.hit_rates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.memory_usage: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        
        # Alerts
        self.active_alerts: List[CacheAlert] = []
        self.alert_history: deque = deque(maxlen=1000)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Monitoring state
        self.monitoring_enabled = True
        
        self.logger.info("Initialized CachePerformanceMonitor")
    
    def record_cache_hit(
        self,
        cache_type: str,
        hit_type: str = "exact",
        lookup_time_ms: float = 0.0
    ) -> None:
        """Record cache hit event."""
        if not self.monitoring_enabled:
            return
        
        with self._lock:
            # Initialize metrics if needed
            if cache_type not in self.current_metrics:
                self.current_metrics[cache_type] = CacheMetrics(cache_type=cache_type)
            
            metrics = self.current_metrics[cache_type]
            
            # Update hit metrics
            metrics.hit_count += 1
            metrics.total_lookups += 1
            metrics.update_hit_rate()
            
            # Update timing
            if lookup_time_ms > 0:
                self._update_lookup_time(cache_type, lookup_time_ms)
            
            # Record historical data
            self.hit_rates[cache_type].append(metrics.hit_rate)
            
            # Check for performance alerts
            self._check_performance_alerts(cache_type)
    
    def record_cache_miss(
        self,
        cache_type: str,
        method: str = "",
        lookup_time_ms: float = 0.0
    ) -> None:
        """Record cache miss event."""
        if not self.monitoring_enabled:
            return
        
        with self._lock:
            # Initialize metrics if needed
            if cache_type not in self.current_metrics:
                self.current_metrics[cache_type] = CacheMetrics(cache_type=cache_type)
            
            metrics = self.current_metrics[cache_type]
            
            # Update miss metrics
            metrics.miss_count += 1
            metrics.total_lookups += 1
            metrics.update_hit_rate()
            
            # Update timing
            if lookup_time_ms > 0:
                self._update_lookup_time(cache_type, lookup_time_ms)
            
            # Record historical data
            self.hit_rates[cache_type].append(metrics.hit_rate)
            
            # Check for performance alerts
            self._check_performance_alerts(cache_type)
    
    def record_cache_store(
        self,
        cache_type: str,
        store_time_ms: float = 0.0,
        computation_time_ms: Optional[float] = None
    ) -> None:
        """Record cache store operation."""
        if not self.monitoring_enabled:
            return
        
        with self._lock:
            # Initialize metrics if needed
            if cache_type not in self.current_metrics:
                self.current_metrics[cache_type] = CacheMetrics(cache_type=cache_type)
            
            # Update timing if provided
            if store_time_ms > 0:
                self._update_lookup_time(cache_type, store_time_ms)
    
    def update_memory_usage(
        self,
        cache_type: str,
        memory_usage_mb: float
    ) -> None:
        """Update memory usage for cache type."""
        if not self.monitoring_enabled:
            return
        
        with self._lock:
            # Initialize metrics if needed
            if cache_type not in self.current_metrics:
                self.current_metrics[cache_type] = CacheMetrics(cache_type=cache_type)
            
            metrics = self.current_metrics[cache_type]
            metrics.memory_usage_mb = memory_usage_mb
            
            # Record historical data
            self.memory_usage[cache_type].append(memory_usage_mb)
            
            # Check memory alerts
            self._check_memory_alerts(cache_type, memory_usage_mb)
    
    def record_eviction(self, cache_type: str, count: int = 1) -> None:
        """Record cache eviction events."""
        if not self.monitoring_enabled:
            return
        
        with self._lock:
            if cache_type in self.current_metrics:
                self.current_metrics[cache_type].eviction_count += count
    
    def record_error(self, cache_type: str, error_message: str) -> None:
        """Record cache error."""
        if not self.monitoring_enabled:
            return
        
        with self._lock:
            if cache_type in self.current_metrics:
                self.current_metrics[cache_type].error_count += 1
            
            # Create critical alert for errors
            alert = CacheAlert(
                cache_type=cache_type,
                severity=AlertSeverity.CRITICAL,
                metric="error_rate",
                message=f"Cache error: {error_message}",
                current_value=1.0,
                threshold_value=0.0
            )
            
            self._add_alert(alert)
    
    def _update_lookup_time(self, cache_type: str, lookup_time_ms: float) -> None:
        """Update average lookup time."""
        metrics = self.current_metrics[cache_type]
        
        # Exponential moving average
        if metrics.average_lookup_time_ms == 0:
            metrics.average_lookup_time_ms = lookup_time_ms
        else:
            alpha = 0.1
            metrics.average_lookup_time_ms = (
                alpha * lookup_time_ms + (1 - alpha) * metrics.average_lookup_time_ms
            )
        
        # Record historical data
        self.lookup_times[cache_type].append(lookup_time_ms)
    
    def _check_performance_alerts(self, cache_type: str) -> None:
        """Check for performance degradation alerts."""
        metrics = self.current_metrics[cache_type]
        
        # Check hit rate alert
        target_hit_rate = self.target_hit_rates.get(cache_type, 0.2)
        alert_threshold = target_hit_rate * self.alert_threshold_factor
        
        if metrics.total_lookups >= 50 and metrics.hit_rate < alert_threshold:
            alert = CacheAlert(
                cache_type=cache_type,
                severity=AlertSeverity.WARNING if metrics.hit_rate > alert_threshold * 0.5 else AlertSeverity.CRITICAL,
                metric="hit_rate",
                message=f"Low hit rate: {metrics.hit_rate:.2%} (target: {target_hit_rate:.2%})",
                current_value=metrics.hit_rate,
                threshold_value=target_hit_rate
            )
            self._add_alert(alert)
        
        # Check lookup time alert
        if metrics.average_lookup_time_ms > self.target_lookup_time_ms * 2:
            alert = CacheAlert(
                cache_type=cache_type,
                severity=AlertSeverity.WARNING,
                metric="lookup_time",
                message=f"High lookup time: {metrics.average_lookup_time_ms:.1f}ms (target: {self.target_lookup_time_ms:.1f}ms)",
                current_value=metrics.average_lookup_time_ms,
                threshold_value=self.target_lookup_time_ms
            )
            self._add_alert(alert)
    
    def _check_memory_alerts(self, cache_type: str, memory_usage_mb: float) -> None:
        """Check for memory usage alerts."""
        # Define memory thresholds (these could be configurable)
        memory_thresholds = {
            "similarity": 600,  # 512MB + 20% buffer
            "quantum": 300,     # 256MB + 20% buffer  
            "embedding": 1200   # 1024MB + 20% buffer
        }
        
        threshold = memory_thresholds.get(cache_type, 500)
        
        if memory_usage_mb > threshold:
            severity = AlertSeverity.CRITICAL if memory_usage_mb > threshold * 1.2 else AlertSeverity.WARNING
            
            alert = CacheAlert(
                cache_type=cache_type,
                severity=severity,
                metric="memory_usage",
                message=f"High memory usage: {memory_usage_mb:.1f}MB (threshold: {threshold:.1f}MB)",
                current_value=memory_usage_mb,
                threshold_value=threshold
            )
            self._add_alert(alert)
    
    def _add_alert(self, alert: CacheAlert) -> None:
        """Add alert if not already active."""
        # Check for duplicate alerts
        existing = any(
            a.cache_type == alert.cache_type and
            a.metric == alert.metric and
            a.severity == alert.severity and
            not a.acknowledged
            for a in self.active_alerts
        )
        
        if not existing:
            self.active_alerts.append(alert)
            self.alert_history.append(alert)
            
            self.logger.warning(
                f"Cache alert ({alert.severity.value}): {alert.cache_type} - {alert.message}"
            )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self._lock:
            summary = {
                "overall_status": self._get_overall_status(),
                "cache_metrics": {},
                "alerts": {
                    "active_count": len(self.active_alerts),
                    "total_alerts": len(self.alert_history),
                    "critical_alerts": len([a for a in self.active_alerts if a.severity == AlertSeverity.CRITICAL])
                },
                "targets": self.target_hit_rates.copy()
            }
            
            # Add per-cache metrics
            for cache_type, metrics in self.current_metrics.items():
                summary["cache_metrics"][cache_type] = {
                    "hit_rate": metrics.hit_rate,
                    "total_lookups": metrics.total_lookups,
                    "average_lookup_time_ms": metrics.average_lookup_time_ms,
                    "memory_usage_mb": metrics.memory_usage_mb,
                    "eviction_count": metrics.eviction_count,
                    "error_count": metrics.error_count,
                    "meets_hit_rate_target": metrics.hit_rate >= self.target_hit_rates.get(cache_type, 0.2),
                    "meets_timing_target": metrics.average_lookup_time_ms <= self.target_lookup_time_ms
                }
            
            # Add trending information
            summary["trends"] = self._compute_trends()
        
        return summary
    
    def _get_overall_status(self) -> str:
        """Determine overall cache system status."""
        critical_alerts = [a for a in self.active_alerts if a.severity == AlertSeverity.CRITICAL]
        warning_alerts = [a for a in self.active_alerts if a.severity == AlertSeverity.WARNING]
        
        if critical_alerts:
            return "critical"
        elif warning_alerts:
            return "warning"
        elif self.current_metrics:
            # Check if majority of caches meet performance targets
            meeting_targets = 0
            total_caches = len(self.current_metrics)
            
            for cache_type, metrics in self.current_metrics.items():
                target_hit_rate = self.target_hit_rates.get(cache_type, 0.2)
                
                if (metrics.hit_rate >= target_hit_rate and 
                    metrics.average_lookup_time_ms <= self.target_lookup_time_ms):
                    meeting_targets += 1
            
            if meeting_targets >= total_caches * 0.8:  # 80% threshold
                return "healthy"
            else:
                return "degraded"
        else:
            return "unknown"
    
    def _compute_trends(self) -> Dict[str, Any]:
        """Compute performance trends."""
        trends = {}
        
        for cache_type in self.current_metrics:
            cache_trends = {}
            
            # Hit rate trend
            if len(self.hit_rates[cache_type]) >= 10:
                recent_hit_rates = list(self.hit_rates[cache_type])[-10:]
                early_avg = sum(recent_hit_rates[:5]) / 5
                late_avg = sum(recent_hit_rates[5:]) / 5
                
                if late_avg > early_avg * 1.05:
                    cache_trends["hit_rate_trend"] = "improving"
                elif late_avg < early_avg * 0.95:
                    cache_trends["hit_rate_trend"] = "degrading"
                else:
                    cache_trends["hit_rate_trend"] = "stable"
            
            # Lookup time trend
            if len(self.lookup_times[cache_type]) >= 10:
                recent_times = list(self.lookup_times[cache_type])[-10:]
                early_avg = sum(recent_times[:5]) / 5
                late_avg = sum(recent_times[5:]) / 5
                
                if late_avg < early_avg * 0.95:
                    cache_trends["lookup_time_trend"] = "improving"
                elif late_avg > early_avg * 1.05:
                    cache_trends["lookup_time_trend"] = "degrading"
                else:
                    cache_trends["lookup_time_trend"] = "stable"
            
            if cache_trends:
                trends[cache_type] = cache_trends
        
        return trends
    
    def get_active_alerts(self) -> List[CacheAlert]:
        """Get currently active alerts."""
        with self._lock:
            return [alert for alert in self.active_alerts if not alert.acknowledged]
    
    def acknowledge_alert(self, alert_id: int) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            if 0 <= alert_id < len(self.active_alerts):
                self.active_alerts[alert_id].acknowledged = True
                return True
            return False
    
    def clear_acknowledged_alerts(self) -> int:
        """Clear acknowledged alerts."""
        with self._lock:
            initial_count = len(self.active_alerts)
            self.active_alerts = [a for a in self.active_alerts if not a.acknowledged]
            return initial_count - len(self.active_alerts)
    
    def reset_metrics(self, cache_types: Optional[List[str]] = None) -> None:
        """Reset metrics for specified cache types."""
        with self._lock:
            target_types = cache_types or list(self.current_metrics.keys())
            
            for cache_type in target_types:
                if cache_type in self.current_metrics:
                    self.current_metrics[cache_type] = CacheMetrics(cache_type=cache_type)
                
                # Clear historical data
                self.lookup_times[cache_type].clear()
                self.hit_rates[cache_type].clear()
                self.memory_usage[cache_type].clear()
            
            # Clear related alerts
            self.active_alerts = [
                a for a in self.active_alerts 
                if a.cache_type not in target_types
            ]
    
    def enable_monitoring(self) -> None:
        """Enable performance monitoring."""
        self.monitoring_enabled = True
        self.logger.info("Cache performance monitoring enabled")
    
    def disable_monitoring(self) -> None:
        """Disable performance monitoring."""
        self.monitoring_enabled = False
        self.logger.info("Cache performance monitoring disabled")


__all__ = [
    "AlertSeverity",
    "CacheMetrics",
    "CacheAlert",
    "CachePerformanceMonitor"
]
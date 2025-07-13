"""
Real-time performance tracking system for comprehensive system monitoring.

This module provides high-performance monitoring with minimal overhead
for tracking all aspects of QuantumRerank performance.
"""

import time
import threading
import psutil
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import numpy as np

from .metrics_collector import MetricsCollector, MetricType
from ..utils import get_logger


@dataclass
class PerformanceTarget:
    """Performance target specification."""
    metric_name: str
    target_value: float
    warning_threshold: float
    critical_threshold: float
    unit: str = ""
    higher_is_better: bool = False


@dataclass
class PerformanceContext:
    """Context for tracking operation performance."""
    operation_name: str
    start_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        success = exc_type is None
        
        # Record metrics through global tracker
        tracker = get_performance_tracker()
        if tracker:
            tracker.record_operation_performance(
                self.operation_name, duration_ms, success, self.metadata
            )


class RealTimePerformanceTracker:
    """
    High-performance real-time monitoring system.
    
    Provides comprehensive performance tracking with minimal overhead,
    supporting quantum computation monitoring, pipeline analytics,
    and automated optimization triggers.
    """
    
    def __init__(self, collection_interval_ms: int = 100):
        self.collection_interval_ms = collection_interval_ms
        self.logger = get_logger(__name__)
        
        # Core components
        self.metrics_collector = MetricsCollector(collection_interval_ms=collection_interval_ms)
        
        # Performance targets
        self.performance_targets = self._initialize_performance_targets()
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        
        # Performance state
        self.monitoring_enabled = True
        self.start_time = time.time()
        
        # Background monitoring
        self._monitoring_thread = None
        self._monitoring_active = False
        
        self.logger.info("Initialized RealTimePerformanceTracker")
    
    def start_monitoring(self) -> None:
        """Start real-time performance monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        
        # Start metrics collector background processing
        self.metrics_collector.start_background_processing()
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        # Start monitoring thread
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self._monitoring_thread.start()
        
        self.logger.info("Started real-time performance monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop real-time performance monitoring."""
        self._monitoring_active = False
        
        # Stop background components
        self.metrics_collector.stop_background_processing()
        self.resource_monitor.stop_monitoring()
        
        # Wait for monitoring thread
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Stopped real-time performance monitoring")
    
    def track_operation(self, operation_name: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> PerformanceContext:
        """
        Create context manager for tracking operation performance.
        
        Args:
            operation_name: Name of the operation to track
            metadata: Additional metadata to track
            
        Returns:
            Performance context manager
        """
        return PerformanceContext(operation_name, time.time(), metadata or {})
    
    def record_operation_performance(self, operation_name: str, 
                                   duration_ms: float, success: bool,
                                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record operation performance metrics."""
        if not self.monitoring_enabled:
            return
        
        tags = {"operation": operation_name, "status": "success" if success else "error"}
        
        # Add metadata as tags
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, bool)):
                    tags[f"meta_{key}"] = str(value)
        
        # Record core metrics
        self.metrics_collector.record_timer(f"operation.{operation_name}.duration", 
                                          duration_ms, tags)
        self.metrics_collector.record_counter(f"operation.{operation_name}.count", 1, tags)
        
        if not success:
            self.metrics_collector.record_counter(f"operation.{operation_name}.errors", 1, tags)
        
        # Record operation-specific metrics
        self._record_operation_specific_metrics(operation_name, duration_ms, success, metadata)
    
    def record_quantum_performance(self, computation_type: str, 
                                 metrics: Dict[str, Any]) -> None:
        """Record quantum computation performance metrics."""
        if not self.monitoring_enabled:
            return
        
        tags = {"computation_type": computation_type, "component": "quantum"}
        
        # Core quantum metrics
        if "execution_time_ms" in metrics:
            self.metrics_collector.record_timer("quantum.execution_time", 
                                              metrics["execution_time_ms"], tags)
        
        if "circuit_depth" in metrics:
            self.metrics_collector.record_gauge("quantum.circuit_depth", 
                                               metrics["circuit_depth"], "", tags)
        
        if "gate_count" in metrics:
            self.metrics_collector.record_gauge("quantum.gate_count", 
                                               metrics["gate_count"], "", tags)
        
        if "fidelity_value" in metrics:
            self.metrics_collector.record_gauge("quantum.fidelity", 
                                               metrics["fidelity_value"], "", tags)
        
        if "parameter_quality_score" in metrics:
            self.metrics_collector.record_gauge("quantum.parameter_quality", 
                                               metrics["parameter_quality_score"], "", tags)
        
        # Memory and resource usage
        if "memory_usage_mb" in metrics:
            self.metrics_collector.record_gauge("quantum.memory_usage", 
                                               metrics["memory_usage_mb"], "MB", tags)
        
        # Success/error tracking
        success = metrics.get("success", True)
        self.metrics_collector.record_counter("quantum.operations", 1, 
                                            {**tags, "status": "success" if success else "error"})
        
        # Anomalies
        if "anomalies" in metrics and metrics["anomalies"]:
            self.metrics_collector.record_counter("quantum.anomalies", 
                                                len(metrics["anomalies"]), tags)
    
    def record_pipeline_performance(self, pipeline_stage: str,
                                   stage_metrics: Dict[str, Any]) -> None:
        """Record pipeline stage performance metrics."""
        if not self.monitoring_enabled:
            return
        
        tags = {"pipeline_stage": pipeline_stage, "component": "pipeline"}
        
        # Stage timing
        if "duration_ms" in stage_metrics:
            self.metrics_collector.record_timer(f"pipeline.{pipeline_stage}.duration", 
                                              stage_metrics["duration_ms"], tags)
        
        # Throughput metrics
        if "items_processed" in stage_metrics:
            self.metrics_collector.record_gauge(f"pipeline.{pipeline_stage}.throughput", 
                                               stage_metrics["items_processed"], "items", tags)
        
        # Quality metrics
        if "accuracy_score" in stage_metrics:
            self.metrics_collector.record_gauge(f"pipeline.{pipeline_stage}.accuracy", 
                                               stage_metrics["accuracy_score"], "", tags)
        
        # Cache performance
        if "cache_hit_rate" in stage_metrics:
            self.metrics_collector.record_gauge(f"pipeline.{pipeline_stage}.cache_hit_rate", 
                                               stage_metrics["cache_hit_rate"], "", tags)
        
        # Resource usage
        if "memory_usage_mb" in stage_metrics:
            self.metrics_collector.record_gauge(f"pipeline.{pipeline_stage}.memory", 
                                               stage_metrics["memory_usage_mb"], "MB", tags)
    
    def get_performance_summary(self, time_window_seconds: int = 300) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.
        
        Args:
            time_window_seconds: Time window for summary (default 5 minutes)
            
        Returns:
            Performance summary with key metrics and health indicators
        """
        summary = {
            "timestamp": time.time(),
            "monitoring_duration_seconds": time.time() - self.start_time,
            "time_window_seconds": time_window_seconds,
            "system_health": self._assess_system_health(),
            "performance_metrics": {},
            "resource_utilization": self.resource_monitor.get_current_usage(),
            "performance_targets": self._check_performance_targets()
        }
        
        # Get metrics summary from collector
        metrics_summary = self.metrics_collector.get_metrics_summary()
        summary["collection_stats"] = {
            "total_metrics": metrics_summary["total_metrics"],
            "total_samples": metrics_summary["total_samples"],
            "collection_overhead_ms": metrics_summary["collection_overhead_ms"]
        }
        
        # Extract key performance indicators
        key_metrics = [
            "operation.similarity_computation.duration",
            "operation.vector_search.duration", 
            "operation.quantum_computation.duration",
            "pipeline.retrieval.duration",
            "pipeline.rerank.duration",
            "quantum.execution_time",
            "quantum.fidelity"
        ]
        
        for metric_name in key_metrics:
            stats = self.metrics_collector.get_metric_statistics(metric_name)
            if stats:
                summary["performance_metrics"][metric_name] = {
                    "mean": stats.mean,
                    "p95": stats.p95,
                    "p99": stats.p99,
                    "count": stats.count,
                    "last_value": stats.last_value
                }
        
        return summary
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics for dashboards."""
        current_time = time.time()
        
        # Get recent performance data
        recent_metrics = {}
        
        # Key latency metrics (last 60 seconds)
        for metric_name in ["operation.similarity_computation.duration", 
                           "operation.vector_search.duration",
                           "quantum.execution_time"]:
            recent_samples = self.metrics_collector.get_samples_in_time_window(
                metric_name, 60
            )
            if recent_samples:
                values = [sample.value for sample in recent_samples]
                recent_metrics[metric_name] = {
                    "current": values[-1] if values else 0,
                    "avg_1min": np.mean(values),
                    "count_1min": len(values)
                }
        
        # System resource metrics
        resource_usage = self.resource_monitor.get_current_usage()
        
        return {
            "timestamp": current_time,
            "latency_metrics": recent_metrics,
            "resource_metrics": resource_usage,
            "system_health": self._assess_system_health(),
            "active_operations": self._get_active_operations_count()
        }
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop for system health checks."""
        while self._monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check performance targets
                self._check_performance_targets()
                
                # Sleep until next collection
                time.sleep(self.collection_interval_ms / 1000.0)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)
    
    def _collect_system_metrics(self) -> None:
        """Collect system-level performance metrics."""
        # CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        
        self.metrics_collector.record_gauge("system.cpu_usage", cpu_percent, "%")
        self.metrics_collector.record_gauge("system.memory_usage", 
                                          memory_info.percent, "%")
        self.metrics_collector.record_gauge("system.memory_available", 
                                          memory_info.available / (1024**3), "GB")
        
        # Process-specific metrics
        process = psutil.Process()
        process_memory_mb = process.memory_info().rss / (1024**2)
        
        self.metrics_collector.record_gauge("process.memory_usage", 
                                          process_memory_mb, "MB")
        self.metrics_collector.record_gauge("process.cpu_usage", 
                                          process.cpu_percent(), "%")
        self.metrics_collector.record_gauge("process.thread_count", 
                                          process.num_threads(), "")
    
    def _record_operation_specific_metrics(self, operation_name: str, 
                                         duration_ms: float, success: bool,
                                         metadata: Optional[Dict[str, Any]]) -> None:
        """Record operation-specific performance metrics."""
        # Quantum-related operations
        if "quantum" in operation_name.lower():
            if metadata and "fidelity" in metadata:
                self.metrics_collector.record_gauge("quantum.operation_fidelity", 
                                                   metadata["fidelity"], "")
        
        # Vector search operations
        if "vector_search" in operation_name or "retrieval" in operation_name:
            if metadata:
                if "result_count" in metadata:
                    self.metrics_collector.record_gauge("search.result_count", 
                                                       metadata["result_count"], "")
                if "cache_hit" in metadata:
                    cache_status = "hit" if metadata["cache_hit"] else "miss"
                    self.metrics_collector.record_counter("search.cache", 1, 
                                                        {"status": cache_status})
        
        # Similarity computation operations
        if "similarity" in operation_name:
            if metadata and "method" in metadata:
                method_tags = {"method": metadata["method"]}
                self.metrics_collector.record_counter("similarity.computations", 1, method_tags)
    
    def _assess_system_health(self) -> str:
        """Assess overall system health based on current metrics."""
        # Check resource usage
        resource_usage = self.resource_monitor.get_current_usage()
        
        # Health indicators
        cpu_health = "healthy" if resource_usage["cpu_percent"] < 80 else "warning"
        memory_health = "healthy" if resource_usage["memory_percent"] < 85 else "warning"
        
        # Check performance metrics
        performance_health = "healthy"
        target_violations = 0
        
        for target in self.performance_targets:
            stats = self.metrics_collector.get_metric_statistics(target.metric_name)
            if stats and stats.count > 0:
                current_value = stats.mean
                
                if target.higher_is_better:
                    if current_value < target.critical_threshold:
                        target_violations += 2
                    elif current_value < target.warning_threshold:
                        target_violations += 1
                else:
                    if current_value > target.critical_threshold:
                        target_violations += 2
                    elif current_value > target.warning_threshold:
                        target_violations += 1
        
        if target_violations >= 4:
            performance_health = "critical"
        elif target_violations >= 2:
            performance_health = "warning"
        
        # Overall health assessment
        health_scores = [cpu_health, memory_health, performance_health]
        
        if "critical" in health_scores:
            return "critical"
        elif "warning" in health_scores:
            return "warning"
        else:
            return "healthy"
    
    def _check_performance_targets(self) -> Dict[str, Any]:
        """Check performance against targets."""
        target_status = {}
        
        for target in self.performance_targets:
            stats = self.metrics_collector.get_metric_statistics(target.metric_name)
            
            if stats and stats.count > 0:
                current_value = stats.mean
                
                if target.higher_is_better:
                    if current_value >= target.target_value:
                        status = "meeting_target"
                    elif current_value >= target.warning_threshold:
                        status = "warning"
                    else:
                        status = "critical"
                else:
                    if current_value <= target.target_value:
                        status = "meeting_target"
                    elif current_value <= target.warning_threshold:
                        status = "warning"
                    else:
                        status = "critical"
                
                target_status[target.metric_name] = {
                    "status": status,
                    "current_value": current_value,
                    "target_value": target.target_value,
                    "unit": target.unit
                }
        
        return target_status
    
    def _get_active_operations_count(self) -> int:
        """Get count of currently active operations."""
        # This would track active contexts, simplified for now
        return threading.active_count() - 1  # Subtract main thread
    
    def _initialize_performance_targets(self) -> List[PerformanceTarget]:
        """Initialize performance targets based on PRD requirements."""
        return [
            PerformanceTarget(
                metric_name="operation.similarity_computation.duration",
                target_value=85.0,  # ms
                warning_threshold=120.0,
                critical_threshold=200.0,
                unit="ms"
            ),
            PerformanceTarget(
                metric_name="quantum.execution_time",
                target_value=60.0,  # ms
                warning_threshold=100.0,
                critical_threshold=150.0,
                unit="ms"
            ),
            PerformanceTarget(
                metric_name="pipeline.retrieval.duration",
                target_value=50.0,  # ms
                warning_threshold=100.0,
                critical_threshold=200.0,
                unit="ms"
            ),
            PerformanceTarget(
                metric_name="pipeline.rerank.duration",
                target_value=300.0,  # ms
                warning_threshold=500.0,
                critical_threshold=750.0,
                unit="ms"
            ),
            PerformanceTarget(
                metric_name="quantum.fidelity",
                target_value=0.95,
                warning_threshold=0.90,
                critical_threshold=0.85,
                unit="",
                higher_is_better=True
            )
        ]


class ResourceMonitor:
    """System resource monitoring component."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._monitoring = False
        self._monitor_thread = None
        self.current_usage = {}
    
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        return self.current_usage.copy()
    
    def _monitoring_loop(self) -> None:
        """Resource monitoring loop."""
        while self._monitoring:
            try:
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=1.0)
                memory_info = psutil.virtual_memory()
                
                # Process metrics
                process = psutil.Process()
                process_memory_mb = process.memory_info().rss / (1024**2)
                
                self.current_usage = {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_info.percent,
                    "memory_available_gb": memory_info.available / (1024**3),
                    "process_memory_mb": process_memory_mb,
                    "process_cpu_percent": process.cpu_percent(),
                    "thread_count": process.num_threads()
                }
                
                time.sleep(5.0)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                time.sleep(5.0)


# Global performance tracker instance
_performance_tracker: Optional[RealTimePerformanceTracker] = None


def get_performance_tracker() -> Optional[RealTimePerformanceTracker]:
    """Get global performance tracker instance."""
    global _performance_tracker
    return _performance_tracker


def initialize_performance_tracking(collection_interval_ms: int = 100) -> None:
    """Initialize global performance tracking."""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = RealTimePerformanceTracker(collection_interval_ms)
        _performance_tracker.start_monitoring()


def shutdown_performance_tracking() -> None:
    """Shutdown global performance tracking."""
    global _performance_tracker
    if _performance_tracker:
        _performance_tracker.stop_monitoring()
        _performance_tracker = None


@contextmanager
def track_operation(operation_name: str, metadata: Optional[Dict[str, Any]] = None):
    """Context manager for tracking operation performance."""
    tracker = get_performance_tracker()
    if tracker:
        with tracker.track_operation(operation_name, metadata) as context:
            yield context
    else:
        yield None


__all__ = [
    "RealTimePerformanceTracker",
    "PerformanceContext",
    "PerformanceTarget",
    "ResourceMonitor",
    "get_performance_tracker",
    "initialize_performance_tracking", 
    "shutdown_performance_tracking",
    "track_operation"
]
"""
Centralized metrics collection system for real-time performance monitoring.

This module provides low-overhead, high-frequency metrics collection with
efficient storage and retrieval for performance analysis.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import numpy as np

from ..utils import get_logger


class MetricType(Enum):
    """Types of performance metrics."""
    COUNTER = "counter"
    GAUGE = "gauge" 
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricSample:
    """Individual metric sample."""
    name: str
    value: Union[float, int]
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class MetricStatistics:
    """Statistical summary of metric samples."""
    count: int
    sum: float
    min: float
    max: float
    mean: float
    std: float
    p50: float
    p95: float
    p99: float
    last_value: float
    rate_per_second: float = 0.0


class MetricsCollector:
    """
    High-performance metrics collection system.
    
    Provides low-latency metric collection with efficient storage
    and statistical analysis for real-time monitoring.
    """
    
    def __init__(self, max_samples_per_metric: int = 10000,
                 collection_interval_ms: int = 100):
        self.max_samples_per_metric = max_samples_per_metric
        self.collection_interval_ms = collection_interval_ms
        self.logger = get_logger(__name__)
        
        # Metric storage
        self.samples: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_samples_per_metric)
        )
        self.statistics: Dict[str, MetricStatistics] = {}
        
        # Metric metadata
        self.metric_types: Dict[str, MetricType] = {}
        self.metric_units: Dict[str, str] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background processing
        self._processing = False
        self._processor_thread = None
        
        # Performance tracking
        self.collection_overhead_ms = 0.0
        self.total_samples_collected = 0
        
        self.logger.info("Initialized MetricsCollector")
    
    def record_counter(self, name: str, value: Union[int, float] = 1,
                      tags: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric (cumulative value)."""
        self._record_sample(MetricSample(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            tags=tags or {}
        ))
    
    def record_gauge(self, name: str, value: Union[int, float],
                    unit: str = "", tags: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric (point-in-time value)."""
        self._record_sample(MetricSample(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            unit=unit,
            tags=tags or {}
        ))
    
    def record_timer(self, name: str, duration_ms: float,
                    tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timer metric (duration measurement)."""
        self._record_sample(MetricSample(
            name=name,
            value=duration_ms,
            metric_type=MetricType.TIMER,
            unit="ms",
            tags=tags or {}
        ))
    
    def record_histogram(self, name: str, value: Union[int, float],
                        unit: str = "", tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric (distribution of values)."""
        self._record_sample(MetricSample(
            name=name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            unit=unit,
            tags=tags or {}
        ))
    
    def record_quantum_computation(self, metrics: Dict[str, Any]) -> None:
        """Record quantum computation metrics."""
        tags = {"component": "quantum"}
        
        # Execution time
        if "execution_time_ms" in metrics:
            self.record_timer("quantum.execution_time", 
                            metrics["execution_time_ms"], tags)
        
        # Memory usage
        if "memory_usage_mb" in metrics:
            self.record_gauge("quantum.memory_usage", 
                            metrics["memory_usage_mb"], "MB", tags)
        
        # Result quality
        if "result_quality" in metrics:
            self.record_gauge("quantum.result_quality", 
                            metrics["result_quality"], "", tags)
        
        # Success/failure
        success = metrics.get("success", True)
        self.record_counter("quantum.operations", 1, 
                          {**tags, "status": "success" if success else "error"})
    
    def record_quantum_error(self, error_metrics: Dict[str, Any]) -> None:
        """Record quantum computation error metrics."""
        tags = {"component": "quantum", "status": "error"}
        
        # Error count
        self.record_counter("quantum.errors", 1, tags)
        
        # Execution time before error
        if "execution_time_ms" in error_metrics:
            self.record_timer("quantum.error_time", 
                            error_metrics["execution_time_ms"], tags)
        
        # Error type
        if "error_type" in error_metrics:
            error_tags = {**tags, "error_type": error_metrics["error_type"]}
            self.record_counter("quantum.error_types", 1, error_tags)
    
    def timer_context(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        return TimerContext(self, name, tags or {})
    
    def get_metric_statistics(self, name: str) -> Optional[MetricStatistics]:
        """Get statistical summary for a metric."""
        with self._lock:
            return self.statistics.get(name)
    
    def get_recent_samples(self, name: str, count: int = 100) -> List[MetricSample]:
        """Get recent samples for a metric."""
        with self._lock:
            if name not in self.samples:
                return []
            
            samples = list(self.samples[name])
            return samples[-count:] if len(samples) >= count else samples
    
    def get_samples_in_time_window(self, name: str, 
                                  time_window_seconds: int = 60) -> List[MetricSample]:
        """Get samples within a time window."""
        with self._lock:
            if name not in self.samples:
                return []
            
            current_time = time.time()
            cutoff_time = current_time - time_window_seconds
            
            return [
                sample for sample in self.samples[name]
                if sample.timestamp >= cutoff_time
            ]
    
    def get_metric_names(self) -> List[str]:
        """Get all tracked metric names."""
        with self._lock:
            return list(self.samples.keys())
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        with self._lock:
            summary = {
                "total_metrics": len(self.samples),
                "total_samples": sum(len(samples) for samples in self.samples.values()),
                "collection_overhead_ms": self.collection_overhead_ms,
                "samples_collected": self.total_samples_collected,
                "metrics": {}
            }
            
            for name, stats in self.statistics.items():
                summary["metrics"][name] = {
                    "type": self.metric_types.get(name, MetricType.GAUGE).value,
                    "unit": self.metric_units.get(name, ""),
                    "statistics": {
                        "count": stats.count,
                        "mean": stats.mean,
                        "min": stats.min,
                        "max": stats.max,
                        "p95": stats.p95,
                        "last_value": stats.last_value,
                        "rate_per_second": stats.rate_per_second
                    }
                }
            
            return summary
    
    def start_background_processing(self) -> None:
        """Start background processing for statistics computation."""
        if self._processing:
            return
        
        self._processing = True
        self._processor_thread = threading.Thread(
            target=self._background_processor, daemon=True
        )
        self._processor_thread.start()
        
        self.logger.info("Started background metrics processing")
    
    def stop_background_processing(self) -> None:
        """Stop background processing."""
        self._processing = False
        if self._processor_thread:
            self._processor_thread.join(timeout=5.0)
        
        self.logger.info("Stopped background metrics processing")
    
    def _record_sample(self, sample: MetricSample) -> None:
        """Record a metric sample with minimal overhead."""
        start_time = time.time()
        
        with self._lock:
            # Store sample
            self.samples[sample.name].append(sample)
            
            # Update metadata
            self.metric_types[sample.name] = sample.metric_type
            if sample.unit:
                self.metric_units[sample.name] = sample.unit
            
            # Track collection stats
            self.total_samples_collected += 1
        
        # Update collection overhead
        collection_time = (time.time() - start_time) * 1000
        if self.collection_overhead_ms == 0:
            self.collection_overhead_ms = collection_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.collection_overhead_ms = (
                alpha * collection_time + 
                (1 - alpha) * self.collection_overhead_ms
            )
    
    def _background_processor(self) -> None:
        """Background thread for computing statistics."""
        while self._processing:
            try:
                self._compute_statistics()
                time.sleep(self.collection_interval_ms / 1000.0)
            except Exception as e:
                self.logger.error(f"Error in background processor: {e}")
                time.sleep(1.0)
    
    def _compute_statistics(self) -> None:
        """Compute statistics for all metrics."""
        with self._lock:
            current_time = time.time()
            
            for name, samples in self.samples.items():
                if not samples:
                    continue
                
                # Convert to numpy array for efficient computation
                values = np.array([sample.value for sample in samples])
                
                if len(values) == 0:
                    continue
                
                # Basic statistics
                count = len(values)
                sum_val = np.sum(values)
                min_val = np.min(values)
                max_val = np.max(values)
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # Percentiles
                p50 = np.percentile(values, 50)
                p95 = np.percentile(values, 95)
                p99 = np.percentile(values, 99)
                
                # Rate calculation (for counters)
                rate_per_second = 0.0
                if (len(samples) >= 2 and 
                    self.metric_types.get(name) == MetricType.COUNTER):
                    
                    time_span = samples[-1].timestamp - samples[0].timestamp
                    if time_span > 0:
                        value_diff = samples[-1].value - samples[0].value
                        rate_per_second = value_diff / time_span
                
                # Update statistics
                self.statistics[name] = MetricStatistics(
                    count=count,
                    sum=float(sum_val),
                    min=float(min_val),
                    max=float(max_val),
                    mean=float(mean_val),
                    std=float(std_val),
                    p50=float(p50),
                    p95=float(p95),
                    p99=float(p99),
                    last_value=float(values[-1]),
                    rate_per_second=rate_per_second
                )
    
    def clear_metrics(self, metric_names: Optional[List[str]] = None) -> None:
        """Clear metrics data."""
        with self._lock:
            if metric_names is None:
                # Clear all metrics
                self.samples.clear()
                self.statistics.clear()
                self.metric_types.clear()
                self.metric_units.clear()
            else:
                # Clear specific metrics
                for name in metric_names:
                    if name in self.samples:
                        del self.samples[name]
                    if name in self.statistics:
                        del self.statistics[name]
                    if name in self.metric_types:
                        del self.metric_types[name]
                    if name in self.metric_units:
                        del self.metric_units[name]


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, name: str, tags: Dict[str, str]):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration_ms = (time.time() - self.start_time) * 1000
            self.collector.record_timer(self.name, duration_ms, self.tags)


__all__ = [
    "MetricType",
    "MetricSample",
    "MetricStatistics", 
    "MetricsCollector",
    "TimerContext"
]
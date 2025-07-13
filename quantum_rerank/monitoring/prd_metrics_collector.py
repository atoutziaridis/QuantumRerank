"""
PRD-aligned metrics collection system for QuantumRerank.

This module provides comprehensive metrics collection specifically aligned with
PRD performance targets and success criteria for production monitoring.
"""

import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from statistics import mean, median, stdev
import threading
from concurrent.futures import ThreadPoolExecutor

from ..utils.logging_config import get_logger
from ..config.manager import ConfigManager

logger = get_logger(__name__)


@dataclass
class PRDMetrics:
    """PRD performance target metrics aligned with specification."""
    
    # Performance targets from PRD
    similarity_computation_time_ms: float = 0.0  # Target: <100ms
    batch_processing_time_ms: float = 0.0        # Target: <500ms
    memory_usage_gb: float = 0.0                 # Target: <2GB for 100 docs
    accuracy_improvement_percent: float = 0.0    # Target: 10-20% over cosine
    
    # Quality metrics
    response_time_ms: float = 0.0                # Overall API response time
    error_rate: float = 0.0                      # Error rate percentage
    cache_hit_rate: float = 0.0                  # Cache effectiveness
    throughput_rps: float = 0.0                  # Requests per second
    
    # System metrics
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MetricWindow:
    """Sliding window for metric aggregation."""
    values: deque = field(default_factory=deque)
    max_size: int = 1000
    window_duration: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    
    def add_value(self, value: float, timestamp: datetime = None) -> None:
        """Add value to window with timestamp."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        self.values.append((value, timestamp))
        
        # Remove old values outside window
        cutoff_time = datetime.utcnow() - self.window_duration
        while self.values and self.values[0][1] < cutoff_time:
            self.values.popleft()
        
        # Limit size
        while len(self.values) > self.max_size:
            self.values.popleft()
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistical summary of values in window."""
        if not self.values:
            return {"count": 0}
        
        values = [v[0] for v in self.values]
        
        stats = {
            "count": len(values),
            "mean": mean(values),
            "median": median(values),
            "min": min(values),
            "max": max(values)
        }
        
        if len(values) > 1:
            stats["std_dev"] = stdev(values)
        else:
            stats["std_dev"] = 0.0
            
        return stats


class PRDMetricsCollector:
    """
    Comprehensive metrics collector aligned with PRD performance targets.
    
    Collects, aggregates, and analyzes metrics to ensure compliance with
    PRD specifications and enables performance optimization.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize PRD metrics collector.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.logger = logger
        
        # PRD performance targets
        self.prd_targets = {
            "similarity_computation_time_ms": 100,
            "batch_processing_time_ms": 500,
            "memory_usage_gb": 2.0,
            "accuracy_improvement_percent": 15.0,  # Mid-range target
            "response_time_ms": 200,
            "error_rate": 0.01,  # 1%
            "cache_hit_rate": 0.20,  # 20% minimum
            "throughput_rps": 100,  # Minimum expected
            "cpu_usage_percent": 80.0,
            "memory_usage_percent": 80.0
        }
        
        # Metric windows for different time scales
        self.metric_windows: Dict[str, Dict[str, MetricWindow]] = {
            "1m": {},   # 1 minute window
            "5m": {},   # 5 minute window
            "1h": {},   # 1 hour window
            "24h": {}   # 24 hour window
        }
        
        # Initialize windows for each metric
        for window_name in self.metric_windows:
            for metric_name in self.prd_targets:
                duration_map = {
                    "1m": timedelta(minutes=1),
                    "5m": timedelta(minutes=5),
                    "1h": timedelta(hours=1),
                    "24h": timedelta(hours=24)
                }
                
                self.metric_windows[window_name][metric_name] = MetricWindow(
                    window_duration=duration_map[window_name]
                )
        
        # Real-time metrics
        self.current_metrics = PRDMetrics()
        
        # Compliance tracking
        self.compliance_history: List[Dict[str, Any]] = []
        
        # Performance analytics
        self.performance_trends: Dict[str, List[float]] = defaultdict(list)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background collection
        self.collection_interval = 10  # seconds
        self.is_collecting = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Start background collection
        self.start_collection()
    
    def record_similarity_computation(self, duration_ms: float, method: str = "quantum") -> None:
        """
        Record similarity computation time.
        
        Args:
            duration_ms: Computation duration in milliseconds
            method: Computation method used
        """
        with self.lock:
            self.current_metrics.similarity_computation_time_ms = duration_ms
            self._add_to_windows("similarity_computation_time_ms", duration_ms)
            
            # Check PRD compliance
            target = self.prd_targets["similarity_computation_time_ms"]
            if duration_ms > target:
                self.logger.warning(
                    f"Similarity computation time exceeded PRD target: {duration_ms:.2f}ms > {target}ms",
                    extra={"method": method, "duration_ms": duration_ms, "target_ms": target}
                )
    
    def record_batch_processing(self, duration_ms: float, batch_size: int) -> None:
        """
        Record batch processing time.
        
        Args:
            duration_ms: Processing duration in milliseconds
            batch_size: Number of items in batch
        """
        with self.lock:
            self.current_metrics.batch_processing_time_ms = duration_ms
            self._add_to_windows("batch_processing_time_ms", duration_ms)
            
            # Check PRD compliance
            target = self.prd_targets["batch_processing_time_ms"]
            if duration_ms > target:
                self.logger.warning(
                    f"Batch processing time exceeded PRD target: {duration_ms:.2f}ms > {target}ms",
                    extra={"duration_ms": duration_ms, "batch_size": batch_size, "target_ms": target}
                )
    
    def record_memory_usage(self, usage_gb: float, context: str = "general") -> None:
        """
        Record memory usage.
        
        Args:
            usage_gb: Memory usage in gigabytes
            context: Context of memory usage
        """
        with self.lock:
            self.current_metrics.memory_usage_gb = usage_gb
            self._add_to_windows("memory_usage_gb", usage_gb)
            
            # Check PRD compliance
            target = self.prd_targets["memory_usage_gb"]
            if usage_gb > target:
                self.logger.warning(
                    f"Memory usage exceeded PRD target: {usage_gb:.2f}GB > {target}GB",
                    extra={"usage_gb": usage_gb, "context": context, "target_gb": target}
                )
    
    def record_accuracy_improvement(self, improvement_percent: float, baseline: str = "cosine") -> None:
        """
        Record accuracy improvement over baseline.
        
        Args:
            improvement_percent: Improvement percentage
            baseline: Baseline method for comparison
        """
        with self.lock:
            self.current_metrics.accuracy_improvement_percent = improvement_percent
            self._add_to_windows("accuracy_improvement_percent", improvement_percent)
            
            # Check PRD compliance
            target = self.prd_targets["accuracy_improvement_percent"]
            if improvement_percent < target * 0.5:  # Below 50% of target
                self.logger.warning(
                    f"Accuracy improvement below expected: {improvement_percent:.2f}% < {target}%",
                    extra={"improvement_percent": improvement_percent, "baseline": baseline, "target": target}
                )
    
    def record_api_response(self, duration_ms: float, status_code: int, endpoint: str) -> None:
        """
        Record API response metrics.
        
        Args:
            duration_ms: Response duration in milliseconds
            status_code: HTTP status code
            endpoint: API endpoint
        """
        with self.lock:
            self.current_metrics.response_time_ms = duration_ms
            self._add_to_windows("response_time_ms", duration_ms)
            
            # Record error if applicable
            if status_code >= 400:
                current_errors = getattr(self, '_error_count', 0) + 1
                setattr(self, '_error_count', current_errors)
                
                total_requests = getattr(self, '_total_requests', 0) + 1
                setattr(self, '_total_requests', total_requests)
                
                error_rate = current_errors / total_requests
                self.current_metrics.error_rate = error_rate
                self._add_to_windows("error_rate", error_rate)
            else:
                total_requests = getattr(self, '_total_requests', 0) + 1
                setattr(self, '_total_requests', total_requests)
                
                error_count = getattr(self, '_error_count', 0)
                error_rate = error_count / total_requests if total_requests > 0 else 0
                self.current_metrics.error_rate = error_rate
    
    def record_cache_metrics(self, hit: bool, cache_type: str = "general") -> None:
        """
        Record cache hit/miss metrics.
        
        Args:
            hit: Whether cache hit occurred
            cache_type: Type of cache
        """
        with self.lock:
            cache_hits = getattr(self, '_cache_hits', 0)
            cache_total = getattr(self, '_cache_total', 0)
            
            if hit:
                cache_hits += 1
            cache_total += 1
            
            setattr(self, '_cache_hits', cache_hits)
            setattr(self, '_cache_total', cache_total)
            
            hit_rate = cache_hits / cache_total if cache_total > 0 else 0
            self.current_metrics.cache_hit_rate = hit_rate
            self._add_to_windows("cache_hit_rate", hit_rate)
    
    def record_system_metrics(self, cpu_percent: float, memory_percent: float, disk_percent: float) -> None:
        """
        Record system resource metrics.
        
        Args:
            cpu_percent: CPU usage percentage
            memory_percent: Memory usage percentage
            disk_percent: Disk usage percentage
        """
        with self.lock:
            self.current_metrics.cpu_usage_percent = cpu_percent
            self.current_metrics.memory_usage_percent = memory_percent
            self.current_metrics.disk_usage_percent = disk_percent
            
            self._add_to_windows("cpu_usage_percent", cpu_percent)
            self._add_to_windows("memory_usage_percent", memory_percent)
    
    def get_current_metrics(self) -> PRDMetrics:
        """Get current metrics snapshot."""
        with self.lock:
            return PRDMetrics(
                similarity_computation_time_ms=self.current_metrics.similarity_computation_time_ms,
                batch_processing_time_ms=self.current_metrics.batch_processing_time_ms,
                memory_usage_gb=self.current_metrics.memory_usage_gb,
                accuracy_improvement_percent=self.current_metrics.accuracy_improvement_percent,
                response_time_ms=self.current_metrics.response_time_ms,
                error_rate=self.current_metrics.error_rate,
                cache_hit_rate=self.current_metrics.cache_hit_rate,
                throughput_rps=self.current_metrics.throughput_rps,
                cpu_usage_percent=self.current_metrics.cpu_usage_percent,
                memory_usage_percent=self.current_metrics.memory_usage_percent,
                disk_usage_percent=self.current_metrics.disk_usage_percent
            )
    
    def get_prd_compliance_report(self) -> Dict[str, Any]:
        """
        Generate PRD compliance report.
        
        Returns:
            Compliance report with violations and recommendations
        """
        with self.lock:
            current = self.get_current_metrics()
            
            compliance = {
                "overall_compliant": True,
                "timestamp": datetime.utcnow().isoformat(),
                "targets": self.prd_targets.copy(),
                "current_values": {},
                "violations": [],
                "recommendations": []
            }
            
            # Check each metric against PRD targets
            metrics_to_check = [
                ("similarity_computation_time_ms", current.similarity_computation_time_ms, "lower"),
                ("batch_processing_time_ms", current.batch_processing_time_ms, "lower"),
                ("memory_usage_gb", current.memory_usage_gb, "lower"),
                ("accuracy_improvement_percent", current.accuracy_improvement_percent, "higher"),
                ("response_time_ms", current.response_time_ms, "lower"),
                ("error_rate", current.error_rate, "lower"),
                ("cache_hit_rate", current.cache_hit_rate, "higher"),
                ("cpu_usage_percent", current.cpu_usage_percent, "lower"),
                ("memory_usage_percent", current.memory_usage_percent, "lower")
            ]
            
            for metric_name, current_value, comparison in metrics_to_check:
                target = self.prd_targets[metric_name]
                compliance["current_values"][metric_name] = current_value
                
                # Check compliance based on comparison type
                is_compliant = True
                if comparison == "lower" and current_value > target:
                    is_compliant = False
                elif comparison == "higher" and current_value < target:
                    is_compliant = False
                
                if not is_compliant:
                    compliance["overall_compliant"] = False
                    violation = {
                        "metric": metric_name,
                        "current": current_value,
                        "target": target,
                        "deviation_percent": abs((current_value - target) / target * 100),
                        "comparison": comparison
                    }
                    compliance["violations"].append(violation)
                    
                    # Add recommendations
                    recommendations = self._get_metric_recommendations(metric_name, current_value, target)
                    compliance["recommendations"].extend(recommendations)
            
            # Store compliance history
            self.compliance_history.append(compliance)
            
            # Keep only recent history
            if len(self.compliance_history) > 1000:
                self.compliance_history = self.compliance_history[-1000:]
            
            return compliance
    
    def get_metric_statistics(self, metric_name: str, window: str = "5m") -> Dict[str, Any]:
        """
        Get statistical analysis for a metric.
        
        Args:
            metric_name: Name of metric
            window: Time window (1m, 5m, 1h, 24h)
            
        Returns:
            Statistical summary
        """
        with self.lock:
            if window not in self.metric_windows or metric_name not in self.metric_windows[window]:
                return {"error": f"No data for {metric_name} in {window} window"}
            
            window_obj = self.metric_windows[window][metric_name]
            stats = window_obj.get_statistics()
            
            # Add PRD compliance info
            target = self.prd_targets.get(metric_name)
            if target and stats.get("count", 0) > 0:
                stats["prd_target"] = target
                stats["target_compliance_percent"] = min(100, max(0, (1 - abs(stats["mean"] - target) / target) * 100))
            
            return {
                "metric": metric_name,
                "window": window,
                "statistics": stats,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_performance_trends(self, hours: int = 1) -> Dict[str, Any]:
        """
        Get performance trends over time.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Trend analysis
        """
        with self.lock:
            trends = {
                "analysis_period_hours": hours,
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": {}
            }
            
            # Analyze trends for key metrics
            key_metrics = [
                "similarity_computation_time_ms",
                "batch_processing_time_ms", 
                "memory_usage_gb",
                "response_time_ms",
                "error_rate"
            ]
            
            for metric_name in key_metrics:
                # Use appropriate window based on hours requested
                window = "1h" if hours <= 1 else "24h"
                stats = self.get_metric_statistics(metric_name, window)
                
                if "error" not in stats:
                    trends["metrics"][metric_name] = {
                        "current_stats": stats["statistics"],
                        "prd_target": self.prd_targets.get(metric_name),
                        "compliance": stats["statistics"].get("target_compliance_percent", 0)
                    }
            
            return trends
    
    def _add_to_windows(self, metric_name: str, value: float) -> None:
        """Add value to all time windows for a metric."""
        timestamp = datetime.utcnow()
        for window_name in self.metric_windows:
            if metric_name in self.metric_windows[window_name]:
                self.metric_windows[window_name][metric_name].add_value(value, timestamp)
    
    def _get_metric_recommendations(self, metric_name: str, current: float, target: float) -> List[str]:
        """Get recommendations for improving a metric."""
        recommendations = []
        
        if metric_name == "similarity_computation_time_ms":
            if current > target:
                recommendations.append("Consider optimizing quantum circuit depth")
                recommendations.append("Implement better caching for repeated computations")
                recommendations.append("Profile quantum computation bottlenecks")
        
        elif metric_name == "batch_processing_time_ms":
            if current > target:
                recommendations.append("Optimize batch size for better throughput")
                recommendations.append("Implement parallel processing for batch items")
                recommendations.append("Review memory usage patterns in batch processing")
        
        elif metric_name == "memory_usage_gb":
            if current > target:
                recommendations.append("Implement more aggressive garbage collection")
                recommendations.append("Optimize embedding storage and caching")
                recommendations.append("Review quantum state vector memory usage")
        
        elif metric_name == "accuracy_improvement_percent":
            if current < target:
                recommendations.append("Fine-tune quantum circuit parameters")
                recommendations.append("Improve training data quality")
                recommendations.append("Experiment with different quantum similarity methods")
        
        return recommendations
    
    def start_collection(self) -> None:
        """Start background metrics collection."""
        if not self.is_collecting:
            self.is_collecting = True
            self.executor.submit(self._background_collection)
    
    def stop_collection(self) -> None:
        """Stop background metrics collection."""
        self.is_collecting = False
    
    def _background_collection(self) -> None:
        """Background thread for automatic metrics collection."""
        while self.is_collecting:
            try:
                # Collect system metrics
                import psutil
                
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                
                self.record_system_metrics(cpu_percent, memory_percent, disk_percent)
                
                # Sleep until next collection
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Background metrics collection error: {str(e)}")
                time.sleep(self.collection_interval)


# Global metrics collector instance
_metrics_collector: Optional[PRDMetricsCollector] = None


def get_prd_metrics_collector(config_manager: ConfigManager) -> PRDMetricsCollector:
    """
    Get or create global PRD metrics collector.
    
    Args:
        config_manager: Configuration manager
        
    Returns:
        PRDMetricsCollector instance
    """
    global _metrics_collector
    
    if _metrics_collector is None:
        _metrics_collector = PRDMetricsCollector(config_manager)
    
    return _metrics_collector
"""
Performance Metrics Collection for QuantumRerank Benchmarking.

Implements specialized metric collection, tracking, and analysis tools
for comprehensive performance monitoring and PRD validation.
"""

import time
import psutil
import threading
import gc
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Generator, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np

import logging
logger = logging.getLogger(__name__)


@dataclass
class LatencyMeasurement:
    """Single latency measurement record."""
    operation: str
    duration_ms: float
    timestamp: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryMeasurement:
    """Single memory measurement record."""
    operation: str
    memory_mb: float
    timestamp: float
    peak_memory_mb: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThroughputMeasurement:
    """Single throughput measurement record."""
    operation: str
    operations_per_second: float
    total_operations: int
    duration_s: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class LatencyTracker:
    """
    High-precision latency tracking for quantum operations.
    
    Supports nested operation tracking and statistical analysis.
    """
    
    def __init__(self, max_history: int = 10000):
        """
        Initialize latency tracker.
        
        Args:
            max_history: Maximum number of measurements to keep in memory
        """
        self.max_history = max_history
        self.measurements: deque = deque(maxlen=max_history)
        self.active_operations: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    @contextmanager
    def track_operation(self, operation_name: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> Generator[None, None, None]:
        """
        Context manager for tracking operation latency.
        
        Args:
            operation_name: Name of the operation being tracked
            metadata: Additional metadata to store with measurement
            
        Usage:
            with tracker.track_operation("similarity_computation"):
                result = compute_similarity()
        """
        start_time = time.perf_counter()
        success = True
        
        try:
            yield
        except Exception as e:
            success = False
            raise
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            measurement = LatencyMeasurement(
                operation=operation_name,
                duration_ms=duration_ms,
                timestamp=time.time(),
                success=success,
                metadata=metadata or {}
            )
            
            with self._lock:
                self.measurements.append(measurement)
    
    def start_operation(self, operation_name: str) -> str:
        """
        Start tracking an operation manually.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Operation ID for stopping the measurement
        """
        operation_id = f"{operation_name}_{time.time()}"
        with self._lock:
            self.active_operations[operation_id] = time.perf_counter()
        return operation_id
    
    def stop_operation(self, operation_id: str, 
                      metadata: Optional[Dict[str, Any]] = None,
                      success: bool = True) -> float:
        """
        Stop tracking an operation manually.
        
        Args:
            operation_id: Operation ID from start_operation
            metadata: Additional metadata
            success: Whether operation succeeded
            
        Returns:
            Duration in milliseconds
        """
        end_time = time.perf_counter()
        
        with self._lock:
            if operation_id not in self.active_operations:
                logger.warning(f"Operation {operation_id} not found in active operations")
                return 0.0
            
            start_time = self.active_operations.pop(operation_id)
            duration_ms = (end_time - start_time) * 1000
            
            # Extract operation name from ID
            operation_name = operation_id.rsplit('_', 1)[0]
            
            measurement = LatencyMeasurement(
                operation=operation_name,
                duration_ms=duration_ms,
                timestamp=time.time(),
                success=success,
                metadata=metadata or {}
            )
            
            self.measurements.append(measurement)
            return duration_ms
    
    def get_statistics(self, operation_name: Optional[str] = None,
                      time_window_s: Optional[float] = None) -> Dict[str, Any]:
        """
        Get statistical analysis of latency measurements.
        
        Args:
            operation_name: Filter by specific operation (None for all)
            time_window_s: Only include measurements from last N seconds
            
        Returns:
            Dictionary with statistical metrics
        """
        with self._lock:
            measurements = list(self.measurements)
        
        # Filter by operation name
        if operation_name:
            measurements = [m for m in measurements if m.operation == operation_name]
        
        # Filter by time window
        if time_window_s:
            cutoff_time = time.time() - time_window_s
            measurements = [m for m in measurements if m.timestamp >= cutoff_time]
        
        if not measurements:
            return {"count": 0, "error": "No measurements found"}
        
        # Calculate statistics
        durations = [m.duration_ms for m in measurements]
        success_count = sum(1 for m in measurements if m.success)
        
        stats = {
            "count": len(measurements),
            "success_rate": success_count / len(measurements),
            "mean_ms": np.mean(durations),
            "median_ms": np.median(durations),
            "std_ms": np.std(durations),
            "min_ms": np.min(durations),
            "max_ms": np.max(durations),
            "p95_ms": np.percentile(durations, 95),
            "p99_ms": np.percentile(durations, 99)
        }
        
        # PRD compliance checks
        if operation_name and "similarity" in operation_name:
            stats["prd_compliant"] = stats["mean_ms"] < 100.0
        elif operation_name and "batch" in operation_name:
            stats["prd_compliant"] = stats["mean_ms"] < 500.0
        
        return stats


class MemoryTracker:
    """
    Memory usage tracking for quantum operations.
    
    Monitors peak memory usage and memory deltas during operations.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize memory tracker.
        
        Args:
            max_history: Maximum number of measurements to keep
        """
        self.max_history = max_history
        self.measurements: deque = deque(maxlen=max_history)
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread = None
    
    @contextmanager
    def track_memory(self, operation_name: str,
                    metadata: Optional[Dict[str, Any]] = None) -> Generator[None, None, None]:
        """
        Context manager for tracking memory usage during operation.
        
        Args:
            operation_name: Name of the operation being tracked
            metadata: Additional metadata to store
        """
        # Force garbage collection before measurement
        gc.collect()
        
        # Get baseline memory
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = baseline_memory
        
        # Start peak memory monitoring
        monitoring = True
        
        def monitor_peak():
            nonlocal peak_memory
            while monitoring:
                try:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    peak_memory = max(peak_memory, current_memory)
                    time.sleep(0.01)  # 10ms sampling
                except:
                    break
        
        monitor_thread = threading.Thread(target=monitor_peak, daemon=True)
        monitor_thread.start()
        
        try:
            yield
        finally:
            monitoring = False
            monitor_thread.join(timeout=0.1)
            
            # Final memory measurement
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_delta = final_memory - baseline_memory
            
            measurement = MemoryMeasurement(
                operation=operation_name,
                memory_mb=memory_delta,
                timestamp=time.time(),
                peak_memory_mb=peak_memory - baseline_memory,
                metadata=metadata or {}
            )
            
            with self._lock:
                self.measurements.append(measurement)
    
    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return psutil.Process().memory_info().rss / 1024 / 1024
    
    def get_statistics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Args:
            operation_name: Filter by specific operation
            
        Returns:
            Dictionary with memory statistics
        """
        with self._lock:
            measurements = list(self.measurements)
        
        if operation_name:
            measurements = [m for m in measurements if m.operation == operation_name]
        
        if not measurements:
            return {"count": 0, "error": "No measurements found"}
        
        memory_deltas = [m.memory_mb for m in measurements]
        peak_memories = [m.peak_memory_mb for m in measurements if m.peak_memory_mb is not None]
        
        stats = {
            "count": len(measurements),
            "mean_delta_mb": np.mean(memory_deltas),
            "max_delta_mb": np.max(memory_deltas),
            "min_delta_mb": np.min(memory_deltas),
            "total_delta_mb": np.sum(memory_deltas)
        }
        
        if peak_memories:
            stats.update({
                "mean_peak_mb": np.mean(peak_memories),
                "max_peak_mb": np.max(peak_memories)
            })
        
        # PRD compliance check (100 docs should use <2GB)
        if operation_name and "100" in operation_name:
            stats["prd_compliant"] = stats["mean_delta_mb"] < 2048
        
        return stats


class ThroughputTracker:
    """
    Throughput measurement for batch operations.
    
    Tracks operations per second for various quantum and classical operations.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize throughput tracker.
        
        Args:
            max_history: Maximum measurements to keep
        """
        self.max_history = max_history
        self.measurements: deque = deque(maxlen=max_history)
        self._lock = threading.Lock()
    
    def measure_throughput(self, operation_name: str,
                          operation_func: Callable,
                          operation_data: Any,
                          target_duration_s: float = 10.0,
                          metadata: Optional[Dict[str, Any]] = None) -> float:
        """
        Measure throughput by running operation repeatedly.
        
        Args:
            operation_name: Name of the operation
            operation_func: Function to benchmark
            operation_data: Data to pass to function
            target_duration_s: How long to run benchmark
            metadata: Additional metadata
            
        Returns:
            Operations per second
        """
        start_time = time.perf_counter()
        operations_count = 0
        
        while (time.perf_counter() - start_time) < target_duration_s:
            try:
                operation_func(operation_data)
                operations_count += 1
            except Exception as e:
                logger.warning(f"Operation {operation_name} failed: {e}")
                break
        
        end_time = time.perf_counter()
        actual_duration = end_time - start_time
        
        if actual_duration > 0:
            ops_per_second = operations_count / actual_duration
        else:
            ops_per_second = 0.0
        
        measurement = ThroughputMeasurement(
            operation=operation_name,
            operations_per_second=ops_per_second,
            total_operations=operations_count,
            duration_s=actual_duration,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self.measurements.append(measurement)
        
        return ops_per_second
    
    def get_statistics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get throughput statistics.
        
        Args:
            operation_name: Filter by operation name
            
        Returns:
            Dictionary with throughput statistics
        """
        with self._lock:
            measurements = list(self.measurements)
        
        if operation_name:
            measurements = [m for m in measurements if m.operation == operation_name]
        
        if not measurements:
            return {"count": 0, "error": "No measurements found"}
        
        throughputs = [m.operations_per_second for m in measurements]
        
        return {
            "count": len(measurements),
            "mean_ops_per_second": np.mean(throughputs),
            "max_ops_per_second": np.max(throughputs),
            "min_ops_per_second": np.min(throughputs),
            "total_operations": sum(m.total_operations for m in measurements)
        }


class BenchmarkMetrics:
    """
    Comprehensive metrics collection system combining all trackers.
    
    Provides unified interface for all performance monitoring needs.
    """
    
    def __init__(self):
        """Initialize comprehensive metrics system."""
        self.latency = LatencyTracker()
        self.memory = MemoryTracker()
        self.throughput = ThroughputTracker()
        
        logger.info("Initialized comprehensive benchmark metrics system")
    
    @contextmanager
    def track_complete_operation(self, operation_name: str,
                                metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager that tracks latency and memory for an operation.
        
        Args:
            operation_name: Name of the operation
            metadata: Additional metadata
        """
        with self.memory.track_memory(operation_name, metadata):
            with self.latency.track_operation(operation_name, metadata):
                yield
    
    def get_comprehensive_report(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            operation_name: Filter by operation name
            
        Returns:
            Complete performance report
        """
        return {
            "latency_stats": self.latency.get_statistics(operation_name),
            "memory_stats": self.memory.get_statistics(operation_name),
            "throughput_stats": self.throughput.get_statistics(operation_name),
            "current_memory_mb": self.memory.get_current_memory_mb(),
            "report_timestamp": time.time()
        }
    
    def check_prd_compliance(self) -> Dict[str, bool]:
        """
        Check PRD compliance across all metrics.
        
        Returns:
            Dictionary with compliance status
        """
        # Get stats for key operations
        similarity_stats = self.latency.get_statistics("similarity")
        batch_stats = self.latency.get_statistics("batch")
        memory_stats = self.memory.get_statistics()
        
        compliance = {
            "similarity_under_100ms": False,
            "batch_under_500ms": False,
            "memory_under_2gb": False,
            "overall_compliant": False
        }
        
        # Check similarity latency
        if similarity_stats.get("count", 0) > 0:
            compliance["similarity_under_100ms"] = similarity_stats.get("mean_ms", float('inf')) < 100.0
        
        # Check batch processing latency
        if batch_stats.get("count", 0) > 0:
            compliance["batch_under_500ms"] = batch_stats.get("mean_ms", float('inf')) < 500.0
        
        # Check memory usage
        if memory_stats.get("count", 0) > 0:
            compliance["memory_under_2gb"] = memory_stats.get("max_delta_mb", float('inf')) < 2048
        
        # Overall compliance
        compliance["overall_compliant"] = all([
            compliance["similarity_under_100ms"],
            compliance["batch_under_500ms"],
            compliance["memory_under_2gb"]
        ])
        
        return compliance
    
    def reset_all_measurements(self):
        """Clear all measurement history."""
        self.latency.measurements.clear()
        self.memory.measurements.clear() 
        self.throughput.measurements.clear()
        logger.info("All measurement history cleared")
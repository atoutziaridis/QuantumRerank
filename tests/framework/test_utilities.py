"""
Test utilities and helper functions for comprehensive testing framework.

This module provides timing utilities, metrics collection, data validation,
and environment management for test execution.
"""

import os
import time
import psutil
import threading
import contextlib
from typing import Dict, List, Optional, Any, Union, Callable, Iterator
from dataclasses import dataclass, field
from collections import defaultdict
import json
import tempfile
import shutil

from quantum_rerank.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TimingResult:
    """Result of timing measurement."""
    operation_name: str
    execution_time_ms: float
    start_time: float
    end_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceMetrics:
    """System resource utilization metrics."""
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_sent_mb: float
    network_io_recv_mb: float
    timestamp: float = field(default_factory=time.time)


class TestTimer:
    """
    High-precision timing utilities for test performance measurement.
    """
    
    def __init__(self):
        self.timings: Dict[str, List[TimingResult]] = defaultdict(list)
        self.active_timers: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    @contextlib.contextmanager
    def time_operation(self, operation_name: str, 
                      metadata: Optional[Dict[str, Any]] = None) -> Iterator[TimingResult]:
        """
        Context manager for timing operations.
        
        Args:
            operation_name: Name of operation being timed
            metadata: Additional metadata to store
            
        Yields:
            TimingResult object that will be populated
        """
        start_time = time.perf_counter()
        
        # Create result object
        result = TimingResult(
            operation_name=operation_name,
            execution_time_ms=0,
            start_time=start_time,
            end_time=0,
            metadata=metadata or {}
        )
        
        try:
            yield result
        finally:
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            
            # Update result
            result.end_time = end_time
            result.execution_time_ms = execution_time_ms
            
            # Store timing
            with self._lock:
                self.timings[operation_name].append(result)
    
    def start_timer(self, operation_name: str) -> None:
        """Start a named timer."""
        with self._lock:
            self.active_timers[operation_name] = time.perf_counter()
    
    def stop_timer(self, operation_name: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> TimingResult:
        """
        Stop a named timer and return result.
        
        Args:
            operation_name: Name of timer to stop
            metadata: Additional metadata
            
        Returns:
            Timing result
        """
        end_time = time.perf_counter()
        
        with self._lock:
            start_time = self.active_timers.pop(operation_name, end_time)
            execution_time_ms = (end_time - start_time) * 1000
            
            result = TimingResult(
                operation_name=operation_name,
                execution_time_ms=execution_time_ms,
                start_time=start_time,
                end_time=end_time,
                metadata=metadata or {}
            )
            
            self.timings[operation_name].append(result)
            return result
    
    def get_timing_statistics(self, operation_name: str) -> Optional[Dict[str, float]]:
        """
        Get statistical summary of timings for an operation.
        
        Args:
            operation_name: Name of operation
            
        Returns:
            Statistics dictionary or None if no timings
        """
        with self._lock:
            timings = self.timings.get(operation_name, [])
        
        if not timings:
            return None
        
        execution_times = [t.execution_time_ms for t in timings]
        
        return {
            "count": len(execution_times),
            "min_ms": min(execution_times),
            "max_ms": max(execution_times),
            "avg_ms": sum(execution_times) / len(execution_times),
            "median_ms": sorted(execution_times)[len(execution_times) // 2],
            "total_ms": sum(execution_times)
        }
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all timed operations."""
        stats = {}
        
        with self._lock:
            for operation_name in self.timings.keys():
                stats[operation_name] = self.get_timing_statistics(operation_name)
        
        return stats
    
    def reset_timings(self, operation_name: Optional[str] = None) -> None:
        """Reset timing data for operation or all operations."""
        with self._lock:
            if operation_name:
                self.timings.pop(operation_name, None)
            else:
                self.timings.clear()
                self.active_timers.clear()


class TestMetricsCollector:
    """
    Collects comprehensive metrics during test execution.
    """
    
    def __init__(self, collection_interval_s: float = 1.0):
        self.collection_interval_s = collection_interval_s
        self.metrics_history: List[ResourceMetrics] = []
        self.custom_metrics: Dict[str, List[float]] = defaultdict(list)
        self._collecting = False
        self._collection_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Initialize baseline metrics
        self.baseline_metrics = self._collect_current_metrics()
    
    def start_collection(self) -> None:
        """Start continuous metrics collection."""
        if self._collecting:
            return
        
        self._collecting = True
        self._collection_thread = threading.Thread(target=self._collect_metrics_loop, daemon=True)
        self._collection_thread.start()
        logger.info("Started metrics collection")
    
    def stop_collection(self) -> None:
        """Stop metrics collection."""
        if not self._collecting:
            return
        
        self._collecting = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
        
        logger.info("Stopped metrics collection")
    
    @contextlib.contextmanager
    def collect_metrics(self) -> Iterator['TestMetricsCollector']:
        """Context manager for automatic metrics collection."""
        self.start_collection()
        try:
            yield self
        finally:
            self.stop_collection()
    
    def record_custom_metric(self, name: str, value: float) -> None:
        """Record a custom metric value."""
        with self._lock:
            self.custom_metrics[name].append(value)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        with self._lock:
            if not self.metrics_history:
                return {}
            
            # Calculate resource usage statistics
            cpu_values = [m.cpu_percent for m in self.metrics_history]
            memory_mb_values = [m.memory_mb for m in self.metrics_history]
            memory_percent_values = [m.memory_percent for m in self.metrics_history]
            
            resource_summary = {
                "cpu_percent": {
                    "min": min(cpu_values),
                    "max": max(cpu_values),
                    "avg": sum(cpu_values) / len(cpu_values),
                    "baseline": self.baseline_metrics.cpu_percent
                },
                "memory_mb": {
                    "min": min(memory_mb_values),
                    "max": max(memory_mb_values),
                    "avg": sum(memory_mb_values) / len(memory_mb_values),
                    "baseline": self.baseline_metrics.memory_mb
                },
                "memory_percent": {
                    "min": min(memory_percent_values),
                    "max": max(memory_percent_values),
                    "avg": sum(memory_percent_values) / len(memory_percent_values),
                    "baseline": self.baseline_metrics.memory_percent
                }
            }
            
            # Custom metrics summary
            custom_summary = {}
            for name, values in self.custom_metrics.items():
                if values:
                    custom_summary[name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values)
                    }
            
            return {
                "collection_duration_s": len(self.metrics_history) * self.collection_interval_s,
                "sample_count": len(self.metrics_history),
                "resource_usage": resource_summary,
                "custom_metrics": custom_summary
            }
    
    def _collect_metrics_loop(self) -> None:
        """Main metrics collection loop."""
        while self._collecting:
            try:
                metrics = self._collect_current_metrics()
                
                with self._lock:
                    self.metrics_history.append(metrics)
                
                time.sleep(self.collection_interval_s)
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                time.sleep(self.collection_interval_s)
    
    def _collect_current_metrics(self) -> ResourceMetrics:
        """Collect current system metrics."""
        # Get process info
        process = psutil.Process()
        
        # CPU and memory usage
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)  # RSS in MB
        memory_percent = process.memory_percent()
        
        # I/O statistics
        try:
            io_stats = process.io_counters()
            disk_io_read_mb = io_stats.read_bytes / (1024 * 1024)
            disk_io_write_mb = io_stats.write_bytes / (1024 * 1024)
        except (AttributeError, psutil.AccessDenied):
            disk_io_read_mb = 0
            disk_io_write_mb = 0
        
        # Network statistics (system-wide)
        try:
            net_io = psutil.net_io_counters()
            network_io_sent_mb = net_io.bytes_sent / (1024 * 1024)
            network_io_recv_mb = net_io.bytes_recv / (1024 * 1024)
        except (AttributeError, psutil.AccessDenied):
            network_io_sent_mb = 0
            network_io_recv_mb = 0
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            disk_io_read_mb=disk_io_read_mb,
            disk_io_write_mb=disk_io_write_mb,
            network_io_sent_mb=network_io_sent_mb,
            network_io_recv_mb=network_io_recv_mb
        )


class TestDataValidator:
    """
    Validates test data for correctness and consistency.
    """
    
    def __init__(self):
        self.validation_rules: Dict[str, List[Callable]] = defaultdict(list)
        self.validation_results: List[Dict[str, Any]] = []
    
    def add_validation_rule(self, data_type: str, validator: Callable[[Any], bool],
                          description: str) -> None:
        """
        Add a validation rule for a data type.
        
        Args:
            data_type: Type of data to validate
            validator: Validation function
            description: Description of validation rule
        """
        # Wrap validator with description
        def wrapped_validator(data):
            try:
                result = validator(data)
                return {"passed": result, "description": description, "error": None}
            except Exception as e:
                return {"passed": False, "description": description, "error": str(e)}
        
        self.validation_rules[data_type].append(wrapped_validator)
    
    def validate_data(self, data_type: str, data: Any) -> Dict[str, Any]:
        """
        Validate data against registered rules.
        
        Args:
            data_type: Type of data being validated
            data: Data to validate
            
        Returns:
            Validation results
        """
        if data_type not in self.validation_rules:
            return {"passed": True, "message": "No validation rules defined"}
        
        results = []
        all_passed = True
        
        for validator in self.validation_rules[data_type]:
            result = validator(data)
            results.append(result)
            if not result["passed"]:
                all_passed = False
        
        validation_result = {
            "data_type": data_type,
            "passed": all_passed,
            "rule_results": results,
            "timestamp": time.time()
        }
        
        self.validation_results.append(validation_result)
        return validation_result
    
    def validate_test_embeddings(self, embeddings: Any) -> Dict[str, Any]:
        """Validate test embedding data."""
        # Standard embedding validation rules
        self.add_validation_rule("embeddings", 
                               lambda x: hasattr(x, 'shape') and len(x.shape) >= 1,
                               "Embeddings should have at least 1 dimension")
        
        self.add_validation_rule("embeddings",
                               lambda x: x.dtype.kind == 'f',  # Float type
                               "Embeddings should be float type")
        
        self.add_validation_rule("embeddings",
                               lambda x: not (x != x).any(),  # Check for NaN
                               "Embeddings should not contain NaN values")
        
        return self.validate_data("embeddings", embeddings)
    
    def validate_quantum_parameters(self, params: Any) -> Dict[str, Any]:
        """Validate quantum parameter data."""
        import numpy as np
        
        self.add_validation_rule("quantum_params",
                               lambda x: isinstance(x, (list, tuple, np.ndarray)),
                               "Parameters should be array-like")
        
        self.add_validation_rule("quantum_params",
                               lambda x: all(abs(p) <= np.pi for p in np.array(x).flatten()),
                               "Parameters should be within [-π, π] range")
        
        return self.validate_data("quantum_params", params)


class TestEnvironmentManager:
    """
    Manages test environment setup and cleanup.
    """
    
    def __init__(self, base_temp_dir: Optional[str] = None):
        self.base_temp_dir = base_temp_dir or tempfile.gettempdir()
        self.temp_directories: List[str] = []
        self.environment_variables: Dict[str, str] = {}
        self.cleanup_functions: List[Callable] = []
        self._original_env: Dict[str, str] = {}
    
    @contextlib.contextmanager
    def temporary_directory(self, prefix: str = "quantumrerank_test_") -> Iterator[str]:
        """
        Create a temporary directory for test use.
        
        Args:
            prefix: Prefix for directory name
            
        Yields:
            Path to temporary directory
        """
        temp_dir = tempfile.mkdtemp(prefix=prefix, dir=self.base_temp_dir)
        self.temp_directories.append(temp_dir)
        
        try:
            yield temp_dir
        finally:
            # Cleanup handled by cleanup_all()
            pass
    
    @contextlib.contextmanager
    def environment_variables(self, env_vars: Dict[str, str]) -> Iterator[None]:
        """
        Temporarily set environment variables.
        
        Args:
            env_vars: Environment variables to set
        """
        # Store original values
        original_values = {}
        for key, value in env_vars.items():
            original_values[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            yield
        finally:
            # Restore original values
            for key, original_value in original_values.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
    
    def create_test_config_file(self, config_data: Dict[str, Any], 
                               filename: str = "test_config.json") -> str:
        """
        Create a temporary configuration file.
        
        Args:
            config_data: Configuration data
            filename: Config file name
            
        Returns:
            Path to config file
        """
        with self.temporary_directory() as temp_dir:
            config_path = os.path.join(temp_dir, filename)
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            return config_path
    
    def register_cleanup_function(self, cleanup_func: Callable) -> None:
        """Register a function to be called during cleanup."""
        self.cleanup_functions.append(cleanup_func)
    
    def cleanup_all(self) -> None:
        """Clean up all test environment resources."""
        # Run custom cleanup functions
        for cleanup_func in self.cleanup_functions:
            try:
                cleanup_func()
            except Exception as e:
                logger.error(f"Cleanup function failed: {e}")
        
        # Remove temporary directories
        for temp_dir in self.temp_directories:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error(f"Failed to remove temp directory {temp_dir}: {e}")
        
        self.temp_directories.clear()
        self.cleanup_functions.clear()
        
        logger.info("Test environment cleanup completed")


# Convenience functions for common test operations
def time_function(func: Callable, *args, **kwargs) -> TimingResult:
    """
    Time execution of a function.
    
    Args:
        func: Function to time
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Timing result
    """
    timer = TestTimer()
    
    with timer.time_operation(func.__name__) as result:
        func(*args, **kwargs)
    
    return result


@contextlib.contextmanager
def measure_performance(operation_name: str) -> Iterator[Dict[str, Any]]:
    """
    Context manager for measuring operation performance.
    
    Args:
        operation_name: Name of operation
        
    Yields:
        Dictionary to store performance metrics
    """
    timer = TestTimer()
    metrics_collector = TestMetricsCollector()
    
    performance_data = {}
    
    with metrics_collector.collect_metrics():
        with timer.time_operation(operation_name) as timing_result:
            yield performance_data
    
    # Combine timing and metrics
    metrics_summary = metrics_collector.get_metrics_summary()
    performance_data.update({
        "timing": {
            "execution_time_ms": timing_result.execution_time_ms,
            "operation_name": timing_result.operation_name
        },
        "resource_usage": metrics_summary.get("resource_usage", {}),
        "custom_metrics": metrics_summary.get("custom_metrics", {})
    })


def validate_test_prerequisites() -> Dict[str, bool]:
    """
    Validate that test prerequisites are met.
    
    Returns:
        Dictionary of prerequisite checks
    """
    checks = {}
    
    # Check required packages
    try:
        import numpy
        checks["numpy_available"] = True
    except ImportError:
        checks["numpy_available"] = False
    
    try:
        import torch
        checks["torch_available"] = True
    except ImportError:
        checks["torch_available"] = False
    
    try:
        import pytest
        checks["pytest_available"] = True
    except ImportError:
        checks["pytest_available"] = False
    
    # Check system resources
    checks["sufficient_memory"] = psutil.virtual_memory().total > 2 * 1024**3  # 2GB
    checks["sufficient_disk"] = psutil.disk_usage('/').free > 1 * 1024**3     # 1GB
    
    return checks
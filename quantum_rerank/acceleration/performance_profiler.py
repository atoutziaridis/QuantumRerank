"""
Performance Profiler for Hardware Acceleration

Provides performance profiling capabilities for hardware acceleration components,
including timing measurements, memory usage tracking, and optimization recommendations.
"""

import time
import psutil
import torch
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from statistics import mean, stdev
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProfilingConfig:
    """Configuration for performance profiling."""
    enable_memory_tracking: bool = True
    enable_cpu_tracking: bool = True
    enable_gpu_tracking: bool = True
    sampling_interval: float = 0.1
    max_history_size: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "enable_memory_tracking": self.enable_memory_tracking,
            "enable_cpu_tracking": self.enable_cpu_tracking,
            "enable_gpu_tracking": self.enable_gpu_tracking,
            "sampling_interval": self.sampling_interval,
            "max_history_size": self.max_history_size
        }


@dataclass
class ProfileMetrics:
    """Profile metrics for acceleration operations."""
    operation_name: str
    mean_time_ms: float
    std_time_ms: float
    throughput_ops_per_sec: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_utilization_percent: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "operation_name": self.operation_name,
            "mean_time_ms": self.mean_time_ms,
            "std_time_ms": self.std_time_ms,
            "throughput_ops_per_sec": self.throughput_ops_per_sec,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "gpu_utilization_percent": self.gpu_utilization_percent
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for acceleration operations."""
    operation_name: str
    mean_time_ms: float
    std_time_ms: float
    throughput_ops_per_sec: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_utilization_percent: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "operation_name": self.operation_name,
            "mean_time_ms": self.mean_time_ms,
            "std_time_ms": self.std_time_ms,
            "throughput_ops_per_sec": self.throughput_ops_per_sec,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "gpu_utilization_percent": self.gpu_utilization_percent
        }


class PerformanceProfiler:
    """Performance profiler for hardware acceleration operations."""
    
    def __init__(self, config: Optional[ProfilingConfig] = None):
        self.config = config or ProfilingConfig()
        self.metrics_history: List[ProfileMetrics] = []
        self.current_operation: Optional[str] = None
        self.operation_start_time: Optional[float] = None
        
    def start_operation(self, operation_name: str):
        """Start profiling an operation."""
        self.current_operation = operation_name
        self.operation_start_time = time.time()
        
    def end_operation(self) -> Optional[float]:
        """End profiling current operation and return elapsed time."""
        if self.operation_start_time is None:
            return None
        
        elapsed_time = time.time() - self.operation_start_time
        self.operation_start_time = None
        self.current_operation = None
        
        return elapsed_time * 1000  # Convert to milliseconds
    
    def profile_operation(self, operation_name: str, operation_func: Callable, *args, **kwargs):
        """Profile a single operation execution."""
        # Measure initial resources
        initial_memory = self.get_memory_usage()
        
        # Execute operation with timing
        start_time = time.time()
        result = operation_func(*args, **kwargs)
        end_time = time.time()
        
        # Measure final resources
        final_memory = self.get_memory_usage()
        cpu_usage = self.get_cpu_usage()
        
        # Calculate metrics
        execution_time_ms = (end_time - start_time) * 1000
        memory_delta_mb = final_memory - initial_memory
        
        return {
            "result": result,
            "execution_time_ms": execution_time_ms,
            "memory_delta_mb": memory_delta_mb,
            "cpu_usage_percent": cpu_usage
        }
    
    def benchmark_operation(self, operation_name: str, operation_func: Callable, 
                          num_trials: int = 10, *args, **kwargs) -> ProfileMetrics:
        """Benchmark an operation over multiple trials."""
        execution_times = []
        memory_usages = []
        cpu_usages = []
        gpu_utilizations = []
        
        for trial in range(num_trials):
            # Clear cache between trials
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Profile single execution
            profile_result = self.profile_operation(
                operation_name, operation_func, *args, **kwargs
            )
            
            execution_times.append(profile_result["execution_time_ms"])
            memory_usages.append(profile_result["memory_delta_mb"])
            cpu_usages.append(profile_result["cpu_usage_percent"])
            
            # GPU utilization if available
            if torch.cuda.is_available():
                gpu_util = self.get_gpu_utilization()
                gpu_utilizations.append(gpu_util)
        
        # Calculate statistics
        mean_time = mean(execution_times)
        std_time = stdev(execution_times) if len(execution_times) > 1 else 0
        throughput = 1000 / mean_time if mean_time > 0 else 0
        
        # Create metrics
        metrics = ProfileMetrics(
            operation_name=operation_name,
            mean_time_ms=mean_time,
            std_time_ms=std_time,
            throughput_ops_per_sec=throughput,
            memory_usage_mb=mean(memory_usages),
            cpu_usage_percent=mean(cpu_usages),
            gpu_utilization_percent=mean(gpu_utilizations) if gpu_utilizations else None
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)
    
    def get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization percentage if available."""
        if torch.cuda.is_available():
            try:
                # Simple GPU utilization estimate based on memory usage
                gpu_memory_used = torch.cuda.memory_allocated()
                gpu_memory_total = torch.cuda.max_memory_allocated()
                
                if gpu_memory_total > 0:
                    return (gpu_memory_used / gpu_memory_total) * 100
                else:
                    return 0.0
            except:
                return None
        return None
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on profiling history."""
        recommendations = []
        
        if not self.metrics_history:
            return ["No profiling data available"]
        
        # Analyze latency patterns
        recent_metrics = self.metrics_history[-5:]  # Last 5 operations
        avg_latency = mean([m.mean_time_ms for m in recent_metrics])
        
        if avg_latency > 100:
            recommendations.append("Consider hardware acceleration for latency optimization")
        
        # Analyze memory usage
        avg_memory = mean([m.memory_usage_mb for m in recent_metrics])
        if avg_memory > 1000:  # > 1GB
            recommendations.append("Consider memory optimization and compression")
        
        # Analyze CPU usage
        avg_cpu = mean([m.cpu_usage_percent for m in recent_metrics])
        if avg_cpu > 80:
            recommendations.append("Consider parallel processing or GPU acceleration")
        
        # Analyze GPU utilization
        gpu_metrics = [m for m in recent_metrics if m.gpu_utilization_percent is not None]
        if gpu_metrics:
            avg_gpu = mean([m.gpu_utilization_percent for m in gpu_metrics])
            if avg_gpu < 30:
                recommendations.append("GPU underutilized - consider batch processing")
            elif avg_gpu > 90:
                recommendations.append("GPU heavily utilized - consider optimization")
        
        return recommendations if recommendations else ["Performance is within acceptable ranges"]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of all profiled operations."""
        if not self.metrics_history:
            return {"status": "No profiling data available"}
        
        # Calculate overall statistics
        all_latencies = [m.mean_time_ms for m in self.metrics_history]
        all_throughputs = [m.throughput_ops_per_sec for m in self.metrics_history]
        all_memory = [m.memory_usage_mb for m in self.metrics_history]
        
        summary = {
            "total_operations_profiled": len(self.metrics_history),
            "overall_stats": {
                "mean_latency_ms": mean(all_latencies),
                "mean_throughput_ops_per_sec": mean(all_throughputs),
                "mean_memory_usage_mb": mean(all_memory)
            },
            "operation_breakdown": {
                m.operation_name: m.to_dict() for m in self.metrics_history
            },
            "optimization_recommendations": self.get_optimization_recommendations()
        }
        
        return summary
    
    def clear_history(self):
        """Clear profiling history."""
        self.metrics_history.clear()
        logger.info("Performance profiling history cleared")
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        import json
        
        summary = self.get_performance_summary()
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Performance metrics exported to {filepath}")


# Global profiler instance
_global_profiler = PerformanceProfiler()


def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    return _global_profiler


def profile_operation(operation_name: str):
    """Decorator for profiling operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            profile_result = profiler.profile_operation(operation_name, func, *args, **kwargs)
            return profile_result["result"]
        return wrapper
    return decorator


def benchmark_function(operation_name: str, num_trials: int = 10):
    """Decorator for benchmarking functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            metrics = profiler.benchmark_operation(
                operation_name, func, num_trials, *args, **kwargs
            )
            
            # Return both result and metrics
            result = func(*args, **kwargs)
            return result, metrics
        return wrapper
    return decorator
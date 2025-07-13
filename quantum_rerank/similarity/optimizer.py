"""
Performance optimization for similarity computation methods.

This module provides dynamic performance optimization strategies to improve
latency, throughput, and resource efficiency of similarity methods.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time

from ..utils import get_logger


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    LATENCY_FOCUSED = "latency_focused"
    THROUGHPUT_FOCUSED = "throughput_focused"
    ACCURACY_FOCUSED = "accuracy_focused"
    RESOURCE_FOCUSED = "resource_focused"
    BALANCED = "balanced"


class PerformanceBottleneck(Enum):
    """Types of performance bottlenecks."""
    LATENCY = "latency"
    MEMORY = "memory"
    CPU = "cpu"
    ACCURACY = "accuracy"
    THROUGHPUT = "throughput"


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    enable_adaptive_batch_size: bool = True
    enable_method_switching: bool = True
    enable_caching: bool = True
    enable_approximation: bool = True
    max_optimization_overhead_ms: float = 5.0
    target_latency_ms: float = 100.0
    target_accuracy: float = 0.9


class LatencyOptimizer:
    """Optimizer focused on reducing computation latency."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.optimization_history = {}
    
    def optimize(
        self,
        method_name: str,
        performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize method for reduced latency."""
        optimizations = {}
        
        current_latency = performance_data.get("avg_latency_ms", 0)
        target_latency = performance_data.get("target_latency_ms", 100)
        
        if current_latency > target_latency:
            # Suggest approximation methods
            if "quantum" in method_name:
                optimizations["use_approximation"] = True
                optimizations["suggested_method"] = "quantum_approximate"
            
            # Suggest batch size optimization
            batch_size = performance_data.get("avg_batch_size", 50)
            if batch_size > 20:
                optimizations["optimal_batch_size"] = min(20, batch_size // 2)
            
            # Suggest caching
            optimizations["enable_aggressive_caching"] = True
            
            self.logger.info(
                f"Latency optimization for {method_name}: "
                f"current={current_latency:.1f}ms, target={target_latency:.1f}ms"
            )
        
        return optimizations


class ThroughputOptimizer:
    """Optimizer focused on maximizing throughput."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def optimize(
        self,
        method_name: str,
        performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize method for increased throughput."""
        optimizations = {}
        
        # Enable batch processing for suitable methods
        if performance_data.get("batch_optimized", False):
            current_batch = performance_data.get("avg_batch_size", 1)
            optimizations["optimal_batch_size"] = min(100, current_batch * 2)
        
        # Suggest parallel processing
        optimizations["enable_parallel_processing"] = True
        optimizations["max_workers"] = 4
        
        # Use faster methods for large batches
        batch_size = performance_data.get("avg_batch_size", 1)
        if batch_size > 50 and "classical" not in method_name:
            optimizations["suggested_method"] = "hybrid_batch"
        
        return optimizations


class AccuracyOptimizer:
    """Optimizer focused on maintaining high accuracy."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def optimize(
        self,
        method_name: str,
        performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize method for better accuracy."""
        optimizations = {}
        
        current_accuracy = performance_data.get("avg_accuracy", 0)
        target_accuracy = performance_data.get("target_accuracy", 0.9)
        
        if current_accuracy < target_accuracy:
            # Suggest higher accuracy methods
            if "classical" in method_name:
                optimizations["suggested_method"] = "quantum_precise"
            elif "approximate" in method_name:
                optimizations["suggested_method"] = "quantum_precise"
            
            # Disable approximations
            optimizations["use_approximation"] = False
            
            # Enable ensemble methods for critical accuracy
            if target_accuracy > 0.95:
                optimizations["suggested_method"] = "ensemble"
            
            self.logger.info(
                f"Accuracy optimization for {method_name}: "
                f"current={current_accuracy:.3f}, target={target_accuracy:.3f}"
            )
        
        return optimizations


class ResourceOptimizer:
    """Optimizer focused on efficient resource usage."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def optimize(
        self,
        method_name: str,
        performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize method for efficient resource usage."""
        optimizations = {}
        
        # Memory optimization
        memory_usage = performance_data.get("memory_usage_mb", 0)
        if memory_usage > 1000:  # High memory usage
            optimizations["reduce_batch_size"] = True
            optimizations["enable_memory_efficient_processing"] = True
            
            if "quantum" in method_name:
                optimizations["suggested_method"] = "classical_fast"
        
        # CPU optimization
        cpu_usage = performance_data.get("cpu_usage", 0)
        if cpu_usage > 0.8:  # High CPU usage
            optimizations["reduce_parallel_workers"] = True
            optimizations["enable_cpu_efficient_methods"] = True
        
        return optimizations


class PerformanceOptimizer:
    """
    Dynamic performance optimizer for similarity computation methods.
    
    This optimizer analyzes method performance and applies appropriate
    optimization strategies to improve overall system performance.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.logger = get_logger(__name__)
        
        # Initialize optimizers
        self.optimizers = {
            OptimizationStrategy.LATENCY_FOCUSED: LatencyOptimizer(),
            OptimizationStrategy.THROUGHPUT_FOCUSED: ThroughputOptimizer(),
            OptimizationStrategy.ACCURACY_FOCUSED: AccuracyOptimizer(),
            OptimizationStrategy.RESOURCE_FOCUSED: ResourceOptimizer()
        }
        
        # Optimization history
        self.optimization_history: Dict[str, List[Dict]] = {}
        
        # Performance baselines
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        
        # Adaptive parameters
        self.adaptive_params: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info(f"Initialized PerformanceOptimizer with strategy: {self.config.strategy.value}")
    
    def optimize_method(
        self,
        method_name: str,
        result: Any,
        requirements: Any
    ) -> Dict[str, Any]:
        """
        Optimize method based on performance result and requirements.
        
        Args:
            method_name: Name of the similarity method
            result: Performance result from method execution
            requirements: Requirements that triggered optimization
            
        Returns:
            Dictionary of optimization recommendations
        """
        start_time = time.time()
        
        # Analyze performance
        performance_data = self._extract_performance_data(result, requirements)
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(method_name, performance_data)
        
        # Apply optimization strategies
        optimizations = self._apply_optimization_strategies(
            method_name, performance_data, bottlenecks
        )
        
        # Update adaptive parameters
        self._update_adaptive_parameters(method_name, performance_data, optimizations)
        
        # Record optimization
        optimization_time = (time.time() - start_time) * 1000
        self._record_optimization(method_name, optimizations, optimization_time)
        
        return optimizations
    
    def get_optimal_batch_size(
        self,
        method_name: str,
        target_latency_ms: float
    ) -> int:
        """Get optimal batch size for method and latency target."""
        # Check adaptive parameters
        if method_name in self.adaptive_params:
            params = self.adaptive_params[method_name]
            if "optimal_batch_size" in params:
                return params["optimal_batch_size"]
        
        # Default heuristics based on method type
        if "classical" in method_name:
            return 100  # Classical methods handle large batches well
        elif "quantum" in method_name:
            return 20   # Quantum methods prefer smaller batches
        elif "hybrid" in method_name:
            return 50   # Hybrid methods are balanced
        else:
            return 32   # Default batch size
    
    def get_method_recommendation(
        self,
        current_method: str,
        performance_issues: List[str]
    ) -> Optional[str]:
        """Get method recommendation based on performance issues."""
        recommendations = {
            "high_latency": {
                "quantum_precise": "quantum_approximate",
                "quantum_approximate": "classical_fast",
                "hybrid_balanced": "hybrid_batch",
                "ensemble": "hybrid_balanced"
            },
            "low_accuracy": {
                "classical_fast": "classical_accurate",
                "classical_accurate": "quantum_approximate",
                "quantum_approximate": "quantum_precise",
                "hybrid_batch": "hybrid_balanced"
            },
            "high_memory": {
                "quantum_precise": "classical_fast",
                "ensemble": "hybrid_balanced",
                "hybrid_balanced": "classical_accurate"
            }
        }
        
        # Find best recommendation
        for issue in performance_issues:
            if issue in recommendations and current_method in recommendations[issue]:
                return recommendations[issue][current_method]
        
        return None
    
    def _extract_performance_data(self, result: Any, requirements: Any) -> Dict[str, Any]:
        """Extract relevant performance data from result."""
        data = {}
        
        # Extract from result
        if hasattr(result, 'computation_time_ms'):
            data["avg_latency_ms"] = result.computation_time_ms
        
        if hasattr(result, 'accuracy_estimate'):
            data["avg_accuracy"] = result.accuracy_estimate
        
        if hasattr(result, 'candidate_scores'):
            data["avg_batch_size"] = len(result.candidate_scores)
        
        # Extract from requirements
        if hasattr(requirements, 'max_latency_ms'):
            data["target_latency_ms"] = requirements.max_latency_ms
        
        if hasattr(requirements, 'min_accuracy'):
            data["target_accuracy"] = requirements.min_accuracy
        
        return data
    
    def _identify_bottlenecks(
        self,
        method_name: str,
        performance_data: Dict[str, Any]
    ) -> List[PerformanceBottleneck]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Latency bottleneck
        current_latency = performance_data.get("avg_latency_ms", 0)
        target_latency = performance_data.get("target_latency_ms", self.config.target_latency_ms)
        
        if current_latency > target_latency:
            bottlenecks.append(PerformanceBottleneck.LATENCY)
        
        # Accuracy bottleneck
        current_accuracy = performance_data.get("avg_accuracy", 1.0)
        target_accuracy = performance_data.get("target_accuracy", self.config.target_accuracy)
        
        if current_accuracy < target_accuracy:
            bottlenecks.append(PerformanceBottleneck.ACCURACY)
        
        # Memory bottleneck (estimated)
        batch_size = performance_data.get("avg_batch_size", 1)
        if "quantum" in method_name and batch_size > 50:
            bottlenecks.append(PerformanceBottleneck.MEMORY)
        
        return bottlenecks
    
    def _apply_optimization_strategies(
        self,
        method_name: str,
        performance_data: Dict[str, Any],
        bottlenecks: List[PerformanceBottleneck]
    ) -> Dict[str, Any]:
        """Apply optimization strategies based on bottlenecks."""
        all_optimizations = {}
        
        # Apply strategy-specific optimizations
        if self.config.strategy == OptimizationStrategy.BALANCED:
            # Apply multiple optimizers based on bottlenecks
            for bottleneck in bottlenecks:
                if bottleneck == PerformanceBottleneck.LATENCY:
                    opts = self.optimizers[OptimizationStrategy.LATENCY_FOCUSED].optimize(
                        method_name, performance_data
                    )
                    all_optimizations.update(opts)
                
                elif bottleneck == PerformanceBottleneck.ACCURACY:
                    opts = self.optimizers[OptimizationStrategy.ACCURACY_FOCUSED].optimize(
                        method_name, performance_data
                    )
                    all_optimizations.update(opts)
                
                elif bottleneck == PerformanceBottleneck.MEMORY:
                    opts = self.optimizers[OptimizationStrategy.RESOURCE_FOCUSED].optimize(
                        method_name, performance_data
                    )
                    all_optimizations.update(opts)
        else:
            # Apply single strategy
            if self.config.strategy in self.optimizers:
                all_optimizations = self.optimizers[self.config.strategy].optimize(
                    method_name, performance_data
                )
        
        # Apply adaptive optimizations
        adaptive_opts = self._get_adaptive_optimizations(method_name, performance_data)
        all_optimizations.update(adaptive_opts)
        
        return all_optimizations
    
    def _get_adaptive_optimizations(
        self,
        method_name: str,
        performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get adaptive optimizations based on learned parameters."""
        optimizations = {}
        
        if not self.config.enable_adaptive_batch_size:
            return optimizations
        
        # Adaptive batch size optimization
        current_batch = performance_data.get("avg_batch_size", 32)
        current_latency = performance_data.get("avg_latency_ms", 0)
        target_latency = performance_data.get("target_latency_ms", self.config.target_latency_ms)
        
        if current_latency > target_latency and current_batch > 10:
            # Reduce batch size
            new_batch_size = max(10, int(current_batch * 0.8))
            optimizations["optimal_batch_size"] = new_batch_size
        
        elif current_latency < target_latency * 0.7 and current_batch < 100:
            # Increase batch size
            new_batch_size = min(100, int(current_batch * 1.2))
            optimizations["optimal_batch_size"] = new_batch_size
        
        return optimizations
    
    def _update_adaptive_parameters(
        self,
        method_name: str,
        performance_data: Dict[str, Any],
        optimizations: Dict[str, Any]
    ) -> None:
        """Update adaptive parameters based on optimization results."""
        if method_name not in self.adaptive_params:
            self.adaptive_params[method_name] = {}
        
        params = self.adaptive_params[method_name]
        
        # Update optimal batch size
        if "optimal_batch_size" in optimizations:
            params["optimal_batch_size"] = optimizations["optimal_batch_size"]
        
        # Update performance baseline
        if method_name not in self.performance_baselines:
            self.performance_baselines[method_name] = {}
        
        baseline = self.performance_baselines[method_name]
        baseline["latency_ms"] = performance_data.get("avg_latency_ms", 0)
        baseline["accuracy"] = performance_data.get("avg_accuracy", 0)
        baseline["last_updated"] = time.time()
    
    def _record_optimization(
        self,
        method_name: str,
        optimizations: Dict[str, Any],
        optimization_time_ms: float
    ) -> None:
        """Record optimization for analysis."""
        if method_name not in self.optimization_history:
            self.optimization_history[method_name] = []
        
        record = {
            "timestamp": time.time(),
            "optimizations": optimizations.copy(),
            "optimization_time_ms": optimization_time_ms
        }
        
        self.optimization_history[method_name].append(record)
        
        # Keep last 100 records
        if len(self.optimization_history[method_name]) > 100:
            self.optimization_history[method_name] = self.optimization_history[method_name][-100:]
        
        self.logger.debug(
            f"Recorded optimization for {method_name}: {len(optimizations)} recommendations"
        )
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = {
            "total_methods_optimized": len(self.optimization_history),
            "total_optimizations": sum(len(history) for history in self.optimization_history.values()),
            "adaptive_parameters": len(self.adaptive_params),
            "performance_baselines": len(self.performance_baselines)
        }
        
        # Method-specific statistics
        method_stats = {}
        for method, history in self.optimization_history.items():
            if history:
                avg_opt_time = np.mean([r["optimization_time_ms"] for r in history])
                common_optimizations = self._get_common_optimizations(history)
                
                method_stats[method] = {
                    "optimization_count": len(history),
                    "avg_optimization_time_ms": avg_opt_time,
                    "common_optimizations": common_optimizations
                }
        
        stats["method_statistics"] = method_stats
        
        return stats
    
    def _get_common_optimizations(self, history: List[Dict]) -> List[str]:
        """Get most common optimizations for a method."""
        optimization_counts = {}
        
        for record in history:
            for opt_key in record["optimizations"].keys():
                optimization_counts[opt_key] = optimization_counts.get(opt_key, 0) + 1
        
        # Sort by frequency
        sorted_opts = sorted(
            optimization_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [opt for opt, count in sorted_opts[:3]]  # Top 3


__all__ = [
    "OptimizationStrategy",
    "PerformanceBottleneck",
    "OptimizationConfig",
    "PerformanceOptimizer"
]
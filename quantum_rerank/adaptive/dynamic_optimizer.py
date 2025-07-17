"""
Dynamic Optimizer for Adaptive Compression

Provides dynamic optimization strategies that adapt compression and performance
parameters based on real-time resource availability and performance requirements.
"""

import time
import torch
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Optimization strategies for adaptive compression."""
    PERFORMANCE_FIRST = "performance_first"
    QUALITY_FIRST = "quality_first"
    BALANCED = "balanced"
    RESOURCE_AWARE = "resource_aware"
    ADAPTIVE = "adaptive"


@dataclass
class OptimizationConfig:
    """Configuration for dynamic optimization."""
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    target_latency_ms: float = 100.0
    target_memory_mb: float = 2048.0
    target_accuracy: float = 0.95
    adaptation_rate: float = 0.1
    stability_threshold: float = 0.05
    optimization_window: int = 10
    enable_auto_tuning: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "strategy": self.strategy.value,
            "target_latency_ms": self.target_latency_ms,
            "target_memory_mb": self.target_memory_mb,
            "target_accuracy": self.target_accuracy,
            "adaptation_rate": self.adaptation_rate,
            "stability_threshold": self.stability_threshold,
            "optimization_window": self.optimization_window,
            "enable_auto_tuning": self.enable_auto_tuning
        }


@dataclass
class OptimizationMetrics:
    """Metrics for optimization performance."""
    latency_ms: float
    memory_mb: float
    accuracy: float
    compression_ratio: float
    throughput_qps: float
    stability_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "latency_ms": self.latency_ms,
            "memory_mb": self.memory_mb,
            "accuracy": self.accuracy,
            "compression_ratio": self.compression_ratio,
            "throughput_qps": self.throughput_qps,
            "stability_score": self.stability_score
        }


class DynamicOptimizer:
    """Dynamic optimizer for adaptive compression systems."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.metrics_history: List[OptimizationMetrics] = []
        self.current_parameters = {
            "compression_level": 0.5,
            "quality_threshold": 0.8,
            "batch_size": 32,
            "precision": "float32"
        }
        self.adaptation_count = 0
        
        logger.info(f"Dynamic Optimizer initialized with strategy: {config.strategy.value}")
    
    def update_metrics(self, metrics: OptimizationMetrics):
        """Update optimization metrics."""
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics
        if len(self.metrics_history) > self.config.optimization_window:
            self.metrics_history = self.metrics_history[-self.config.optimization_window:]
        
        # Trigger adaptation if enabled
        if self.config.enable_auto_tuning:
            self._adapt_parameters()
    
    def _adapt_parameters(self):
        """Adapt parameters based on current metrics."""
        if len(self.metrics_history) < 2:
            return
        
        current_metrics = self.metrics_history[-1]
        
        # Calculate adaptation based on strategy
        if self.config.strategy == OptimizationStrategy.PERFORMANCE_FIRST:
            self._adapt_for_performance(current_metrics)
        elif self.config.strategy == OptimizationStrategy.QUALITY_FIRST:
            self._adapt_for_quality(current_metrics)
        elif self.config.strategy == OptimizationStrategy.BALANCED:
            self._adapt_balanced(current_metrics)
        elif self.config.strategy == OptimizationStrategy.RESOURCE_AWARE:
            self._adapt_resource_aware(current_metrics)
        elif self.config.strategy == OptimizationStrategy.ADAPTIVE:
            self._adapt_adaptive(current_metrics)
        
        self.adaptation_count += 1
        logger.debug(f"Parameters adapted (count: {self.adaptation_count})")
    
    def _adapt_for_performance(self, metrics: OptimizationMetrics):
        """Adapt parameters prioritizing performance."""
        adaptation_rate = self.config.adaptation_rate
        
        # Reduce compression if latency is too high
        if metrics.latency_ms > self.config.target_latency_ms:
            self.current_parameters["compression_level"] = max(
                0.1, self.current_parameters["compression_level"] - adaptation_rate
            )
        
        # Reduce quality threshold if memory usage is high
        if metrics.memory_mb > self.config.target_memory_mb:
            self.current_parameters["quality_threshold"] = max(
                0.5, self.current_parameters["quality_threshold"] - adaptation_rate
            )
        
        # Increase batch size if throughput is low
        if metrics.throughput_qps < 10.0:
            self.current_parameters["batch_size"] = min(
                64, int(self.current_parameters["batch_size"] * (1 + adaptation_rate))
            )
    
    def _adapt_for_quality(self, metrics: OptimizationMetrics):
        """Adapt parameters prioritizing quality."""
        adaptation_rate = self.config.adaptation_rate
        
        # Increase compression if accuracy is sufficient
        if metrics.accuracy > self.config.target_accuracy:
            self.current_parameters["compression_level"] = min(
                1.0, self.current_parameters["compression_level"] + adaptation_rate
            )
        
        # Increase quality threshold if accuracy is low
        if metrics.accuracy < self.config.target_accuracy:
            self.current_parameters["quality_threshold"] = min(
                1.0, self.current_parameters["quality_threshold"] + adaptation_rate
            )
        
        # Use higher precision if needed
        if metrics.accuracy < 0.9 and self.current_parameters["precision"] == "float16":
            self.current_parameters["precision"] = "float32"
    
    def _adapt_balanced(self, metrics: OptimizationMetrics):
        """Adapt parameters with balanced approach."""
        adaptation_rate = self.config.adaptation_rate * 0.5  # More conservative
        
        # Balance latency and accuracy
        latency_score = min(1.0, self.config.target_latency_ms / max(1.0, metrics.latency_ms))
        accuracy_score = min(1.0, metrics.accuracy / self.config.target_accuracy)
        
        combined_score = (latency_score + accuracy_score) / 2
        
        if combined_score < 0.8:
            # Adjust compression based on which metric is worse
            if latency_score < accuracy_score:
                # Prioritize latency
                self.current_parameters["compression_level"] = max(
                    0.1, self.current_parameters["compression_level"] - adaptation_rate
                )
            else:
                # Prioritize accuracy
                self.current_parameters["quality_threshold"] = min(
                    1.0, self.current_parameters["quality_threshold"] + adaptation_rate
                )
    
    def _adapt_resource_aware(self, metrics: OptimizationMetrics):
        """Adapt parameters based on resource availability."""
        adaptation_rate = self.config.adaptation_rate
        
        # Memory-based adaptation
        memory_utilization = metrics.memory_mb / self.config.target_memory_mb
        if memory_utilization > 0.8:
            self.current_parameters["compression_level"] = min(
                1.0, self.current_parameters["compression_level"] + adaptation_rate
            )
            self.current_parameters["batch_size"] = max(
                16, int(self.current_parameters["batch_size"] * 0.8)
            )
        
        # CPU/latency-based adaptation
        if metrics.latency_ms > self.config.target_latency_ms * 1.2:
            self.current_parameters["precision"] = "float16"
            self.current_parameters["quality_threshold"] = max(
                0.6, self.current_parameters["quality_threshold"] - adaptation_rate
            )
    
    def _adapt_adaptive(self, metrics: OptimizationMetrics):
        """Fully adaptive optimization strategy."""
        if len(self.metrics_history) < 3:
            return
        
        # Calculate trends
        recent_metrics = self.metrics_history[-3:]
        latency_trend = self._calculate_trend([m.latency_ms for m in recent_metrics])
        accuracy_trend = self._calculate_trend([m.accuracy for m in recent_metrics])
        memory_trend = self._calculate_trend([m.memory_mb for m in recent_metrics])
        
        # Adaptive response based on trends
        if latency_trend > 0.1:  # Increasing latency
            self.current_parameters["compression_level"] = max(
                0.2, self.current_parameters["compression_level"] - 0.1
            )
        
        if accuracy_trend < -0.05:  # Decreasing accuracy
            self.current_parameters["quality_threshold"] = min(
                1.0, self.current_parameters["quality_threshold"] + 0.1
            )
        
        if memory_trend > 0.1:  # Increasing memory usage
            self.current_parameters["batch_size"] = max(
                16, int(self.current_parameters["batch_size"] * 0.9)
            )
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in a list of values."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        n = len(values)
        x = list(range(n))
        
        # Calculate slope
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        if n * sum_x2 - sum_x ** 2 == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        return slope
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current optimization parameters."""
        return self.current_parameters.copy()
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """Set optimization parameters."""
        self.current_parameters.update(parameters)
        logger.info(f"Parameters updated: {parameters}")
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get optimization recommendations based on current metrics."""
        if not self.metrics_history:
            return {"status": "no_data", "recommendations": []}
        
        current_metrics = self.metrics_history[-1]
        recommendations = []
        
        # Latency recommendations
        if current_metrics.latency_ms > self.config.target_latency_ms:
            recommendations.append({
                "type": "latency",
                "current": current_metrics.latency_ms,
                "target": self.config.target_latency_ms,
                "suggestion": "Reduce compression level or lower quality threshold"
            })
        
        # Memory recommendations
        if current_metrics.memory_mb > self.config.target_memory_mb:
            recommendations.append({
                "type": "memory",
                "current": current_metrics.memory_mb,
                "target": self.config.target_memory_mb,
                "suggestion": "Increase compression or reduce batch size"
            })
        
        # Accuracy recommendations
        if current_metrics.accuracy < self.config.target_accuracy:
            recommendations.append({
                "type": "accuracy",
                "current": current_metrics.accuracy,
                "target": self.config.target_accuracy,
                "suggestion": "Increase quality threshold or reduce compression"
            })
        
        return {
            "status": "ready",
            "recommendations": recommendations,
            "current_metrics": current_metrics.to_dict(),
            "optimization_strategy": self.config.strategy.value
        }
    
    def calculate_stability_score(self) -> float:
        """Calculate stability score based on metrics variance."""
        if len(self.metrics_history) < 3:
            return 1.0
        
        recent_metrics = self.metrics_history[-5:]
        
        # Calculate variance for key metrics
        latencies = [m.latency_ms for m in recent_metrics]
        accuracies = [m.accuracy for m in recent_metrics]
        
        latency_variance = self._calculate_variance(latencies)
        accuracy_variance = self._calculate_variance(accuracies)
        
        # Normalize variances
        latency_stability = max(0.0, 1.0 - latency_variance / 100.0)
        accuracy_stability = max(0.0, 1.0 - accuracy_variance / 0.1)
        
        return (latency_stability + accuracy_stability) / 2
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def reset_optimization(self):
        """Reset optimization state."""
        self.metrics_history.clear()
        self.current_parameters = {
            "compression_level": 0.5,
            "quality_threshold": 0.8,
            "batch_size": 32,
            "precision": "float32"
        }
        self.adaptation_count = 0
        logger.info("Optimization state reset")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary and statistics."""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = self.metrics_history[-min(5, len(self.metrics_history)):]
        
        summary = {
            "optimization_strategy": self.config.strategy.value,
            "adaptation_count": self.adaptation_count,
            "metrics_collected": len(self.metrics_history),
            "current_parameters": self.current_parameters,
            "stability_score": self.calculate_stability_score(),
            "recent_performance": {
                "avg_latency_ms": sum(m.latency_ms for m in recent_metrics) / len(recent_metrics),
                "avg_accuracy": sum(m.accuracy for m in recent_metrics) / len(recent_metrics),
                "avg_memory_mb": sum(m.memory_mb for m in recent_metrics) / len(recent_metrics),
                "avg_compression_ratio": sum(m.compression_ratio for m in recent_metrics) / len(recent_metrics)
            },
            "target_compliance": {
                "latency": recent_metrics[-1].latency_ms <= self.config.target_latency_ms,
                "accuracy": recent_metrics[-1].accuracy >= self.config.target_accuracy,
                "memory": recent_metrics[-1].memory_mb <= self.config.target_memory_mb
            }
        }
        
        return summary


# Utility functions
def create_optimization_config(strategy: OptimizationStrategy) -> OptimizationConfig:
    """Create optimization configuration for specific strategy."""
    config = OptimizationConfig(strategy=strategy)
    
    if strategy == OptimizationStrategy.PERFORMANCE_FIRST:
        config.target_latency_ms = 50.0
        config.adaptation_rate = 0.2
    elif strategy == OptimizationStrategy.QUALITY_FIRST:
        config.target_accuracy = 0.98
        config.adaptation_rate = 0.05
    elif strategy == OptimizationStrategy.RESOURCE_AWARE:
        config.target_memory_mb = 1024.0
        config.adaptation_rate = 0.15
    
    return config


def evaluate_optimization_effectiveness(optimizer: DynamicOptimizer) -> Dict[str, Any]:
    """Evaluate optimization effectiveness."""
    summary = optimizer.get_optimization_summary()
    
    if summary.get("status") == "no_data":
        return {"effectiveness": "unknown", "reason": "insufficient_data"}
    
    recent_perf = summary["recent_performance"]
    target_compliance = summary["target_compliance"]
    
    effectiveness_score = sum(target_compliance.values()) / len(target_compliance)
    
    return {
        "effectiveness": "high" if effectiveness_score > 0.8 else "medium" if effectiveness_score > 0.5 else "low",
        "effectiveness_score": effectiveness_score,
        "stability_score": summary["stability_score"],
        "adaptation_count": summary["adaptation_count"],
        "recommendations": optimizer.get_optimization_recommendations()
    }
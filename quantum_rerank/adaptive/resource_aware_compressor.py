"""
Resource-Aware Adaptive Compression.

This module implements dynamic compression that adapts to real-time resource
constraints, automatically adjusting quality vs performance trade-offs to
maintain optimal system performance under varying conditions.

Based on:
- Phase 3 adaptive compression requirements
- Dynamic quality vs resource trade-off research
- Edge deployment resource constraints
- Real-time performance optimization
"""

import torch
import torch.nn as nn
import numpy as np
import time
import psutil
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CompressionLevel(Enum):
    """Compression levels with different quality/performance trade-offs."""
    MINIMAL = "minimal"      # 2x compression, highest quality
    LOW = "low"             # 4x compression, high quality  
    MEDIUM = "medium"       # 8x compression, medium quality
    HIGH = "high"           # 16x compression, low quality
    MAXIMUM = "maximum"     # 32x compression, minimal quality
    ADAPTIVE = "adaptive"   # Dynamic based on resources


@dataclass
class CompressionConfig:
    """Configuration for adaptive compression."""
    default_level: CompressionLevel = CompressionLevel.MEDIUM
    enable_adaptive: bool = True
    min_quality_threshold: float = 0.85  # Minimum acceptable quality
    max_compression_ratio: float = 32.0  # Maximum compression allowed
    latency_target_ms: float = 100.0     # Target latency constraint
    memory_limit_mb: float = 2048.0      # Memory usage limit
    cpu_usage_threshold: float = 0.8     # CPU usage threshold
    adaptation_interval: float = 1.0     # Seconds between adaptations
    quality_weight: float = 0.6          # Weight for quality vs speed
    enable_caching: bool = True          # Cache compression decisions


class ResourceMetrics:
    """Real-time resource usage metrics."""
    
    def __init__(self):
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.memory_used_mb = 0.0
        self.memory_available_mb = 0.0
        self.gpu_memory_used_mb = 0.0
        self.gpu_memory_total_mb = 0.0
        self.disk_io_read_mb = 0.0
        self.disk_io_write_mb = 0.0
        self.network_io_sent_mb = 0.0
        self.network_io_recv_mb = 0.0
        self.timestamp = time.time()
    
    @classmethod
    def collect_current_metrics(cls) -> 'ResourceMetrics':
        """Collect current system resource metrics."""
        metrics = cls()
        
        # CPU metrics
        metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.memory_percent = memory.percent
        metrics.memory_used_mb = memory.used / 1024 / 1024
        metrics.memory_available_mb = memory.available / 1024 / 1024
        
        # GPU metrics (if available)
        if torch.cuda.is_available():
            metrics.gpu_memory_used_mb = torch.cuda.memory_allocated() / 1024 / 1024
            metrics.gpu_memory_total_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        
        # Disk I/O metrics
        disk_io = psutil.disk_io_counters()
        if disk_io:
            metrics.disk_io_read_mb = disk_io.read_bytes / 1024 / 1024
            metrics.disk_io_write_mb = disk_io.write_bytes / 1024 / 1024
        
        # Network I/O metrics
        network_io = psutil.net_io_counters()
        if network_io:
            metrics.network_io_sent_mb = network_io.bytes_sent / 1024 / 1024
            metrics.network_io_recv_mb = network_io.bytes_recv / 1024 / 1024
        
        metrics.timestamp = time.time()
        
        return metrics
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_mb": self.memory_used_mb,
            "memory_available_mb": self.memory_available_mb,
            "gpu_memory_used_mb": self.gpu_memory_used_mb,
            "gpu_memory_total_mb": self.gpu_memory_total_mb,
            "disk_io_read_mb": self.disk_io_read_mb,
            "disk_io_write_mb": self.disk_io_write_mb,
            "network_io_sent_mb": self.network_io_sent_mb,
            "network_io_recv_mb": self.network_io_recv_mb,
            "timestamp": self.timestamp
        }


class CompressionStrategy:
    """
    Individual compression strategy with specific parameters.
    
    Each strategy defines a set of compression parameters and 
    their expected quality/performance characteristics.
    """
    
    def __init__(self, 
                 name: str,
                 compression_ratio: float,
                 expected_quality: float,
                 expected_latency_ms: float,
                 memory_overhead_mb: float):
        self.name = name
        self.compression_ratio = compression_ratio
        self.expected_quality = expected_quality
        self.expected_latency_ms = expected_latency_ms
        self.memory_overhead_mb = memory_overhead_mb
        self.actual_performance = {}  # Learned performance metrics
        self.usage_count = 0
    
    def update_performance(self, 
                          actual_quality: float,
                          actual_latency_ms: float,
                          actual_memory_mb: float):
        """Update strategy with actual performance data."""
        self.usage_count += 1
        
        # Exponential moving average for performance metrics
        alpha = 0.1  # Learning rate
        
        if 'quality' not in self.actual_performance:
            self.actual_performance = {
                'quality': actual_quality,
                'latency_ms': actual_latency_ms,
                'memory_mb': actual_memory_mb
            }
        else:
            self.actual_performance['quality'] = (
                alpha * actual_quality + 
                (1 - alpha) * self.actual_performance['quality']
            )
            self.actual_performance['latency_ms'] = (
                alpha * actual_latency_ms + 
                (1 - alpha) * self.actual_performance['latency_ms']
            )
            self.actual_performance['memory_mb'] = (
                alpha * actual_memory_mb + 
                (1 - alpha) * self.actual_performance['memory_mb']
            )
    
    def get_score(self, 
                 resource_pressure: float,
                 quality_weight: float) -> float:
        """
        Calculate strategy score based on current conditions.
        
        Args:
            resource_pressure: 0-1 indicating resource constraint level
            quality_weight: 0-1 weight for quality vs performance
            
        Returns:
            Strategy score (higher is better)
        """
        # Use actual performance if available, otherwise use expected
        if self.actual_performance:
            quality = self.actual_performance['quality']
            latency = self.actual_performance['latency_ms']
            memory = self.actual_performance['memory_mb']
        else:
            quality = self.expected_quality
            latency = self.expected_latency_ms
            memory = self.memory_overhead_mb
        
        # Quality score (higher is better)
        quality_score = quality * quality_weight
        
        # Performance score (lower latency/memory is better)
        performance_score = (1 - quality_weight) * (
            (100.0 / max(latency, 1.0)) * 0.5 +  # Latency component
            (1000.0 / max(memory, 1.0)) * 0.5    # Memory component
        )
        
        # Resource pressure penalty (penalize heavy strategies under pressure)
        resource_penalty = resource_pressure * (self.compression_ratio / 32.0)
        
        total_score = quality_score + performance_score - resource_penalty
        
        return max(0.0, total_score)


class ResourceAwareCompressor:
    """
    Main adaptive compression engine.
    
    Dynamically selects optimal compression strategies based on real-time
    resource availability and performance requirements.
    """
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.current_strategy = None
        self.strategies = self._initialize_strategies()
        self.resource_history = []
        self.performance_history = []
        self.last_adaptation = 0.0
        self.compression_cache = {} if config.enable_caching else None
        
        logger.info(f"ResourceAwareCompressor initialized with {len(self.strategies)} strategies")
    
    def _initialize_strategies(self) -> List[CompressionStrategy]:
        """Initialize compression strategies with different trade-offs."""
        strategies = [
            CompressionStrategy(
                name="minimal_compression",
                compression_ratio=2.0,
                expected_quality=0.98,
                expected_latency_ms=20.0,
                memory_overhead_mb=50.0
            ),
            CompressionStrategy(
                name="low_compression", 
                compression_ratio=4.0,
                expected_quality=0.95,
                expected_latency_ms=30.0,
                memory_overhead_mb=40.0
            ),
            CompressionStrategy(
                name="medium_compression",
                compression_ratio=8.0,
                expected_quality=0.90,
                expected_latency_ms=50.0,
                memory_overhead_mb=30.0
            ),
            CompressionStrategy(
                name="high_compression",
                compression_ratio=16.0,
                expected_quality=0.82,
                expected_latency_ms=80.0,
                memory_overhead_mb=20.0
            ),
            CompressionStrategy(
                name="maximum_compression",
                compression_ratio=32.0,
                expected_quality=0.75,
                expected_latency_ms=120.0,
                memory_overhead_mb=15.0
            )
        ]
        
        return strategies
    
    def compress_embeddings(self, 
                           embeddings: torch.Tensor,
                           target_compression: Optional[float] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Adaptively compress embeddings based on current resource conditions.
        
        Args:
            embeddings: Input embeddings to compress
            target_compression: Optional specific compression ratio target
            
        Returns:
            Tuple of (compressed_embeddings, compression_metadata)
        """
        start_time = time.time()
        
        # Collect current resource metrics
        current_metrics = ResourceMetrics.collect_current_metrics()
        
        # Select optimal compression strategy
        if target_compression:
            strategy = self._select_strategy_by_ratio(target_compression)
        else:
            strategy = self._select_adaptive_strategy(current_metrics)
        
        # Apply compression
        compressed_embeddings, compression_stats = self._apply_compression(
            embeddings, strategy
        )
        
        # Calculate actual metrics
        compression_time = time.time() - start_time
        actual_compression_ratio = (embeddings.numel() * 4) / (compressed_embeddings.numel() * 4)
        memory_used = compressed_embeddings.numel() * 4 / 1024 / 1024  # MB
        
        # Update strategy performance
        quality_estimate = self._estimate_quality_loss(strategy.compression_ratio)
        strategy.update_performance(
            actual_quality=quality_estimate,
            actual_latency_ms=compression_time * 1000,
            actual_memory_mb=memory_used
        )
        
        # Prepare metadata
        metadata = {
            "strategy_name": strategy.name,
            "target_compression_ratio": strategy.compression_ratio,
            "actual_compression_ratio": actual_compression_ratio,
            "compression_time_ms": compression_time * 1000,
            "memory_used_mb": memory_used,
            "estimated_quality": quality_estimate,
            "resource_metrics": current_metrics.to_dict(),
            "original_shape": embeddings.shape,
            "compressed_shape": compressed_embeddings.shape
        }
        
        # Update history
        self.performance_history.append(metadata)
        self.resource_history.append(current_metrics)
        
        # Maintain history size
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        if len(self.resource_history) > 100:
            self.resource_history = self.resource_history[-100:]
        
        logger.debug(f"Compressed embeddings using {strategy.name}: "
                    f"{actual_compression_ratio:.2f}x in {compression_time*1000:.2f}ms")
        
        return compressed_embeddings, metadata
    
    def _select_adaptive_strategy(self, metrics: ResourceMetrics) -> CompressionStrategy:
        """Select optimal strategy based on current resource conditions."""
        
        # Calculate resource pressure (0-1, higher means more constrained)
        memory_pressure = min(metrics.memory_percent / 100.0, 1.0)
        cpu_pressure = min(metrics.cpu_percent / 100.0, 1.0)
        
        # GPU memory pressure if available
        gpu_pressure = 0.0
        if metrics.gpu_memory_total_mb > 0:
            gpu_pressure = min(metrics.gpu_memory_used_mb / metrics.gpu_memory_total_mb, 1.0)
        
        # Overall resource pressure
        resource_pressure = max(memory_pressure, cpu_pressure, gpu_pressure)
        
        # Score all strategies
        strategy_scores = []
        for strategy in self.strategies:
            score = strategy.get_score(resource_pressure, self.config.quality_weight)
            strategy_scores.append((strategy, score))
        
        # Sort by score (descending)
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select best strategy
        best_strategy = strategy_scores[0][0]
        
        # Check if we need to adapt due to constraints
        if resource_pressure > 0.8:
            # High resource pressure - favor compression
            high_compression_strategies = [s for s in self.strategies 
                                         if s.compression_ratio >= 16.0]
            if high_compression_strategies:
                best_strategy = max(high_compression_strategies, 
                                  key=lambda s: s.compression_ratio)
        
        elif resource_pressure < 0.3:
            # Low resource pressure - favor quality
            low_compression_strategies = [s for s in self.strategies 
                                        if s.compression_ratio <= 8.0]
            if low_compression_strategies:
                best_strategy = min(low_compression_strategies,
                                  key=lambda s: s.compression_ratio)
        
        self.current_strategy = best_strategy
        return best_strategy
    
    def _select_strategy_by_ratio(self, target_ratio: float) -> CompressionStrategy:
        """Select strategy closest to target compression ratio."""
        
        # Find strategy with closest compression ratio
        best_strategy = min(self.strategies, 
                          key=lambda s: abs(s.compression_ratio - target_ratio))
        
        return best_strategy
    
    def _apply_compression(self, 
                          embeddings: torch.Tensor,
                          strategy: CompressionStrategy) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply the selected compression strategy."""
        
        # Check cache if enabled
        cache_key = None
        if self.compression_cache is not None:
            cache_key = self._get_cache_key(embeddings, strategy.name)
            if cache_key in self.compression_cache:
                cached_result = self.compression_cache[cache_key]
                return cached_result['compressed'], cached_result['stats']
        
        # Apply compression based on strategy
        if strategy.compression_ratio <= 4.0:
            compressed = self._apply_quantization_compression(embeddings, bits=8)
        elif strategy.compression_ratio <= 8.0:
            compressed = self._apply_pca_compression(embeddings, ratio=0.5)
        elif strategy.compression_ratio <= 16.0:
            compressed = self._apply_combined_compression(embeddings, pca_ratio=0.25, bits=8)
        else:
            compressed = self._apply_aggressive_compression(embeddings)
        
        stats = {
            "method": strategy.name,
            "original_size": embeddings.numel() * 4,
            "compressed_size": compressed.numel() * 4,
            "compression_ratio": (embeddings.numel() * 4) / (compressed.numel() * 4)
        }
        
        # Cache result if enabled
        if self.compression_cache is not None and cache_key is not None:
            self.compression_cache[cache_key] = {
                'compressed': compressed,
                'stats': stats
            }
            
            # Limit cache size
            if len(self.compression_cache) > 1000:
                # Remove oldest entries
                oldest_keys = list(self.compression_cache.keys())[:100]
                for key in oldest_keys:
                    del self.compression_cache[key]
        
        return compressed, stats
    
    def _apply_quantization_compression(self, embeddings: torch.Tensor, bits: int = 8) -> torch.Tensor:
        """Apply quantization compression."""
        if bits == 8:
            # 8-bit quantization
            scale = 127.0 / torch.max(torch.abs(embeddings))
            quantized = torch.round(embeddings * scale).clamp(-127, 127)
            return quantized.to(torch.int8)
        else:
            # 4-bit quantization
            scale = 7.0 / torch.max(torch.abs(embeddings))
            quantized = torch.round(embeddings * scale).clamp(-7, 7)
            return quantized.to(torch.int8)
    
    def _apply_pca_compression(self, embeddings: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
        """Apply PCA-based dimensionality reduction."""
        original_shape = embeddings.shape
        
        # Flatten to 2D for PCA
        if embeddings.dim() > 2:
            embeddings_2d = embeddings.view(-1, embeddings.size(-1))
        else:
            embeddings_2d = embeddings
        
        # Compute PCA
        U, S, V = torch.svd(embeddings_2d.T)
        
        # Select top components
        n_components = int(embeddings_2d.size(1) * ratio)
        compressed = torch.matmul(embeddings_2d, U[:, :n_components])
        
        return compressed
    
    def _apply_combined_compression(self, 
                                  embeddings: torch.Tensor,
                                  pca_ratio: float = 0.5,
                                  bits: int = 8) -> torch.Tensor:
        """Apply combined PCA + quantization compression."""
        # First apply PCA
        pca_compressed = self._apply_pca_compression(embeddings, pca_ratio)
        
        # Then apply quantization
        quantized = self._apply_quantization_compression(pca_compressed, bits)
        
        return quantized
    
    def _apply_aggressive_compression(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply aggressive compression with multiple techniques."""
        # 1. PCA to 25% of dimensions
        pca_compressed = self._apply_pca_compression(embeddings, ratio=0.25)
        
        # 2. 4-bit quantization
        quantized = self._apply_quantization_compression(pca_compressed, bits=4)
        
        # 3. Sparse encoding (zero out small values)
        threshold = torch.quantile(torch.abs(quantized.float()), 0.9)
        sparse = torch.where(torch.abs(quantized.float()) > threshold, quantized, 0)
        
        return sparse
    
    def _estimate_quality_loss(self, compression_ratio: float) -> float:
        """Estimate quality retention based on compression ratio."""
        # Empirical quality loss function
        if compression_ratio <= 2.0:
            return 0.98
        elif compression_ratio <= 4.0:
            return 0.95
        elif compression_ratio <= 8.0:
            return 0.90
        elif compression_ratio <= 16.0:
            return 0.82
        else:
            return 0.75
    
    def _get_cache_key(self, embeddings: torch.Tensor, strategy_name: str) -> str:
        """Generate cache key for embeddings and strategy."""
        # Use hash of embeddings shape and a sample of values
        shape_str = str(embeddings.shape)
        sample_values = embeddings.flatten()[:10].tolist()
        content_str = str(sample_values)
        
        # Combine with strategy name
        key_content = f"{shape_str}_{content_str}_{strategy_name}"
        
        # Generate hash
        import hashlib
        return hashlib.md5(key_content.encode()).hexdigest()
    
    def get_adaptation_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for system adaptation."""
        if not self.resource_history:
            return {"status": "insufficient_data"}
        
        # Analyze recent resource trends
        recent_metrics = self.resource_history[-10:]  # Last 10 measurements
        
        avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
        avg_memory = np.mean([m.memory_percent for m in recent_metrics])
        
        recommendations = {
            "current_strategy": self.current_strategy.name if self.current_strategy else "none",
            "resource_pressure": {
                "cpu_average": avg_cpu,
                "memory_average": avg_memory,
                "status": "high" if max(avg_cpu, avg_memory) > 80 else 
                         "medium" if max(avg_cpu, avg_memory) > 60 else "low"
            },
            "recommendations": []
        }
        
        # Generate specific recommendations
        if avg_memory > 80:
            recommendations["recommendations"].append({
                "type": "memory_optimization",
                "action": "increase_compression_level",
                "reason": "High memory usage detected"
            })
        
        if avg_cpu > 80:
            recommendations["recommendations"].append({
                "type": "cpu_optimization", 
                "action": "reduce_compression_complexity",
                "reason": "High CPU usage detected"
            })
        
        if max(avg_cpu, avg_memory) < 30:
            recommendations["recommendations"].append({
                "type": "quality_optimization",
                "action": "reduce_compression_level", 
                "reason": "Resources available for better quality"
            })
        
        return recommendations
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        if not self.performance_history:
            return {"status": "no_data"}
        
        recent_performance = self.performance_history[-50:]  # Last 50 operations
        
        stats = {
            "operations_count": len(self.performance_history),
            "recent_operations": len(recent_performance),
            "average_compression_ratio": np.mean([p["actual_compression_ratio"] 
                                                for p in recent_performance]),
            "average_latency_ms": np.mean([p["compression_time_ms"] 
                                         for p in recent_performance]),
            "average_memory_mb": np.mean([p["memory_used_mb"] 
                                        for p in recent_performance]),
            "strategy_usage": {},
            "quality_metrics": {
                "average_estimated_quality": np.mean([p["estimated_quality"] 
                                                    for p in recent_performance])
            }
        }
        
        # Strategy usage statistics
        for strategy in self.strategies:
            strategy_ops = [p for p in recent_performance 
                          if p["strategy_name"] == strategy.name]
            stats["strategy_usage"][strategy.name] = {
                "count": len(strategy_ops),
                "percentage": len(strategy_ops) / len(recent_performance) * 100,
                "average_latency_ms": np.mean([p["compression_time_ms"] 
                                             for p in strategy_ops]) if strategy_ops else 0
            }
        
        return stats


def create_resource_aware_compressor(
    default_level: CompressionLevel = CompressionLevel.MEDIUM,
    enable_adaptive: bool = True,
    latency_target_ms: float = 100.0,
    memory_limit_mb: float = 2048.0
) -> ResourceAwareCompressor:
    """
    Factory function to create resource-aware compressor.
    
    Args:
        default_level: Default compression level
        enable_adaptive: Enable adaptive compression
        latency_target_ms: Target latency constraint
        memory_limit_mb: Memory usage limit
        
    Returns:
        Configured ResourceAwareCompressor
    """
    config = CompressionConfig(
        default_level=default_level,
        enable_adaptive=enable_adaptive,
        latency_target_ms=latency_target_ms,
        memory_limit_mb=memory_limit_mb
    )
    
    return ResourceAwareCompressor(config)
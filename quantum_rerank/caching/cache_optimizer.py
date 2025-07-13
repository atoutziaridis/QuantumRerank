"""
Cache performance optimizer for intelligent cache tuning.

This module provides optimization strategies to improve cache performance
based on usage patterns and performance metrics.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..utils import get_logger


class OptimizationStrategy(Enum):
    """Cache optimization strategies."""
    MEMORY_FOCUSED = "memory_focused"
    HIT_RATE_FOCUSED = "hit_rate_focused"
    LATENCY_FOCUSED = "latency_focused"
    BALANCED = "balanced"


@dataclass
class PerformanceAnalyzer:
    """Analyzer for cache performance patterns."""
    cache_type: str
    hit_rate_trend: str = "stable"  # "improving", "degrading", "stable"
    memory_trend: str = "stable"
    latency_trend: str = "stable"
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


class CacheOptimizer:
    """
    Intelligent cache optimizer for performance tuning.
    
    This optimizer analyzes cache performance and applies appropriate
    optimization strategies to improve overall system performance.
    """
    
    def __init__(
        self,
        target_memory_gb: float = 1.0,
        optimization_enabled: bool = True,
        optimization_interval_minutes: int = 15
    ):
        self.target_memory_gb = target_memory_gb
        self.optimization_enabled = optimization_enabled
        self.optimization_interval_minutes = optimization_interval_minutes
        
        self.logger = get_logger(__name__)
        
        # Optimization history
        self.optimization_history: List[Dict[str, Any]] = []
        self.last_optimization_time = 0.0
        
        # Performance baselines
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        
        self.logger.info(f"Initialized CacheOptimizer: target_memory={target_memory_gb}GB")
    
    def should_optimize(self) -> bool:
        """Check if optimization should be triggered."""
        if not self.optimization_enabled:
            return False
        
        current_time = time.time()
        time_since_last = current_time - self.last_optimization_time
        
        return time_since_last >= (self.optimization_interval_minutes * 60)
    
    def optimize_cache(
        self,
        cache: Any,
        cache_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize individual cache performance."""
        optimization_result = {
            "cache_type": getattr(cache, 'cache_type', 'unknown'),
            "optimizations_applied": [],
            "performance_impact": {}
        }
        
        try:
            # Analyze current performance
            analyzer = self._analyze_cache_performance(cache, cache_metrics)
            
            # Apply optimizations based on analysis
            if hasattr(cache, 'optimize'):
                cache_optimization = cache.optimize()
                optimization_result["cache_optimizations"] = cache_optimization
            
            # Record optimization
            self.optimization_history.append({
                "timestamp": time.time(),
                "cache_type": optimization_result["cache_type"],
                "result": optimization_result
            })
            
            # Trim history
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]
        
        except Exception as e:
            self.logger.error(f"Cache optimization failed: {e}")
            optimization_result["error"] = str(e)
        
        return optimization_result
    
    def optimize_global_strategy(
        self,
        caches: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize global caching strategy."""
        global_optimization = {
            "strategy_adjustments": [],
            "memory_reallocation": {},
            "recommendations": []
        }
        
        try:
            # Check overall memory usage
            total_memory_mb = sum(
                cache.get_memory_usage_mb() 
                for cache in caches.values()
                if hasattr(cache, 'get_memory_usage_mb')
            )
            
            target_memory_mb = self.target_memory_gb * 1024
            
            if total_memory_mb > target_memory_mb:
                # Need to reduce memory usage
                memory_reduction = self._optimize_memory_allocation(
                    caches, metrics, total_memory_mb, target_memory_mb
                )
                global_optimization["memory_reallocation"] = memory_reduction
            
            # Analyze cross-cache performance
            cross_cache_analysis = self._analyze_cross_cache_performance(caches, metrics)
            global_optimization["cross_cache_analysis"] = cross_cache_analysis
            
            self.last_optimization_time = time.time()
        
        except Exception as e:
            self.logger.error(f"Global optimization failed: {e}")
            global_optimization["error"] = str(e)
        
        return global_optimization
    
    def _analyze_cache_performance(
        self,
        cache: Any,
        metrics: Dict[str, Any]
    ) -> PerformanceAnalyzer:
        """Analyze individual cache performance."""
        cache_type = getattr(cache, 'cache_type', 'unknown')
        analyzer = PerformanceAnalyzer(cache_type=cache_type)
        
        # Analyze hit rate
        hit_rate = metrics.get("hit_rate", 0.0)
        if hit_rate < 0.1:
            analyzer.hit_rate_trend = "degrading"
            analyzer.recommendations.append("Consider increasing cache size")
        elif hit_rate > 0.3:
            analyzer.hit_rate_trend = "improving"
        
        # Analyze memory usage
        memory_usage = metrics.get("memory_usage_mb", 0.0)
        if memory_usage > 500:  # Arbitrary threshold
            analyzer.memory_trend = "degrading"
            analyzer.recommendations.append("Consider memory optimization")
        
        return analyzer
    
    def _optimize_memory_allocation(
        self,
        caches: Dict[str, Any],
        metrics: Dict[str, Any],
        current_memory_mb: float,
        target_memory_mb: float
    ) -> Dict[str, Any]:
        """Optimize memory allocation across caches."""
        reduction_needed = current_memory_mb - target_memory_mb
        
        # Calculate efficiency scores for each cache
        cache_efficiency = {}
        for cache_name, cache in caches.items():
            if hasattr(cache, 'get_memory_usage_mb'):
                memory_usage = cache.get_memory_usage_mb()
                hit_rate = metrics.get(f"{cache_name}_hit_rate", 0.0)
                
                # Efficiency = hit_rate / memory_usage
                efficiency = hit_rate / max(memory_usage, 1.0)
                cache_efficiency[cache_name] = efficiency
        
        # Reduce memory for least efficient caches
        memory_reallocation = {}
        remaining_reduction = reduction_needed
        
        # Sort by efficiency (least efficient first)
        sorted_caches = sorted(cache_efficiency.items(), key=lambda x: x[1])
        
        for cache_name, efficiency in sorted_caches:
            if remaining_reduction <= 0:
                break
            
            current_memory = caches[cache_name].get_memory_usage_mb()
            
            # Reduce by up to 20% of current memory
            max_reduction = current_memory * 0.2
            actual_reduction = min(max_reduction, remaining_reduction)
            
            if actual_reduction > 0:
                memory_reallocation[cache_name] = {
                    "current_mb": current_memory,
                    "reduction_mb": actual_reduction,
                    "efficiency_score": efficiency
                }
                remaining_reduction -= actual_reduction
        
        return memory_reallocation
    
    def _analyze_cross_cache_performance(
        self,
        caches: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze performance patterns across caches."""
        analysis = {
            "best_performing_cache": None,
            "worst_performing_cache": None,
            "overall_efficiency": 0.0,
            "recommendations": []
        }
        
        cache_scores = {}
        for cache_name in caches:
            hit_rate = metrics.get(f"{cache_name}_hit_rate", 0.0)
            memory_usage = metrics.get(f"{cache_name}_memory_usage_mb", 1.0)
            
            # Score = hit_rate / normalized_memory_usage
            normalized_memory = memory_usage / 100.0  # Normalize to 0-10 range
            score = hit_rate / max(normalized_memory, 0.1)
            cache_scores[cache_name] = score
        
        if cache_scores:
            best_cache = max(cache_scores.items(), key=lambda x: x[1])
            worst_cache = min(cache_scores.items(), key=lambda x: x[1])
            
            analysis["best_performing_cache"] = {
                "name": best_cache[0],
                "score": best_cache[1]
            }
            analysis["worst_performing_cache"] = {
                "name": worst_cache[0],
                "score": worst_cache[1]
            }
            
            analysis["overall_efficiency"] = sum(cache_scores.values()) / len(cache_scores)
            
            # Generate recommendations
            if worst_cache[1] < best_cache[1] * 0.5:
                analysis["recommendations"].append(
                    f"Consider optimizing {worst_cache[0]} cache configuration"
                )
        
        return analysis
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization performance statistics."""
        return {
            "total_optimizations": len(self.optimization_history),
            "optimization_enabled": self.optimization_enabled,
            "target_memory_gb": self.target_memory_gb,
            "last_optimization": self.last_optimization_time,
            "recent_optimizations": self.optimization_history[-10:] if self.optimization_history else []
        }


__all__ = [
    "OptimizationStrategy",
    "PerformanceAnalyzer",
    "CacheOptimizer"
]
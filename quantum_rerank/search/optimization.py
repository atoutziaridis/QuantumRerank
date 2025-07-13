"""
Pipeline performance optimization engine for intelligent tuning.

This module provides comprehensive optimization strategies for search and
reranking pipelines, including dynamic parameter tuning and performance analysis.
"""

import time
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .performance_monitor import SearchPerformanceMonitor, SearchPerformanceTargets
from ..utils import get_logger


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    LATENCY_FOCUSED = "latency_focused"
    THROUGHPUT_FOCUSED = "throughput_focused"
    ACCURACY_FOCUSED = "accuracy_focused"
    BALANCED = "balanced"
    RESOURCE_EFFICIENT = "resource_efficient"


@dataclass
class OptimizationGoals:
    """Optimization goals and constraints."""
    target_latency_ms: Optional[float] = None
    target_throughput_qps: Optional[float] = None
    target_accuracy: Optional[float] = None
    max_memory_mb: Optional[float] = None
    max_cpu_utilization: Optional[float] = None
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED


@dataclass
class OptimizationResult:
    """Result of optimization analysis."""
    strategy: OptimizationStrategy
    recommendations: List[str]
    parameter_changes: Dict[str, Any]
    expected_improvements: Dict[str, float]
    confidence_score: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class PerformanceProfile:
    """Performance profile for different configurations."""
    configuration: Dict[str, Any]
    latency_ms: float
    throughput_qps: float
    accuracy_score: float
    memory_usage_mb: float
    stability_score: float
    sample_count: int = 1


class PipelineOptimizer:
    """
    Intelligent pipeline optimizer for search and reranking performance.
    
    This optimizer analyzes performance patterns and automatically suggests
    or applies optimizations to improve pipeline efficiency.
    """
    
    def __init__(self, performance_monitor: SearchPerformanceMonitor,
                 auto_optimize: bool = False):
        self.performance_monitor = performance_monitor
        self.auto_optimize = auto_optimize
        self.logger = get_logger(__name__)
        
        # Optimization state
        self.performance_profiles: List[PerformanceProfile] = []
        self.optimization_history: List[OptimizationResult] = []
        self.current_configuration: Dict[str, Any] = {}
        
        # Optimization knowledge base
        self.optimization_rules = self._initialize_optimization_rules()
        self.parameter_ranges = self._initialize_parameter_ranges()
        
        # Learning and adaptation
        self.adaptation_enabled = True
        self.learning_rate = 0.1
        self.exploration_factor = 0.2
        
        self.logger.info(f"Initialized PipelineOptimizer (auto_optimize={auto_optimize})")
    
    def analyze_performance(self, time_window_minutes: int = 30) -> Dict[str, Any]:
        """
        Analyze current performance and identify optimization opportunities.
        
        Args:
            time_window_minutes: Time window for performance analysis
            
        Returns:
            Performance analysis with optimization recommendations
        """
        # Get performance report
        performance_report = self.performance_monitor.get_performance_report(time_window_minutes)
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(performance_report)
        
        # Analyze trends
        trend_analysis = self._analyze_performance_trends(performance_report)
        
        # Generate optimization opportunities
        opportunities = self._identify_optimization_opportunities(
            performance_report, bottlenecks, trend_analysis
        )
        
        analysis = {
            "performance_summary": self._summarize_performance(performance_report),
            "bottlenecks": bottlenecks,
            "trends": trend_analysis,
            "optimization_opportunities": opportunities,
            "urgency_score": self._calculate_urgency_score(bottlenecks),
            "recommended_actions": self._generate_recommended_actions(opportunities)
        }
        
        return analysis
    
    def optimize_pipeline(self, goals: OptimizationGoals) -> OptimizationResult:
        """
        Optimize pipeline based on specified goals.
        
        Args:
            goals: Optimization goals and constraints
            
        Returns:
            Optimization result with recommendations
        """
        try:
            # Analyze current state
            performance_analysis = self.analyze_performance()
            
            # Select optimization strategy
            strategy = self._select_optimization_strategy(goals, performance_analysis)
            
            # Generate optimizations based on strategy
            optimizations = self._generate_optimizations(strategy, goals, performance_analysis)
            
            # Evaluate optimization potential
            evaluation = self._evaluate_optimizations(optimizations, goals)
            
            # Create optimization result
            result = OptimizationResult(
                strategy=strategy,
                recommendations=optimizations["recommendations"],
                parameter_changes=optimizations["parameter_changes"],
                expected_improvements=evaluation["expected_improvements"],
                confidence_score=evaluation["confidence_score"]
            )
            
            # Record optimization
            self.optimization_history.append(result)
            
            # Apply optimizations if auto-optimize is enabled
            if self.auto_optimize and result.confidence_score > 0.7:
                self._apply_optimizations(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline optimization failed: {e}")
            return OptimizationResult(
                strategy=goals.strategy,
                recommendations=[f"Optimization failed: {e}"],
                parameter_changes={},
                expected_improvements={},
                confidence_score=0.0
            )
    
    def optimize_retrieval_size(self, current_performance: Dict[str, Any],
                               target_latency_ms: float,
                               target_accuracy: float) -> Dict[str, Any]:
        """
        Optimize retrieval size for best performance balance.
        
        Args:
            current_performance: Current performance metrics
            target_latency_ms: Target latency constraint
            target_accuracy: Target accuracy requirement
            
        Returns:
            Retrieval size optimization recommendations
        """
        # Get current retrieval statistics
        current_latency = current_performance.get("total_latency_ms", {}).get("average", 600)
        current_retrieval_size = current_performance.get("retrieval_result_count", {}).get("average", 100)
        
        # Model performance vs retrieval size relationship
        optimal_size = self._calculate_optimal_retrieval_size(
            current_retrieval_size, current_latency, target_latency_ms, target_accuracy
        )
        
        # Calculate expected improvements
        expected_latency_improvement = self._estimate_latency_improvement(
            current_retrieval_size, optimal_size, current_latency
        )
        
        return {
            "current_retrieval_size": current_retrieval_size,
            "optimal_retrieval_size": optimal_size,
            "expected_latency_reduction_ms": expected_latency_improvement,
            "confidence": self._calculate_retrieval_optimization_confidence(current_performance)
        }
    
    def optimize_backend_selection(self, collection_characteristics: Dict[str, Any],
                                 performance_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize backend selection based on workload characteristics.
        
        Args:
            collection_characteristics: Dataset characteristics
            performance_requirements: Performance requirements
            
        Returns:
            Backend selection optimization
        """
        # Analyze collection characteristics
        collection_size = collection_characteristics.get("size", 100000)
        dimension = collection_characteristics.get("dimension", 512)
        query_patterns = collection_characteristics.get("query_patterns", {})
        
        # Score different backends
        backend_scores = self._score_backends_for_workload(
            collection_size, dimension, query_patterns, performance_requirements
        )
        
        # Select optimal backend
        optimal_backend = max(backend_scores.items(), key=lambda x: x[1]["total_score"])
        
        return {
            "recommended_backend": optimal_backend[0],
            "backend_scores": backend_scores,
            "expected_improvements": optimal_backend[1]["expected_improvements"],
            "migration_complexity": optimal_backend[1]["migration_complexity"]
        }
    
    def optimize_caching_strategy(self, cache_performance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize caching strategy based on performance data.
        
        Args:
            cache_performance: Current cache performance metrics
            
        Returns:
            Caching optimization recommendations
        """
        optimizations = []
        
        # Analyze cache hit rates
        for cache_type, metrics in cache_performance.items():
            hit_rate = metrics.get("hit_rate", 0.0)
            memory_usage = metrics.get("memory_usage_mb", 0.0)
            
            if hit_rate < 0.3:  # Low hit rate
                optimizations.append({
                    "cache_type": cache_type,
                    "issue": "low_hit_rate",
                    "current_hit_rate": hit_rate,
                    "recommendation": "increase_cache_size",
                    "expected_improvement": min(0.6, hit_rate * 2)
                })
            
            if memory_usage > 1024:  # High memory usage
                optimizations.append({
                    "cache_type": cache_type,
                    "issue": "high_memory_usage",
                    "current_memory_mb": memory_usage,
                    "recommendation": "optimize_cache_policy",
                    "expected_memory_reduction": memory_usage * 0.2
                })
        
        return {
            "cache_optimizations": optimizations,
            "overall_strategy": self._recommend_overall_cache_strategy(cache_performance)
        }
    
    def _identify_bottlenecks(self, performance_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks from metrics."""
        bottlenecks = []
        
        metrics = performance_report.get("performance_metrics", {})
        targets = self.performance_monitor.targets
        
        # Check retrieval latency
        if "retrieval_latency_ms" in metrics:
            avg_latency = metrics["retrieval_latency_ms"].get("average", 0)
            if avg_latency > targets.retrieval_latency_ms:
                bottlenecks.append({
                    "type": "retrieval_latency",
                    "severity": "high" if avg_latency > targets.retrieval_latency_ms * 2 else "medium",
                    "current_value": avg_latency,
                    "target_value": targets.retrieval_latency_ms,
                    "impact_score": min(avg_latency / targets.retrieval_latency_ms, 5.0)
                })
        
        # Check rerank latency
        if "rerank_latency_ms" in metrics:
            avg_latency = metrics["rerank_latency_ms"].get("average", 0)
            if avg_latency > targets.rerank_latency_ms:
                bottlenecks.append({
                    "type": "rerank_latency", 
                    "severity": "high" if avg_latency > targets.rerank_latency_ms * 2 else "medium",
                    "current_value": avg_latency,
                    "target_value": targets.rerank_latency_ms,
                    "impact_score": min(avg_latency / targets.rerank_latency_ms, 5.0)
                })
        
        # Check memory usage
        if "memory_usage_mb" in metrics:
            avg_memory = metrics["memory_usage_mb"].get("average", 0)
            if avg_memory > targets.memory_usage_mb:
                bottlenecks.append({
                    "type": "memory_usage",
                    "severity": "high" if avg_memory > targets.memory_usage_mb * 1.5 else "medium",
                    "current_value": avg_memory,
                    "target_value": targets.memory_usage_mb,
                    "impact_score": min(avg_memory / targets.memory_usage_mb, 3.0)
                })
        
        # Sort by impact score
        bottlenecks.sort(key=lambda x: x["impact_score"], reverse=True)
        
        return bottlenecks
    
    def _analyze_performance_trends(self, performance_report: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        trends = performance_report.get("trends", {})
        
        trend_analysis = {
            "degrading_metrics": [],
            "improving_metrics": [],
            "stable_metrics": [],
            "volatility_issues": []
        }
        
        for metric_name, trend in trends.items():
            if trend == "increasing" and "latency" in metric_name:
                trend_analysis["degrading_metrics"].append(metric_name)
            elif trend == "decreasing" and "latency" in metric_name:
                trend_analysis["improving_metrics"].append(metric_name)
            elif trend == "increasing" and "throughput" in metric_name:
                trend_analysis["improving_metrics"].append(metric_name)
            elif trend == "decreasing" and "throughput" in metric_name:
                trend_analysis["degrading_metrics"].append(metric_name)
            else:
                trend_analysis["stable_metrics"].append(metric_name)
        
        return trend_analysis
    
    def _identify_optimization_opportunities(self, performance_report: Dict[str, Any],
                                           bottlenecks: List[Dict[str, Any]],
                                           trends: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific optimization opportunities."""
        opportunities = []
        
        # Opportunities from bottlenecks
        for bottleneck in bottlenecks[:3]:  # Top 3 bottlenecks
            if bottleneck["type"] == "retrieval_latency":
                opportunities.append({
                    "type": "optimize_retrieval_size",
                    "priority": "high",
                    "description": "Reduce retrieval size to improve latency",
                    "expected_impact": "20-40% latency reduction"
                })
                
                opportunities.append({
                    "type": "optimize_backend_selection",
                    "priority": "medium", 
                    "description": "Consider faster vector search backend",
                    "expected_impact": "30-60% latency reduction"
                })
            
            elif bottleneck["type"] == "rerank_latency":
                opportunities.append({
                    "type": "optimize_rerank_method",
                    "priority": "high",
                    "description": "Use faster similarity computation method",
                    "expected_impact": "25-50% rerank latency reduction"
                })
            
            elif bottleneck["type"] == "memory_usage":
                opportunities.append({
                    "type": "optimize_memory_usage",
                    "priority": "medium",
                    "description": "Reduce index size or enable compression",
                    "expected_impact": "15-30% memory reduction"
                })
        
        # Opportunities from trends
        if "retrieval_latency_ms" in trends.get("degrading_metrics", []):
            opportunities.append({
                "type": "investigate_latency_degradation",
                "priority": "high",
                "description": "Investigate cause of increasing retrieval latency",
                "expected_impact": "Prevent further degradation"
            })
        
        return opportunities
    
    def _calculate_optimal_retrieval_size(self, current_size: float, current_latency: float,
                                        target_latency: float, target_accuracy: float) -> int:
        """Calculate optimal retrieval size using performance modeling."""
        # Simple model: latency scales roughly linearly with retrieval size
        # Accuracy improves with retrieval size but with diminishing returns
        
        if target_latency >= current_latency:
            # Can maintain or increase size
            optimal_size = current_size
        else:
            # Need to reduce size for latency target
            latency_reduction_needed = (current_latency - target_latency) / current_latency
            size_reduction = latency_reduction_needed * 0.8  # Conservative estimate
            optimal_size = current_size * (1 - size_reduction)
        
        # Ensure minimum size for accuracy
        min_size_for_accuracy = max(50, target_accuracy * 100)
        optimal_size = max(optimal_size, min_size_for_accuracy)
        
        return int(optimal_size)
    
    def _estimate_latency_improvement(self, current_size: float, optimal_size: float,
                                    current_latency: float) -> float:
        """Estimate latency improvement from retrieval size change."""
        if optimal_size >= current_size:
            return 0.0  # No improvement expected
        
        size_reduction_ratio = (current_size - optimal_size) / current_size
        
        # Estimate latency improvement (conservative)
        # Assume 70% of retrieval size reduction translates to latency improvement
        latency_improvement_ratio = size_reduction_ratio * 0.7
        
        return current_latency * latency_improvement_ratio
    
    def _score_backends_for_workload(self, collection_size: int, dimension: int,
                                   query_patterns: Dict[str, Any],
                                   requirements: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Score different backends for specific workload characteristics."""
        backends = {
            "faiss_exact": {
                "latency_score": 0.9 if collection_size < 100000 else 0.3,
                "accuracy_score": 1.0,
                "memory_score": 0.7,
                "scalability_score": 0.4
            },
            "faiss_ivf": {
                "latency_score": 0.8,
                "accuracy_score": 0.95,
                "memory_score": 0.8,
                "scalability_score": 0.8
            },
            "faiss_hnsw": {
                "latency_score": 0.9,
                "accuracy_score": 0.93,
                "memory_score": 0.6,
                "scalability_score": 0.9
            }
        }
        
        # Calculate weighted scores based on requirements
        weighted_scores = {}
        weights = {
            "latency_score": requirements.get("latency_weight", 0.3),
            "accuracy_score": requirements.get("accuracy_weight", 0.3),
            "memory_score": requirements.get("memory_weight", 0.2),
            "scalability_score": requirements.get("scalability_weight", 0.2)
        }
        
        for backend_name, scores in backends.items():
            total_score = sum(scores[metric] * weights[metric] for metric in scores)
            
            weighted_scores[backend_name] = {
                "total_score": total_score,
                "individual_scores": scores,
                "expected_improvements": {
                    "latency_improvement": (scores["latency_score"] - 0.5) * 100,
                    "accuracy_improvement": (scores["accuracy_score"] - 0.9) * 100
                },
                "migration_complexity": "low" if backend_name.startswith("faiss") else "medium"
            }
        
        return weighted_scores
    
    def _generate_optimizations(self, strategy: OptimizationStrategy,
                              goals: OptimizationGoals,
                              analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific optimizations based on strategy and analysis."""
        recommendations = []
        parameter_changes = {}
        
        bottlenecks = analysis.get("bottlenecks", [])
        opportunities = analysis.get("optimization_opportunities", [])
        
        if strategy == OptimizationStrategy.LATENCY_FOCUSED:
            # Focus on reducing latency
            recommendations.extend([
                "Reduce retrieval size to minimum required for accuracy",
                "Switch to fastest available backend (HNSW or exact)",
                "Enable aggressive caching for repeated queries"
            ])
            
            if bottlenecks and bottlenecks[0]["type"] == "retrieval_latency":
                parameter_changes["retrieval_multiplier"] = 2.0  # Reduce from default 3.0
            
        elif strategy == OptimizationStrategy.THROUGHPUT_FOCUSED:
            # Focus on maximizing throughput
            recommendations.extend([
                "Enable batch processing for multiple queries",
                "Optimize backend parameters for throughput",
                "Implement connection pooling and parallel processing"
            ])
            
            parameter_changes["batch_size"] = 32
            parameter_changes["parallel_queries"] = 4
            
        elif strategy == OptimizationStrategy.ACCURACY_FOCUSED:
            # Focus on maximizing accuracy
            recommendations.extend([
                "Increase retrieval size for better candidate coverage",
                "Use exact search backend when possible",
                "Enable hybrid reranking methods"
            ])
            
            parameter_changes["retrieval_multiplier"] = 5.0
            parameter_changes["rerank_method"] = "hybrid"
            
        elif strategy == OptimizationStrategy.RESOURCE_EFFICIENT:
            # Focus on minimizing resource usage
            recommendations.extend([
                "Enable memory-efficient index formats",
                "Use compressed embeddings where possible",
                "Implement intelligent cache eviction policies"
            ])
            
            parameter_changes["enable_compression"] = True
            parameter_changes["cache_policy"] = "lru_with_frequency"
        
        else:  # BALANCED
            # Balanced optimization
            recommendations.extend([
                "Optimize retrieval size for latency-accuracy balance",
                "Use IVF index for balanced performance",
                "Enable moderate caching with performance monitoring"
            ])
            
            parameter_changes["retrieval_multiplier"] = 3.0
            parameter_changes["backend_type"] = "faiss_ivf"
        
        return {
            "recommendations": recommendations,
            "parameter_changes": parameter_changes
        }
    
    def _evaluate_optimizations(self, optimizations: Dict[str, Any],
                              goals: OptimizationGoals) -> Dict[str, Any]:
        """Evaluate potential impact of optimizations."""
        # Simplified evaluation based on historical data and models
        expected_improvements = {}
        confidence_factors = []
        
        parameter_changes = optimizations.get("parameter_changes", {})
        
        # Estimate improvements from parameter changes
        if "retrieval_multiplier" in parameter_changes:
            new_multiplier = parameter_changes["retrieval_multiplier"]
            current_multiplier = self.current_configuration.get("retrieval_multiplier", 3.0)
            
            if new_multiplier < current_multiplier:
                # Reducing retrieval size should improve latency
                improvement_ratio = (current_multiplier - new_multiplier) / current_multiplier
                expected_improvements["latency_improvement_percent"] = improvement_ratio * 30
                confidence_factors.append(0.8)
        
        if "backend_type" in parameter_changes:
            # Backend change can have significant impact
            expected_improvements["latency_improvement_percent"] = expected_improvements.get("latency_improvement_percent", 0) + 25
            expected_improvements["accuracy_change_percent"] = -2  # Slight accuracy trade-off
            confidence_factors.append(0.7)
        
        if "enable_compression" in parameter_changes:
            expected_improvements["memory_reduction_percent"] = 20
            confidence_factors.append(0.9)
        
        # Calculate overall confidence
        confidence_score = np.mean(confidence_factors) if confidence_factors else 0.5
        
        return {
            "expected_improvements": expected_improvements,
            "confidence_score": confidence_score
        }
    
    def _select_optimization_strategy(self, goals: OptimizationGoals,
                                    analysis: Dict[str, Any]) -> OptimizationStrategy:
        """Select appropriate optimization strategy based on goals and current state."""
        if goals.strategy != OptimizationStrategy.BALANCED:
            return goals.strategy
        
        # Auto-select based on bottlenecks and goals
        bottlenecks = analysis.get("bottlenecks", [])
        
        if goals.target_latency_ms and goals.target_latency_ms < 300:
            return OptimizationStrategy.LATENCY_FOCUSED
        
        if goals.target_throughput_qps and goals.target_throughput_qps > 100:
            return OptimizationStrategy.THROUGHPUT_FOCUSED
        
        if goals.target_accuracy and goals.target_accuracy > 0.98:
            return OptimizationStrategy.ACCURACY_FOCUSED
        
        if goals.max_memory_mb and goals.max_memory_mb < 1024:
            return OptimizationStrategy.RESOURCE_EFFICIENT
        
        # Default to balanced if no clear preference
        return OptimizationStrategy.BALANCED
    
    def _apply_optimizations(self, result: OptimizationResult) -> None:
        """Apply optimization recommendations (placeholder for actual implementation)."""
        self.logger.info(f"Applying optimizations: {result.parameter_changes}")
        
        # Update current configuration
        self.current_configuration.update(result.parameter_changes)
        
        # In a real implementation, this would:
        # 1. Update backend configurations
        # 2. Adjust retrieval parameters
        # 3. Modify caching settings
        # 4. Restart components if necessary
    
    def _initialize_optimization_rules(self) -> Dict[str, Any]:
        """Initialize optimization rule knowledge base."""
        return {
            "latency_rules": [
                {"condition": "retrieval_latency_high", "action": "reduce_retrieval_size"},
                {"condition": "rerank_latency_high", "action": "use_faster_method"},
                {"condition": "total_latency_high", "action": "optimize_pipeline"}
            ],
            "throughput_rules": [
                {"condition": "low_throughput", "action": "enable_batching"},
                {"condition": "high_concurrency", "action": "scale_resources"}
            ],
            "accuracy_rules": [
                {"condition": "low_accuracy", "action": "increase_retrieval_size"},
                {"condition": "accuracy_degradation", "action": "review_backend"}
            ]
        }
    
    def _initialize_parameter_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Initialize valid parameter ranges for optimization."""
        return {
            "retrieval_multiplier": {"min": 1.5, "max": 10.0, "default": 3.0},
            "batch_size": {"min": 1, "max": 128, "default": 32},
            "cache_size_mb": {"min": 64, "max": 4096, "default": 512},
            "parallel_queries": {"min": 1, "max": 16, "default": 1}
        }
    
    def _summarize_performance(self, performance_report: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of current performance state."""
        metrics = performance_report.get("performance_metrics", {})
        
        summary = {
            "overall_health": "unknown",
            "key_metrics": {},
            "meets_targets": {}
        }
        
        # Extract key metrics
        if "total_latency_ms" in metrics:
            summary["key_metrics"]["average_latency_ms"] = metrics["total_latency_ms"].get("average", 0)
        
        if "total_throughput_qps" in metrics:
            summary["key_metrics"]["average_throughput_qps"] = metrics["total_throughput_qps"].get("average", 0)
        
        # Check target compliance
        compliance = performance_report.get("target_compliance", {})
        meets_all_targets = all(
            target_info.get("meets_target", False) 
            for target_info in compliance.values()
        )
        
        summary["overall_health"] = "good" if meets_all_targets else "needs_attention"
        summary["meets_targets"] = {
            key: info.get("meets_target", False) 
            for key, info in compliance.items()
        }
        
        return summary
    
    def _calculate_urgency_score(self, bottlenecks: List[Dict[str, Any]]) -> float:
        """Calculate urgency score for optimization (0-1 scale)."""
        if not bottlenecks:
            return 0.0
        
        # Weight by severity and impact
        total_score = 0.0
        for bottleneck in bottlenecks:
            severity_weight = 0.8 if bottleneck["severity"] == "high" else 0.4
            impact_weight = min(bottleneck["impact_score"] / 5.0, 1.0)
            total_score += severity_weight * impact_weight
        
        return min(total_score / len(bottlenecks), 1.0)
    
    def _generate_recommended_actions(self, opportunities: List[Dict[str, Any]]) -> List[str]:
        """Generate prioritized list of recommended actions."""
        # Sort opportunities by priority
        priority_order = {"high": 3, "medium": 2, "low": 1}
        sorted_opportunities = sorted(
            opportunities,
            key=lambda x: priority_order.get(x.get("priority", "low"), 1),
            reverse=True
        )
        
        # Generate action list
        actions = []
        for opp in sorted_opportunities[:5]:  # Top 5 opportunities
            actions.append(f"[{opp.get('priority', 'medium').upper()}] {opp.get('description', 'Unknown action')}")
        
        return actions
    
    def _calculate_retrieval_optimization_confidence(self, performance_data: Dict[str, Any]) -> float:
        """Calculate confidence in retrieval size optimization."""
        # Base confidence on data quality and consistency
        sample_count = performance_data.get("query_statistics", {}).get("total_queries", 0)
        
        if sample_count < 10:
            return 0.3  # Low confidence with little data
        elif sample_count < 100:
            return 0.6  # Medium confidence
        else:
            return 0.8  # High confidence with sufficient data
    
    def _recommend_overall_cache_strategy(self, cache_performance: Dict[str, Any]) -> str:
        """Recommend overall caching strategy based on performance."""
        avg_hit_rate = np.mean([
            metrics.get("hit_rate", 0.0) 
            for metrics in cache_performance.values()
        ])
        
        if avg_hit_rate < 0.3:
            return "aggressive_caching"
        elif avg_hit_rate > 0.7:
            return "conservative_caching" 
        else:
            return "balanced_caching"


__all__ = [
    "OptimizationStrategy",
    "OptimizationGoals",
    "OptimizationResult", 
    "PerformanceProfile",
    "PipelineOptimizer"
]
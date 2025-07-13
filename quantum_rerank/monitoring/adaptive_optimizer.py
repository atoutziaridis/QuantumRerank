"""
Adaptive performance optimization engine for real-time system tuning.

This module provides intelligent optimization strategies that automatically
adjust system parameters based on performance monitoring data.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import numpy as np

from .metrics_collector import MetricsCollector
from ..utils import get_logger


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    LATENCY_OPTIMIZATION = "latency_optimization"
    THROUGHPUT_OPTIMIZATION = "throughput_optimization"
    ACCURACY_OPTIMIZATION = "accuracy_optimization"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    BALANCED_OPTIMIZATION = "balanced_optimization"


class OptimizationPriority(Enum):
    """Priority levels for optimizations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class OptimizationTarget:
    """Target specification for optimization."""
    metric_name: str
    target_value: float
    current_value: float
    importance_weight: float = 1.0
    optimization_direction: str = "minimize"  # minimize or maximize
    unit: str = ""


@dataclass
class OptimizationAction:
    """Specific optimization action to be performed."""
    action_type: str
    component: str
    parameter: str
    current_value: Any
    recommended_value: Any
    expected_improvement: float
    confidence_score: float
    priority: OptimizationPriority
    description: str
    estimated_impact: Dict[str, float] = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)
    rollback_plan: Optional[str] = None


@dataclass
class OptimizationResult:
    """Result of an optimization execution."""
    action: OptimizationAction
    executed: bool
    execution_time_ms: float
    success: bool
    actual_improvement: Optional[float] = None
    side_effects: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class AdaptivePerformanceOptimizer:
    """
    Real-time performance optimization engine.
    
    Automatically analyzes performance metrics and applies optimization
    strategies to improve system performance across multiple dimensions.
    """
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None,
                 optimization_interval_seconds: int = 60,
                 enable_auto_optimization: bool = False):
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.optimization_interval_seconds = optimization_interval_seconds
        self.enable_auto_optimization = enable_auto_optimization
        self.logger = get_logger(__name__)
        
        # Optimization state
        self.optimization_targets = self._initialize_optimization_targets()
        self.optimization_strategies = self._initialize_optimization_strategies()
        self.optimization_history: deque = deque(maxlen=1000)
        
        # Performance models
        self.performance_models = {}
        self.baseline_metrics = {}
        
        # Auto-optimization
        self._auto_optimizer_thread = None
        self._auto_optimization_active = False
        
        # Optimization rules
        self.optimization_rules = self._initialize_optimization_rules()
        
        # Safety constraints
        self.safety_constraints = self._initialize_safety_constraints()
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info("Initialized AdaptivePerformanceOptimizer")
    
    def start_auto_optimization(self) -> None:
        """Start automatic optimization loop."""
        if self._auto_optimization_active or not self.enable_auto_optimization:
            return
        
        self._auto_optimization_active = True
        self._auto_optimizer_thread = threading.Thread(
            target=self._auto_optimization_loop, daemon=True
        )
        self._auto_optimizer_thread.start()
        
        self.logger.info("Started automatic performance optimization")
    
    def stop_auto_optimization(self) -> None:
        """Stop automatic optimization loop."""
        self._auto_optimization_active = False
        if self._auto_optimizer_thread:
            self._auto_optimizer_thread.join(timeout=10.0)
        
        self.logger.info("Stopped automatic performance optimization")
    
    def analyze_performance_and_optimize(self, time_window_seconds: int = 300) -> List[OptimizationAction]:
        """
        Analyze current performance and generate optimization recommendations.
        
        Args:
            time_window_seconds: Time window for performance analysis
            
        Returns:
            List of recommended optimization actions
        """
        with self._lock:
            # Update current performance metrics
            self._update_current_metrics(time_window_seconds)
            
            # Identify performance gaps
            performance_gaps = self._identify_performance_gaps()
            
            # Generate optimization actions
            optimization_actions = []
            
            for gap in performance_gaps:
                actions = self._generate_optimization_actions(gap)
                optimization_actions.extend(actions)
            
            # Prioritize and filter actions
            optimization_actions = self._prioritize_optimization_actions(optimization_actions)
            
            # Apply safety constraints
            optimization_actions = self._apply_safety_constraints(optimization_actions)
            
            self.logger.info(f"Generated {len(optimization_actions)} optimization recommendations")
            
            return optimization_actions
    
    def execute_optimization_action(self, action: OptimizationAction) -> OptimizationResult:
        """
        Execute a specific optimization action.
        
        Args:
            action: Optimization action to execute
            
        Returns:
            Result of the optimization execution
        """
        start_time = time.time()
        
        try:
            # Validate action
            if not self._validate_optimization_action(action):
                return OptimizationResult(
                    action=action,
                    executed=False,
                    execution_time_ms=0.0,
                    success=False,
                    error_message="Action validation failed"
                )
            
            # Get baseline metrics
            baseline_metrics = self._capture_baseline_metrics(action)
            
            # Execute the optimization
            success = self._apply_optimization_action(action)
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            if success:
                # Measure impact
                time.sleep(5.0)  # Allow time for metrics to reflect changes
                post_optimization_metrics = self._capture_baseline_metrics(action)
                
                actual_improvement = self._calculate_actual_improvement(
                    action, baseline_metrics, post_optimization_metrics
                )
                
                side_effects = self._detect_side_effects(
                    baseline_metrics, post_optimization_metrics
                )
                
                result = OptimizationResult(
                    action=action,
                    executed=True,
                    execution_time_ms=execution_time_ms,
                    success=True,
                    actual_improvement=actual_improvement,
                    side_effects=side_effects
                )
                
                self.logger.info(
                    f"Successfully executed optimization: {action.description} "
                    f"(improvement: {actual_improvement:.3f})"
                )
            else:
                result = OptimizationResult(
                    action=action,
                    executed=False,
                    execution_time_ms=execution_time_ms,
                    success=False,
                    error_message="Optimization execution failed"
                )
            
            # Record result
            self.optimization_history.append(result)
            
            return result
        
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            
            result = OptimizationResult(
                action=action,
                executed=False,
                execution_time_ms=execution_time_ms,
                success=False,
                error_message=str(e)
            )
            
            self.optimization_history.append(result)
            self.logger.error(f"Optimization execution failed: {e}")
            
            return result
    
    def get_optimization_recommendations(self, strategy: OptimizationStrategy,
                                       max_recommendations: int = 5) -> List[OptimizationAction]:
        """Get optimization recommendations for specific strategy."""
        # Analyze current performance
        actions = self.analyze_performance_and_optimize()
        
        # Filter by strategy
        strategy_actions = []
        for action in actions:
            if self._action_matches_strategy(action, strategy):
                strategy_actions.append(action)
        
        # Sort by priority and confidence
        strategy_actions.sort(
            key=lambda a: (a.priority.value, -a.confidence_score, -a.expected_improvement)
        )
        
        return strategy_actions[:max_recommendations]
    
    def get_optimization_history(self, limit: int = 50) -> List[OptimizationResult]:
        """Get recent optimization history."""
        with self._lock:
            return list(self.optimization_history)[-limit:]
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        with self._lock:
            if not self.optimization_history:
                return {"total_optimizations": 0}
            
            successful_optimizations = [r for r in self.optimization_history if r.success]
            
            stats = {
                "total_optimizations": len(self.optimization_history),
                "successful_optimizations": len(successful_optimizations),
                "success_rate": len(successful_optimizations) / len(self.optimization_history),
                "average_improvement": np.mean([
                    r.actual_improvement for r in successful_optimizations
                    if r.actual_improvement is not None
                ]) if successful_optimizations else 0.0,
                "total_execution_time_ms": sum(r.execution_time_ms for r in self.optimization_history),
                "optimization_types": self._get_optimization_type_stats(),
                "recent_optimizations": len([
                    r for r in self.optimization_history
                    if r.timestamp > time.time() - 3600  # Last hour
                ])
            }
            
            return stats
    
    def _auto_optimization_loop(self) -> None:
        """Main auto-optimization loop."""
        while self._auto_optimization_active:
            try:
                # Analyze and generate recommendations
                recommendations = self.analyze_performance_and_optimize()
                
                # Execute high-priority optimizations
                for action in recommendations:
                    if action.priority in [OptimizationPriority.CRITICAL, OptimizationPriority.HIGH]:
                        result = self.execute_optimization_action(action)
                        
                        if not result.success:
                            self.logger.warning(f"Auto-optimization failed: {result.error_message}")
                
                # Wait for next optimization cycle
                time.sleep(self.optimization_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in auto-optimization loop: {e}")
                time.sleep(60.0)  # Wait 1 minute on error
    
    def _update_current_metrics(self, time_window_seconds: int) -> None:
        """Update current performance metrics for all targets."""
        for target in self.optimization_targets:
            stats = self.metrics_collector.get_metric_statistics(target.metric_name)
            
            if stats and stats.count > 0:
                # Use appropriate statistic based on metric type
                if "latency" in target.metric_name or "time" in target.metric_name:
                    target.current_value = stats.p95  # Use 95th percentile for latency
                else:
                    target.current_value = stats.mean
    
    def _identify_performance_gaps(self) -> List[OptimizationTarget]:
        """Identify significant performance gaps."""
        gaps = []
        
        for target in self.optimization_targets:
            if target.optimization_direction == "minimize":
                gap_ratio = target.current_value / target.target_value
                if gap_ratio > 1.1:  # 10% worse than target
                    gaps.append(target)
            else:  # maximize
                gap_ratio = target.target_value / target.current_value
                if gap_ratio > 1.1:  # 10% worse than target
                    gaps.append(target)
        
        # Sort by importance and gap size
        gaps.sort(key=lambda t: t.importance_weight * (
            t.current_value / t.target_value if t.optimization_direction == "minimize"
            else t.target_value / t.current_value
        ), reverse=True)
        
        return gaps
    
    def _generate_optimization_actions(self, target: OptimizationTarget) -> List[OptimizationAction]:
        """Generate optimization actions for a performance gap."""
        actions = []
        
        # Get applicable rules for this metric
        applicable_rules = [
            rule for rule in self.optimization_rules
            if rule["metric_pattern"] in target.metric_name
        ]
        
        for rule in applicable_rules:
            action = self._create_optimization_action_from_rule(rule, target)
            if action:
                actions.append(action)
        
        return actions
    
    def _create_optimization_action_from_rule(self, rule: Dict[str, Any],
                                            target: OptimizationTarget) -> Optional[OptimizationAction]:
        """Create optimization action from rule."""
        try:
            # Calculate expected improvement
            gap_ratio = (target.current_value / target.target_value 
                        if target.optimization_direction == "minimize"
                        else target.target_value / target.current_value)
            
            expected_improvement = min(0.5, rule["max_improvement"] * (gap_ratio - 1.0))
            
            # Determine priority
            if gap_ratio > 2.0:
                priority = OptimizationPriority.CRITICAL
            elif gap_ratio > 1.5:
                priority = OptimizationPriority.HIGH
            elif gap_ratio > 1.2:
                priority = OptimizationPriority.MEDIUM
            else:
                priority = OptimizationPriority.LOW
            
            action = OptimizationAction(
                action_type=rule["action_type"],
                component=rule["component"],
                parameter=rule["parameter"],
                current_value=rule.get("current_value"),
                recommended_value=rule["recommended_value"],
                expected_improvement=expected_improvement,
                confidence_score=rule["confidence"],
                priority=priority,
                description=rule["description"].format(
                    metric=target.metric_name,
                    current=target.current_value,
                    target=target.target_value
                ),
                estimated_impact={target.metric_name: expected_improvement}
            )
            
            return action
        
        except Exception as e:
            self.logger.warning(f"Failed to create optimization action: {e}")
            return None
    
    def _prioritize_optimization_actions(self, actions: List[OptimizationAction]) -> List[OptimizationAction]:
        """Prioritize optimization actions by impact and feasibility."""
        # Sort by priority, then by expected improvement and confidence
        actions.sort(key=lambda a: (
            ["critical", "high", "medium", "low"].index(a.priority.value),
            -a.expected_improvement,
            -a.confidence_score
        ))
        
        return actions
    
    def _apply_safety_constraints(self, actions: List[OptimizationAction]) -> List[OptimizationAction]:
        """Apply safety constraints to filter risky optimizations."""
        safe_actions = []
        
        for action in actions:
            # Check if action violates safety constraints
            constraint_key = f"{action.component}.{action.parameter}"
            
            if constraint_key in self.safety_constraints:
                constraint = self.safety_constraints[constraint_key]
                
                # Check value bounds
                if "min_value" in constraint and action.recommended_value < constraint["min_value"]:
                    continue
                if "max_value" in constraint and action.recommended_value > constraint["max_value"]:
                    continue
                
                # Check change rate
                if "max_change_rate" in constraint:
                    current_val = action.current_value or 0
                    change_rate = abs(action.recommended_value - current_val) / max(current_val, 1)
                    if change_rate > constraint["max_change_rate"]:
                        continue
            
            safe_actions.append(action)
        
        return safe_actions
    
    def _validate_optimization_action(self, action: OptimizationAction) -> bool:
        """Validate that an optimization action is safe to execute."""
        # Basic validation
        if not action.component or not action.parameter:
            return False
        
        # Check prerequisites
        for prereq in action.prerequisites:
            if not self._check_prerequisite(prereq):
                self.logger.warning(f"Prerequisite not met: {prereq}")
                return False
        
        # Check confidence threshold
        if action.confidence_score < 0.3:
            return False
        
        return True
    
    def _apply_optimization_action(self, action: OptimizationAction) -> bool:
        """Apply the optimization action to the system."""
        # This is a placeholder - in a real implementation, this would
        # actually modify system parameters based on the action type
        
        self.logger.info(f"Applying optimization: {action.description}")
        
        # Simulate different optimization types
        if action.action_type == "cache_resize":
            # Would resize cache
            return True
        elif action.action_type == "parameter_tune":
            # Would tune algorithm parameters
            return True
        elif action.action_type == "resource_reallocation":
            # Would reallocate system resources
            return True
        elif action.action_type == "method_selection":
            # Would change similarity computation method
            return True
        
        return False
    
    def _capture_baseline_metrics(self, action: OptimizationAction) -> Dict[str, float]:
        """Capture baseline metrics before optimization."""
        baseline = {}
        
        # Capture metrics that might be affected by this optimization
        affected_metrics = action.estimated_impact.keys()
        
        for metric_name in affected_metrics:
            stats = self.metrics_collector.get_metric_statistics(metric_name)
            if stats:
                baseline[metric_name] = stats.mean
        
        return baseline
    
    def _calculate_actual_improvement(self, action: OptimizationAction,
                                    baseline: Dict[str, float],
                                    post_optimization: Dict[str, float]) -> float:
        """Calculate actual improvement achieved by optimization."""
        improvements = []
        
        for metric_name in action.estimated_impact.keys():
            if metric_name in baseline and metric_name in post_optimization:
                baseline_val = baseline[metric_name]
                post_val = post_optimization[metric_name]
                
                # Calculate improvement based on optimization direction
                if "latency" in metric_name or "time" in metric_name:
                    # Lower is better
                    improvement = (baseline_val - post_val) / baseline_val
                else:
                    # Higher is better
                    improvement = (post_val - baseline_val) / baseline_val
                
                improvements.append(improvement)
        
        return np.mean(improvements) if improvements else 0.0
    
    def _detect_side_effects(self, baseline: Dict[str, float],
                           post_optimization: Dict[str, float]) -> Dict[str, float]:
        """Detect any side effects from the optimization."""
        side_effects = {}
        
        # Compare all available metrics to detect unintended changes
        all_metrics = set(baseline.keys()) | set(post_optimization.keys())
        
        for metric_name in all_metrics:
            if metric_name in baseline and metric_name in post_optimization:
                baseline_val = baseline[metric_name]
                post_val = post_optimization[metric_name]
                
                # Calculate relative change
                relative_change = (post_val - baseline_val) / baseline_val
                
                # Consider significant changes as potential side effects
                if abs(relative_change) > 0.1:  # 10% change
                    side_effects[metric_name] = relative_change
        
        return side_effects
    
    def _action_matches_strategy(self, action: OptimizationAction,
                               strategy: OptimizationStrategy) -> bool:
        """Check if action matches optimization strategy."""
        strategy_mappings = {
            OptimizationStrategy.LATENCY_OPTIMIZATION: ["latency", "time", "duration"],
            OptimizationStrategy.THROUGHPUT_OPTIMIZATION: ["throughput", "qps", "rate"],
            OptimizationStrategy.ACCURACY_OPTIMIZATION: ["accuracy", "fidelity", "quality"],
            OptimizationStrategy.RESOURCE_OPTIMIZATION: ["memory", "cpu", "resource"]
        }
        
        if strategy == OptimizationStrategy.BALANCED_OPTIMIZATION:
            return True
        
        keywords = strategy_mappings.get(strategy, [])
        return any(keyword in action.description.lower() for keyword in keywords)
    
    def _get_optimization_type_stats(self) -> Dict[str, int]:
        """Get statistics by optimization type."""
        type_counts = defaultdict(int)
        
        for result in self.optimization_history:
            type_counts[result.action.action_type] += 1
        
        return dict(type_counts)
    
    def _check_prerequisite(self, prerequisite: str) -> bool:
        """Check if prerequisite condition is met."""
        # Simplified prerequisite checking
        if "system_stable" in prerequisite:
            # Check if system is stable (no recent errors)
            error_stats = self.metrics_collector.get_metric_statistics("system.errors")
            return not error_stats or error_stats.rate_per_second < 0.1
        
        return True
    
    def _initialize_optimization_targets(self) -> List[OptimizationTarget]:
        """Initialize optimization targets."""
        return [
            OptimizationTarget(
                metric_name="operation.similarity_computation.duration",
                target_value=85.0,
                current_value=0.0,
                importance_weight=1.0,
                optimization_direction="minimize",
                unit="ms"
            ),
            OptimizationTarget(
                metric_name="quantum.execution_time",
                target_value=60.0,
                current_value=0.0,
                importance_weight=1.0,
                optimization_direction="minimize",
                unit="ms"
            ),
            OptimizationTarget(
                metric_name="pipeline.retrieval.duration",
                target_value=50.0,
                current_value=0.0,
                importance_weight=0.8,
                optimization_direction="minimize",
                unit="ms"
            ),
            OptimizationTarget(
                metric_name="quantum.fidelity",
                target_value=0.95,
                current_value=0.0,
                importance_weight=1.2,
                optimization_direction="maximize",
                unit=""
            ),
            OptimizationTarget(
                metric_name="search.cache_hit_rate",
                target_value=0.6,
                current_value=0.0,
                importance_weight=0.6,
                optimization_direction="maximize",
                unit=""
            )
        ]
    
    def _initialize_optimization_strategies(self) -> Dict[OptimizationStrategy, Dict[str, Any]]:
        """Initialize optimization strategies."""
        return {
            OptimizationStrategy.LATENCY_OPTIMIZATION: {
                "priority_metrics": ["latency", "duration", "time"],
                "acceptable_accuracy_loss": 0.05,
                "max_resource_increase": 0.2
            },
            OptimizationStrategy.THROUGHPUT_OPTIMIZATION: {
                "priority_metrics": ["throughput", "qps", "rate"],
                "acceptable_latency_increase": 0.15,
                "max_resource_increase": 0.3
            },
            OptimizationStrategy.ACCURACY_OPTIMIZATION: {
                "priority_metrics": ["accuracy", "fidelity", "quality"],
                "acceptable_latency_increase": 0.25,
                "acceptable_throughput_loss": 0.1
            },
            OptimizationStrategy.RESOURCE_OPTIMIZATION: {
                "priority_metrics": ["memory", "cpu", "resource"],
                "acceptable_performance_loss": 0.1,
                "target_resource_reduction": 0.2
            }
        }
    
    def _initialize_optimization_rules(self) -> List[Dict[str, Any]]:
        """Initialize optimization rules."""
        return [
            {
                "metric_pattern": "similarity_computation.duration",
                "action_type": "method_selection",
                "component": "similarity_engine",
                "parameter": "computation_method",
                "recommended_value": "classical",
                "max_improvement": 0.4,
                "confidence": 0.8,
                "description": "Switch to classical similarity for {metric} (current: {current}ms, target: {target}ms)"
            },
            {
                "metric_pattern": "quantum.execution_time",
                "action_type": "parameter_tune",
                "component": "quantum_engine",
                "parameter": "circuit_depth",
                "recommended_value": "reduce",
                "max_improvement": 0.3,
                "confidence": 0.7,
                "description": "Reduce quantum circuit depth for {metric} (current: {current}ms, target: {target}ms)"
            },
            {
                "metric_pattern": "cache_hit_rate",
                "action_type": "cache_resize",
                "component": "cache_manager",
                "parameter": "cache_size",
                "recommended_value": "increase",
                "max_improvement": 0.2,
                "confidence": 0.9,
                "description": "Increase cache size for {metric} (current: {current}, target: {target})"
            },
            {
                "metric_pattern": "retrieval.duration",
                "action_type": "parameter_tune",
                "component": "search_engine",
                "parameter": "retrieval_size",
                "recommended_value": "decrease",
                "max_improvement": 0.25,
                "confidence": 0.8,
                "description": "Reduce retrieval size for {metric} (current: {current}ms, target: {target}ms)"
            }
        ]
    
    def _initialize_safety_constraints(self) -> Dict[str, Dict[str, Any]]:
        """Initialize safety constraints for optimizations."""
        return {
            "cache_manager.cache_size": {
                "min_value": 64,  # MB
                "max_value": 4096,  # MB
                "max_change_rate": 0.5  # 50% change max
            },
            "search_engine.retrieval_size": {
                "min_value": 10,
                "max_value": 1000,
                "max_change_rate": 0.3
            },
            "quantum_engine.circuit_depth": {
                "min_value": 5,
                "max_value": 100,
                "max_change_rate": 0.2
            }
        }


__all__ = [
    "OptimizationStrategy",
    "OptimizationPriority", 
    "OptimizationTarget",
    "OptimizationAction",
    "OptimizationResult",
    "AdaptivePerformanceOptimizer"
]
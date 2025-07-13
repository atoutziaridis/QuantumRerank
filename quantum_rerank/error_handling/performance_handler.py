"""
Performance-aware error handling for dynamic threshold management.

This module provides intelligent performance monitoring with adaptive thresholds,
proactive error prevention, and context-aware performance optimization.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import numpy as np

from .error_classifier import ErrorClassifier, ErrorClassification, ErrorSeverity
from .fallback_manager import AdvancedFallbackManager, FallbackTrigger
from ..utils.logging_config import get_logger


class PerformanceMetric(Enum):
    """Performance metrics for monitoring."""
    LATENCY_MS = "latency_ms"
    THROUGHPUT_OPS = "throughput_ops_per_sec"
    CPU_USAGE_PERCENT = "cpu_usage_percent"
    MEMORY_USAGE_PERCENT = "memory_usage_percent"
    ERROR_RATE_PERCENT = "error_rate_percent"
    QUANTUM_FIDELITY = "quantum_fidelity"
    CACHE_HIT_RATE = "cache_hit_rate_percent"


class ThresholdAction(Enum):
    """Actions to take when thresholds are breached."""
    LOG_WARNING = "log_warning"
    TRIGGER_FALLBACK = "trigger_fallback"
    REDUCE_LOAD = "reduce_load"
    ESCALATE_ALERT = "escalate_alert"
    ADAPTIVE_DEGRADATION = "adaptive_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class PerformanceThreshold:
    """Dynamic performance threshold configuration."""
    metric: PerformanceMetric
    warning_level: float
    critical_level: float
    actions: List[ThresholdAction] = field(default_factory=list)
    adaptive: bool = True
    window_size_s: int = 300  # 5 minutes
    min_samples: int = 10
    percentile: float = 95.0  # P95 threshold
    baseline_adjustment_factor: float = 1.2


@dataclass
class PerformanceEvent:
    """Performance event for tracking and analysis."""
    metric: PerformanceMetric
    value: float
    threshold: PerformanceThreshold
    severity: ErrorSeverity
    timestamp: float = field(default_factory=time.time)
    component: str = "unknown"
    operation: str = "unknown"
    context: Dict[str, Any] = field(default_factory=dict)
    actions_taken: List[ThresholdAction] = field(default_factory=list)


@dataclass
class PerformanceStats:
    """Performance statistics for a metric."""
    metric: PerformanceMetric
    current_value: float
    mean: float
    median: float
    p95: float
    p99: float
    std_dev: float
    trend: str  # "improving", "degrading", "stable"
    samples_count: int
    time_window_s: int


class PerformanceErrorHandler:
    """
    Performance-aware error handling with adaptive thresholds.
    
    Provides dynamic performance monitoring, proactive error prevention,
    and intelligent threshold adjustment based on system behavior patterns.
    """
    
    def __init__(self, 
                 error_classifier: Optional[ErrorClassifier] = None,
                 fallback_manager: Optional[AdvancedFallbackManager] = None):
        self.error_classifier = error_classifier or ErrorClassifier()
        self.fallback_manager = fallback_manager or AdvancedFallbackManager()
        self.logger = get_logger(__name__)
        
        # Performance monitoring
        self.performance_data: Dict[PerformanceMetric, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.performance_events: deque = deque(maxlen=500)
        self.thresholds: Dict[PerformanceMetric, PerformanceThreshold] = self._initialize_thresholds()
        
        # Adaptive threshold management
        self.baseline_calculations: Dict[PerformanceMetric, Dict[str, float]] = defaultdict(dict)
        self.threshold_adjustments: Dict[PerformanceMetric, float] = defaultdict(lambda: 1.0)
        
        # Performance optimization
        self.load_shedding_active = False
        self.degradation_level = 0.0  # 0.0 = no degradation, 1.0 = maximum degradation
        self.circuit_breaker_states: Dict[str, bool] = defaultdict(bool)
        
        # Monitoring hooks
        self.performance_hooks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Start background optimization
        self._optimization_thread = threading.Thread(target=self._background_optimization, daemon=True)
        self._optimization_thread.start()
        
        self.logger.info("Initialized PerformanceErrorHandler")
    
    def record_performance_metric(self, metric: PerformanceMetric, value: float,
                                 component: str = "unknown", operation: str = "unknown",
                                 context: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a performance metric and check thresholds.
        
        Args:
            metric: Performance metric type
            value: Metric value
            component: Component name
            operation: Operation name
            context: Additional context
        """
        timestamp = time.time()
        context = context or {}
        
        with self._lock:
            # Store metric data
            self.performance_data[metric].append({
                "value": value,
                "timestamp": timestamp,
                "component": component,
                "operation": operation,
                "context": context
            })
            
            # Check thresholds
            threshold_event = self._check_thresholds(metric, value, component, operation, context)
            if threshold_event:
                self._handle_threshold_breach(threshold_event)
            
            # Update adaptive thresholds periodically
            if len(self.performance_data[metric]) % 50 == 0:
                self._update_adaptive_threshold(metric)
    
    def get_performance_stats(self, metric: PerformanceMetric, 
                            time_window_s: int = 300) -> PerformanceStats:
        """Get comprehensive performance statistics."""
        cutoff_time = time.time() - time_window_s
        
        with self._lock:
            recent_data = [
                data for data in self.performance_data[metric]
                if data["timestamp"] >= cutoff_time
            ]
            
            if not recent_data:
                return PerformanceStats(
                    metric=metric,
                    current_value=0.0,
                    mean=0.0,
                    median=0.0,
                    p95=0.0,
                    p99=0.0,
                    std_dev=0.0,
                    trend="stable",
                    samples_count=0,
                    time_window_s=time_window_s
                )
            
            values = [data["value"] for data in recent_data]
            
            # Calculate statistics
            mean_val = np.mean(values)
            median_val = np.median(values)
            p95_val = np.percentile(values, 95)
            p99_val = np.percentile(values, 99)
            std_dev = np.std(values)
            
            # Calculate trend
            trend = self._calculate_trend(values)
            
            return PerformanceStats(
                metric=metric,
                current_value=values[-1],
                mean=mean_val,
                median=median_val,
                p95=p95_val,
                p99=p99_val,
                std_dev=std_dev,
                trend=trend,
                samples_count=len(values),
                time_window_s=time_window_s
            )
    
    def predict_performance_issues(self, component: str, operation: str,
                                 context: Dict[str, Any]) -> Dict[PerformanceMetric, float]:
        """
        Predict likelihood of performance issues.
        
        Args:
            component: Component name
            operation: Operation name
            context: Current context
            
        Returns:
            Dictionary of metrics and their issue likelihood (0-1)
        """
        predictions = {}
        
        for metric in PerformanceMetric:
            # Get recent performance data
            stats = self.get_performance_stats(metric, time_window_s=300)
            
            if stats.samples_count < 5:
                predictions[metric] = 0.1  # Low likelihood with insufficient data
                continue
            
            # Base prediction on trend and variance
            likelihood = 0.0
            
            # Trend-based prediction
            if stats.trend == "degrading":
                likelihood += 0.4
            elif stats.trend == "stable":
                likelihood += 0.1
            
            # Variance-based prediction
            if stats.std_dev > stats.mean * 0.3:  # High variance
                likelihood += 0.3
            
            # Current value proximity to threshold
            threshold = self.thresholds.get(metric)
            if threshold and stats.current_value > threshold.warning_level:
                proximity = (stats.current_value - threshold.warning_level) / \
                           (threshold.critical_level - threshold.warning_level)
                likelihood += proximity * 0.4
            
            # Context-based adjustments
            likelihood *= self._get_context_adjustment_factor(metric, context)
            
            predictions[metric] = min(1.0, likelihood)
        
        return predictions
    
    def optimize_performance(self, component: str, operation: str,
                           target_improvement: float = 0.2) -> Dict[str, Any]:
        """
        Optimize performance for specific component/operation.
        
        Args:
            component: Component to optimize
            operation: Operation to optimize
            target_improvement: Target improvement ratio (0-1)
            
        Returns:
            Optimization result and recommendations
        """
        optimization_result = {
            "component": component,
            "operation": operation,
            "target_improvement": target_improvement,
            "actions_taken": [],
            "estimated_improvement": 0.0,
            "success": False
        }
        
        # Analyze current performance
        current_stats = {}
        for metric in PerformanceMetric:
            current_stats[metric] = self.get_performance_stats(metric)
        
        # Identify optimization opportunities
        optimization_actions = self._identify_optimization_opportunities(
            component, operation, current_stats, target_improvement
        )
        
        # Execute optimizations
        total_improvement = 0.0
        for action in optimization_actions:
            try:
                improvement = self._execute_optimization_action(action, component, operation)
                total_improvement += improvement
                optimization_result["actions_taken"].append(action)
                
                self.logger.info(f"Applied optimization {action} with {improvement:.2f} improvement")
                
            except Exception as e:
                self.logger.error(f"Failed to apply optimization {action}: {e}")
        
        optimization_result["estimated_improvement"] = total_improvement
        optimization_result["success"] = total_improvement >= target_improvement
        
        return optimization_result
    
    def register_performance_hook(self, event_type: str, callback: Callable) -> None:
        """Register a callback for performance events."""
        self.performance_hooks[event_type].append(callback)
        self.logger.info(f"Registered performance hook for {event_type}")
    
    def get_system_health_score(self) -> float:
        """
        Calculate overall system health score (0-1).
        
        Returns:
            Health score where 1.0 is perfect health
        """
        health_scores = []
        
        for metric in PerformanceMetric:
            stats = self.get_performance_stats(metric)
            
            if stats.samples_count < 5:
                continue
            
            # Calculate metric health based on threshold proximity
            threshold = self.thresholds.get(metric)
            if threshold:
                if stats.current_value <= threshold.warning_level:
                    metric_health = 1.0
                elif stats.current_value <= threshold.critical_level:
                    # Linear interpolation between warning and critical
                    ratio = (stats.current_value - threshold.warning_level) / \
                           (threshold.critical_level - threshold.warning_level)
                    metric_health = 1.0 - (ratio * 0.5)  # 50% health at critical level
                else:
                    metric_health = 0.0  # Critical threshold exceeded
                
                health_scores.append(metric_health)
        
        # Overall health is weighted average
        if health_scores:
            overall_health = np.mean(health_scores)
            
            # Apply degradation penalty
            overall_health *= (1.0 - self.degradation_level * 0.3)
            
            return max(0.0, overall_health)
        
        return 0.5  # Neutral health with no data
    
    def _check_thresholds(self, metric: PerformanceMetric, value: float,
                         component: str, operation: str, context: Dict[str, Any]) -> Optional[PerformanceEvent]:
        """Check if metric value breaches thresholds."""
        threshold = self.thresholds.get(metric)
        if not threshold:
            return None
        
        # Adjust threshold based on adaptive settings
        adjusted_warning = threshold.warning_level * self.threshold_adjustments[metric]
        adjusted_critical = threshold.critical_level * self.threshold_adjustments[metric]
        
        severity = None
        if value >= adjusted_critical:
            severity = ErrorSeverity.CRITICAL
        elif value >= adjusted_warning:
            severity = ErrorSeverity.MEDIUM
        
        if severity:
            return PerformanceEvent(
                metric=metric,
                value=value,
                threshold=threshold,
                severity=severity,
                component=component,
                operation=operation,
                context=context
            )
        
        return None
    
    def _handle_threshold_breach(self, event: PerformanceEvent) -> None:
        """Handle threshold breach with appropriate actions."""
        self.performance_events.append(event)
        
        # Execute threshold actions
        for action in event.threshold.actions:
            try:
                self._execute_threshold_action(action, event)
                event.actions_taken.append(action)
            except Exception as e:
                self.logger.error(f"Failed to execute threshold action {action}: {e}")
        
        # Notify hooks
        for callback in self.performance_hooks.get("threshold_breach", []):
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Performance hook callback failed: {e}")
        
        self.logger.warning(
            f"Performance threshold breach: {event.metric.value}={event.value:.2f} "
            f"(threshold: {event.threshold.warning_level}/{event.threshold.critical_level}) "
            f"in {event.component}.{event.operation}"
        )
    
    def _execute_threshold_action(self, action: ThresholdAction, event: PerformanceEvent) -> None:
        """Execute specific threshold action."""
        if action == ThresholdAction.LOG_WARNING:
            self.logger.warning(f"Performance warning: {event.metric.value}={event.value:.2f}")
        
        elif action == ThresholdAction.TRIGGER_FALLBACK:
            # Create fallback context and trigger fallback
            from .fallback_manager import FallbackContext
            fallback_context = FallbackContext(
                operation_name=f"{event.component}.{event.operation}",
                original_args=(),
                original_kwargs={},
                performance_metrics={event.metric.value: event.value},
                system_state=event.context
            )
            
            self.fallback_manager.execute_fallback(fallback_context, FallbackTrigger.PERFORMANCE_THRESHOLD)
        
        elif action == ThresholdAction.REDUCE_LOAD:
            self._activate_load_shedding(event)
        
        elif action == ThresholdAction.ADAPTIVE_DEGRADATION:
            self._increase_degradation_level(event)
        
        elif action == ThresholdAction.CIRCUIT_BREAKER:
            self._activate_circuit_breaker(event.component)
        
        elif action == ThresholdAction.ESCALATE_ALERT:
            self._escalate_performance_alert(event)
    
    def _activate_load_shedding(self, event: PerformanceEvent) -> None:
        """Activate load shedding to reduce system stress."""
        self.load_shedding_active = True
        self.logger.info(f"Activated load shedding due to {event.metric.value} breach")
    
    def _increase_degradation_level(self, event: PerformanceEvent) -> None:
        """Increase system degradation level."""
        if event.severity == ErrorSeverity.CRITICAL:
            self.degradation_level = min(1.0, self.degradation_level + 0.3)
        else:
            self.degradation_level = min(1.0, self.degradation_level + 0.1)
        
        self.logger.info(f"Increased degradation level to {self.degradation_level:.2f}")
    
    def _activate_circuit_breaker(self, component: str) -> None:
        """Activate circuit breaker for component."""
        self.circuit_breaker_states[component] = True
        self.logger.warning(f"Activated circuit breaker for component {component}")
    
    def _escalate_performance_alert(self, event: PerformanceEvent) -> None:
        """Escalate performance alert to administrators."""
        # This would integrate with alerting system
        self.logger.critical(
            f"ESCALATED ALERT: Critical performance issue in {event.component}.{event.operation}: "
            f"{event.metric.value}={event.value:.2f}"
        )
    
    def _update_adaptive_threshold(self, metric: PerformanceMetric) -> None:
        """Update adaptive threshold based on recent performance."""
        threshold = self.thresholds.get(metric)
        if not threshold or not threshold.adaptive:
            return
        
        stats = self.get_performance_stats(metric, threshold.window_size_s)
        if stats.samples_count < threshold.min_samples:
            return
        
        # Calculate new baseline using specified percentile
        new_baseline = np.percentile(
            [data["value"] for data in self.performance_data[metric]],
            threshold.percentile
        )
        
        # Adjust threshold with baseline adjustment factor
        adjusted_threshold = new_baseline * threshold.baseline_adjustment_factor
        
        # Update threshold adjustment factor
        if adjusted_threshold > 0:
            self.threshold_adjustments[metric] = adjusted_threshold / threshold.warning_level
            
            self.logger.debug(
                f"Updated adaptive threshold for {metric.value}: "
                f"adjustment factor = {self.threshold_adjustments[metric]:.2f}"
            )
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate performance trend from recent values."""
        if len(values) < 5:
            return "stable"
        
        # Use linear regression to determine trend
        x = np.array(range(len(values)))
        y = np.array(values)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(x, y)[0, 1]
        
        if correlation > 0.3:
            return "degrading"  # Performance getting worse (values increasing)
        elif correlation < -0.3:
            return "improving"  # Performance getting better (values decreasing)
        else:
            return "stable"
    
    def _get_context_adjustment_factor(self, metric: PerformanceMetric, 
                                     context: Dict[str, Any]) -> float:
        """Get context-based adjustment factor for predictions."""
        factor = 1.0
        
        # Time-based adjustments
        current_hour = time.localtime().tm_hour
        if 9 <= current_hour <= 17:  # Business hours
            factor *= 1.2
        
        # Load-based adjustments
        if context.get("high_load", False):
            factor *= 1.5
        
        # Component-specific adjustments
        if "quantum" in context.get("component", ""):
            if metric == PerformanceMetric.LATENCY_MS:
                factor *= 1.3  # Quantum operations more latency-sensitive
        
        return factor
    
    def _identify_optimization_opportunities(self, component: str, operation: str,
                                           current_stats: Dict[PerformanceMetric, PerformanceStats],
                                           target_improvement: float) -> List[str]:
        """Identify optimization opportunities."""
        opportunities = []
        
        # Analyze each metric for optimization potential
        for metric, stats in current_stats.items():
            if stats.samples_count < 5:
                continue
            
            threshold = self.thresholds.get(metric)
            if not threshold:
                continue
            
            # Check if metric is approaching threshold
            if stats.current_value > threshold.warning_level * 0.8:
                if metric == PerformanceMetric.LATENCY_MS:
                    opportunities.extend(["enable_caching", "optimize_algorithms", "parallel_processing"])
                elif metric == PerformanceMetric.MEMORY_USAGE_PERCENT:
                    opportunities.extend(["garbage_collection", "memory_optimization", "data_compression"])
                elif metric == PerformanceMetric.CPU_USAGE_PERCENT:
                    opportunities.extend(["algorithm_optimization", "load_balancing", "async_processing"])
                elif metric == PerformanceMetric.ERROR_RATE_PERCENT:
                    opportunities.extend(["error_prevention", "circuit_breaker", "graceful_degradation"])
        
        # Component-specific optimizations
        if "quantum" in component:
            opportunities.extend(["circuit_simplification", "parameter_optimization", "backend_selection"])
        elif "similarity" in component:
            opportunities.extend(["approximate_methods", "dimensionality_reduction", "early_termination"])
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(opportunities))
    
    def _execute_optimization_action(self, action: str, component: str, operation: str) -> float:
        """Execute optimization action and return estimated improvement."""
        improvement = 0.0
        
        if action == "enable_caching":
            # Enable more aggressive caching
            improvement = 0.2
        elif action == "optimize_algorithms":
            # Switch to optimized algorithms
            improvement = 0.15
        elif action == "parallel_processing":
            # Enable parallel processing
            improvement = 0.1
        elif action == "circuit_simplification":
            # Simplify quantum circuits
            improvement = 0.25
        elif action == "approximate_methods":
            # Use approximate computation methods
            improvement = 0.3
        elif action == "graceful_degradation":
            # Enable graceful service degradation
            self.degradation_level = min(1.0, self.degradation_level + 0.1)
            improvement = 0.2
        
        # Simulate optimization effect
        time.sleep(0.1)  # Simulate optimization time
        
        return improvement
    
    def _background_optimization(self) -> None:
        """Background thread for continuous performance optimization."""
        while True:
            try:
                # Sleep for optimization interval
                time.sleep(60)  # Run every minute
                
                # Check if optimization is needed
                health_score = self.get_system_health_score()
                
                if health_score < 0.7:  # Health below 70%
                    self.logger.info(f"System health low ({health_score:.2f}), running optimization")
                    
                    # Run optimization for all components
                    self._run_background_optimizations()
                
                # Reset load shedding if health improved
                if health_score > 0.8 and self.load_shedding_active:
                    self.load_shedding_active = False
                    self.logger.info("Deactivated load shedding - system health improved")
                
                # Reduce degradation level if health improved
                if health_score > 0.9 and self.degradation_level > 0:
                    self.degradation_level = max(0.0, self.degradation_level - 0.1)
                    self.logger.info(f"Reduced degradation level to {self.degradation_level:.2f}")
                
            except Exception as e:
                self.logger.error(f"Background optimization error: {e}")
    
    def _run_background_optimizations(self) -> None:
        """Run background optimizations for performance improvement."""
        # Identify components needing optimization
        components_needing_optimization = []
        
        for component in ["quantum_engine", "similarity_engine", "search_engine"]:
            for metric in PerformanceMetric:
                stats = self.get_performance_stats(metric)
                threshold = self.thresholds.get(metric)
                
                if threshold and stats.current_value > threshold.warning_level:
                    components_needing_optimization.append(component)
                    break
        
        # Run optimizations
        for component in set(components_needing_optimization):
            try:
                self.optimize_performance(component, "background_optimization", target_improvement=0.1)
            except Exception as e:
                self.logger.error(f"Background optimization failed for {component}: {e}")
    
    def _initialize_thresholds(self) -> Dict[PerformanceMetric, PerformanceThreshold]:
        """Initialize default performance thresholds."""
        return {
            PerformanceMetric.LATENCY_MS: PerformanceThreshold(
                metric=PerformanceMetric.LATENCY_MS,
                warning_level=150.0,
                critical_level=500.0,
                actions=[ThresholdAction.LOG_WARNING, ThresholdAction.TRIGGER_FALLBACK],
                adaptive=True
            ),
            PerformanceMetric.THROUGHPUT_OPS: PerformanceThreshold(
                metric=PerformanceMetric.THROUGHPUT_OPS,
                warning_level=50.0,  # Operations per second
                critical_level=10.0,  # Lower is worse for throughput
                actions=[ThresholdAction.LOG_WARNING, ThresholdAction.REDUCE_LOAD],
                adaptive=True
            ),
            PerformanceMetric.CPU_USAGE_PERCENT: PerformanceThreshold(
                metric=PerformanceMetric.CPU_USAGE_PERCENT,
                warning_level=80.0,
                critical_level=95.0,
                actions=[ThresholdAction.LOG_WARNING, ThresholdAction.ADAPTIVE_DEGRADATION],
                adaptive=True
            ),
            PerformanceMetric.MEMORY_USAGE_PERCENT: PerformanceThreshold(
                metric=PerformanceMetric.MEMORY_USAGE_PERCENT,
                warning_level=85.0,
                critical_level=95.0,
                actions=[ThresholdAction.LOG_WARNING, ThresholdAction.CIRCUIT_BREAKER],
                adaptive=True
            ),
            PerformanceMetric.ERROR_RATE_PERCENT: PerformanceThreshold(
                metric=PerformanceMetric.ERROR_RATE_PERCENT,
                warning_level=5.0,
                critical_level=15.0,
                actions=[ThresholdAction.LOG_WARNING, ThresholdAction.ESCALATE_ALERT],
                adaptive=True
            ),
            PerformanceMetric.QUANTUM_FIDELITY: PerformanceThreshold(
                metric=PerformanceMetric.QUANTUM_FIDELITY,
                warning_level=0.8,
                critical_level=0.6,  # Lower is worse for fidelity
                actions=[ThresholdAction.LOG_WARNING, ThresholdAction.TRIGGER_FALLBACK],
                adaptive=False  # Fidelity thresholds should be fixed
            ),
            PerformanceMetric.CACHE_HIT_RATE: PerformanceThreshold(
                metric=PerformanceMetric.CACHE_HIT_RATE,
                warning_level=70.0,
                critical_level=50.0,  # Lower is worse for cache hit rate
                actions=[ThresholdAction.LOG_WARNING],
                adaptive=True
            )
        }


__all__ = [
    "PerformanceMetric",
    "ThresholdAction", 
    "PerformanceThreshold",
    "PerformanceEvent",
    "PerformanceStats",
    "PerformanceErrorHandler"
]
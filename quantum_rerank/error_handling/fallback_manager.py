"""
Advanced fallback management system with intelligent strategy selection.

This module provides context-aware fallback strategies, quality vs performance trade-offs,
and seamless integration with the existing error recovery system.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import numpy as np

from .error_classifier import ErrorClassifier, ErrorClassification, ErrorCategory
from ..utils.exceptions import QuantumRerankException
from ..utils.logging_config import get_logger


class FallbackStrategy(Enum):
    """Enhanced fallback strategies."""
    # Quantum-specific fallbacks
    QUANTUM_TO_CLASSICAL = "quantum_to_classical"
    SIMPLIFIED_QUANTUM = "simplified_quantum"
    APPROXIMATE_QUANTUM = "approximate_quantum"
    
    # Classical computation fallbacks
    HIGH_ACCURACY_TO_FAST = "high_accuracy_to_fast"
    COMPLEX_TO_SIMPLE = "complex_to_simple"
    
    # System-level fallbacks
    REAL_TIME_TO_CACHED = "real_time_to_cached"
    FULL_TO_PARTIAL = "full_to_partial"
    SYNCHRONOUS_TO_ASYNC = "synchronous_to_async"
    
    # Service degradation
    GRACEFUL_DEGRADATION = "graceful_degradation"
    LOAD_SHEDDING = "load_shedding"
    CIRCUIT_BREAKER = "circuit_breaker"
    
    # Default fallbacks
    CACHED_RESULT = "cached_result"
    DEFAULT_VALUE = "default_value"
    NONE = "none"


class FallbackTrigger(Enum):
    """Triggers that activate fallback strategies."""
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    TIMEOUT = "timeout"
    QUALITY_DEGRADATION = "quality_degradation"
    PROACTIVE_OPTIMIZATION = "proactive_optimization"


@dataclass
class FallbackConfig:
    """Configuration for a specific fallback strategy."""
    strategy: FallbackStrategy
    triggers: List[FallbackTrigger]
    priority: int  # Lower number = higher priority
    quality_impact: float  # 0-1, impact on result quality
    performance_gain: float  # 0-1, expected performance improvement
    resource_reduction: float  # 0-1, expected resource usage reduction
    success_probability: float  # 0-1, probability of successful fallback
    max_usage_per_hour: int  # Rate limiting for fallback usage
    prerequisites: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FallbackResult:
    """Result of a fallback execution."""
    strategy: FallbackStrategy
    success: bool
    result: Any
    execution_time_ms: float
    quality_impact: float
    trigger: FallbackTrigger
    original_error: Optional[Exception] = None
    fallback_error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class FallbackContext:
    """Context information for fallback decisions."""
    operation_name: str
    original_args: tuple
    original_kwargs: dict
    error_classification: Optional[ErrorClassification] = None
    performance_metrics: Optional[Dict[str, float]] = None
    system_state: Optional[Dict[str, Any]] = None
    user_preferences: Optional[Dict[str, Any]] = None
    deadline_ms: Optional[float] = None
    quality_requirements: Optional[Dict[str, float]] = None


class AdvancedFallbackManager:
    """
    Advanced fallback management with intelligent strategy selection.
    
    Provides context-aware fallback decisions, quality vs performance optimization,
    and seamless integration with error classification and recovery systems.
    """
    
    def __init__(self, error_classifier: Optional[ErrorClassifier] = None):
        self.error_classifier = error_classifier or ErrorClassifier()
        self.logger = get_logger(__name__)
        
        # Fallback configurations
        self.fallback_configs = self._initialize_fallback_configs()
        self.fallback_handlers: Dict[FallbackStrategy, Callable] = {}
        
        # Execution tracking
        self.fallback_history: deque = deque(maxlen=1000)
        self.fallback_statistics: Dict[str, Any] = defaultdict(int)
        self.strategy_performance: Dict[FallbackStrategy, List[float]] = defaultdict(list)
        
        # Rate limiting
        self.usage_counters: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Adaptive learning
        self.strategy_success_rates: Dict[FallbackStrategy, float] = {}
        self.context_patterns: Dict[str, List[FallbackStrategy]] = defaultdict(list)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize handlers
        self._register_fallback_handlers()
        
        self.logger.info("Initialized AdvancedFallbackManager")
    
    def execute_fallback(self, context: FallbackContext, 
                        trigger: FallbackTrigger) -> FallbackResult:
        """
        Execute the most appropriate fallback strategy.
        
        Args:
            context: Fallback execution context
            trigger: What triggered the fallback
            
        Returns:
            Fallback execution result
        """
        start_time = time.time()
        
        # Select best fallback strategy
        strategy = self._select_optimal_strategy(context, trigger)
        
        if strategy == FallbackStrategy.NONE:
            return FallbackResult(
                strategy=strategy,
                success=False,
                result=None,
                execution_time_ms=0,
                quality_impact=1.0,
                trigger=trigger,
                fallback_error=Exception("No suitable fallback strategy available")
            )
        
        # Check rate limiting
        if not self._check_rate_limit(strategy, context.operation_name):
            self.logger.warning(f"Rate limit exceeded for fallback strategy {strategy.value}")
            return self._try_alternative_strategy(context, trigger, [strategy])
        
        # Execute fallback
        try:
            handler = self.fallback_handlers.get(strategy)
            if not handler:
                raise Exception(f"No handler registered for strategy {strategy.value}")
            
            self.logger.info(f"Executing fallback strategy {strategy.value} for {context.operation_name}")
            
            # Execute the handler
            result = handler(context)
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Get strategy configuration
            config = self.fallback_configs.get(strategy)
            quality_impact = config.quality_impact if config else 0.5
            
            fallback_result = FallbackResult(
                strategy=strategy,
                success=True,
                result=result,
                execution_time_ms=execution_time_ms,
                quality_impact=quality_impact,
                trigger=trigger,
                metadata={
                    "fallback_reason": f"Triggered by {trigger.value}",
                    "operation": context.operation_name
                }
            )
            
            # Record success
            self._record_fallback_execution(fallback_result, context)
            
            return fallback_result
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            
            self.logger.error(f"Fallback strategy {strategy.value} failed: {e}")
            
            fallback_result = FallbackResult(
                strategy=strategy,
                success=False,
                result=None,
                execution_time_ms=execution_time_ms,
                quality_impact=1.0,
                trigger=trigger,
                fallback_error=e
            )
            
            # Try alternative strategy
            alternative_result = self._try_alternative_strategy(context, trigger, [strategy])
            if alternative_result.success:
                return alternative_result
            
            # Record failure
            self._record_fallback_execution(fallback_result, context)
            
            return fallback_result
    
    def register_fallback_handler(self, strategy: FallbackStrategy, 
                                 handler: Callable[[FallbackContext], Any]) -> None:
        """Register a custom fallback handler."""
        self.fallback_handlers[strategy] = handler
        self.logger.info(f"Registered handler for fallback strategy {strategy.value}")
    
    def get_fallback_recommendation(self, context: FallbackContext,
                                  trigger: FallbackTrigger) -> List[FallbackStrategy]:
        """Get ranked list of recommended fallback strategies."""
        # Get all applicable strategies
        applicable_strategies = self._get_applicable_strategies(context, trigger)
        
        # Rank strategies by effectiveness
        ranked_strategies = self._rank_strategies(applicable_strategies, context, trigger)
        
        return ranked_strategies[:5]  # Return top 5 recommendations
    
    def get_fallback_statistics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive fallback statistics."""
        cutoff_time = time.time() - (time_window_hours * 3600)
        
        with self._lock:
            recent_fallbacks = [
                fb for fb in self.fallback_history
                if fb.timestamp >= cutoff_time
            ]
            
            if not recent_fallbacks:
                return {"fallback_count": 0, "time_window_hours": time_window_hours}
            
            # Calculate statistics
            successful_fallbacks = [fb for fb in recent_fallbacks if fb.success]
            
            stats = {
                "time_window_hours": time_window_hours,
                "total_fallbacks": len(recent_fallbacks),
                "successful_fallbacks": len(successful_fallbacks),
                "success_rate": len(successful_fallbacks) / len(recent_fallbacks),
                "average_execution_time_ms": np.mean([fb.execution_time_ms for fb in successful_fallbacks]) if successful_fallbacks else 0,
                "average_quality_impact": np.mean([fb.quality_impact for fb in successful_fallbacks]) if successful_fallbacks else 0,
                "strategy_distribution": self._calculate_strategy_distribution(recent_fallbacks),
                "trigger_distribution": self._calculate_trigger_distribution(recent_fallbacks),
                "most_effective_strategies": self._get_most_effective_strategies(recent_fallbacks)
            }
            
            return stats
    
    def _select_optimal_strategy(self, context: FallbackContext, 
                               trigger: FallbackTrigger) -> FallbackStrategy:
        """Select the optimal fallback strategy for the given context."""
        # Get applicable strategies
        applicable_strategies = self._get_applicable_strategies(context, trigger)
        
        if not applicable_strategies:
            return FallbackStrategy.NONE
        
        # Rank strategies by effectiveness
        ranked_strategies = self._rank_strategies(applicable_strategies, context, trigger)
        
        # Select the best strategy that passes rate limiting
        for strategy in ranked_strategies:
            if self._check_rate_limit(strategy, context.operation_name):
                return strategy
        
        # If all strategies are rate limited, return NONE
        return FallbackStrategy.NONE
    
    def _get_applicable_strategies(self, context: FallbackContext,
                                 trigger: FallbackTrigger) -> List[FallbackStrategy]:
        """Get strategies applicable to the current context and trigger."""
        applicable = []
        
        for strategy, config in self.fallback_configs.items():
            # Check if trigger matches
            if trigger not in config.triggers:
                continue
            
            # Check prerequisites
            if not self._check_prerequisites(config.prerequisites, context):
                continue
            
            # Context-specific filtering
            if not self._is_strategy_applicable(strategy, context):
                continue
            
            applicable.append(strategy)
        
        return applicable
    
    def _rank_strategies(self, strategies: List[FallbackStrategy],
                        context: FallbackContext, trigger: FallbackTrigger) -> List[FallbackStrategy]:
        """Rank strategies by effectiveness for the given context."""
        if not strategies:
            return []
        
        strategy_scores = []
        
        for strategy in strategies:
            score = self._calculate_strategy_score(strategy, context, trigger)
            strategy_scores.append((strategy, score))
        
        # Sort by score (higher is better)
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [strategy for strategy, _ in strategy_scores]
    
    def _calculate_strategy_score(self, strategy: FallbackStrategy,
                                context: FallbackContext, trigger: FallbackTrigger) -> float:
        """Calculate effectiveness score for a strategy."""
        config = self.fallback_configs.get(strategy)
        if not config:
            return 0.0
        
        score = 0.0
        
        # Base score from configuration
        score += config.success_probability * 0.4
        score += (1.0 - config.quality_impact) * 0.3  # Lower quality impact is better
        score += config.performance_gain * 0.2
        score += (1.0 / config.priority) * 0.1  # Lower priority number is better
        
        # Historical success rate
        historical_success = self.strategy_success_rates.get(strategy, 0.5)
        score *= historical_success
        
        # Context-specific adjustments
        score *= self._get_context_adjustment_factor(strategy, context, trigger)
        
        # Recent performance adjustment
        recent_performance = self._get_recent_performance_factor(strategy)
        score *= recent_performance
        
        return score
    
    def _get_context_adjustment_factor(self, strategy: FallbackStrategy,
                                     context: FallbackContext, trigger: FallbackTrigger) -> float:
        """Get context-specific adjustment factor for strategy scoring."""
        factor = 1.0
        
        # Operation-specific adjustments
        if context.operation_name:
            if "quantum" in context.operation_name and strategy == FallbackStrategy.QUANTUM_TO_CLASSICAL:
                factor *= 1.2  # Boost quantum-to-classical for quantum operations
            elif "similarity" in context.operation_name and strategy == FallbackStrategy.HIGH_ACCURACY_TO_FAST:
                factor *= 1.1  # Boost fast methods for similarity computation
        
        # Performance requirements
        if context.deadline_ms:
            remaining_time = context.deadline_ms
            if remaining_time < 100:  # Tight deadline
                if strategy in [FallbackStrategy.REAL_TIME_TO_CACHED, FallbackStrategy.HIGH_ACCURACY_TO_FAST]:
                    factor *= 1.3
        
        # Quality requirements
        if context.quality_requirements:
            min_quality = context.quality_requirements.get("min_accuracy", 0.0)
            config = self.fallback_configs.get(strategy)
            if config and config.quality_impact > (1.0 - min_quality):
                factor *= 0.5  # Penalize strategies that impact quality too much
        
        # System state considerations
        if context.system_state:
            cpu_usage = context.system_state.get("cpu_usage", 0)
            memory_usage = context.system_state.get("memory_usage", 0)
            
            if cpu_usage > 80:
                if strategy in [FallbackStrategy.COMPLEX_TO_SIMPLE, FallbackStrategy.REAL_TIME_TO_CACHED]:
                    factor *= 1.2
            
            if memory_usage > 85:
                if strategy in [FallbackStrategy.FULL_TO_PARTIAL, FallbackStrategy.LOAD_SHEDDING]:
                    factor *= 1.3
        
        return factor
    
    def _get_recent_performance_factor(self, strategy: FallbackStrategy) -> float:
        """Get performance factor based on recent strategy execution."""
        if strategy not in self.strategy_performance:
            return 1.0
        
        recent_times = self.strategy_performance[strategy][-10:]  # Last 10 executions
        
        if len(recent_times) < 3:
            return 1.0
        
        # Better recent performance = higher factor
        avg_time = np.mean(recent_times)
        target_time = 100.0  # 100ms target
        
        if avg_time <= target_time:
            return 1.2
        elif avg_time <= target_time * 2:
            return 1.0
        else:
            return 0.8
    
    def _is_strategy_applicable(self, strategy: FallbackStrategy, 
                              context: FallbackContext) -> bool:
        """Check if strategy is applicable to the current context."""
        # Quantum-specific strategies
        if strategy in [FallbackStrategy.QUANTUM_TO_CLASSICAL, 
                       FallbackStrategy.SIMPLIFIED_QUANTUM,
                       FallbackStrategy.APPROXIMATE_QUANTUM]:
            return "quantum" in context.operation_name.lower()
        
        # Classical computation strategies
        if strategy in [FallbackStrategy.HIGH_ACCURACY_TO_FAST,
                       FallbackStrategy.COMPLEX_TO_SIMPLE]:
            return "similarity" in context.operation_name.lower() or "embedding" in context.operation_name.lower()
        
        # System-level strategies are generally applicable
        return True
    
    def _check_prerequisites(self, prerequisites: List[str], 
                           context: FallbackContext) -> bool:
        """Check if strategy prerequisites are met."""
        for prereq in prerequisites:
            if prereq == "classical_similarity_available":
                # Check if classical similarity computation is available
                continue  # Assume available for now
            elif prereq == "cache_available":
                # Check if cache system is available
                continue  # Assume available for now
            elif prereq == "alternative_backend_available":
                # Check if alternative quantum backend is available
                continue  # Assume available for now
        
        return True
    
    def _check_rate_limit(self, strategy: FallbackStrategy, operation_name: str) -> bool:
        """Check if strategy usage is within rate limits."""
        config = self.fallback_configs.get(strategy)
        if not config:
            return True
        
        rate_limit_key = f"{strategy.value}_{operation_name}"
        current_time = time.time()
        
        # Clean old entries (older than 1 hour)
        usage_times = self.usage_counters[rate_limit_key]
        while usage_times and current_time - usage_times[0] > 3600:
            usage_times.popleft()
        
        # Check if under rate limit
        return len(usage_times) < config.max_usage_per_hour
    
    def _record_usage(self, strategy: FallbackStrategy, operation_name: str) -> None:
        """Record strategy usage for rate limiting."""
        rate_limit_key = f"{strategy.value}_{operation_name}"
        self.usage_counters[rate_limit_key].append(time.time())
    
    def _try_alternative_strategy(self, context: FallbackContext,
                                trigger: FallbackTrigger,
                                failed_strategies: List[FallbackStrategy]) -> FallbackResult:
        """Try alternative strategies when primary strategy fails."""
        applicable_strategies = self._get_applicable_strategies(context, trigger)
        
        # Remove failed strategies
        available_strategies = [s for s in applicable_strategies if s not in failed_strategies]
        
        if not available_strategies:
            return FallbackResult(
                strategy=FallbackStrategy.NONE,
                success=False,
                result=None,
                execution_time_ms=0,
                quality_impact=1.0,
                trigger=trigger,
                fallback_error=Exception("No alternative strategies available")
            )
        
        # Try the next best strategy
        ranked_strategies = self._rank_strategies(available_strategies, context, trigger)
        
        for strategy in ranked_strategies:
            if self._check_rate_limit(strategy, context.operation_name):
                return self.execute_fallback(context, trigger)
        
        # All alternatives exhausted
        return FallbackResult(
            strategy=FallbackStrategy.NONE,
            success=False,
            result=None,
            execution_time_ms=0,
            quality_impact=1.0,
            trigger=trigger,
            fallback_error=Exception("All alternative strategies failed or rate limited")
        )
    
    def _record_fallback_execution(self, result: FallbackResult, 
                                 context: FallbackContext) -> None:
        """Record fallback execution for learning and statistics."""
        with self._lock:
            # Store result
            self.fallback_history.append(result)
            
            # Update statistics
            self.fallback_statistics["total_executions"] += 1
            if result.success:
                self.fallback_statistics["successful_executions"] += 1
            
            # Update strategy performance
            if result.success:
                self.strategy_performance[result.strategy].append(result.execution_time_ms)
            
            # Update success rates
            self._update_strategy_success_rate(result.strategy, result.success)
            
            # Record usage for rate limiting
            self._record_usage(result.strategy, context.operation_name)
            
            # Learn context patterns
            self._learn_context_patterns(context, result)
    
    def _update_strategy_success_rate(self, strategy: FallbackStrategy, success: bool) -> None:
        """Update rolling success rate for strategy."""
        current_rate = self.strategy_success_rates.get(strategy, 0.5)
        
        # Exponential moving average with alpha=0.1
        alpha = 0.1
        new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * current_rate
        
        self.strategy_success_rates[strategy] = new_rate
    
    def _learn_context_patterns(self, context: FallbackContext, result: FallbackResult) -> None:
        """Learn patterns between context and successful strategies."""
        if not result.success:
            return
        
        # Create context signature
        context_signature = self._create_context_signature(context)
        
        # Store successful strategy for this context
        self.context_patterns[context_signature].append(result.strategy)
        
        # Keep only recent patterns (last 50)
        if len(self.context_patterns[context_signature]) > 50:
            self.context_patterns[context_signature] = self.context_patterns[context_signature][-50:]
    
    def _create_context_signature(self, context: FallbackContext) -> str:
        """Create a signature for context pattern learning."""
        signature_parts = [
            context.operation_name or "unknown",
            str(bool(context.error_classification)),
            str(bool(context.deadline_ms)),
            str(bool(context.quality_requirements))
        ]
        
        # Add system state indicators
        if context.system_state:
            cpu_high = context.system_state.get("cpu_usage", 0) > 70
            memory_high = context.system_state.get("memory_usage", 0) > 80
            signature_parts.extend([str(cpu_high), str(memory_high)])
        
        return "_".join(signature_parts)
    
    def _register_fallback_handlers(self) -> None:
        """Register default fallback handlers."""
        self.fallback_handlers[FallbackStrategy.QUANTUM_TO_CLASSICAL] = self._quantum_to_classical_handler
        self.fallback_handlers[FallbackStrategy.SIMPLIFIED_QUANTUM] = self._simplified_quantum_handler
        self.fallback_handlers[FallbackStrategy.HIGH_ACCURACY_TO_FAST] = self._high_accuracy_to_fast_handler
        self.fallback_handlers[FallbackStrategy.REAL_TIME_TO_CACHED] = self._real_time_to_cached_handler
        self.fallback_handlers[FallbackStrategy.FULL_TO_PARTIAL] = self._full_to_partial_handler
        self.fallback_handlers[FallbackStrategy.GRACEFUL_DEGRADATION] = self._graceful_degradation_handler
        self.fallback_handlers[FallbackStrategy.DEFAULT_VALUE] = self._default_value_handler
    
    def _quantum_to_classical_handler(self, context: FallbackContext) -> Any:
        """Handle quantum to classical fallback."""
        # Extract arguments for classical similarity computation
        if len(context.original_args) >= 2:
            # Import here to avoid circular dependencies
            from ..core.similarity.engines import SimilarityEngine
            
            engine = SimilarityEngine()
            
            # Use classical cosine similarity as fallback
            result = engine.compute_similarity(
                context.original_args[0],
                context.original_args[1],
                method="classical_cosine"
            )
            
            # Add metadata about fallback
            if isinstance(result, tuple):
                similarity, metadata = result
                metadata["fallback_method"] = "quantum_to_classical"
                metadata["quality_impact"] = 0.1
                return similarity, metadata
            else:
                return result, {"fallback_method": "quantum_to_classical", "quality_impact": 0.1}
        
        raise Exception("Invalid arguments for quantum to classical fallback")
    
    def _simplified_quantum_handler(self, context: FallbackContext) -> Any:
        """Handle simplified quantum computation fallback."""
        # Modify kwargs to use simplified quantum parameters
        simplified_kwargs = context.original_kwargs.copy()
        
        # Reduce quantum circuit complexity
        if 'n_qubits' in simplified_kwargs:
            simplified_kwargs['n_qubits'] = min(simplified_kwargs['n_qubits'], 2)
        if 'circuit_depth' in simplified_kwargs:
            simplified_kwargs['circuit_depth'] = min(simplified_kwargs['circuit_depth'], 5)
        if 'n_layers' in simplified_kwargs:
            simplified_kwargs['n_layers'] = 1
        
        # Re-execute with simplified parameters
        # This would call the original function with modified parameters
        raise NotImplementedError("Simplified quantum handler needs original function reference")
    
    def _high_accuracy_to_fast_handler(self, context: FallbackContext) -> Any:
        """Handle high accuracy to fast computation fallback."""
        # Switch to faster, less accurate similarity methods
        if "similarity" in context.operation_name:
            from ..core.similarity.engines import SimilarityEngine
            
            engine = SimilarityEngine()
            
            # Use fast approximate similarity
            result = engine.compute_similarity(
                context.original_args[0],
                context.original_args[1],
                method="approximate_cosine"
            )
            
            if isinstance(result, tuple):
                similarity, metadata = result
                metadata["fallback_method"] = "high_accuracy_to_fast"
                metadata["quality_impact"] = 0.15
                return similarity, metadata
            else:
                return result, {"fallback_method": "high_accuracy_to_fast", "quality_impact": 0.15}
        
        raise Exception("High accuracy to fast fallback not applicable")
    
    def _real_time_to_cached_handler(self, context: FallbackContext) -> Any:
        """Handle real-time to cached result fallback."""
        # Try to get cached result
        cache_key = self._generate_cache_key(context)
        
        # This would integrate with the caching system
        # For now, return a placeholder
        return None, {"fallback_method": "real_time_to_cached", "cache_hit": False}
    
    def _full_to_partial_handler(self, context: FallbackContext) -> Any:
        """Handle full to partial computation fallback."""
        # Reduce scope of computation
        modified_args = list(context.original_args)
        modified_kwargs = context.original_kwargs.copy()
        
        # Reduce batch size if applicable
        if 'batch_size' in modified_kwargs:
            modified_kwargs['batch_size'] = max(1, modified_kwargs['batch_size'] // 2)
        
        # Reduce result count if applicable  
        if 'top_k' in modified_kwargs:
            modified_kwargs['top_k'] = max(1, modified_kwargs['top_k'] // 2)
        
        # Reduce input size if it's a list
        if modified_args and isinstance(modified_args[0], list) and len(modified_args[0]) > 5:
            modified_args[0] = modified_args[0][:len(modified_args[0])//2]
        
        # This would re-execute with reduced scope
        # For now, return a placeholder
        return None, {"fallback_method": "full_to_partial", "scope_reduction": 0.5}
    
    def _graceful_degradation_handler(self, context: FallbackContext) -> Any:
        """Handle graceful service degradation."""
        # Return a degraded but functional result
        degraded_result = {
            "status": "degraded",
            "message": "Service operating in degraded mode",
            "quality_impact": 0.3,
            "fallback_method": "graceful_degradation"
        }
        
        # Try to provide some useful result even in degraded mode
        if "similarity" in context.operation_name:
            degraded_result["similarity"] = 0.5  # Neutral similarity
        
        return degraded_result
    
    def _default_value_handler(self, context: FallbackContext) -> Any:
        """Handle default value fallback."""
        # Return safe default based on operation type
        if "similarity" in context.operation_name:
            return 0.0, {"fallback_method": "default_value", "quality_impact": 1.0}
        elif "embedding" in context.operation_name:
            return [], {"fallback_method": "default_value", "quality_impact": 1.0}
        else:
            return None, {"fallback_method": "default_value", "quality_impact": 1.0}
    
    def _generate_cache_key(self, context: FallbackContext) -> str:
        """Generate cache key for fallback operations."""
        # Simple cache key generation
        args_hash = hash(str(context.original_args))
        kwargs_hash = hash(str(sorted(context.original_kwargs.items())))
        
        return f"{context.operation_name}_{args_hash}_{kwargs_hash}"
    
    def _calculate_strategy_distribution(self, fallbacks: List[FallbackResult]) -> Dict[str, int]:
        """Calculate distribution of fallback strategies used."""
        distribution = defaultdict(int)
        for fb in fallbacks:
            distribution[fb.strategy.value] += 1
        return dict(distribution)
    
    def _calculate_trigger_distribution(self, fallbacks: List[FallbackResult]) -> Dict[str, int]:
        """Calculate distribution of fallback triggers."""
        distribution = defaultdict(int)
        for fb in fallbacks:
            distribution[fb.trigger.value] += 1
        return dict(distribution)
    
    def _get_most_effective_strategies(self, fallbacks: List[FallbackResult]) -> List[Dict[str, Any]]:
        """Get most effective strategies based on recent performance."""
        strategy_stats = defaultdict(lambda: {"total": 0, "successful": 0, "avg_time": []})
        
        for fb in fallbacks:
            stats = strategy_stats[fb.strategy.value]
            stats["total"] += 1
            if fb.success:
                stats["successful"] += 1
                stats["avg_time"].append(fb.execution_time_ms)
        
        effectiveness = []
        for strategy, stats in strategy_stats.items():
            if stats["total"] > 0:
                success_rate = stats["successful"] / stats["total"]
                avg_time = np.mean(stats["avg_time"]) if stats["avg_time"] else 0
                
                effectiveness.append({
                    "strategy": strategy,
                    "success_rate": success_rate,
                    "average_time_ms": avg_time,
                    "usage_count": stats["total"]
                })
        
        # Sort by success rate, then by average time
        effectiveness.sort(key=lambda x: (x["success_rate"], -x["average_time_ms"]), reverse=True)
        
        return effectiveness[:5]  # Top 5 most effective
    
    def _initialize_fallback_configs(self) -> Dict[FallbackStrategy, FallbackConfig]:
        """Initialize fallback strategy configurations."""
        configs = {}
        
        # Quantum fallbacks
        configs[FallbackStrategy.QUANTUM_TO_CLASSICAL] = FallbackConfig(
            strategy=FallbackStrategy.QUANTUM_TO_CLASSICAL,
            triggers=[FallbackTrigger.ERROR_OCCURRED, FallbackTrigger.TIMEOUT],
            priority=1,
            quality_impact=0.1,
            performance_gain=0.8,
            resource_reduction=0.6,
            success_probability=0.95,
            max_usage_per_hour=100,
            prerequisites=["classical_similarity_available"]
        )
        
        configs[FallbackStrategy.SIMPLIFIED_QUANTUM] = FallbackConfig(
            strategy=FallbackStrategy.SIMPLIFIED_QUANTUM,
            triggers=[FallbackTrigger.PERFORMANCE_THRESHOLD, FallbackTrigger.RESOURCE_EXHAUSTION],
            priority=2,
            quality_impact=0.2,
            performance_gain=0.5,
            resource_reduction=0.4,
            success_probability=0.8,
            max_usage_per_hour=50
        )
        
        # Classical fallbacks
        configs[FallbackStrategy.HIGH_ACCURACY_TO_FAST] = FallbackConfig(
            strategy=FallbackStrategy.HIGH_ACCURACY_TO_FAST,
            triggers=[FallbackTrigger.PERFORMANCE_THRESHOLD, FallbackTrigger.TIMEOUT],
            priority=2,
            quality_impact=0.15,
            performance_gain=0.7,
            resource_reduction=0.3,
            success_probability=0.9,
            max_usage_per_hour=200
        )
        
        # System fallbacks
        configs[FallbackStrategy.REAL_TIME_TO_CACHED] = FallbackConfig(
            strategy=FallbackStrategy.REAL_TIME_TO_CACHED,
            triggers=[FallbackTrigger.PERFORMANCE_THRESHOLD, FallbackTrigger.RESOURCE_EXHAUSTION],
            priority=1,
            quality_impact=0.0,  # No quality impact if cache is fresh
            performance_gain=0.95,
            resource_reduction=0.9,
            success_probability=0.7,  # Depends on cache hit rate
            max_usage_per_hour=1000,
            prerequisites=["cache_available"]
        )
        
        configs[FallbackStrategy.FULL_TO_PARTIAL] = FallbackConfig(
            strategy=FallbackStrategy.FULL_TO_PARTIAL,
            triggers=[FallbackTrigger.RESOURCE_EXHAUSTION, FallbackTrigger.TIMEOUT],
            priority=3,
            quality_impact=0.3,
            performance_gain=0.6,
            resource_reduction=0.5,
            success_probability=0.85,
            max_usage_per_hour=50
        )
        
        configs[FallbackStrategy.GRACEFUL_DEGRADATION] = FallbackConfig(
            strategy=FallbackStrategy.GRACEFUL_DEGRADATION,
            triggers=[FallbackTrigger.ERROR_OCCURRED, FallbackTrigger.RESOURCE_EXHAUSTION],
            priority=4,
            quality_impact=0.5,
            performance_gain=0.3,
            resource_reduction=0.2,
            success_probability=0.9,
            max_usage_per_hour=20
        )
        
        configs[FallbackStrategy.DEFAULT_VALUE] = FallbackConfig(
            strategy=FallbackStrategy.DEFAULT_VALUE,
            triggers=[FallbackTrigger.ERROR_OCCURRED],
            priority=10,  # Last resort
            quality_impact=1.0,  # Complete quality loss
            performance_gain=0.99,
            resource_reduction=0.99,
            success_probability=1.0,
            max_usage_per_hour=10
        )
        
        return configs


__all__ = [
    "FallbackStrategy",
    "FallbackTrigger",
    "FallbackConfig",
    "FallbackResult",
    "FallbackContext",
    "AdvancedFallbackManager"
]
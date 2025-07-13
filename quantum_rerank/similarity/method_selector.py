"""
Intelligent method selection for similarity computation.

This module provides adaptive method selection based on performance requirements,
accuracy targets, and historical performance data.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json

from ..utils import get_logger


@dataclass
class MethodProfile:
    """Profile of a similarity computation method."""
    name: str
    latency_ms: float
    accuracy: float
    memory_usage_mb: float
    batch_optimized: bool
    use_cases: List[str]
    resource_requirements: Dict[str, str]
    
    def matches_requirements(
        self,
        max_latency_ms: float,
        min_accuracy: float,
        max_memory_mb: float
    ) -> bool:
        """Check if method matches requirements."""
        return (
            self.latency_ms <= max_latency_ms and
            self.accuracy >= min_accuracy and
            self.memory_usage_mb <= max_memory_mb
        )


@dataclass
class MethodSelectionContext:
    """Context for method selection decision."""
    query_size: int
    batch_size: int
    embedding_dim: int
    accuracy_requirement: float = 0.9
    latency_requirement_ms: float = 100.0
    memory_limit_mb: Optional[float] = None
    domain: Optional[str] = None
    previous_method: Optional[str] = None
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert context to feature vector for ML-based selection."""
        features = [
            np.log1p(self.query_size),
            np.log1p(self.batch_size),
            np.log1p(self.embedding_dim),
            self.accuracy_requirement,
            np.log1p(self.latency_requirement_ms),
            1.0 if self.batch_size > 50 else 0.0,
            1.0 if self.accuracy_requirement > 0.95 else 0.0,
            1.0 if self.latency_requirement_ms < 50 else 0.0
        ]
        return np.array(features)


class SelectionStrategy(Enum):
    """Method selection strategies."""
    PERFORMANCE_FIRST = "performance_first"
    ACCURACY_FIRST = "accuracy_first"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"


class MethodSelector:
    """
    Intelligent selector for optimal similarity computation method.
    
    This selector uses contextual information, performance history, and
    requirements to choose the best method for each query.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Method profiles
        self.method_profiles = self._initialize_method_profiles()
        
        # Performance history
        self.performance_history: Dict[str, List[float]] = {}
        self.selection_history: List[Tuple[MethodSelectionContext, str]] = []
        
        # Selection strategy
        self.strategy = SelectionStrategy.ADAPTIVE
        
        # Performance targets
        self.default_latency_target_ms = 100.0
        self.default_memory_limit_mb = 2000.0
        
        # Adaptive weights for scoring
        self.scoring_weights = {
            "latency": 0.4,
            "accuracy": 0.4,
            "reliability": 0.2
        }
    
    def _initialize_method_profiles(self) -> Dict[str, MethodProfile]:
        """Initialize method profiles with characteristics."""
        return {
            "classical_fast": MethodProfile(
                name="classical_fast",
                latency_ms=15.0,
                accuracy=0.85,
                memory_usage_mb=50.0,
                batch_optimized=True,
                use_cases=["high_throughput", "large_batch", "real_time"],
                resource_requirements={"cpu": "low", "memory": "minimal"}
            ),
            "classical_accurate": MethodProfile(
                name="classical_accurate",
                latency_ms=25.0,
                accuracy=0.88,
                memory_usage_mb=80.0,
                batch_optimized=True,
                use_cases=["general_purpose", "medium_accuracy"],
                resource_requirements={"cpu": "low", "memory": "low"}
            ),
            "quantum_precise": MethodProfile(
                name="quantum_precise",
                latency_ms=95.0,
                accuracy=0.99,
                memory_usage_mb=200.0,
                batch_optimized=False,
                use_cases=["high_accuracy", "small_batch", "research"],
                resource_requirements={"cpu": "high", "memory": "moderate"}
            ),
            "quantum_approximate": MethodProfile(
                name="quantum_approximate",
                latency_ms=45.0,
                accuracy=0.93,
                memory_usage_mb=150.0,
                batch_optimized=True,
                use_cases=["balanced", "quantum_advantage"],
                resource_requirements={"cpu": "medium", "memory": "moderate"}
            ),
            "hybrid_balanced": MethodProfile(
                name="hybrid_balanced",
                latency_ms=60.0,
                accuracy=0.93,
                memory_usage_mb=180.0,
                batch_optimized=True,
                use_cases=["general_purpose", "production"],
                resource_requirements={"cpu": "medium", "memory": "moderate"}
            ),
            "hybrid_batch": MethodProfile(
                name="hybrid_batch",
                latency_ms=20.0,
                accuracy=0.90,
                memory_usage_mb=100.0,
                batch_optimized=True,
                use_cases=["large_scale", "production", "batch_processing"],
                resource_requirements={"cpu": "medium", "memory": "high"}
            ),
            "ensemble": MethodProfile(
                name="ensemble",
                latency_ms=120.0,
                accuracy=0.96,
                memory_usage_mb=300.0,
                batch_optimized=False,
                use_cases=["high_accuracy", "consensus", "critical"],
                resource_requirements={"cpu": "high", "memory": "high"}
            )
        }
    
    def select_method(
        self,
        context: MethodSelectionContext,
        performance_cache: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Select optimal method based on context and requirements.
        
        Args:
            context: Selection context with requirements
            performance_cache: Recent performance data for methods
            
        Returns:
            Selected method name
        """
        # Quick selection for edge cases
        if context.batch_size > 100 and context.latency_requirement_ms < 50:
            return "hybrid_batch"
        
        if context.accuracy_requirement > 0.98:
            if context.batch_size <= 10:
                return "quantum_precise"
            else:
                return "ensemble"
        
        if context.latency_requirement_ms < 20:
            return "classical_fast"
        
        # Score all methods
        method_scores = self._score_methods(context, performance_cache)
        
        # Select best method
        best_method = max(method_scores.items(), key=lambda x: x[1])[0]
        
        # Log selection
        self._log_selection(context, best_method, method_scores)
        
        # Update history
        self.selection_history.append((context, best_method))
        if len(self.selection_history) > 1000:
            self.selection_history = self.selection_history[-1000:]
        
        return best_method
    
    def _score_methods(
        self,
        context: MethodSelectionContext,
        performance_cache: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Score all methods based on context."""
        scores = {}
        
        for method_name, profile in self.method_profiles.items():
            # Base score from profile
            score = self._compute_base_score(profile, context)
            
            # Adjust based on recent performance
            if performance_cache and method_name in performance_cache:
                score = self._adjust_score_with_performance(
                    score, profile, performance_cache[method_name]
                )
            
            # Apply strategy-specific adjustments
            score = self._apply_strategy_adjustment(score, profile, context)
            
            scores[method_name] = score
        
        return scores
    
    def _compute_base_score(
        self,
        profile: MethodProfile,
        context: MethodSelectionContext
    ) -> float:
        """Compute base score for method."""
        score = 0.0
        
        # Latency score (inverse relationship)
        if profile.latency_ms <= context.latency_requirement_ms:
            latency_score = 1.0 - (profile.latency_ms / context.latency_requirement_ms)
            score += self.scoring_weights["latency"] * latency_score
        else:
            # Penalty for not meeting latency requirement
            score -= 0.5
        
        # Accuracy score
        if profile.accuracy >= context.accuracy_requirement:
            accuracy_score = profile.accuracy
            score += self.scoring_weights["accuracy"] * accuracy_score
        else:
            # Penalty for not meeting accuracy requirement
            score -= 0.3
        
        # Batch optimization bonus
        if profile.batch_optimized and context.batch_size > 20:
            score += 0.1
        
        # Use case matching bonus
        if self._matches_use_case(profile, context):
            score += 0.1
        
        return max(0.0, score)
    
    def _adjust_score_with_performance(
        self,
        base_score: float,
        profile: MethodProfile,
        recent_performance: Dict[str, List[float]]
    ) -> float:
        """Adjust score based on recent performance data."""
        if "latency_ms" in recent_performance and recent_performance["latency_ms"]:
            # Check if recent performance matches profile
            recent_latency = np.mean(recent_performance["latency_ms"][-10:])
            
            if recent_latency < profile.latency_ms:
                # Better than expected
                base_score += 0.05
            elif recent_latency > profile.latency_ms * 1.5:
                # Worse than expected
                base_score -= 0.1
        
        # Reliability score based on consistency
        if "latency_ms" in recent_performance and len(recent_performance["latency_ms"]) > 5:
            latency_std = np.std(recent_performance["latency_ms"][-10:])
            latency_mean = np.mean(recent_performance["latency_ms"][-10:])
            
            if latency_mean > 0:
                cv = latency_std / latency_mean  # Coefficient of variation
                reliability_score = max(0, 1.0 - cv)
                base_score += self.scoring_weights["reliability"] * reliability_score * 0.5
        
        return base_score
    
    def _apply_strategy_adjustment(
        self,
        score: float,
        profile: MethodProfile,
        context: MethodSelectionContext
    ) -> float:
        """Apply strategy-specific score adjustments."""
        if self.strategy == SelectionStrategy.PERFORMANCE_FIRST:
            # Boost score for faster methods
            if profile.latency_ms < 30:
                score *= 1.2
        
        elif self.strategy == SelectionStrategy.ACCURACY_FIRST:
            # Boost score for more accurate methods
            if profile.accuracy > 0.95:
                score *= 1.2
        
        elif self.strategy == SelectionStrategy.ADAPTIVE:
            # Adaptive based on context
            if context.batch_size > 50:
                # Prefer batch-optimized methods
                if profile.batch_optimized:
                    score *= 1.1
            
            if context.embedding_dim > 1024:
                # Prefer methods that handle high dimensions well
                if "high_dimensional" in profile.use_cases:
                    score *= 1.1
        
        return score
    
    def _matches_use_case(
        self,
        profile: MethodProfile,
        context: MethodSelectionContext
    ) -> bool:
        """Check if method matches the use case."""
        # Large batch processing
        if context.batch_size > 50 and "large_batch" in profile.use_cases:
            return True
        
        # High accuracy requirement
        if context.accuracy_requirement > 0.95 and "high_accuracy" in profile.use_cases:
            return True
        
        # Real-time requirement
        if context.latency_requirement_ms < 30 and "real_time" in profile.use_cases:
            return True
        
        # General purpose fallback
        if "general_purpose" in profile.use_cases:
            return True
        
        return False
    
    def _log_selection(
        self,
        context: MethodSelectionContext,
        selected_method: str,
        scores: Dict[str, float]
    ) -> None:
        """Log method selection decision."""
        self.logger.debug(
            f"Selected method '{selected_method}' for batch_size={context.batch_size}, "
            f"accuracy_req={context.accuracy_requirement:.2f}, "
            f"latency_req={context.latency_requirement_ms:.0f}ms"
        )
        
        # Log top 3 methods
        sorted_methods = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        self.logger.debug(f"Top methods: {sorted_methods}")
    
    def update_performance_history(
        self,
        method: str,
        latency_ms: float,
        accuracy_estimate: float
    ) -> None:
        """Update performance history for a method."""
        if method not in self.performance_history:
            self.performance_history[method] = []
        
        self.performance_history[method].append({
            "latency_ms": latency_ms,
            "accuracy": accuracy_estimate,
            "timestamp": np.datetime64('now')
        })
        
        # Keep last 100 entries
        if len(self.performance_history[method]) > 100:
            self.performance_history[method] = self.performance_history[method][-100:]
    
    def update_scoring_weights(
        self,
        latency_weight: float = None,
        accuracy_weight: float = None,
        reliability_weight: float = None
    ) -> None:
        """Update scoring weights for method selection."""
        if latency_weight is not None:
            self.scoring_weights["latency"] = latency_weight
        
        if accuracy_weight is not None:
            self.scoring_weights["accuracy"] = accuracy_weight
        
        if reliability_weight is not None:
            self.scoring_weights["reliability"] = reliability_weight
        
        # Normalize weights
        total_weight = sum(self.scoring_weights.values())
        if total_weight > 0:
            for key in self.scoring_weights:
                self.scoring_weights[key] /= total_weight
    
    def set_strategy(self, strategy: SelectionStrategy) -> None:
        """Set selection strategy."""
        self.strategy = strategy
        self.logger.info(f"Method selection strategy set to: {strategy.value}")
    
    def update_performance_targets(
        self,
        latency_ms: Optional[float] = None,
        memory_gb: Optional[float] = None
    ) -> None:
        """Update default performance targets."""
        if latency_ms is not None:
            self.default_latency_target_ms = latency_ms
        
        if memory_gb is not None:
            self.default_memory_limit_mb = memory_gb * 1024
    
    def get_method_recommendation(
        self,
        use_case: str
    ) -> List[str]:
        """Get method recommendations for specific use case."""
        recommendations = []
        
        for method_name, profile in self.method_profiles.items():
            if use_case in profile.use_cases:
                recommendations.append(method_name)
        
        # Sort by default preference
        preference_order = [
            "hybrid_balanced",
            "quantum_approximate", 
            "classical_accurate",
            "classical_fast",
            "hybrid_batch",
            "quantum_precise",
            "ensemble"
        ]
        
        recommendations.sort(
            key=lambda x: preference_order.index(x) if x in preference_order else 999
        )
        
        return recommendations
    
    def analyze_selection_history(self) -> Dict[str, Any]:
        """Analyze method selection history for insights."""
        if not self.selection_history:
            return {"message": "No selection history available"}
        
        # Method usage frequency
        method_counts = {}
        context_patterns = {
            "small_batch": 0,
            "large_batch": 0,
            "high_accuracy": 0,
            "low_latency": 0
        }
        
        for context, method in self.selection_history:
            # Count method usage
            method_counts[method] = method_counts.get(method, 0) + 1
            
            # Analyze context patterns
            if context.batch_size < 10:
                context_patterns["small_batch"] += 1
            elif context.batch_size > 50:
                context_patterns["large_batch"] += 1
            
            if context.accuracy_requirement > 0.95:
                context_patterns["high_accuracy"] += 1
            
            if context.latency_requirement_ms < 50:
                context_patterns["low_latency"] += 1
        
        total_selections = len(self.selection_history)
        
        return {
            "total_selections": total_selections,
            "method_usage": {
                method: f"{count/total_selections*100:.1f}%"
                for method, count in method_counts.items()
            },
            "context_patterns": {
                pattern: f"{count/total_selections*100:.1f}%"
                for pattern, count in context_patterns.items()
            },
            "most_used_method": max(method_counts.items(), key=lambda x: x[1])[0]
        }


__all__ = [
    "MethodProfile",
    "MethodSelectionContext", 
    "SelectionStrategy",
    "MethodSelector"
]
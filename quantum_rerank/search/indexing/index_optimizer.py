"""
Index performance optimizer for automatic parameter tuning.
"""

import time
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np

from ..backends.faiss_backend import FAISSBackend
from ...utils import get_logger


@dataclass
class OptimizationTarget:
    """Optimization target specification."""
    metric: str  # "latency", "accuracy", "memory", "throughput"
    target_value: float
    weight: float = 1.0
    tolerance: float = 0.1


@dataclass
class ParameterRange:
    """Valid range for a parameter."""
    name: str
    min_value: Any
    max_value: Any
    step: Optional[Any] = None
    values: Optional[List[Any]] = None  # Discrete values


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    optimized_parameters: Dict[str, Any]
    performance_improvement: Dict[str, float]
    optimization_time_seconds: float
    iterations: int
    final_score: float


class IndexOptimizer:
    """
    Intelligent index optimizer for automatic parameter tuning.
    
    This optimizer uses performance feedback to automatically tune
    index parameters for optimal search performance.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.parameter_ranges = self._initialize_parameter_ranges()
        self.optimization_history: List[Dict[str, Any]] = []
    
    def optimize_index_parameters(self, backend: FAISSBackend,
                                 targets: List[OptimizationTarget],
                                 test_queries: Optional[np.ndarray] = None,
                                 max_iterations: int = 20) -> OptimizationResult:
        """
        Optimize index parameters for specified targets.
        
        Args:
            backend: FAISS backend to optimize
            targets: Optimization targets
            test_queries: Test queries for performance evaluation
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimization result with best parameters
        """
        start_time = time.time()
        
        # Get current parameters
        current_params = self._extract_current_parameters(backend)
        best_params = current_params.copy()
        best_score = 0.0
        
        # Generate test queries if not provided
        if test_queries is None:
            test_queries = self._generate_test_queries(backend)
        
        self.logger.info(f"Starting parameter optimization with {len(targets)} targets")
        
        # Optimization loop
        for iteration in range(max_iterations):
            # Generate parameter candidates
            candidate_params = self._generate_parameter_candidates(
                current_params, iteration, max_iterations
            )
            
            best_candidate_score = 0.0
            best_candidate_params = current_params
            
            # Evaluate each candidate
            for params in candidate_params:
                try:
                    # Apply parameters
                    self._apply_parameters(backend, params)
                    
                    # Measure performance
                    performance = self._measure_performance(backend, test_queries)
                    
                    # Calculate score based on targets
                    score = self._calculate_score(performance, targets)
                    
                    if score > best_candidate_score:
                        best_candidate_score = score
                        best_candidate_params = params
                        
                except Exception as e:
                    self.logger.warning(f"Parameter evaluation failed: {e}")
                    continue
            
            # Update best parameters if improved
            if best_candidate_score > best_score:
                best_score = best_candidate_score
                best_params = best_candidate_params
                current_params = best_candidate_params
                
                self.logger.debug(f"Iteration {iteration}: improved score to {best_score:.3f}")
            else:
                self.logger.debug(f"Iteration {iteration}: no improvement")
            
            # Early stopping if target achieved
            if best_score > 0.95:  # 95% of perfect score
                self.logger.info(f"Target achieved early at iteration {iteration}")
                break
        
        # Apply best parameters
        self._apply_parameters(backend, best_params)
        
        # Calculate performance improvement
        final_performance = self._measure_performance(backend, test_queries)
        initial_performance = self._measure_performance_with_params(
            backend, current_params, test_queries
        )
        
        improvement = self._calculate_improvement(initial_performance, final_performance)
        
        optimization_time = time.time() - start_time
        
        result = OptimizationResult(
            optimized_parameters=best_params,
            performance_improvement=improvement,
            optimization_time_seconds=optimization_time,
            iterations=iteration + 1,
            final_score=best_score
        )
        
        # Record optimization
        self.optimization_history.append({
            "timestamp": time.time(),
            "targets": [target.__dict__ for target in targets],
            "result": result.__dict__
        })
        
        self.logger.info(
            f"Optimization completed: {result.iterations} iterations, "
            f"{optimization_time:.2f}s, score: {best_score:.3f}"
        )
        
        return result
    
    def optimize_for_latency(self, backend: FAISSBackend,
                           target_latency_ms: float,
                           test_queries: Optional[np.ndarray] = None) -> OptimizationResult:
        """Optimize index specifically for low latency."""
        targets = [
            OptimizationTarget(metric="latency", target_value=target_latency_ms, weight=1.0)
        ]
        return self.optimize_index_parameters(backend, targets, test_queries)
    
    def optimize_for_accuracy(self, backend: FAISSBackend,
                            target_accuracy: float,
                            test_queries: Optional[np.ndarray] = None) -> OptimizationResult:
        """Optimize index specifically for high accuracy."""
        targets = [
            OptimizationTarget(metric="accuracy", target_value=target_accuracy, weight=1.0)
        ]
        return self.optimize_index_parameters(backend, targets, test_queries)
    
    def optimize_for_throughput(self, backend: FAISSBackend,
                              target_qps: float,
                              test_queries: Optional[np.ndarray] = None) -> OptimizationResult:
        """Optimize index specifically for high throughput."""
        targets = [
            OptimizationTarget(metric="throughput", target_value=target_qps, weight=1.0)
        ]
        return self.optimize_index_parameters(backend, targets, test_queries)
    
    def _extract_current_parameters(self, backend: FAISSBackend) -> Dict[str, Any]:
        """Extract current tunable parameters from backend."""
        params = {}
        
        if hasattr(backend, 'index'):
            index = backend.index
            
            # FAISS IVF parameters
            if hasattr(index, 'nprobe'):
                params['nprobe'] = index.nprobe
            
            # FAISS HNSW parameters
            if hasattr(index, 'hnsw'):
                hnsw = index.hnsw
                if hasattr(hnsw, 'efSearch'):
                    params['efSearch'] = hnsw.efSearch
        
        # Backend-level parameters
        if hasattr(backend, 'nprobe'):
            params['nprobe'] = backend.nprobe
        if hasattr(backend, 'ef_search'):
            params['efSearch'] = backend.ef_search
        
        return params
    
    def _apply_parameters(self, backend: FAISSBackend, params: Dict[str, Any]) -> None:
        """Apply parameters to backend."""
        if hasattr(backend, 'index') and backend.index is not None:
            index = backend.index
            
            # Apply IVF parameters
            if 'nprobe' in params and hasattr(index, 'nprobe'):
                index.nprobe = params['nprobe']
                backend.nprobe = params['nprobe']
            
            # Apply HNSW parameters
            if 'efSearch' in params and hasattr(index, 'hnsw'):
                index.hnsw.efSearch = params['efSearch']
                backend.ef_search = params['efSearch']
    
    def _generate_parameter_candidates(self, current_params: Dict[str, Any],
                                     iteration: int, max_iterations: int) -> List[Dict[str, Any]]:
        """Generate parameter candidates for evaluation."""
        candidates = []
        
        # Calculate exploration factor (high initially, decreases over time)
        exploration_factor = 1.0 - (iteration / max_iterations)
        
        for param_name, current_value in current_params.items():
            if param_name not in self.parameter_ranges:
                continue
            
            param_range = self.parameter_ranges[param_name]
            
            # Generate variations around current value
            if param_range.values:
                # Discrete values
                candidates.extend(self._generate_discrete_candidates(
                    param_name, current_value, param_range, exploration_factor
                ))
            else:
                # Continuous range
                candidates.extend(self._generate_continuous_candidates(
                    param_name, current_value, param_range, exploration_factor
                ))
        
        # Add current parameters as baseline
        candidates.append(current_params.copy())
        
        return candidates
    
    def _generate_discrete_candidates(self, param_name: str, current_value: Any,
                                    param_range: ParameterRange, 
                                    exploration_factor: float) -> List[Dict[str, Any]]:
        """Generate candidates for discrete parameter values."""
        candidates = []
        
        if current_value not in param_range.values:
            # Use closest valid value
            current_value = min(param_range.values, key=lambda x: abs(x - current_value))
        
        current_idx = param_range.values.index(current_value)
        
        # Generate neighbors based on exploration factor
        max_distance = max(1, int(len(param_range.values) * exploration_factor * 0.5))
        
        for distance in range(1, max_distance + 1):
            # Try values before and after current
            for direction in [-1, 1]:
                new_idx = current_idx + (direction * distance)
                if 0 <= new_idx < len(param_range.values):
                    candidate = {param_name: param_range.values[new_idx]}
                    candidates.append(candidate)
        
        return candidates
    
    def _generate_continuous_candidates(self, param_name: str, current_value: float,
                                      param_range: ParameterRange,
                                      exploration_factor: float) -> List[Dict[str, Any]]:
        """Generate candidates for continuous parameter values."""
        candidates = []
        
        # Calculate search range
        value_range = param_range.max_value - param_range.min_value
        max_step = value_range * exploration_factor * 0.2
        
        # Generate variations
        step_sizes = [max_step * factor for factor in [0.1, 0.3, 0.5, 1.0]]
        
        for step_size in step_sizes:
            for direction in [-1, 1]:
                new_value = current_value + (direction * step_size)
                
                # Clamp to valid range
                new_value = max(param_range.min_value, min(param_range.max_value, new_value))
                
                # Apply step constraint if specified
                if param_range.step:
                    new_value = round(new_value / param_range.step) * param_range.step
                
                candidate = {param_name: new_value}
                candidates.append(candidate)
        
        return candidates
    
    def _measure_performance(self, backend: FAISSBackend, 
                           test_queries: np.ndarray) -> Dict[str, float]:
        """Measure current performance metrics."""
        metrics = {}
        
        # Measure latency
        start_time = time.time()
        for query in test_queries[:10]:  # Use first 10 queries for latency test
            backend.search(query, 10)
        latency_ms = ((time.time() - start_time) / 10) * 1000
        metrics["latency"] = latency_ms
        
        # Measure throughput
        start_time = time.time()
        query_count = min(100, len(test_queries))
        for i in range(query_count):
            backend.search(test_queries[i], 10)
        throughput_time = time.time() - start_time
        metrics["throughput"] = query_count / throughput_time if throughput_time > 0 else 0
        
        # Memory usage
        metrics["memory"] = backend.get_memory_usage_mb()
        
        # Accuracy estimation (simplified)
        # In practice, this would require ground truth data
        metrics["accuracy"] = 0.95  # Placeholder
        
        return metrics
    
    def _measure_performance_with_params(self, backend: FAISSBackend,
                                       params: Dict[str, Any],
                                       test_queries: np.ndarray) -> Dict[str, float]:
        """Measure performance with specific parameters."""
        # Save current parameters
        original_params = self._extract_current_parameters(backend)
        
        try:
            # Apply test parameters
            self._apply_parameters(backend, params)
            
            # Measure performance
            performance = self._measure_performance(backend, test_queries)
            
            return performance
        
        finally:
            # Restore original parameters
            self._apply_parameters(backend, original_params)
    
    def _calculate_score(self, performance: Dict[str, float],
                        targets: List[OptimizationTarget]) -> float:
        """Calculate optimization score based on targets."""
        total_score = 0.0
        total_weight = 0.0
        
        for target in targets:
            if target.metric not in performance:
                continue
            
            actual_value = performance[target.metric]
            target_value = target.target_value
            
            # Calculate normalized score (0-1 range)
            if target.metric == "latency":
                # Lower is better
                score = min(1.0, target_value / max(actual_value, 0.1))
            elif target.metric in ["accuracy", "throughput"]:
                # Higher is better
                score = min(1.0, actual_value / target_value)
            elif target.metric == "memory":
                # Lower is better (within reasonable bounds)
                score = min(1.0, target_value / max(actual_value, 0.1))
            else:
                score = 0.5  # Default neutral score
            
            # Apply tolerance
            if abs(actual_value - target_value) / target_value <= target.tolerance:
                score = 1.0  # Perfect score if within tolerance
            
            total_score += score * target.weight
            total_weight += target.weight
        
        return total_score / max(total_weight, 1.0)
    
    def _calculate_improvement(self, initial: Dict[str, float],
                             final: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance improvement percentages."""
        improvement = {}
        
        for metric in initial:
            if metric in final:
                initial_val = initial[metric]
                final_val = final[metric]
                
                if initial_val > 0:
                    if metric == "latency" or metric == "memory":
                        # Lower is better
                        improvement[f"{metric}_improvement_percent"] = (
                            (initial_val - final_val) / initial_val * 100
                        )
                    else:
                        # Higher is better
                        improvement[f"{metric}_improvement_percent"] = (
                            (final_val - initial_val) / initial_val * 100
                        )
        
        return improvement
    
    def _generate_test_queries(self, backend: FAISSBackend,
                              count: int = 100) -> np.ndarray:
        """Generate test queries for performance evaluation."""
        # Get embedding dimension from backend
        if hasattr(backend, 'dimension') and backend.dimension:
            dimension = backend.dimension
        else:
            dimension = 512  # Default dimension
        
        # Generate random normalized queries
        queries = np.random.randn(count, dimension).astype(np.float32)
        
        # Normalize to unit vectors
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / (norms + 1e-8)
        
        return queries
    
    def _initialize_parameter_ranges(self) -> Dict[str, ParameterRange]:
        """Initialize valid parameter ranges."""
        return {
            "nprobe": ParameterRange(
                name="nprobe",
                min_value=1,
                max_value=512,
                values=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
            ),
            "efSearch": ParameterRange(
                name="efSearch",
                min_value=16,
                max_value=800,
                values=[16, 32, 50, 64, 100, 128, 200, 256, 400, 512, 800]
            )
        }


__all__ = [
    "OptimizationTarget",
    "ParameterRange",
    "OptimizationResult",
    "IndexOptimizer"
]
"""
Result aggregation and validation for multi-method similarity.

This module provides aggregation strategies for combining results from multiple
similarity methods and validating consistency across methods.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import warnings

from ..utils import get_logger


class AggregationStrategy(Enum):
    """Available aggregation strategies."""
    MEAN = "mean"
    WEIGHTED_AVERAGE = "weighted_average"
    MEDIAN = "median"
    MAX = "max"
    MIN = "min"
    TRIMMED_MEAN = "trimmed_mean"
    RANK_FUSION = "rank_fusion"


class ConsensusAlgorithm(Enum):
    """Consensus algorithms for multi-method results."""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    BORDA_COUNT = "borda_count"
    KEMENY_YOUNG = "kemeny_young"


@dataclass
class AggregationConfig:
    """Configuration for result aggregation."""
    strategy: AggregationStrategy = AggregationStrategy.WEIGHTED_AVERAGE
    consensus_algorithm: ConsensusAlgorithm = ConsensusAlgorithm.WEIGHTED_VOTE
    outlier_detection: bool = True
    outlier_threshold: float = 2.0  # Standard deviations
    normalize_scores: bool = True
    min_methods_required: int = 2


class ResultAggregator:
    """
    Aggregates and validates results from multiple similarity methods.
    
    This aggregator provides various strategies for combining similarity scores
    and detecting inconsistencies across methods.
    """
    
    def __init__(self, config: Optional[AggregationConfig] = None):
        self.config = config or AggregationConfig()
        self.logger = get_logger(__name__)
        
        # Method reliability weights (learned from performance)
        self.method_weights = {
            "classical_fast": 0.7,
            "classical_accurate": 0.8,
            "quantum_precise": 0.95,
            "quantum_approximate": 0.85,
            "hybrid_balanced": 0.9,
            "hybrid_batch": 0.8,
            "ensemble": 1.0
        }
        
        # Validation thresholds
        self.consistency_threshold = 0.8
        self.confidence_threshold = 0.7
    
    def compute_consensus(
        self,
        method_results: Dict[str, np.ndarray],
        strategy: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute consensus scores from multiple methods.
        
        Args:
            method_results: Dictionary mapping method names to score arrays
            strategy: Aggregation strategy to use
            
        Returns:
            Consensus score array
        """
        if not method_results:
            raise ValueError("No method results provided for consensus")
        
        # Use provided strategy or default
        strategy = strategy or self.config.strategy.value
        
        # Validate input dimensions
        self._validate_result_dimensions(method_results)
        
        # Normalize scores if configured
        if self.config.normalize_scores:
            method_results = self._normalize_scores(method_results)
        
        # Detect and handle outliers
        if self.config.outlier_detection:
            method_results = self._handle_outliers(method_results)
        
        # Apply aggregation strategy
        if strategy == AggregationStrategy.MEAN.value:
            consensus = self._aggregate_mean(method_results)
        elif strategy == AggregationStrategy.WEIGHTED_AVERAGE.value:
            consensus = self._aggregate_weighted_average(method_results)
        elif strategy == AggregationStrategy.MEDIAN.value:
            consensus = self._aggregate_median(method_results)
        elif strategy == AggregationStrategy.TRIMMED_MEAN.value:
            consensus = self._aggregate_trimmed_mean(method_results)
        elif strategy == AggregationStrategy.RANK_FUSION.value:
            consensus = self._aggregate_rank_fusion(method_results)
        else:
            self.logger.warning(f"Unknown strategy '{strategy}', using mean")
            consensus = self._aggregate_mean(method_results)
        
        # Ensure scores are in valid range
        consensus = np.clip(consensus, 0.0, 1.0)
        
        return consensus
    
    def validate_consistency(
        self,
        method_results: Dict[str, np.ndarray]
    ) -> Tuple[bool, float, List[str]]:
        """
        Validate consistency across method results.
        
        Args:
            method_results: Dictionary mapping method names to score arrays
            
        Returns:
            Tuple of (is_consistent, consistency_score, warnings)
        """
        if len(method_results) < 2:
            return True, 1.0, []
        
        warnings = []
        consistency_scores = []
        
        # Compute pairwise consistency
        methods = list(method_results.keys())
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method1, method2 = methods[i], methods[j]
                scores1 = method_results[method1]
                scores2 = method_results[method2]
                
                # Compute correlation
                if len(scores1) > 1:
                    correlation = np.corrcoef(scores1, scores2)[0, 1]
                    
                    if not np.isnan(correlation):
                        consistency_scores.append(correlation)
                        
                        if correlation < self.consistency_threshold:
                            warnings.append(
                                f"Low consistency between {method1} and {method2}: "
                                f"{correlation:.3f}"
                            )
        
        # Overall consistency
        if consistency_scores:
            avg_consistency = np.mean(consistency_scores)
            is_consistent = avg_consistency >= self.consistency_threshold
        else:
            avg_consistency = 0.0
            is_consistent = False
            warnings.append("Unable to compute consistency scores")
        
        return is_consistent, float(avg_consistency), warnings
    
    def process_results(
        self,
        scores: np.ndarray,
        method_used: str,
        requirements: Any
    ) -> Any:
        """Process and validate single method results."""
        # Validate score range
        if np.any(scores < 0) or np.any(scores > 1):
            self.logger.warning(f"Scores outside [0,1] range for {method_used}, clipping")
            scores = np.clip(scores, 0.0, 1.0)
        
        # Check for anomalies
        anomalies = self._detect_anomalies(scores)
        if anomalies:
            self.logger.warning(f"Detected anomalies in {method_used} results: {anomalies}")
        
        # Compute quality metrics
        quality_metrics = self._compute_quality_metrics(scores)
        
        return scores, quality_metrics
    
    def _validate_result_dimensions(self, method_results: Dict[str, np.ndarray]) -> None:
        """Validate that all result arrays have same dimensions."""
        if not method_results:
            return
        
        # Get reference shape
        ref_shape = None
        ref_method = None
        
        for method, scores in method_results.items():
            if ref_shape is None:
                ref_shape = scores.shape
                ref_method = method
            elif scores.shape != ref_shape:
                raise ValueError(
                    f"Shape mismatch: {method} has shape {scores.shape}, "
                    f"but {ref_method} has shape {ref_shape}"
                )
    
    def _normalize_scores(
        self,
        method_results: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Normalize scores to [0,1] range."""
        normalized = {}
        
        for method, scores in method_results.items():
            if len(scores) == 0:
                normalized[method] = scores
                continue
            
            # Min-max normalization
            min_score = np.min(scores)
            max_score = np.max(scores)
            
            if max_score > min_score:
                normalized[method] = (scores - min_score) / (max_score - min_score)
            else:
                # All scores are same
                normalized[method] = np.ones_like(scores) * 0.5
        
        return normalized
    
    def _handle_outliers(
        self,
        method_results: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Detect and handle outlier scores."""
        # Compute median across methods for each candidate
        all_scores = np.stack(list(method_results.values()))
        median_scores = np.median(all_scores, axis=0)
        std_scores = np.std(all_scores, axis=0)
        
        cleaned_results = {}
        
        for method, scores in method_results.items():
            # Detect outliers
            z_scores = np.abs((scores - median_scores) / (std_scores + 1e-8))
            outlier_mask = z_scores > self.config.outlier_threshold
            
            if np.any(outlier_mask):
                self.logger.debug(
                    f"Found {np.sum(outlier_mask)} outliers in {method} results"
                )
                
                # Replace outliers with median
                cleaned_scores = scores.copy()
                cleaned_scores[outlier_mask] = median_scores[outlier_mask]
                cleaned_results[method] = cleaned_scores
            else:
                cleaned_results[method] = scores
        
        return cleaned_results
    
    def _aggregate_mean(self, method_results: Dict[str, np.ndarray]) -> np.ndarray:
        """Simple mean aggregation."""
        return np.mean(list(method_results.values()), axis=0)
    
    def _aggregate_weighted_average(
        self,
        method_results: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Weighted average based on method reliability."""
        weighted_sum = np.zeros_like(next(iter(method_results.values())))
        total_weight = 0.0
        
        for method, scores in method_results.items():
            weight = self.method_weights.get(method, 0.5)
            weighted_sum += weight * scores
            total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return self._aggregate_mean(method_results)
    
    def _aggregate_median(self, method_results: Dict[str, np.ndarray]) -> np.ndarray:
        """Median aggregation (robust to outliers)."""
        return np.median(list(method_results.values()), axis=0)
    
    def _aggregate_trimmed_mean(
        self,
        method_results: Dict[str, np.ndarray],
        trim_percent: float = 0.1
    ) -> np.ndarray:
        """Trimmed mean aggregation."""
        all_scores = np.stack(list(method_results.values()))
        
        # Sort along method axis
        sorted_scores = np.sort(all_scores, axis=0)
        
        # Determine trim indices
        n_methods = len(method_results)
        trim_count = int(n_methods * trim_percent)
        
        if trim_count > 0 and n_methods > 2 * trim_count:
            # Trim top and bottom
            trimmed_scores = sorted_scores[trim_count:-trim_count]
            return np.mean(trimmed_scores, axis=0)
        else:
            # Not enough methods to trim
            return self._aggregate_mean(method_results)
    
    def _aggregate_rank_fusion(
        self,
        method_results: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Rank-based fusion aggregation."""
        n_candidates = len(next(iter(method_results.values())))
        rank_sum = np.zeros(n_candidates)
        
        for method, scores in method_results.items():
            # Convert scores to ranks (higher score = better rank = lower number)
            ranks = np.argsort(np.argsort(-scores)) + 1
            
            # Weight ranks by method reliability
            weight = self.method_weights.get(method, 0.5)
            rank_sum += weight * ranks
        
        # Convert back to scores (lower rank sum = higher score)
        # Normalize to [0, 1]
        min_rank = np.min(rank_sum)
        max_rank = np.max(rank_sum)
        
        if max_rank > min_rank:
            fusion_scores = 1.0 - (rank_sum - min_rank) / (max_rank - min_rank)
        else:
            fusion_scores = np.ones(n_candidates) * 0.5
        
        return fusion_scores
    
    def _detect_anomalies(self, scores: np.ndarray) -> List[str]:
        """Detect anomalies in score distribution."""
        anomalies = []
        
        # Check for NaN or infinite values
        if np.any(np.isnan(scores)):
            anomalies.append("NaN values detected")
        
        if np.any(np.isinf(scores)):
            anomalies.append("Infinite values detected")
        
        # Check for extreme clustering
        unique_scores = np.unique(scores)
        if len(unique_scores) == 1:
            anomalies.append("All scores are identical")
        elif len(unique_scores) < len(scores) * 0.1:
            anomalies.append("Very low score diversity")
        
        # Check for extreme skew
        if len(scores) > 10:
            skewness = self._compute_skewness(scores)
            if abs(skewness) > 2.0:
                anomalies.append(f"Extreme skewness: {skewness:.2f}")
        
        return anomalies
    
    def _compute_skewness(self, scores: np.ndarray) -> float:
        """Compute skewness of score distribution."""
        mean = np.mean(scores)
        std = np.std(scores)
        
        if std == 0:
            return 0.0
        
        skew = np.mean(((scores - mean) / std) ** 3)
        return float(skew)
    
    def _compute_quality_metrics(self, scores: np.ndarray) -> Dict[str, float]:
        """Compute quality metrics for scores."""
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "range": float(np.max(scores) - np.min(scores)),
            "cv": float(np.std(scores) / (np.mean(scores) + 1e-8))
        }
    
    def update_method_weights(
        self,
        method_performance: Dict[str, float]
    ) -> None:
        """Update method weights based on performance."""
        for method, performance in method_performance.items():
            if 0 <= performance <= 1:
                self.method_weights[method] = performance
            else:
                self.logger.warning(
                    f"Invalid performance score {performance} for {method}, skipping"
                )
    
    def get_aggregation_statistics(self) -> Dict[str, Any]:
        """Get statistics about aggregation performance."""
        return {
            "config": {
                "strategy": self.config.strategy.value,
                "consensus_algorithm": self.config.consensus_algorithm.value,
                "outlier_detection": self.config.outlier_detection,
                "normalize_scores": self.config.normalize_scores
            },
            "method_weights": self.method_weights,
            "thresholds": {
                "consistency": self.consistency_threshold,
                "confidence": self.confidence_threshold,
                "outlier": self.config.outlier_threshold
            }
        }


__all__ = [
    "AggregationStrategy",
    "ConsensusAlgorithm",
    "AggregationConfig",
    "ResultAggregator"
]
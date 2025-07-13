"""
Comparative Analysis Tools for QuantumRerank Benchmarking.

Implements statistical comparison between quantum and classical approaches,
enabling comprehensive evaluation of performance trade-offs.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of statistical comparison between two methods."""
    method1_name: str
    method2_name: str
    metric_name: str
    
    # Statistical Test Results
    statistically_significant: bool
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    
    # Descriptive Statistics
    method1_stats: Dict[str, float]
    method2_stats: Dict[str, float]
    
    # Performance Comparison
    winner: str  # 'method1', 'method2', or 'tie'
    improvement_percent: float
    
    # Test Details
    test_used: str
    sample_sizes: Tuple[int, int]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccuracyComparison:
    """Comparison of accuracy metrics between methods."""
    method_name: str
    ndcg_scores: Dict[str, float]  # @1, @5, @10
    mrr_score: float
    precision_scores: Dict[str, float]  # @1, @5, @10
    recall_scores: Dict[str, float]  # @1, @5, @10
    map_score: float  # Mean Average Precision


class ComparativeAnalyzer:
    """
    Statistical analysis and comparison framework for benchmarking results.
    
    Provides rigorous statistical testing and effect size calculations
    for comparing quantum vs classical approaches.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize comparative analyzer.
        
        Args:
            confidence_level: Statistical confidence level (default 95%)
        """
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        
        logger.info(f"Initialized ComparativeAnalyzer with {confidence_level*100}% confidence")
    
    def compare_latency_distributions(self, 
                                    method1_latencies: List[float],
                                    method2_latencies: List[float],
                                    method1_name: str = "Method1",
                                    method2_name: str = "Method2") -> ComparisonResult:
        """
        Compare latency distributions between two methods.
        
        Args:
            method1_latencies: Latency measurements for method 1 (ms)
            method2_latencies: Latency measurements for method 2 (ms)
            method1_name: Name of first method
            method2_name: Name of second method
            
        Returns:
            ComparisonResult with statistical analysis
        """
        if not method1_latencies or not method2_latencies:
            raise ValueError("Both methods must have latency measurements")
        
        # Convert to numpy arrays
        data1 = np.array(method1_latencies)
        data2 = np.array(method2_latencies)
        
        # Descriptive statistics
        method1_stats = self._calculate_descriptive_stats(data1)
        method2_stats = self._calculate_descriptive_stats(data2)
        
        # Choose appropriate statistical test
        if self._is_normally_distributed(data1) and self._is_normally_distributed(data2):
            # Use t-test for normally distributed data
            statistic, p_value = stats.ttest_ind(data1, data2)
            test_used = "Independent t-test"
            
            # Calculate Cohen's d for effect size
            pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                 (len(data2) - 1) * np.var(data2, ddof=1)) / 
                                (len(data1) + len(data2) - 2))
            effect_size = (np.mean(data1) - np.mean(data2)) / pooled_std
        else:
            # Use Mann-Whitney U test for non-normal data
            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            test_used = "Mann-Whitney U test"
            
            # Calculate rank-biserial correlation for effect size
            effect_size = 1 - (2 * statistic) / (len(data1) * len(data2))
        
        # Determine statistical significance
        significant = p_value < self.alpha
        
        # Calculate confidence interval for difference in means
        ci_lower, ci_upper = self._calculate_confidence_interval(data1, data2)
        
        # Determine winner and improvement
        mean1, mean2 = np.mean(data1), np.mean(data2)
        if mean1 < mean2:  # Lower latency is better
            winner = method1_name
            improvement_percent = ((mean2 - mean1) / mean2) * 100
        elif mean2 < mean1:
            winner = method2_name
            improvement_percent = ((mean1 - mean2) / mean1) * 100
        else:
            winner = "tie"
            improvement_percent = 0.0
        
        return ComparisonResult(
            method1_name=method1_name,
            method2_name=method2_name,
            metric_name="latency",
            statistically_significant=significant,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            method1_stats=method1_stats,
            method2_stats=method2_stats,
            winner=winner,
            improvement_percent=improvement_percent,
            test_used=test_used,
            sample_sizes=(len(data1), len(data2))
        )
    
    def compare_memory_usage(self,
                           method1_memory: List[float],
                           method2_memory: List[float],
                           method1_name: str = "Method1",
                           method2_name: str = "Method2") -> ComparisonResult:
        """
        Compare memory usage between two methods.
        
        Args:
            method1_memory: Memory usage measurements for method 1 (MB)
            method2_memory: Memory usage measurements for method 2 (MB)
            method1_name: Name of first method
            method2_name: Name of second method
            
        Returns:
            ComparisonResult with statistical analysis
        """
        # Use same logic as latency comparison (lower is better)
        result = self.compare_latency_distributions(
            method1_memory, method2_memory, method1_name, method2_name
        )
        result.metric_name = "memory_usage"
        return result
    
    def compare_throughput(self,
                         method1_throughput: List[float],
                         method2_throughput: List[float],
                         method1_name: str = "Method1",
                         method2_name: str = "Method2") -> ComparisonResult:
        """
        Compare throughput between two methods.
        
        Args:
            method1_throughput: Throughput measurements for method 1 (ops/sec)
            method2_throughput: Throughput measurements for method 2 (ops/sec)
            method1_name: Name of first method
            method2_name: Name of second method
            
        Returns:
            ComparisonResult with statistical analysis
        """
        if not method1_throughput or not method2_throughput:
            raise ValueError("Both methods must have throughput measurements")
        
        data1 = np.array(method1_throughput)
        data2 = np.array(method2_throughput)
        
        # Calculate stats (same as other metrics)
        method1_stats = self._calculate_descriptive_stats(data1)
        method2_stats = self._calculate_descriptive_stats(data2)
        
        # Statistical test
        if self._is_normally_distributed(data1) and self._is_normally_distributed(data2):
            statistic, p_value = stats.ttest_ind(data1, data2)
            test_used = "Independent t-test"
            pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                 (len(data2) - 1) * np.var(data2, ddof=1)) / 
                                (len(data1) + len(data2) - 2))
            effect_size = (np.mean(data1) - np.mean(data2)) / pooled_std
        else:
            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            test_used = "Mann-Whitney U test"
            effect_size = 1 - (2 * statistic) / (len(data1) * len(data2))
        
        significant = p_value < self.alpha
        ci_lower, ci_upper = self._calculate_confidence_interval(data1, data2)
        
        # For throughput, higher is better
        mean1, mean2 = np.mean(data1), np.mean(data2)
        if mean1 > mean2:
            winner = method1_name
            improvement_percent = ((mean1 - mean2) / mean2) * 100
        elif mean2 > mean1:
            winner = method2_name
            improvement_percent = ((mean2 - mean1) / mean1) * 100
        else:
            winner = "tie"
            improvement_percent = 0.0
        
        return ComparisonResult(
            method1_name=method1_name,
            method2_name=method2_name,
            metric_name="throughput",
            statistically_significant=significant,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            method1_stats=method1_stats,
            method2_stats=method2_stats,
            winner=winner,
            improvement_percent=improvement_percent,
            test_used=test_used,
            sample_sizes=(len(data1), len(data2))
        )
    
    def compare_accuracy_metrics(self,
                               method1_results: List[Dict[str, float]],
                               method2_results: List[Dict[str, float]],
                               method1_name: str = "Method1",
                               method2_name: str = "Method2") -> Dict[str, ComparisonResult]:
        """
        Compare accuracy metrics (NDCG, MRR, etc.) between two methods.
        
        Args:
            method1_results: List of accuracy results for method 1
            method2_results: List of accuracy results for method 2
            method1_name: Name of first method
            method2_name: Name of second method
            
        Returns:
            Dictionary of ComparisonResult objects by metric name
        """
        if not method1_results or not method2_results:
            raise ValueError("Both methods must have accuracy results")
        
        # Extract metric arrays
        metric_names = set()
        for result in method1_results + method2_results:
            metric_names.update(result.keys())
        
        comparisons = {}
        
        for metric_name in metric_names:
            # Extract values for this metric
            method1_values = [r.get(metric_name, 0.0) for r in method1_results]
            method2_values = [r.get(metric_name, 0.0) for r in method2_results]
            
            # Skip if no data for this metric
            if not any(method1_values) or not any(method2_values):
                continue
            
            # For accuracy metrics, higher is better (use throughput logic)
            comparison = self.compare_throughput(
                method1_values, method2_values, method1_name, method2_name
            )
            comparison.metric_name = metric_name
            
            comparisons[metric_name] = comparison
        
        return comparisons
    
    def analyze_quantum_vs_classical(self,
                                   quantum_results: Dict[str, List[float]],
                                   classical_results: Dict[str, List[float]]) -> Dict[str, ComparisonResult]:
        """
        Comprehensive analysis of quantum vs classical performance.
        
        Args:
            quantum_results: Dictionary of quantum method results by metric
            classical_results: Dictionary of classical method results by metric
            
        Returns:
            Dictionary of ComparisonResult objects by metric
        """
        logger.info("Performing quantum vs classical comparative analysis")
        
        comparisons = {}
        
        # Compare metrics that exist in both results
        common_metrics = set(quantum_results.keys()) & set(classical_results.keys())
        
        for metric_name in common_metrics:
            quantum_data = quantum_results[metric_name]
            classical_data = classical_results[metric_name]
            
            if not quantum_data or not classical_data:
                logger.warning(f"Insufficient data for metric {metric_name}")
                continue
            
            # Choose comparison method based on metric type
            if "latency" in metric_name.lower() or "memory" in metric_name.lower():
                # Lower is better
                comparison = self.compare_latency_distributions(
                    quantum_data, classical_data, "Quantum", "Classical"
                )
            else:
                # Higher is better (accuracy, throughput)
                comparison = self.compare_throughput(
                    quantum_data, classical_data, "Quantum", "Classical"
                )
            
            comparison.metric_name = metric_name
            comparisons[metric_name] = comparison
        
        return comparisons
    
    def generate_performance_summary(self, 
                                   comparisons: Dict[str, ComparisonResult]) -> Dict[str, Any]:
        """
        Generate overall performance summary from comparisons.
        
        Args:
            comparisons: Dictionary of comparison results
            
        Returns:
            Summary dictionary with overall findings
        """
        if not comparisons:
            return {"error": "No comparisons provided"}
        
        # Count wins and significant differences
        quantum_wins = 0
        classical_wins = 0
        ties = 0
        significant_differences = 0
        total_comparisons = len(comparisons)
        
        effect_sizes = []
        improvements = []
        
        for metric_name, comparison in comparisons.items():
            if comparison.winner == "Quantum":
                quantum_wins += 1
            elif comparison.winner == "Classical":
                classical_wins += 1
            else:
                ties += 1
            
            if comparison.statistically_significant:
                significant_differences += 1
            
            effect_sizes.append(abs(comparison.effect_size))
            improvements.append(comparison.improvement_percent)
        
        # Calculate average effect size and improvement
        avg_effect_size = np.mean(effect_sizes) if effect_sizes else 0.0
        avg_improvement = np.mean(improvements) if improvements else 0.0
        
        # Determine overall winner
        if quantum_wins > classical_wins:
            overall_winner = "Quantum"
        elif classical_wins > quantum_wins:
            overall_winner = "Classical"
        else:
            overall_winner = "Tie"
        
        summary = {
            "overall_winner": overall_winner,
            "quantum_wins": quantum_wins,
            "classical_wins": classical_wins,
            "ties": ties,
            "total_comparisons": total_comparisons,
            "significant_differences": significant_differences,
            "significance_rate": significant_differences / total_comparisons,
            "average_effect_size": avg_effect_size,
            "average_improvement_percent": avg_improvement,
            "confidence_level": self.confidence_level,
            "detailed_results": {
                metric: {
                    "winner": comp.winner,
                    "significant": comp.statistically_significant,
                    "p_value": comp.p_value,
                    "effect_size": comp.effect_size,
                    "improvement": comp.improvement_percent
                }
                for metric, comp in comparisons.items()
            }
        }
        
        return summary
    
    def _calculate_descriptive_stats(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate descriptive statistics for data array."""
        return {
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "std": float(np.std(data, ddof=1)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "q25": float(np.percentile(data, 25)),
            "q75": float(np.percentile(data, 75)),
            "count": len(data)
        }
    
    def _is_normally_distributed(self, data: np.ndarray, alpha: float = 0.05) -> bool:
        """Test if data is normally distributed using Shapiro-Wilk test."""
        if len(data) < 3:
            return False  # Too few samples for test
        
        if len(data) > 5000:
            # Use Kolmogorov-Smirnov test for large samples
            statistic, p_value = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
        else:
            # Use Shapiro-Wilk test for smaller samples
            statistic, p_value = stats.shapiro(data)
        
        return p_value > alpha
    
    def _calculate_confidence_interval(self, data1: np.ndarray, data2: np.ndarray) -> Tuple[float, float]:
        """Calculate confidence interval for difference in means."""
        mean1, mean2 = np.mean(data1), np.mean(data2)
        var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        n1, n2 = len(data1), len(data2)
        
        # Standard error of difference
        se_diff = np.sqrt(var1/n1 + var2/n2)
        
        # Degrees of freedom (Welch's formula)
        df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
        
        # Critical t-value
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        
        # Confidence interval
        diff = mean1 - mean2
        margin = t_critical * se_diff
        
        return (diff - margin, diff + margin)
    
    def interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_effect = abs(effect_size)
        
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"
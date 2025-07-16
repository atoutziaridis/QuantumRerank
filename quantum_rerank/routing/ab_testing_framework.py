"""
A/B Testing Framework for Routing Optimization.

This module implements comprehensive A/B testing for routing decisions,
enabling data-driven optimization of quantum-classical routing performance.

Based on:
- QMMR-02 task specification
- Industry-standard A/B testing methodology
- Statistical significance testing
- Performance comparison framework
"""

import time
import logging
import numpy as np
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json
import random
from datetime import datetime, timedelta

from .complexity_metrics import ComplexityAssessmentResult
from .routing_decision_engine import RoutingDecision, RoutingMethod
from ..config.routing_config import ABTestConfig, ABTestStrategy

logger = logging.getLogger(__name__)

# Optional imports for advanced statistical analysis
try:
    import scipy.stats as stats
    from scipy.stats import chi2_contingency, ttest_ind
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available, statistical tests will be limited")


class ABTestStatus(Enum):
    """Status of A/B test."""
    PLANNING = "planning"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ABTestMetrics:
    """Metrics collected during A/B testing."""
    
    # Performance metrics
    accuracy_scores: List[float] = field(default_factory=list)
    latency_measurements: List[float] = field(default_factory=list)
    memory_usage_measurements: List[float] = field(default_factory=list)
    
    # User satisfaction metrics
    user_satisfaction_scores: List[float] = field(default_factory=list)
    
    # Routing-specific metrics
    routing_decisions: List[RoutingMethod] = field(default_factory=list)
    complexity_scores: List[float] = field(default_factory=list)
    quantum_advantage_scores: List[float] = field(default_factory=list)
    
    # Error tracking
    error_count: int = 0
    timeout_count: int = 0
    
    # Timing
    sample_timestamps: List[float] = field(default_factory=list)
    
    def add_sample(self, accuracy: float, latency: float, memory: float,
                   routing_method: RoutingMethod, complexity: float,
                   quantum_advantage: float = 0.0, user_satisfaction: float = 0.0,
                   error: bool = False, timeout: bool = False):
        """Add a sample to the metrics."""
        self.accuracy_scores.append(accuracy)
        self.latency_measurements.append(latency)
        self.memory_usage_measurements.append(memory)
        self.routing_decisions.append(routing_method)
        self.complexity_scores.append(complexity)
        self.quantum_advantage_scores.append(quantum_advantage)
        self.user_satisfaction_scores.append(user_satisfaction)
        self.sample_timestamps.append(time.time())
        
        if error:
            self.error_count += 1
        if timeout:
            self.timeout_count += 1
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the metrics."""
        if not self.accuracy_scores:
            return {}
        
        return {
            'sample_count': len(self.accuracy_scores),
            'accuracy': {
                'mean': np.mean(self.accuracy_scores),
                'std': np.std(self.accuracy_scores),
                'min': np.min(self.accuracy_scores),
                'max': np.max(self.accuracy_scores)
            },
            'latency': {
                'mean': np.mean(self.latency_measurements),
                'std': np.std(self.latency_measurements),
                'p95': np.percentile(self.latency_measurements, 95),
                'p99': np.percentile(self.latency_measurements, 99)
            },
            'memory': {
                'mean': np.mean(self.memory_usage_measurements),
                'std': np.std(self.memory_usage_measurements),
                'max': np.max(self.memory_usage_measurements)
            },
            'routing_distribution': {
                'classical': sum(1 for r in self.routing_decisions if r == RoutingMethod.CLASSICAL) / len(self.routing_decisions),
                'quantum': sum(1 for r in self.routing_decisions if r == RoutingMethod.QUANTUM) / len(self.routing_decisions),
                'hybrid': sum(1 for r in self.routing_decisions if r == RoutingMethod.HYBRID) / len(self.routing_decisions)
            },
            'error_rate': self.error_count / len(self.accuracy_scores),
            'timeout_rate': self.timeout_count / len(self.accuracy_scores),
            'avg_complexity': np.mean(self.complexity_scores),
            'avg_quantum_advantage': np.mean(self.quantum_advantage_scores)
        }


@dataclass
class ABTestResult:
    """Result of an individual A/B test assignment."""
    
    test_id: str
    user_id: str
    group: str  # 'control' or 'treatment'
    routing_method: RoutingMethod
    complexity_score: float
    timestamp: float
    
    # Outcome metrics
    accuracy: float = 0.0
    latency_ms: float = 0.0
    memory_mb: float = 0.0
    user_satisfaction: float = 0.0
    
    # Success indicators
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ABTestAnalysis:
    """Analysis results from A/B test."""
    
    test_name: str
    test_duration_hours: float
    
    # Group performance
    control_metrics: Dict[str, Any]
    treatment_metrics: Dict[str, Any]
    
    # Statistical tests
    statistical_significance: Dict[str, Any]
    
    # Recommendations
    winner: str  # 'control', 'treatment', or 'inconclusive'
    confidence_level: float
    effect_size: float
    
    # Detailed analysis
    recommendations: List[str]
    insights: List[str]
    
    # Meta information
    analysis_timestamp: float = field(default_factory=time.time)


class ABTestMetricsCollector:
    """Collects and aggregates metrics during A/B testing."""
    
    def __init__(self):
        self.test_results: Dict[str, List[ABTestResult]] = defaultdict(list)
        self.group_metrics: Dict[str, Dict[str, ABTestMetrics]] = defaultdict(lambda: defaultdict(ABTestMetrics))
        
    def record_result(self, result: ABTestResult):
        """Record an A/B test result."""
        self.test_results[result.test_id].append(result)
        
        # Update group metrics
        group_key = f"{result.test_id}_{result.group}"
        self.group_metrics[result.test_id][result.group].add_sample(
            accuracy=result.accuracy,
            latency=result.latency_ms,
            memory=result.memory_mb,
            routing_method=result.routing_method,
            complexity=result.complexity_score,
            user_satisfaction=result.user_satisfaction,
            error=not result.success
        )
    
    def get_test_metrics(self, test_id: str) -> Dict[str, ABTestMetrics]:
        """Get metrics for a specific test."""
        return dict(self.group_metrics[test_id])
    
    def get_all_results(self, test_id: str) -> List[ABTestResult]:
        """Get all results for a test."""
        return self.test_results[test_id]


class StatisticalAnalyzer:
    """Performs statistical analysis on A/B test results."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def analyze_test(self, control_metrics: ABTestMetrics, 
                    treatment_metrics: ABTestMetrics,
                    primary_metric: str = "accuracy") -> Dict[str, Any]:
        """Perform statistical analysis on A/B test results."""
        if not SCIPY_AVAILABLE:
            return self._basic_analysis(control_metrics, treatment_metrics, primary_metric)
        
        results = {}
        
        # Get primary metric data
        control_data = self._get_metric_data(control_metrics, primary_metric)
        treatment_data = self._get_metric_data(treatment_metrics, primary_metric)
        
        if len(control_data) < 2 or len(treatment_data) < 2:
            return {'error': 'Insufficient data for statistical analysis'}
        
        # T-test for primary metric
        t_stat, p_value = ttest_ind(control_data, treatment_data)
        
        results['primary_metric'] = {
            'metric': primary_metric,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'control_mean': np.mean(control_data),
            'treatment_mean': np.mean(treatment_data),
            'effect_size': self._compute_cohens_d(control_data, treatment_data)
        }
        
        # Secondary metrics analysis
        secondary_metrics = ['latency_measurements', 'memory_usage_measurements']
        results['secondary_metrics'] = {}
        
        for metric in secondary_metrics:
            control_secondary = self._get_metric_data(control_metrics, metric)
            treatment_secondary = self._get_metric_data(treatment_metrics, metric)
            
            if len(control_secondary) >= 2 and len(treatment_secondary) >= 2:
                t_stat_sec, p_value_sec = ttest_ind(control_secondary, treatment_secondary)
                
                results['secondary_metrics'][metric] = {
                    't_statistic': t_stat_sec,
                    'p_value': p_value_sec,
                    'significant': p_value_sec < self.significance_level,
                    'control_mean': np.mean(control_secondary),
                    'treatment_mean': np.mean(treatment_secondary),
                    'effect_size': self._compute_cohens_d(control_secondary, treatment_secondary)
                }
        
        # Routing distribution analysis
        control_routing = control_metrics.routing_decisions
        treatment_routing = treatment_metrics.routing_decisions
        
        if control_routing and treatment_routing:
            routing_analysis = self._analyze_routing_distribution(control_routing, treatment_routing)
            results['routing_analysis'] = routing_analysis
        
        return results
    
    def _basic_analysis(self, control_metrics: ABTestMetrics, 
                       treatment_metrics: ABTestMetrics,
                       primary_metric: str) -> Dict[str, Any]:
        """Basic analysis without scipy."""
        control_data = self._get_metric_data(control_metrics, primary_metric)
        treatment_data = self._get_metric_data(treatment_metrics, primary_metric)
        
        if not control_data or not treatment_data:
            return {'error': 'No data available for analysis'}
        
        control_mean = np.mean(control_data)
        treatment_mean = np.mean(treatment_data)
        
        return {
            'primary_metric': {
                'metric': primary_metric,
                'control_mean': control_mean,
                'treatment_mean': treatment_mean,
                'difference': treatment_mean - control_mean,
                'percent_change': ((treatment_mean - control_mean) / control_mean) * 100 if control_mean != 0 else 0
            },
            'note': 'Limited analysis - scipy not available for significance testing'
        }
    
    def _get_metric_data(self, metrics: ABTestMetrics, metric_name: str) -> List[float]:
        """Extract metric data from ABTestMetrics."""
        if metric_name == 'accuracy':
            return metrics.accuracy_scores
        elif metric_name == 'latency_measurements':
            return metrics.latency_measurements
        elif metric_name == 'memory_usage_measurements':
            return metrics.memory_usage_measurements
        elif metric_name == 'user_satisfaction_scores':
            return metrics.user_satisfaction_scores
        else:
            return []
    
    def _compute_cohens_d(self, control_data: List[float], treatment_data: List[float]) -> float:
        """Compute Cohen's d effect size."""
        control_mean = np.mean(control_data)
        treatment_mean = np.mean(treatment_data)
        
        control_std = np.std(control_data, ddof=1)
        treatment_std = np.std(treatment_data, ddof=1)
        
        pooled_std = np.sqrt(((len(control_data) - 1) * control_std**2 + 
                             (len(treatment_data) - 1) * treatment_std**2) / 
                            (len(control_data) + len(treatment_data) - 2))
        
        return (treatment_mean - control_mean) / pooled_std
    
    def _analyze_routing_distribution(self, control_routing: List[RoutingMethod], 
                                    treatment_routing: List[RoutingMethod]) -> Dict[str, Any]:
        """Analyze routing distribution differences."""
        # Count routing methods
        control_counts = defaultdict(int)
        treatment_counts = defaultdict(int)
        
        for method in control_routing:
            control_counts[method] += 1
        
        for method in treatment_routing:
            treatment_counts[method] += 1
        
        # Calculate percentages
        control_total = len(control_routing)
        treatment_total = len(treatment_routing)
        
        control_percentages = {method: count / control_total * 100 
                              for method, count in control_counts.items()}
        treatment_percentages = {method: count / treatment_total * 100 
                               for method, count in treatment_counts.items()}
        
        return {
            'control_distribution': control_percentages,
            'treatment_distribution': treatment_percentages,
            'control_sample_size': control_total,
            'treatment_sample_size': treatment_total
        }


class ABTest:
    """Individual A/B test implementation."""
    
    def __init__(self, config: ABTestConfig):
        self.config = config
        self.test_id = str(uuid.uuid4())
        self.status = ABTestStatus.PLANNING
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # Sample tracking
        self.control_samples = 0
        self.treatment_samples = 0
        
        # Metrics collection
        self.metrics_collector = ABTestMetricsCollector()
        
        # Statistical analyzer
        self.analyzer = StatisticalAnalyzer(config.significance_level)
        
        logger.info(f"ABTest {self.test_id} created: {config.test_name}")
    
    def start(self) -> bool:
        """Start the A/B test."""
        if self.status != ABTestStatus.PLANNING:
            logger.error(f"Cannot start test {self.test_id} in status {self.status}")
            return False
        
        self.status = ABTestStatus.ACTIVE
        self.start_time = time.time()
        
        logger.info(f"ABTest {self.test_id} started")
        return True
    
    def should_apply(self, complexity_result: ComplexityAssessmentResult) -> bool:
        """Check if test should be applied to this query."""
        if self.status != ABTestStatus.ACTIVE:
            return False
        
        # Check if test has expired
        if self.start_time and time.time() - self.start_time > self.config.max_duration_hours * 3600:
            self.status = ABTestStatus.COMPLETED
            self.end_time = time.time()
            return False
        
        # Check sample size limits
        total_samples = self.control_samples + self.treatment_samples
        if total_samples >= self.config.sample_size:
            return False
        
        # Apply strategy-specific logic
        if self.config.strategy == ABTestStrategy.COMPLEXITY_BASED:
            return self._complexity_based_should_apply(complexity_result)
        elif self.config.strategy == ABTestStrategy.PERFORMANCE_BASED:
            return self._performance_based_should_apply(complexity_result)
        else:  # RANDOM
            return random.random() < 0.5  # 50% chance
    
    def get_routing_decision(self, complexity_result: ComplexityAssessmentResult) -> Optional[RoutingDecision]:
        """Get routing decision for A/B test."""
        if self.status != ABTestStatus.ACTIVE:
            return None
        
        # Determine group assignment
        group = self._assign_group()
        
        # Get routing method based on group
        if group == 'control':
            method = self.config.control_method
            self.control_samples += 1
        else:
            method = self.config.treatment_method
            self.treatment_samples += 1
        
        # Create routing decision
        decision = RoutingDecision(
            method=method,
            confidence=0.8,  # A/B test confidence
            complexity_score=complexity_result.overall_complexity.overall_complexity,
            complexity_factors=complexity_result.complexity_breakdown,
            estimated_latency_ms=100.0,  # Placeholder
            estimated_memory_mb=512.0,  # Placeholder
            reasoning=f"A/B test {self.config.test_name} - {group} group",
            decision_time_ms=1.0
        )
        
        return decision
    
    def record_decision(self, decision: RoutingDecision, user_id: str = "anonymous"):
        """Record A/B test decision."""
        # Determine group
        group = 'control' if decision.method == self.config.control_method else 'treatment'
        
        result = ABTestResult(
            test_id=self.test_id,
            user_id=user_id,
            group=group,
            routing_method=decision.method,
            complexity_score=decision.complexity_score,
            timestamp=time.time()
        )
        
        self.metrics_collector.record_result(result)
    
    def record_outcome(self, user_id: str, accuracy: float, latency_ms: float, 
                      memory_mb: float, user_satisfaction: float = 0.0):
        """Record outcome metrics for a user."""
        # Find the most recent result for this user
        recent_results = [r for r in self.metrics_collector.test_results[self.test_id] 
                         if r.user_id == user_id]
        
        if not recent_results:
            logger.warning(f"No A/B test result found for user {user_id}")
            return
        
        # Update the most recent result
        recent_result = recent_results[-1]
        recent_result.accuracy = accuracy
        recent_result.latency_ms = latency_ms
        recent_result.memory_mb = memory_mb
        recent_result.user_satisfaction = user_satisfaction
        
        # Update metrics
        group_metrics = self.metrics_collector.group_metrics[self.test_id][recent_result.group]
        group_metrics.add_sample(
            accuracy=accuracy,
            latency=latency_ms,
            memory=memory_mb,
            routing_method=recent_result.routing_method,
            complexity=recent_result.complexity_score,
            user_satisfaction=user_satisfaction
        )
    
    def _assign_group(self) -> str:
        """Assign user to control or treatment group."""
        # Simple randomization based on allocation
        if random.random() < self.config.control_allocation:
            return 'control'
        else:
            return 'treatment'
    
    def _complexity_based_should_apply(self, complexity_result: ComplexityAssessmentResult) -> bool:
        """Apply complexity-based filtering."""
        complexity_score = complexity_result.overall_complexity.overall_complexity
        
        # Apply to queries within complexity bands
        for min_complexity, max_complexity in self.config.complexity_bands:
            if min_complexity <= complexity_score <= max_complexity:
                return True
        
        return False
    
    def _performance_based_should_apply(self, complexity_result: ComplexityAssessmentResult) -> bool:
        """Apply performance-based filtering."""
        # Apply to queries likely to benefit from performance optimization
        return (complexity_result.overall_complexity.overall_complexity > 0.5 and
                complexity_result.query_complexity.modality_count > 1)
    
    def should_stop_early(self) -> Tuple[bool, str]:
        """Check if test should be stopped early."""
        if not self.config.enable_early_stopping:
            return False, "early_stopping_disabled"
        
        # Check minimum sample size
        total_samples = self.control_samples + self.treatment_samples
        if total_samples < self.config.min_sample_size:
            return False, "insufficient_samples"
        
        # Perform statistical analysis
        control_metrics = self.metrics_collector.group_metrics[self.test_id]['control']
        treatment_metrics = self.metrics_collector.group_metrics[self.test_id]['treatment']
        
        if not control_metrics.accuracy_scores or not treatment_metrics.accuracy_scores:
            return False, "insufficient_data"
        
        analysis = self.analyzer.analyze_test(control_metrics, treatment_metrics)
        
        if 'primary_metric' in analysis:
            primary_result = analysis['primary_metric']
            if (primary_result.get('significant', False) and 
                abs(primary_result.get('effect_size', 0)) > 0.5):
                return True, "significant_result_detected"
        
        return False, "no_early_stopping_condition"
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current test status."""
        return {
            'test_id': self.test_id,
            'test_name': self.config.test_name,
            'status': self.status.value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'control_samples': self.control_samples,
            'treatment_samples': self.treatment_samples,
            'target_sample_size': self.config.sample_size,
            'progress': (self.control_samples + self.treatment_samples) / self.config.sample_size
        }


class ABTestingFramework:
    """
    A/B testing framework for routing decisions and performance comparison.
    
    Manages multiple concurrent A/B tests with statistical analysis and optimization.
    """
    
    def __init__(self):
        self.active_tests: Dict[str, ABTest] = {}
        self.completed_tests: Dict[str, ABTest] = {}
        
        # Global metrics
        self.test_stats = {
            'total_tests': 0,
            'active_tests': 0,
            'completed_tests': 0
        }
        
        logger.info("ABTestingFramework initialized")
    
    def create_routing_test(self, config: ABTestConfig) -> ABTest:
        """Create a new A/B test for routing decisions."""
        test = ABTest(config)
        
        self.test_stats['total_tests'] += 1
        self.test_stats['active_tests'] += 1
        
        return test
    
    def start_test(self, test: ABTest) -> bool:
        """Start an A/B test."""
        if test.start():
            self.active_tests[test.test_id] = test
            return True
        return False
    
    def is_active(self) -> bool:
        """Check if any A/B tests are active."""
        return len(self.active_tests) > 0
    
    def get_routing_decision(self, complexity_result: ComplexityAssessmentResult) -> Optional[RoutingDecision]:
        """Get routing decision from active A/B tests."""
        for test in self.active_tests.values():
            if test.should_apply(complexity_result):
                return test.get_routing_decision(complexity_result)
        
        return None
    
    def record_test_outcome(self, test_id: str, user_id: str, accuracy: float, 
                           latency_ms: float, memory_mb: float, user_satisfaction: float = 0.0):
        """Record outcome for a specific test."""
        if test_id in self.active_tests:
            self.active_tests[test_id].record_outcome(user_id, accuracy, latency_ms, memory_mb, user_satisfaction)
        elif test_id in self.completed_tests:
            self.completed_tests[test_id].record_outcome(user_id, accuracy, latency_ms, memory_mb, user_satisfaction)
    
    def check_early_stopping(self):
        """Check all active tests for early stopping conditions."""
        tests_to_stop = []
        
        for test_id, test in self.active_tests.items():
            should_stop, reason = test.should_stop_early()
            if should_stop:
                tests_to_stop.append((test_id, reason))
        
        # Stop tests that meet early stopping criteria
        for test_id, reason in tests_to_stop:
            self.stop_test(test_id, reason)
    
    def stop_test(self, test_id: str, reason: str = "manual_stop"):
        """Stop an active A/B test."""
        if test_id in self.active_tests:
            test = self.active_tests[test_id]
            test.status = ABTestStatus.COMPLETED
            test.end_time = time.time()
            
            # Move to completed tests
            self.completed_tests[test_id] = test
            del self.active_tests[test_id]
            
            self.test_stats['active_tests'] -= 1
            self.test_stats['completed_tests'] += 1
            
            logger.info(f"A/B test {test_id} stopped: {reason}")
    
    def analyze_test_results(self, test_id: str) -> ABTestAnalysis:
        """Analyze results from completed A/B test."""
        if test_id in self.completed_tests:
            test = self.completed_tests[test_id]
        elif test_id in self.active_tests:
            test = self.active_tests[test_id]
        else:
            raise ValueError(f"Test {test_id} not found")
        
        # Get metrics
        control_metrics = test.metrics_collector.group_metrics[test_id]['control']
        treatment_metrics = test.metrics_collector.group_metrics[test_id]['treatment']
        
        # Perform statistical analysis
        statistical_results = test.analyzer.analyze_test(control_metrics, treatment_metrics, test.config.primary_metric)
        
        # Determine winner
        winner = "inconclusive"
        effect_size = 0.0
        confidence_level = 0.0
        
        if 'primary_metric' in statistical_results:
            primary_result = statistical_results['primary_metric']
            if primary_result.get('significant', False):
                if primary_result['treatment_mean'] > primary_result['control_mean']:
                    winner = "treatment"
                else:
                    winner = "control"
                effect_size = abs(primary_result.get('effect_size', 0))
                confidence_level = 1.0 - primary_result.get('p_value', 1.0)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(statistical_results, winner, effect_size)
        insights = self._generate_insights(control_metrics, treatment_metrics, statistical_results)
        
        # Calculate test duration
        test_duration = 0.0
        if test.start_time:
            end_time = test.end_time or time.time()
            test_duration = (end_time - test.start_time) / 3600  # Convert to hours
        
        return ABTestAnalysis(
            test_name=test.config.test_name,
            test_duration_hours=test_duration,
            control_metrics=control_metrics.get_summary_stats(),
            treatment_metrics=treatment_metrics.get_summary_stats(),
            statistical_significance=statistical_results,
            winner=winner,
            confidence_level=confidence_level,
            effect_size=effect_size,
            recommendations=recommendations,
            insights=insights
        )
    
    def _generate_recommendations(self, statistical_results: Dict[str, Any], 
                                winner: str, effect_size: float) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if winner == "treatment":
            recommendations.append("Implement treatment method - shows significant improvement")
            if effect_size > 0.5:
                recommendations.append("Large effect size detected - high impact change")
        elif winner == "control":
            recommendations.append("Keep current method - treatment shows no improvement")
        else:
            recommendations.append("Results inconclusive - consider longer test or larger sample")
        
        # Performance recommendations
        if 'secondary_metrics' in statistical_results:
            latency_results = statistical_results['secondary_metrics'].get('latency_measurements', {})
            if latency_results.get('significant', False):
                if latency_results['treatment_mean'] < latency_results['control_mean']:
                    recommendations.append("Treatment method shows better latency performance")
                else:
                    recommendations.append("Control method has better latency performance")
        
        return recommendations
    
    def _generate_insights(self, control_metrics: ABTestMetrics, 
                          treatment_metrics: ABTestMetrics,
                          statistical_results: Dict[str, Any]) -> List[str]:
        """Generate insights from test results."""
        insights = []
        
        # Routing distribution insights
        if 'routing_analysis' in statistical_results:
            routing_analysis = statistical_results['routing_analysis']
            control_dist = routing_analysis.get('control_distribution', {})
            treatment_dist = routing_analysis.get('treatment_distribution', {})
            
            # Compare quantum usage
            control_quantum = control_dist.get(RoutingMethod.QUANTUM, 0)
            treatment_quantum = treatment_dist.get(RoutingMethod.QUANTUM, 0)
            
            if abs(control_quantum - treatment_quantum) > 10:
                insights.append(f"Quantum routing usage differs significantly: {control_quantum:.1f}% vs {treatment_quantum:.1f}%")
        
        # Complexity insights
        if control_metrics.complexity_scores and treatment_metrics.complexity_scores:
            control_avg_complexity = np.mean(control_metrics.complexity_scores)
            treatment_avg_complexity = np.mean(treatment_metrics.complexity_scores)
            
            if abs(control_avg_complexity - treatment_avg_complexity) > 0.1:
                insights.append(f"Groups showed different complexity patterns: {control_avg_complexity:.3f} vs {treatment_avg_complexity:.3f}")
        
        # Error rate insights
        control_error_rate = control_metrics.error_count / len(control_metrics.accuracy_scores) if control_metrics.accuracy_scores else 0
        treatment_error_rate = treatment_metrics.error_count / len(treatment_metrics.accuracy_scores) if treatment_metrics.accuracy_scores else 0
        
        if abs(control_error_rate - treatment_error_rate) > 0.05:
            insights.append(f"Error rates differ: {control_error_rate:.3f} vs {treatment_error_rate:.3f}")
        
        return insights
    
    def get_all_test_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all tests."""
        all_statuses = {}
        
        for test_id, test in self.active_tests.items():
            all_statuses[test_id] = test.get_current_status()
        
        for test_id, test in self.completed_tests.items():
            all_statuses[test_id] = test.get_current_status()
        
        return all_statuses
    
    def get_framework_stats(self) -> Dict[str, Any]:
        """Get overall framework statistics."""
        return {
            **self.test_stats,
            'active_test_ids': list(self.active_tests.keys()),
            'completed_test_ids': list(self.completed_tests.keys())
        }
    
    def cleanup_old_tests(self, days_old: int = 30):
        """Clean up old completed tests."""
        cutoff_time = time.time() - (days_old * 24 * 3600)
        
        tests_to_remove = []
        for test_id, test in self.completed_tests.items():
            if test.end_time and test.end_time < cutoff_time:
                tests_to_remove.append(test_id)
        
        for test_id in tests_to_remove:
            del self.completed_tests[test_id]
        
        logger.info(f"Cleaned up {len(tests_to_remove)} old A/B tests")
    
    def export_test_results(self, test_id: str) -> Dict[str, Any]:
        """Export test results for external analysis."""
        if test_id in self.completed_tests:
            test = self.completed_tests[test_id]
        elif test_id in self.active_tests:
            test = self.active_tests[test_id]
        else:
            raise ValueError(f"Test {test_id} not found")
        
        # Get all raw data
        all_results = test.metrics_collector.get_all_results(test_id)
        
        export_data = {
            'test_config': {
                'test_name': test.config.test_name,
                'test_description': test.config.test_description,
                'strategy': test.config.strategy.value,
                'control_method': test.config.control_method.value,
                'treatment_method': test.config.treatment_method.value,
                'sample_size': test.config.sample_size,
                'significance_level': test.config.significance_level
            },
            'test_status': test.get_current_status(),
            'raw_results': [
                {
                    'user_id': r.user_id,
                    'group': r.group,
                    'routing_method': r.routing_method.value,
                    'complexity_score': r.complexity_score,
                    'accuracy': r.accuracy,
                    'latency_ms': r.latency_ms,
                    'memory_mb': r.memory_mb,
                    'user_satisfaction': r.user_satisfaction,
                    'timestamp': r.timestamp,
                    'success': r.success
                }
                for r in all_results
            ],
            'export_timestamp': time.time()
        }
        
        return export_data
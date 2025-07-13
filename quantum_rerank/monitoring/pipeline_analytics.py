"""
Pipeline analytics and performance trend analysis for QuantumRerank.

This module provides end-to-end pipeline performance analysis,
trend detection, and comprehensive reporting capabilities.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import numpy as np

from .metrics_collector import MetricsCollector
from ..utils import get_logger


class PipelineStage(Enum):
    """Pipeline stages for performance analysis."""
    DOCUMENT_RETRIEVAL = "document_retrieval"
    EMBEDDING_COMPUTATION = "embedding_computation"
    QUANTUM_PREPARATION = "quantum_preparation"
    SIMILARITY_COMPUTATION = "similarity_computation"
    RANKING_RECOMPUTATION = "ranking_recomputation"
    RESULT_AGGREGATION = "result_aggregation"


class TrendDirection(Enum):
    """Performance trend directions."""
    IMPROVING = "improving"
    DEGRADING = "degrading"
    STABLE = "stable"
    VOLATILE = "volatile"


@dataclass
class PerformanceTrend:
    """Performance trend analysis result."""
    metric_name: str
    direction: TrendDirection
    change_rate: float  # Percentage change
    confidence: float  # Confidence in trend (0-1)
    period_days: int
    significance: str  # "low", "medium", "high"
    description: str


@dataclass
class PipelineStageMetrics:
    """Comprehensive metrics for a pipeline stage."""
    stage: PipelineStage
    avg_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    throughput_per_second: float
    success_rate: float
    error_count: int
    cache_hit_rate: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_utilization: Optional[float] = None


@dataclass
class PerformanceTrendReport:
    """Comprehensive performance trend report."""
    report_timestamp: float
    analysis_period_hours: int
    overall_health_score: float  # 0-100
    key_trends: List[PerformanceTrend]
    stage_performance: Dict[str, PipelineStageMetrics]
    bottlenecks: List[str]
    recommendations: List[str]
    anomalies_detected: int
    performance_regression_alerts: List[str]


@dataclass
class PerformanceRegression:
    """Performance regression detection result."""
    metric_name: str
    regression_severity: str  # "minor", "moderate", "severe"
    current_value: float
    baseline_value: float
    degradation_percentage: float
    detected_at: float
    affected_operations: List[str]


class PipelineAnalytics:
    """
    Comprehensive pipeline performance analytics engine.
    
    Provides end-to-end analysis of pipeline performance,
    trend detection, bottleneck identification, and optimization recommendations.
    """
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.logger = get_logger(__name__)
        
        # Analytics state
        self.performance_baselines = self._initialize_baselines()
        self.trend_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.regression_alerts: deque = deque(maxlen=50)
        
        # Performance tracking
        self.pipeline_metrics_history: Dict[PipelineStage, deque] = {
            stage: deque(maxlen=1000) for stage in PipelineStage
        }
        
        # Analysis configuration
        self.trend_analysis_window_hours = 24
        self.regression_threshold = 0.15  # 15% degradation threshold
        self.volatility_threshold = 0.3  # 30% coefficient of variation
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info("Initialized PipelineAnalytics")
    
    def analyze_end_to_end_performance(self, time_window_hours: int = 1) -> Dict[str, Any]:
        """
        Analyze complete pipeline performance end-to-end.
        
        Args:
            time_window_hours: Analysis time window
            
        Returns:
            Comprehensive end-to-end performance analysis
        """
        cutoff_time = time.time() - (time_window_hours * 3600)
        
        analysis = {
            "analysis_timestamp": time.time(),
            "time_window_hours": time_window_hours,
            "pipeline_stages": {},
            "overall_metrics": {},
            "bottlenecks": [],
            "critical_path_analysis": {},
            "resource_efficiency": {}
        }
        
        # Analyze each pipeline stage
        total_pipeline_time = 0.0
        stage_durations = {}
        
        for stage in PipelineStage:
            stage_metrics = self._analyze_stage_performance(stage, cutoff_time)
            analysis["pipeline_stages"][stage.value] = stage_metrics
            
            if stage_metrics["avg_duration_ms"] > 0:
                stage_durations[stage.value] = stage_metrics["avg_duration_ms"]
                total_pipeline_time += stage_metrics["avg_duration_ms"]
        
        # Overall pipeline metrics
        if total_pipeline_time > 0:
            analysis["overall_metrics"] = {
                "total_pipeline_duration_ms": total_pipeline_time,
                "pipeline_throughput_per_minute": 60000 / total_pipeline_time if total_pipeline_time > 0 else 0,
                "bottleneck_stage": max(stage_durations.items(), key=lambda x: x[1])[0] if stage_durations else None,
                "parallel_efficiency": self._calculate_parallel_efficiency(stage_durations)
            }
        
        # Identify bottlenecks
        analysis["bottlenecks"] = self._identify_bottlenecks(stage_durations)
        
        # Critical path analysis
        analysis["critical_path_analysis"] = self._analyze_critical_path(stage_durations)
        
        # Resource efficiency
        analysis["resource_efficiency"] = self._analyze_resource_efficiency(cutoff_time)
        
        return analysis
    
    def detect_performance_trends(self, analysis_period_hours: int = 24) -> List[PerformanceTrend]:
        """
        Detect performance trends across key metrics.
        
        Args:
            analysis_period_hours: Period for trend analysis
            
        Returns:
            List of detected performance trends
        """
        trends = []
        cutoff_time = time.time() - (analysis_period_hours * 3600)
        
        # Key metrics to analyze for trends
        key_metrics = [
            "operation.similarity_computation.duration",
            "quantum.execution_time",
            "pipeline.retrieval.duration",
            "pipeline.rerank.duration",
            "quantum.fidelity",
            "search.cache_hit_rate"
        ]
        
        for metric_name in key_metrics:
            # Get time-series data
            samples = self.metrics_collector.get_samples_in_time_window(
                metric_name, analysis_period_hours * 3600
            )
            
            if len(samples) >= 10:  # Need sufficient data points
                trend = self._analyze_metric_trend(metric_name, samples, analysis_period_hours)
                if trend:
                    trends.append(trend)
        
        # Sort trends by significance
        trends.sort(key=lambda t: (
            ["high", "medium", "low"].index(t.significance),
            -abs(t.change_rate)
        ))
        
        return trends
    
    def detect_performance_regressions(self, detection_window_hours: int = 6) -> List[PerformanceRegression]:
        """
        Detect performance regressions by comparing against baselines.
        
        Args:
            detection_window_hours: Window for regression detection
            
        Returns:
            List of detected performance regressions
        """
        regressions = []
        cutoff_time = time.time() - (detection_window_hours * 3600)
        
        for metric_name, baseline_value in self.performance_baselines.items():
            # Get recent performance
            recent_samples = self.metrics_collector.get_samples_in_time_window(
                metric_name, detection_window_hours * 3600
            )
            
            if len(recent_samples) >= 5:
                current_value = np.mean([s.value for s in recent_samples])
                
                # Check for regression
                regression = self._detect_metric_regression(
                    metric_name, current_value, baseline_value
                )
                
                if regression:
                    regressions.append(regression)
                    
                    # Store in regression alerts
                    with self._lock:
                        self.regression_alerts.append(regression)
        
        return regressions
    
    def generate_performance_report(self, analysis_period_hours: int = 24) -> PerformanceTrendReport:
        """
        Generate comprehensive performance trend report.
        
        Args:
            analysis_period_hours: Period for analysis
            
        Returns:
            Comprehensive performance trend report
        """
        # Analyze trends
        trends = self.detect_performance_trends(analysis_period_hours)
        
        # Analyze pipeline stages
        stage_performance = {}
        for stage in PipelineStage:
            cutoff_time = time.time() - (analysis_period_hours * 3600)
            stage_metrics = self._analyze_stage_performance(stage, cutoff_time)
            
            stage_performance[stage.value] = PipelineStageMetrics(
                stage=stage,
                avg_duration_ms=stage_metrics.get("avg_duration_ms", 0),
                p95_duration_ms=stage_metrics.get("p95_duration_ms", 0),
                p99_duration_ms=stage_metrics.get("p99_duration_ms", 0),
                throughput_per_second=stage_metrics.get("throughput_per_second", 0),
                success_rate=stage_metrics.get("success_rate", 1.0),
                error_count=stage_metrics.get("error_count", 0),
                cache_hit_rate=stage_metrics.get("cache_hit_rate"),
                memory_usage_mb=stage_metrics.get("memory_usage_mb"),
                cpu_utilization=stage_metrics.get("cpu_utilization")
            )
        
        # Detect regressions
        regressions = self.detect_performance_regressions()
        
        # Calculate overall health score
        overall_health_score = self._calculate_overall_health_score(trends, stage_performance, regressions)
        
        # Identify bottlenecks
        bottlenecks = self._identify_system_bottlenecks(stage_performance)
        
        # Generate recommendations
        recommendations = self._generate_performance_recommendations(trends, stage_performance, regressions)
        
        # Count anomalies
        anomalies_detected = len([t for t in trends if t.direction == TrendDirection.VOLATILE])
        
        return PerformanceTrendReport(
            report_timestamp=time.time(),
            analysis_period_hours=analysis_period_hours,
            overall_health_score=overall_health_score,
            key_trends=trends[:10],  # Top 10 trends
            stage_performance=stage_performance,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            anomalies_detected=anomalies_detected,
            performance_regression_alerts=[r.metric_name for r in regressions]
        )
    
    def get_pipeline_health_score(self) -> float:
        """
        Calculate current pipeline health score (0-100).
        
        Returns:
            Health score from 0 (critical) to 100 (optimal)
        """
        health_factors = []
        
        # Performance against targets
        for metric_name, baseline in self.performance_baselines.items():
            stats = self.metrics_collector.get_metric_statistics(metric_name)
            if stats and stats.count > 0:
                if "latency" in metric_name or "duration" in metric_name:
                    # Lower is better
                    factor = min(1.0, baseline / stats.mean) if stats.mean > 0 else 1.0
                else:
                    # Higher is better
                    factor = min(1.0, stats.mean / baseline) if baseline > 0 else 1.0
                
                health_factors.append(factor)
        
        # Error rates
        error_stats = self.metrics_collector.get_metric_statistics("system.errors")
        if error_stats:
            error_factor = max(0.0, 1.0 - (error_stats.rate_per_second / 10.0))  # Penalize >10 errors/sec
            health_factors.append(error_factor)
        
        # Resource utilization
        cpu_stats = self.metrics_collector.get_metric_statistics("system.cpu_usage")
        if cpu_stats:
            cpu_factor = max(0.0, 1.0 - max(0.0, cpu_stats.mean - 80) / 20.0)  # Penalize >80% CPU
            health_factors.append(cpu_factor)
        
        memory_stats = self.metrics_collector.get_metric_statistics("system.memory_usage")
        if memory_stats:
            memory_factor = max(0.0, 1.0 - max(0.0, memory_stats.mean - 85) / 15.0)  # Penalize >85% memory
            health_factors.append(memory_factor)
        
        # Calculate weighted average
        if health_factors:
            return np.mean(health_factors) * 100
        else:
            return 100.0  # No data assumes healthy
    
    def _analyze_stage_performance(self, stage: PipelineStage, cutoff_time: float) -> Dict[str, Any]:
        """Analyze performance metrics for a specific pipeline stage."""
        stage_metrics = {
            "stage_name": stage.value,
            "avg_duration_ms": 0.0,
            "p95_duration_ms": 0.0,
            "p99_duration_ms": 0.0,
            "throughput_per_second": 0.0,
            "success_rate": 1.0,
            "error_count": 0,
            "total_operations": 0
        }
        
        # Map pipeline stages to metric names
        metric_mapping = {
            PipelineStage.DOCUMENT_RETRIEVAL: "pipeline.retrieval.duration",
            PipelineStage.EMBEDDING_COMPUTATION: "operation.embedding_computation.duration",
            PipelineStage.QUANTUM_PREPARATION: "operation.quantum_preparation.duration",
            PipelineStage.SIMILARITY_COMPUTATION: "operation.similarity_computation.duration",
            PipelineStage.RANKING_RECOMPUTATION: "pipeline.rerank.duration",
            PipelineStage.RESULT_AGGREGATION: "operation.result_aggregation.duration"
        }
        
        metric_name = metric_mapping.get(stage)
        if not metric_name:
            return stage_metrics
        
        # Get performance statistics
        stats = self.metrics_collector.get_metric_statistics(metric_name)
        if stats and stats.count > 0:
            stage_metrics.update({
                "avg_duration_ms": stats.mean,
                "p95_duration_ms": stats.p95,
                "p99_duration_ms": stats.p99,
                "total_operations": stats.count,
                "throughput_per_second": 1000.0 / stats.mean if stats.mean > 0 else 0.0
            })
        
        # Get error statistics
        error_metric = f"{metric_name}.errors"
        error_stats = self.metrics_collector.get_metric_statistics(error_metric)
        if error_stats:
            stage_metrics["error_count"] = error_stats.count
            if stats and stats.count > 0:
                stage_metrics["success_rate"] = max(0.0, 1.0 - error_stats.count / stats.count)
        
        return stage_metrics
    
    def _analyze_metric_trend(self, metric_name: str, samples: List, period_hours: int) -> Optional[PerformanceTrend]:
        """Analyze trend for a specific metric."""
        if len(samples) < 10:
            return None
        
        # Extract values and timestamps
        values = np.array([s.value for s in samples])
        timestamps = np.array([s.timestamp for s in samples])
        
        # Normalize timestamps to hours from start
        time_hours = (timestamps - timestamps[0]) / 3600.0
        
        # Linear regression for trend
        if len(time_hours) > 1 and np.std(time_hours) > 0:
            correlation = np.corrcoef(time_hours, values)[0, 1]
            slope = np.cov(time_hours, values)[0, 1] / np.var(time_hours)
        else:
            correlation = 0.0
            slope = 0.0
        
        # Calculate percentage change
        initial_mean = np.mean(values[:max(1, len(values)//4)])  # First quarter
        final_mean = np.mean(values[-(len(values)//4):])  # Last quarter
        
        if initial_mean != 0:
            change_rate = ((final_mean - initial_mean) / initial_mean) * 100
        else:
            change_rate = 0.0
        
        # Determine trend direction
        if abs(correlation) < 0.3:
            # Check for volatility
            cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
            if cv > self.volatility_threshold:
                direction = TrendDirection.VOLATILE
            else:
                direction = TrendDirection.STABLE
        elif correlation > 0:
            direction = TrendDirection.DEGRADING if "latency" in metric_name or "duration" in metric_name else TrendDirection.IMPROVING
        else:
            direction = TrendDirection.IMPROVING if "latency" in metric_name or "duration" in metric_name else TrendDirection.DEGRADING
        
        # Confidence based on correlation strength
        confidence = abs(correlation)
        
        # Significance based on change magnitude and confidence
        if abs(change_rate) > 20 and confidence > 0.6:
            significance = "high"
        elif abs(change_rate) > 10 and confidence > 0.4:
            significance = "medium"
        else:
            significance = "low"
        
        # Generate description
        direction_text = {
            TrendDirection.IMPROVING: "improving",
            TrendDirection.DEGRADING: "degrading", 
            TrendDirection.STABLE: "stable",
            TrendDirection.VOLATILE: "volatile"
        }
        
        description = f"{metric_name} is {direction_text[direction]} with {change_rate:+.1f}% change over {period_hours}h"
        
        return PerformanceTrend(
            metric_name=metric_name,
            direction=direction,
            change_rate=change_rate,
            confidence=confidence,
            period_days=period_hours // 24,
            significance=significance,
            description=description
        )
    
    def _detect_metric_regression(self, metric_name: str, current_value: float, baseline_value: float) -> Optional[PerformanceRegression]:
        """Detect performance regression for a metric."""
        if baseline_value <= 0:
            return None
        
        # For latency/duration metrics, higher is worse
        if "latency" in metric_name or "duration" in metric_name or "time" in metric_name:
            degradation = (current_value - baseline_value) / baseline_value
        else:
            # For other metrics, lower is worse
            degradation = (baseline_value - current_value) / baseline_value
        
        if degradation > self.regression_threshold:
            if degradation > 0.5:
                severity = "severe"
            elif degradation > 0.3:
                severity = "moderate"
            else:
                severity = "minor"
            
            return PerformanceRegression(
                metric_name=metric_name,
                regression_severity=severity,
                current_value=current_value,
                baseline_value=baseline_value,
                degradation_percentage=degradation * 100,
                detected_at=time.time(),
                affected_operations=[metric_name.split('.')[1] if '.' in metric_name else metric_name]
            )
        
        return None
    
    def _calculate_parallel_efficiency(self, stage_durations: Dict[str, float]) -> float:
        """Calculate parallel processing efficiency."""
        if not stage_durations:
            return 1.0
        
        # Theoretical parallel time (longest stage)
        max_duration = max(stage_durations.values())
        
        # Actual sequential time (sum of all stages)
        total_duration = sum(stage_durations.values())
        
        # Parallel efficiency
        if total_duration > 0:
            return max_duration / total_duration
        else:
            return 1.0
    
    def _identify_bottlenecks(self, stage_durations: Dict[str, float]) -> List[str]:
        """Identify performance bottlenecks."""
        if not stage_durations:
            return []
        
        bottlenecks = []
        total_time = sum(stage_durations.values())
        
        for stage, duration in stage_durations.items():
            # Consider a stage a bottleneck if it takes >40% of total time
            if duration / total_time > 0.4:
                bottlenecks.append(f"{stage} (takes {duration/total_time*100:.1f}% of pipeline time)")
        
        return bottlenecks
    
    def _analyze_critical_path(self, stage_durations: Dict[str, float]) -> Dict[str, Any]:
        """Analyze critical path through pipeline."""
        if not stage_durations:
            return {}
        
        # Sort stages by duration
        sorted_stages = sorted(stage_durations.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "critical_stage": sorted_stages[0][0] if sorted_stages else None,
            "critical_stage_duration_ms": sorted_stages[0][1] if sorted_stages else 0,
            "critical_path_percentage": (sorted_stages[0][1] / sum(stage_durations.values()) * 100) if sorted_stages and sum(stage_durations.values()) > 0 else 0,
            "optimization_potential": f"Optimizing {sorted_stages[0][0]} could improve overall performance by up to {sorted_stages[0][1] / sum(stage_durations.values()) * 100:.1f}%" if sorted_stages and sum(stage_durations.values()) > 0 else ""
        }
    
    def _analyze_resource_efficiency(self, cutoff_time: float) -> Dict[str, Any]:
        """Analyze resource utilization efficiency."""
        efficiency = {
            "cpu_efficiency": 0.0,
            "memory_efficiency": 0.0,
            "cache_efficiency": 0.0,
            "quantum_efficiency": 0.0
        }
        
        # CPU efficiency
        cpu_stats = self.metrics_collector.get_metric_statistics("system.cpu_usage")
        if cpu_stats:
            # Good efficiency is 60-80% utilization
            if 60 <= cpu_stats.mean <= 80:
                efficiency["cpu_efficiency"] = 1.0
            elif cpu_stats.mean < 60:
                efficiency["cpu_efficiency"] = cpu_stats.mean / 60.0
            else:
                efficiency["cpu_efficiency"] = max(0.0, (100 - cpu_stats.mean) / 20.0)
        
        # Memory efficiency
        memory_stats = self.metrics_collector.get_metric_statistics("system.memory_usage")
        if memory_stats:
            # Good efficiency is 50-75% utilization
            if 50 <= memory_stats.mean <= 75:
                efficiency["memory_efficiency"] = 1.0
            elif memory_stats.mean < 50:
                efficiency["memory_efficiency"] = memory_stats.mean / 50.0
            else:
                efficiency["memory_efficiency"] = max(0.0, (100 - memory_stats.mean) / 25.0)
        
        # Cache efficiency
        cache_stats = self.metrics_collector.get_metric_statistics("search.cache_hit_rate")
        if cache_stats:
            efficiency["cache_efficiency"] = cache_stats.mean
        
        # Quantum efficiency (based on fidelity and execution time)
        fidelity_stats = self.metrics_collector.get_metric_statistics("quantum.fidelity")
        quantum_time_stats = self.metrics_collector.get_metric_statistics("quantum.execution_time")
        
        if fidelity_stats and quantum_time_stats:
            # Efficiency = fidelity * (target_time / actual_time)
            target_time = 60.0  # Target 60ms execution time
            time_efficiency = min(1.0, target_time / quantum_time_stats.mean) if quantum_time_stats.mean > 0 else 1.0
            efficiency["quantum_efficiency"] = fidelity_stats.mean * time_efficiency
        
        return efficiency
    
    def _calculate_overall_health_score(self, trends: List[PerformanceTrend], 
                                      stage_performance: Dict[str, Any], 
                                      regressions: List[PerformanceRegression]) -> float:
        """Calculate overall system health score."""
        score = 100.0
        
        # Penalize based on regressions
        for regression in regressions:
            if regression.regression_severity == "severe":
                score -= 20
            elif regression.regression_severity == "moderate":
                score -= 10
            else:
                score -= 5
        
        # Penalize based on negative trends
        for trend in trends:
            if trend.direction == TrendDirection.DEGRADING and trend.significance == "high":
                score -= 15
            elif trend.direction == TrendDirection.DEGRADING and trend.significance == "medium":
                score -= 8
            elif trend.direction == TrendDirection.VOLATILE:
                score -= 5
        
        # Penalize based on poor stage performance
        for stage_name, metrics in stage_performance.items():
            if hasattr(metrics, 'success_rate') and metrics.success_rate < 0.95:
                score -= (1.0 - metrics.success_rate) * 20
        
        return max(0.0, score)
    
    def _identify_system_bottlenecks(self, stage_performance: Dict[str, Any]) -> List[str]:
        """Identify system-wide bottlenecks."""
        bottlenecks = []
        
        # Find stages with poor performance
        for stage_name, metrics in stage_performance.items():
            if hasattr(metrics, 'p95_duration_ms'):
                # Check against targets
                if stage_name == "similarity_computation" and metrics.p95_duration_ms > 120:
                    bottlenecks.append(f"Similarity computation P95 latency is {metrics.p95_duration_ms:.1f}ms (target: <120ms)")
                elif stage_name == "document_retrieval" and metrics.p95_duration_ms > 100:
                    bottlenecks.append(f"Document retrieval P95 latency is {metrics.p95_duration_ms:.1f}ms (target: <100ms)")
            
            if hasattr(metrics, 'success_rate') and metrics.success_rate < 0.98:
                bottlenecks.append(f"{stage_name} success rate is {metrics.success_rate*100:.1f}% (target: >98%)")
        
        return bottlenecks
    
    def _generate_performance_recommendations(self, trends: List[PerformanceTrend], 
                                           stage_performance: Dict[str, Any], 
                                           regressions: List[PerformanceRegression]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Recommendations based on regressions
        for regression in regressions:
            if "quantum" in regression.metric_name:
                recommendations.append(f"Investigate quantum computation performance - {regression.degradation_percentage:.1f}% degradation detected")
            elif "similarity" in regression.metric_name:
                recommendations.append("Consider switching to classical similarity computation temporarily")
            elif "cache" in regression.metric_name:
                recommendations.append("Review cache configuration and hit rates")
        
        # Recommendations based on trends
        for trend in trends:
            if trend.direction == TrendDirection.DEGRADING and "memory" in trend.metric_name:
                recommendations.append("Monitor memory usage - upward trend detected")
            elif trend.direction == TrendDirection.VOLATILE:
                recommendations.append(f"Investigate {trend.metric_name} stability - high volatility detected")
        
        # Recommendations based on stage performance
        for stage_name, metrics in stage_performance.items():
            if hasattr(metrics, 'avg_duration_ms'):
                if stage_name == "similarity_computation" and metrics.avg_duration_ms > 100:
                    recommendations.append("Consider quantum circuit optimization or classical fallback")
                elif stage_name == "document_retrieval" and metrics.avg_duration_ms > 75:
                    recommendations.append("Optimize vector search parameters or increase cache size")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _initialize_baselines(self) -> Dict[str, float]:
        """Initialize performance baselines."""
        return {
            "operation.similarity_computation.duration": 85.0,  # ms
            "quantum.execution_time": 60.0,  # ms
            "pipeline.retrieval.duration": 50.0,  # ms
            "pipeline.rerank.duration": 300.0,  # ms
            "quantum.fidelity": 0.95,
            "search.cache_hit_rate": 0.6
        }


__all__ = [
    "PipelineStage",
    "TrendDirection", 
    "PerformanceTrend",
    "PipelineStageMetrics",
    "PerformanceTrendReport",
    "PerformanceRegression",
    "PipelineAnalytics"
]
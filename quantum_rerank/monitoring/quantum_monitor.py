"""
Quantum-specific performance monitoring and analysis.

This module provides specialized monitoring for quantum computations,
including fidelity tracking, circuit analysis, and quantum-classical correlation.
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


class QuantumComputationType(Enum):
    """Types of quantum computations to monitor."""
    FIDELITY_COMPUTATION = "fidelity_computation"
    PARAMETER_PREDICTION = "parameter_prediction"
    CIRCUIT_EXECUTION = "circuit_execution"
    SIMILARITY_CALCULATION = "similarity_calculation"


@dataclass
class QuantumMetrics:
    """Quantum computation performance metrics."""
    computation_type: QuantumComputationType
    execution_time_ms: float
    circuit_depth: int
    gate_count: int
    qubit_count: int
    fidelity_value: Optional[float] = None
    classical_correlation: Optional[float] = None
    parameter_quality: Optional[float] = None
    memory_usage_mb: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantumHealthIndicator:
    """Health indicator for quantum computations."""
    name: str
    current_value: float
    target_value: float
    warning_threshold: float
    critical_threshold: float
    status: str = "unknown"  # healthy, warning, critical
    trend: str = "stable"  # improving, degrading, stable
    unit: str = ""


@dataclass
class QuantumAnomalyDetection:
    """Quantum computation anomaly detection result."""
    anomaly_type: str
    severity: str  # low, medium, high
    description: str
    affected_metrics: List[str]
    confidence_score: float
    timestamp: float = field(default_factory=time.time)


class QuantumPerformanceMonitor:
    """
    Specialized monitoring for quantum computations.
    
    Provides detailed monitoring of quantum computation performance,
    including circuit analysis, fidelity tracking, and anomaly detection.
    """
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.logger = get_logger(__name__)
        
        # Quantum-specific tracking
        self.quantum_metrics_history: Dict[QuantumComputationType, deque] = {
            comp_type: deque(maxlen=1000) for comp_type in QuantumComputationType
        }
        
        # Performance baselines
        self.performance_baselines = self._initialize_baselines()
        
        # Health indicators
        self.health_indicators = self._initialize_health_indicators()
        
        # Anomaly detection
        self.anomaly_detector = QuantumAnomalyDetector()
        self.detected_anomalies: deque = deque(maxlen=100)
        
        # Correlation tracking
        self.quantum_classical_correlations: deque = deque(maxlen=500)
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info("Initialized QuantumPerformanceMonitor")
    
    def monitor_fidelity_computation(self, embedding1: np.ndarray, 
                                   embedding2: np.ndarray,
                                   computation_func: callable) -> Tuple[Any, QuantumMetrics]:
        """
        Monitor quantum fidelity computation with detailed metrics.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector  
            computation_func: Function to compute fidelity
            
        Returns:
            Tuple of (computation_result, metrics)
        """
        start_time = time.time()
        
        # Initialize metrics
        metrics = QuantumMetrics(
            computation_type=QuantumComputationType.FIDELITY_COMPUTATION,
            execution_time_ms=0.0,
            circuit_depth=0,
            gate_count=0,
            qubit_count=len(embedding1)  # Simplified assumption
        )
        
        try:
            # Monitor circuit preparation
            prep_start = time.time()
            
            # Execute computation
            result = computation_func(embedding1, embedding2)
            
            # Extract fidelity value
            if isinstance(result, dict) and "fidelity" in result:
                metrics.fidelity_value = result["fidelity"]
            elif isinstance(result, (int, float)):
                metrics.fidelity_value = float(result)
            
            # Calculate execution time
            metrics.execution_time_ms = (time.time() - start_time) * 1000
            
            # Estimate circuit properties (would be more accurate with actual circuit)
            metrics.circuit_depth = self._estimate_circuit_depth(embedding1, embedding2)
            metrics.gate_count = self._estimate_gate_count(embedding1, embedding2)
            
            # Assess computation quality
            metrics.parameter_quality = self._assess_parameter_quality(
                embedding1, embedding2, metrics.fidelity_value
            )
            
            # Calculate classical correlation for comparison
            classical_similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            metrics.classical_correlation = float(classical_similarity)
            
            metrics.success = True
            
        except Exception as e:
            metrics.success = False
            metrics.error_message = str(e)
            metrics.execution_time_ms = (time.time() - start_time) * 1000
            result = None
        
        # Record metrics
        self._record_quantum_metrics(metrics)
        
        # Detect anomalies
        anomalies = self.anomaly_detector.detect_computation_anomalies(metrics)
        if anomalies:
            for anomaly in anomalies:
                self.detected_anomalies.append(anomaly)
                self.logger.warning(f"Quantum anomaly detected: {anomaly.description}")
        
        return result, metrics
    
    def monitor_parameter_prediction(self, embedding: np.ndarray,
                                   prediction_model: Any) -> QuantumMetrics:
        """Monitor quantum parameter prediction performance."""
        start_time = time.time()
        
        metrics = QuantumMetrics(
            computation_type=QuantumComputationType.PARAMETER_PREDICTION,
            execution_time_ms=0.0,
            circuit_depth=0,
            gate_count=0,
            qubit_count=len(embedding)
        )
        
        try:
            # Execute parameter prediction
            predicted_params = prediction_model.predict(embedding.reshape(1, -1))
            
            # Calculate execution time
            metrics.execution_time_ms = (time.time() - start_time) * 1000
            
            # Assess parameter quality
            if predicted_params is not None and len(predicted_params) > 0:
                params = predicted_params[0] if len(predicted_params.shape) > 1 else predicted_params
                metrics.parameter_quality = self._assess_predicted_parameters(params)
                
                # Store parameters in metadata
                metrics.metadata["predicted_parameters"] = params.tolist() if hasattr(params, 'tolist') else params
                metrics.metadata["parameter_count"] = len(params)
                metrics.metadata["parameter_variance"] = float(np.var(params)) if hasattr(params, '__len__') else 0.0
            
            metrics.success = True
            
        except Exception as e:
            metrics.success = False
            metrics.error_message = str(e)
            metrics.execution_time_ms = (time.time() - start_time) * 1000
        
        # Record metrics
        self._record_quantum_metrics(metrics)
        
        return metrics
    
    def monitor_circuit_execution(self, circuit_func: callable, 
                                *args, **kwargs) -> Tuple[Any, QuantumMetrics]:
        """Monitor quantum circuit execution."""
        start_time = time.time()
        
        metrics = QuantumMetrics(
            computation_type=QuantumComputationType.CIRCUIT_EXECUTION,
            execution_time_ms=0.0,
            circuit_depth=0,
            gate_count=0,
            qubit_count=0
        )
        
        try:
            # Execute circuit
            result = circuit_func(*args, **kwargs)
            
            # Calculate execution time
            metrics.execution_time_ms = (time.time() - start_time) * 1000
            
            # Extract circuit properties if available
            if hasattr(result, 'depth'):
                metrics.circuit_depth = result.depth()
            if hasattr(result, 'size'):
                metrics.gate_count = result.size()
            if hasattr(result, 'num_qubits'):
                metrics.qubit_count = result.num_qubits
            
            metrics.success = True
            
        except Exception as e:
            metrics.success = False
            metrics.error_message = str(e)
            metrics.execution_time_ms = (time.time() - start_time) * 1000
            result = None
        
        # Record metrics
        self._record_quantum_metrics(metrics)
        
        return result, metrics
    
    def get_quantum_health_status(self) -> Dict[str, Any]:
        """Get comprehensive quantum system health status."""
        with self._lock:
            # Update health indicators
            self._update_health_indicators()
            
            # Calculate overall health
            overall_health = self._calculate_overall_quantum_health()
            
            # Get recent performance trends
            performance_trends = self._analyze_performance_trends()
            
            # Get anomaly summary
            anomaly_summary = self._get_anomaly_summary()
            
            return {
                "overall_health": overall_health,
                "health_indicators": {
                    indicator.name: {
                        "current_value": indicator.current_value,
                        "target_value": indicator.target_value,
                        "status": indicator.status,
                        "trend": indicator.trend,
                        "unit": indicator.unit
                    }
                    for indicator in self.health_indicators
                },
                "performance_trends": performance_trends,
                "anomaly_summary": anomaly_summary,
                "quantum_classical_correlation": self._get_correlation_stats()
            }
    
    def get_quantum_performance_report(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Generate comprehensive quantum performance report."""
        cutoff_time = time.time() - (time_window_minutes * 60)
        
        report = {
            "time_window_minutes": time_window_minutes,
            "computation_types": {},
            "performance_summary": {},
            "anomalies": [],
            "recommendations": []
        }
        
        # Analyze each computation type
        for comp_type in QuantumComputationType:
            recent_metrics = [
                metric for metric in self.quantum_metrics_history[comp_type]
                if metric.timestamp >= cutoff_time
            ]
            
            if recent_metrics:
                comp_analysis = self._analyze_computation_type_performance(comp_type, recent_metrics)
                report["computation_types"][comp_type.value] = comp_analysis
        
        # Overall performance summary
        all_recent_metrics = []
        for metrics_list in self.quantum_metrics_history.values():
            all_recent_metrics.extend([
                metric for metric in metrics_list
                if metric.timestamp >= cutoff_time
            ])
        
        if all_recent_metrics:
            report["performance_summary"] = self._generate_performance_summary(all_recent_metrics)
        
        # Recent anomalies
        recent_anomalies = [
            anomaly for anomaly in self.detected_anomalies
            if anomaly.timestamp >= cutoff_time
        ]
        report["anomalies"] = [anomaly.__dict__ for anomaly in recent_anomalies]
        
        # Generate recommendations
        report["recommendations"] = self._generate_performance_recommendations(report)
        
        return report
    
    def _record_quantum_metrics(self, metrics: QuantumMetrics) -> None:
        """Record quantum metrics in history and metrics collector."""
        with self._lock:
            # Store in history
            self.quantum_metrics_history[metrics.computation_type].append(metrics)
            
            # Record in metrics collector
            tags = {
                "computation_type": metrics.computation_type.value,
                "component": "quantum",
                "success": str(metrics.success)
            }
            
            self.metrics_collector.record_timer("quantum.execution_time", 
                                              metrics.execution_time_ms, tags)
            self.metrics_collector.record_gauge("quantum.circuit_depth", 
                                               metrics.circuit_depth, "", tags)
            self.metrics_collector.record_gauge("quantum.gate_count", 
                                               metrics.gate_count, "", tags)
            self.metrics_collector.record_gauge("quantum.qubit_count", 
                                               metrics.qubit_count, "", tags)
            
            if metrics.fidelity_value is not None:
                self.metrics_collector.record_gauge("quantum.fidelity", 
                                                   metrics.fidelity_value, "", tags)
            
            if metrics.classical_correlation is not None:
                self.metrics_collector.record_gauge("quantum.classical_correlation", 
                                                   metrics.classical_correlation, "", tags)
                
                # Track correlation for analysis
                self.quantum_classical_correlations.append({
                    "quantum_fidelity": metrics.fidelity_value,
                    "classical_similarity": metrics.classical_correlation,
                    "timestamp": metrics.timestamp
                })
            
            if metrics.parameter_quality is not None:
                self.metrics_collector.record_gauge("quantum.parameter_quality", 
                                                   metrics.parameter_quality, "", tags)
            
            # Count operations
            self.metrics_collector.record_counter("quantum.operations", 1, tags)
            
            if not metrics.success:
                self.metrics_collector.record_counter("quantum.errors", 1, tags)
    
    def _estimate_circuit_depth(self, embedding1: np.ndarray, embedding2: np.ndarray) -> int:
        """Estimate circuit depth based on embedding dimensions."""
        # Simplified estimation - would be more accurate with actual circuit
        dim = len(embedding1)
        # Typical depth for fidelity computation scales with log(dim)
        return max(10, int(np.log2(dim) * 3))
    
    def _estimate_gate_count(self, embedding1: np.ndarray, embedding2: np.ndarray) -> int:
        """Estimate gate count based on embedding dimensions."""
        # Simplified estimation
        dim = len(embedding1)
        # Typical gate count scales with dimension
        return dim * 5  # Rough estimate
    
    def _assess_parameter_quality(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                                fidelity: Optional[float]) -> float:
        """Assess quality of quantum parameters."""
        if fidelity is None:
            return 0.5  # Neutral score
        
        # Quality based on fidelity value and embedding properties
        embedding_norm1 = np.linalg.norm(embedding1)
        embedding_norm2 = np.linalg.norm(embedding2)
        
        # Check if embeddings are well-normalized
        norm_quality = 1.0 - abs(1.0 - embedding_norm1) - abs(1.0 - embedding_norm2)
        norm_quality = max(0.0, min(1.0, norm_quality))
        
        # Combine fidelity and normalization quality
        overall_quality = 0.7 * fidelity + 0.3 * norm_quality
        
        return float(overall_quality)
    
    def _assess_predicted_parameters(self, parameters: np.ndarray) -> float:
        """Assess quality of predicted quantum parameters."""
        if len(parameters) == 0:
            return 0.0
        
        # Check parameter distribution
        param_variance = np.var(parameters)
        param_range = np.max(parameters) - np.min(parameters)
        
        # Good parameters should have reasonable variance and range
        variance_score = min(1.0, param_variance / 0.1)  # Normalized to expected variance
        range_score = min(1.0, param_range / (2 * np.pi))  # Normalized to parameter range
        
        # Check for NaN or infinite values
        validity_score = 1.0 if np.all(np.isfinite(parameters)) else 0.0
        
        # Combine scores
        quality = 0.4 * variance_score + 0.4 * range_score + 0.2 * validity_score
        
        return float(quality)
    
    def _update_health_indicators(self) -> None:
        """Update quantum health indicators based on recent metrics."""
        # Get recent metrics (last 5 minutes)
        recent_time = time.time() - 300
        
        for indicator in self.health_indicators:
            if indicator.name == "fidelity_computation_time":
                recent_metrics = [
                    m for m in self.quantum_metrics_history[QuantumComputationType.FIDELITY_COMPUTATION]
                    if m.timestamp >= recent_time and m.success
                ]
                if recent_metrics:
                    avg_time = np.mean([m.execution_time_ms for m in recent_metrics])
                    indicator.current_value = avg_time
                    indicator.status = self._assess_indicator_status(indicator)
            
            elif indicator.name == "quantum_fidelity_accuracy":
                recent_metrics = [
                    m for m in self.quantum_metrics_history[QuantumComputationType.FIDELITY_COMPUTATION]
                    if m.timestamp >= recent_time and m.success and m.fidelity_value is not None
                ]
                if recent_metrics:
                    avg_fidelity = np.mean([m.fidelity_value for m in recent_metrics])
                    indicator.current_value = avg_fidelity
                    indicator.status = self._assess_indicator_status(indicator)
            
            elif indicator.name == "quantum_classical_correlation":
                if self.quantum_classical_correlations:
                    recent_correlations = [
                        corr for corr in self.quantum_classical_correlations
                        if corr["timestamp"] >= recent_time
                    ]
                    if recent_correlations:
                        correlations = [
                            abs(corr["quantum_fidelity"] - corr["classical_similarity"])
                            for corr in recent_correlations
                            if corr["quantum_fidelity"] is not None
                        ]
                        if correlations:
                            avg_correlation = 1.0 - np.mean(correlations)  # Higher is better
                            indicator.current_value = max(0.0, avg_correlation)
                            indicator.status = self._assess_indicator_status(indicator)
    
    def _assess_indicator_status(self, indicator: QuantumHealthIndicator) -> str:
        """Assess health indicator status."""
        if indicator.name in ["fidelity_computation_time"]:
            # Lower is better
            if indicator.current_value <= indicator.target_value:
                return "healthy"
            elif indicator.current_value <= indicator.warning_threshold:
                return "warning"
            else:
                return "critical"
        else:
            # Higher is better
            if indicator.current_value >= indicator.target_value:
                return "healthy"
            elif indicator.current_value >= indicator.warning_threshold:
                return "warning"
            else:
                return "critical"
    
    def _calculate_overall_quantum_health(self) -> str:
        """Calculate overall quantum system health."""
        statuses = [indicator.status for indicator in self.health_indicators]
        
        if "critical" in statuses:
            return "critical"
        elif "warning" in statuses:
            return "warning"
        elif "healthy" in statuses:
            return "healthy"
        else:
            return "unknown"
    
    def _analyze_performance_trends(self) -> Dict[str, str]:
        """Analyze performance trends for quantum metrics."""
        trends = {}
        
        # Analyze trends for each computation type
        for comp_type in QuantumComputationType:
            recent_metrics = list(self.quantum_metrics_history[comp_type])[-20:]  # Last 20 operations
            
            if len(recent_metrics) >= 10:
                # Split into two halves for trend analysis
                first_half = recent_metrics[:len(recent_metrics)//2]
                second_half = recent_metrics[len(recent_metrics)//2:]
                
                first_avg_time = np.mean([m.execution_time_ms for m in first_half if m.success])
                second_avg_time = np.mean([m.execution_time_ms for m in second_half if m.success])
                
                if second_avg_time > first_avg_time * 1.1:
                    trends[f"{comp_type.value}_performance"] = "degrading"
                elif second_avg_time < first_avg_time * 0.9:
                    trends[f"{comp_type.value}_performance"] = "improving"
                else:
                    trends[f"{comp_type.value}_performance"] = "stable"
        
        return trends
    
    def _get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of recent anomalies."""
        recent_anomalies = list(self.detected_anomalies)[-10:]  # Last 10 anomalies
        
        return {
            "total_recent_anomalies": len(recent_anomalies),
            "anomaly_types": list(set(a.anomaly_type for a in recent_anomalies)),
            "high_severity_count": len([a for a in recent_anomalies if a.severity == "high"]),
            "latest_anomaly": recent_anomalies[-1].__dict__ if recent_anomalies else None
        }
    
    def _get_correlation_stats(self) -> Dict[str, float]:
        """Get quantum-classical correlation statistics."""
        if not self.quantum_classical_correlations:
            return {"correlation_coefficient": 0.0, "sample_count": 0}
        
        quantum_values = [c["quantum_fidelity"] for c in self.quantum_classical_correlations 
                         if c["quantum_fidelity"] is not None]
        classical_values = [c["classical_similarity"] for c in self.quantum_classical_correlations
                           if c["quantum_fidelity"] is not None]
        
        if len(quantum_values) >= 2:
            correlation = np.corrcoef(quantum_values, classical_values)[0, 1]
            return {
                "correlation_coefficient": float(correlation) if not np.isnan(correlation) else 0.0,
                "sample_count": len(quantum_values)
            }
        
        return {"correlation_coefficient": 0.0, "sample_count": len(quantum_values)}
    
    def _analyze_computation_type_performance(self, comp_type: QuantumComputationType,
                                            metrics: List[QuantumMetrics]) -> Dict[str, Any]:
        """Analyze performance for specific computation type."""
        if not metrics:
            return {"count": 0}
        
        execution_times = [m.execution_time_ms for m in metrics if m.success]
        success_rate = len([m for m in metrics if m.success]) / len(metrics)
        
        analysis = {
            "count": len(metrics),
            "success_rate": success_rate,
            "average_execution_time_ms": np.mean(execution_times) if execution_times else 0,
            "p95_execution_time_ms": np.percentile(execution_times, 95) if execution_times else 0,
            "max_execution_time_ms": np.max(execution_times) if execution_times else 0
        }
        
        # Type-specific analysis
        if comp_type == QuantumComputationType.FIDELITY_COMPUTATION:
            fidelity_values = [m.fidelity_value for m in metrics if m.success and m.fidelity_value is not None]
            if fidelity_values:
                analysis["average_fidelity"] = np.mean(fidelity_values)
                analysis["min_fidelity"] = np.min(fidelity_values)
        
        return analysis
    
    def _generate_performance_summary(self, all_metrics: List[QuantumMetrics]) -> Dict[str, Any]:
        """Generate overall performance summary."""
        if not all_metrics:
            return {}
        
        successful_metrics = [m for m in all_metrics if m.success]
        
        return {
            "total_operations": len(all_metrics),
            "successful_operations": len(successful_metrics),
            "overall_success_rate": len(successful_metrics) / len(all_metrics),
            "average_execution_time_ms": np.mean([m.execution_time_ms for m in successful_metrics]) if successful_metrics else 0,
            "total_quantum_time_ms": sum(m.execution_time_ms for m in all_metrics),
            "average_circuit_depth": np.mean([m.circuit_depth for m in successful_metrics]) if successful_metrics else 0,
            "average_gate_count": np.mean([m.gate_count for m in successful_metrics]) if successful_metrics else 0
        }
    
    def _generate_performance_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        performance_summary = report.get("performance_summary", {})
        
        # Check success rate
        success_rate = performance_summary.get("overall_success_rate", 1.0)
        if success_rate < 0.95:
            recommendations.append("Consider improving quantum computation stability - success rate below 95%")
        
        # Check execution time
        avg_time = performance_summary.get("average_execution_time_ms", 0)
        if avg_time > 100:
            recommendations.append("Quantum execution time is high - consider circuit optimization")
        
        # Check anomalies
        anomaly_summary = report.get("anomaly_summary", {})
        high_severity_count = anomaly_summary.get("high_severity_count", 0)
        if high_severity_count > 0:
            recommendations.append("High severity anomalies detected - investigate quantum computation issues")
        
        return recommendations
    
    def _initialize_baselines(self) -> Dict[str, float]:
        """Initialize performance baselines."""
        return {
            "fidelity_computation_time_ms": 85.0,
            "parameter_prediction_time_ms": 15.0,
            "circuit_execution_time_ms": 60.0,
            "quantum_fidelity": 0.95,
            "parameter_quality": 0.8,
            "quantum_classical_correlation": 0.95
        }
    
    def _initialize_health_indicators(self) -> List[QuantumHealthIndicator]:
        """Initialize quantum health indicators."""
        return [
            QuantumHealthIndicator(
                name="fidelity_computation_time",
                current_value=0.0,
                target_value=85.0,
                warning_threshold=120.0,
                critical_threshold=200.0,
                unit="ms"
            ),
            QuantumHealthIndicator(
                name="quantum_fidelity_accuracy",
                current_value=0.0,
                target_value=0.95,
                warning_threshold=0.90,
                critical_threshold=0.85,
                unit=""
            ),
            QuantumHealthIndicator(
                name="quantum_classical_correlation",
                current_value=0.0,
                target_value=0.95,
                warning_threshold=0.90,
                critical_threshold=0.80,
                unit=""
            )
        ]


class QuantumAnomalyDetector:
    """Anomaly detection for quantum computations."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def detect_computation_anomalies(self, metrics: QuantumMetrics) -> List[QuantumAnomalyDetection]:
        """Detect anomalies in quantum computation metrics."""
        anomalies = []
        
        # Check execution time anomalies
        if metrics.execution_time_ms > 500:  # Very high execution time
            anomalies.append(QuantumAnomalyDetection(
                anomaly_type="high_execution_time",
                severity="high",
                description=f"Quantum computation took {metrics.execution_time_ms:.1f}ms (expected <100ms)",
                affected_metrics=["execution_time_ms"],
                confidence_score=0.9
            ))
        
        # Check fidelity anomalies
        if metrics.fidelity_value is not None:
            if metrics.fidelity_value < 0.7:
                anomalies.append(QuantumAnomalyDetection(
                    anomaly_type="low_fidelity",
                    severity="medium",
                    description=f"Low quantum fidelity: {metrics.fidelity_value:.3f} (expected >0.9)",
                    affected_metrics=["fidelity_value"],
                    confidence_score=0.8
                ))
            elif metrics.fidelity_value > 1.0:
                anomalies.append(QuantumAnomalyDetection(
                    anomaly_type="invalid_fidelity",
                    severity="high",
                    description=f"Invalid fidelity value: {metrics.fidelity_value:.3f} (should be ≤1.0)",
                    affected_metrics=["fidelity_value"],
                    confidence_score=1.0
                ))
        
        # Check parameter quality anomalies
        if metrics.parameter_quality is not None and metrics.parameter_quality < 0.5:
            anomalies.append(QuantumAnomalyDetection(
                anomaly_type="low_parameter_quality",
                severity="medium",
                description=f"Low parameter quality: {metrics.parameter_quality:.3f}",
                affected_metrics=["parameter_quality"],
                confidence_score=0.7
            ))
        
        # Check quantum-classical correlation
        if (metrics.fidelity_value is not None and 
            metrics.classical_correlation is not None):
            
            correlation_diff = abs(metrics.fidelity_value - metrics.classical_correlation)
            if correlation_diff > 0.5:
                anomalies.append(QuantumAnomalyDetection(
                    anomaly_type="quantum_classical_mismatch",
                    severity="medium",
                    description=f"Large difference between quantum and classical results: {correlation_diff:.3f}",
                    affected_metrics=["fidelity_value", "classical_correlation"],
                    confidence_score=0.8
                ))
        
        return anomalies


__all__ = [
    "QuantumComputationType",
    "QuantumMetrics",
    "QuantumHealthIndicator",
    "QuantumAnomalyDetection",
    "QuantumPerformanceMonitor",
    "QuantumAnomalyDetector"
]
"""
Specialized quantum computation metric collector.

This module provides detailed quantum-specific metric collection
for circuits, gates, fidelity, and quantum state management.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from collections import deque
import numpy as np

from ..metrics_collector import MetricsCollector
from ...utils import get_logger


@dataclass
class QuantumCircuitMetrics:
    """Detailed quantum circuit metrics."""
    circuit_id: str
    depth: int
    gate_count: int
    qubit_count: int
    execution_time_ms: float
    fidelity_score: Optional[float]
    success: bool
    error_type: Optional[str] = None
    memory_usage_mb: float = 0.0
    timestamp: float = 0.0


class QuantumMetricsCollector:
    """
    Specialized collector for quantum computation metrics.
    
    Provides detailed tracking of quantum circuit execution,
    gate operations, fidelity measurements, and quantum state quality.
    """
    
    def __init__(self, base_collector: Optional[MetricsCollector] = None):
        self.base_collector = base_collector or MetricsCollector()
        self.logger = get_logger(__name__)
        
        # Quantum-specific tracking
        self.circuit_metrics: deque = deque(maxlen=1000)
        self.gate_operation_counts: Dict[str, int] = {}
        self.fidelity_history: deque = deque(maxlen=500)
        
        # Performance tracking
        self.quantum_execution_overhead = 0.0
        self.circuit_compilation_times: deque = deque(maxlen=100)
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info("Initialized QuantumMetricsCollector")
    
    def record_circuit_execution(self, circuit_metrics: QuantumCircuitMetrics) -> None:
        """Record comprehensive circuit execution metrics."""
        with self._lock:
            # Store detailed metrics
            circuit_metrics.timestamp = time.time()
            self.circuit_metrics.append(circuit_metrics)
            
            # Record in base collector
            tags = {
                "circuit_id": circuit_metrics.circuit_id,
                "success": str(circuit_metrics.success),
                "component": "quantum"
            }
            
            self.base_collector.record_timer("quantum.circuit.execution_time",
                                            circuit_metrics.execution_time_ms, tags)
            self.base_collector.record_gauge("quantum.circuit.depth",
                                           circuit_metrics.depth, "", tags)
            self.base_collector.record_gauge("quantum.circuit.gate_count",
                                           circuit_metrics.gate_count, "", tags)
            self.base_collector.record_gauge("quantum.circuit.qubit_count",
                                           circuit_metrics.qubit_count, "", tags)
            
            if circuit_metrics.fidelity_score is not None:
                self.base_collector.record_gauge("quantum.fidelity",
                                                circuit_metrics.fidelity_score, "", tags)
                self.fidelity_history.append({
                    "value": circuit_metrics.fidelity_score,
                    "timestamp": circuit_metrics.timestamp,
                    "circuit_id": circuit_metrics.circuit_id
                })
            
            if circuit_metrics.memory_usage_mb > 0:
                self.base_collector.record_gauge("quantum.memory_usage",
                                                circuit_metrics.memory_usage_mb, "MB", tags)
            
            # Count operations
            self.base_collector.record_counter("quantum.circuits.executed", 1, tags)
            
            if not circuit_metrics.success:
                error_tags = {**tags, "error_type": circuit_metrics.error_type or "unknown"}
                self.base_collector.record_counter("quantum.circuits.errors", 1, error_tags)
    
    def record_gate_operation(self, gate_type: str, execution_time_ms: float, 
                            success: bool = True) -> None:
        """Record individual gate operation metrics."""
        with self._lock:
            # Update gate counts
            if gate_type not in self.gate_operation_counts:
                self.gate_operation_counts[gate_type] = 0
            self.gate_operation_counts[gate_type] += 1
            
            # Record metrics
            tags = {
                "gate_type": gate_type,
                "success": str(success),
                "component": "quantum"
            }
            
            self.base_collector.record_timer("quantum.gate.execution_time",
                                            execution_time_ms, tags)
            self.base_collector.record_counter("quantum.gates.executed", 1, tags)
            
            if not success:
                self.base_collector.record_counter("quantum.gates.errors", 1, tags)
    
    def record_fidelity_measurement(self, embedding1: np.ndarray, embedding2: np.ndarray,
                                  quantum_fidelity: float, classical_similarity: float,
                                  computation_time_ms: float) -> None:
        """Record fidelity measurement with comparison to classical."""
        tags = {"component": "quantum", "measurement": "fidelity"}
        
        # Record fidelity values
        self.base_collector.record_gauge("quantum.fidelity.quantum_value",
                                        quantum_fidelity, "", tags)
        self.base_collector.record_gauge("quantum.fidelity.classical_value",
                                        classical_similarity, "", tags)
        
        # Record correlation between quantum and classical
        correlation_diff = abs(quantum_fidelity - classical_similarity)
        self.base_collector.record_gauge("quantum.fidelity.correlation_diff",
                                        correlation_diff, "", tags)
        
        # Record computation efficiency
        self.base_collector.record_timer("quantum.fidelity.computation_time",
                                        computation_time_ms, tags)
        
        # Assess measurement quality
        quality_score = self._assess_fidelity_quality(quantum_fidelity, classical_similarity)
        self.base_collector.record_gauge("quantum.fidelity.quality_score",
                                        quality_score, "", tags)
        
        # Store in history for trend analysis
        with self._lock:
            self.fidelity_history.append({
                "quantum_fidelity": quantum_fidelity,
                "classical_similarity": classical_similarity,
                "correlation_diff": correlation_diff,
                "computation_time_ms": computation_time_ms,
                "quality_score": quality_score,
                "timestamp": time.time()
            })
    
    def record_quantum_state_preparation(self, embedding: np.ndarray, 
                                       preparation_time_ms: float,
                                       state_quality_score: float) -> None:
        """Record quantum state preparation metrics."""
        tags = {"component": "quantum", "operation": "state_preparation"}
        
        # Record preparation metrics
        self.base_collector.record_timer("quantum.state.preparation_time",
                                        preparation_time_ms, tags)
        self.base_collector.record_gauge("quantum.state.quality_score",
                                        state_quality_score, "", tags)
        self.base_collector.record_gauge("quantum.state.embedding_dimension",
                                        len(embedding), "", tags)
        
        # Analyze embedding properties
        embedding_norm = np.linalg.norm(embedding)
        self.base_collector.record_gauge("quantum.state.embedding_norm",
                                        embedding_norm, "", tags)
        
        # Count state preparations
        self.base_collector.record_counter("quantum.states.prepared", 1, tags)
    
    def record_parameter_prediction(self, prediction_time_ms: float,
                                  parameter_count: int, parameter_quality: float) -> None:
        """Record quantum parameter prediction metrics."""
        tags = {"component": "quantum", "operation": "parameter_prediction"}
        
        self.base_collector.record_timer("quantum.parameters.prediction_time",
                                        prediction_time_ms, tags)
        self.base_collector.record_gauge("quantum.parameters.count",
                                        parameter_count, "", tags)
        self.base_collector.record_gauge("quantum.parameters.quality_score",
                                        parameter_quality, "", tags)
        self.base_collector.record_counter("quantum.parameters.predicted", 1, tags)
    
    def get_quantum_performance_summary(self, time_window_seconds: int = 300) -> Dict[str, Any]:
        """Get comprehensive quantum performance summary."""
        cutoff_time = time.time() - time_window_seconds
        
        with self._lock:
            # Recent circuit metrics
            recent_circuits = [
                c for c in self.circuit_metrics
                if c.timestamp >= cutoff_time
            ]
            
            # Recent fidelity measurements
            recent_fidelity = [
                f for f in self.fidelity_history
                if f["timestamp"] >= cutoff_time
            ]
            
            summary = {
                "time_window_seconds": time_window_seconds,
                "circuit_execution": self._analyze_circuit_performance(recent_circuits),
                "fidelity_analysis": self._analyze_fidelity_performance(recent_fidelity),
                "gate_operations": dict(self.gate_operation_counts),
                "quantum_efficiency": self._calculate_quantum_efficiency(recent_circuits, recent_fidelity)
            }
            
            return summary
    
    def get_quantum_health_indicators(self) -> Dict[str, Any]:
        """Get quantum system health indicators."""
        # Recent performance (last 5 minutes)
        recent_time = time.time() - 300
        
        with self._lock:
            recent_circuits = [
                c for c in self.circuit_metrics
                if c.timestamp >= recent_time
            ]
            
            recent_fidelity = [
                f for f in self.fidelity_history
                if f["timestamp"] >= recent_time
            ]
            
            indicators = {
                "circuit_success_rate": self._calculate_success_rate(recent_circuits),
                "average_fidelity": self._calculate_average_fidelity(recent_fidelity),
                "quantum_classical_correlation": self._calculate_quantum_classical_correlation(recent_fidelity),
                "execution_time_trend": self._analyze_execution_time_trend(recent_circuits),
                "gate_error_rate": self._calculate_gate_error_rate(),
                "overall_quantum_health": "unknown"
            }
            
            # Calculate overall health
            indicators["overall_quantum_health"] = self._assess_overall_quantum_health(indicators)
            
            return indicators
    
    def _assess_fidelity_quality(self, quantum_fidelity: float, classical_similarity: float) -> float:
        """Assess the quality of a fidelity measurement."""
        # Quality based on fidelity value and correlation with classical
        fidelity_quality = quantum_fidelity  # Higher fidelity is better
        
        # Correlation quality (lower difference is better)
        correlation_diff = abs(quantum_fidelity - classical_similarity)
        correlation_quality = max(0.0, 1.0 - correlation_diff)
        
        # Combined quality score
        quality_score = 0.7 * fidelity_quality + 0.3 * correlation_quality
        
        return float(quality_score)
    
    def _analyze_circuit_performance(self, circuits: List[QuantumCircuitMetrics]) -> Dict[str, Any]:
        """Analyze circuit execution performance."""
        if not circuits:
            return {"count": 0}
        
        successful_circuits = [c for c in circuits if c.success]
        
        analysis = {
            "total_circuits": len(circuits),
            "successful_circuits": len(successful_circuits),
            "success_rate": len(successful_circuits) / len(circuits),
            "average_execution_time_ms": 0.0,
            "average_depth": 0.0,
            "average_gate_count": 0.0
        }
        
        if successful_circuits:
            analysis.update({
                "average_execution_time_ms": np.mean([c.execution_time_ms for c in successful_circuits]),
                "p95_execution_time_ms": np.percentile([c.execution_time_ms for c in successful_circuits], 95),
                "average_depth": np.mean([c.depth for c in successful_circuits]),
                "average_gate_count": np.mean([c.gate_count for c in successful_circuits])
            })
        
        return analysis
    
    def _analyze_fidelity_performance(self, fidelity_measurements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze fidelity measurement performance."""
        if not fidelity_measurements:
            return {"count": 0}
        
        quantum_values = [f["quantum_fidelity"] for f in fidelity_measurements]
        classical_values = [f["classical_similarity"] for f in fidelity_measurements]
        correlation_diffs = [f["correlation_diff"] for f in fidelity_measurements]
        quality_scores = [f["quality_score"] for f in fidelity_measurements]
        
        analysis = {
            "measurement_count": len(fidelity_measurements),
            "average_quantum_fidelity": np.mean(quantum_values),
            "average_classical_similarity": np.mean(classical_values),
            "average_correlation_diff": np.mean(correlation_diffs),
            "average_quality_score": np.mean(quality_scores),
            "quantum_advantage": np.mean(quantum_values) - np.mean(classical_values)
        }
        
        return analysis
    
    def _calculate_quantum_efficiency(self, circuits: List[QuantumCircuitMetrics],
                                    fidelity_measurements: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate quantum computation efficiency metrics."""
        efficiency = {
            "temporal_efficiency": 0.0,
            "fidelity_efficiency": 0.0,
            "resource_efficiency": 0.0,
            "overall_efficiency": 0.0
        }
        
        # Temporal efficiency (target execution time vs actual)
        if circuits:
            avg_execution_time = np.mean([c.execution_time_ms for c in circuits if c.success])
            target_time = 60.0  # Target 60ms
            efficiency["temporal_efficiency"] = min(1.0, target_time / avg_execution_time) if avg_execution_time > 0 else 1.0
        
        # Fidelity efficiency (actual fidelity vs target)
        if fidelity_measurements:
            avg_fidelity = np.mean([f["quantum_fidelity"] for f in fidelity_measurements])
            target_fidelity = 0.95  # Target 95% fidelity
            efficiency["fidelity_efficiency"] = avg_fidelity / target_fidelity
        
        # Resource efficiency (based on gate count and depth)
        if circuits:
            successful_circuits = [c for c in circuits if c.success]
            if successful_circuits:
                avg_gates = np.mean([c.gate_count for c in successful_circuits])
                avg_depth = np.mean([c.depth for c in successful_circuits])
                # Efficiency inversely related to resource usage
                efficiency["resource_efficiency"] = 1.0 / (1.0 + avg_gates / 100.0 + avg_depth / 50.0)
        
        # Overall efficiency
        efficiency["overall_efficiency"] = np.mean([
            efficiency["temporal_efficiency"],
            efficiency["fidelity_efficiency"], 
            efficiency["resource_efficiency"]
        ])
        
        return efficiency
    
    def _calculate_success_rate(self, circuits: List[QuantumCircuitMetrics]) -> float:
        """Calculate circuit execution success rate."""
        if not circuits:
            return 1.0
        
        return len([c for c in circuits if c.success]) / len(circuits)
    
    def _calculate_average_fidelity(self, fidelity_measurements: List[Dict[str, Any]]) -> float:
        """Calculate average quantum fidelity."""
        if not fidelity_measurements:
            return 0.0
        
        return np.mean([f["quantum_fidelity"] for f in fidelity_measurements])
    
    def _calculate_quantum_classical_correlation(self, fidelity_measurements: List[Dict[str, Any]]) -> float:
        """Calculate correlation between quantum and classical results."""
        if len(fidelity_measurements) < 2:
            return 0.0
        
        quantum_values = [f["quantum_fidelity"] for f in fidelity_measurements]
        classical_values = [f["classical_similarity"] for f in fidelity_measurements]
        
        correlation = np.corrcoef(quantum_values, classical_values)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def _analyze_execution_time_trend(self, circuits: List[QuantumCircuitMetrics]) -> str:
        """Analyze execution time trend."""
        if len(circuits) < 5:
            return "insufficient_data"
        
        execution_times = [c.execution_time_ms for c in circuits if c.success]
        if len(execution_times) < 3:
            return "insufficient_data"
        
        # Simple trend analysis
        recent_avg = np.mean(execution_times[-3:])
        earlier_avg = np.mean(execution_times[:-3])
        
        if recent_avg > earlier_avg * 1.1:
            return "degrading"
        elif recent_avg < earlier_avg * 0.9:
            return "improving"
        else:
            return "stable"
    
    def _calculate_gate_error_rate(self) -> float:
        """Calculate gate operation error rate."""
        # This would need to be tracked per gate operation
        # For now, return a placeholder
        return 0.01  # 1% error rate placeholder
    
    def _assess_overall_quantum_health(self, indicators: Dict[str, Any]) -> str:
        """Assess overall quantum system health."""
        health_score = 0
        
        # Success rate contribution
        if indicators["circuit_success_rate"] >= 0.95:
            health_score += 25
        elif indicators["circuit_success_rate"] >= 0.90:
            health_score += 15
        
        # Fidelity contribution
        if indicators["average_fidelity"] >= 0.95:
            health_score += 25
        elif indicators["average_fidelity"] >= 0.90:
            health_score += 15
        
        # Correlation contribution
        if indicators["quantum_classical_correlation"] >= 0.95:
            health_score += 25
        elif indicators["quantum_classical_correlation"] >= 0.85:
            health_score += 15
        
        # Trend contribution
        if indicators["execution_time_trend"] == "improving":
            health_score += 25
        elif indicators["execution_time_trend"] == "stable":
            health_score += 20
        elif indicators["execution_time_trend"] == "degrading":
            health_score += 5
        
        if health_score >= 80:
            return "excellent"
        elif health_score >= 60:
            return "good"
        elif health_score >= 40:
            return "fair"
        else:
            return "poor"


__all__ = [
    "QuantumCircuitMetrics",
    "QuantumMetricsCollector"
]
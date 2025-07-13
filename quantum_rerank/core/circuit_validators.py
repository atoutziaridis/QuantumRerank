"""
Circuit validation and performance utilities for QuantumRerank.

This module provides comprehensive validation and performance analysis
for quantum circuits to ensure PRD compliance and optimal performance.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from ..config.settings import QuantumConfig, PerformanceConfig

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue found in a circuit."""
    severity: ValidationSeverity
    message: str
    constraint: str
    actual_value: Any
    expected_value: Any


@dataclass
class ValidationResult:
    """Result of circuit validation."""
    is_valid: bool
    issues: List[ValidationIssue]
    performance_score: float
    recommendations: List[str]


@dataclass
class PerformanceMetrics:
    """Performance metrics for circuit operations."""
    encoding_time_ms: float
    simulation_time_ms: float
    total_time_ms: float
    memory_usage_mb: float
    circuit_depth: int
    circuit_size: int
    success_rate: float
    prd_compliant: bool


class CircuitValidator:
    """
    Comprehensive circuit validation against PRD requirements.
    
    Validates circuits against:
    - PRD Section 4.1: System Requirements (2-4 qubits, ≤15 gates)
    - Performance targets (<100ms simulation supporting similarity computation)
    - Memory and resource constraints
    """
    
    def __init__(self, quantum_config: Optional[QuantumConfig] = None,
                 performance_config: Optional[PerformanceConfig] = None):
        """
        Initialize circuit validator.
        
        Args:
            quantum_config: Quantum configuration for constraints
            performance_config: Performance configuration for targets
        """
        self.quantum_config = quantum_config or QuantumConfig()
        self.performance_config = performance_config or PerformanceConfig()
        
        logger.info("Initialized CircuitValidator with PRD compliance checks")
    
    def validate_circuit(self, circuit: QuantumCircuit) -> ValidationResult:
        """
        Comprehensive validation of a quantum circuit.
        
        Args:
            circuit: Quantum circuit to validate
            
        Returns:
            ValidationResult with detailed analysis
        """
        issues = []
        recommendations = []
        
        # PRD Constraint Validations
        issues.extend(self._validate_qubit_constraints(circuit))
        issues.extend(self._validate_depth_constraints(circuit))
        issues.extend(self._validate_gate_constraints(circuit))
        
        # Performance Validations
        issues.extend(self._validate_performance_constraints(circuit))
        
        # Resource Validations
        issues.extend(self._validate_resource_constraints(circuit))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(circuit, issues)
        
        # Calculate overall validity and performance score
        is_valid = not any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                          for issue in issues)
        performance_score = self._calculate_performance_score(circuit, issues)
        
        logger.debug(f"Validated circuit '{circuit.name}': valid={is_valid}, score={performance_score:.2f}")
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            performance_score=performance_score,
            recommendations=recommendations
        )
    
    def _validate_qubit_constraints(self, circuit: QuantumCircuit) -> List[ValidationIssue]:
        """Validate qubit count constraints from PRD."""
        issues = []
        
        min_qubits = 2
        max_qubits = 4
        actual_qubits = circuit.num_qubits
        
        if actual_qubits < min_qubits:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Circuit has too few qubits for quantum advantage",
                constraint="min_qubits",
                actual_value=actual_qubits,
                expected_value=f">= {min_qubits}"
            ))
        elif actual_qubits > max_qubits:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                message=f"Circuit exceeds PRD qubit limit for classical simulation",
                constraint="max_qubits",
                actual_value=actual_qubits,
                expected_value=f"<= {max_qubits}"
            ))
        
        return issues
    
    def _validate_depth_constraints(self, circuit: QuantumCircuit) -> List[ValidationIssue]:
        """Validate circuit depth constraints from PRD."""
        issues = []
        
        max_depth = self.quantum_config.max_circuit_depth
        actual_depth = circuit.depth()
        
        if actual_depth > max_depth:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                message=f"Circuit depth exceeds PRD limit",
                constraint="max_circuit_depth",
                actual_value=actual_depth,
                expected_value=f"<= {max_depth}"
            ))
        elif actual_depth > max_depth * 0.8:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Circuit depth approaching PRD limit",
                constraint="circuit_depth_warning",
                actual_value=actual_depth,
                expected_value=f"< {max_depth * 0.8}"
            ))
        
        return issues
    
    def _validate_gate_constraints(self, circuit: QuantumCircuit) -> List[ValidationIssue]:
        """Validate gate count and types."""
        issues = []
        
        circuit_size = circuit.size()
        gate_counts = circuit.count_ops()
        
        # Check for excessive gate count
        max_gates = 50  # Reasonable limit for small circuits
        if circuit_size > max_gates:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Circuit has many gates, may impact performance",
                constraint="gate_count",
                actual_value=circuit_size,
                expected_value=f"<= {max_gates}"
            ))
        
        # Check for unsupported or inefficient gates
        inefficient_gates = ['u1', 'u2', 'u3']  # Gates that could be optimized
        for gate in inefficient_gates:
            if gate in gate_counts:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Circuit uses {gate} gates which could be optimized",
                    constraint="gate_efficiency",
                    actual_value=gate_counts[gate],
                    expected_value="0"
                ))
        
        return issues
    
    def _validate_performance_constraints(self, circuit: QuantumCircuit) -> List[ValidationIssue]:
        """Validate performance-related constraints."""
        issues = []
        
        # Estimate simulation complexity
        estimated_complexity = 2 ** circuit.num_qubits * circuit.depth()
        complexity_threshold = 1000  # Empirical threshold for fast simulation
        
        if estimated_complexity > complexity_threshold:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Circuit complexity may impact performance",
                constraint="simulation_complexity",
                actual_value=estimated_complexity,
                expected_value=f"< {complexity_threshold}"
            ))
        
        # Check for performance-impacting patterns
        two_qubit_gates = sum(1 for instruction in circuit.data 
                             if instruction.operation.num_qubits == 2)
        if two_qubit_gates > 10:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"Many two-qubit gates may slow simulation",
                constraint="two_qubit_gate_count",
                actual_value=two_qubit_gates,
                expected_value="<= 10"
            ))
        
        return issues
    
    def _validate_resource_constraints(self, circuit: QuantumCircuit) -> List[ValidationIssue]:
        """Validate resource usage constraints."""
        issues = []
        
        # Estimate memory usage
        estimated_memory_mb = (2 ** circuit.num_qubits * 16) / (1024 * 1024)  # Complex amplitudes
        memory_limit_mb = 100  # Conservative limit for statevector simulation
        
        if estimated_memory_mb > memory_limit_mb:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Circuit may exceed memory limits",
                constraint="memory_usage",
                actual_value=f"{estimated_memory_mb:.2f} MB",
                expected_value=f"< {memory_limit_mb} MB"
            ))
        
        return issues
    
    def _generate_recommendations(self, circuit: QuantumCircuit, 
                                issues: List[ValidationIssue]) -> List[str]:
        """Generate optimization recommendations based on validation issues."""
        recommendations = []
        
        # Recommendations based on issues
        for issue in issues:
            if issue.constraint == "max_qubits":
                recommendations.append("Consider using amplitude encoding to fit embeddings in fewer qubits")
            elif issue.constraint == "max_circuit_depth":
                recommendations.append("Simplify circuit or use gate optimization to reduce depth")
            elif issue.constraint == "gate_count":
                recommendations.append("Use circuit transpilation to reduce gate count")
            elif issue.constraint == "simulation_complexity":
                recommendations.append("Consider angle encoding instead of amplitude encoding for large embeddings")
            elif issue.constraint == "two_qubit_gate_count":
                recommendations.append("Minimize entangling gates for better simulation performance")
        
        # General recommendations
        if circuit.depth() > 5:
            recommendations.append("Consider circuit optimization passes to reduce depth")
        
        if circuit.num_parameters > 0:
            recommendations.append("Parameterized circuits may benefit from parameter binding")
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _calculate_performance_score(self, circuit: QuantumCircuit, 
                                   issues: List[ValidationIssue]) -> float:
        """Calculate overall performance score (0-100)."""
        base_score = 100.0
        
        # Deduct points for issues
        for issue in issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                base_score -= 30
            elif issue.severity == ValidationSeverity.ERROR:
                base_score -= 20
            elif issue.severity == ValidationSeverity.WARNING:
                base_score -= 10
            elif issue.severity == ValidationSeverity.INFO:
                base_score -= 5
        
        # Bonus for efficient circuits
        if circuit.depth() <= 5:
            base_score += 5
        
        if circuit.num_qubits == 2:  # Most efficient
            base_score += 5
        
        return max(0, min(100, base_score))


class PerformanceAnalyzer:
    """
    Analyzes and benchmarks quantum circuit performance.
    
    Provides detailed performance analysis against PRD targets.
    """
    
    def __init__(self, quantum_config: Optional[QuantumConfig] = None):
        """Initialize performance analyzer."""
        self.quantum_config = quantum_config or QuantumConfig()
        logger.info("Initialized PerformanceAnalyzer")
    
    def analyze_circuit_performance(self, circuit: QuantumCircuit,
                                  num_trials: int = 10) -> PerformanceMetrics:
        """
        Analyze performance characteristics of a circuit.
        
        Args:
            circuit: Circuit to analyze
            num_trials: Number of trials for statistical analysis
            
        Returns:
            PerformanceMetrics with detailed analysis
        """
        from .quantum_circuits import BasicQuantumCircuits
        
        # Initialize circuit handler for simulation
        circuit_handler = BasicQuantumCircuits(self.quantum_config)
        
        encoding_times = []
        simulation_times = []
        success_count = 0
        
        logger.debug(f"Analyzing performance of circuit '{circuit.name}' over {num_trials} trials")
        
        for trial in range(num_trials):
            # Measure simulation time
            start_time = time.time()
            result = circuit_handler.simulate_circuit(circuit)
            simulation_time = (time.time() - start_time) * 1000
            
            simulation_times.append(simulation_time)
            if result.success:
                success_count += 1
        
        # Calculate statistics
        avg_simulation_time = np.mean(simulation_times)
        total_time = avg_simulation_time  # No encoding time for pre-built circuit
        success_rate = success_count / num_trials
        
        # Estimate memory usage
        memory_usage_mb = (2 ** circuit.num_qubits * 16) / (1024 * 1024)
        
        # Check PRD compliance
        prd_compliant = (avg_simulation_time < 100 and  # <100ms simulation target
                        circuit.num_qubits <= 4 and    # ≤4 qubits
                        circuit.depth() <= 15)         # ≤15 gate depth
        
        return PerformanceMetrics(
            encoding_time_ms=0.0,  # Not applicable for pre-built circuit
            simulation_time_ms=avg_simulation_time,
            total_time_ms=total_time,
            memory_usage_mb=memory_usage_mb,
            circuit_depth=circuit.depth(),
            circuit_size=circuit.size(),
            success_rate=success_rate,
            prd_compliant=prd_compliant
        )
    
    def benchmark_encoding_performance(self, embeddings: List[np.ndarray],
                                     encoding_methods: List[str] = None) -> Dict[str, PerformanceMetrics]:
        """
        Benchmark performance of different encoding methods.
        
        Args:
            embeddings: List of embeddings to test
            encoding_methods: Methods to test (default: all available)
            
        Returns:
            Dictionary mapping method names to performance metrics
        """
        from .quantum_circuits import BasicQuantumCircuits
        
        if encoding_methods is None:
            encoding_methods = ['amplitude', 'angle', 'dense_angle']
        
        circuit_handler = BasicQuantumCircuits(self.quantum_config)
        results = {}
        
        logger.info(f"Benchmarking {len(encoding_methods)} encoding methods on {len(embeddings)} embeddings")
        
        for method in encoding_methods:
            encoding_times = []
            simulation_times = []
            success_count = 0
            circuit_depths = []
            circuit_sizes = []
            
            for embedding in embeddings:
                # Measure encoding time
                start_time = time.time()
                
                if method == 'amplitude':
                    circuit = circuit_handler.amplitude_encode_embedding(embedding)
                elif method == 'angle':
                    circuit = circuit_handler.angle_encode_embedding(embedding)
                elif method == 'dense_angle':
                    circuit = circuit_handler.dense_angle_encoding(embedding)
                else:
                    raise ValueError(f"Unknown encoding method: {method}")
                
                encoding_time = (time.time() - start_time) * 1000
                
                # Measure simulation time
                start_time = time.time()
                result = circuit_handler.simulate_circuit(circuit)
                simulation_time = (time.time() - start_time) * 1000
                
                # Collect metrics
                encoding_times.append(encoding_time)
                simulation_times.append(simulation_time)
                circuit_depths.append(circuit.depth())
                circuit_sizes.append(circuit.size())
                
                if result.success:
                    success_count += 1
            
            # Calculate statistics
            avg_encoding_time = np.mean(encoding_times)
            avg_simulation_time = np.mean(simulation_times)
            total_time = avg_encoding_time + avg_simulation_time
            success_rate = success_count / len(embeddings)
            avg_depth = np.mean(circuit_depths)
            avg_size = np.mean(circuit_sizes)
            
            # Estimate memory usage (worst case for this method)
            memory_usage_mb = (2 ** self.quantum_config.n_qubits * 16) / (1024 * 1024)
            
            # Check PRD compliance
            prd_compliant = (total_time < 100 and           # <100ms total time
                           avg_depth <= 15 and             # ≤15 gate depth
                           self.quantum_config.n_qubits <= 4)  # ≤4 qubits
            
            results[method] = PerformanceMetrics(
                encoding_time_ms=avg_encoding_time,
                simulation_time_ms=avg_simulation_time,
                total_time_ms=total_time,
                memory_usage_mb=memory_usage_mb,
                circuit_depth=int(avg_depth),
                circuit_size=int(avg_size),
                success_rate=success_rate,
                prd_compliant=prd_compliant
            )
            
            logger.info(f"Method '{method}': {total_time:.2f}ms total, PRD compliant: {prd_compliant}")
        
        return results
    
    def generate_performance_report(self, metrics: Dict[str, PerformanceMetrics]) -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            Formatted performance report string
        """
        report = []
        report.append("=" * 60)
        report.append("QUANTUM CIRCUIT PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary statistics
        total_methods = len(metrics)
        compliant_methods = sum(1 for m in metrics.values() if m.prd_compliant)
        avg_time = np.mean([m.total_time_ms for m in metrics.values()])
        avg_success = np.mean([m.success_rate for m in metrics.values()])
        
        report.append(f"Summary:")
        report.append(f"  Methods tested: {total_methods}")
        report.append(f"  PRD compliant: {compliant_methods}/{total_methods} ({compliant_methods/total_methods*100:.1f}%)")
        report.append(f"  Average time: {avg_time:.2f}ms")
        report.append(f"  Average success rate: {avg_success*100:.1f}%")
        report.append("")
        
        # PRD targets
        report.append("PRD Targets:")
        report.append("  ✓ <100ms total computation time")
        report.append("  ✓ ≤4 qubits")
        report.append("  ✓ ≤15 gate depth")
        report.append("")
        
        # Detailed results
        report.append("Detailed Results:")
        report.append("-" * 40)
        
        for method, metric in metrics.items():
            status = "✓" if metric.prd_compliant else "✗"
            report.append(f"{status} {method.upper()}:")
            report.append(f"    Encoding:    {metric.encoding_time_ms:.2f}ms")
            report.append(f"    Simulation:  {metric.simulation_time_ms:.2f}ms")
            report.append(f"    Total:       {metric.total_time_ms:.2f}ms")
            report.append(f"    Success:     {metric.success_rate*100:.1f}%")
            report.append(f"    Depth:       {metric.circuit_depth}")
            report.append(f"    Memory:      {metric.memory_usage_mb:.2f}MB")
            report.append("")
        
        # Recommendations
        report.append("Recommendations:")
        if compliant_methods == total_methods:
            report.append("  ✓ All methods meet PRD requirements")
        else:
            fastest_method = min(metrics.keys(), key=lambda k: metrics[k].total_time_ms)
            report.append(f"  • Use '{fastest_method}' for best performance ({metrics[fastest_method].total_time_ms:.2f}ms)")
            
            for method, metric in metrics.items():
                if not metric.prd_compliant:
                    if metric.total_time_ms >= 100:
                        report.append(f"  • Optimize '{method}' to reduce computation time")
                    if metric.circuit_depth > 15:
                        report.append(f"  • Simplify '{method}' circuits to reduce depth")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
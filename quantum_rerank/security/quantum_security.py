"""
Quantum-specific security framework for QuantumRerank.

This module provides security measures specifically designed for quantum
computations including circuit validation, parameter integrity checking,
side-channel attack prevention, and quantum computation monitoring.
"""

import numpy as np
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict

from ..utils.logging_config import get_logger
from ..utils.exceptions import QuantumSecurityError, SecurityError

logger = get_logger(__name__)


class SecurityRiskLevel(Enum):
    """Security risk levels for quantum operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityValidationResult:
    """Result of quantum security validation."""
    secure: bool
    risk_level: SecurityRiskLevel
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    security_score: float = 1.0
    validation_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityMonitoringResult:
    """Result of quantum computation security monitoring."""
    secure: bool
    anomaly: Optional[str] = None
    threat_level: float = 0.0
    side_channel_risk: float = 0.0
    resource_anomalies: List[str] = field(default_factory=list)
    timing_anomalies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuantumCircuitValidator:
    """Validates quantum circuits for security vulnerabilities."""
    
    def __init__(self):
        """Initialize quantum circuit validator."""
        self.security_limits = {
            "max_circuit_depth": 100,
            "max_gate_count": 1000,
            "max_qubits": 50,
            "max_parameters": 500,
            "max_execution_time_s": 300
        }
        
        # Suspicious gate patterns
        self.suspicious_patterns = {
            "excessive_rotation_gates": 0.8,  # >80% rotation gates
            "deep_entanglement": 50,  # >50 consecutive entangling gates
            "parameter_concentration": 0.9,  # >90% parameters in small range
            "gate_repetition": 20  # >20 identical consecutive gates
        }
        
        self.logger = logger
        logger.info("Initialized QuantumCircuitValidator")
    
    def validate_circuit_structure(self, circuit_data: Dict[str, Any]) -> SecurityValidationResult:
        """
        Validate quantum circuit structure for security.
        
        Args:
            circuit_data: Circuit information including gates, qubits, parameters
            
        Returns:
            SecurityValidationResult with validation results
        """
        start_time = time.time()
        issues = []
        warnings = []
        metadata = {}
        
        try:
            # Extract circuit information
            num_qubits = circuit_data.get("num_qubits", 0)
            gates = circuit_data.get("gates", [])
            parameters = circuit_data.get("parameters", [])
            depth = circuit_data.get("depth", len(gates))
            
            metadata.update({
                "num_qubits": num_qubits,
                "num_gates": len(gates),
                "circuit_depth": depth,
                "num_parameters": len(parameters)
            })
            
            # Check basic limits
            if depth > self.security_limits["max_circuit_depth"]:
                issues.append(f"Circuit depth {depth} exceeds security limit {self.security_limits['max_circuit_depth']}")
            
            if len(gates) > self.security_limits["max_gate_count"]:
                issues.append(f"Gate count {len(gates)} exceeds security limit {self.security_limits['max_gate_count']}")
            
            if num_qubits > self.security_limits["max_qubits"]:
                issues.append(f"Qubit count {num_qubits} exceeds security limit {self.security_limits['max_qubits']}")
            
            if len(parameters) > self.security_limits["max_parameters"]:
                issues.append(f"Parameter count {len(parameters)} exceeds security limit {self.security_limits['max_parameters']}")
            
            # Analyze gate patterns for suspicious behavior
            if gates:
                pattern_analysis = self._analyze_gate_patterns(gates)
                metadata["pattern_analysis"] = pattern_analysis
                
                # Check for suspicious patterns
                if pattern_analysis["rotation_gate_ratio"] > self.suspicious_patterns["excessive_rotation_gates"]:
                    warnings.append(f"Excessive rotation gates: {pattern_analysis['rotation_gate_ratio']:.2f}")
                
                if pattern_analysis["max_consecutive_entangling"] > self.suspicious_patterns["deep_entanglement"]:
                    warnings.append(f"Deep entanglement pattern detected: {pattern_analysis['max_consecutive_entangling']} consecutive gates")
                
                if pattern_analysis["max_gate_repetition"] > self.suspicious_patterns["gate_repetition"]:
                    warnings.append(f"Excessive gate repetition: {pattern_analysis['max_gate_repetition']} consecutive identical gates")
            
            # Analyze parameters for anomalies
            if parameters:
                param_analysis = self._analyze_parameters(parameters)
                metadata["parameter_analysis"] = param_analysis
                
                if param_analysis["concentration_ratio"] > self.suspicious_patterns["parameter_concentration"]:
                    warnings.append(f"High parameter concentration: {param_analysis['concentration_ratio']:.2f}")
                
                if param_analysis["has_extreme_values"]:
                    warnings.append("Extreme parameter values detected")
            
            # Calculate security score and risk level
            security_score = self._calculate_security_score(issues, warnings)
            risk_level = self._assess_risk_level(issues, warnings, security_score)
            
            validation_time_ms = (time.time() - start_time) * 1000
            
            return SecurityValidationResult(
                secure=len(issues) == 0,
                risk_level=risk_level,
                issues=issues,
                warnings=warnings,
                security_score=security_score,
                validation_time_ms=validation_time_ms,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Circuit validation error: {e}")
            return SecurityValidationResult(
                secure=False,
                risk_level=SecurityRiskLevel.CRITICAL,
                issues=[f"Validation failed: {str(e)}"],
                validation_time_ms=(time.time() - start_time) * 1000
            )
    
    def _analyze_gate_patterns(self, gates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze gate patterns for suspicious behavior."""
        if not gates:
            return {}
        
        gate_types = [gate.get("type", "unknown") for gate in gates]
        gate_type_counts = defaultdict(int)
        
        for gate_type in gate_types:
            gate_type_counts[gate_type] += 1
        
        # Rotation gates (parameterized gates that could be used for side-channel attacks)
        rotation_gates = {"rx", "ry", "rz", "u1", "u2", "u3", "rxx", "ryy", "rzz"}
        rotation_count = sum(gate_type_counts[gate] for gate in rotation_gates)
        rotation_ratio = rotation_count / len(gates)
        
        # Entangling gates
        entangling_gates = {"cx", "cnot", "cz", "swap", "ccx", "toffoli"}
        entangling_indices = [i for i, gate in enumerate(gates) if gate.get("type") in entangling_gates]
        
        max_consecutive_entangling = 0
        current_consecutive = 0
        for i in range(len(gates)):
            if i in entangling_indices:
                current_consecutive += 1
                max_consecutive_entangling = max(max_consecutive_entangling, current_consecutive)
            else:
                current_consecutive = 0
        
        # Gate repetition analysis
        max_repetition = 1
        current_repetition = 1
        for i in range(1, len(gate_types)):
            if gate_types[i] == gate_types[i-1]:
                current_repetition += 1
                max_repetition = max(max_repetition, current_repetition)
            else:
                current_repetition = 1
        
        return {
            "rotation_gate_ratio": rotation_ratio,
            "max_consecutive_entangling": max_consecutive_entangling,
            "max_gate_repetition": max_repetition,
            "gate_type_distribution": dict(gate_type_counts),
            "total_gates": len(gates)
        }
    
    def _analyze_parameters(self, parameters: List[float]) -> Dict[str, Any]:
        """Analyze quantum parameters for security anomalies."""
        if not parameters:
            return {}
        
        params_array = np.array(parameters)
        
        # Check parameter distribution
        param_range = np.max(params_array) - np.min(params_array)
        param_std = np.std(params_array)
        param_mean = np.mean(params_array)
        
        # Check for concentration in small range
        sorted_params = np.sort(params_array)
        mid_90_percent_range = sorted_params[int(0.95 * len(parameters))] - sorted_params[int(0.05 * len(parameters))]
        concentration_ratio = 1 - (mid_90_percent_range / (2 * np.pi)) if param_range > 0 else 1
        
        # Check for extreme values
        has_extreme_values = np.any(np.abs(params_array) > 2 * np.pi)
        
        # Check for potential adversarial patterns
        param_entropy = self._calculate_parameter_entropy(params_array)
        
        return {
            "concentration_ratio": concentration_ratio,
            "has_extreme_values": has_extreme_values,
            "parameter_entropy": param_entropy,
            "std_deviation": param_std,
            "mean_value": param_mean,
            "value_range": param_range
        }
    
    def _calculate_parameter_entropy(self, parameters: np.ndarray) -> float:
        """Calculate entropy of parameter distribution."""
        try:
            # Discretize parameters for entropy calculation
            hist, _ = np.histogram(parameters, bins=20, range=(-np.pi, np.pi))
            hist = hist + 1e-10  # Avoid log(0)
            hist = hist / np.sum(hist)  # Normalize
            
            entropy = -np.sum(hist * np.log2(hist))
            return float(entropy)
        except:
            return 0.0
    
    def _calculate_security_score(self, issues: List[str], warnings: List[str]) -> float:
        """Calculate security score based on issues and warnings."""
        if issues:
            return 0.0
        
        score = 1.0
        score -= len(warnings) * 0.1  # Deduct 0.1 per warning
        
        return max(0.0, min(1.0, score))
    
    def _assess_risk_level(self, issues: List[str], warnings: List[str], 
                          security_score: float) -> SecurityRiskLevel:
        """Assess security risk level."""
        if issues:
            return SecurityRiskLevel.CRITICAL
        elif len(warnings) >= 3:
            return SecurityRiskLevel.HIGH
        elif len(warnings) >= 1:
            return SecurityRiskLevel.MEDIUM
        else:
            return SecurityRiskLevel.LOW


class ParameterIntegrityChecker:
    """Checks integrity of quantum parameters."""
    
    def __init__(self):
        """Initialize parameter integrity checker."""
        self.parameter_history: Dict[str, List[np.ndarray]] = {}
        self.integrity_checksums: Dict[str, str] = {}
        self.logger = logger
        
        logger.info("Initialized ParameterIntegrityChecker")
    
    def store_parameters(self, context_id: str, parameters: np.ndarray) -> str:
        """
        Store parameters with integrity checksum.
        
        Args:
            context_id: Unique context identifier
            parameters: Parameters to store
            
        Returns:
            Integrity checksum
        """
        # Calculate checksum
        param_bytes = parameters.tobytes()
        checksum = hashlib.sha256(param_bytes).hexdigest()
        
        # Store parameters and checksum
        if context_id not in self.parameter_history:
            self.parameter_history[context_id] = []
        
        self.parameter_history[context_id].append(parameters.copy())
        self.integrity_checksums[context_id] = checksum
        
        self.logger.debug(f"Stored parameters for context {context_id} with checksum {checksum[:8]}...")
        return checksum
    
    def verify_parameters(self, context_id: str, parameters: np.ndarray,
                         expected_checksum: Optional[str] = None) -> bool:
        """
        Verify parameter integrity.
        
        Args:
            context_id: Context identifier
            parameters: Parameters to verify
            expected_checksum: Expected checksum (uses stored if None)
            
        Returns:
            True if integrity is verified
        """
        try:
            # Calculate current checksum
            param_bytes = parameters.tobytes()
            current_checksum = hashlib.sha256(param_bytes).hexdigest()
            
            # Get expected checksum
            if expected_checksum is None:
                expected_checksum = self.integrity_checksums.get(context_id)
                if expected_checksum is None:
                    self.logger.warning(f"No stored checksum for context {context_id}")
                    return False
            
            # Verify integrity
            integrity_verified = current_checksum == expected_checksum
            
            if not integrity_verified:
                self.logger.warning(f"Parameter integrity failed for context {context_id}")
                self.logger.warning(f"Expected: {expected_checksum[:8]}..., Got: {current_checksum[:8]}...")
            
            return integrity_verified
            
        except Exception as e:
            self.logger.error(f"Parameter integrity verification error: {e}")
            return False
    
    def detect_parameter_tampering(self, context_id: str, 
                                 parameters: np.ndarray) -> Dict[str, Any]:
        """
        Detect potential parameter tampering.
        
        Args:
            context_id: Context identifier
            parameters: Current parameters
            
        Returns:
            Dictionary with tampering analysis
        """
        if context_id not in self.parameter_history:
            return {"tampering_detected": False, "reason": "No history available"}
        
        history = self.parameter_history[context_id]
        if not history:
            return {"tampering_detected": False, "reason": "Empty history"}
        
        latest_stored = history[-1]
        
        # Check for exact match
        if np.array_equal(parameters, latest_stored):
            return {"tampering_detected": False, "reason": "Parameters match exactly"}
        
        # Analyze differences
        diff = np.abs(parameters - latest_stored)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # Check for suspicious modifications
        tampering_indicators = {
            "large_single_change": max_diff > np.pi / 2,  # Large single parameter change
            "systematic_shift": mean_diff > 0.1,  # Systematic parameter shift
            "sign_flips": np.sum(np.sign(parameters) != np.sign(latest_stored)) > len(parameters) * 0.5,
            "extreme_values": np.any(np.abs(parameters) > 3 * np.pi)
        }
        
        tampering_detected = any(tampering_indicators.values())
        
        return {
            "tampering_detected": tampering_detected,
            "indicators": tampering_indicators,
            "max_difference": float(max_diff),
            "mean_difference": float(mean_diff),
            "parameter_count": len(parameters)
        }


class QuantumComputationMonitor:
    """Monitors quantum computations for security anomalies."""
    
    def __init__(self):
        """Initialize quantum computation monitor."""
        self.computation_history: List[Dict[str, Any]] = []
        self.baseline_metrics: Dict[str, float] = {}
        self.anomaly_thresholds = {
            "execution_time_multiplier": 3.0,  # 3x normal execution time
            "memory_usage_multiplier": 2.0,    # 2x normal memory usage
            "error_rate_threshold": 0.1,       # 10% error rate
            "side_channel_threshold": 0.5      # Side-channel risk threshold
        }
        
        self.logger = logger
        self._monitoring_lock = threading.Lock()
        
        logger.info("Initialized QuantumComputationMonitor")
    
    def start_monitoring(self, computation_context: Dict[str, Any]) -> str:
        """
        Start monitoring quantum computation.
        
        Args:
            computation_context: Computation context information
            
        Returns:
            Monitoring session ID
        """
        session_id = f"monitor_{int(time.time() * 1000000)}"
        
        monitoring_data = {
            "session_id": session_id,
            "start_time": time.time(),
            "context": computation_context.copy(),
            "metrics": {},
            "anomalies": []
        }
        
        with self._monitoring_lock:
            self.computation_history.append(monitoring_data)
        
        self.logger.debug(f"Started monitoring session {session_id}")
        return session_id
    
    def record_metric(self, session_id: str, metric_name: str, value: float) -> None:
        """
        Record computation metric.
        
        Args:
            session_id: Monitoring session ID
            metric_name: Name of metric
            value: Metric value
        """
        with self._monitoring_lock:
            for session in self.computation_history:
                if session["session_id"] == session_id:
                    session["metrics"][metric_name] = value
                    break
    
    def end_monitoring(self, session_id: str) -> SecurityMonitoringResult:
        """
        End monitoring and analyze results.
        
        Args:
            session_id: Monitoring session ID
            
        Returns:
            SecurityMonitoringResult with analysis
        """
        with self._monitoring_lock:
            session_data = None
            for session in self.computation_history:
                if session["session_id"] == session_id:
                    session_data = session
                    break
        
        if session_data is None:
            return SecurityMonitoringResult(
                secure=False,
                anomaly="Monitoring session not found"
            )
        
        # Calculate execution time
        execution_time = time.time() - session_data["start_time"]
        session_data["execution_time"] = execution_time
        
        # Analyze for anomalies
        anomalies = self._analyze_computation_anomalies(session_data)
        
        # Calculate threat level
        threat_level = self._calculate_threat_level(anomalies)
        
        # Assess side-channel risk
        side_channel_risk = self._assess_side_channel_risk(session_data)
        
        # Determine if computation is secure
        secure = (len(anomalies) == 0 and 
                 threat_level < 0.5 and 
                 side_channel_risk < self.anomaly_thresholds["side_channel_threshold"])
        
        self.logger.debug(f"Ended monitoring session {session_id}: secure={secure}")
        
        return SecurityMonitoringResult(
            secure=secure,
            anomaly=anomalies[0] if anomalies else None,
            threat_level=threat_level,
            side_channel_risk=side_channel_risk,
            resource_anomalies=anomalies,
            metadata={
                "session_id": session_id,
                "execution_time": execution_time,
                "total_anomalies": len(anomalies)
            }
        )
    
    def _analyze_computation_anomalies(self, session_data: Dict[str, Any]) -> List[str]:
        """Analyze computation for anomalies."""
        anomalies = []
        metrics = session_data["metrics"]
        execution_time = session_data["execution_time"]
        
        # Check execution time anomaly
        if "expected_execution_time" in session_data["context"]:
            expected_time = session_data["context"]["expected_execution_time"]
            if execution_time > expected_time * self.anomaly_thresholds["execution_time_multiplier"]:
                anomalies.append(f"Excessive execution time: {execution_time:.2f}s vs expected {expected_time:.2f}s")
        
        # Check memory usage anomaly
        if "memory_usage_mb" in metrics:
            memory_usage = metrics["memory_usage_mb"]
            expected_memory = session_data["context"].get("expected_memory_mb", 100)
            if memory_usage > expected_memory * self.anomaly_thresholds["memory_usage_multiplier"]:
                anomalies.append(f"Excessive memory usage: {memory_usage:.1f}MB vs expected {expected_memory:.1f}MB")
        
        # Check error rate
        if "error_rate" in metrics:
            error_rate = metrics["error_rate"]
            if error_rate > self.anomaly_thresholds["error_rate_threshold"]:
                anomalies.append(f"High error rate: {error_rate:.2f}")
        
        # Check for unusual patterns in metrics
        for metric_name, value in metrics.items():
            if metric_name in self.baseline_metrics:
                baseline = self.baseline_metrics[metric_name]
                if abs(value - baseline) > baseline * 2:  # >200% deviation
                    anomalies.append(f"Unusual {metric_name}: {value:.2f} vs baseline {baseline:.2f}")
        
        return anomalies
    
    def _calculate_threat_level(self, anomalies: List[str]) -> float:
        """Calculate threat level based on anomalies."""
        if not anomalies:
            return 0.0
        
        # Simple threat level calculation
        threat_level = min(1.0, len(anomalies) * 0.3)
        return threat_level
    
    def _assess_side_channel_risk(self, session_data: Dict[str, Any]) -> float:
        """Assess side-channel attack risk."""
        risk_factors = 0.0
        
        # Timing-based side-channel risk
        execution_time = session_data["execution_time"]
        if execution_time > 10:  # Long execution increases timing attack risk
            risk_factors += 0.3
        
        # Resource usage side-channel risk
        metrics = session_data["metrics"]
        if "memory_usage_mb" in metrics and metrics["memory_usage_mb"] > 1000:
            risk_factors += 0.2
        
        # Parameter-based side-channel risk
        context = session_data["context"]
        if "parameters" in context:
            params = context["parameters"]
            if isinstance(params, (list, np.ndarray)) and len(params) > 100:
                risk_factors += 0.2
        
        return min(1.0, risk_factors)


class QuantumSecurityFramework:
    """
    Comprehensive quantum security framework.
    
    Integrates circuit validation, parameter integrity, and computation monitoring
    for complete quantum computation security.
    """
    
    def __init__(self):
        """Initialize quantum security framework."""
        self.circuit_validator = QuantumCircuitValidator()
        self.parameter_integrity = ParameterIntegrityChecker()
        self.computation_monitor = QuantumComputationMonitor()
        
        self.logger = logger
        logger.info("Initialized QuantumSecurityFramework")
    
    def validate_quantum_circuit(self, circuit_data: Dict[str, Any]) -> SecurityValidationResult:
        """Validate quantum circuit for security."""
        return self.circuit_validator.validate_circuit_structure(circuit_data)
    
    def secure_quantum_computation(self, circuit_data: Dict[str, Any],
                                 parameters: np.ndarray,
                                 context: Optional[Dict[str, Any]] = None) -> str:
        """
        Secure quantum computation with comprehensive monitoring.
        
        Args:
            circuit_data: Quantum circuit information
            parameters: Quantum parameters
            context: Optional computation context
            
        Returns:
            Monitoring session ID
        """
        # Validate circuit
        circuit_validation = self.validate_quantum_circuit(circuit_data)
        if not circuit_validation.secure:
            raise QuantumSecurityError(f"Circuit validation failed: {circuit_validation.issues}")
        
        # Store parameter integrity
        context_id = context.get("context_id", f"comp_{int(time.time()*1000)}") if context else f"comp_{int(time.time()*1000)}"
        checksum = self.parameter_integrity.store_parameters(context_id, parameters)
        
        # Start monitoring
        monitoring_context = {
            **(context or {}),
            "context_id": context_id,
            "parameter_checksum": checksum,
            "circuit_info": circuit_data
        }
        
        session_id = self.computation_monitor.start_monitoring(monitoring_context)
        
        self.logger.info(f"Secured quantum computation with session {session_id}")
        return session_id
    
    def monitor_quantum_computation(self, session_id: str, 
                                  computation_context: Dict[str, Any]) -> SecurityMonitoringResult:
        """Monitor quantum computation for security anomalies."""
        # Record any metrics from the computation context
        for metric_name, value in computation_context.items():
            if isinstance(value, (int, float)):
                self.computation_monitor.record_metric(session_id, metric_name, value)
        
        # End monitoring and return results
        return self.computation_monitor.end_monitoring(session_id)
    
    def verify_computation_integrity(self, context_id: str, 
                                   parameters: np.ndarray) -> bool:
        """Verify integrity of quantum computation parameters."""
        return self.parameter_integrity.verify_parameters(context_id, parameters)
    
    def detect_security_threats(self, computation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive security threat detection.
        
        Args:
            computation_data: Complete computation data
            
        Returns:
            Threat detection results
        """
        threats = {
            "circuit_threats": [],
            "parameter_threats": [],
            "execution_threats": [],
            "overall_threat_level": 0.0
        }
        
        # Circuit-level threat detection
        if "circuit" in computation_data:
            circuit_validation = self.validate_quantum_circuit(computation_data["circuit"])
            if not circuit_validation.secure:
                threats["circuit_threats"] = circuit_validation.issues
        
        # Parameter-level threat detection
        if "parameters" in computation_data and "context_id" in computation_data:
            tampering_analysis = self.parameter_integrity.detect_parameter_tampering(
                computation_data["context_id"],
                computation_data["parameters"]
            )
            if tampering_analysis["tampering_detected"]:
                threats["parameter_threats"].append("Parameter tampering detected")
        
        # Execution-level threat detection
        if "monitoring_session" in computation_data:
            monitoring_result = self.computation_monitor.end_monitoring(
                computation_data["monitoring_session"]
            )
            if not monitoring_result.secure:
                threats["execution_threats"].append(monitoring_result.anomaly)
            threats["overall_threat_level"] = monitoring_result.threat_level
        
        return threats
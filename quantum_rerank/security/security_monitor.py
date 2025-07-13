"""
Security Monitoring and Threat Detection System for QuantumRerank.

This module provides real-time security monitoring, threat detection,
anomaly detection, audit logging, and compliance monitoring to ensure
comprehensive security visibility and incident response.
"""

import time
import json
import hashlib
import threading
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta
import numpy as np

from ..utils.logging_config import get_logger
from ..utils.exceptions import SecurityError

logger = get_logger(__name__)


class ThreatType(Enum):
    """Types of security threats."""
    BRUTE_FORCE = "brute_force"
    DDOS = "ddos"
    INJECTION = "injection"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    MALICIOUS_INPUT = "malicious_input"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    QUANTUM_ATTACK = "quantum_attack"


class ThreatSeverity(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditEventType(Enum):
    """Types of audit events."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_VIOLATION = "security_violation"
    SYSTEM_EVENT = "system_event"
    QUANTUM_OPERATION = "quantum_operation"


@dataclass
class SecurityEvent:
    """Security event information."""
    event_id: str
    event_type: str
    timestamp: float
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatDetection:
    """Threat detection result."""
    threat_id: str
    threat_type: ThreatType
    severity: ThreatSeverity
    confidence: float
    timestamp: float
    source_events: List[SecurityEvent] = field(default_factory=list)
    indicators: Dict[str, Any] = field(default_factory=dict)
    recommended_actions: List[str] = field(default_factory=list)


@dataclass
class AnomalyDetection:
    """Anomaly detection result."""
    anomaly_id: str
    anomaly_type: str
    score: float
    timestamp: float
    baseline_value: float
    observed_value: float
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditEvent:
    """Audit log event."""
    event_id: str
    event_type: AuditEventType
    timestamp: float
    user_id: Optional[str] = None
    resource: Optional[str] = None
    action: str
    result: str  # success, failure, error
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class ThreatDetector:
    """Real-time threat detection engine."""
    
    def __init__(self):
        """Initialize threat detector."""
        self.event_buffer: deque = deque(maxlen=10000)
        self.threat_patterns = self._initialize_threat_patterns()
        self.detection_rules = self._initialize_detection_rules()
        self.lock = threading.Lock()
        
        self.logger = logger
        logger.info("Initialized ThreatDetector")
    
    def _initialize_threat_patterns(self) -> Dict[ThreatType, Dict[str, Any]]:
        """Initialize threat detection patterns."""
        return {
            ThreatType.BRUTE_FORCE: {
                "failed_login_threshold": 5,
                "time_window_seconds": 300,
                "detection_window": 60
            },
            ThreatType.DDOS: {
                "request_rate_threshold": 1000,
                "time_window_seconds": 60,
                "unique_ip_threshold": 100
            },
            ThreatType.INJECTION: {
                "pattern_matches_threshold": 3,
                "suspicious_patterns": [
                    "union select", "' or ", "'; drop", "<script>",
                    "javascript:", "eval(", "system(", "../"
                ]
            },
            ThreatType.DATA_EXFILTRATION: {
                "large_response_threshold": 10 * 1024 * 1024,  # 10MB
                "rapid_requests_threshold": 100,
                "time_window_seconds": 300
            },
            ThreatType.QUANTUM_ATTACK: {
                "circuit_complexity_threshold": 1000,
                "parameter_anomaly_threshold": 0.8,
                "computation_time_threshold": 300
            }
        }
    
    def _initialize_detection_rules(self) -> List[Callable]:
        """Initialize threat detection rules."""
        return [
            self._detect_brute_force,
            self._detect_ddos,
            self._detect_injection_attacks,
            self._detect_data_exfiltration,
            self._detect_quantum_attacks
        ]
    
    def add_event(self, event: SecurityEvent) -> None:
        """Add security event to detection pipeline."""
        with self.lock:
            self.event_buffer.append(event)
    
    def detect_threats(self) -> List[ThreatDetection]:
        """Detect threats from recent events."""
        threats = []
        
        with self.lock:
            recent_events = list(self.event_buffer)
        
        # Apply detection rules
        for rule in self.detection_rules:
            try:
                detected_threats = rule(recent_events)
                threats.extend(detected_threats)
            except Exception as e:
                self.logger.error(f"Threat detection rule failed: {e}")
        
        return threats
    
    def _detect_brute_force(self, events: List[SecurityEvent]) -> List[ThreatDetection]:
        """Detect brute force attacks."""
        threats = []
        current_time = time.time()
        pattern = self.threat_patterns[ThreatType.BRUTE_FORCE]
        
        # Group failed login events by IP
        failed_logins = defaultdict(list)
        
        for event in events:
            if (event.event_type == "authentication_failure" and 
                current_time - event.timestamp < pattern["time_window_seconds"]):
                failed_logins[event.source_ip].append(event)
        
        # Check for brute force patterns
        for ip, login_events in failed_logins.items():
            if len(login_events) >= pattern["failed_login_threshold"]:
                threats.append(ThreatDetection(
                    threat_id=f"bf_{hashlib.md5(f'{ip}_{current_time}'.encode()).hexdigest()[:8]}",
                    threat_type=ThreatType.BRUTE_FORCE,
                    severity=ThreatSeverity.HIGH,
                    confidence=min(1.0, len(login_events) / pattern["failed_login_threshold"]),
                    timestamp=current_time,
                    source_events=login_events,
                    indicators={
                        "source_ip": ip,
                        "failed_attempts": len(login_events),
                        "time_span": max(e.timestamp for e in login_events) - min(e.timestamp for e in login_events)
                    },
                    recommended_actions=[
                        f"Block IP address {ip}",
                        "Enable additional authentication factors",
                        "Monitor for continued attempts"
                    ]
                ))
        
        return threats
    
    def _detect_ddos(self, events: List[SecurityEvent]) -> List[ThreatDetection]:
        """Detect DDoS attacks."""
        threats = []
        current_time = time.time()
        pattern = self.threat_patterns[ThreatType.DDOS]
        
        # Analyze request patterns
        recent_requests = [
            e for e in events 
            if e.event_type == "api_request" and 
            current_time - e.timestamp < pattern["time_window_seconds"]
        ]
        
        if len(recent_requests) >= pattern["request_rate_threshold"]:
            unique_ips = set(e.source_ip for e in recent_requests if e.source_ip)
            
            # High request rate with low IP diversity indicates DDoS
            if len(unique_ips) < pattern["unique_ip_threshold"]:
                threats.append(ThreatDetection(
                    threat_id=f"ddos_{hashlib.md5(f'{current_time}'.encode()).hexdigest()[:8]}",
                    threat_type=ThreatType.DDOS,
                    severity=ThreatSeverity.CRITICAL,
                    confidence=min(1.0, len(recent_requests) / pattern["request_rate_threshold"]),
                    timestamp=current_time,
                    source_events=recent_requests[-100:],  # Sample of recent events
                    indicators={
                        "request_rate": len(recent_requests),
                        "unique_ips": len(unique_ips),
                        "top_source_ips": [ip for ip, _ in 
                                         sorted([(ip, len([e for e in recent_requests if e.source_ip == ip])) 
                                               for ip in unique_ips], key=lambda x: x[1], reverse=True)[:5]]
                    },
                    recommended_actions=[
                        "Enable DDoS protection",
                        "Implement rate limiting",
                        "Block suspicious IP ranges",
                        "Scale infrastructure if needed"
                    ]
                ))
        
        return threats
    
    def _detect_injection_attacks(self, events: List[SecurityEvent]) -> List[ThreatDetection]:
        """Detect injection attacks."""
        threats = []
        current_time = time.time()
        pattern = self.threat_patterns[ThreatType.INJECTION]
        
        # Look for injection patterns in requests
        injection_events = []
        
        for event in events:
            if event.event_type in ["api_request", "query_execution"]:
                content = str(event.details.get("content", "")).lower()
                matches = sum(1 for p in pattern["suspicious_patterns"] if p in content)
                
                if matches >= pattern["pattern_matches_threshold"]:
                    injection_events.append(event)
        
        if injection_events:
            # Group by source IP for threat assessment
            by_ip = defaultdict(list)
            for event in injection_events:
                by_ip[event.source_ip].append(event)
            
            for ip, ip_events in by_ip.items():
                threats.append(ThreatDetection(
                    threat_id=f"inj_{hashlib.md5(f'{ip}_{current_time}'.encode()).hexdigest()[:8]}",
                    threat_type=ThreatType.INJECTION,
                    severity=ThreatSeverity.HIGH,
                    confidence=min(1.0, len(ip_events) / 3),
                    timestamp=current_time,
                    source_events=ip_events,
                    indicators={
                        "source_ip": ip,
                        "injection_attempts": len(ip_events),
                        "patterns_detected": list(set(
                            p for event in ip_events 
                            for p in pattern["suspicious_patterns"]
                            if p in str(event.details.get("content", "")).lower()
                        ))
                    },
                    recommended_actions=[
                        f"Block IP address {ip}",
                        "Review input validation",
                        "Check for data compromise",
                        "Update security filters"
                    ]
                ))
        
        return threats
    
    def _detect_data_exfiltration(self, events: List[SecurityEvent]) -> List[ThreatDetection]:
        """Detect data exfiltration attempts."""
        threats = []
        current_time = time.time()
        pattern = self.threat_patterns[ThreatType.DATA_EXFILTRATION]
        
        # Look for large data transfers or rapid requests
        data_events = [
            e for e in events 
            if e.event_type == "data_access" and 
            current_time - e.timestamp < pattern["time_window_seconds"]
        ]
        
        # Group by user/IP
        by_actor = defaultdict(list)
        for event in data_events:
            actor = event.user_id or event.source_ip
            by_actor[actor].append(event)
        
        for actor, actor_events in by_actor.items():
            # Check for rapid access patterns
            if len(actor_events) >= pattern["rapid_requests_threshold"]:
                total_data = sum(event.details.get("response_size", 0) for event in actor_events)
                
                if total_data >= pattern["large_response_threshold"]:
                    threats.append(ThreatDetection(
                        threat_id=f"exfil_{hashlib.md5(f'{actor}_{current_time}'.encode()).hexdigest()[:8]}",
                        threat_type=ThreatType.DATA_EXFILTRATION,
                        severity=ThreatSeverity.HIGH,
                        confidence=min(1.0, total_data / pattern["large_response_threshold"]),
                        timestamp=current_time,
                        source_events=actor_events,
                        indicators={
                            "actor": actor,
                            "total_data_bytes": total_data,
                            "request_count": len(actor_events),
                            "time_span": max(e.timestamp for e in actor_events) - min(e.timestamp for e in actor_events)
                        },
                        recommended_actions=[
                            f"Investigate actor {actor}",
                            "Review data access logs",
                            "Implement data loss prevention",
                            "Monitor for continued activity"
                        ]
                    ))
        
        return threats
    
    def _detect_quantum_attacks(self, events: List[SecurityEvent]) -> List[ThreatDetection]:
        """Detect quantum-specific attacks."""
        threats = []
        current_time = time.time()
        pattern = self.threat_patterns[ThreatType.QUANTUM_ATTACK]
        
        # Look for quantum computation anomalies
        quantum_events = [
            e for e in events 
            if e.event_type == "quantum_computation"
        ]
        
        for event in quantum_events:
            details = event.details
            suspicious_indicators = []
            
            # Check circuit complexity
            if details.get("circuit_gates", 0) > pattern["circuit_complexity_threshold"]:
                suspicious_indicators.append("excessive_circuit_complexity")
            
            # Check computation time
            if details.get("computation_time", 0) > pattern["computation_time_threshold"]:
                suspicious_indicators.append("excessive_computation_time")
            
            # Check parameter anomalies
            if details.get("parameter_anomaly_score", 0) > pattern["parameter_anomaly_threshold"]:
                suspicious_indicators.append("parameter_anomaly")
            
            if suspicious_indicators:
                threats.append(ThreatDetection(
                    threat_id=f"qattack_{hashlib.md5(f'{event.event_id}_{current_time}'.encode()).hexdigest()[:8]}",
                    threat_type=ThreatType.QUANTUM_ATTACK,
                    severity=ThreatSeverity.MEDIUM,
                    confidence=len(suspicious_indicators) / 3,
                    timestamp=current_time,
                    source_events=[event],
                    indicators={
                        "quantum_indicators": suspicious_indicators,
                        "circuit_complexity": details.get("circuit_gates", 0),
                        "computation_time": details.get("computation_time", 0),
                        "parameter_anomaly": details.get("parameter_anomaly_score", 0)
                    },
                    recommended_actions=[
                        "Review quantum computation parameters",
                        "Validate circuit authenticity",
                        "Monitor quantum resource usage",
                        "Check for side-channel attacks"
                    ]
                ))
        
        return threats


class AnomalyDetector:
    """Statistical anomaly detection for security monitoring."""
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize anomaly detector.
        
        Args:
            window_size: Size of sliding window for baseline calculation
        """
        self.window_size = window_size
        self.metric_baselines: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.lock = threading.Lock()
        
        self.logger = logger
        logger.info(f"Initialized AnomalyDetector with window size {window_size}")
    
    def update_metric(self, metric_name: str, value: float) -> None:
        """Update metric baseline."""
        with self.lock:
            self.metric_baselines[metric_name].append(value)
    
    def detect_anomaly(self, metric_name: str, current_value: float,
                      sensitivity: float = 2.0) -> Optional[AnomalyDetection]:
        """
        Detect anomalies using statistical methods.
        
        Args:
            metric_name: Name of metric to check
            current_value: Current metric value
            sensitivity: Number of standard deviations for anomaly threshold
            
        Returns:
            AnomalyDetection if anomaly detected, None otherwise
        """
        with self.lock:
            baseline_values = list(self.metric_baselines[metric_name])
        
        if len(baseline_values) < 10:  # Need minimum data for baseline
            return None
        
        baseline_array = np.array(baseline_values)
        baseline_mean = np.mean(baseline_array)
        baseline_std = np.std(baseline_array)
        
        if baseline_std == 0:  # Constant values
            return None
        
        # Calculate z-score
        z_score = abs((current_value - baseline_mean) / baseline_std)
        
        if z_score > sensitivity:
            anomaly_id = f"anomaly_{hashlib.md5(f'{metric_name}_{time.time()}'.encode()).hexdigest()[:8]}"
            
            return AnomalyDetection(
                anomaly_id=anomaly_id,
                anomaly_type="statistical_outlier",
                score=z_score,
                timestamp=time.time(),
                baseline_value=baseline_mean,
                observed_value=current_value,
                description=f"{metric_name} value {current_value:.2f} deviates {z_score:.2f}σ from baseline {baseline_mean:.2f}",
                metadata={
                    "metric_name": metric_name,
                    "z_score": z_score,
                    "baseline_std": baseline_std,
                    "sensitivity_threshold": sensitivity
                }
            )
        
        return None


class SecurityAuditLogger:
    """Comprehensive security audit logging system."""
    
    def __init__(self, log_retention_days: int = 90):
        """
        Initialize security audit logger.
        
        Args:
            log_retention_days: Number of days to retain audit logs
        """
        self.log_retention_days = log_retention_days
        self.audit_buffer: deque = deque(maxlen=100000)
        self.lock = threading.Lock()
        
        self.logger = logger
        logger.info(f"Initialized SecurityAuditLogger with {log_retention_days} day retention")
    
    def log_event(self, event: AuditEvent) -> None:
        """Log audit event."""
        with self.lock:
            self.audit_buffer.append(event)
        
        # Log to standard logger as well
        self.logger.info(f"AUDIT: {event.event_type.value} - {event.action} by {event.user_id} - {event.result}")
    
    def query_events(self, start_time: Optional[float] = None,
                    end_time: Optional[float] = None,
                    event_type: Optional[AuditEventType] = None,
                    user_id: Optional[str] = None) -> List[AuditEvent]:
        """Query audit events with filters."""
        with self.lock:
            events = list(self.audit_buffer)
        
        filtered_events = []
        
        for event in events:
            # Apply filters
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            if event_type and event.event_type != event_type:
                continue
            if user_id and event.user_id != user_id:
                continue
            
            filtered_events.append(event)
        
        return filtered_events
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit log statistics."""
        with self.lock:
            events = list(self.audit_buffer)
        
        if not events:
            return {"total_events": 0}
        
        # Calculate statistics
        event_types = defaultdict(int)
        users = defaultdict(int)
        results = defaultdict(int)
        
        for event in events:
            event_types[event.event_type.value] += 1
            if event.user_id:
                users[event.user_id] += 1
            results[event.result] += 1
        
        return {
            "total_events": len(events),
            "event_types": dict(event_types),
            "active_users": len(users),
            "results": dict(results),
            "time_range": {
                "earliest": min(e.timestamp for e in events),
                "latest": max(e.timestamp for e in events)
            }
        }


class ComplianceMonitor:
    """Monitor compliance with security policies and regulations."""
    
    def __init__(self):
        """Initialize compliance monitor."""
        self.compliance_rules = self._initialize_compliance_rules()
        self.violations: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        
        self.logger = logger
        logger.info("Initialized ComplianceMonitor")
    
    def _initialize_compliance_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize compliance rules."""
        return {
            "data_retention": {
                "max_retention_days": 2555,  # 7 years
                "min_retention_days": 30
            },
            "access_logging": {
                "required_events": [
                    AuditEventType.AUTHENTICATION,
                    AuditEventType.AUTHORIZATION,
                    AuditEventType.DATA_ACCESS
                ],
                "log_completeness_threshold": 0.95
            },
            "authentication": {
                "max_failed_attempts": 5,
                "password_policy": {
                    "min_length": 8,
                    "require_special_chars": True
                }
            },
            "encryption": {
                "data_at_rest": True,
                "data_in_transit": True,
                "key_rotation_days": 90
            }
        }
    
    def check_compliance(self, audit_events: List[AuditEvent]) -> Dict[str, Any]:
        """Check compliance against defined rules."""
        compliance_results = {}
        violations = []
        
        # Check data retention compliance
        current_time = time.time()
        retention_days = self.compliance_rules["data_retention"]["max_retention_days"]
        cutoff_time = current_time - (retention_days * 24 * 3600)
        
        old_events = [e for e in audit_events if e.timestamp < cutoff_time]
        if old_events:
            violations.append({
                "rule": "data_retention",
                "description": f"Found {len(old_events)} events older than retention policy",
                "severity": "medium"
            })
        
        # Check access logging completeness
        required_events = self.compliance_rules["access_logging"]["required_events"]
        event_coverage = {}
        
        for event_type in required_events:
            events_of_type = [e for e in audit_events if e.event_type == event_type]
            event_coverage[event_type.value] = len(events_of_type)
        
        # Check authentication compliance
        auth_events = [e for e in audit_events if e.event_type == AuditEventType.AUTHENTICATION]
        failed_auth_by_user = defaultdict(int)
        
        for event in auth_events:
            if event.result == "failure":
                failed_auth_by_user[event.user_id] += 1
        
        max_failures = self.compliance_rules["authentication"]["max_failed_attempts"]
        for user, failures in failed_auth_by_user.items():
            if failures > max_failures:
                violations.append({
                    "rule": "authentication",
                    "description": f"User {user} exceeded max failed attempts ({failures} > {max_failures})",
                    "severity": "high"
                })
        
        compliance_results = {
            "overall_compliant": len(violations) == 0,
            "violations": violations,
            "event_coverage": event_coverage,
            "compliance_score": max(0.0, 1.0 - len(violations) * 0.1)
        }
        
        with self.lock:
            self.violations.extend(violations)
        
        return compliance_results


class SecurityMonitoringSystem:
    """
    Comprehensive security monitoring system.
    
    Integrates threat detection, anomaly detection, audit logging,
    and compliance monitoring for complete security visibility.
    """
    
    def __init__(self):
        """Initialize security monitoring system."""
        self.threat_detector = ThreatDetector()
        self.anomaly_detector = AnomalyDetector()
        self.audit_logger = SecurityAuditLogger()
        self.compliance_monitor = ComplianceMonitor()
        
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        self.logger = logger
        logger.info("Initialized SecurityMonitoringSystem")
    
    def start_monitoring(self) -> None:
        """Start real-time security monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Started security monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop security monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Stopped security monitoring")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Detect threats
                threats = self.threat_detector.detect_threats()
                
                # Log detected threats
                for threat in threats:
                    self.logger.warning(f"THREAT DETECTED: {threat.threat_type.value} - Severity: {threat.severity.value}")
                    
                    # Log as audit event
                    audit_event = AuditEvent(
                        event_id=threat.threat_id,
                        event_type=AuditEventType.SECURITY_VIOLATION,
                        timestamp=threat.timestamp,
                        action="threat_detected",
                        result="detected",
                        details={
                            "threat_type": threat.threat_type.value,
                            "severity": threat.severity.value,
                            "confidence": threat.confidence,
                            "indicators": threat.indicators
                        }
                    )
                    self.audit_logger.log_event(audit_event)
                
                # Sleep before next iteration
                time.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(30)  # Longer sleep on error
    
    def add_security_event(self, event: SecurityEvent) -> None:
        """Add security event for monitoring."""
        self.threat_detector.add_event(event)
    
    def log_audit_event(self, event: AuditEvent) -> None:
        """Log audit event."""
        self.audit_logger.log_event(event)
    
    def update_metric(self, metric_name: str, value: float) -> Optional[AnomalyDetection]:
        """Update metric and check for anomalies."""
        anomaly = self.anomaly_detector.detect_anomaly(metric_name, value)
        self.anomaly_detector.update_metric(metric_name, value)
        
        if anomaly:
            self.logger.warning(f"ANOMALY DETECTED: {anomaly.description}")
            
            # Log as audit event
            audit_event = AuditEvent(
                event_id=anomaly.anomaly_id,
                event_type=AuditEventType.SYSTEM_EVENT,
                timestamp=anomaly.timestamp,
                action="anomaly_detected",
                result="detected",
                details={
                    "anomaly_type": anomaly.anomaly_type,
                    "score": anomaly.score,
                    "metric_name": metric_name,
                    "observed_value": anomaly.observed_value,
                    "baseline_value": anomaly.baseline_value
                }
            )
            self.audit_logger.log_event(audit_event)
        
        return anomaly
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard data."""
        audit_events = self.audit_logger.query_events(
            start_time=time.time() - 24 * 3600  # Last 24 hours
        )
        
        threats = self.threat_detector.detect_threats()
        compliance_status = self.compliance_monitor.check_compliance(audit_events)
        audit_stats = self.audit_logger.get_audit_statistics()
        
        return {
            "timestamp": time.time(),
            "monitoring_active": self.monitoring_active,
            "threats": {
                "total": len(threats),
                "by_severity": {
                    severity.value: len([t for t in threats if t.severity == severity])
                    for severity in ThreatSeverity
                },
                "recent_threats": threats[-10:]  # Last 10 threats
            },
            "compliance": compliance_status,
            "audit_statistics": audit_stats,
            "system_health": {
                "monitoring_status": "active" if self.monitoring_active else "inactive",
                "event_buffer_size": len(self.threat_detector.event_buffer),
                "audit_buffer_size": len(self.audit_logger.audit_buffer)
            }
        }
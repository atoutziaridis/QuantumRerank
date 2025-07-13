"""
Unified Security Manager for QuantumRerank.

This module provides a centralized security management interface that integrates
all security components including authentication, validation, monitoring,
incident response, and quantum-specific security measures.
"""

import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import threading

from ..utils.logging_config import get_logger
from ..utils.exceptions import SecurityError, AuthenticationError, ValidationError

# Import security components
from .input_validator import SecurityInputValidator, ValidationResult, ValidationLevel
from .auth_framework import QuantumRerankAuthFramework, AuthenticationResult, AuthorizationResult, Permission
from .quantum_security import QuantumSecurityFramework, SecurityValidationResult
from .api_security import APISecurityFramework, SecureRequestResult, APIRequest
from .security_monitor import SecurityMonitoringSystem, SecurityEvent, AuditEvent, AuditEventType
from .incident_response import SecurityIncidentResponse, NotificationConfig

logger = get_logger(__name__)


class SecurityMode(Enum):
    """Security operation modes."""
    PERMISSIVE = "permissive"  # Minimal security, log warnings
    STANDARD = "standard"     # Standard security enforcement
    STRICT = "strict"         # Strict security enforcement
    PARANOID = "paranoid"     # Maximum security, block on any suspicion


@dataclass
class SecurityConfiguration:
    """Comprehensive security configuration."""
    mode: SecurityMode = SecurityMode.STANDARD
    validation_level: ValidationLevel = ValidationLevel.STRICT
    enable_monitoring: bool = True
    enable_incident_response: bool = True
    enable_audit_logging: bool = True
    auto_block_threats: bool = True
    session_timeout_minutes: int = 60
    token_expiry_hours: int = 24
    max_failed_auth_attempts: int = 5
    rate_limit_enabled: bool = True
    quantum_security_enabled: bool = True
    notification_configs: List[NotificationConfig] = field(default_factory=list)


@dataclass
class SecuredRequest:
    """Secured request container with validation results."""
    original_request: Any
    validated: bool
    sanitized_request: Optional[Any] = None
    security_score: float = 0.0
    validation_errors: List[str] = field(default_factory=list)
    auth_result: Optional[AuthenticationResult] = None
    authz_result: Optional[AuthorizationResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecuredQuantumResult:
    """Secured quantum computation result."""
    computation_result: Any
    security_validated: bool
    quantum_security_score: float = 0.0
    monitoring_session_id: Optional[str] = None
    security_warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuantumRerankSecurityManager:
    """
    Centralized security management for QuantumRerank.
    
    Provides unified interface for all security operations including
    authentication, authorization, validation, monitoring, and incident response.
    """
    
    def __init__(self, config: Optional[SecurityConfiguration] = None):
        """
        Initialize security manager.
        
        Args:
            config: Security configuration
        """
        self.config = config or SecurityConfiguration()
        self.logger = logger
        
        # Initialize security components
        self._initialize_security_components()
        
        # Security state
        self.security_enabled = True
        self.blocked_ips: set = set()
        self.disabled_users: set = set()
        self.security_stats = {
            "requests_processed": 0,
            "threats_detected": 0,
            "incidents_created": 0,
            "validations_failed": 0,
            "auth_failures": 0
        }
        self.lock = threading.Lock()
        
        logger.info(f"Initialized QuantumRerankSecurityManager in {self.config.mode.value} mode")
    
    def _initialize_security_components(self) -> None:
        """Initialize all security components."""
        try:
            # Input validation
            self.input_validator = SecurityInputValidator(self.config.validation_level)
            
            # Authentication and authorization
            auth_config = {
                "token_expiry_hours": self.config.token_expiry_hours,
                "session_timeout_minutes": self.config.session_timeout_minutes
            }
            self.auth_framework = QuantumRerankAuthFramework(auth_config)
            
            # Quantum security
            if self.config.quantum_security_enabled:
                self.quantum_security = QuantumSecurityFramework()
            
            # API security
            if self.config.rate_limit_enabled:
                self.api_security = APISecurityFramework()
            
            # Security monitoring
            if self.config.enable_monitoring:
                self.security_monitor = SecurityMonitoringSystem()
                self.security_monitor.start_monitoring()
            
            # Incident response
            if self.config.enable_incident_response:
                self.incident_response = SecurityIncidentResponse(self.config.notification_configs)
            
            self.logger.info("All security components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize security components: {e}")
            raise SecurityError(f"Security manager initialization failed: {e}")
    
    def secure_request(self, request_data: Dict[str, Any], 
                      auth_header: Optional[str] = None,
                      api_key: Optional[str] = None,
                      required_permission: Optional[Permission] = None) -> SecuredRequest:
        """
        Comprehensively secure incoming request.
        
        Args:
            request_data: Request data to secure
            auth_header: Optional authentication header
            api_key: Optional API key
            required_permission: Required permission for authorization
            
        Returns:
            SecuredRequest with security validation results
        """
        if not self.security_enabled:
            return SecuredRequest(
                original_request=request_data,
                validated=True,
                sanitized_request=request_data,
                security_score=1.0
            )
        
        with self.lock:
            self.security_stats["requests_processed"] += 1
        
        secured_request = SecuredRequest(original_request=request_data)
        validation_errors = []
        
        try:
            # 1. Input validation
            validation_result = self._validate_request_input(request_data)
            secured_request.validation_errors.extend(validation_result.errors)
            
            if not validation_result.valid:
                with self.lock:
                    self.security_stats["validations_failed"] += 1
                
                if self.config.mode in [SecurityMode.STRICT, SecurityMode.PARANOID]:
                    secured_request.validated = False
                    return secured_request
            
            # 2. Authentication
            auth_result = None
            if auth_header or api_key:
                auth_result = self.auth_framework.authenticate_request(
                    auth_header=auth_header,
                    api_key=api_key
                )
                secured_request.auth_result = auth_result
                
                if not auth_result.authenticated:
                    with self.lock:
                        self.security_stats["auth_failures"] += 1
                    
                    if self.config.mode in [SecurityMode.STANDARD, SecurityMode.STRICT, SecurityMode.PARANOID]:
                        secured_request.validated = False
                        secured_request.validation_errors.append("Authentication failed")
                        return secured_request
            
            # 3. Authorization
            if auth_result and auth_result.user and required_permission:
                authz_result = self.auth_framework.authorize_operation(
                    auth_result.user, required_permission
                )
                secured_request.authz_result = authz_result
                
                if not authz_result.authorized:
                    if self.config.mode in [SecurityMode.STANDARD, SecurityMode.STRICT, SecurityMode.PARANOID]:
                        secured_request.validated = False
                        secured_request.validation_errors.append("Authorization failed")
                        return secured_request
            
            # 4. API security (rate limiting, etc.)
            if hasattr(self, 'api_security'):
                api_request = APIRequest(
                    method=request_data.get("method", "GET"),
                    endpoint=request_data.get("endpoint", "/"),
                    headers=request_data.get("headers", {}),
                    query_params=request_data.get("query_params", {}),
                    body=request_data.get("body"),
                    client_ip=request_data.get("client_ip")
                )
                
                api_result = self.api_security.secure_request(
                    api_request,
                    auth_result.user.user_id if auth_result and auth_result.user else request_data.get("client_ip", "unknown")
                )
                
                if not api_result.allowed:
                    secured_request.validated = False
                    secured_request.validation_errors.extend(api_result.reasons)
                    
                    if self.config.mode in [SecurityMode.STRICT, SecurityMode.PARANOID]:
                        return secured_request
            
            # 5. Security monitoring
            if hasattr(self, 'security_monitor'):
                security_event = SecurityEvent(
                    event_id=f"req_{int(time.time() * 1000000)}",
                    event_type="api_request",
                    timestamp=time.time(),
                    source_ip=request_data.get("client_ip"),
                    user_id=auth_result.user.user_id if auth_result and auth_result.user else None,
                    details=request_data,
                    metadata={"security_score": validation_result.security_score}
                )
                self.security_monitor.add_security_event(security_event)
            
            # 6. Create sanitized request
            secured_request.sanitized_request = validation_result.sanitized_input or request_data
            secured_request.security_score = validation_result.security_score
            secured_request.validated = True
            
            # Audit logging
            if self.config.enable_audit_logging and hasattr(self, 'security_monitor'):
                audit_event = AuditEvent(
                    event_id=f"audit_{int(time.time() * 1000000)}",
                    event_type=AuditEventType.DATA_ACCESS,
                    timestamp=time.time(),
                    user_id=auth_result.user.user_id if auth_result and auth_result.user else None,
                    resource=request_data.get("endpoint", "unknown"),
                    action="request_processed",
                    result="success" if secured_request.validated else "failure",
                    ip_address=request_data.get("client_ip"),
                    details={
                        "security_score": secured_request.security_score,
                        "validation_errors": secured_request.validation_errors
                    }
                )
                self.security_monitor.log_audit_event(audit_event)
            
            return secured_request
            
        except Exception as e:
            self.logger.error(f"Request security processing failed: {e}")
            secured_request.validated = False
            secured_request.validation_errors.append(f"Security processing error: {str(e)}")
            return secured_request
    
    def secure_quantum_computation(self, circuit_data: Dict[str, Any],
                                 parameters: Any,
                                 user_context: Optional[Dict[str, Any]] = None) -> SecuredQuantumResult:
        """
        Secure quantum computation with comprehensive validation.
        
        Args:
            circuit_data: Quantum circuit data
            parameters: Quantum parameters
            user_context: Optional user context
            
        Returns:
            SecuredQuantumResult with security validation
        """
        if not self.config.quantum_security_enabled:
            return SecuredQuantumResult(
                computation_result=None,
                security_validated=True,
                quantum_security_score=1.0
            )
        
        result = SecuredQuantumResult(computation_result=None)
        
        try:
            # Validate quantum circuit
            circuit_validation = self.quantum_security.validate_quantum_circuit(circuit_data)
            
            if not circuit_validation.secure:
                result.security_validated = False
                result.security_warnings.extend(circuit_validation.issues)
                
                if self.config.mode in [SecurityMode.STRICT, SecurityMode.PARANOID]:
                    return result
            
            # Start secure computation monitoring
            monitoring_session = self.quantum_security.secure_quantum_computation(
                circuit_data, parameters, user_context
            )
            result.monitoring_session_id = monitoring_session
            
            # Security monitoring
            if hasattr(self, 'security_monitor'):
                security_event = SecurityEvent(
                    event_id=f"quantum_{int(time.time() * 1000000)}",
                    event_type="quantum_computation",
                    timestamp=time.time(),
                    user_id=user_context.get("user_id") if user_context else None,
                    details={
                        "circuit_gates": circuit_data.get("gates", 0),
                        "parameter_count": len(parameters) if hasattr(parameters, '__len__') else 0,
                        "computation_time": 0,  # Would be updated after computation
                        "circuit_validation_score": circuit_validation.security_score
                    }
                )
                self.security_monitor.add_security_event(security_event)
            
            result.security_validated = True
            result.quantum_security_score = circuit_validation.security_score
            
            return result
            
        except Exception as e:
            self.logger.error(f"Quantum computation security validation failed: {e}")
            result.security_validated = False
            result.security_warnings.append(f"Security validation error: {str(e)}")
            return result
    
    def _validate_request_input(self, request_data: Dict[str, Any]) -> ValidationResult:
        """Validate request input data."""
        # Extract different types of input for validation
        validation_results = []
        
        # Validate text inputs
        if "text" in request_data:
            text_result = self.input_validator.validate_text_input(request_data["text"])
            validation_results.append(text_result)
        
        # Validate embedding inputs
        if "embedding" in request_data:
            embedding_result = self.input_validator.validate_embedding_input(request_data["embedding"])
            validation_results.append(embedding_result)
        
        # Validate quantum parameters
        if "parameters" in request_data:
            param_result = self.input_validator.validate_quantum_parameters(request_data["parameters"])
            validation_results.append(param_result)
        
        # Aggregate results
        if not validation_results:
            return ValidationResult(valid=True, security_score=1.0)
        
        all_valid = all(result.valid for result in validation_results)
        avg_score = sum(result.security_score for result in validation_results) / len(validation_results)
        all_errors = []
        all_warnings = []
        
        for result in validation_results:
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
        
        return ValidationResult(
            valid=all_valid,
            sanitized_input=request_data,  # Simplified - would apply sanitization
            errors=all_errors,
            warnings=all_warnings,
            security_score=avg_score
        )
    
    def handle_security_threat(self, threat_data: Dict[str, Any]) -> None:
        """Handle detected security threat."""
        if not hasattr(self, 'incident_response'):
            return
        
        with self.lock:
            self.security_stats["threats_detected"] += 1
        
        # Would integrate with threat detection from monitoring system
        self.logger.warning(f"Security threat detected: {threat_data}")
    
    def block_ip(self, ip_address: str, reason: str = "Security violation") -> None:
        """Block IP address."""
        with self.lock:
            self.blocked_ips.add(ip_address)
        
        self.logger.warning(f"Blocked IP {ip_address}: {reason}")
        
        # Audit log
        if self.config.enable_audit_logging and hasattr(self, 'security_monitor'):
            audit_event = AuditEvent(
                event_id=f"block_{int(time.time() * 1000000)}",
                event_type=AuditEventType.SECURITY_VIOLATION,
                timestamp=time.time(),
                action="block_ip",
                result="success",
                ip_address=ip_address,
                details={"reason": reason}
            )
            self.security_monitor.log_audit_event(audit_event)
    
    def disable_user(self, user_id: str, reason: str = "Security violation") -> None:
        """Disable user account."""
        with self.lock:
            self.disabled_users.add(user_id)
        
        self.logger.warning(f"Disabled user {user_id}: {reason}")
        
        # Audit log
        if self.config.enable_audit_logging and hasattr(self, 'security_monitor'):
            audit_event = AuditEvent(
                event_id=f"disable_{int(time.time() * 1000000)}",
                event_type=AuditEventType.SECURITY_VIOLATION,
                timestamp=time.time(),
                user_id=user_id,
                action="disable_user",
                result="success",
                details={"reason": reason}
            )
            self.security_monitor.log_audit_event(audit_event)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        status = {
            "security_enabled": self.security_enabled,
            "security_mode": self.config.mode.value,
            "validation_level": self.config.validation_level.value,
            "monitoring_active": self.config.enable_monitoring,
            "incident_response_active": self.config.enable_incident_response,
            "statistics": dict(self.security_stats),
            "blocked_ips": len(self.blocked_ips),
            "disabled_users": len(self.disabled_users)
        }
        
        # Add component-specific status
        if hasattr(self, 'security_monitor'):
            status["monitoring_dashboard"] = self.security_monitor.get_security_dashboard()
        
        if hasattr(self, 'incident_response'):
            status["incident_statistics"] = self.incident_response.get_incident_statistics()
        
        return status
    
    def update_security_configuration(self, new_config: SecurityConfiguration) -> None:
        """Update security configuration."""
        old_mode = self.config.mode
        self.config = new_config
        
        # Reinitialize components if necessary
        if old_mode != new_config.mode:
            self.logger.info(f"Security mode changed from {old_mode.value} to {new_config.mode.value}")
            # Would reinitialize components as needed
    
    def emergency_lockdown(self, reason: str = "Emergency security lockdown") -> None:
        """Emergency security lockdown."""
        self.config.mode = SecurityMode.PARANOID
        self.security_enabled = True
        
        self.logger.critical(f"EMERGENCY LOCKDOWN ACTIVATED: {reason}")
        
        # Audit log
        if self.config.enable_audit_logging and hasattr(self, 'security_monitor'):
            audit_event = AuditEvent(
                event_id=f"lockdown_{int(time.time() * 1000000)}",
                event_type=AuditEventType.SYSTEM_EVENT,
                timestamp=time.time(),
                action="emergency_lockdown",
                result="activated",
                details={"reason": reason}
            )
            self.security_monitor.log_audit_event(audit_event)
    
    def shutdown(self) -> None:
        """Shutdown security manager."""
        if hasattr(self, 'security_monitor'):
            self.security_monitor.stop_monitoring()
        
        self.logger.info("Security manager shutdown complete")
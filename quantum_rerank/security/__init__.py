"""
Comprehensive Security Framework for QuantumRerank.

This module provides enterprise-grade security including input validation,
authentication, authorization, quantum-specific security measures, and
incident response capabilities.

Security Components:
- Input validation and sanitization
- Authentication and authorization framework
- Quantum computation security
- API security and rate limiting
- Security monitoring and incident response
- Audit logging and compliance

Features:
- Multi-factor authentication support
- Role-based access control (RBAC)
- Quantum circuit validation and integrity
- Real-time threat detection and response
- Comprehensive audit logging
- OWASP security best practices

Integration:
- Works with existing monitoring and error handling
- Integrates with configuration management
- Extends the testing framework with security tests
"""

from .input_validator import (
    SecurityInputValidator,
    ValidationResult,
    EmbeddingValidator,
    TextValidator,
    ParameterValidator
)

from .auth_framework import (
    QuantumRerankAuthFramework,
    AuthenticationResult,
    AuthorizationResult,
    JWTTokenManager,
    APIKeyManager,
    PermissionManager,
    SessionManager
)

from .quantum_security import (
    QuantumSecurityFramework,
    SecurityValidationResult,
    SecurityMonitoringResult,
    QuantumCircuitValidator,
    ParameterIntegrityChecker,
    QuantumComputationMonitor
)

from .api_security import (
    APISecurityFramework,
    SecureRequestResult,
    EnhancedRateLimiter,
    APIRequestValidator,
    APIResponseSanitizer,
    SecurityHeaderManager
)

from .security_monitor import (
    SecurityMonitoringSystem,
    ThreatDetector,
    AnomalyDetector,
    SecurityAuditLogger,
    ComplianceMonitor
)

from .incident_response import (
    SecurityIncidentResponse,
    SecurityResponse,
    IncidentClassifier,
    ResponseActionManager,
    SecurityNotificationSystem
)

from .security_manager import (
    QuantumRerankSecurityManager,
    SecuredRequest,
    SecuredQuantumResult,
    SecurityConfiguration
)

# Security configuration constants
SECURITY_REQUIREMENTS = {
    "authentication": {
        "multi_factor_support": True,
        "token_expiration_hours": 24,
        "session_timeout_minutes": 60,
        "password_complexity": "high"
    },
    "authorization": {
        "role_based_access": True,
        "resource_level_permissions": True,
        "audit_logging": True,
        "permission_inheritance": True
    },
    "input_validation": {
        "strict_type_checking": True,
        "content_sanitization": True,
        "size_limits": True,
        "encoding_validation": True
    },
    "api_security": {
        "rate_limiting": True,
        "ddos_protection": True,
        "request_signing": True,
        "response_filtering": True
    },
    "quantum_security": {
        "circuit_validation": True,
        "parameter_integrity": True,
        "computation_monitoring": True,
        "side_channel_protection": True
    }
}

SECURITY_MONITORING_TARGETS = {
    "threat_detection": {
        "detection_latency_s": 5,
        "false_positive_rate": 0.02,
        "threat_classification_accuracy": 0.95
    },
    "incident_response": {
        "response_latency_s": 30,
        "automated_response_coverage": 0.80,
        "escalation_accuracy": 0.98
    },
    "compliance": {
        "audit_log_completeness": 1.0,
        "policy_compliance": 0.99,
        "vulnerability_detection": 0.95
    }
}

__all__ = [
    # Input Validation
    "SecurityInputValidator",
    "ValidationResult",
    "EmbeddingValidator",
    "TextValidator", 
    "ParameterValidator",
    
    # Authentication & Authorization
    "QuantumRerankAuthFramework",
    "AuthenticationResult",
    "AuthorizationResult",
    "JWTTokenManager",
    "APIKeyManager",
    "PermissionManager",
    "SessionManager",
    
    # Quantum Security
    "QuantumSecurityFramework",
    "SecurityValidationResult",
    "SecurityMonitoringResult",
    "QuantumCircuitValidator",
    "ParameterIntegrityChecker",
    "QuantumComputationMonitor",
    
    # API Security
    "APISecurityFramework",
    "SecureRequestResult", 
    "EnhancedRateLimiter",
    "APIRequestValidator",
    "APIResponseSanitizer",
    "SecurityHeaderManager",
    
    # Security Monitoring
    "SecurityMonitoringSystem",
    "ThreatDetector",
    "AnomalyDetector",
    "SecurityAuditLogger",
    "ComplianceMonitor",
    
    # Incident Response
    "SecurityIncidentResponse",
    "SecurityResponse",
    "IncidentClassifier",
    "ResponseActionManager",
    "SecurityNotificationSystem",
    
    # Security Manager
    "QuantumRerankSecurityManager",
    "SecuredRequest",
    "SecuredQuantumResult",
    "SecurityConfiguration",
    
    # Constants
    "SECURITY_REQUIREMENTS",
    "SECURITY_MONITORING_TARGETS"
]
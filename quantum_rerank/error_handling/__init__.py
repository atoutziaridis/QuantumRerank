"""
Advanced Error Handling System for QuantumRerank.

This module provides comprehensive error handling with intelligent recovery,
fallback strategies, and performance-aware error management for quantum computations.

Implements PRD Section 6.1: Technical Risk Mitigation through robust error handling.
"""

from .error_classifier import ErrorClassifier, ErrorSeverity, ErrorCategory, ErrorClassification
from .fallback_manager import AdvancedFallbackManager, FallbackStrategy, FallbackResult
from .quantum_recovery import QuantumErrorRecovery, QuantumRecoveryResult, QuantumErrorType
from .performance_handler import PerformanceErrorHandler, PerformanceThreshold
from .recovery_coordinator import RecoveryCoordinator, RecoveryAction, RecoveryResult
from .circuit_breaker import AdvancedCircuitBreaker, CircuitBreakerState
from .error_metrics import ErrorMetricsCollector, ErrorMetrics

__all__ = [
    "ErrorClassifier",
    "ErrorSeverity", 
    "ErrorCategory",
    "ErrorClassification",
    "AdvancedFallbackManager",
    "FallbackStrategy",
    "FallbackResult",
    "QuantumErrorRecovery",
    "QuantumRecoveryResult",
    "QuantumErrorType",
    "PerformanceErrorHandler",
    "PerformanceThreshold",
    "RecoveryCoordinator",
    "RecoveryAction",
    "RecoveryResult",
    "AdvancedCircuitBreaker",
    "CircuitBreakerState",
    "ErrorMetricsCollector",
    "ErrorMetrics"
]
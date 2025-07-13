"""
Utility functions and error handling for QuantumRerank.

This module provides comprehensive error handling, logging, health monitoring,
and recovery mechanisms for robust system operation.
"""

from .logging_config import (
    LogConfig,
    QuantumRerankFormatter,
    ComponentLoggerAdapter,
    LoggingConfigManager,
    setup_logging,
    get_logger,
    configure_logging_from_dict,
    get_logging_stats
)

from .exceptions import (
    QuantumRerankException,
    QuantumCircuitError,
    EmbeddingProcessingError,
    SimilarityComputationError,
    ConfigurationError,
    PerformanceError,
    DependencyError,
    RetryableError,
    TemporaryError,
    ResourceExhaustedError,
    ErrorContext,
    create_circuit_error,
    create_embedding_error,
    create_similarity_error,
    create_performance_error
)

from .error_recovery import (
    FallbackStrategy,
    RecoveryConfig,
    RecoveryMetrics,
    CircuitBreaker,
    FallbackManager,
    ErrorRecoveryManager,
    get_recovery_manager,
    with_recovery,
    configure_recovery
)

from .health_monitor import (
    HealthStatus,
    HealthCheck,
    SystemHealth,
    HealthMonitor,
    get_health_monitor,
    start_health_monitoring,
    stop_health_monitoring,
    get_system_health,
    record_operation_metric
)

__all__ = [
    # Logging
    "LogConfig",
    "QuantumRerankFormatter",
    "ComponentLoggerAdapter", 
    "LoggingConfigManager",
    "setup_logging",
    "get_logger",
    "configure_logging_from_dict",
    "get_logging_stats",
    
    # Exceptions
    "QuantumRerankException",
    "QuantumCircuitError",
    "EmbeddingProcessingError",
    "SimilarityComputationError",
    "ConfigurationError",
    "PerformanceError",
    "DependencyError",
    "RetryableError",
    "TemporaryError",
    "ResourceExhaustedError",
    "ErrorContext",
    "create_circuit_error",
    "create_embedding_error",
    "create_similarity_error",
    "create_performance_error",
    
    # Error Recovery
    "FallbackStrategy",
    "RecoveryConfig",
    "RecoveryMetrics",
    "CircuitBreaker",
    "FallbackManager",
    "ErrorRecoveryManager",
    "get_recovery_manager",
    "with_recovery",
    "configure_recovery",
    
    # Health Monitoring
    "HealthStatus",
    "HealthCheck",
    "SystemHealth",
    "HealthMonitor",
    "get_health_monitor",
    "start_health_monitoring",
    "stop_health_monitoring",
    "get_system_health",
    "record_operation_metric"
]
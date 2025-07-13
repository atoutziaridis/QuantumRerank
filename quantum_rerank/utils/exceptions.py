"""
Exception hierarchy and error handling for QuantumRerank.

This module defines custom exceptions with rich context information,
error recovery suggestions, and integration with the logging system.

Implements PRD Section 6.1: Technical Risks and Mitigation through structured error handling.
"""

from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import traceback


@dataclass
class ErrorContext:
    """Rich context information for errors."""
    component: str
    operation: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Input context
    input_data_info: Optional[Dict[str, Any]] = None
    configuration: Optional[Dict[str, Any]] = None
    
    # Performance context
    performance_metrics: Optional[Dict[str, Any]] = None
    memory_usage_mb: Optional[float] = None
    
    # System context
    system_state: Optional[Dict[str, Any]] = None
    dependencies_status: Optional[Dict[str, str]] = None
    
    # Recovery suggestions
    suggested_actions: List[str] = field(default_factory=list)
    fallback_available: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error context to dictionary."""
        return {
            "component": self.component,
            "operation": self.operation,
            "timestamp": self.timestamp.isoformat(),
            "input_data_info": self.input_data_info,
            "configuration": self.configuration,
            "performance_metrics": self.performance_metrics,
            "memory_usage_mb": self.memory_usage_mb,
            "system_state": self.system_state,
            "dependencies_status": self.dependencies_status,
            "suggested_actions": self.suggested_actions,
            "fallback_available": self.fallback_available
        }


class QuantumRerankException(Exception):
    """
    Base exception for all QuantumRerank errors.
    
    Provides rich error context and integration with logging system.
    """
    
    def __init__(self, 
                 message: str,
                 error_context: Optional[ErrorContext] = None,
                 cause: Optional[Exception] = None,
                 error_code: Optional[str] = None):
        """
        Initialize quantum rerank exception.
        
        Args:
            message: Human-readable error message
            error_context: Rich context information
            cause: Underlying exception that caused this error
            error_code: Machine-readable error code
        """
        super().__init__(message)
        self.message = message
        self.error_context = error_context or ErrorContext(
            component="unknown",
            operation="unknown"
        )
        self.cause = cause
        self.error_code = error_code or self.__class__.__name__
        self.timestamp = datetime.now()
    
    def get_detailed_info(self) -> Dict[str, Any]:
        """Get detailed error information for logging."""
        info = {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "context": self.error_context.to_dict()
        }
        
        if self.cause:
            info["underlying_cause"] = {
                "type": type(self.cause).__name__,
                "message": str(self.cause),
                "traceback": traceback.format_exception(
                    type(self.cause), self.cause, self.cause.__traceback__
                )
            }
        
        return info
    
    def get_recovery_suggestions(self) -> List[str]:
        """Get suggested recovery actions."""
        suggestions = self.error_context.suggested_actions.copy()
        
        # Add default suggestions based on error type
        if not suggestions:
            suggestions.extend(self._get_default_suggestions())
        
        return suggestions
    
    def _get_default_suggestions(self) -> List[str]:
        """Get default recovery suggestions for this error type."""
        return [
            "Check system logs for additional details",
            "Verify system configuration and dependencies",
            "Retry the operation after a brief delay"
        ]
    
    def __str__(self) -> str:
        """String representation with context."""
        base_msg = f"{self.__class__.__name__}: {self.message}"
        
        if self.error_context.component != "unknown":
            base_msg += f" (Component: {self.error_context.component}"
            if self.error_context.operation != "unknown":
                base_msg += f", Operation: {self.error_context.operation}"
            base_msg += ")"
        
        return base_msg


class QuantumCircuitError(QuantumRerankException):
    """Errors related to quantum circuit operations."""
    
    def __init__(self, 
                 message: str,
                 circuit_info: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize quantum circuit error.
        
        Args:
            message: Error message
            circuit_info: Information about the circuit that failed
            **kwargs: Additional arguments for base exception
        """
        super().__init__(message, **kwargs)
        self.circuit_info = circuit_info or {}
    
    def _get_default_suggestions(self) -> List[str]:
        """Get circuit-specific recovery suggestions."""
        suggestions = [
            "Verify quantum circuit parameters are within valid ranges",
            "Check if circuit depth exceeds maximum allowed (15 gates)",
            "Ensure qubit count is within supported range (2-4 qubits)"
        ]
        
        # Add specific suggestions based on circuit info
        if self.circuit_info.get("depth", 0) > 15:
            suggestions.append("Reduce circuit depth by simplifying parameterization")
        
        if self.circuit_info.get("n_qubits", 0) > 4:
            suggestions.append("Reduce number of qubits to supported range (2-4)")
        
        suggestions.append("Fall back to classical similarity computation")
        
        return suggestions


class EmbeddingProcessingError(QuantumRerankException):
    """Errors related to embedding processing and encoding."""
    
    def __init__(self,
                 message: str,
                 embedding_info: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize embedding processing error.
        
        Args:
            message: Error message
            embedding_info: Information about embedding processing
            **kwargs: Additional arguments for base exception
        """
        super().__init__(message, **kwargs)
        self.embedding_info = embedding_info or {}
    
    def _get_default_suggestions(self) -> List[str]:
        """Get embedding-specific recovery suggestions."""
        suggestions = [
            "Verify input text is not empty and within length limits",
            "Check if embedding model is properly loaded",
            "Ensure sufficient memory for embedding processing"
        ]
        
        # Add specific suggestions based on embedding info
        if self.embedding_info.get("text_length", 0) > 10000:
            suggestions.append("Truncate input text to maximum supported length")
        
        if self.embedding_info.get("batch_size", 0) > 100:
            suggestions.append("Reduce batch size for embedding processing")
        
        suggestions.extend([
            "Retry with simplified text preprocessing",
            "Check embedding model availability and configuration"
        ])
        
        return suggestions


class SimilarityComputationError(QuantumRerankException):
    """Errors related to similarity computation."""
    
    def __init__(self,
                 message: str,
                 similarity_method: Optional[str] = None,
                 computation_info: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize similarity computation error.
        
        Args:
            message: Error message
            similarity_method: Method used for similarity computation
            computation_info: Information about the computation
            **kwargs: Additional arguments for base exception
        """
        super().__init__(message, **kwargs)
        self.similarity_method = similarity_method
        self.computation_info = computation_info or {}
    
    def _get_default_suggestions(self) -> List[str]:
        """Get similarity computation recovery suggestions."""
        suggestions = [
            "Verify input embeddings are valid and normalized",
            "Check if similarity method configuration is correct"
        ]
        
        # Method-specific suggestions
        if self.similarity_method == "quantum_fidelity":
            suggestions.extend([
                "Fall back to classical cosine similarity",
                "Verify quantum circuit parameters are valid",
                "Check if quantum simulation is within resource limits"
            ])
        elif self.similarity_method == "hybrid_weighted":
            suggestions.extend([
                "Verify hybrid weighting configuration",
                "Fall back to single method (classical or quantum)",
                "Check if both classical and quantum methods are available"
            ])
        
        suggestions.append("Retry with different similarity method")
        
        return suggestions


class ConfigurationError(QuantumRerankException):
    """Errors related to system configuration."""
    
    def __init__(self,
                 message: str,
                 config_key: Optional[str] = None,
                 config_value: Optional[Any] = None,
                 **kwargs):
        """
        Initialize configuration error.
        
        Args:
            message: Error message
            config_key: Configuration key that caused the error
            config_value: Invalid configuration value
            **kwargs: Additional arguments for base exception
        """
        super().__init__(message, **kwargs)
        self.config_key = config_key
        self.config_value = config_value
    
    def _get_default_suggestions(self) -> List[str]:
        """Get configuration-specific recovery suggestions."""
        suggestions = [
            "Verify configuration file format and syntax",
            "Check if all required configuration keys are present",
            "Validate configuration values against expected types and ranges"
        ]
        
        if self.config_key:
            suggestions.append(f"Review configuration for key: {self.config_key}")
        
        suggestions.extend([
            "Reset to default configuration and reconfigure",
            "Check configuration documentation for valid values"
        ])
        
        return suggestions


class PerformanceError(QuantumRerankException):
    """Errors related to performance violations."""
    
    def __init__(self,
                 message: str,
                 performance_metric: Optional[str] = None,
                 actual_value: Optional[float] = None,
                 target_value: Optional[float] = None,
                 **kwargs):
        """
        Initialize performance error.
        
        Args:
            message: Error message
            performance_metric: Metric that exceeded threshold
            actual_value: Actual measured value
            target_value: Target/threshold value
            **kwargs: Additional arguments for base exception
        """
        super().__init__(message, **kwargs)
        self.performance_metric = performance_metric
        self.actual_value = actual_value
        self.target_value = target_value
    
    def _get_default_suggestions(self) -> List[str]:
        """Get performance-specific recovery suggestions."""
        suggestions = []
        
        if self.performance_metric == "latency":
            suggestions.extend([
                "Enable caching to reduce computation time",
                "Use simpler similarity methods (classical instead of quantum)",
                "Reduce batch size for processing",
                "Optimize quantum circuit parameters"
            ])
        elif self.performance_metric == "memory":
            suggestions.extend([
                "Reduce batch size to lower memory usage",
                "Clear caches and force garbage collection",
                "Use memory-efficient data structures",
                "Process data in smaller chunks"
            ])
        elif self.performance_metric == "accuracy":
            suggestions.extend([
                "Verify training data quality and quantity",
                "Adjust quantum circuit parameters",
                "Use hybrid methods combining quantum and classical approaches",
                "Retrain models with updated data"
            ])
        
        suggestions.extend([
            "Monitor system resources and optimize accordingly",
            "Consider scaling hardware resources if possible"
        ])
        
        return suggestions


class DependencyError(QuantumRerankException):
    """Errors related to external dependencies."""
    
    def __init__(self,
                 message: str,
                 dependency_name: Optional[str] = None,
                 dependency_version: Optional[str] = None,
                 **kwargs):
        """
        Initialize dependency error.
        
        Args:
            message: Error message
            dependency_name: Name of the failing dependency
            dependency_version: Version of the dependency
            **kwargs: Additional arguments for base exception
        """
        super().__init__(message, **kwargs)
        self.dependency_name = dependency_name
        self.dependency_version = dependency_version
    
    def _get_default_suggestions(self) -> List[str]:
        """Get dependency-specific recovery suggestions."""
        suggestions = [
            "Verify all required dependencies are installed",
            "Check dependency versions match requirements",
            "Update or reinstall problematic dependencies"
        ]
        
        if self.dependency_name:
            suggestions.append(f"Specifically check dependency: {self.dependency_name}")
        
        suggestions.extend([
            "Run dependency verification script",
            "Check for compatibility issues between dependencies",
            "Consult installation documentation"
        ])
        
        return suggestions


class RetryableError(QuantumRerankException):
    """Base class for errors that can be retried."""
    
    def __init__(self, 
                 message: str,
                 max_retries: int = 3,
                 retry_delay_s: float = 1.0,
                 **kwargs):
        """
        Initialize retryable error.
        
        Args:
            message: Error message
            max_retries: Maximum number of retry attempts
            retry_delay_s: Delay between retries in seconds
            **kwargs: Additional arguments for base exception
        """
        super().__init__(message, **kwargs)
        self.max_retries = max_retries
        self.retry_delay_s = retry_delay_s
        self.retry_count = 0
    
    def should_retry(self) -> bool:
        """Check if this error should be retried."""
        return self.retry_count < self.max_retries
    
    def increment_retry(self):
        """Increment retry counter."""
        self.retry_count += 1


class TemporaryError(RetryableError):
    """Temporary errors that are likely to resolve on retry."""
    
    def _get_default_suggestions(self) -> List[str]:
        return [
            f"Retry operation (attempt {self.retry_count + 1}/{self.max_retries})",
            f"Wait {self.retry_delay_s} seconds before retry",
            "Check system resources and temporary conditions"
        ]


class ResourceExhaustedError(PerformanceError):
    """Errors due to resource exhaustion."""
    
    def __init__(self,
                 message: str,
                 resource_type: str,
                 **kwargs):
        """
        Initialize resource exhausted error.
        
        Args:
            message: Error message
            resource_type: Type of resource exhausted (memory, cpu, etc.)
            **kwargs: Additional arguments for base exception
        """
        super().__init__(message, **kwargs)
        self.resource_type = resource_type
    
    def _get_default_suggestions(self) -> List[str]:
        suggestions = [
            f"Free up {self.resource_type} resources",
            "Reduce operation complexity or batch size",
            "Wait for resources to become available"
        ]
        
        if self.resource_type == "memory":
            suggestions.extend([
                "Force garbage collection",
                "Clear caches and temporary data",
                "Process data in smaller chunks"
            ])
        elif self.resource_type == "cpu":
            suggestions.extend([
                "Reduce concurrent operations",
                "Use simpler algorithms or methods",
                "Wait for CPU load to decrease"
            ])
        
        return suggestions


# Convenience functions for creating common errors

def create_circuit_error(message: str, 
                        component: str = "quantum_circuits",
                        operation: str = "circuit_creation",
                        circuit_info: Optional[Dict] = None) -> QuantumCircuitError:
    """Create a quantum circuit error with context."""
    context = ErrorContext(
        component=component,
        operation=operation,
        suggested_actions=["Fall back to classical similarity computation"],
        fallback_available=True
    )
    
    return QuantumCircuitError(
        message=message,
        error_context=context,
        circuit_info=circuit_info
    )


def create_embedding_error(message: str,
                          component: str = "embeddings",
                          operation: str = "text_encoding",
                          embedding_info: Optional[Dict] = None) -> EmbeddingProcessingError:
    """Create an embedding processing error with context."""
    context = ErrorContext(
        component=component,
        operation=operation,
        suggested_actions=["Verify input text format and length"],
        fallback_available=False
    )
    
    return EmbeddingProcessingError(
        message=message,
        error_context=context,
        embedding_info=embedding_info
    )


def create_similarity_error(message: str,
                           similarity_method: str,
                           component: str = "similarity_engine",
                           operation: str = "similarity_computation") -> SimilarityComputationError:
    """Create a similarity computation error with context."""
    context = ErrorContext(
        component=component,
        operation=operation,
        suggested_actions=["Try alternative similarity method"],
        fallback_available=True
    )
    
    return SimilarityComputationError(
        message=message,
        error_context=context,
        similarity_method=similarity_method
    )


def create_performance_error(message: str,
                            metric: str,
                            actual: float,
                            target: float,
                            component: str = "performance_monitor") -> PerformanceError:
    """Create a performance error with context."""
    context = ErrorContext(
        component=component,
        operation="performance_check",
        performance_metrics={
            "metric": metric,
            "actual_value": actual,
            "target_value": target,
            "violation_ratio": actual / target if target > 0 else float('inf')
        },
        suggested_actions=["Optimize performance or adjust targets"],
        fallback_available=True
    )
    
    return PerformanceError(
        message=message,
        error_context=context,
        performance_metric=metric,
        actual_value=actual,
        target_value=target
    )
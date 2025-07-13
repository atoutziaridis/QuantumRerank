"""
Error recovery and fallback mechanisms for QuantumRerank.

This module provides automated error recovery, fallback strategies, and
resilient operation patterns to maintain system stability.

Implements PRD Section 6.1: Technical Risks and Mitigation through robust error recovery.
"""

import time
import random
import functools
import threading
from typing import Callable, Any, Optional, Dict, List, Union, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from .exceptions import (
    QuantumRerankException, RetryableError, TemporaryError,
    QuantumCircuitError, SimilarityComputationError, PerformanceError,
    ResourceExhaustedError, ErrorContext
)
from .logging_config import get_logger

logger = get_logger("error_recovery")


class FallbackStrategy(Enum):
    """Available fallback strategies."""
    NONE = "none"
    CLASSICAL_SIMILARITY = "classical_similarity"
    SIMPLIFIED_QUANTUM = "simplified_quantum"
    CACHED_RESULT = "cached_result"
    DEFAULT_VALUE = "default_value"
    GRACEFUL_DEGRADATION = "graceful_degradation"


@dataclass
class RecoveryConfig:
    """Configuration for error recovery mechanisms."""
    # Retry configuration
    max_retries: int = 3
    initial_delay_s: float = 1.0
    max_delay_s: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    
    # Circuit breaker configuration
    enable_circuit_breaker: bool = True
    failure_threshold: int = 5
    recovery_timeout_s: float = 60.0
    half_open_max_calls: int = 3
    
    # Fallback configuration
    enable_fallback: bool = True
    fallback_strategies: Dict[str, FallbackStrategy] = field(default_factory=lambda: {
        "quantum_fidelity": FallbackStrategy.CLASSICAL_SIMILARITY,
        "quantum_circuits": FallbackStrategy.SIMPLIFIED_QUANTUM,
        "batch_processing": FallbackStrategy.GRACEFUL_DEGRADATION
    })
    
    # Performance monitoring
    track_recovery_metrics: bool = True
    alert_on_frequent_failures: bool = True
    failure_alert_threshold: int = 10


@dataclass
class RecoveryMetrics:
    """Metrics for error recovery operations."""
    operation_name: str
    total_attempts: int = 0
    successful_recoveries: int = 0
    fallback_activations: int = 0
    circuit_breaker_trips: int = 0
    last_failure_time: Optional[datetime] = None
    recovery_times: List[float] = field(default_factory=list)
    
    def get_success_rate(self) -> float:
        """Calculate recovery success rate."""
        if self.total_attempts == 0:
            return 0.0
        return self.successful_recoveries / self.total_attempts
    
    def get_avg_recovery_time(self) -> float:
        """Calculate average recovery time."""
        if not self.recovery_times:
            return 0.0
        return sum(self.recovery_times) / len(self.recovery_times)


class CircuitBreaker:
    """
    Circuit breaker pattern for preventing cascading failures.
    
    Implements three states: CLOSED (normal), OPEN (failing), HALF_OPEN (testing recovery).
    """
    
    class State(Enum):
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"
    
    def __init__(self, config: RecoveryConfig):
        self.config = config
        self.state = self.State.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            QuantumRerankException: If circuit is open
        """
        with self._lock:
            if self.state == self.State.OPEN:
                if self._should_attempt_reset():
                    self.state = self.State.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info(f"Circuit breaker entering HALF_OPEN state")
                else:
                    raise QuantumRerankException(
                        "Circuit breaker is OPEN - operation not allowed",
                        error_context=ErrorContext(
                            component="circuit_breaker",
                            operation="state_check",
                            suggested_actions=["Wait for circuit breaker recovery timeout"]
                        )
                    )
            
            elif self.state == self.State.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    self.state = self.State.OPEN
                    self.last_failure_time = datetime.now()
                    raise QuantumRerankException(
                        "Circuit breaker HALF_OPEN limit exceeded",
                        error_context=ErrorContext(
                            component="circuit_breaker",
                            operation="half_open_limit"
                        )
                    )
                self.half_open_calls += 1
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout_s
    
    def _on_success(self):
        """Handle successful operation."""
        with self._lock:
            if self.state == self.State.HALF_OPEN:
                self.state = self.State.CLOSED
                logger.info("Circuit breaker reset to CLOSED state")
            
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed operation."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if (self.state == self.State.CLOSED and 
                self.failure_count >= self.config.failure_threshold):
                self.state = self.State.OPEN
                logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
            elif self.state == self.State.HALF_OPEN:
                self.state = self.State.OPEN
                logger.warning("Circuit breaker returned to OPEN from HALF_OPEN")


class FallbackManager:
    """
    Manager for fallback strategies when primary operations fail.
    """
    
    def __init__(self, config: RecoveryConfig):
        self.config = config
        self._fallback_handlers: Dict[str, Callable] = {}
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default fallback handlers."""
        self._fallback_handlers[FallbackStrategy.CLASSICAL_SIMILARITY.value] = self._classical_similarity_fallback
        self._fallback_handlers[FallbackStrategy.SIMPLIFIED_QUANTUM.value] = self._simplified_quantum_fallback
        self._fallback_handlers[FallbackStrategy.CACHED_RESULT.value] = self._cached_result_fallback
        self._fallback_handlers[FallbackStrategy.DEFAULT_VALUE.value] = self._default_value_fallback
        self._fallback_handlers[FallbackStrategy.GRACEFUL_DEGRADATION.value] = self._graceful_degradation_fallback
    
    def execute_fallback(self, 
                        operation_name: str,
                        original_func: Callable,
                        original_args: tuple,
                        original_kwargs: dict,
                        error: Exception) -> Any:
        """
        Execute appropriate fallback strategy.
        
        Args:
            operation_name: Name of the failed operation
            original_func: Original function that failed
            original_args: Original function arguments
            original_kwargs: Original function keyword arguments
            error: Exception that triggered fallback
            
        Returns:
            Fallback result
        """
        strategy = self.config.fallback_strategies.get(operation_name, FallbackStrategy.NONE)
        
        if strategy == FallbackStrategy.NONE:
            logger.warning(f"No fallback strategy configured for {operation_name}")
            raise error
        
        handler = self._fallback_handlers.get(strategy.value)
        if not handler:
            logger.error(f"No handler found for fallback strategy: {strategy.value}")
            raise error
        
        logger.info(f"Executing fallback strategy {strategy.value} for {operation_name}")
        
        try:
            return handler(operation_name, original_func, original_args, original_kwargs, error)
        except Exception as fallback_error:
            logger.error(f"Fallback strategy {strategy.value} also failed: {fallback_error}")
            raise error  # Raise original error if fallback fails
    
    def _classical_similarity_fallback(self, operation_name: str, original_func: Callable,
                                     args: tuple, kwargs: dict, error: Exception) -> Any:
        """Fallback to classical similarity computation."""
        logger.info("Falling back to classical similarity computation")
        
        # Import here to avoid circular dependencies
        from ..core.embeddings import EmbeddingProcessor
        
        try:
            # Extract text arguments for similarity computation
            if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], str):
                processor = EmbeddingProcessor()
                embeddings = processor.encode_texts([args[0], args[1]])
                similarity = processor.compute_classical_similarity(embeddings[0], embeddings[1])
                
                return similarity, {
                    "method": "classical_cosine_fallback",
                    "fallback_reason": str(error),
                    "success": True
                }
            else:
                raise ValueError("Cannot perform classical similarity fallback - invalid arguments")
                
        except Exception as e:
            logger.error(f"Classical similarity fallback failed: {e}")
            raise error
    
    def _simplified_quantum_fallback(self, operation_name: str, original_func: Callable,
                                   args: tuple, kwargs: dict, error: Exception) -> Any:
        """Fallback to simplified quantum computation."""
        logger.info("Falling back to simplified quantum computation")
        
        # Simplify quantum parameters
        simplified_kwargs = kwargs.copy()
        
        # Reduce circuit complexity
        if 'n_qubits' in simplified_kwargs:
            simplified_kwargs['n_qubits'] = min(simplified_kwargs['n_qubits'], 2)
        if 'n_layers' in simplified_kwargs:
            simplified_kwargs['n_layers'] = min(simplified_kwargs['n_layers'], 1)
        if 'max_depth' in simplified_kwargs:
            simplified_kwargs['max_depth'] = min(simplified_kwargs['max_depth'], 10)
        
        try:
            return original_func(*args, **simplified_kwargs)
        except Exception as e:
            logger.error(f"Simplified quantum fallback failed: {e}")
            raise error
    
    def _cached_result_fallback(self, operation_name: str, original_func: Callable,
                              args: tuple, kwargs: dict, error: Exception) -> Any:
        """Fallback to cached result if available."""
        logger.info("Attempting to use cached result")
        
        # Simple cache lookup based on function name and arguments
        cache_key = f"{operation_name}_{hash(str(args))}_{hash(str(sorted(kwargs.items())))}"
        
        # This would integrate with a proper cache system
        # For now, return a placeholder indicating cache miss
        logger.warning("Cache system not implemented - fallback unavailable")
        raise error
    
    def _default_value_fallback(self, operation_name: str, original_func: Callable,
                              args: tuple, kwargs: dict, error: Exception) -> Any:
        """Fallback to safe default value."""
        logger.info("Using default value fallback")
        
        # Return safe defaults based on operation type
        if "similarity" in operation_name.lower():
            return 0.0, {"method": "default_fallback", "fallback_reason": str(error)}
        elif "embedding" in operation_name.lower():
            return [], {"method": "default_fallback", "fallback_reason": str(error)}
        else:
            return None
    
    def _graceful_degradation_fallback(self, operation_name: str, original_func: Callable,
                                     args: tuple, kwargs: dict, error: Exception) -> Any:
        """Implement graceful degradation strategy."""
        logger.info("Implementing graceful degradation")
        
        # Reduce batch size or complexity
        degraded_kwargs = kwargs.copy()
        
        if 'batch_size' in degraded_kwargs:
            degraded_kwargs['batch_size'] = max(1, degraded_kwargs['batch_size'] // 2)
        if 'top_k' in degraded_kwargs:
            degraded_kwargs['top_k'] = max(1, degraded_kwargs['top_k'] // 2)
        
        # Reduce argument complexity
        degraded_args = args
        if args and isinstance(args[0], list) and len(args[0]) > 10:
            degraded_args = (args[0][:10],) + args[1:]
        
        try:
            return original_func(*degraded_args, **degraded_kwargs)
        except Exception as e:
            logger.error(f"Graceful degradation fallback failed: {e}")
            raise error


class ErrorRecoveryManager:
    """
    Central manager for error recovery, retries, and fallback mechanisms.
    """
    
    def __init__(self, config: Optional[RecoveryConfig] = None):
        self.config = config or RecoveryConfig()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_manager = FallbackManager(self.config)
        self.metrics: Dict[str, RecoveryMetrics] = {}
        self._lock = threading.Lock()
    
    def get_circuit_breaker(self, operation_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for operation."""
        if operation_name not in self.circuit_breakers:
            with self._lock:
                if operation_name not in self.circuit_breakers:
                    self.circuit_breakers[operation_name] = CircuitBreaker(self.config)
        
        return self.circuit_breakers[operation_name]
    
    def execute_with_recovery(self, 
                            operation_name: str,
                            operation_func: Callable,
                            *args,
                            **kwargs) -> Any:
        """
        Execute operation with full error recovery capabilities.
        
        Args:
            operation_name: Name of the operation for metrics and recovery
            operation_func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Operation result
        """
        start_time = time.time()
        
        # Initialize metrics
        if operation_name not in self.metrics:
            self.metrics[operation_name] = RecoveryMetrics(operation_name)
        
        metrics = self.metrics[operation_name]
        metrics.total_attempts += 1
        
        # Get circuit breaker
        circuit_breaker = self.get_circuit_breaker(operation_name) if self.config.enable_circuit_breaker else None
        
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Execute through circuit breaker if enabled
                if circuit_breaker:
                    result = circuit_breaker.call(operation_func, *args, **kwargs)
                else:
                    result = operation_func(*args, **kwargs)
                
                # Success - update metrics
                metrics.successful_recoveries += 1
                recovery_time = time.time() - start_time
                metrics.recovery_times.append(recovery_time)
                
                if attempt > 0:
                    logger.info(f"Operation {operation_name} succeeded after {attempt} retries")
                
                return result
                
            except Exception as e:
                last_exception = e
                metrics.last_failure_time = datetime.now()
                
                logger.warning(f"Operation {operation_name} failed (attempt {attempt + 1}): {e}")
                
                # Check if this is a retryable error
                if isinstance(e, RetryableError):
                    e.increment_retry()
                    if not e.should_retry():
                        break
                elif not isinstance(e, (TemporaryError, ResourceExhaustedError)):
                    # Non-retryable error
                    break
                
                # Don't retry on last attempt
                if attempt == self.config.max_retries:
                    break
                
                # Calculate retry delay
                delay = self._calculate_retry_delay(attempt)
                logger.info(f"Retrying {operation_name} in {delay:.2f} seconds")
                time.sleep(delay)
        
        # All retries failed - try fallback if enabled
        if self.config.enable_fallback:
            try:
                logger.info(f"Attempting fallback for {operation_name}")
                result = self.fallback_manager.execute_fallback(
                    operation_name, operation_func, args, kwargs, last_exception
                )
                metrics.fallback_activations += 1
                recovery_time = time.time() - start_time
                metrics.recovery_times.append(recovery_time)
                
                return result
                
            except Exception as fallback_error:
                logger.error(f"Fallback for {operation_name} also failed: {fallback_error}")
        
        # Complete failure
        logger.error(f"Operation {operation_name} failed after all recovery attempts")
        raise last_exception or QuantumRerankException(f"Unknown error in {operation_name}")
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = self.config.initial_delay_s * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay_s)
        
        # Add jitter if enabled
        if self.config.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)
    
    def get_recovery_metrics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get recovery metrics for operations."""
        if operation_name:
            if operation_name in self.metrics:
                metrics = self.metrics[operation_name]
                return {
                    "operation_name": metrics.operation_name,
                    "total_attempts": metrics.total_attempts,
                    "success_rate": metrics.get_success_rate(),
                    "fallback_activations": metrics.fallback_activations,
                    "circuit_breaker_trips": metrics.circuit_breaker_trips,
                    "avg_recovery_time": metrics.get_avg_recovery_time(),
                    "last_failure": metrics.last_failure_time.isoformat() if metrics.last_failure_time else None
                }
            else:
                return {"error": f"No metrics found for operation: {operation_name}"}
        else:
            return {
                name: {
                    "total_attempts": metrics.total_attempts,
                    "success_rate": metrics.get_success_rate(),
                    "fallback_activations": metrics.fallback_activations,
                    "avg_recovery_time": metrics.get_avg_recovery_time()
                }
                for name, metrics in self.metrics.items()
            }


# Global recovery manager instance
_recovery_manager: Optional[ErrorRecoveryManager] = None


def get_recovery_manager() -> ErrorRecoveryManager:
    """Get global error recovery manager."""
    global _recovery_manager
    if _recovery_manager is None:
        _recovery_manager = ErrorRecoveryManager()
    return _recovery_manager


def with_recovery(operation_name: str):
    """
    Decorator for automatic error recovery.
    
    Args:
        operation_name: Name of the operation for recovery tracking
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_recovery_manager()
            return manager.execute_with_recovery(operation_name, func, *args, **kwargs)
        return wrapper
    return decorator


def configure_recovery(config: RecoveryConfig):
    """
    Configure global error recovery settings.
    
    Args:
        config: Recovery configuration
    """
    global _recovery_manager
    _recovery_manager = ErrorRecoveryManager(config)
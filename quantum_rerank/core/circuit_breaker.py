"""
Circuit breaker implementation for quantum operations.

Provides resilience against cascading failures by monitoring quantum operation
success rates and falling back to classical methods when needed.
"""

import time
import asyncio
from enum import Enum
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Blocking requests, using fallback
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5       # Number of failures before opening
    success_threshold: int = 3       # Number of successes before closing
    timeout_seconds: int = 60        # Time before trying half-open
    operation_timeout: float = 30.0  # Individual operation timeout


class CircuitBreakerMetrics:
    """Metrics tracking for circuit breaker."""
    
    def __init__(self):
        self.total_requests = 0
        self.failed_requests = 0
        self.successful_requests = 0
        self.fallback_requests = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        self.state_changes: Dict[str, int] = {
            "closed_to_open": 0,
            "open_to_half_open": 0,
            "half_open_to_closed": 0,
            "half_open_to_open": 0
        }
    
    def record_success(self):
        """Record a successful operation."""
        self.total_requests += 1
        self.successful_requests += 1
        self.last_success_time = datetime.utcnow()
    
    def record_failure(self):
        """Record a failed operation."""
        self.total_requests += 1
        self.failed_requests += 1
        self.last_failure_time = datetime.utcnow()
    
    def record_fallback(self):
        """Record a fallback operation."""
        self.fallback_requests += 1
    
    def record_state_change(self, from_state: CircuitState, to_state: CircuitState):
        """Record a state transition."""
        transition_key = f"{from_state.value}_to_{to_state.value}"
        if transition_key in self.state_changes:
            self.state_changes[transition_key] += 1
    
    def get_failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "fallback_requests": self.fallback_requests,
            "failure_rate": self.get_failure_rate(),
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
            "state_changes": self.state_changes
        }


class CircuitBreaker:
    """
    Circuit breaker for quantum operations.
    
    Monitors quantum operation success/failure rates and provides automatic
    fallback to classical methods when quantum operations are unreliable.
    """
    
    def __init__(
        self, 
        name: str = "quantum_circuit_breaker",
        config: Optional[CircuitBreakerConfig] = None
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker identifier
            config: Configuration parameters
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_state_change: Optional[datetime] = None
        
        # Metrics
        self.metrics = CircuitBreakerMetrics()
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info(f"Circuit breaker '{name}' initialized", extra={
            "failure_threshold": self.config.failure_threshold,
            "timeout_seconds": self.config.timeout_seconds
        })
    
    async def call(
        self, 
        func: Callable, 
        fallback_func: Optional[Callable] = None,
        *args, 
        **kwargs
    ) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Primary function to execute
            fallback_func: Fallback function if circuit is open
            *args: Arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Result from function execution
            
        Raises:
            Exception: If both primary and fallback functions fail
        """
        async with self._lock:
            # Check if circuit should transition states
            await self._check_state_transition()
            
            # If circuit is open, use fallback
            if self.state == CircuitState.OPEN:
                if fallback_func:
                    logger.debug(f"Circuit '{self.name}' is OPEN, using fallback")
                    self.metrics.record_fallback()
                    return await self._execute_with_timeout(fallback_func, *args, **kwargs)
                else:
                    raise Exception(f"Circuit breaker '{self.name}' is OPEN - service unavailable")
            
            # Try primary function
            try:
                result = await self._execute_with_timeout(func, *args, **kwargs)
                await self._on_success()
                return result
                
            except Exception as e:
                await self._on_failure(e)
                
                # If we have a fallback, use it
                if fallback_func:
                    logger.warning(f"Primary function failed, using fallback: {e}")
                    self.metrics.record_fallback()
                    return await self._execute_with_timeout(fallback_func, *args, **kwargs)
                else:
                    raise e
    
    async def _execute_with_timeout(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with timeout protection."""
        try:
            # Add timeout to async functions
            if asyncio.iscoroutinefunction(func):
                return await asyncio.wait_for(
                    func(*args, **kwargs), 
                    timeout=self.config.operation_timeout
                )
            else:
                # For sync functions, run in executor with timeout
                loop = asyncio.get_event_loop()
                return await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: func(*args, **kwargs)),
                    timeout=self.config.operation_timeout
                )
        except asyncio.TimeoutError:
            raise Exception(f"Operation timeout after {self.config.operation_timeout}s")
    
    async def _check_state_transition(self):
        """Check if circuit breaker should change state."""
        current_time = datetime.utcnow()
        
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed and we should try half-open
            if (self.last_failure_time and 
                current_time - self.last_failure_time >= timedelta(seconds=self.config.timeout_seconds)):
                await self._transition_to_state(CircuitState.HALF_OPEN)
        
        elif self.state == CircuitState.HALF_OPEN:
            # In half-open, we'll transition based on success/failure in _on_success/_on_failure
            pass
    
    async def _on_success(self):
        """Handle successful operation."""
        self.metrics.record_success()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            
            if self.success_count >= self.config.success_threshold:
                await self._transition_to_state(CircuitState.CLOSED)
        
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    async def _on_failure(self, exception: Exception):
        """Handle failed operation."""
        self.metrics.record_failure()
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        logger.warning(f"Circuit '{self.name}' recorded failure", extra={
            "failure_count": self.failure_count,
            "exception": str(exception),
            "state": self.state.value
        })
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                await self._transition_to_state(CircuitState.OPEN)
        
        elif self.state == CircuitState.HALF_OPEN:
            # Back to open on any failure in half-open
            await self._transition_to_state(CircuitState.OPEN)
    
    async def _transition_to_state(self, new_state: CircuitState):
        """Transition circuit breaker to new state."""
        old_state = self.state
        self.state = new_state
        self.last_state_change = datetime.utcnow()
        
        # Reset counters based on new state
        if new_state == CircuitState.CLOSED:
            self.failure_count = 0
            self.success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self.success_count = 0
        elif new_state == CircuitState.OPEN:
            self.success_count = 0
        
        # Record state change
        self.metrics.record_state_change(old_state, new_state)
        
        logger.info(f"Circuit '{self.name}' transitioned from {old_state.value} to {new_state.value}")
    
    def get_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self.state
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_state_change": self.last_state_change.isoformat() if self.last_state_change else None,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout_seconds,
                "operation_timeout": self.config.operation_timeout
            },
            "metrics": self.metrics.get_summary()
        }
    
    async def force_open(self):
        """Force circuit breaker to OPEN state (for testing/maintenance)."""
        async with self._lock:
            await self._transition_to_state(CircuitState.OPEN)
    
    async def force_closed(self):
        """Force circuit breaker to CLOSED state (for testing/recovery)."""
        async with self._lock:
            await self._transition_to_state(CircuitState.CLOSED)


# Global circuit breaker instances
quantum_circuit_breaker = CircuitBreaker(
    name="quantum_operations",
    config=CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=30,
        operation_timeout=15.0
    )
)

embedding_circuit_breaker = CircuitBreaker(
    name="embedding_operations", 
    config=CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=3,
        timeout_seconds=60,
        operation_timeout=20.0
    )
)


async def get_circuit_breaker_status() -> Dict[str, Any]:
    """Get status of all circuit breakers."""
    return {
        "quantum_operations": quantum_circuit_breaker.get_metrics(),
        "embedding_operations": embedding_circuit_breaker.get_metrics()
    }
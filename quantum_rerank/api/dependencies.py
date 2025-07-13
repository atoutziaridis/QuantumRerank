"""
FastAPI dependency injection for QuantumRerank API.

This module provides dependency functions for injecting services,
configurations, and other resources into API endpoints.
"""

from typing import Optional, Dict, Any
from functools import lru_cache
import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader

from ..core.rag_reranker import QuantumRAGReranker
from ..config.manager import ConfigManager
from ..utils.logging_config import get_logger
from ..monitoring.performance_tracker import RealTimePerformanceTracker


# Global instances (initialized in app startup)
_quantum_reranker: Optional[QuantumRAGReranker] = None
_config_manager: Optional[ConfigManager] = None
_performance_monitor: Optional[RealTimePerformanceTracker] = None
_logger = get_logger(__name__)


# API Key header (optional, configured via settings)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def initialize_dependencies(config_path: Optional[str] = None) -> None:
    """
    Initialize all global dependencies during app startup.
    
    Args:
        config_path: Optional path to configuration file
    """
    global _quantum_reranker, _config_manager, _performance_monitor
    
    try:
        # Initialize configuration manager
        _config_manager = ConfigManager(config_path)
        config = _config_manager.get_config()
        
        # Create SimilarityEngineConfig from loaded configuration
        from ..core.quantum_similarity_engine import SimilarityEngineConfig, SimilarityMethod
        
        # Map configuration to SimilarityEngineConfig
        engine_config = SimilarityEngineConfig(
            n_qubits=config.quantum.n_qubits,
            n_layers=min(config.quantum.max_circuit_depth // 2, 3),  # Approximate layers from depth
            similarity_method=SimilarityMethod.HYBRID_WEIGHTED,  # Default to hybrid
            enable_caching=config.performance.enable_caching,
            max_cache_size=config.performance.cache_size,
            performance_monitoring=True  # Always enable for monitoring
        )
        
        # Initialize quantum reranker with proper config
        _quantum_reranker = QuantumRAGReranker(config=engine_config)
        
        # Initialize performance monitor
        _performance_monitor = RealTimePerformanceTracker()
        
        _logger.info("All dependencies initialized successfully")
        
    except Exception as e:
        _logger.error(f"Failed to initialize dependencies: {e}")
        raise


async def cleanup_dependencies() -> None:
    """Cleanup dependencies during app shutdown."""
    global _quantum_reranker, _config_manager, _performance_monitor
    
    try:
        if _config_manager and hasattr(_config_manager, 'stop_watching'):
            _config_manager.stop_watching()
        
        # Cleanup other resources as needed
        _logger.info("Dependencies cleaned up successfully")
        
    except Exception as e:
        _logger.error(f"Error during dependency cleanup: {e}")


@lru_cache()
def get_quantum_reranker() -> QuantumRAGReranker:
    """
    Get the quantum reranker instance.
    
    Returns:
        QuantumRAGReranker: The initialized quantum reranker
        
    Raises:
        HTTPException: If reranker not initialized
    """
    if _quantum_reranker is None:
        raise HTTPException(
            status_code=503,
            detail="Quantum reranker not initialized"
        )
    return _quantum_reranker


@lru_cache()
def get_config_manager() -> ConfigManager:
    """
    Get the configuration manager instance.
    
    Returns:
        ConfigManager: The configuration manager
        
    Raises:
        HTTPException: If config manager not initialized
    """
    if _config_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Configuration manager not initialized"
        )
    return _config_manager


@lru_cache()
def get_performance_monitor() -> RealTimePerformanceTracker:
    """
    Get the performance monitor instance.
    
    Returns:
        RealTimePerformanceTracker: The performance monitor
        
    Raises:
        HTTPException: If monitor not initialized
    """
    if _performance_monitor is None:
        raise HTTPException(
            status_code=503,
            detail="Performance monitor not initialized"
        )
    return _performance_monitor


def get_current_config(
    config_manager: ConfigManager = Depends(get_config_manager)
) -> Dict[str, Any]:
    """
    Get current configuration as dependency.
    
    Args:
        config_manager: Injected configuration manager
        
    Returns:
        Dict containing current configuration
    """
    config = config_manager.get_config()
    return {
        "quantum": {
            "num_qubits": config.quantum.n_qubits,
            "backends": config.quantum.quantum_backends,
            "max_circuit_depth": config.quantum.max_circuit_depth
        },
        "ml": {
            "embedding_model": config.ml.embedding_model,
            "batch_size": config.ml.batch_size
        },
        "api": {
            "max_request_size": config.api.max_request_size,
            "rate_limit": config.api.rate_limit_per_minute
        }
    }


async def verify_api_key(
    api_key: Optional[str] = Depends(api_key_header),
    config_manager: ConfigManager = Depends(get_config_manager)
) -> Optional[str]:
    """
    Verify API key if authentication is enabled.
    
    Args:
        api_key: API key from request header
        config_manager: Configuration manager
        
    Returns:
        Validated API key or None if auth disabled
        
    Raises:
        HTTPException: If authentication fails
    """
    config = config_manager.get_config()
    
    # Check if authentication is enabled
    if not config.api.require_auth:
        return None
    
    # Validate API key
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    # In production, validate against database or key service
    # For now, check against configured keys
    valid_keys = getattr(config.api, 'valid_api_keys', [])
    if api_key not in valid_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return api_key


class RequestContext:
    """Context manager for request-scoped resources."""
    
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.logger = get_logger(f"api.request.{request_id}")
        self.start_time = None
        self.metadata = {}
    
    async def __aenter__(self):
        """Enter context and start timing."""
        self.start_time = asyncio.get_event_loop().time()
        self.logger.info(f"Request {self.request_id} started")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context and log completion."""
        duration = asyncio.get_event_loop().time() - self.start_time
        self.metadata['duration_ms'] = duration * 1000
        
        if exc_type:
            self.logger.error(
                f"Request {self.request_id} failed after {duration:.3f}s: {exc_val}"
            )
        else:
            self.logger.info(
                f"Request {self.request_id} completed in {duration:.3f}s"
            )
        
        return False  # Don't suppress exceptions


async def get_request_context(request: Request) -> RequestContext:
    """
    Get request context for the current request.
    
    Args:
        request: FastAPI request object
        
    Returns:
        RequestContext for the request
    """
    # Extract or generate request ID
    request_id = request.headers.get("X-Request-ID", str(id(request)))
    return RequestContext(request_id)


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self):
        self.requests = {}
        self._lock = asyncio.Lock()
    
    async def check_rate_limit(
        self, 
        key: str, 
        max_requests: int, 
        window_seconds: int
    ) -> bool:
        """
        Check if request is within rate limit.
        
        Args:
            key: Client identifier (IP or API key)
            max_requests: Maximum requests allowed
            window_seconds: Time window in seconds
            
        Returns:
            True if within limit, False otherwise
        """
        async with self._lock:
            now = asyncio.get_event_loop().time()
            
            # Clean old entries
            self.requests = {
                k: [t for t in times if now - t < window_seconds]
                for k, times in self.requests.items()
            }
            
            # Check current client
            client_requests = self.requests.get(key, [])
            if len(client_requests) >= max_requests:
                return False
            
            # Add current request
            client_requests.append(now)
            self.requests[key] = client_requests
            return True


# Global rate limiter instance
_rate_limiter = RateLimiter()


async def check_rate_limit(
    request: Request,
    config_manager: ConfigManager = Depends(get_config_manager),
    api_key: Optional[str] = Depends(verify_api_key)
) -> None:
    """
    Check rate limit for the current request.
    
    Args:
        request: FastAPI request
        config_manager: Configuration manager
        api_key: Validated API key if auth enabled
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    config = config_manager.get_config()
    
    # Use API key or client IP as rate limit key
    limit_key = api_key or request.client.host
    
    # Check rate limit
    allowed = await _rate_limiter.check_rate_limit(
        limit_key,
        config.api.rate_limit_per_minute,
        60  # 1 minute window
    )
    
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded"
        )


# Dependency for endpoints that need all core services
async def get_core_services(
    quantum_reranker: QuantumRAGReranker = Depends(get_quantum_reranker),
    config_manager: ConfigManager = Depends(get_config_manager),
    performance_monitor: RealTimePerformanceTracker = Depends(get_performance_monitor),
    _: None = Depends(check_rate_limit)
) -> Dict[str, Any]:
    """
    Get all core services as a dictionary.
    
    Returns:
        Dictionary with core service instances
    """
    return {
        "quantum_reranker": quantum_reranker,
        "config_manager": config_manager,
        "performance_monitor": performance_monitor
    }
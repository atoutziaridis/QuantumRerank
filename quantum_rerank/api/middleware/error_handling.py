"""
Error handling middleware for QuantumRerank API.

This middleware catches and formats exceptions into appropriate HTTP responses
with structured error information and proper status codes.
"""

import traceback
from typing import Callable, Dict, Any
from datetime import datetime

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import ValidationError

from ...utils.logging_config import get_logger
from ...utils.exceptions import (
    QuantumCircuitError, 
    EmbeddingProcessingError, 
    ConfigurationError,
    PerformanceError
)

logger = get_logger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive error handling middleware.
    
    Catches all exceptions and converts them to appropriate HTTP responses
    with structured error information.
    """
    
    def __init__(self, app, include_traceback: bool = False):
        """
        Initialize error handling middleware.
        
        Args:
            app: FastAPI application
            include_traceback: Whether to include tracebacks in error responses
        """
        super().__init__(app)
        self.include_traceback = include_traceback
        self.logger = logger
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with comprehensive error handling.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware or endpoint handler
            
        Returns:
            HTTP response or formatted error response
        """
        request_id = request.headers.get("X-Request-ID", str(id(request)))
        
        try:
            response = await call_next(request)
            return response
            
        except HTTPException as exc:
            # FastAPI HTTP exceptions - pass through but log
            self.logger.warning(
                "HTTP exception occurred",
                extra={
                    "request_id": request_id,
                    "status_code": exc.status_code,
                    "detail": exc.detail,
                    "path": request.url.path
                }
            )
            raise exc
            
        except ValidationError as exc:
            # Pydantic validation errors
            return await self._handle_validation_error(exc, request_id)
            
        except QuantumCircuitError as exc:
            # Quantum computation errors
            return await self._handle_quantum_error(exc, request_id)
            
        except EmbeddingProcessingError as exc:
            # Embedding processing errors
            return await self._handle_embedding_error(exc, request_id)
            
        except ConfigurationError as exc:
            # Configuration errors
            return await self._handle_configuration_error(exc, request_id)
            
        except PerformanceError as exc:
            # Performance/timeout errors
            return await self._handle_performance_error(exc, request_id)
            
        except Exception as exc:
            # Unexpected errors
            return await self._handle_unexpected_error(exc, request_id, request)
    
    async def _handle_validation_error(self, exc: ValidationError, request_id: str) -> JSONResponse:
        """Handle Pydantic validation errors."""
        errors = []
        for error in exc.errors():
            errors.append({
                "field": ".".join(str(x) for x in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            })
        
        error_response = {
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": {
                    "validation_errors": errors,
                    "request_id": request_id
                },
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
        
        self.logger.warning(
            "Validation error",
            extra={
                "request_id": request_id,
                "validation_errors": errors
            }
        )
        
        return JSONResponse(
            status_code=422,
            content=error_response
        )
    
    async def _handle_quantum_error(self, exc: QuantumCircuitError, request_id: str) -> JSONResponse:
        """Handle quantum computation errors."""
        error_response = {
            "error": {
                "code": "QUANTUM_COMPUTATION_ERROR",
                "message": str(exc),
                "details": {
                    "error_type": type(exc).__name__,
                    "component": "quantum_engine",
                    "suggested_action": "retry_with_classical_method",
                    "request_id": request_id
                },
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
        
        self.logger.error(
            "Quantum computation error",
            extra={
                "request_id": request_id,
                "error": str(exc),
                "error_type": type(exc).__name__
            }
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response
        )
    
    async def _handle_embedding_error(self, exc: EmbeddingProcessingError, request_id: str) -> JSONResponse:
        """Handle embedding processing errors."""
        error_response = {
            "error": {
                "code": "EMBEDDING_PROCESSING_ERROR",
                "message": str(exc),
                "details": {
                    "error_type": type(exc).__name__,
                    "component": "embedding_processor",
                    "suggested_action": "check_input_text_format",
                    "request_id": request_id
                },
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
        
        self.logger.error(
            "Embedding processing error",
            extra={
                "request_id": request_id,
                "error": str(exc),
                "error_type": type(exc).__name__
            }
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response
        )
    
    async def _handle_configuration_error(self, exc: ConfigurationError, request_id: str) -> JSONResponse:
        """Handle configuration errors."""
        error_response = {
            "error": {
                "code": "CONFIGURATION_ERROR",
                "message": str(exc),
                "details": {
                    "error_type": type(exc).__name__,
                    "component": "configuration_manager",
                    "suggested_action": "check_service_configuration",
                    "request_id": request_id
                },
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
        
        self.logger.error(
            "Configuration error",
            extra={
                "request_id": request_id,
                "error": str(exc),
                "error_type": type(exc).__name__
            }
        )
        
        return JSONResponse(
            status_code=503,
            content=error_response
        )
    
    async def _handle_performance_error(self, exc: PerformanceError, request_id: str) -> JSONResponse:
        """Handle performance/timeout errors."""
        error_response = {
            "error": {
                "code": "PERFORMANCE_ERROR",
                "message": str(exc),
                "details": {
                    "error_type": type(exc).__name__,
                    "component": "performance_monitor",
                    "suggested_action": "reduce_request_size_or_retry",
                    "request_id": request_id
                },
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
        
        self.logger.error(
            "Performance error",
            extra={
                "request_id": request_id,
                "error": str(exc),
                "error_type": type(exc).__name__
            }
        )
        
        return JSONResponse(
            status_code=503,
            content=error_response
        )
    
    async def _handle_unexpected_error(self, exc: Exception, request_id: str, request: Request) -> JSONResponse:
        """Handle unexpected errors."""
        error_details = {
            "error_type": type(exc).__name__,
            "component": "unknown",
            "suggested_action": "contact_support",
            "request_id": request_id,
            "path": str(request.url.path),
            "method": request.method
        }
        
        # Include traceback if enabled
        if self.include_traceback:
            error_details["traceback"] = traceback.format_exc()
        
        error_response = {
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "details": error_details,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
        
        self.logger.error(
            "Unexpected error",
            extra={
                "request_id": request_id,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "path": request.url.path,
                "method": request.method,
                "traceback": traceback.format_exc()
            }
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response
        )


class RateLimitErrorHandler:
    """
    Specialized error handler for rate limiting errors.
    """
    
    @staticmethod
    def create_rate_limit_response(request_id: str, retry_after: int = 60) -> JSONResponse:
        """
        Create a rate limit exceeded response.
        
        Args:
            request_id: Request identifier
            retry_after: Seconds to wait before retrying
            
        Returns:
            JSON response for rate limit error
        """
        error_response = {
            "error": {
                "code": "RATE_LIMIT_EXCEEDED",
                "message": "Too many requests",
                "details": {
                    "retry_after_seconds": retry_after,
                    "request_id": request_id,
                    "suggested_action": "wait_and_retry"
                },
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
        
        headers = {"Retry-After": str(retry_after)}
        
        return JSONResponse(
            status_code=429,
            content=error_response,
            headers=headers
        )


class SecurityErrorHandler:
    """
    Specialized error handler for security-related errors.
    """
    
    @staticmethod
    def create_authentication_error(request_id: str) -> JSONResponse:
        """Create authentication error response."""
        error_response = {
            "error": {
                "code": "AUTHENTICATION_REQUIRED",
                "message": "Valid authentication required",
                "details": {
                    "request_id": request_id,
                    "suggested_action": "provide_valid_api_key"
                },
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
        
        return JSONResponse(
            status_code=401,
            content=error_response,
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    @staticmethod
    def create_authorization_error(request_id: str) -> JSONResponse:
        """Create authorization error response."""
        error_response = {
            "error": {
                "code": "INSUFFICIENT_PERMISSIONS",
                "message": "Insufficient permissions for this operation",
                "details": {
                    "request_id": request_id,
                    "suggested_action": "contact_administrator"
                },
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
        
        return JSONResponse(
            status_code=403,
            content=error_response
        )


class ErrorMetricsCollector:
    """
    Collector for error metrics and statistics.
    """
    
    def __init__(self):
        self.error_counts = {}
        self.error_details = []
    
    def record_error(self, error_code: str, error_type: str, details: Dict[str, Any]) -> None:
        """
        Record an error occurrence for metrics.
        
        Args:
            error_code: Error code identifier
            error_type: Type of error
            details: Additional error details
        """
        # Update error counts
        if error_code not in self.error_counts:
            self.error_counts[error_code] = 0
        self.error_counts[error_code] += 1
        
        # Store recent error details (keep last 100)
        error_record = {
            "timestamp": datetime.utcnow(),
            "error_code": error_code,
            "error_type": error_type,
            "details": details
        }
        
        self.error_details.append(error_record)
        
        # Keep only recent errors
        if len(self.error_details) > 100:
            self.error_details = self.error_details[-100:]
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get summary of error metrics.
        
        Returns:
            Dictionary with error statistics
        """
        total_errors = sum(self.error_counts.values())
        
        return {
            "total_errors": total_errors,
            "error_counts_by_code": dict(self.error_counts),
            "recent_errors": len(self.error_details),
            "most_common_errors": sorted(
                self.error_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }
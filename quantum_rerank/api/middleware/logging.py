"""
Logging middleware for QuantumRerank API.

This middleware provides comprehensive request/response logging with
structured JSON format for monitoring and debugging.
"""

import json
import time
from typing import Callable, Dict, Any, Optional
from datetime import datetime

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ...utils.logging_config import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive logging middleware for API requests and responses.
    
    Logs all incoming requests and outgoing responses with structured
    information for monitoring and debugging purposes.
    """
    
    def __init__(
        self, 
        app,
        log_requests: bool = True,
        log_responses: bool = True,
        log_request_body: bool = False,
        log_response_body: bool = False,
        max_body_size: int = 1024
    ):
        """
        Initialize logging middleware.
        
        Args:
            app: FastAPI application
            log_requests: Whether to log incoming requests
            log_responses: Whether to log outgoing responses
            log_request_body: Whether to log request body content
            log_response_body: Whether to log response body content
            max_body_size: Maximum body size to log (bytes)
        """
        super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.max_body_size = max_body_size
        self.logger = logger
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with comprehensive logging.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware or endpoint handler
            
        Returns:
            HTTP response with logging completed
        """
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", str(id(request)))
        start_time = time.perf_counter()
        
        # Log incoming request
        if self.log_requests:
            await self._log_request(request, request_id)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate timing
            process_time = time.perf_counter() - start_time
            
            # Log outgoing response
            if self.log_responses:
                await self._log_response(response, request_id, process_time)
            
            return response
            
        except Exception as exc:
            # Log failed request
            process_time = time.perf_counter() - start_time
            await self._log_error(request, exc, request_id, process_time)
            raise exc
    
    async def _log_request(self, request: Request, request_id: str) -> None:
        """
        Log incoming request details.
        
        Args:
            request: HTTP request object
            request_id: Unique request identifier
        """
        # Extract request information
        request_info = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": self._sanitize_headers(dict(request.headers)),
            "client": {
                "host": getattr(request.client, 'host', 'unknown'),
                "port": getattr(request.client, 'port', 'unknown')
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Add request body if enabled
        if self.log_request_body:
            try:
                body = await self._get_request_body(request)
                if body:
                    request_info["body"] = body
            except Exception as e:
                request_info["body_error"] = str(e)
        
        self.logger.info(
            "Incoming request",
            extra={"request_data": request_info}
        )
    
    async def _log_response(self, response: Response, request_id: str, process_time: float) -> None:
        """
        Log outgoing response details.
        
        Args:
            response: HTTP response object
            request_id: Request identifier
            process_time: Request processing time in seconds
        """
        # Extract response information
        response_info = {
            "request_id": request_id,
            "status_code": response.status_code,
            "headers": self._sanitize_headers(dict(response.headers)),
            "process_time_ms": process_time * 1000,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Add response body if enabled
        if self.log_response_body:
            try:
                body = await self._get_response_body(response)
                if body:
                    response_info["body"] = body
            except Exception as e:
                response_info["body_error"] = str(e)
        
        # Determine log level based on status code
        if response.status_code >= 500:
            self.logger.error("Outgoing response", extra={"response_data": response_info})
        elif response.status_code >= 400:
            self.logger.warning("Outgoing response", extra={"response_data": response_info})
        else:
            self.logger.info("Outgoing response", extra={"response_data": response_info})
    
    async def _log_error(self, request: Request, exc: Exception, request_id: str, process_time: float) -> None:
        """
        Log request processing error.
        
        Args:
            request: HTTP request object
            exc: Exception that occurred
            request_id: Request identifier
            process_time: Processing time before error
        """
        error_info = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "exception": str(exc),
            "exception_type": type(exc).__name__,
            "process_time_ms": process_time * 1000,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        self.logger.error(
            "Request processing error",
            extra={"error_data": error_info}
        )
    
    async def _get_request_body(self, request: Request) -> Optional[str]:
        """
        Extract and format request body for logging.
        
        Args:
            request: HTTP request object
            
        Returns:
            Formatted request body or None
        """
        try:
            body = await request.body()
            if not body:
                return None
            
            # Limit body size
            if len(body) > self.max_body_size:
                truncated_body = body[:self.max_body_size]
                return f"{truncated_body.decode('utf-8', errors='ignore')}... [TRUNCATED]"
            
            # Try to parse as JSON for better formatting
            try:
                parsed_json = json.loads(body)
                return json.dumps(parsed_json, indent=2)
            except json.JSONDecodeError:
                return body.decode('utf-8', errors='ignore')
                
        except Exception:
            return None
    
    async def _get_response_body(self, response: Response) -> Optional[str]:
        """
        Extract and format response body for logging.
        
        Args:
            response: HTTP response object
            
        Returns:
            Formatted response body or None
        """
        try:
            # This is complex for streaming responses
            # For now, just return content type info
            content_type = response.headers.get("content-type", "unknown")
            content_length = response.headers.get("content-length", "unknown")
            
            return f"Content-Type: {content_type}, Content-Length: {content_length}"
            
        except Exception:
            return None
    
    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Sanitize headers by removing sensitive information.
        
        Args:
            headers: Original headers dictionary
            
        Returns:
            Sanitized headers dictionary
        """
        # Headers to sanitize
        sensitive_headers = {
            "authorization", "x-api-key", "cookie", "set-cookie",
            "x-auth-token", "x-access-token", "x-refresh-token"
        }
        
        sanitized = {}
        for key, value in headers.items():
            if key.lower() in sensitive_headers:
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value
        
        return sanitized


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """
    Advanced structured logging middleware with custom fields and contexts.
    """
    
    def __init__(
        self,
        app,
        service_name: str = "quantum-rerank-api",
        service_version: str = "1.0.0",
        include_user_agent: bool = True,
        include_referer: bool = True
    ):
        """
        Initialize structured logging middleware.
        
        Args:
            app: FastAPI application
            service_name: Name of the service
            service_version: Version of the service
            include_user_agent: Whether to log user agent
            include_referer: Whether to log referer
        """
        super().__init__(app)
        self.service_name = service_name
        self.service_version = service_version
        self.include_user_agent = include_user_agent
        self.include_referer = include_referer
        self.logger = logger
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with structured logging.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware or endpoint handler
            
        Returns:
            HTTP response with structured logging
        """
        request_id = request.headers.get("X-Request-ID", str(id(request)))
        start_time = time.perf_counter()
        
        # Create structured log context
        log_context = self._create_log_context(request, request_id)
        
        # Log request start
        self.logger.info(
            "Request started",
            extra={
                "event_type": "request_start",
                "context": log_context
            }
        )
        
        try:
            response = await call_next(request)
            
            # Calculate timing
            process_time = time.perf_counter() - start_time
            
            # Update context with response data
            log_context.update({
                "response": {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "content_type": response.headers.get("content-type")
                },
                "timing": {
                    "duration_ms": process_time * 1000,
                    "duration_seconds": process_time
                }
            })
            
            # Log successful completion
            self.logger.info(
                "Request completed",
                extra={
                    "event_type": "request_completed",
                    "context": log_context
                }
            )
            
            return response
            
        except Exception as exc:
            # Calculate timing for failed requests
            process_time = time.perf_counter() - start_time
            
            # Update context with error data
            log_context.update({
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc)
                },
                "timing": {
                    "duration_ms": process_time * 1000,
                    "duration_seconds": process_time
                }
            })
            
            # Log error
            self.logger.error(
                "Request failed",
                extra={
                    "event_type": "request_failed",
                    "context": log_context
                }
            )
            
            raise exc
    
    def _create_log_context(self, request: Request, request_id: str) -> Dict[str, Any]:
        """
        Create structured log context for the request.
        
        Args:
            request: HTTP request object
            request_id: Request identifier
            
        Returns:
            Structured log context dictionary
        """
        context = {
            "service": {
                "name": self.service_name,
                "version": self.service_version
            },
            "request": {
                "id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query_string": str(request.url.query),
                "scheme": request.url.scheme,
                "root_path": request.scope.get("root_path", "")
            },
            "client": {
                "host": getattr(request.client, 'host', 'unknown'),
                "port": getattr(request.client, 'port', 'unknown')
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Add optional fields
        if self.include_user_agent:
            context["request"]["user_agent"] = request.headers.get("user-agent")
        
        if self.include_referer:
            context["request"]["referer"] = request.headers.get("referer")
        
        # Add custom headers if present
        custom_headers = {}
        for header, value in request.headers.items():
            if header.lower().startswith("x-custom-"):
                custom_headers[header] = value
        
        if custom_headers:
            context["request"]["custom_headers"] = custom_headers
        
        return context


class APIMetricsLogger:
    """
    Specialized logger for API metrics and analytics.
    """
    
    def __init__(self):
        self.logger = get_logger("api.metrics")
    
    def log_request_metrics(self, request_data: Dict[str, Any]) -> None:
        """
        Log request metrics for analytics.
        
        Args:
            request_data: Request metrics data
        """
        self.logger.info(
            "API request metrics",
            extra={
                "event_type": "api_metrics",
                "metrics": request_data
            }
        )
    
    def log_performance_metrics(self, performance_data: Dict[str, Any]) -> None:
        """
        Log performance metrics.
        
        Args:
            performance_data: Performance metrics data
        """
        self.logger.info(
            "API performance metrics",
            extra={
                "event_type": "performance_metrics",
                "metrics": performance_data
            }
        )
    
    def log_error_metrics(self, error_data: Dict[str, Any]) -> None:
        """
        Log error metrics for monitoring.
        
        Args:
            error_data: Error metrics data
        """
        self.logger.error(
            "API error metrics",
            extra={
                "event_type": "error_metrics",
                "metrics": error_data
            }
        )
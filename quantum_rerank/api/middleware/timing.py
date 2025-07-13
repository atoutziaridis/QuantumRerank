"""
Timing middleware for QuantumRerank API.

This middleware measures request processing time and adds timing headers
to responses for performance monitoring and debugging.
"""

import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ...utils.logging_config import get_logger

logger = get_logger(__name__)


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to track request processing time.
    
    Adds timing information to response headers and logs performance metrics.
    """
    
    def __init__(self, app, include_in_headers: bool = True):
        """
        Initialize timing middleware.
        
        Args:
            app: FastAPI application
            include_in_headers: Whether to include timing in response headers
        """
        super().__init__(app)
        self.include_in_headers = include_in_headers
        self.logger = logger
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with timing measurement.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware or endpoint handler
            
        Returns:
            HTTP response with timing headers
        """
        # Record start time
        start_time = time.perf_counter()
        
        # Extract request metadata for logging
        request_id = request.headers.get("X-Request-ID", str(id(request)))
        method = request.method
        path = request.url.path
        client_ip = getattr(request.client, 'host', 'unknown')
        
        # Log request start
        self.logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "client_ip": client_ip,
                "user_agent": request.headers.get("User-Agent", "unknown")
            }
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate timing
            process_time = time.perf_counter() - start_time
            process_time_ms = process_time * 1000
            
            # Add timing headers if enabled
            if self.include_in_headers:
                response.headers["X-Process-Time"] = str(process_time)
                response.headers["X-Process-Time-MS"] = f"{process_time_ms:.2f}"
                response.headers["X-Request-ID"] = request_id
            
            # Log successful request completion
            self.logger.info(
                "Request completed successfully",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "status_code": response.status_code,
                    "process_time_ms": process_time_ms,
                    "client_ip": client_ip
                }
            )
            
            # Add performance warning for slow requests
            if process_time_ms > 1000:  # > 1 second
                self.logger.warning(
                    "Slow request detected",
                    extra={
                        "request_id": request_id,
                        "process_time_ms": process_time_ms,
                        "path": path,
                        "threshold_ms": 1000
                    }
                )
            
            return response
            
        except Exception as exc:
            # Calculate timing for failed requests
            process_time = time.perf_counter() - start_time
            process_time_ms = process_time * 1000
            
            # Log failed request
            self.logger.error(
                "Request failed with exception",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "process_time_ms": process_time_ms,
                    "client_ip": client_ip,
                    "exception": str(exc),
                    "exception_type": type(exc).__name__
                }
            )
            
            # Re-raise the exception to be handled by error middleware
            raise exc


class DetailedTimingMiddleware(BaseHTTPMiddleware):
    """
    Advanced timing middleware with detailed performance breakdown.
    
    Provides more granular timing information for performance optimization.
    """
    
    def __init__(self, app, enable_detailed_logging: bool = False):
        """
        Initialize detailed timing middleware.
        
        Args:
            app: FastAPI application
            enable_detailed_logging: Whether to log detailed timing breakdowns
        """
        super().__init__(app)
        self.enable_detailed_logging = enable_detailed_logging
        self.logger = logger
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with detailed timing measurement.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware or endpoint handler
            
        Returns:
            HTTP response with detailed timing information
        """
        # Initialize timing data
        timing_data = {
            "request_start": time.perf_counter(),
            "middleware_overhead": 0,
            "request_processing": 0,
            "response_generation": 0
        }
        
        request_id = request.headers.get("X-Request-ID", str(id(request)))
        
        try:
            # Measure middleware overhead (time spent in this middleware)
            middleware_start = time.perf_counter()
            
            # Process request through application
            response = await call_next(request)
            
            # Calculate detailed timings
            total_time = time.perf_counter() - timing_data["request_start"]
            timing_data["middleware_overhead"] = middleware_start - timing_data["request_start"]
            timing_data["request_processing"] = total_time - timing_data["middleware_overhead"]
            
            # Add detailed timing headers
            response.headers["X-Total-Time-MS"] = f"{total_time * 1000:.2f}"
            response.headers["X-Middleware-Overhead-MS"] = f"{timing_data['middleware_overhead'] * 1000:.2f}"
            response.headers["X-Processing-Time-MS"] = f"{timing_data['request_processing'] * 1000:.2f}"
            
            # Detailed logging if enabled
            if self.enable_detailed_logging:
                self.logger.info(
                    "Detailed timing information",
                    extra={
                        "request_id": request_id,
                        "path": request.url.path,
                        "timing_breakdown": {
                            "total_ms": total_time * 1000,
                            "middleware_overhead_ms": timing_data['middleware_overhead'] * 1000,
                            "processing_ms": timing_data['request_processing'] * 1000,
                            "overhead_percentage": (timing_data['middleware_overhead'] / total_time) * 100
                        }
                    }
                )
            
            return response
            
        except Exception as exc:
            # Log timing data even for failed requests
            total_time = time.perf_counter() - timing_data["request_start"]
            
            self.logger.error(
                "Request failed - timing data",
                extra={
                    "request_id": request_id,
                    "total_time_ms": total_time * 1000,
                    "exception": str(exc)
                }
            )
            
            raise exc


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting performance metrics and statistics.
    """
    
    def __init__(self, app, collect_metrics: bool = True):
        """
        Initialize performance monitoring middleware.
        
        Args:
            app: FastAPI application
            collect_metrics: Whether to collect and store metrics
        """
        super().__init__(app)
        self.collect_metrics = collect_metrics
        self.logger = logger
        
        # In-memory metrics storage (in production, use Redis or database)
        self.metrics = {
            "request_count": 0,
            "total_time": 0.0,
            "error_count": 0,
            "endpoint_stats": {},
            "method_stats": {}
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and collect performance metrics.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware or endpoint handler
            
        Returns:
            HTTP response with updated metrics
        """
        if not self.collect_metrics:
            return await call_next(request)
        
        start_time = time.perf_counter()
        path = request.url.path
        method = request.method
        
        try:
            response = await call_next(request)
            
            # Calculate timing
            process_time = time.perf_counter() - start_time
            
            # Update metrics
            self._update_metrics(path, method, process_time, response.status_code)
            
            return response
            
        except Exception as exc:
            # Update error metrics
            process_time = time.perf_counter() - start_time
            self._update_metrics(path, method, process_time, 500, error=True)
            raise exc
    
    def _update_metrics(self, path: str, method: str, process_time: float, 
                       status_code: int, error: bool = False) -> None:
        """
        Update internal metrics storage.
        
        Args:
            path: Request path
            method: HTTP method
            process_time: Processing time in seconds
            status_code: HTTP status code
            error: Whether this was an error
        """
        # Update global metrics
        self.metrics["request_count"] += 1
        self.metrics["total_time"] += process_time
        
        if error:
            self.metrics["error_count"] += 1
        
        # Update endpoint-specific metrics
        endpoint_key = f"{method} {path}"
        if endpoint_key not in self.metrics["endpoint_stats"]:
            self.metrics["endpoint_stats"][endpoint_key] = {
                "count": 0,
                "total_time": 0.0,
                "errors": 0,
                "min_time": float('inf'),
                "max_time": 0.0
            }
        
        endpoint_stats = self.metrics["endpoint_stats"][endpoint_key]
        endpoint_stats["count"] += 1
        endpoint_stats["total_time"] += process_time
        endpoint_stats["min_time"] = min(endpoint_stats["min_time"], process_time)
        endpoint_stats["max_time"] = max(endpoint_stats["max_time"], process_time)
        
        if error:
            endpoint_stats["errors"] += 1
        
        # Update method-specific metrics
        if method not in self.metrics["method_stats"]:
            self.metrics["method_stats"][method] = {"count": 0, "total_time": 0.0}
        
        self.metrics["method_stats"][method]["count"] += 1
        self.metrics["method_stats"][method]["total_time"] += process_time
    
    def get_metrics(self) -> dict:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        if not self.collect_metrics:
            return {}
        
        # Calculate averages and rates
        total_requests = self.metrics["request_count"]
        if total_requests == 0:
            return self.metrics
        
        avg_time = self.metrics["total_time"] / total_requests
        error_rate = (self.metrics["error_count"] / total_requests) * 100
        
        # Calculate endpoint averages
        endpoint_metrics = {}
        for endpoint, stats in self.metrics["endpoint_stats"].items():
            if stats["count"] > 0:
                endpoint_metrics[endpoint] = {
                    **stats,
                    "avg_time": stats["total_time"] / stats["count"],
                    "error_rate": (stats["errors"] / stats["count"]) * 100
                }
        
        return {
            **self.metrics,
            "avg_response_time": avg_time,
            "error_rate_percent": error_rate,
            "endpoint_details": endpoint_metrics
        }
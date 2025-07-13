"""
Security middleware for QuantumRerank API.

This module implements security headers, request filtering, and protection
mechanisms for production deployment security.
"""

import re
import time
import hashlib
from typing import Dict, Any, List, Optional, Set
from urllib.parse import urlparse

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ...utils.logging_config import get_logger
from ...config.manager import ConfigManager

logger = get_logger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive security middleware for API protection.
    
    Provides security headers, request filtering, and basic attack protection.
    """
    
    def __init__(
        self, 
        app, 
        config_manager: ConfigManager,
        enable_security_headers: bool = True,
        enable_request_filtering: bool = True,
        enable_size_limits: bool = True
    ):
        """
        Initialize security middleware.
        
        Args:
            app: FastAPI application
            config_manager: Configuration manager
            enable_security_headers: Whether to add security headers
            enable_request_filtering: Whether to filter malicious requests
            enable_size_limits: Whether to enforce request size limits
        """
        super().__init__(app)
        self.config_manager = config_manager
        self.enable_security_headers = enable_security_headers
        self.enable_request_filtering = enable_request_filtering
        self.enable_size_limits = enable_size_limits
        self.logger = logger
        
        # Security headers to add
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()"
        }
        
        # Malicious patterns to detect
        self.malicious_patterns = [
            # SQL injection patterns
            r"(\b(union|select|insert|update|delete|drop|create|alter|exec)\b)",
            r"(\b(or|and)\s+\w+\s*=\s*\w+)",
            r"([\'\"](\s*)(or|and)(\s*)\w+(\s*)=(\s*)\w+)",
            
            # XSS patterns
            r"(<script[^>]*>.*?</script>)",
            r"(javascript\s*:)",
            r"(on\w+\s*=)",
            
            # Command injection patterns
            r"(;\s*(cat|ls|pwd|whoami|uname|id)\s)",
            r"(\||\&\&|\|\|)\s*(cat|ls|pwd|whoami)",
            
            # Path traversal patterns
            r"(\.\./|\.\.\\)",
            r"(/etc/passwd|/etc/shadow)",
            
            # Common attack vectors
            r"(<iframe|<object|<embed)",
            r"(eval\s*\(|setTimeout\s*\(|setInterval\s*\()"
        ]
        
        # Compiled regex patterns for performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.malicious_patterns]
        
        # Request size limits (bytes)
        self.size_limits = {
            "max_request_size": 10 * 1024 * 1024,  # 10MB
            "max_header_size": 8 * 1024,  # 8KB
            "max_url_length": 2048,  # 2KB
            "max_json_depth": 10
        }
        
        # Blocked user agents (bots, scrapers, etc.)
        self.blocked_user_agents = {
            "curl", "wget", "python-requests", "scrapy", "bot", "crawler", 
            "spider", "scraper", "scanner", "nikto", "sqlmap"
        }
        
        # Rate limiting for suspicious IPs
        self.suspicious_ips: Dict[str, Dict[str, Any]] = {}
        self.ip_block_duration = 3600  # 1 hour
        
        # Security event counters
        self.security_events = {
            "blocked_requests": 0,
            "malicious_patterns_detected": 0,
            "size_limit_violations": 0,
            "blocked_user_agents": 0,
            "suspicious_ips_blocked": 0
        }
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request through security middleware.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware or endpoint handler
            
        Returns:
            HTTP response with security measures applied
        """
        # Extract client information
        client_ip = getattr(request.client, 'host', 'unknown')
        user_agent = request.headers.get("User-Agent", "")
        
        # Check if IP is blocked
        if self._is_ip_blocked(client_ip):
            self.security_events["suspicious_ips_blocked"] += 1
            return self._create_security_error("IP_BLOCKED", "Client IP is temporarily blocked")
        
        # Check request size limits
        if self.enable_size_limits and not self._check_size_limits(request):
            self.security_events["size_limit_violations"] += 1
            return self._create_security_error("REQUEST_TOO_LARGE", "Request exceeds size limits")
        
        # Check for blocked user agents
        if self._is_blocked_user_agent(user_agent):
            self.security_events["blocked_user_agents"] += 1
            self._mark_suspicious_ip(client_ip, "blocked_user_agent")
            return self._create_security_error("BLOCKED_USER_AGENT", "User agent not allowed")
        
        # Check for malicious patterns
        if self.enable_request_filtering and await self._contains_malicious_patterns(request):
            self.security_events["malicious_patterns_detected"] += 1
            self._mark_suspicious_ip(client_ip, "malicious_pattern")
            return self._create_security_error("MALICIOUS_REQUEST", "Request contains suspicious patterns")
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        if self.enable_security_headers:
            self._add_security_headers(response)
        
        # Log security-relevant information
        self._log_request_security(request, response)
        
        return response
    
    def _check_size_limits(self, request: Request) -> bool:
        """
        Check if request meets size limits.
        
        Args:
            request: HTTP request
            
        Returns:
            True if within limits, False otherwise
        """
        # Check URL length
        if len(str(request.url)) > self.size_limits["max_url_length"]:
            self.logger.warning(f"URL length exceeds limit: {len(str(request.url))}")
            return False
        
        # Check header size
        total_header_size = sum(len(k) + len(v) for k, v in request.headers.items())
        if total_header_size > self.size_limits["max_header_size"]:
            self.logger.warning(f"Header size exceeds limit: {total_header_size}")
            return False
        
        # Content length check (if available)
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > self.size_limits["max_request_size"]:
                    self.logger.warning(f"Content length exceeds limit: {content_length}")
                    return False
            except ValueError:
                pass
        
        return True
    
    def _is_blocked_user_agent(self, user_agent: str) -> bool:
        """
        Check if user agent is blocked.
        
        Args:
            user_agent: User agent string
            
        Returns:
            True if blocked, False otherwise
        """
        user_agent_lower = user_agent.lower()
        return any(blocked in user_agent_lower for blocked in self.blocked_user_agents)
    
    async def _contains_malicious_patterns(self, request: Request) -> bool:
        """
        Check if request contains malicious patterns.
        
        Args:
            request: HTTP request
            
        Returns:
            True if malicious patterns found, False otherwise
        """
        # Check URL parameters
        url_str = str(request.url)
        if self._scan_for_patterns(url_str):
            self.logger.warning(f"Malicious pattern in URL: {url_str}")
            return True
        
        # Check headers
        for name, value in request.headers.items():
            if self._scan_for_patterns(f"{name}: {value}"):
                self.logger.warning(f"Malicious pattern in header: {name}")
                return True
        
        # Check request body (if available and not too large)
        try:
            content_length = request.headers.get("Content-Length")
            if content_length and int(content_length) < 1024 * 1024:  # Only check if < 1MB
                # Note: This is a simplified check. In production, you'd need
                # more sophisticated body inspection that doesn't consume the stream
                pass
        except (ValueError, TypeError):
            pass
        
        return False
    
    def _scan_for_patterns(self, text: str) -> bool:
        """
        Scan text for malicious patterns.
        
        Args:
            text: Text to scan
            
        Returns:
            True if malicious patterns found
        """
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                return True
        return False
    
    def _is_ip_blocked(self, client_ip: str) -> bool:
        """
        Check if IP is currently blocked.
        
        Args:
            client_ip: Client IP address
            
        Returns:
            True if blocked, False otherwise
        """
        if client_ip not in self.suspicious_ips:
            return False
        
        ip_data = self.suspicious_ips[client_ip]
        
        # Check if block duration has expired
        if time.time() - ip_data["blocked_at"] > self.ip_block_duration:
            del self.suspicious_ips[client_ip]
            return False
        
        return ip_data.get("blocked", False)
    
    def _mark_suspicious_ip(self, client_ip: str, reason: str) -> None:
        """
        Mark IP as suspicious and potentially block it.
        
        Args:
            client_ip: Client IP address
            reason: Reason for marking as suspicious
        """
        current_time = time.time()
        
        if client_ip not in self.suspicious_ips:
            self.suspicious_ips[client_ip] = {
                "violations": 0,
                "last_violation": current_time,
                "reasons": [],
                "blocked": False
            }
        
        ip_data = self.suspicious_ips[client_ip]
        ip_data["violations"] += 1
        ip_data["last_violation"] = current_time
        ip_data["reasons"].append(reason)
        
        # Block IP if too many violations
        if ip_data["violations"] >= 5:
            ip_data["blocked"] = True
            ip_data["blocked_at"] = current_time
            
            self.logger.warning(
                f"IP blocked due to suspicious activity: {client_ip}",
                extra={
                    "violations": ip_data["violations"],
                    "reasons": ip_data["reasons"]
                }
            )
    
    def _add_security_headers(self, response: Response) -> None:
        """
        Add security headers to response.
        
        Args:
            response: HTTP response
        """
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        # Add custom headers
        response.headers["X-Security-Middleware"] = "QuantumRerank-Security"
        response.headers["X-Request-ID"] = hashlib.md5(str(time.time()).encode()).hexdigest()[:16]
    
    def _log_request_security(self, request: Request, response: Response) -> None:
        """
        Log security-relevant request information.
        
        Args:
            request: HTTP request
            response: HTTP response
        """
        # Only log if there are security concerns or errors
        if (response.status_code >= 400 or 
            getattr(request.client, 'host', '') in self.suspicious_ips):
            
            self.logger.info(
                "Security event",
                extra={
                    "client_ip": getattr(request.client, 'host', 'unknown'),
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "user_agent": request.headers.get("User-Agent", ""),
                    "referer": request.headers.get("Referer", ""),
                    "content_length": request.headers.get("Content-Length", "0")
                }
            )
    
    def _create_security_error(self, error_code: str, message: str) -> JSONResponse:
        """
        Create security error response.
        
        Args:
            error_code: Error code
            message: Error message
            
        Returns:
            JSON error response
        """
        self.security_events["blocked_requests"] += 1
        
        error_response = {
            "error": {
                "code": error_code,
                "message": message,
                "details": {
                    "security_violation": True,
                    "timestamp": time.time()
                }
            }
        }
        
        response = JSONResponse(
            status_code=403,
            content=error_response
        )
        
        # Add security headers even to error responses
        if self.enable_security_headers:
            self._add_security_headers(response)
        
        return response
    
    def get_security_stats(self) -> Dict[str, Any]:
        """
        Get security middleware statistics.
        
        Returns:
            Dictionary with security statistics
        """
        return {
            "security_events": self.security_events.copy(),
            "suspicious_ips_count": len(self.suspicious_ips),
            "blocked_ips": [
                ip for ip, data in self.suspicious_ips.items() 
                if data.get("blocked", False)
            ],
            "configuration": {
                "security_headers_enabled": self.enable_security_headers,
                "request_filtering_enabled": self.enable_request_filtering,
                "size_limits_enabled": self.enable_size_limits,
                "size_limits": self.size_limits
            }
        }
    
    def unblock_ip(self, client_ip: str) -> bool:
        """
        Manually unblock an IP address.
        
        Args:
            client_ip: IP address to unblock
            
        Returns:
            True if IP was unblocked, False if not found
        """
        if client_ip in self.suspicious_ips:
            del self.suspicious_ips[client_ip]
            self.logger.info(f"Manually unblocked IP: {client_ip}")
            return True
        return False


class CORSSecurityMiddleware(BaseHTTPMiddleware):
    """
    Secure CORS middleware with configurable origins.
    """
    
    def __init__(
        self, 
        app,
        allowed_origins: List[str] = None,
        allowed_methods: List[str] = None,
        allowed_headers: List[str] = None,
        expose_headers: List[str] = None,
        allow_credentials: bool = False,
        max_age: int = 600
    ):
        """
        Initialize CORS security middleware.
        
        Args:
            app: FastAPI application
            allowed_origins: List of allowed origins
            allowed_methods: List of allowed HTTP methods
            allowed_headers: List of allowed headers
            expose_headers: List of headers to expose
            allow_credentials: Whether to allow credentials
            max_age: Max age for preflight requests
        """
        super().__init__(app)
        self.allowed_origins = set(allowed_origins or [])
        self.allowed_methods = allowed_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allowed_headers = allowed_headers or ["*"]
        self.expose_headers = expose_headers or []
        self.allow_credentials = allow_credentials
        self.max_age = max_age
        self.logger = logger
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process CORS security.
        
        Args:
            request: HTTP request
            call_next: Next handler
            
        Returns:
            HTTP response with CORS headers
        """
        origin = request.headers.get("Origin")
        
        # Handle preflight requests
        if request.method == "OPTIONS":
            return self._create_preflight_response(origin)
        
        # Process normal request
        response = await call_next(request)
        
        # Add CORS headers
        if origin and self._is_origin_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            
            if self.allow_credentials:
                response.headers["Access-Control-Allow-Credentials"] = "true"
            
            if self.expose_headers:
                response.headers["Access-Control-Expose-Headers"] = ", ".join(self.expose_headers)
        
        return response
    
    def _is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed."""
        if not self.allowed_origins:
            return True
        
        return origin in self.allowed_origins or "*" in self.allowed_origins
    
    def _create_preflight_response(self, origin: str) -> Response:
        """Create preflight response."""
        response = Response()
        
        if origin and self._is_origin_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
            response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
            response.headers["Access-Control-Max-Age"] = str(self.max_age)
            
            if self.allow_credentials:
                response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response
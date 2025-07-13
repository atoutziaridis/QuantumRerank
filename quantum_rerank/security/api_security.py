"""
API Security Framework for QuantumRerank.

This module provides comprehensive API security including rate limiting,
request validation, response sanitization, DDoS protection, and security headers
to protect against various API-based attacks.
"""

import time
import hashlib
import secrets
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from datetime import datetime, timedelta

from ..utils.logging_config import get_logger
from ..utils.exceptions import SecurityError, RateLimitError

logger = get_logger(__name__)


class RequestMethod(Enum):
    """HTTP request methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class SecurityThreatLevel(Enum):
    """Security threat levels for API requests."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class APIRequest:
    """API request information for security analysis."""
    method: RequestMethod
    endpoint: str
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, Any] = field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    request_id: str = field(default_factory=lambda: secrets.token_hex(16))


@dataclass
class SecureRequestResult:
    """Result of API security validation."""
    allowed: bool
    threat_level: SecurityThreatLevel = SecurityThreatLevel.LOW
    rate_limited: bool = False
    sanitized_request: Optional[APIRequest] = None
    security_headers: Dict[str, str] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""
    requests_per_window: int
    window_seconds: int
    burst_allowance: int = 0
    identifier_type: str = "ip"  # ip, user, api_key


class EnhancedRateLimiter:
    """Advanced rate limiter with burst protection and adaptive limits."""
    
    def __init__(self, rules: List[RateLimitRule]):
        """
        Initialize enhanced rate limiter.
        
        Args:
            rules: List of rate limiting rules
        """
        self.rules = rules
        self.request_history: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(deque)
        )
        self.burst_tokens: Dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()
        self.logger = logger
        
        logger.info(f"Initialized EnhancedRateLimiter with {len(rules)} rules")
    
    def check_rate_limit(self, identifier: str, rule_type: str = "default") -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limits.
        
        Args:
            identifier: Unique identifier (IP, user ID, etc.)
            rule_type: Type of rate limit rule to apply
            
        Returns:
            Tuple of (allowed, metadata)
        """
        current_time = time.time()
        
        with self.lock:
            # Find applicable rule
            rule = None
            for r in self.rules:
                if r.identifier_type == rule_type:
                    rule = r
                    break
            
            if not rule:
                return True, {"reason": "No applicable rule"}
            
            # Get request history for this identifier
            history = self.request_history[identifier][rule_type]
            
            # Clean old requests outside the window
            cutoff_time = current_time - rule.window_seconds
            while history and history[0] < cutoff_time:
                history.popleft()
            
            # Check if within limits
            current_requests = len(history)
            
            # Check burst allowance
            burst_tokens = self.burst_tokens[f"{identifier}:{rule_type}"]
            
            if current_requests >= rule.requests_per_window:
                if burst_tokens > 0:
                    # Use burst token
                    self.burst_tokens[f"{identifier}:{rule_type}"] -= 1
                    history.append(current_time)
                    
                    return True, {
                        "allowed": True,
                        "used_burst_token": True,
                        "remaining_burst_tokens": self.burst_tokens[f"{identifier}:{rule_type}"],
                        "requests_in_window": current_requests + 1
                    }
                else:
                    # Rate limited
                    return False, {
                        "allowed": False,
                        "requests_in_window": current_requests,
                        "limit": rule.requests_per_window,
                        "window_seconds": rule.window_seconds,
                        "retry_after": rule.window_seconds - (current_time - history[0]) if history else rule.window_seconds
                    }
            
            # Within limits
            history.append(current_time)
            
            # Regenerate burst tokens periodically
            if current_requests == 0:  # First request in window
                self.burst_tokens[f"{identifier}:{rule_type}"] = rule.burst_allowance
            
            return True, {
                "allowed": True,
                "requests_in_window": current_requests + 1,
                "limit": rule.requests_per_window,
                "remaining_burst_tokens": self.burst_tokens[f"{identifier}:{rule_type}"]
            }
    
    def add_rule(self, rule: RateLimitRule) -> None:
        """Add new rate limiting rule."""
        with self.lock:
            self.rules.append(rule)
            self.logger.info(f"Added rate limit rule: {rule.requests_per_window}/{rule.window_seconds}s for {rule.identifier_type}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        with self.lock:
            return {
                "active_identifiers": len(self.request_history),
                "total_rules": len(self.rules),
                "burst_tokens_active": len([k for k, v in self.burst_tokens.items() if v > 0])
            }


class APIRequestValidator:
    """Validates API requests for security threats."""
    
    def __init__(self):
        """Initialize API request validator."""
        self.suspicious_patterns = [
            # SQL injection patterns
            r"(union|select|insert|update|delete|drop|create|alter)\s",
            r"'.*'.*or.*'.*'",
            r"--|#|/\*|\*/",
            
            # XSS patterns
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"data:text/html",
            
            # Command injection
            r"[;&|`$]",
            r"(rm|cat|ls|ps|kill|wget|curl)\s",
            
            # Path traversal
            r"\.\.\/",
            r"\.\.\\"
        ]
        
        self.max_request_sizes = {
            "query_params": 1000,  # max query string length
            "body_size": 10 * 1024 * 1024,  # 10MB max body
            "header_size": 8192,  # 8KB max headers
            "url_length": 2048  # 2KB max URL
        }
        
        self.forbidden_headers = {
            "x-forwarded-for",  # Can be spoofed
            "x-real-ip",  # Can be spoofed
            "x-cluster-client-ip"  # Can be spoofed
        }
        
        self.required_headers = {
            "user-agent",
            "content-type"
        }
        
        self.logger = logger
        logger.info("Initialized APIRequestValidator")
    
    def validate_request(self, request: APIRequest) -> SecureRequestResult:
        """
        Validate API request for security threats.
        
        Args:
            request: API request to validate
            
        Returns:
            SecureRequestResult with validation results
        """
        reasons = []
        metadata = {}
        threat_level = SecurityThreatLevel.LOW
        
        try:
            # Basic request validation
            validation_results = []
            
            # 1. Size validation
            validation_results.append(self._validate_request_sizes(request))
            
            # 2. Header validation
            validation_results.append(self._validate_headers(request))
            
            # 3. Content validation
            validation_results.append(self._validate_content(request))
            
            # 4. Method validation
            validation_results.append(self._validate_method(request))
            
            # 5. Rate limiting patterns
            validation_results.append(self._detect_abuse_patterns(request))
            
            # Aggregate results
            for result in validation_results:
                if not result["valid"]:
                    reasons.extend(result["reasons"])
                    if result["threat_level"].value > threat_level.value:
                        threat_level = result["threat_level"]
                metadata.update(result["metadata"])
            
            # Create sanitized request
            sanitized_request = self._sanitize_request(request) if not reasons else None
            
            # Generate security headers
            security_headers = self._generate_security_headers(request)
            
            return SecureRequestResult(
                allowed=len(reasons) == 0,
                threat_level=threat_level,
                sanitized_request=sanitized_request,
                security_headers=security_headers,
                reasons=reasons,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Request validation error: {e}")
            return SecureRequestResult(
                allowed=False,
                threat_level=SecurityThreatLevel.CRITICAL,
                reasons=[f"Validation failed: {str(e)}"]
            )
    
    def _validate_request_sizes(self, request: APIRequest) -> Dict[str, Any]:
        """Validate request size limits."""
        reasons = []
        metadata = {}
        
        # Check URL length
        full_url = f"{request.endpoint}?" + "&".join(f"{k}={v}" for k, v in request.query_params.items())
        if len(full_url) > self.max_request_sizes["url_length"]:
            reasons.append(f"URL length {len(full_url)} exceeds limit {self.max_request_sizes['url_length']}")
        
        # Check query params size
        query_string = "&".join(f"{k}={v}" for k, v in request.query_params.items())
        if len(query_string) > self.max_request_sizes["query_params"]:
            reasons.append(f"Query parameters size {len(query_string)} exceeds limit {self.max_request_sizes['query_params']}")
        
        # Check headers size
        headers_size = sum(len(f"{k}: {v}") for k, v in request.headers.items())
        if headers_size > self.max_request_sizes["header_size"]:
            reasons.append(f"Headers size {headers_size} exceeds limit {self.max_request_sizes['header_size']}")
        
        # Check body size
        if request.body:
            body_size = len(str(request.body))
            if body_size > self.max_request_sizes["body_size"]:
                reasons.append(f"Body size {body_size} exceeds limit {self.max_request_sizes['body_size']}")
        
        metadata["size_validation"] = {
            "url_length": len(full_url),
            "query_size": len(query_string),
            "headers_size": headers_size,
            "body_size": len(str(request.body)) if request.body else 0
        }
        
        return {
            "valid": len(reasons) == 0,
            "reasons": reasons,
            "threat_level": SecurityThreatLevel.HIGH if reasons else SecurityThreatLevel.LOW,
            "metadata": metadata
        }
    
    def _validate_headers(self, request: APIRequest) -> Dict[str, Any]:
        """Validate request headers."""
        reasons = []
        metadata = {}
        
        # Check for forbidden headers
        for header in self.forbidden_headers:
            if header.lower() in [h.lower() for h in request.headers.keys()]:
                reasons.append(f"Forbidden header detected: {header}")
        
        # Check for required headers
        for header in self.required_headers:
            if header.lower() not in [h.lower() for h in request.headers.keys()]:
                reasons.append(f"Missing required header: {header}")
        
        # Check for suspicious header values
        for header, value in request.headers.items():
            # Check for injection patterns in headers
            for pattern in self.suspicious_patterns:
                import re
                if re.search(pattern, str(value), re.IGNORECASE):
                    reasons.append(f"Suspicious pattern in header {header}")
                    break
        
        metadata["header_validation"] = {
            "header_count": len(request.headers),
            "forbidden_headers_found": [h for h in self.forbidden_headers if h.lower() in [h.lower() for h in request.headers.keys()]],
            "missing_required_headers": [h for h in self.required_headers if h.lower() not in [h.lower() for h in request.headers.keys()]]
        }
        
        return {
            "valid": len(reasons) == 0,
            "reasons": reasons,
            "threat_level": SecurityThreatLevel.MEDIUM if reasons else SecurityThreatLevel.LOW,
            "metadata": metadata
        }
    
    def _validate_content(self, request: APIRequest) -> Dict[str, Any]:
        """Validate request content for injection attacks."""
        reasons = []
        metadata = {}
        
        import re
        
        # Check query parameters
        for param, value in request.query_params.items():
            for pattern in self.suspicious_patterns:
                if re.search(pattern, str(value), re.IGNORECASE):
                    reasons.append(f"Suspicious pattern in query parameter {param}")
                    break
        
        # Check body content
        if request.body:
            body_str = str(request.body)
            for pattern in self.suspicious_patterns:
                if re.search(pattern, body_str, re.IGNORECASE):
                    reasons.append("Suspicious pattern in request body")
                    break
        
        metadata["content_validation"] = {
            "query_params_checked": len(request.query_params),
            "body_checked": request.body is not None,
            "patterns_checked": len(self.suspicious_patterns)
        }
        
        return {
            "valid": len(reasons) == 0,
            "reasons": reasons,
            "threat_level": SecurityThreatLevel.HIGH if reasons else SecurityThreatLevel.LOW,
            "metadata": metadata
        }
    
    def _validate_method(self, request: APIRequest) -> Dict[str, Any]:
        """Validate HTTP method appropriateness."""
        reasons = []
        metadata = {}
        
        # Check method-endpoint compatibility
        if request.method == RequestMethod.GET and request.body:
            reasons.append("GET request should not have body")
        
        if request.method in [RequestMethod.POST, RequestMethod.PUT, RequestMethod.PATCH]:
            if not request.body and "content-length" not in [h.lower() for h in request.headers.keys()]:
                reasons.append("POST/PUT/PATCH request missing body or content-length")
        
        metadata["method_validation"] = {
            "method": request.method.value,
            "has_body": request.body is not None
        }
        
        return {
            "valid": len(reasons) == 0,
            "reasons": reasons,
            "threat_level": SecurityThreatLevel.LOW,
            "metadata": metadata
        }
    
    def _detect_abuse_patterns(self, request: APIRequest) -> Dict[str, Any]:
        """Detect potential abuse patterns."""
        reasons = []
        metadata = {}
        
        # Check for automation patterns
        user_agent = request.headers.get("user-agent", "").lower()
        automated_indicators = ["bot", "crawler", "spider", "scraper", "curl", "wget", "python", "requests"]
        
        is_automated = any(indicator in user_agent for indicator in automated_indicators)
        
        # Check for suspicious timing patterns (would need historical data)
        # This is a simplified check
        if request.client_ip:
            # Check for rapid requests (simplified - would need request history)
            metadata["client_analysis"] = {
                "is_automated": is_automated,
                "user_agent": user_agent[:100],  # Truncated for safety
                "client_ip": request.client_ip
            }
        
        return {
            "valid": True,  # Don't block based on automation alone
            "reasons": reasons,
            "threat_level": SecurityThreatLevel.LOW,
            "metadata": metadata
        }
    
    def _sanitize_request(self, request: APIRequest) -> APIRequest:
        """Sanitize request by removing/encoding dangerous content."""
        import html
        
        # Create copy for sanitization
        sanitized = APIRequest(
            method=request.method,
            endpoint=request.endpoint,
            headers=request.headers.copy(),
            query_params={},
            body=None,
            client_ip=request.client_ip,
            user_agent=request.user_agent,
            timestamp=request.timestamp,
            request_id=request.request_id
        )
        
        # Sanitize query parameters
        for param, value in request.query_params.items():
            sanitized_value = html.escape(str(value))
            # Remove potentially dangerous characters
            sanitized_value = sanitized_value.replace('<', '').replace('>', '').replace('"', '')
            sanitized.query_params[param] = sanitized_value
        
        # Sanitize body
        if request.body:
            # This is a simplified sanitization - in practice would be more sophisticated
            sanitized.body = {k: html.escape(str(v)) for k, v in request.body.items()}
        
        return sanitized
    
    def _generate_security_headers(self, request: APIRequest) -> Dict[str, str]:
        """Generate security headers for response."""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "X-Request-ID": request.request_id
        }


class APIResponseSanitizer:
    """Sanitizes API responses to prevent data leakage."""
    
    def __init__(self):
        """Initialize API response sanitizer."""
        self.sensitive_fields = {
            "password", "secret", "key", "token", "api_key",
            "private", "confidential", "internal", "admin"
        }
        
        self.logger = logger
        logger.info("Initialized APIResponseSanitizer")
    
    def sanitize_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize response data to remove sensitive information.
        
        Args:
            response_data: Response data to sanitize
            
        Returns:
            Sanitized response data
        """
        if not isinstance(response_data, dict):
            return response_data
        
        sanitized = {}
        
        for key, value in response_data.items():
            key_lower = key.lower()
            
            # Check if field is sensitive
            if any(sensitive in key_lower for sensitive in self.sensitive_fields):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_response(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self.sanitize_response(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized


class SecurityHeaderManager:
    """Manages security headers for API responses."""
    
    def __init__(self):
        """Initialize security header manager."""
        self.default_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }
        
        self.logger = logger
        logger.info("Initialized SecurityHeaderManager")
    
    def get_security_headers(self, request: APIRequest) -> Dict[str, str]:
        """Get appropriate security headers for request."""
        headers = self.default_headers.copy()
        
        # Add request-specific headers
        headers["X-Request-ID"] = request.request_id
        headers["X-Timestamp"] = str(int(time.time()))
        
        return headers


class APISecurityFramework:
    """
    Comprehensive API security framework.
    
    Integrates rate limiting, request validation, response sanitization,
    and security headers for complete API protection.
    """
    
    def __init__(self, rate_limit_rules: Optional[List[RateLimitRule]] = None):
        """
        Initialize API security framework.
        
        Args:
            rate_limit_rules: Optional rate limiting rules
        """
        # Default rate limiting rules
        default_rules = [
            RateLimitRule(requests_per_window=100, window_seconds=60, burst_allowance=10, identifier_type="ip"),
            RateLimitRule(requests_per_window=1000, window_seconds=3600, burst_allowance=50, identifier_type="user"),
            RateLimitRule(requests_per_window=10000, window_seconds=3600, burst_allowance=100, identifier_type="api_key")
        ]
        
        self.rate_limiter = EnhancedRateLimiter(rate_limit_rules or default_rules)
        self.request_validator = APIRequestValidator()
        self.response_sanitizer = APIResponseSanitizer()
        self.header_manager = SecurityHeaderManager()
        
        self.logger = logger
        logger.info("Initialized APISecurityFramework")
    
    def secure_request(self, request: APIRequest, identifier: str, 
                      identifier_type: str = "ip") -> SecureRequestResult:
        """
        Secure API request with comprehensive validation.
        
        Args:
            request: API request to secure
            identifier: Unique identifier for rate limiting
            identifier_type: Type of identifier (ip, user, api_key)
            
        Returns:
            SecureRequestResult with security validation
        """
        # Check rate limits
        rate_allowed, rate_metadata = self.rate_limiter.check_rate_limit(identifier, identifier_type)
        
        if not rate_allowed:
            return SecureRequestResult(
                allowed=False,
                rate_limited=True,
                reasons=["Rate limit exceeded"],
                metadata={"rate_limit": rate_metadata}
            )
        
        # Validate request
        validation_result = self.request_validator.validate_request(request)
        
        # Add rate limit metadata
        validation_result.metadata["rate_limit"] = rate_metadata
        
        # Update rate limited flag
        validation_result.rate_limited = not rate_allowed
        
        return validation_result
    
    def secure_response(self, response_data: Dict[str, Any], 
                       request: APIRequest) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Secure API response with sanitization and headers.
        
        Args:
            response_data: Response data to secure
            request: Original request for context
            
        Returns:
            Tuple of (sanitized_data, security_headers)
        """
        sanitized_data = self.response_sanitizer.sanitize_response(response_data)
        security_headers = self.header_manager.get_security_headers(request)
        
        return sanitized_data, security_headers
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get comprehensive security statistics."""
        return {
            "rate_limiter": self.rate_limiter.get_statistics(),
            "timestamp": time.time()
        }
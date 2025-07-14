"""
Exception classes for QuantumRerank client.

This module defines custom exceptions for handling different error conditions
that can occur when interacting with the QuantumRerank API.
"""


class QuantumRerankError(Exception):
    """Base exception for all QuantumRerank client errors."""
    
    def __init__(self, message: str, status_code: int = None, response_data: dict = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_data = response_data or {}


class AuthenticationError(QuantumRerankError):
    """Raised when API authentication fails."""
    
    def __init__(self, message: str = "Authentication failed. Check your API key."):
        super().__init__(message, status_code=401)


class RateLimitError(QuantumRerankError):
    """Raised when API rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: int = None):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class ValidationError(QuantumRerankError):
    """Raised when request validation fails."""
    
    def __init__(self, message: str, field: str = None):
        super().__init__(message, status_code=422)
        self.field = field


class ServiceUnavailableError(QuantumRerankError):
    """Raised when the QuantumRerank service is unavailable."""
    
    def __init__(self, message: str = "QuantumRerank service is temporarily unavailable"):
        super().__init__(message, status_code=503)
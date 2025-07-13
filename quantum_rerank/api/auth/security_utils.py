"""
Security utilities for QuantumRerank API authentication.

This module provides cryptographic functions for secure API key hashing,
token generation, and security-related utilities.
"""

import secrets
import hashlib
import hmac
import time
import base64
from typing import Optional, Dict, Any

from ...utils.logging_config import get_logger

logger = get_logger(__name__)


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token.
    
    Args:
        length: Length of the token in characters
        
    Returns:
        Secure random token string
    """
    # Generate random bytes and encode as URL-safe base64
    random_bytes = secrets.token_bytes(length)
    return base64.urlsafe_b64encode(random_bytes).decode('utf-8')[:length]


def hash_api_key(api_key: str, salt: Optional[str] = None) -> str:
    """
    Hash an API key for secure storage.
    
    Args:
        api_key: Raw API key to hash
        salt: Optional salt (generated if not provided)
        
    Returns:
        Hashed API key with embedded salt
    """
    if salt is None:
        # Generate random salt
        salt = secrets.token_hex(16)
    
    # Create hash using PBKDF2 with SHA-256
    key_hash = hashlib.pbkdf2_hmac(
        'sha256',
        api_key.encode('utf-8'),
        salt.encode('utf-8'),
        100000  # 100,000 iterations
    )
    
    # Combine salt and hash for storage
    combined = f"{salt}:{base64.b64encode(key_hash).decode('utf-8')}"
    return combined


def verify_api_key(api_key: str, stored_hash: str) -> bool:
    """
    Verify an API key against its stored hash.
    
    Args:
        api_key: Raw API key to verify
        stored_hash: Stored hash with embedded salt
        
    Returns:
        True if the key matches the hash
    """
    try:
        # Extract salt and hash
        salt, encoded_hash = stored_hash.split(':', 1)
        stored_key_hash = base64.b64decode(encoded_hash)
        
        # Hash the provided key with the stored salt
        test_hash = hashlib.pbkdf2_hmac(
            'sha256',
            api_key.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        
        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(stored_key_hash, test_hash)
        
    except (ValueError, IndexError) as e:
        logger.warning(f"Invalid hash format in verify_api_key: {e}")
        return False


def generate_api_key_with_prefix(prefix: str = "qr", length: int = 32) -> str:
    """
    Generate an API key with a specific prefix for identification.
    
    Args:
        prefix: Prefix for the API key
        length: Total length of the key (including prefix)
        
    Returns:
        API key with prefix
    """
    # Calculate remaining length after prefix and separator
    token_length = length - len(prefix) - 1
    if token_length < 16:
        raise ValueError("Length too short for secure token generation")
    
    # Generate the token part
    token = generate_secure_token(token_length)
    
    return f"{prefix}_{token}"


def create_signature(data: str, secret_key: str) -> str:
    """
    Create HMAC signature for data integrity verification.
    
    Args:
        data: Data to sign
        secret_key: Secret key for signing
        
    Returns:
        Base64-encoded HMAC signature
    """
    signature = hmac.new(
        secret_key.encode('utf-8'),
        data.encode('utf-8'),
        hashlib.sha256
    ).digest()
    
    return base64.b64encode(signature).decode('utf-8')


def verify_signature(data: str, signature: str, secret_key: str) -> bool:
    """
    Verify HMAC signature for data integrity.
    
    Args:
        data: Original data
        signature: Base64-encoded signature to verify
        secret_key: Secret key used for signing
        
    Returns:
        True if signature is valid
    """
    try:
        expected_signature = create_signature(data, secret_key)
        return hmac.compare_digest(signature, expected_signature)
    except Exception as e:
        logger.warning(f"Signature verification error: {e}")
        return False


def generate_session_token(user_id: str, expires_in: int = 3600) -> Dict[str, Any]:
    """
    Generate a temporary session token for authenticated users.
    
    Args:
        user_id: User identifier
        expires_in: Token expiration time in seconds
        
    Returns:
        Dictionary with token and metadata
    """
    current_time = int(time.time())
    expires_at = current_time + expires_in
    
    # Create token payload
    token_data = f"{user_id}:{expires_at}"
    
    # Generate secure token
    token = generate_secure_token(24)
    
    # Create signature (in production, use a secure secret)
    secret_key = "secure-session-secret"  # Should be from configuration
    signature = create_signature(token_data, secret_key)
    
    return {
        "token": token,
        "user_id": user_id,
        "expires_at": expires_at,
        "signature": signature,
        "token_type": "session"
    }


def validate_session_token(token_data: Dict[str, Any], secret_key: str = "secure-session-secret") -> bool:
    """
    Validate a session token.
    
    Args:
        token_data: Token data dictionary
        secret_key: Secret key for signature verification
        
    Returns:
        True if token is valid and not expired
    """
    try:
        # Check expiration
        current_time = int(time.time())
        if current_time > token_data["expires_at"]:
            return False
        
        # Verify signature
        payload = f"{token_data['user_id']}:{token_data['expires_at']}"
        return verify_signature(payload, token_data["signature"], secret_key)
        
    except (KeyError, TypeError) as e:
        logger.warning(f"Invalid session token format: {e}")
        return False


def sanitize_api_key_for_logging(api_key: str) -> str:
    """
    Sanitize API key for safe logging (show only first 8 characters).
    
    Args:
        api_key: Raw API key
        
    Returns:
        Sanitized key for logging
    """
    if len(api_key) <= 8:
        return "*" * len(api_key)
    
    return api_key[:8] + "*" * (len(api_key) - 8)


def generate_webhook_signature(payload: str, secret: str) -> str:
    """
    Generate webhook signature for secure webhook delivery.
    
    Args:
        payload: Webhook payload
        secret: Webhook secret
        
    Returns:
        Webhook signature
    """
    signature = hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    return f"sha256={signature}"


def verify_webhook_signature(payload: str, signature: str, secret: str) -> bool:
    """
    Verify webhook signature.
    
    Args:
        payload: Webhook payload
        signature: Received signature (should start with 'sha256=')
        secret: Webhook secret
        
    Returns:
        True if signature is valid
    """
    if not signature.startswith('sha256='):
        return False
    
    expected_signature = generate_webhook_signature(payload, secret)
    return hmac.compare_digest(signature, expected_signature)


def mask_sensitive_data(data: Dict[str, Any], sensitive_keys: Optional[list] = None) -> Dict[str, Any]:
    """
    Mask sensitive data in dictionaries for safe logging.
    
    Args:
        data: Dictionary with potentially sensitive data
        sensitive_keys: List of keys to mask (default: common sensitive keys)
        
    Returns:
        Dictionary with sensitive values masked
    """
    if sensitive_keys is None:
        sensitive_keys = [
            'api_key', 'password', 'secret', 'token', 'auth', 
            'authorization', 'x-api-key', 'key', 'credential'
        ]
    
    masked_data = data.copy()
    
    for key, value in data.items():
        key_lower = key.lower()
        
        # Check if key contains sensitive information
        if any(sensitive_key in key_lower for sensitive_key in sensitive_keys):
            if isinstance(value, str) and len(value) > 8:
                masked_data[key] = value[:4] + "*" * (len(value) - 8) + value[-4:]
            else:
                masked_data[key] = "*" * len(str(value))
        elif isinstance(value, dict):
            # Recursively mask nested dictionaries
            masked_data[key] = mask_sensitive_data(value, sensitive_keys)
    
    return masked_data


class SecurityMetrics:
    """Security-related metrics collection."""
    
    def __init__(self):
        """Initialize security metrics."""
        self.metrics = {
            "api_key_validations": 0,
            "failed_authentications": 0,
            "successful_authentications": 0,
            "token_generations": 0,
            "signature_verifications": 0,
            "failed_signature_verifications": 0
        }
    
    def increment(self, metric_name: str, count: int = 1) -> None:
        """Increment a security metric."""
        if metric_name in self.metrics:
            self.metrics[metric_name] += count
    
    def get_metrics(self) -> Dict[str, int]:
        """Get current security metrics."""
        return self.metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset all security metrics."""
        for key in self.metrics:
            self.metrics[key] = 0


# Global security metrics instance
_security_metrics = SecurityMetrics()


def get_security_metrics() -> SecurityMetrics:
    """Get the global security metrics instance."""
    return _security_metrics
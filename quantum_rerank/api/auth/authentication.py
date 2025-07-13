"""
Authentication framework for QuantumRerank API.

This module provides API key authentication with support for development
mode bypass and enterprise-grade security patterns.
"""

import time
import hashlib
from typing import Optional, Dict, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from fastapi import HTTPException, Request
from fastapi.security import APIKeyHeader

from ...utils.logging_config import get_logger
from ...config.manager import ConfigManager

logger = get_logger(__name__)


class UserTier(str, Enum):
    """User tier levels for rate limiting and features."""
    PUBLIC = "public"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


@dataclass
class AuthenticatedUser:
    """Represents an authenticated user."""
    user_id: str
    api_key_id: str
    tier: UserTier
    permissions: list
    rate_limits: Dict[str, int]
    metadata: Dict[str, Any]
    authenticated_at: float = None
    
    def __post_init__(self):
        if self.authenticated_at is None:
            self.authenticated_at = time.time()


class APIKeyAuthenticator:
    """
    API Key authentication handler with configurable security levels.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize API key authenticator.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.logger = logger
        
        # API key header extractor
        self.api_key_header = APIKeyHeader(
            name="X-API-Key",
            auto_error=False,
            description="API key for authentication"
        )
        
        # Cache for validated keys (in production, use Redis)
        self.key_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Development mode keys (insecure - for development only)
        self.dev_keys = {
            "dev-key-123": {
                "user_id": "dev-user",
                "tier": UserTier.PREMIUM,
                "permissions": ["all"],
                "rate_limits": {"requests_per_minute": 1000}
            },
            "test-key-456": {
                "user_id": "test-user", 
                "tier": UserTier.STANDARD,
                "permissions": ["read", "compute"],
                "rate_limits": {"requests_per_minute": 300}
            }
        }
    
    async def authenticate_request(self, request: Request, api_key: Optional[str] = None) -> Optional[AuthenticatedUser]:
        """
        Authenticate an incoming request.
        
        Args:
            request: FastAPI request object
            api_key: Optional API key from header
            
        Returns:
            AuthenticatedUser if authenticated, None otherwise
            
        Raises:
            HTTPException: For authentication errors
        """
        config = self.config_manager.get_config()
        
        # Check if authentication is disabled (development mode)
        if not config.api.require_auth:
            return self._create_development_user()
        
        # Extract API key from header if not provided
        if not api_key:
            api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization", "").replace("Bearer ", "")
        
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "code": "MISSING_API_KEY",
                        "message": "API key is required",
                        "details": {
                            "header": "X-API-Key",
                            "alternative": "Authorization: Bearer <key>"
                        }
                    }
                }
            )
        
        # Validate API key
        user_data = await self._validate_api_key(api_key)
        
        if not user_data:
            # Log authentication failure
            self.logger.warning(
                "Authentication failed",
                extra={
                    "api_key_hash": hashlib.sha256(api_key.encode()).hexdigest()[:16],
                    "client_ip": getattr(request.client, 'host', 'unknown'),
                    "user_agent": request.headers.get("User-Agent", "unknown")
                }
            )
            
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "code": "INVALID_API_KEY",
                        "message": "API key is invalid or expired",
                        "details": {
                            "suggestion": "Check your API key and ensure it's active"
                        }
                    }
                }
            )
        
        # Create authenticated user
        authenticated_user = AuthenticatedUser(
            user_id=user_data["user_id"],
            api_key_id=user_data.get("api_key_id", api_key[:8]),
            tier=UserTier(user_data.get("tier", "standard")),
            permissions=user_data.get("permissions", []),
            rate_limits=user_data.get("rate_limits", {}),
            metadata=user_data.get("metadata", {})
        )
        
        # Log successful authentication
        self.logger.info(
            "Authentication successful",
            extra={
                "user_id": authenticated_user.user_id,
                "tier": authenticated_user.tier.value,
                "client_ip": getattr(request.client, 'host', 'unknown')
            }
        )
        
        return authenticated_user
    
    async def _validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Validate an API key and return user data.
        
        Args:
            api_key: API key to validate
            
        Returns:
            User data if valid, None otherwise
        """
        # Check cache first
        cache_key = hashlib.sha256(api_key.encode()).hexdigest()
        current_time = time.time()
        
        if cache_key in self.key_cache:
            cached_data, cached_time = self.key_cache[cache_key]
            if current_time - cached_time < self.cache_ttl:
                return cached_data
        
        # Development mode - check dev keys
        config = self.config_manager.get_config()
        if hasattr(config, 'development_mode') and config.development_mode:
            if api_key in self.dev_keys:
                user_data = self.dev_keys[api_key].copy()
                user_data["api_key_id"] = cache_key[:16]
                
                # Cache the result
                self.key_cache[cache_key] = (user_data, current_time)
                return user_data
        
        # Production mode - validate against key store
        # In a real implementation, this would query a database or external service
        return await self._validate_production_key(api_key, cache_key, current_time)
    
    async def _validate_production_key(self, api_key: str, cache_key: str, current_time: float) -> Optional[Dict[str, Any]]:
        """
        Validate API key in production mode.
        
        Args:
            api_key: Raw API key
            cache_key: Hashed cache key
            current_time: Current timestamp
            
        Returns:
            User data if valid
        """
        # For now, implement a simple hardcoded validation
        # In production, this would integrate with a key management service
        
        production_keys = {
            # Production API keys would be stored securely
            "prod-key-quantum-123": {
                "user_id": "quantum-research-lab",
                "tier": UserTier.ENTERPRISE,
                "permissions": ["all"],
                "rate_limits": {"requests_per_minute": 10000}
            },
            "prod-key-standard-456": {
                "user_id": "standard-user-1",
                "tier": UserTier.STANDARD,
                "permissions": ["read", "compute"],
                "rate_limits": {"requests_per_minute": 300}
            }
        }
        
        if api_key in production_keys:
            user_data = production_keys[api_key].copy()
            user_data["api_key_id"] = cache_key[:16]
            
            # Cache the result
            self.key_cache[cache_key] = (user_data, current_time)
            return user_data
        
        return None
    
    def _create_development_user(self) -> AuthenticatedUser:
        """Create a development user when auth is disabled."""
        return AuthenticatedUser(
            user_id="dev-user-default",
            api_key_id="dev-mode",
            tier=UserTier.PREMIUM,
            permissions=["all"],
            rate_limits={"requests_per_minute": 10000},
            metadata={"development_mode": True}
        )
    
    def invalidate_cache(self, api_key: Optional[str] = None) -> None:
        """
        Invalidate authentication cache.
        
        Args:
            api_key: Specific key to invalidate, or None for all
        """
        if api_key:
            cache_key = hashlib.sha256(api_key.encode()).hexdigest()
            self.key_cache.pop(cache_key, None)
        else:
            self.key_cache.clear()
        
        self.logger.info("Authentication cache invalidated")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get authentication cache statistics."""
        current_time = time.time()
        valid_entries = 0
        expired_entries = 0
        
        for cached_data, cached_time in self.key_cache.values():
            if current_time - cached_time < self.cache_ttl:
                valid_entries += 1
            else:
                expired_entries += 1
        
        return {
            "total_entries": len(self.key_cache),
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "cache_ttl_seconds": self.cache_ttl
        }


# Global authenticator instance
_authenticator: Optional[APIKeyAuthenticator] = None


def get_authenticator(config_manager: ConfigManager) -> APIKeyAuthenticator:
    """
    Get or create the global authenticator instance.
    
    Args:
        config_manager: Configuration manager
        
    Returns:
        APIKeyAuthenticator instance
    """
    global _authenticator
    
    if _authenticator is None:
        _authenticator = APIKeyAuthenticator(config_manager)
    
    return _authenticator


async def authenticate_request(request: Request, config_manager: ConfigManager) -> Optional[AuthenticatedUser]:
    """
    Convenience function to authenticate a request.
    
    Args:
        request: FastAPI request
        config_manager: Configuration manager
        
    Returns:
        Authenticated user or None
    """
    authenticator = get_authenticator(config_manager)
    return await authenticator.authenticate_request(request)


def get_current_user() -> Optional[AuthenticatedUser]:
    """
    Get the current authenticated user from request context.
    
    Note: This is a placeholder. In a real implementation,
    this would use FastAPI's dependency injection to get
    the user from the request context.
    
    Returns:
        Current authenticated user
    """
    # This would be implemented with proper FastAPI dependency injection
    # For now, return None to indicate no user context available
    return None
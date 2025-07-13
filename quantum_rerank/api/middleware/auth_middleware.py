"""
Authentication middleware for QuantumRerank API.

This module integrates authentication with the existing dependency injection
system to provide seamless authentication and authorization across endpoints.
"""

import time
from typing import Optional, Dict, Any, List
from fastapi import Request, Response, HTTPException, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware

from ...utils.logging_config import get_logger
from ...config.manager import ConfigManager
from ...security.auth import AuthenticationManager, APIKeyManager
from ...security.models import User, UserTier

logger = get_logger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware that integrates with dependency injection.
    
    Provides authentication validation, user context injection, and
    authorization enforcement across API endpoints.
    """
    
    def __init__(
        self, 
        app, 
        config_manager: ConfigManager,
        auth_manager: AuthenticationManager,
        api_key_manager: APIKeyManager,
        exempt_paths: List[str] = None
    ):
        """
        Initialize authentication middleware.
        
        Args:
            app: FastAPI application
            config_manager: Configuration manager
            auth_manager: Authentication manager
            api_key_manager: API key manager
            exempt_paths: Paths that don't require authentication
        """
        super().__init__(app)
        self.config_manager = config_manager
        self.auth_manager = auth_manager
        self.api_key_manager = api_key_manager
        self.logger = logger
        
        # Default exempt paths (health checks, docs, etc.)
        self.exempt_paths = set(exempt_paths or [
            "/health",
            "/health/ready",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json"
        ])
        
        # Authentication statistics
        self.auth_stats = {
            "total_requests": 0,
            "authenticated_requests": 0,
            "failed_authentications": 0,
            "api_key_authentications": 0,
            "token_authentications": 0,
            "exempt_requests": 0
        }
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request with authentication.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware or endpoint handler
            
        Returns:
            HTTP response with authentication context
        """
        self.auth_stats["total_requests"] += 1
        
        # Check if path is exempt from authentication
        if self._is_exempt_path(request.url.path):
            self.auth_stats["exempt_requests"] += 1
            return await call_next(request)
        
        # Extract authentication credentials
        auth_result = await self._authenticate_request(request)
        
        if not auth_result["authenticated"]:
            self.auth_stats["failed_authentications"] += 1
            return self._create_auth_error(auth_result["error"])
        
        # Add user context to request state
        request.state.user = auth_result["user"]
        request.state.auth_method = auth_result["method"]
        request.state.permissions = auth_result["permissions"]
        
        self.auth_stats["authenticated_requests"] += 1
        
        # Log authentication success
        self.logger.info(
            "Authentication successful",
            extra={
                "user_id": auth_result["user"].user_id,
                "method": auth_result["method"],
                "endpoint": request.url.path,
                "tier": auth_result["user"].tier.value
            }
        )
        
        # Process request
        response = await call_next(request)
        
        # Add authentication headers to response
        self._add_auth_headers(response, auth_result["user"])
        
        return response
    
    def _is_exempt_path(self, path: str) -> bool:
        """
        Check if path is exempt from authentication.
        
        Args:
            path: Request path
            
        Returns:
            True if exempt, False otherwise
        """
        # Exact path match
        if path in self.exempt_paths:
            return True
        
        # Pattern matching for dynamic paths
        for exempt_path in self.exempt_paths:
            if path.startswith(exempt_path):
                return True
        
        return False
    
    async def _authenticate_request(self, request: Request) -> Dict[str, Any]:
        """
        Authenticate incoming request.
        
        Args:
            request: HTTP request
            
        Returns:
            Authentication result dictionary
        """
        # Try API key authentication first
        api_key = self._extract_api_key(request)
        if api_key:
            return await self._authenticate_api_key(api_key)
        
        # Try Bearer token authentication
        token = self._extract_bearer_token(request)
        if token:
            return await self._authenticate_token(token)
        
        # No valid authentication found
        return {
            "authenticated": False,
            "error": "No valid authentication credentials provided",
            "error_code": "MISSING_CREDENTIALS"
        }
    
    def _extract_api_key(self, request: Request) -> Optional[str]:
        """
        Extract API key from request headers.
        
        Args:
            request: HTTP request
            
        Returns:
            API key if found, None otherwise
        """
        # Check X-API-Key header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key
        
        # Check Authorization header with API key format
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("ApiKey "):
            return auth_header[7:]  # Remove "ApiKey " prefix
        
        return None
    
    def _extract_bearer_token(self, request: Request) -> Optional[str]:
        """
        Extract Bearer token from request headers.
        
        Args:
            request: HTTP request
            
        Returns:
            Bearer token if found, None otherwise
        """
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix
        
        return None
    
    async def _authenticate_api_key(self, api_key: str) -> Dict[str, Any]:
        """
        Authenticate using API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            Authentication result
        """
        try:
            # Validate API key
            user_data = await self.api_key_manager.validate_api_key(api_key)
            
            if not user_data:
                return {
                    "authenticated": False,
                    "error": "Invalid API key",
                    "error_code": "INVALID_API_KEY"
                }
            
            # Create user object
            user = User(
                user_id=user_data["user_id"],
                email=user_data.get("email", ""),
                tier=UserTier(user_data.get("tier", "public")),
                is_active=user_data.get("is_active", True),
                created_at=user_data.get("created_at"),
                metadata=user_data.get("metadata", {})
            )
            
            self.auth_stats["api_key_authentications"] += 1
            
            return {
                "authenticated": True,
                "user": user,
                "method": "api_key",
                "permissions": user_data.get("permissions", [])
            }
            
        except Exception as e:
            self.logger.error(f"API key authentication error: {str(e)}")
            return {
                "authenticated": False,
                "error": "Authentication service error",
                "error_code": "AUTH_SERVICE_ERROR"
            }
    
    async def _authenticate_token(self, token: str) -> Dict[str, Any]:
        """
        Authenticate using Bearer token.
        
        Args:
            token: Bearer token to validate
            
        Returns:
            Authentication result
        """
        try:
            # Validate session token
            user_data = await self.auth_manager.validate_session_token(token)
            
            if not user_data:
                return {
                    "authenticated": False,
                    "error": "Invalid or expired token",
                    "error_code": "INVALID_TOKEN"
                }
            
            # Create user object
            user = User(
                user_id=user_data["user_id"],
                email=user_data.get("email", ""),
                tier=UserTier(user_data.get("tier", "public")),
                is_active=user_data.get("is_active", True),
                created_at=user_data.get("created_at"),
                metadata=user_data.get("metadata", {})
            )
            
            self.auth_stats["token_authentications"] += 1
            
            return {
                "authenticated": True,
                "user": user,
                "method": "token",
                "permissions": user_data.get("permissions", [])
            }
            
        except Exception as e:
            self.logger.error(f"Token authentication error: {str(e)}")
            return {
                "authenticated": False,
                "error": "Authentication service error",
                "error_code": "AUTH_SERVICE_ERROR"
            }
    
    def _add_auth_headers(self, response: Response, user: User) -> None:
        """
        Add authentication headers to response.
        
        Args:
            response: HTTP response
            user: Authenticated user
        """
        response.headers["X-Authenticated-User"] = user.user_id
        response.headers["X-User-Tier"] = user.tier.value
        response.headers["X-Auth-Timestamp"] = str(int(time.time()))
    
    def _create_auth_error(self, error_info: str) -> Response:
        """
        Create authentication error response.
        
        Args:
            error_info: Error information
            
        Returns:
            JSON error response
        """
        from fastapi.responses import JSONResponse
        
        error_response = {
            "error": {
                "code": "AUTHENTICATION_FAILED",
                "message": "Authentication required",
                "details": {
                    "reason": error_info,
                    "timestamp": time.time(),
                    "suggestion": "Provide valid API key or Bearer token"
                }
            }
        }
        
        return JSONResponse(
            status_code=401,
            content=error_response,
            headers={
                "WWW-Authenticate": "Bearer, ApiKey"
            }
        )
    
    def get_auth_stats(self) -> Dict[str, Any]:
        """
        Get authentication middleware statistics.
        
        Returns:
            Dictionary with authentication statistics
        """
        total = self.auth_stats["total_requests"]
        
        return {
            "authentication_stats": self.auth_stats.copy(),
            "success_rate": (
                self.auth_stats["authenticated_requests"] / total 
                if total > 0 else 0
            ),
            "failure_rate": (
                self.auth_stats["failed_authentications"] / total 
                if total > 0 else 0
            ),
            "exempt_rate": (
                self.auth_stats["exempt_requests"] / total 
                if total > 0 else 0
            )
        }


# Dependency injection functions for FastAPI

def get_current_user(request: Request) -> User:
    """
    Get current authenticated user from request.
    
    Args:
        request: HTTP request with user context
        
    Returns:
        Authenticated user object
        
    Raises:
        HTTPException: If user not authenticated
    """
    user = getattr(request.state, 'user', None)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )
    return user


def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Get current active user.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Active user object
        
    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=403,
            detail="User account is inactive"
        )
    return current_user


def require_tier(required_tier: UserTier):
    """
    Create dependency to require specific user tier.
    
    Args:
        required_tier: Minimum required user tier
        
    Returns:
        Dependency function
    """
    def check_tier(current_user: User = Depends(get_current_active_user)) -> User:
        """Check if user meets tier requirement."""
        # Define tier hierarchy (higher number = higher tier)
        tier_levels = {
            UserTier.PUBLIC: 0,
            UserTier.STANDARD: 1,
            UserTier.PREMIUM: 2,
            UserTier.ENTERPRISE: 3
        }
        
        user_level = tier_levels.get(current_user.tier, 0)
        required_level = tier_levels.get(required_tier, 0)
        
        if user_level < required_level:
            raise HTTPException(
                status_code=403,
                detail=f"Requires {required_tier.value} tier or higher"
            )
        
        return current_user
    
    return check_tier


def require_permission(permission: str):
    """
    Create dependency to require specific permission.
    
    Args:
        permission: Required permission
        
    Returns:
        Dependency function
    """
    def check_permission(request: Request, current_user: User = Depends(get_current_active_user)) -> User:
        """Check if user has required permission."""
        user_permissions = getattr(request.state, 'permissions', [])
        
        if permission not in user_permissions:
            raise HTTPException(
                status_code=403,
                detail=f"Missing required permission: {permission}"
            )
        
        return current_user
    
    return check_permission


# Global authentication middleware instance
_auth_middleware: Optional[AuthenticationMiddleware] = None


def get_auth_middleware(
    config_manager: ConfigManager,
    auth_manager: AuthenticationManager,
    api_key_manager: APIKeyManager
) -> AuthenticationMiddleware:
    """
    Get or create global authentication middleware.
    
    Args:
        config_manager: Configuration manager
        auth_manager: Authentication manager
        api_key_manager: API key manager
        
    Returns:
        AuthenticationMiddleware instance
    """
    global _auth_middleware
    
    if _auth_middleware is None:
        _auth_middleware = AuthenticationMiddleware(
            app=None,  # Will be set when added to FastAPI
            config_manager=config_manager,
            auth_manager=auth_manager,
            api_key_manager=api_key_manager
        )
    
    return _auth_middleware
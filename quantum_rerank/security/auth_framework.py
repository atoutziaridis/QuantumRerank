"""
Authentication and Authorization Framework for QuantumRerank.

This module provides comprehensive authentication and authorization capabilities
including JWT tokens, API keys, role-based access control, and session management
with enterprise-grade security features.
"""

import jwt
import hashlib
import secrets
import time
import uuid
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import bcrypt

from ..utils.logging_config import get_logger
from ..utils.exceptions import (
    AuthenticationError,
    AuthorizationError,
    SecurityError
)

logger = get_logger(__name__)


class UserRole(Enum):
    """User roles for access control."""
    ADMIN = "admin"
    DEVELOPER = "developer"
    USER = "user"
    READONLY = "readonly"
    SERVICE = "service"


class Permission(Enum):
    """System permissions."""
    # Quantum operations
    QUANTUM_COMPUTE = "quantum:compute"
    QUANTUM_OPTIMIZE = "quantum:optimize"
    QUANTUM_ANALYZE = "quantum:analyze"
    
    # Search operations
    SEARCH_QUERY = "search:query"
    SEARCH_RERANK = "search:rerank"
    SEARCH_INDEX = "search:index"
    
    # Data operations
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_DELETE = "data:delete"
    
    # Admin operations
    ADMIN_USER_MANAGE = "admin:user_manage"
    ADMIN_SYSTEM_CONFIG = "admin:system_config"
    ADMIN_SECURITY = "admin:security"
    
    # Monitoring
    MONITOR_VIEW = "monitor:view"
    MONITOR_ADMIN = "monitor:admin"


@dataclass
class User:
    """User information for authentication and authorization."""
    user_id: str
    username: str
    email: str
    roles: Set[UserRole]
    permissions: Set[Permission] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthenticationResult:
    """Result of authentication operation."""
    authenticated: bool
    user: Optional[User] = None
    token: Optional[str] = None
    expires_at: Optional[datetime] = None
    reason: Optional[str] = None
    security_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthorizationResult:
    """Result of authorization operation."""
    authorized: bool
    required_permission: Optional[Permission] = None
    user_permissions: Set[Permission] = field(default_factory=set)
    reason: Optional[str] = None


@dataclass
class Session:
    """User session information."""
    session_id: str
    user_id: str
    created_at: datetime
    last_accessed: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True


class JWTTokenManager:
    """JWT token management for secure authentication."""
    
    def __init__(self, secret_key: Optional[str] = None, 
                 token_expiry_hours: int = 24):
        """
        Initialize JWT token manager.
        
        Args:
            secret_key: Secret key for JWT signing (auto-generated if None)
            token_expiry_hours: Token expiration time in hours
        """
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.token_expiry_hours = token_expiry_hours
        self.algorithm = "HS256"
        self.logger = logger
        
        # Token blacklist for logout/revocation
        self.blacklisted_tokens: Set[str] = set()
        
        logger.info(f"Initialized JWT token manager with {token_expiry_hours}h expiry")
    
    def generate_token(self, user: User) -> str:
        """
        Generate JWT token for user.
        
        Args:
            user: User to generate token for
            
        Returns:
            JWT token string
        """
        now = datetime.utcnow()
        expires_at = now + timedelta(hours=self.token_expiry_hours)
        
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "roles": [role.value for role in user.roles],
            "permissions": [perm.value for perm in user.permissions],
            "iat": now.timestamp(),
            "exp": expires_at.timestamp(),
            "jti": str(uuid.uuid4())  # Token ID for revocation
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        self.logger.info(f"Generated JWT token for user {user.username}")
        return token
    
    def validate_token(self, token: str) -> AuthenticationResult:
        """
        Validate JWT token.
        
        Args:
            token: JWT token to validate
            
        Returns:
            AuthenticationResult with validation results
        """
        try:
            # Check if token is blacklisted
            if token in self.blacklisted_tokens:
                return AuthenticationResult(
                    authenticated=False,
                    reason="Token has been revoked"
                )
            
            # Decode and validate token
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            
            # Extract user information
            user = User(
                user_id=payload["user_id"],
                username=payload["username"],
                email="",  # Not stored in token for security
                roles={UserRole(role) for role in payload.get("roles", [])},
                permissions={Permission(perm) for perm in payload.get("permissions", [])}
            )
            
            expires_at = datetime.fromtimestamp(payload["exp"])
            
            return AuthenticationResult(
                authenticated=True,
                user=user,
                token=token,
                expires_at=expires_at,
                security_metadata={
                    "token_id": payload.get("jti"),
                    "issued_at": datetime.fromtimestamp(payload["iat"])
                }
            )
            
        except jwt.ExpiredSignatureError:
            return AuthenticationResult(
                authenticated=False,
                reason="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            return AuthenticationResult(
                authenticated=False,
                reason=f"Invalid token: {str(e)}"
            )
        except Exception as e:
            self.logger.error(f"Token validation error: {e}")
            return AuthenticationResult(
                authenticated=False,
                reason="Token validation failed"
            )
    
    def revoke_token(self, token: str) -> bool:
        """
        Revoke JWT token by adding to blacklist.
        
        Args:
            token: Token to revoke
            
        Returns:
            True if successfully revoked
        """
        try:
            # Validate token structure first
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                options={"verify_exp": False}  # Allow expired tokens to be revoked
            )
            
            # Add to blacklist
            self.blacklisted_tokens.add(token)
            
            self.logger.info(f"Revoked token for user {payload.get('username', 'unknown')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Token revocation error: {e}")
            return False


class APIKeyManager:
    """API key management for service-to-service authentication."""
    
    def __init__(self):
        """Initialize API key manager."""
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.logger = logger
        
        logger.info("Initialized API key manager")
    
    def generate_api_key(self, user_id: str, name: str, 
                        permissions: Set[Permission],
                        expires_at: Optional[datetime] = None) -> str:
        """
        Generate new API key.
        
        Args:
            user_id: User ID for the API key
            name: Human-readable name for the key
            permissions: Permissions granted to this key
            expires_at: Optional expiration time
            
        Returns:
            Generated API key
        """
        # Generate secure API key
        api_key = f"qr_{secrets.token_urlsafe(32)}"
        
        # Hash the key for storage
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Store key metadata
        self.api_keys[key_hash] = {
            "user_id": user_id,
            "name": name,
            "permissions": permissions,
            "created_at": datetime.utcnow(),
            "expires_at": expires_at,
            "last_used": None,
            "is_active": True,
            "usage_count": 0
        }
        
        self.logger.info(f"Generated API key '{name}' for user {user_id}")
        return api_key
    
    def validate_api_key(self, api_key: str) -> AuthenticationResult:
        """
        Validate API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            AuthenticationResult with validation results
        """
        try:
            # Hash the provided key
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            # Check if key exists
            if key_hash not in self.api_keys:
                return AuthenticationResult(
                    authenticated=False,
                    reason="Invalid API key"
                )
            
            key_info = self.api_keys[key_hash]
            
            # Check if key is active
            if not key_info["is_active"]:
                return AuthenticationResult(
                    authenticated=False,
                    reason="API key is disabled"
                )
            
            # Check expiration
            if key_info["expires_at"] and datetime.utcnow() > key_info["expires_at"]:
                return AuthenticationResult(
                    authenticated=False,
                    reason="API key has expired"
                )
            
            # Update usage statistics
            key_info["last_used"] = datetime.utcnow()
            key_info["usage_count"] += 1
            
            # Create user object from key info
            user = User(
                user_id=key_info["user_id"],
                username=f"api_key_{key_info['name']}",
                email="",
                roles={UserRole.SERVICE},
                permissions=key_info["permissions"]
            )
            
            return AuthenticationResult(
                authenticated=True,
                user=user,
                security_metadata={
                    "api_key_name": key_info["name"],
                    "usage_count": key_info["usage_count"]
                }
            )
            
        except Exception as e:
            self.logger.error(f"API key validation error: {e}")
            return AuthenticationResult(
                authenticated=False,
                reason="API key validation failed"
            )
    
    def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke API key.
        
        Args:
            api_key: API key to revoke
            
        Returns:
            True if successfully revoked
        """
        try:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            if key_hash in self.api_keys:
                self.api_keys[key_hash]["is_active"] = False
                self.logger.info(f"Revoked API key {self.api_keys[key_hash]['name']}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"API key revocation error: {e}")
            return False


class PermissionManager:
    """Role-based access control and permission management."""
    
    def __init__(self):
        """Initialize permission manager."""
        self.role_permissions = self._initialize_role_permissions()
        self.logger = logger
        
        logger.info("Initialized permission manager")
    
    def _initialize_role_permissions(self) -> Dict[UserRole, Set[Permission]]:
        """Initialize default role-permission mappings."""
        return {
            UserRole.READONLY: {
                Permission.SEARCH_QUERY,
                Permission.DATA_READ,
                Permission.MONITOR_VIEW
            },
            UserRole.USER: {
                Permission.SEARCH_QUERY,
                Permission.SEARCH_RERANK,
                Permission.QUANTUM_COMPUTE,
                Permission.DATA_READ,
                Permission.MONITOR_VIEW
            },
            UserRole.DEVELOPER: {
                Permission.SEARCH_QUERY,
                Permission.SEARCH_RERANK,
                Permission.SEARCH_INDEX,
                Permission.QUANTUM_COMPUTE,
                Permission.QUANTUM_OPTIMIZE,
                Permission.QUANTUM_ANALYZE,
                Permission.DATA_READ,
                Permission.DATA_WRITE,
                Permission.MONITOR_VIEW,
                Permission.MONITOR_ADMIN
            },
            UserRole.SERVICE: {
                Permission.SEARCH_QUERY,
                Permission.SEARCH_RERANK,
                Permission.QUANTUM_COMPUTE,
                Permission.DATA_READ
            },
            UserRole.ADMIN: set(Permission)  # All permissions
        }
    
    def get_user_permissions(self, user: User) -> Set[Permission]:
        """
        Get all permissions for a user based on roles.
        
        Args:
            user: User to get permissions for
            
        Returns:
            Set of permissions
        """
        permissions = set(user.permissions)  # Direct permissions
        
        # Add role-based permissions
        for role in user.roles:
            permissions.update(self.role_permissions.get(role, set()))
        
        return permissions
    
    def check_permission(self, user: User, required_permission: Permission) -> bool:
        """
        Check if user has required permission.
        
        Args:
            user: User to check
            required_permission: Permission to check for
            
        Returns:
            True if user has permission
        """
        user_permissions = self.get_user_permissions(user)
        return required_permission in user_permissions
    
    def add_role_permission(self, role: UserRole, permission: Permission) -> None:
        """Add permission to role."""
        if role not in self.role_permissions:
            self.role_permissions[role] = set()
        self.role_permissions[role].add(permission)
        
        self.logger.info(f"Added permission {permission.value} to role {role.value}")
    
    def remove_role_permission(self, role: UserRole, permission: Permission) -> None:
        """Remove permission from role."""
        if role in self.role_permissions:
            self.role_permissions[role].discard(permission)
            self.logger.info(f"Removed permission {permission.value} from role {role.value}")


class SessionManager:
    """User session management."""
    
    def __init__(self, session_timeout_minutes: int = 60):
        """
        Initialize session manager.
        
        Args:
            session_timeout_minutes: Session timeout in minutes
        """
        self.session_timeout_minutes = session_timeout_minutes
        self.sessions: Dict[str, Session] = {}
        self.logger = logger
        
        logger.info(f"Initialized session manager with {session_timeout_minutes}m timeout")
    
    def create_session(self, user_id: str, ip_address: Optional[str] = None,
                      user_agent: Optional[str] = None) -> str:
        """
        Create new user session.
        
        Args:
            user_id: User ID for session
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_accessed=now,
            expires_at=now + timedelta(minutes=self.session_timeout_minutes),
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session_id] = session
        
        self.logger.info(f"Created session {session_id} for user {user_id}")
        return session_id
    
    def validate_session(self, session_id: str) -> bool:
        """
        Validate session.
        
        Args:
            session_id: Session ID to validate
            
        Returns:
            True if session is valid
        """
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        # Check if session is active
        if not session.is_active:
            return False
        
        # Check if session has expired
        if datetime.utcnow() > session.expires_at:
            session.is_active = False
            return False
        
        # Update last accessed time
        session.last_accessed = datetime.utcnow()
        session.expires_at = session.last_accessed + timedelta(minutes=self.session_timeout_minutes)
        
        return True
    
    def destroy_session(self, session_id: str) -> bool:
        """
        Destroy session.
        
        Args:
            session_id: Session ID to destroy
            
        Returns:
            True if session was destroyed
        """
        if session_id in self.sessions:
            self.sessions[session_id].is_active = False
            del self.sessions[session_id]
            self.logger.info(f"Destroyed session {session_id}")
            return True
        return False


class QuantumRerankAuthFramework:
    """
    Comprehensive authentication and authorization framework.
    
    Integrates JWT tokens, API keys, RBAC, and session management
    for secure access control.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize authentication framework.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.token_manager = JWTTokenManager(
            secret_key=self.config.get("jwt_secret"),
            token_expiry_hours=self.config.get("token_expiry_hours", 24)
        )
        
        self.api_key_manager = APIKeyManager()
        self.permission_manager = PermissionManager()
        self.session_manager = SessionManager(
            session_timeout_minutes=self.config.get("session_timeout_minutes", 60)
        )
        
        self.logger = logger
        logger.info("Initialized QuantumRerank authentication framework")
    
    def authenticate_request(self, auth_header: Optional[str] = None,
                           api_key: Optional[str] = None,
                           session_id: Optional[str] = None) -> AuthenticationResult:
        """
        Authenticate request using multiple methods.
        
        Args:
            auth_header: Authorization header (Bearer token)
            api_key: API key for service authentication
            session_id: Session ID for session-based auth
            
        Returns:
            AuthenticationResult
        """
        # Try JWT token authentication
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            result = self.token_manager.validate_token(token)
            if result.authenticated:
                return result
        
        # Try API key authentication
        if api_key:
            result = self.api_key_manager.validate_api_key(api_key)
            if result.authenticated:
                return result
        
        # Try session authentication
        if session_id:
            if self.session_manager.validate_session(session_id):
                # Would need to look up user from session
                # For now, return a basic result
                return AuthenticationResult(
                    authenticated=True,
                    reason="Session authentication successful"
                )
        
        # Authentication failed
        return AuthenticationResult(
            authenticated=False,
            reason="No valid authentication provided"
        )
    
    def authorize_operation(self, user: User, required_permission: Permission) -> AuthorizationResult:
        """
        Authorize user operation.
        
        Args:
            user: User to authorize
            required_permission: Permission required for operation
            
        Returns:
            AuthorizationResult
        """
        user_permissions = self.permission_manager.get_user_permissions(user)
        authorized = required_permission in user_permissions
        
        return AuthorizationResult(
            authorized=authorized,
            required_permission=required_permission,
            user_permissions=user_permissions,
            reason=None if authorized else f"Missing permission: {required_permission.value}"
        )
    
    def create_user_token(self, user: User) -> str:
        """Create JWT token for user."""
        return self.token_manager.generate_token(user)
    
    def create_api_key(self, user_id: str, name: str, 
                      permissions: Set[Permission]) -> str:
        """Create API key for user."""
        return self.api_key_manager.generate_api_key(user_id, name, permissions)
    
    def logout_user(self, token: Optional[str] = None, 
                   session_id: Optional[str] = None) -> bool:
        """
        Logout user by revoking token and/or destroying session.
        
        Args:
            token: JWT token to revoke
            session_id: Session to destroy
            
        Returns:
            True if logout successful
        """
        success = True
        
        if token:
            success &= self.token_manager.revoke_token(token)
        
        if session_id:
            success &= self.session_manager.destroy_session(session_id)
        
        return success
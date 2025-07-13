"""
Authentication and authorization module for QuantumRerank API.
"""

from .authentication import (
    APIKeyAuthenticator,
    authenticate_request,
    get_current_user
)
from .authorization import (
    AuthorizationManager,
    check_permissions,
    require_permission
)
from .key_management import (
    APIKeyManager,
    generate_api_key,
    validate_api_key
)
from .security_utils import (
    hash_api_key,
    verify_api_key,
    generate_secure_token
)

__all__ = [
    "APIKeyAuthenticator",
    "authenticate_request", 
    "get_current_user",
    "AuthorizationManager",
    "check_permissions",
    "require_permission",
    "APIKeyManager",
    "generate_api_key",
    "validate_api_key",
    "hash_api_key",
    "verify_api_key",
    "generate_secure_token"
]
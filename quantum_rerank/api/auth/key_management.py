"""
API Key Management system for QuantumRerank API.

This module provides secure API key generation, validation, rotation,
and lifecycle management with usage tracking and analytics.
"""

import time
import secrets
import hashlib
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from ...utils.logging_config import get_logger
from .security_utils import hash_api_key, verify_api_key, generate_secure_token

logger = get_logger(__name__)


class KeyStatus(str, Enum):
    """API key status values."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"


@dataclass
class APIKey:
    """API key record with metadata."""
    key_id: str
    user_id: str
    key_hash: str
    name: str
    tier: str = "standard"
    status: KeyStatus = KeyStatus.ACTIVE
    permissions: List[str] = field(default_factory=list)
    rate_limits: Dict[str, int] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    last_used_at: Optional[float] = None
    usage_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if the key is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def is_active(self) -> bool:
        """Check if the key is active and usable."""
        return (
            self.status == KeyStatus.ACTIVE and
            not self.is_expired()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key_id": self.key_id,
            "user_id": self.user_id,
            "name": self.name,
            "tier": self.tier,
            "status": self.status.value,
            "permissions": self.permissions,
            "rate_limits": self.rate_limits,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "last_used_at": self.last_used_at,
            "usage_count": self.usage_count,
            "metadata": self.metadata,
            "is_expired": self.is_expired(),
            "is_active": self.is_active()
        }


class APIKeyManager:
    """
    Comprehensive API key management system.
    
    Handles key generation, validation, rotation, and lifecycle management
    with secure storage and usage analytics.
    """
    
    def __init__(self, storage_backend: str = "memory"):
        """
        Initialize API key manager.
        
        Args:
            storage_backend: Storage backend type ("memory", "file", "database")
        """
        self.storage_backend = storage_backend
        self.logger = logger
        
        # In-memory storage (for development)
        self.api_keys: Dict[str, APIKey] = {}
        self.key_hash_index: Dict[str, str] = {}  # hash -> key_id
        
        # Default rate limits by tier
        self.default_rate_limits = {
            "public": {
                "requests_per_minute": 60,
                "similarity_per_hour": 1000,
                "batch_size_limit": 10
            },
            "standard": {
                "requests_per_minute": 300,
                "similarity_per_hour": 10000,
                "batch_size_limit": 50
            },
            "premium": {
                "requests_per_minute": 1000,
                "similarity_per_hour": 50000,
                "batch_size_limit": 100
            },
            "enterprise": {
                "requests_per_minute": 10000,
                "similarity_per_hour": 500000,
                "batch_size_limit": 1000
            }
        }
        
        # Default permissions by tier
        self.default_permissions = {
            "public": ["read"],
            "standard": ["read", "compute"],
            "premium": ["read", "compute", "batch"],
            "enterprise": ["read", "compute", "batch", "admin"]
        }
    
    def generate_api_key(
        self,
        user_id: str,
        name: str,
        tier: str = "standard",
        expires_in_days: Optional[int] = None,
        custom_permissions: Optional[List[str]] = None,
        custom_rate_limits: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> tuple[str, APIKey]:
        """
        Generate a new API key.
        
        Args:
            user_id: User identifier
            name: Human-readable key name
            tier: User tier (public, standard, premium, enterprise)
            expires_in_days: Days until expiration (None for no expiration)
            custom_permissions: Custom permissions (overrides tier defaults)
            custom_rate_limits: Custom rate limits (overrides tier defaults)
            metadata: Additional metadata
            
        Returns:
            Tuple of (raw_key, api_key_record)
        """
        # Generate secure API key
        raw_key = generate_secure_token(length=32)
        key_hash = hash_api_key(raw_key)
        key_id = generate_secure_token(length=16)
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = time.time() + (expires_in_days * 24 * 3600)
        
        # Set permissions and rate limits
        permissions = custom_permissions or self.default_permissions.get(tier, ["read"])
        rate_limits = custom_rate_limits or self.default_rate_limits.get(tier, {})
        
        # Create API key record
        api_key = APIKey(
            key_id=key_id,
            user_id=user_id,
            key_hash=key_hash,
            name=name,
            tier=tier,
            permissions=permissions,
            rate_limits=rate_limits,
            expires_at=expires_at,
            metadata=metadata or {}
        )
        
        # Store the key
        self.api_keys[key_id] = api_key
        self.key_hash_index[key_hash] = key_id
        
        self.logger.info(
            "API key generated",
            extra={
                "key_id": key_id,
                "user_id": user_id,
                "tier": tier,
                "expires_at": expires_at
            }
        )
        
        return raw_key, api_key
    
    def validate_api_key(self, raw_key: str) -> Optional[APIKey]:
        """
        Validate an API key and return the key record.
        
        Args:
            raw_key: Raw API key string
            
        Returns:
            APIKey record if valid, None otherwise
        """
        try:
            # Hash the key for lookup
            key_hash = hash_api_key(raw_key)
            
            # Find key record
            key_id = self.key_hash_index.get(key_hash)
            if not key_id:
                return None
            
            api_key = self.api_keys.get(key_id)
            if not api_key:
                return None
            
            # Verify hash matches (additional security)
            if not verify_api_key(raw_key, api_key.key_hash):
                return None
            
            # Check if key is active
            if not api_key.is_active():
                self.logger.warning(
                    "Inactive API key used",
                    extra={
                        "key_id": key_id,
                        "status": api_key.status.value,
                        "is_expired": api_key.is_expired()
                    }
                )
                return None
            
            # Update usage statistics
            api_key.last_used_at = time.time()
            api_key.usage_count += 1
            
            return api_key
            
        except Exception as e:
            self.logger.error(f"API key validation error: {e}")
            return None
    
    def revoke_api_key(self, key_id: str, reason: str = "manual_revocation") -> bool:
        """
        Revoke an API key.
        
        Args:
            key_id: Key identifier to revoke
            reason: Reason for revocation
            
        Returns:
            True if revoked successfully
        """
        if key_id not in self.api_keys:
            return False
        
        api_key = self.api_keys[key_id]
        api_key.status = KeyStatus.REVOKED
        api_key.metadata["revoked_at"] = time.time()
        api_key.metadata["revocation_reason"] = reason
        
        # Remove from hash index
        self.key_hash_index.pop(api_key.key_hash, None)
        
        self.logger.info(
            "API key revoked",
            extra={
                "key_id": key_id,
                "user_id": api_key.user_id,
                "reason": reason
            }
        )
        
        return True
    
    def list_user_keys(self, user_id: str) -> List[Dict[str, Any]]:
        """
        List all API keys for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of key information (without sensitive data)
        """
        user_keys = []
        
        for api_key in self.api_keys.values():
            if api_key.user_id == user_id:
                key_info = api_key.to_dict()
                # Remove sensitive information
                key_info.pop("key_hash", None)
                user_keys.append(key_info)
        
        return user_keys
    
    def get_key_usage_stats(self, key_id: str) -> Optional[Dict[str, Any]]:
        """
        Get usage statistics for an API key.
        
        Args:
            key_id: Key identifier
            
        Returns:
            Usage statistics or None if key not found
        """
        if key_id not in self.api_keys:
            return None
        
        api_key = self.api_keys[key_id]
        
        # Calculate usage metrics
        age_days = (time.time() - api_key.created_at) / (24 * 3600)
        avg_daily_usage = api_key.usage_count / max(age_days, 1)
        
        last_used_days_ago = None
        if api_key.last_used_at:
            last_used_days_ago = (time.time() - api_key.last_used_at) / (24 * 3600)
        
        return {
            "key_id": key_id,
            "total_usage": api_key.usage_count,
            "age_days": age_days,
            "average_daily_usage": avg_daily_usage,
            "last_used_days_ago": last_used_days_ago,
            "status": api_key.status.value,
            "is_active": api_key.is_active(),
            "tier": api_key.tier,
            "permissions": api_key.permissions,
            "rate_limits": api_key.rate_limits
        }
    
    def cleanup_expired_keys(self) -> int:
        """
        Remove expired keys from storage.
        
        Returns:
            Number of keys cleaned up
        """
        expired_keys = []
        
        for key_id, api_key in self.api_keys.items():
            if api_key.is_expired() and api_key.status != KeyStatus.REVOKED:
                api_key.status = KeyStatus.EXPIRED
                expired_keys.append(key_id)
        
        # Remove from hash index
        for key_id in expired_keys:
            api_key = self.api_keys[key_id]
            self.key_hash_index.pop(api_key.key_hash, None)
        
        if expired_keys:
            self.logger.info(
                f"Cleaned up {len(expired_keys)} expired API keys",
                extra={"expired_key_ids": expired_keys}
            )
        
        return len(expired_keys)
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """
        Get API key manager statistics.
        
        Returns:
            Manager statistics and metrics
        """
        total_keys = len(self.api_keys)
        active_keys = len([k for k in self.api_keys.values() if k.is_active()])
        expired_keys = len([k for k in self.api_keys.values() if k.is_expired()])
        revoked_keys = len([k for k in self.api_keys.values() if k.status == KeyStatus.REVOKED])
        
        # Tier distribution
        tier_distribution = {}
        for api_key in self.api_keys.values():
            tier_distribution[api_key.tier] = tier_distribution.get(api_key.tier, 0) + 1
        
        # Usage statistics
        total_usage = sum(k.usage_count for k in self.api_keys.values())
        avg_usage_per_key = total_usage / max(total_keys, 1)
        
        return {
            "total_keys": total_keys,
            "active_keys": active_keys,
            "expired_keys": expired_keys,
            "revoked_keys": revoked_keys,
            "tier_distribution": tier_distribution,
            "total_usage": total_usage,
            "average_usage_per_key": avg_usage_per_key,
            "storage_backend": self.storage_backend
        }


# Global key manager instance
_key_manager: Optional[APIKeyManager] = None


def get_key_manager() -> APIKeyManager:
    """
    Get or create the global key manager instance.
    
    Returns:
        APIKeyManager instance
    """
    global _key_manager
    
    if _key_manager is None:
        _key_manager = APIKeyManager()
    
    return _key_manager


def generate_api_key(user_id: str, name: str, **kwargs) -> tuple[str, APIKey]:
    """
    Convenience function to generate an API key.
    
    Args:
        user_id: User identifier
        name: Key name
        **kwargs: Additional arguments for generate_api_key
        
    Returns:
        Tuple of (raw_key, api_key_record)
    """
    manager = get_key_manager()
    return manager.generate_api_key(user_id, name, **kwargs)


def validate_api_key(raw_key: str) -> Optional[APIKey]:
    """
    Convenience function to validate an API key.
    
    Args:
        raw_key: Raw API key string
        
    Returns:
        APIKey record if valid
    """
    manager = get_key_manager()
    return manager.validate_api_key(raw_key)
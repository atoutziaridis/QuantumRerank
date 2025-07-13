"""
Rate limiting middleware for QuantumRerank API.

This module implements token bucket algorithm for rate limiting with
configurable limits per user tier and endpoint-specific quotas.
"""

import time
import asyncio
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ...utils.logging_config import get_logger
from ...config.manager import ConfigManager

logger = get_logger(__name__)


class RateLimitType(str, Enum):
    """Types of rate limits."""
    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_HOUR = "requests_per_hour" 
    SIMILARITY_PER_HOUR = "similarity_per_hour"
    BATCH_SIZE_LIMIT = "batch_size_limit"


@dataclass
class TokenBucket:
    """Token bucket for rate limiting algorithm."""
    capacity: int  # Maximum tokens
    tokens: float  # Current tokens
    refill_rate: float  # Tokens per second
    last_refill: float  # Last refill timestamp
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if insufficient
        """
        now = time.time()
        
        # Refill bucket based on time elapsed
        time_passed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + (time_passed * self.refill_rate))
        self.last_refill = now
        
        # Try to consume tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False
    
    def time_until_available(self, tokens: int = 1) -> float:
        """
        Calculate time until enough tokens are available.
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Seconds until tokens available
        """
        if self.tokens >= tokens:
            return 0.0
        
        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


class RateLimitManager:
    """
    Comprehensive rate limiting manager with token bucket algorithm.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize rate limit manager.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.logger = logger
        
        # User buckets: user_id -> limit_type -> TokenBucket
        self.user_buckets: Dict[str, Dict[str, TokenBucket]] = defaultdict(dict)
        
        # Global buckets for IP-based limiting
        self.ip_buckets: Dict[str, Dict[str, TokenBucket]] = defaultdict(dict)
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
        
        # Rate limit configurations by tier
        self.tier_limits = {
            "public": {
                RateLimitType.REQUESTS_PER_MINUTE: 60,
                RateLimitType.REQUESTS_PER_HOUR: 1000,
                RateLimitType.SIMILARITY_PER_HOUR: 1000,
                RateLimitType.BATCH_SIZE_LIMIT: 10
            },
            "standard": {
                RateLimitType.REQUESTS_PER_MINUTE: 300,
                RateLimitType.REQUESTS_PER_HOUR: 10000,
                RateLimitType.SIMILARITY_PER_HOUR: 10000,
                RateLimitType.BATCH_SIZE_LIMIT: 50
            },
            "premium": {
                RateLimitType.REQUESTS_PER_MINUTE: 1000,
                RateLimitType.REQUESTS_PER_HOUR: 50000,
                RateLimitType.SIMILARITY_PER_HOUR: 50000,
                RateLimitType.BATCH_SIZE_LIMIT: 100
            },
            "enterprise": {
                RateLimitType.REQUESTS_PER_MINUTE: 10000,
                RateLimitType.REQUESTS_PER_HOUR: 500000,
                RateLimitType.SIMILARITY_PER_HOUR: 500000,
                RateLimitType.BATCH_SIZE_LIMIT: 1000
            }
        }
        
        # Endpoint-specific cost multipliers
        self.endpoint_costs = {
            "/v1/rerank": 2,  # Higher cost for quantum computation
            "/v1/similarity": 1,
            "/v1/batch-similarity": 3,  # Highest cost for batch processing
            "/v1/health": 0,  # No cost for health checks
            "/v1/metrics": 0
        }
    
    async def check_rate_limit(
        self,
        user_id: str,
        user_tier: str,
        client_ip: str,
        endpoint: str,
        request_size: int = 1
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limits.
        
        Args:
            user_id: User identifier
            user_tier: User tier (public, standard, premium, enterprise)
            client_ip: Client IP address
            endpoint: API endpoint being accessed
            request_size: Size/cost of the request
            
        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        async with self.lock:
            # Get rate limits for user tier
            tier_limits = self.tier_limits.get(user_tier, self.tier_limits["public"])
            
            # Get endpoint cost multiplier
            cost_multiplier = self.endpoint_costs.get(endpoint, 1)
            effective_cost = request_size * cost_multiplier
            
            # Skip rate limiting for zero-cost endpoints
            if cost_multiplier == 0:
                return True, {"exempt": True, "reason": "zero_cost_endpoint"}
            
            # Check requests per minute
            rpm_limit = tier_limits[RateLimitType.REQUESTS_PER_MINUTE]
            rpm_bucket = self._get_or_create_bucket(
                user_id, RateLimitType.REQUESTS_PER_MINUTE, rpm_limit, rpm_limit / 60.0
            )
            
            if not rpm_bucket.consume(effective_cost):
                retry_after = int(rpm_bucket.time_until_available(effective_cost)) + 1
                
                self.logger.warning(
                    "Rate limit exceeded",
                    extra={
                        "user_id": user_id,
                        "tier": user_tier,
                        "limit_type": "requests_per_minute",
                        "endpoint": endpoint,
                        "retry_after": retry_after
                    }
                )
                
                return False, {
                    "limit_type": "requests_per_minute",
                    "limit": rpm_limit,
                    "retry_after_seconds": retry_after,
                    "cost": effective_cost
                }
            
            # Check requests per hour
            rph_limit = tier_limits[RateLimitType.REQUESTS_PER_HOUR]
            rph_bucket = self._get_or_create_bucket(
                user_id, RateLimitType.REQUESTS_PER_HOUR, rph_limit, rph_limit / 3600.0
            )
            
            if not rph_bucket.consume(effective_cost):
                retry_after = int(rph_bucket.time_until_available(effective_cost)) + 1
                
                return False, {
                    "limit_type": "requests_per_hour",
                    "limit": rph_limit,
                    "retry_after_seconds": retry_after,
                    "cost": effective_cost
                }
            
            # All checks passed
            return True, {
                "allowed": True,
                "tier": user_tier,
                "cost": effective_cost,
                "remaining_rpm": int(rpm_bucket.tokens),
                "remaining_rph": int(rph_bucket.tokens)
            }
    
    def _get_or_create_bucket(
        self, 
        user_id: str, 
        limit_type: RateLimitType, 
        capacity: int, 
        refill_rate: float
    ) -> TokenBucket:
        """
        Get or create a token bucket for user and limit type.
        
        Args:
            user_id: User identifier
            limit_type: Type of rate limit
            capacity: Bucket capacity
            refill_rate: Refill rate in tokens per second
            
        Returns:
            TokenBucket instance
        """
        if limit_type not in self.user_buckets[user_id]:
            self.user_buckets[user_id][limit_type] = TokenBucket(
                capacity=capacity,
                tokens=capacity,  # Start with full bucket
                refill_rate=refill_rate,
                last_refill=time.time()
            )
        
        return self.user_buckets[user_id][limit_type]
    
    async def cleanup_expired_buckets(self, max_age_seconds: int = 3600) -> int:
        """
        Clean up old, unused token buckets.
        
        Args:
            max_age_seconds: Maximum age for keeping buckets
            
        Returns:
            Number of buckets cleaned up
        """
        async with self.lock:
            current_time = time.time()
            cleaned_count = 0
            
            # Clean user buckets
            expired_users = []
            for user_id, buckets in self.user_buckets.items():
                expired_buckets = []
                for limit_type, bucket in buckets.items():
                    if current_time - bucket.last_refill > max_age_seconds:
                        expired_buckets.append(limit_type)
                
                for limit_type in expired_buckets:
                    del buckets[limit_type]
                    cleaned_count += 1
                
                if not buckets:
                    expired_users.append(user_id)
            
            for user_id in expired_users:
                del self.user_buckets[user_id]
            
            self.logger.info(f"Cleaned up {cleaned_count} expired rate limit buckets")
            return cleaned_count
    
    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """
        Get rate limiting statistics.
        
        Returns:
            Dictionary with rate limiting statistics
        """
        stats = {
            "total_users": len(self.user_buckets),
            "total_buckets": sum(len(buckets) for buckets in self.user_buckets.values()),
            "tier_limits": self.tier_limits,
            "endpoint_costs": self.endpoint_costs
        }
        
        # User bucket details
        user_stats = {}
        for user_id, buckets in self.user_buckets.items():
            user_stats[user_id] = {
                limit_type: {
                    "capacity": bucket.capacity,
                    "current_tokens": int(bucket.tokens),
                    "refill_rate": bucket.refill_rate,
                    "last_refill": bucket.last_refill
                }
                for limit_type, bucket in buckets.items()
            }
        
        stats["user_buckets"] = user_stats
        return stats


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting requests.
    """
    
    def __init__(self, app, config_manager: ConfigManager):
        """
        Initialize rate limiting middleware.
        
        Args:
            app: FastAPI application
            config_manager: Configuration manager
        """
        super().__init__(app)
        self.config_manager = config_manager
        self.rate_limit_manager = RateLimitManager(config_manager)
        self.logger = logger
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request with rate limiting.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware or endpoint handler
            
        Returns:
            HTTP response or rate limit error
        """
        # Check if rate limiting is enabled
        config = self.config_manager.get_config()
        if not getattr(config.api, 'rate_limiting_enabled', True):
            return await call_next(request)
        
        # Extract user information from request
        user_info = self._extract_user_info(request)
        
        # Determine request size/cost
        request_cost = self._calculate_request_cost(request)
        
        # Check rate limits
        allowed, rate_info = await self.rate_limit_manager.check_rate_limit(
            user_id=user_info["user_id"],
            user_tier=user_info["tier"],
            client_ip=user_info["client_ip"],
            endpoint=str(request.url.path),
            request_size=request_cost
        )
        
        if not allowed:
            # Log rate limit violation
            self.logger.warning(
                "Rate limit exceeded",
                extra={
                    "user_id": user_info["user_id"],
                    "client_ip": user_info["client_ip"],
                    "endpoint": request.url.path,
                    "rate_info": rate_info
                }
            )
            
            # Return rate limit error
            return self._create_rate_limit_response(rate_info)
        
        # Add rate limit info to response headers
        response = await call_next(request)
        
        if "remaining_rpm" in rate_info:
            response.headers["X-RateLimit-Remaining-Minute"] = str(rate_info["remaining_rpm"])
        if "remaining_rph" in rate_info:
            response.headers["X-RateLimit-Remaining-Hour"] = str(rate_info["remaining_rph"])
        
        response.headers["X-RateLimit-Tier"] = user_info["tier"]
        
        return response
    
    def _extract_user_info(self, request: Request) -> Dict[str, str]:
        """
        Extract user information from request.
        
        Args:
            request: HTTP request
            
        Returns:
            Dictionary with user information
        """
        # Try to get user from authentication context
        # For now, use default values (would be improved with proper auth integration)
        
        client_ip = getattr(request.client, 'host', '127.0.0.1')
        
        # Check for API key in headers
        api_key = request.headers.get("X-API-Key")
        if api_key:
            # In production, would look up user tier from API key
            user_id = f"user_{api_key[:8]}"
            tier = "standard"  # Default tier
        else:
            # Anonymous user
            user_id = f"anon_{client_ip}"
            tier = "public"
        
        return {
            "user_id": user_id,
            "tier": tier,
            "client_ip": client_ip
        }
    
    def _calculate_request_cost(self, request: Request) -> int:
        """
        Calculate the cost/weight of a request.
        
        Args:
            request: HTTP request
            
        Returns:
            Request cost multiplier
        """
        # Base cost is 1
        cost = 1
        
        # Increase cost for batch requests
        if "batch" in str(request.url.path):
            # Would check actual batch size from request body
            cost *= 2
        
        # Increase cost for quantum methods
        # Would check request body for method parameter
        
        return cost
    
    def _create_rate_limit_response(self, rate_info: Dict[str, Any]) -> JSONResponse:
        """
        Create rate limit exceeded response.
        
        Args:
            rate_info: Rate limit information
            
        Returns:
            JSON response with rate limit error
        """
        error_response = {
            "error": {
                "code": "RATE_LIMIT_EXCEEDED",
                "message": "Request rate limit exceeded",
                "details": {
                    "limit_type": rate_info.get("limit_type"),
                    "limit": rate_info.get("limit"),
                    "retry_after_seconds": rate_info.get("retry_after_seconds"),
                    "cost": rate_info.get("cost", 1),
                    "suggestion": "Wait before making additional requests or upgrade your tier"
                },
                "timestamp": time.time()
            }
        }
        
        headers = {}
        if "retry_after_seconds" in rate_info:
            headers["Retry-After"] = str(rate_info["retry_after_seconds"])
        
        return JSONResponse(
            status_code=429,
            content=error_response,
            headers=headers
        )


# Global rate limit manager instance
_rate_limit_manager: Optional[RateLimitManager] = None


def get_rate_limit_manager(config_manager: ConfigManager) -> RateLimitManager:
    """
    Get or create global rate limit manager.
    
    Args:
        config_manager: Configuration manager
        
    Returns:
        RateLimitManager instance
    """
    global _rate_limit_manager
    
    if _rate_limit_manager is None:
        _rate_limit_manager = RateLimitManager(config_manager)
    
    return _rate_limit_manager
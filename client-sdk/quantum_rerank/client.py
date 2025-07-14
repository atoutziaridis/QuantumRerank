"""
QuantumRerank API Client

This module provides the main Client class for interacting with the QuantumRerank API.
"""

import requests
import time
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin

from .models import RerankResponse, HealthStatus
from .exceptions import (
    QuantumRerankError, 
    AuthenticationError, 
    RateLimitError, 
    ValidationError,
    ServiceUnavailableError
)


class Client:
    """
    QuantumRerank API client for semantic similarity and document reranking.
    
    This client provides a simple interface to the QuantumRerank API, handling
    authentication, request formatting, error handling, and response parsing.
    
    Args:
        api_key: Your QuantumRerank API key
        base_url: Base URL for the API (default: http://localhost:8000)
        timeout: Request timeout in seconds (default: 30)
        max_retries: Maximum number of retry attempts (default: 3)
    
    Example:
        >>> client = Client(api_key="your-api-key")
        >>> result = client.rerank(
        ...     query="What is machine learning?",
        ...     documents=["ML is AI subset", "Python is a language"]
        ... )
        >>> print(f"Top result: {result.documents[0].text}")
    """
    
    def __init__(
        self, 
        api_key: str,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        max_retries: int = 3
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Configure session
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "quantum-rerank-client/1.0.0"
        })
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        method: str = "hybrid"
    ) -> RerankResponse:
        """
        Rerank documents based on their similarity to the query.
        
        Args:
            query: Query text to compare against documents
            documents: List of document texts to rerank
            top_k: Number of top results to return (default: all documents)
            method: Similarity method to use ("classical", "quantum", "hybrid")
        
        Returns:
            RerankResponse containing ranked documents with scores
            
        Raises:
            ValidationError: If request parameters are invalid
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit is exceeded
            QuantumRerankError: For other API errors
        """
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty")
        
        if not documents:
            raise ValidationError("At least one document is required")
        
        if top_k is not None and top_k <= 0:
            raise ValidationError("top_k must be greater than 0")
        
        if top_k is not None and top_k > len(documents):
            top_k = len(documents)
        
        # Prepare request payload (matching API exactly)
        payload = {
            "query": query,
            "candidates": documents,  # API uses 'candidates', not 'documents'
            "method": method
        }
        
        if top_k is not None:
            payload["top_k"] = top_k
        
        # Make API request
        response_data = self._make_request("POST", "/v1/rerank", json=payload)
        
        # Parse response
        return RerankResponse.from_api_response(response_data, query)
    
    def health(self) -> HealthStatus:
        """
        Check the health status of the QuantumRerank API.
        
        Returns:
            HealthStatus with service status information
            
        Raises:
            QuantumRerankError: If health check fails
        """
        response_data = self._make_request("GET", "/health")
        return HealthStatus.from_api_response(response_data)
    
    def get_similarity_methods(self) -> Dict[str, Any]:
        """
        Get information about available similarity methods.
        
        Returns:
            Dictionary with method information and recommendations
        """
        return self._make_request("GET", "/v1/rerank/methods")
    
    def get_limits(self) -> Dict[str, Any]:
        """
        Get current API limits and constraints.
        
        Returns:
            Dictionary with rate limits and request constraints
        """
        return self._make_request("GET", "/v1/rerank/limits")
    
    def validate_request(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        method: str = "hybrid"
    ) -> Dict[str, Any]:
        """
        Validate a rerank request without executing it.
        
        Args:
            query: Query text
            documents: List of documents
            top_k: Number of top results
            method: Similarity method
            
        Returns:
            Validation result with errors and suggestions
        """
        payload = {
            "query": query,
            "candidates": documents,
            "method": method
        }
        
        if top_k is not None:
            payload["top_k"] = top_k
        
        return self._make_request("POST", "/v1/rerank/validate", json=payload)
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the API with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments for requests
            
        Returns:
            Response JSON data
            
        Raises:
            Various QuantumRerankError subclasses based on response
        """
        url = urljoin(self.base_url, endpoint)
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    timeout=self.timeout,
                    **kwargs
                )
                
                # Handle specific status codes
                if response.status_code == 401:
                    raise AuthenticationError("Invalid API key")
                
                elif response.status_code == 403:
                    raise AuthenticationError("Access forbidden. Check your API key permissions.")
                
                elif response.status_code == 422:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("detail", "Validation error")
                        raise ValidationError(str(error_msg))
                    except (ValueError, KeyError):
                        raise ValidationError("Request validation failed")
                
                elif response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    raise RateLimitError(
                        f"Rate limit exceeded. Retry after {retry_after} seconds",
                        retry_after=retry_after
                    )
                
                elif response.status_code == 503:
                    if attempt < self.max_retries - 1:
                        # Retry on service unavailable
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        continue
                    else:
                        raise ServiceUnavailableError()
                
                elif response.status_code >= 500:
                    if attempt < self.max_retries - 1:
                        # Retry on server errors
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        continue
                    else:
                        raise QuantumRerankError(
                            f"Server error: {response.status_code}",
                            status_code=response.status_code
                        )
                
                # Raise for other HTTP errors
                response.raise_for_status()
                
                # Return JSON response
                try:
                    return response.json()
                except ValueError as e:
                    raise QuantumRerankError(f"Invalid JSON response: {e}")
                
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                else:
                    raise QuantumRerankError(
                        f"Request timeout after {self.timeout} seconds"
                    )
            
            except requests.exceptions.ConnectionError:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                else:
                    raise QuantumRerankError(
                        f"Connection error. Could not reach {self.base_url}"
                    )
            
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                else:
                    raise QuantumRerankError(f"Request failed: {e}")
        
        # This should never be reached due to the loop structure
        raise QuantumRerankError("Max retries exceeded")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close session."""
        self.session.close()
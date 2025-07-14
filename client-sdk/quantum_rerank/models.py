"""
Data models for QuantumRerank client.

This module defines data classes that mirror the API request/response models
for type safety and ease of use.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class Document:
    """A document with its similarity score and metadata."""
    text: str
    score: float
    rank: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @classmethod
    def from_api_response(cls, result_data: dict) -> "Document":
        """Create Document from API response data."""
        return cls(
            text=result_data["text"],
            score=result_data["similarity_score"],
            rank=result_data["rank"],
            metadata=result_data.get("metadata", {})
        )


@dataclass 
class RerankResponse:
    """Response from the rerank API endpoint."""
    documents: List[Document]
    query: str
    method: str
    processing_time_ms: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @classmethod
    def from_api_response(cls, response_data: dict, original_query: str) -> "RerankResponse":
        """Create RerankResponse from API response data."""
        documents = [
            Document.from_api_response(result) 
            for result in response_data["results"]
        ]
        
        return cls(
            documents=documents,
            query=original_query,
            method=response_data["method_used"],
            processing_time_ms=response_data["computation_time_ms"],
            metadata=response_data.get("query_metadata", {})
        )


@dataclass
class HealthStatus:
    """Health status response from the API."""
    status: str
    timestamp: str
    version: str
    components: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.components is None:
            self.components = {}
    
    @property
    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        return self.status.lower() in ["healthy", "ok"]
    
    @classmethod
    def from_api_response(cls, response_data: dict) -> "HealthStatus":
        """Create HealthStatus from API response data."""
        return cls(
            status=response_data["status"],
            timestamp=response_data.get("timestamp", ""),
            version=response_data.get("version", "unknown"),
            components=response_data.get("components", {})
        )
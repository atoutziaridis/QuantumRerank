"""
API models and schemas for QuantumRerank FastAPI service.

This module defines Pydantic models for request validation, response formatting,
and data transfer objects used throughout the API endpoints.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator, conlist
from enum import Enum
from datetime import datetime


class SimilarityMethod(str, Enum):
    """Available similarity computation methods."""
    CLASSICAL = "classical"
    QUANTUM = "quantum"
    HYBRID = "hybrid"


class RerankRequest(BaseModel):
    """
    Request model for document reranking.
    
    Accepts a query and multiple candidate documents for quantum-enhanced reranking.
    """
    query: str = Field(
        ...,
        description="Query text to compare against candidates",
        min_length=1,
        max_length=10000  # Extended limit as per Task 28
    )
    candidates: List[str] = Field(
        ...,
        description="List of candidate documents to rerank",
        min_items=1,
        max_items=1000  # Extended to 1000 as per Task 28
    )
    top_k: Optional[int] = Field(
        default=10,
        description="Number of top results to return",
        ge=1,
        le=100
    )
    method: Optional[SimilarityMethod] = Field(
        default=SimilarityMethod.HYBRID,
        description="Similarity computation method"
    )
    user_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional user context for personalization"
    )
    
    @validator('candidates')
    def validate_candidates(cls, v):
        """Ensure all candidates are non-empty strings with proper length limits."""
        for i, candidate in enumerate(v):
            if not candidate or not candidate.strip():
                raise ValueError(f"Candidate at index {i} is empty")
            if len(candidate) > 50000:  # 50K character limit per document
                raise ValueError(f"Candidate at index {i} exceeds maximum length (50000 characters)")
        return [candidate.strip() for candidate in v]
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query string with enhanced checks."""
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        if len(v.strip()) > 10000:
            raise ValueError('Query too long (max 10000 characters)')
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "query": "quantum computing applications in machine learning",
                "candidates": [
                    "Quantum algorithms for optimization problems",
                    "Classical ML approaches to data analysis",
                    "Hybrid quantum-classical machine learning"
                ],
                "top_k": 2,
                "method": "hybrid",
                "user_context": {"domain": "research", "preference": "recent"}
            }
        }


class SimilarityRequest(BaseModel):
    """
    Request model for direct similarity computation.
    
    Computes similarity between two text inputs using specified method.
    """
    text1: str = Field(
        ...,
        description="First text for similarity comparison",
        min_length=1,
        max_length=5000
    )
    text2: str = Field(
        ...,
        description="Second text for similarity comparison",
        min_length=1,
        max_length=5000
    )
    method: Optional[SimilarityMethod] = Field(
        default=SimilarityMethod.HYBRID,
        description="Similarity computation method"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "text1": "Quantum computing fundamentals",
                "text2": "Basics of quantum mechanics in computing",
                "method": "quantum"
            }
        }


class BatchSimilarityRequest(BaseModel):
    """
    Request model for batch similarity computation.
    
    Efficiently processes multiple query-candidate pairs.
    """
    query: str = Field(
        ...,
        description="Query text to compare against all candidates",
        min_length=1,
        max_length=5000
    )
    candidates: List[str] = Field(
        ...,
        description="List of candidate texts for batch similarity",
        min_items=1,
        max_items=100
    )
    method: Optional[SimilarityMethod] = Field(
        default=SimilarityMethod.HYBRID,
        description="Similarity computation method"
    )
    return_all: Optional[bool] = Field(
        default=False,
        description="Return all results or only top matches"
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Minimum similarity threshold",
        ge=0.0,
        le=1.0
    )


class RankedResult(BaseModel):
    """Individual ranked result with metadata."""
    text: str = Field(..., description="The candidate text")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    rank: int = Field(..., description="Rank position (1-based)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the computation"
    )


class RerankResponse(BaseModel):
    """
    Response model for document reranking.
    
    Contains ranked results with scores and computation metadata.
    """
    results: List[RankedResult] = Field(
        ...,
        description="Ranked list of documents with scores"
    )
    query_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the query processing"
    )
    computation_time_ms: float = Field(
        ...,
        description="Total computation time in milliseconds"
    )
    method_used: SimilarityMethod = Field(
        ...,
        description="Actual method used for computation"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {
                        "text": "Hybrid quantum-classical machine learning",
                        "similarity_score": 0.92,
                        "rank": 1,
                        "metadata": {
                            "classical_score": 0.88,
                            "quantum_score": 0.95,
                            "weight_classical": 0.5,
                            "weight_quantum": 0.5
                        }
                    }
                ],
                "query_metadata": {
                    "total_candidates": 3,
                    "embedding_model": "all-MiniLM-L6-v2",
                    "quantum_backend": "statevector_simulator"
                },
                "computation_time_ms": 156.7,
                "method_used": "hybrid"
            }
        }


class SimilarityResponse(BaseModel):
    """
    Response model for direct similarity computation.
    
    Contains similarity score and detailed computation metadata.
    """
    similarity_score: float = Field(
        ...,
        description="Computed similarity score (0-1)",
        ge=0.0,
        le=1.0
    )
    method_used: SimilarityMethod = Field(
        ...,
        description="Method used for computation"
    )
    computation_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed breakdown of computation"
    )
    computation_time_ms: float = Field(
        ...,
        description="Computation time in milliseconds"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "similarity_score": 0.87,
                "method_used": "quantum",
                "computation_details": {
                    "quantum_fidelity": 0.87,
                    "circuit_depth": 15,
                    "num_qubits": 4,
                    "backend": "statevector_simulator"
                },
                "computation_time_ms": 45.3
            }
        }


class BatchSimilarityResponse(BaseModel):
    """Response model for batch similarity computation."""
    results: List[Dict[str, Any]] = Field(
        ...,
        description="List of similarity results for each candidate"
    )
    total_processed: int = Field(
        ...,
        description="Total number of candidates processed"
    )
    computation_time_ms: float = Field(
        ...,
        description="Total computation time"
    )
    method_used: SimilarityMethod = Field(
        ...,
        description="Method used for computation"
    )


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoints."""
    status: str = Field(..., description="Service health status")
    timestamp: datetime = Field(..., description="Current server time")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(
        default_factory=dict,
        description="Status of individual components"
    )


class DetailedHealthResponse(BaseModel):
    """Detailed health check response with performance metrics."""
    status: str = Field(..., description="Overall service status")
    timestamp: datetime = Field(..., description="Current server time")
    version: str = Field(..., description="API version")
    components: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Detailed status of all components"
    )
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Current performance metrics"
    )
    resource_usage: Dict[str, Any] = Field(
        default_factory=dict,
        description="System resource usage"
    )


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: Dict[str, Any] = Field(
        ...,
        description="Error details"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid request parameters",
                    "details": {
                        "field": "candidates",
                        "reason": "Maximum 100 candidates allowed"
                    },
                    "timestamp": "2024-01-15T10:30:00Z"
                }
            }
        }


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint."""
    request_count: int = Field(..., description="Total requests processed")
    average_response_time_ms: float = Field(..., description="Average response time")
    method_usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Usage count by method"
    )
    error_rate: float = Field(..., description="Error rate percentage")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    
    
# Request/Response type aliases for easier imports
RequestModels = Union[RerankRequest, SimilarityRequest, BatchSimilarityRequest]
ResponseModels = Union[
    RerankResponse, 
    SimilarityResponse, 
    BatchSimilarityResponse,
    HealthCheckResponse,
    DetailedHealthResponse,
    ErrorResponse,
    MetricsResponse
]
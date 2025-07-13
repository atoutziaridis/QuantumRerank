"""
Document reranking endpoint for QuantumRerank API.

This module implements the main reranking functionality endpoint that accepts
a query and candidate documents and returns them ranked by quantum-enhanced similarity.
"""

import asyncio
import time
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from ..models import RerankRequest, RerankResponse, ErrorResponse
from ..dependencies import get_core_services, get_request_context, RequestContext
from ..services.similarity_service import SimilarityService
from ...utils.logging_config import get_logger

logger = get_logger(__name__)

# Create router for reranking endpoints
router = APIRouter()

# Initialize similarity service (will be properly injected via dependency)
_similarity_service = None


def get_similarity_service(
    services: Dict[str, Any] = Depends(get_core_services)
) -> SimilarityService:
    """
    Get or create similarity service instance.
    
    Args:
        services: Core services dictionary
        
    Returns:
        SimilarityService instance
    """
    global _similarity_service
    
    if _similarity_service is None:
        _similarity_service = SimilarityService(
            quantum_reranker=services["quantum_reranker"],
            performance_monitor=services["performance_monitor"],
            max_workers=4
        )
    
    return _similarity_service


@router.post(
    "/rerank",
    response_model=RerankResponse,
    responses={
        200: {"description": "Successful reranking"},
        400: {"description": "Invalid request", "model": ErrorResponse},
        422: {"description": "Validation error", "model": ErrorResponse},
        429: {"description": "Rate limit exceeded", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    },
    summary="Rerank documents using quantum similarity",
    description="""
    Rerank a list of candidate documents based on their quantum-enhanced similarity 
    to a query text. This endpoint supports multiple similarity methods and returns
    ranked results with detailed metadata.
    
    **Features:**
    - Quantum, classical, and hybrid similarity methods
    - Configurable top-k results
    - Performance timing and metadata
    - User context support for personalization
    
    **Performance:**
    - Target response time: <500ms
    - Supports up to 100 candidate documents
    - Batch processing for efficiency
    """
)
async def rerank_documents(
    request: RerankRequest,
    similarity_service: SimilarityService = Depends(get_similarity_service),
    request_context: RequestContext = Depends(get_request_context)
) -> RerankResponse:
    """
    Rerank documents using quantum-enhanced similarity.
    
    Args:
        request: Reranking request with query and candidates
        similarity_service: Injected similarity service
        request_context: Request context for logging and tracking
        
    Returns:
        RerankResponse with ranked documents and metadata
        
    Raises:
        HTTPException: For various error conditions
    """
    async with request_context:
        request_context.logger.info(
            "Processing rerank request",
            extra={
                "query_length": len(request.query),
                "candidate_count": len(request.candidates),
                "method": request.method.value,
                "top_k": request.top_k
            }
        )
        
        try:
            # Validate request parameters
            if len(request.candidates) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="At least one candidate document is required"
                )
            
            if request.top_k > len(request.candidates):
                request_context.logger.warning(
                    f"top_k ({request.top_k}) is greater than candidate count ({len(request.candidates)}), "
                    f"adjusting to {len(request.candidates)}"
                )
                request.top_k = len(request.candidates)
            
            # Execute reranking
            start_time = time.perf_counter()
            
            result = await similarity_service.rerank_documents(
                query=request.query,
                candidates=request.candidates,
                top_k=request.top_k,
                method=request.method,
                user_context=request.user_context,
                request_id=request_context.request_id
            )
            
            processing_time = time.perf_counter() - start_time
            
            # Create response
            response = RerankResponse(
                results=result["results"],
                query_metadata=result["query_metadata"],
                computation_time_ms=result["computation_time_ms"],
                method_used=result["method_used"]
            )
            
            # Log successful completion
            request_context.logger.info(
                "Rerank request completed successfully",
                extra={
                    "results_count": len(response.results),
                    "computation_time_ms": response.computation_time_ms,
                    "method_used": response.method_used.value
                }
            )
            
            # Add performance warning if needed
            if response.computation_time_ms > 500:  # PRD target: <500ms
                request_context.logger.warning(
                    "Rerank request exceeded target response time",
                    extra={
                        "computation_time_ms": response.computation_time_ms,
                        "target_ms": 500
                    }
                )
            
            return response
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
            
        except Exception as e:
            # Log error with context
            request_context.logger.error(
                "Rerank request failed with unexpected error",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            
            # Raise internal server error
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "RERANK_PROCESSING_ERROR",
                    "message": "Failed to process reranking request",
                    "details": {
                        "error_type": type(e).__name__,
                        "request_id": request_context.request_id
                    }
                }
            )


@router.get(
    "/rerank/methods",
    summary="Get available similarity methods",
    description="List all available similarity computation methods with descriptions"
)
async def get_similarity_methods():
    """
    Get available similarity methods.
    
    Returns:
        Dictionary of available methods with descriptions
    """
    return {
        "methods": {
            "classical": {
                "name": "Classical Cosine Similarity",
                "description": "Traditional cosine similarity using embeddings",
                "performance": "Fast, reliable baseline method",
                "use_case": "General purpose similarity computation"
            },
            "quantum": {
                "name": "Quantum Fidelity Similarity", 
                "description": "Quantum fidelity-based similarity using quantum circuits",
                "performance": "Higher accuracy, moderate computational cost",
                "use_case": "High-precision similarity for critical applications"
            },
            "hybrid": {
                "name": "Hybrid Quantum-Classical",
                "description": "Weighted combination of quantum and classical methods",
                "performance": "Balanced accuracy and speed",
                "use_case": "Recommended for most applications"
            }
        },
        "default_method": "hybrid",
        "recommended_method": "hybrid"
    }


@router.get(
    "/rerank/limits",
    summary="Get request limits and constraints",
    description="Get current API limits for reranking requests"
)
async def get_rerank_limits(
    services: Dict[str, Any] = Depends(get_core_services)
):
    """
    Get current request limits and constraints.
    
    Args:
        services: Core services for configuration access
        
    Returns:
        Dictionary with current limits
    """
    try:
        config = services["config_manager"].get_config()
        
        return {
            "limits": {
                "max_candidates": 100,  # PRD specification
                "max_query_length": 5000,
                "max_candidate_length": 5000,
                "max_top_k": 100,
                "target_response_time_ms": 500
            },
            "rate_limits": {
                "requests_per_minute": config.api.rate_limit_per_minute,
                "max_request_size_mb": config.api.max_request_size / (1024 * 1024)
            },
            "performance_targets": {
                "avg_response_time_ms": 200,
                "max_response_time_ms": 500,
                "success_rate_percent": 99.5
            }
        }
        
    except Exception as e:
        logger.warning(f"Failed to get limits from config: {e}")
        
        # Return default limits
        return {
            "limits": {
                "max_candidates": 100,
                "max_query_length": 5000,
                "max_candidate_length": 5000,
                "max_top_k": 100,
                "target_response_time_ms": 500
            },
            "rate_limits": {
                "requests_per_minute": 100,
                "max_request_size_mb": 10
            },
            "performance_targets": {
                "avg_response_time_ms": 200,
                "max_response_time_ms": 500,
                "success_rate_percent": 99.5
            }
        }


@router.post(
    "/rerank/validate",
    summary="Validate rerank request without processing",
    description="Validate a rerank request format and parameters without executing the reranking"
)
async def validate_rerank_request(
    request: RerankRequest,
    services: Dict[str, Any] = Depends(get_core_services)
):
    """
    Validate a rerank request without processing.
    
    Args:
        request: Reranking request to validate
        services: Core services for configuration access
        
    Returns:
        Validation result with details
    """
    validation_result = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "suggestions": []
    }
    
    try:
        config = services["config_manager"].get_config()
        
        # Validate query
        if len(request.query.strip()) == 0:
            validation_result["errors"].append("Query cannot be empty")
            validation_result["valid"] = False
        
        if len(request.query) > 5000:
            validation_result["errors"].append("Query exceeds maximum length of 5000 characters")
            validation_result["valid"] = False
        
        # Validate candidates
        if len(request.candidates) == 0:
            validation_result["errors"].append("At least one candidate is required")
            validation_result["valid"] = False
        
        if len(request.candidates) > 100:
            validation_result["errors"].append("Maximum 100 candidates allowed")
            validation_result["valid"] = False
        
        # Check individual candidates
        for i, candidate in enumerate(request.candidates):
            if len(candidate.strip()) == 0:
                validation_result["errors"].append(f"Candidate {i} is empty")
                validation_result["valid"] = False
            
            if len(candidate) > 5000:
                validation_result["errors"].append(f"Candidate {i} exceeds maximum length")
                validation_result["valid"] = False
        
        # Validate top_k
        if request.top_k <= 0:
            validation_result["errors"].append("top_k must be greater than 0")
            validation_result["valid"] = False
        
        if request.top_k > len(request.candidates):
            validation_result["warnings"].append(
                f"top_k ({request.top_k}) is greater than candidate count ({len(request.candidates)})"
            )
            validation_result["suggestions"].append("Consider reducing top_k to candidate count")
        
        # Performance suggestions
        if len(request.candidates) > 50:
            validation_result["suggestions"].append(
                "Large candidate sets may increase response time. Consider batch processing."
            )
        
        if request.method.value == "quantum" and len(request.candidates) > 20:
            validation_result["suggestions"].append(
                "Quantum method with many candidates may be slow. Consider hybrid method."
            )
        
        return validation_result
        
    except Exception as e:
        return {
            "valid": False,
            "errors": [f"Validation error: {str(e)}"],
            "warnings": [],
            "suggestions": []
        }
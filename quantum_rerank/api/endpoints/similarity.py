"""
Direct similarity computation endpoint for QuantumRerank API.

This module implements the direct similarity endpoint that computes similarity
between two texts using quantum-enhanced methods.
"""

import time
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from ..models import SimilarityRequest, SimilarityResponse, ErrorResponse
from ..dependencies import get_core_services, get_request_context, RequestContext
from ..services.similarity_service import SimilarityService
from ...utils.logging_config import get_logger

logger = get_logger(__name__)

# Create router for similarity endpoints
router = APIRouter()

# Reuse similarity service from rerank module
def get_similarity_service(
    services: Dict[str, Any] = Depends(get_core_services)
) -> SimilarityService:
    """Get similarity service instance."""
    from .rerank import get_similarity_service as get_rerank_service
    return get_rerank_service(services)


@router.post(
    "/similarity",
    response_model=SimilarityResponse,
    responses={
        200: {"description": "Successful similarity computation"},
        400: {"description": "Invalid request", "model": ErrorResponse},
        422: {"description": "Validation error", "model": ErrorResponse},
        429: {"description": "Rate limit exceeded", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    },
    summary="Compute direct similarity between two texts",
    description="""
    Compute similarity score between two text inputs using quantum-enhanced methods.
    
    **Features:**
    - Support for quantum, classical, and hybrid methods
    - Detailed computation metadata
    - Performance timing information
    - Method-specific score breakdowns
    
    **Use Cases:**
    - A/B testing different similarity methods
    - Research and analysis workflows
    - Similarity threshold validation
    - Direct text comparison
    """
)
async def compute_similarity(
    request: SimilarityRequest,
    similarity_service: SimilarityService = Depends(get_similarity_service),
    request_context: RequestContext = Depends(get_request_context)
) -> SimilarityResponse:
    """
    Compute similarity between two texts.
    
    Args:
        request: Similarity request with two texts
        similarity_service: Injected similarity service
        request_context: Request context for logging
        
    Returns:
        SimilarityResponse with score and metadata
    """
    async with request_context:
        request_context.logger.info(
            "Processing similarity request",
            extra={
                "text1_length": len(request.text1),
                "text2_length": len(request.text2),
                "method": request.method.value
            }
        )
        
        try:
            # Validate input texts
            if not request.text1.strip() or not request.text2.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Both text inputs must be non-empty"
                )
            
            # Execute similarity computation
            start_time = time.perf_counter()
            
            result = await similarity_service.compute_similarity(
                text1=request.text1,
                text2=request.text2,
                method=request.method,
                request_id=request_context.request_id
            )
            
            processing_time = time.perf_counter() - start_time
            
            # Create response
            response = SimilarityResponse(
                similarity_score=result["similarity_score"],
                method_used=result["method_used"],
                computation_details=result["computation_details"],
                computation_time_ms=result["computation_time_ms"]
            )
            
            # Log successful completion
            request_context.logger.info(
                "Similarity request completed successfully",
                extra={
                    "similarity_score": response.similarity_score,
                    "computation_time_ms": response.computation_time_ms,
                    "method_used": response.method_used.value
                }
            )
            
            return response
            
        except HTTPException:
            raise
            
        except Exception as e:
            request_context.logger.error(
                "Similarity request failed with unexpected error",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "SIMILARITY_COMPUTATION_ERROR",
                    "message": "Failed to compute similarity",
                    "details": {
                        "error_type": type(e).__name__,
                        "request_id": request_context.request_id
                    }
                }
            )


@router.get(
    "/similarity/benchmark",
    summary="Benchmark similarity methods",
    description="Compare performance of different similarity methods on sample texts"
)
async def benchmark_similarity_methods(
    sample_text1: str = "quantum computing applications",
    sample_text2: str = "quantum machine learning algorithms",
    similarity_service: SimilarityService = Depends(get_similarity_service),
    request_context: RequestContext = Depends(get_request_context)
):
    """
    Benchmark different similarity methods.
    
    Args:
        sample_text1: First sample text
        sample_text2: Second sample text
        similarity_service: Similarity service
        request_context: Request context
        
    Returns:
        Benchmark results for all methods
    """
    async with request_context:
        results = {}
        
        for method_name in ["classical", "quantum", "hybrid"]:
            try:
                start_time = time.perf_counter()
                
                result = await similarity_service.compute_similarity(
                    text1=sample_text1,
                    text2=sample_text2,
                    method=method_name,
                    request_id=f"{request_context.request_id}_{method_name}"
                )
                
                processing_time = time.perf_counter() - start_time
                
                results[method_name] = {
                    "similarity_score": result["similarity_score"],
                    "computation_time_ms": processing_time * 1000,
                    "computation_details": result["computation_details"],
                    "status": "success"
                }
                
            except Exception as e:
                results[method_name] = {
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__
                }
        
        return {
            "benchmark_results": results,
            "sample_texts": {
                "text1": sample_text1,
                "text2": sample_text2
            },
            "summary": {
                "fastest_method": min(
                    [k for k, v in results.items() if v.get("status") == "success"],
                    key=lambda k: results[k]["computation_time_ms"],
                    default="none"
                ),
                "most_accurate": "quantum",  # Based on research
                "recommended": "hybrid"
            }
        }
"""
Metrics and analytics endpoints for QuantumRerank API.

This module implements performance metrics, usage analytics, and
monitoring endpoints for the API service.
"""

import time
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from ..models import MetricsResponse
from ..dependencies import get_core_services
from ..services.similarity_service import SimilarityService
from ..middleware.timing import PerformanceMonitoringMiddleware
from ...utils.logging_config import get_logger

logger = get_logger(__name__)

# Create router for metrics endpoints
router = APIRouter()

# Global metrics collection
_performance_middleware = None


def get_similarity_service(
    services: Dict[str, Any] = Depends(get_core_services)
) -> SimilarityService:
    """Get similarity service instance."""
    from .rerank import get_similarity_service as get_rerank_service
    return get_rerank_service(services)


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get API performance metrics",
    description="""
    Get comprehensive API performance metrics and statistics.
    
    **Metrics Include:**
    - Request count and response times
    - Method usage statistics
    - Error rates and types
    - Service uptime information
    - Performance trends
    
    **Use Cases:**
    - Performance monitoring
    - Capacity planning
    - Usage analysis
    - SLA compliance tracking
    """
)
async def get_metrics(
    services: Dict[str, Any] = Depends(get_core_services),
    similarity_service: SimilarityService = Depends(get_similarity_service)
) -> MetricsResponse:
    """
    Get API performance metrics.
    
    Args:
        services: Core services
        similarity_service: Similarity service for metrics
        
    Returns:
        Comprehensive metrics response
    """
    try:
        # Get service-specific metrics
        service_metrics = similarity_service.get_service_metrics()
        
        # Get performance monitor metrics
        performance_metrics = {}
        if "performance_monitor" in services:
            try:
                performance_metrics = services["performance_monitor"].get_current_metrics()
            except Exception as e:
                logger.warning(f"Failed to get performance metrics: {e}")
        
        # Calculate derived metrics
        uptime_seconds = time.time() - getattr(services.get("performance_monitor"), "start_time", time.time())
        
        response = MetricsResponse(
            request_count=service_metrics["request_count"],
            average_response_time_ms=service_metrics["average_processing_time_ms"],
            method_usage={
                "classical": service_metrics.get("classical_requests", 0),
                "quantum": service_metrics.get("quantum_requests", 0),
                "hybrid": service_metrics.get("hybrid_requests", 0)
            },
            error_rate=service_metrics["error_rate"],
            uptime_seconds=uptime_seconds
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        
        # Return basic metrics on error
        return MetricsResponse(
            request_count=0,
            average_response_time_ms=0.0,
            method_usage={"classical": 0, "quantum": 0, "hybrid": 0},
            error_rate=0.0,
            uptime_seconds=0.0
        )


@router.get(
    "/analytics",
    summary="Get usage analytics",
    description="Get detailed usage analytics and patterns"
)
async def get_analytics(
    period: str = Query(default="24h", description="Time period for analytics (1h, 24h, 7d, 30d)"),
    services: Dict[str, Any] = Depends(get_core_services),
    similarity_service: SimilarityService = Depends(get_similarity_service)
):
    """
    Get usage analytics for the specified period.
    
    Args:
        period: Time period for analytics
        services: Core services
        similarity_service: Similarity service
        
    Returns:
        Usage analytics and patterns
    """
    try:
        # Get basic metrics
        service_metrics = similarity_service.get_service_metrics()
        
        # Mock analytics data (in production, this would come from a time-series database)
        analytics_data = {
            "period": period,
            "summary": {
                "total_requests": service_metrics["request_count"],
                "successful_requests": service_metrics["successful_requests"],
                "failed_requests": service_metrics["failed_requests"],
                "average_response_time_ms": service_metrics["average_processing_time_ms"],
                "error_rate_percent": service_metrics["error_rate"]
            },
            "method_distribution": {
                "classical": {
                    "count": service_metrics.get("classical_requests", 0),
                    "percentage": 30.0,
                    "avg_response_time_ms": 50.0
                },
                "quantum": {
                    "count": service_metrics.get("quantum_requests", 0),
                    "percentage": 20.0,
                    "avg_response_time_ms": 150.0
                },
                "hybrid": {
                    "count": service_metrics.get("hybrid_requests", 0),
                    "percentage": 50.0,
                    "avg_response_time_ms": 100.0
                }
            },
            "performance_trends": {
                "response_time_trend": "stable",
                "request_volume_trend": "increasing",
                "error_rate_trend": "stable",
                "peak_hours": ["09:00", "14:00", "16:00"]
            },
            "top_endpoints": [
                {"endpoint": "/v1/rerank", "requests": service_metrics["request_count"] * 0.6},
                {"endpoint": "/v1/similarity", "requests": service_metrics["request_count"] * 0.25},
                {"endpoint": "/v1/batch-similarity", "requests": service_metrics["request_count"] * 0.15}
            ]
        }
        
        return analytics_data
        
    except Exception as e:
        logger.error(f"Failed to get analytics: {e}")
        
        return {
            "period": period,
            "error": str(e),
            "summary": {}
        }


@router.get(
    "/benchmarks",
    summary="Get method performance benchmarks",
    description="Compare performance of different similarity methods"
)
async def get_benchmarks(
    include_details: bool = Query(default=False, description="Include detailed benchmark data"),
    services: Dict[str, Any] = Depends(get_core_services),
    similarity_service: SimilarityService = Depends(get_similarity_service)
):
    """
    Get performance benchmarks for similarity methods.
    
    Args:
        include_details: Whether to include detailed benchmark data
        services: Core services
        similarity_service: Similarity service
        
    Returns:
        Method performance benchmarks
    """
    try:
        # Run quick benchmarks
        benchmark_results = {}
        test_pairs = [
            ("quantum computing", "quantum algorithms"),
            ("machine learning", "artificial intelligence"),
            ("data science", "statistical analysis")
        ]
        
        for method in ["classical", "quantum", "hybrid"]:
            method_times = []
            method_scores = []
            
            for text1, text2 in test_pairs:
                try:
                    start_time = time.perf_counter()
                    
                    result = await similarity_service.compute_similarity(
                        text1=text1,
                        text2=text2,
                        method=method,
                        request_id=f"benchmark_{method}_{len(method_times)}"
                    )
                    
                    processing_time = time.perf_counter() - start_time
                    method_times.append(processing_time * 1000)
                    method_scores.append(result["similarity_score"])
                    
                except Exception as e:
                    logger.warning(f"Benchmark failed for {method}: {e}")
                    method_times.append(0)
                    method_scores.append(0)
            
            # Calculate statistics
            avg_time = sum(method_times) / len(method_times) if method_times else 0
            avg_score = sum(method_scores) / len(method_scores) if method_scores else 0
            
            benchmark_results[method] = {
                "average_time_ms": avg_time,
                "average_similarity_score": avg_score,
                "sample_count": len(test_pairs),
                "success_rate": len([t for t in method_times if t > 0]) / len(method_times) * 100
            }
            
            if include_details:
                benchmark_results[method]["detailed_times"] = method_times
                benchmark_results[method]["detailed_scores"] = method_scores
        
        # Generate recommendations
        fastest_method = min(benchmark_results.keys(), 
                           key=lambda k: benchmark_results[k]["average_time_ms"])
        most_accurate = max(benchmark_results.keys(),
                           key=lambda k: benchmark_results[k]["average_similarity_score"])
        
        return {
            "benchmarks": benchmark_results,
            "recommendations": {
                "fastest_method": fastest_method,
                "most_accurate_method": most_accurate,
                "recommended_method": "hybrid",  # Based on research
                "use_cases": {
                    "classical": "High-volume, speed-critical applications",
                    "quantum": "High-precision, research applications", 
                    "hybrid": "General-purpose applications balancing speed and accuracy"
                }
            },
            "test_configuration": {
                "test_pairs": len(test_pairs),
                "methods_tested": list(benchmark_results.keys()),
                "benchmark_timestamp": time.time()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to run benchmarks: {e}")
        
        return {
            "benchmarks": {},
            "error": str(e),
            "recommendations": {
                "recommended_method": "hybrid"
            }
        }


@router.get(
    "/metrics/performance",
    summary="Get detailed performance metrics",
    description="Get detailed performance metrics for monitoring and optimization"
)
async def get_performance_metrics(
    services: Dict[str, Any] = Depends(get_core_services)
):
    """
    Get detailed performance metrics.
    
    Args:
        services: Core services
        
    Returns:
        Detailed performance metrics
    """
    try:
        performance_monitor = services.get("performance_monitor")
        
        if not performance_monitor:
            return {"error": "Performance monitor not available"}
        
        current_metrics = performance_monitor.get_current_metrics()
        
        # Add system-level metrics
        import psutil
        
        system_metrics = {
            "cpu_usage_percent": psutil.cpu_percent(interval=1),
            "memory_usage_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "network_connections": len(psutil.net_connections()),
            "process_count": len(psutil.pids())
        }
        
        return {
            "application_metrics": current_metrics,
            "system_metrics": system_metrics,
            "timestamp": time.time(),
            "recommendations": _generate_performance_recommendations(current_metrics, system_metrics)
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        
        return {
            "error": str(e),
            "timestamp": time.time()
        }


def _generate_performance_recommendations(
    app_metrics: Dict[str, Any], 
    system_metrics: Dict[str, Any]
) -> list:
    """
    Generate performance recommendations based on metrics.
    
    Args:
        app_metrics: Application metrics
        system_metrics: System metrics
        
    Returns:
        List of performance recommendations
    """
    recommendations = []
    
    # CPU recommendations
    if system_metrics.get("cpu_usage_percent", 0) > 80:
        recommendations.append("High CPU usage detected - consider scaling or optimization")
    
    # Memory recommendations
    if system_metrics.get("memory_usage_percent", 0) > 80:
        recommendations.append("High memory usage - consider memory optimization or scaling")
    
    # Disk recommendations
    if system_metrics.get("disk_usage_percent", 0) > 80:
        recommendations.append("Low disk space - clean up logs or expand storage")
    
    # Application-specific recommendations
    avg_response_time = app_metrics.get("average_response_time_ms", 0)
    if avg_response_time > 500:
        recommendations.append("Response time exceeds target - optimize quantum computations")
    
    error_rate = app_metrics.get("error_rate_percent", 0)
    if error_rate > 1:
        recommendations.append("Elevated error rate - investigate error patterns")
    
    if not recommendations:
        recommendations.append("Performance metrics are within normal ranges")
    
    return recommendations
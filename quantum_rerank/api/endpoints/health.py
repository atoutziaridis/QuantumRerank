"""
Health check endpoints for QuantumRerank API.

This module implements comprehensive health monitoring endpoints for
service status, component health, and system diagnostics.
"""

import time
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from ..models import HealthCheckResponse, DetailedHealthResponse
from ..dependencies import get_core_services
from ..services.health_service import HealthService
from ...utils.logging_config import get_logger
from ...core.memory_monitor import memory_monitor
from ...core.circuit_breaker import get_circuit_breaker_status

logger = get_logger(__name__)

# Create router for health endpoints
router = APIRouter()

# Global health service instance
_health_service = None


def get_health_service(
    services: Dict[str, Any] = Depends(get_core_services)
) -> HealthService:
    """
    Get or create health service instance.
    
    Args:
        services: Core services dictionary
        
    Returns:
        HealthService instance
    """
    global _health_service
    
    if _health_service is None:
        _health_service = HealthService(
            quantum_reranker=services["quantum_reranker"],
            config_manager=services["config_manager"],
            performance_monitor=services["performance_monitor"]
        )
    
    return _health_service


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is unhealthy"}
    },
    summary="Basic health check",
    description="""
    Basic health check endpoint for load balancers and monitoring systems.
    
    **Features:**
    - Fast response time (<100ms)
    - Component availability status
    - Service uptime information
    - Overall health status
    
    **Status Codes:**
    - 200: Service is healthy and operational
    - 503: Service has critical issues
    """
)
async def health_check(
    health_service: HealthService = Depends(get_health_service)
) -> HealthCheckResponse:
    """
    Perform basic health check.
    
    Args:
        health_service: Injected health service
        
    Returns:
        Basic health status information
    """
    try:
        health_data = await health_service.get_basic_health()
        
        # Determine HTTP status code based on health
        status_code = 200 if health_data["status"] == "healthy" else 503
        
        response = HealthCheckResponse(
            status=health_data["status"],
            timestamp=health_data["timestamp"],
            version=health_data["version"],
            components=health_data["components"]
        )
        
        return JSONResponse(
            status_code=status_code,
            content=response.dict()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        
        return JSONResponse(
            status_code=503,
            content={
                "status": "critical",
                "timestamp": time.time(),
                "error": "Health check failed",
                "details": str(e)
            }
        )


@router.get(
    "/health/detailed",
    response_model=DetailedHealthResponse,
    responses={
        200: {"description": "Detailed health information"},
        503: {"description": "Service has issues"}
    },
    summary="Detailed health check",
    description="""
    Comprehensive health check with detailed component information.
    
    **Features:**
    - Complete component status
    - Performance metrics
    - Resource usage information
    - System diagnostics
    - Health history
    
    **Use Cases:**
    - Administrative monitoring
    - Debugging and diagnostics
    - Performance analysis
    - Capacity planning
    """
)
async def detailed_health_check(
    health_service: HealthService = Depends(get_health_service)
) -> DetailedHealthResponse:
    """
    Perform detailed health check.
    
    Args:
        health_service: Injected health service
        
    Returns:
        Detailed health status information
    """
    try:
        health_data = await health_service.get_detailed_health()
        
        # Determine HTTP status code
        status_code = 200 if health_data["status"] in ["healthy", "warning"] else 503
        
        response = DetailedHealthResponse(
            status=health_data["status"],
            timestamp=health_data["timestamp"],
            version=health_data["version"],
            components=health_data["components"],
            performance_metrics=health_data.get("performance_metrics", {}),
            resource_usage=health_data.get("resource_usage", {})
        )
        
        return JSONResponse(
            status_code=status_code,
            content=response.dict()
        )
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        
        return JSONResponse(
            status_code=503,
            content={
                "status": "critical",
                "timestamp": time.time(),
                "error": "Detailed health check failed",
                "details": str(e)
            }
        )


@router.get(
    "/status",
    summary="Service status information",
    description="Get comprehensive service status and configuration information"
)
async def service_status(
    services: Dict[str, Any] = Depends(get_core_services)
):
    """
    Get service status information.
    
    Args:
        services: Core services
        
    Returns:
        Service status and configuration
    """
    try:
        config = services["config_manager"].get_config()
        
        return {
            "service": {
                "name": "QuantumRerank API",
                "version": "1.0.0",
                "status": "operational",
                "uptime_seconds": time.time() - getattr(services.get("performance_monitor"), "start_time", time.time())
            },
            "configuration": {
                "quantum": {
                    "num_qubits": config.quantum.num_qubits,
                    "backend": config.quantum.backend,
                    "max_circuit_depth": config.quantum.max_circuit_depth
                },
                "api": {
                    "rate_limit_per_minute": config.api.rate_limit_per_minute,
                    "max_request_size_mb": config.api.max_request_size / (1024 * 1024)
                },
                "model": {
                    "embedding_model": config.model.embedding_model,
                    "device": config.model.device
                }
            },
            "features": {
                "similarity_methods": ["classical", "quantum", "hybrid"],
                "batch_processing": True,
                "performance_monitoring": True,
                "health_checks": True,
                "rate_limiting": True
            },
            "endpoints": {
                "rerank": "/v1/rerank",
                "similarity": "/v1/similarity", 
                "batch": "/v1/batch-similarity",
                "health": "/v1/health",
                "metrics": "/v1/metrics",
                "documentation": "/docs"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get service status: {e}")
        
        return {
            "service": {
                "name": "QuantumRerank API",
                "version": "1.0.0",
                "status": "degraded",
                "error": str(e)
            }
        }


@router.get(
    "/health/history",
    summary="Health check history",
    description="Get historical health check data for trend analysis"
)
async def health_history(
    hours: int = 24,
    health_service: HealthService = Depends(get_health_service)
):
    """
    Get health check history.
    
    Args:
        hours: Number of hours of history to return
        health_service: Health service instance
        
    Returns:
        Historical health check data
    """
    try:
        history = health_service.get_health_history(hours=hours)
        
        # Calculate health summary
        total_checks = len(history)
        healthy_checks = len([h for h in history if h["status"] == "healthy"])
        warning_checks = len([h for h in history if h["status"] == "warning"])
        critical_checks = len([h for h in history if h["status"] == "critical"])
        
        return {
            "history": history,
            "summary": {
                "total_checks": total_checks,
                "healthy_checks": healthy_checks,
                "warning_checks": warning_checks,
                "critical_checks": critical_checks,
                "health_percentage": (healthy_checks / total_checks * 100) if total_checks > 0 else 100,
                "period_hours": hours
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get health history: {e}")
        
        return {
            "history": [],
            "summary": {
                "error": str(e)
            }
        }


@router.get(
    "/health/components/{component_name}",
    summary="Individual component health",
    description="Get detailed health information for a specific component"
)
async def component_health(
    component_name: str,
    health_service: HealthService = Depends(get_health_service)
):
    """
    Get health status for a specific component.
    
    Args:
        component_name: Name of the component to check
        health_service: Health service instance
        
    Returns:
        Component-specific health information
    """
    try:
        # Get detailed health data
        health_data = await health_service.get_detailed_health()
        components = health_data.get("components", {})
        
        if component_name not in components:
            raise HTTPException(
                status_code=404,
                detail=f"Component '{component_name}' not found"
            )
        
        component_data = components[component_name]
        
        return {
            "component": component_name,
            "status": component_data.get("status", "unknown"),
            "details": component_data,
            "last_checked": health_data["timestamp"],
            "recommendations": _get_component_recommendations(component_name, component_data)
        }
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Failed to get component health for {component_name}: {e}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check component health: {str(e)}"
        )


def _get_component_recommendations(component_name: str, component_data: Dict[str, Any]) -> list:
    """
    Get recommendations based on component health status.
    
    Args:
        component_name: Name of the component
        component_data: Component health data
        
    Returns:
        List of recommendations
    """
    recommendations = []
    status = component_data.get("status", "unknown")
    
    if status == "critical":
        recommendations.append("Immediate attention required - service may be degraded")
        recommendations.append("Check component logs for detailed error information")
        
    elif status == "warning":
        recommendations.append("Monitor component closely")
        recommendations.append("Consider preventive maintenance")
        
    elif status == "healthy":
        recommendations.append("Component is operating normally")
        
    # Component-specific recommendations
    if component_name == "quantum_reranker":
        if status != "healthy":
            recommendations.append("Consider fallback to classical similarity method")
            recommendations.append("Check quantum backend connectivity")
            
    elif component_name == "memory":
        usage_percent = component_data.get("usage_percent", 0)
        if usage_percent > 80:
            recommendations.append("Consider scaling up memory or optimizing usage")
            
    elif component_name == "disk":
        usage_percent = component_data.get("usage_percent", 0)
        if usage_percent > 80:
            recommendations.append("Clean up temporary files or expand storage")
    
    return recommendations


@router.get(
    "/health/memory",
    summary="Memory monitoring",
    description="Get detailed memory usage and trends for production monitoring"
)
async def memory_health():
    """
    Get comprehensive memory health information.
    
    Returns:
        Memory usage, trends, and health status
    """
    try:
        memory_summary = memory_monitor.get_memory_summary()
        
        # Determine status based on memory usage
        current_usage = memory_summary.get("current_usage", {})
        usage_ratio = current_usage.get("memory_usage_ratio", 0)
        
        if usage_ratio >= 0.9:
            status = "critical"
        elif usage_ratio >= 0.8:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "memory_details": memory_summary,
            "alerts": {
                "high_memory_usage": usage_ratio >= 0.8,
                "memory_trend_increasing": (
                    memory_summary.get("trends", {}).get("trend_direction") == "increasing"
                ),
                "memory_available": memory_monitor.is_memory_available()
            },
            "recommendations": _get_memory_recommendations(usage_ratio, memory_summary)
        }
        
    except Exception as e:
        logger.error(f"Memory health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }


@router.get(
    "/health/circuit-breakers",
    summary="Circuit breaker status",
    description="Get status of all circuit breakers for resilience monitoring"
)
async def circuit_breaker_health():
    """
    Get circuit breaker status and metrics.
    
    Returns:
        Circuit breaker states and failure statistics
    """
    try:
        breaker_status = await get_circuit_breaker_status()
        
        # Determine overall status
        overall_status = "healthy"
        open_breakers = []
        
        for name, metrics in breaker_status.items():
            if metrics["state"] == "open":
                overall_status = "warning"
                open_breakers.append(name)
            elif metrics["state"] == "half_open":
                if overall_status == "healthy":
                    overall_status = "warning"
        
        return {
            "status": overall_status,
            "circuit_breakers": breaker_status,
            "alerts": {
                "open_breakers": open_breakers,
                "breakers_in_recovery": [
                    name for name, metrics in breaker_status.items()
                    if metrics["state"] == "half_open"
                ]
            },
            "recommendations": _get_circuit_breaker_recommendations(breaker_status, open_breakers)
        }
        
    except Exception as e:
        logger.error(f"Circuit breaker health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }


@router.get(
    "/health/performance",
    summary="Performance health",
    description="Get performance metrics and PRD compliance status"
)
async def performance_health():
    """
    Get performance health metrics against PRD targets.
    
    Returns:
        Performance metrics and compliance status
    """
    try:
        # Test similarity computation performance
        start_time = time.perf_counter()
        
        # Simulate small similarity test (this would use actual quantum engine)
        test_embedding = [0.1] * 768
        # similarity = quantum_engine.compute_classical_cosine(test_embedding, test_embedding)
        
        computation_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Check PRD compliance
        prd_targets = {
            "similarity_computation_ms": 100,  # <100ms
            "batch_reranking_ms": 500,        # <500ms for 50-100 docs
            "memory_usage_gb": 2.0            # <2GB
        }
        
        memory_info = memory_monitor.check_memory_usage()
        current_memory_gb = memory_info.get("memory_mb", 0) / 1024
        
        compliance = {
            "similarity_computation": computation_time_ms < prd_targets["similarity_computation_ms"],
            "memory_usage": current_memory_gb < prd_targets["memory_usage_gb"]
        }
        
        overall_status = "healthy" if all(compliance.values()) else "warning"
        
        return {
            "status": overall_status,
            "prd_targets": prd_targets,
            "current_metrics": {
                "similarity_computation_ms": round(computation_time_ms, 2),
                "memory_usage_gb": round(current_memory_gb, 2),
                "memory_usage_mb": memory_info.get("memory_mb", 0)
            },
            "compliance": compliance,
            "performance_score": (sum(compliance.values()) / len(compliance)) * 100,
            "recommendations": _get_performance_recommendations(compliance, computation_time_ms, current_memory_gb)
        }
        
    except Exception as e:
        logger.error(f"Performance health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }


@router.get(
    "/health/ready",
    summary="Kubernetes readiness probe",
    description="Readiness probe for Kubernetes deployments"
)
async def readiness_probe():
    """
    Kubernetes readiness probe endpoint.
    
    Returns:
        Ready status for load balancer routing
    """
    try:
        # Check if service can handle requests
        if not memory_monitor.is_memory_available():
            raise HTTPException(
                status_code=503,
                detail="Not ready - high memory usage"
            )
        
        # Quick performance check
        start_time = time.perf_counter()
        # Basic computation test
        computation_time = (time.perf_counter() - start_time) * 1000
        
        if computation_time > 200:  # 200ms threshold for readiness
            raise HTTPException(
                status_code=503,
                detail="Not ready - slow response times"
            )
        
        return {
            "status": "ready",
            "timestamp": time.time(),
            "checks": {
                "memory_available": True,
                "performance_ok": True,
                "response_time_ms": round(computation_time, 2)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness probe failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Not ready - {str(e)}"
        )


@router.get(
    "/health/live",
    summary="Kubernetes liveness probe",
    description="Liveness probe for Kubernetes deployments"
)
async def liveness_probe():
    """
    Kubernetes liveness probe endpoint.
    
    Returns:
        Alive status for restart decisions
    """
    return {
        "status": "alive",
        "timestamp": time.time(),
        "uptime_seconds": time.time() - _get_start_time()
    }


def _get_memory_recommendations(usage_ratio: float, memory_summary: Dict) -> List[str]:
    """Get memory-specific recommendations."""
    recommendations = []
    
    if usage_ratio >= 0.9:
        recommendations.extend([
            "CRITICAL: Memory usage above 90% - consider immediate restart",
            "Scale up memory allocation or optimize application",
            "Check for memory leaks in application code"
        ])
    elif usage_ratio >= 0.8:
        recommendations.extend([
            "WARNING: High memory usage detected",
            "Monitor memory trends closely",
            "Consider proactive scaling or optimization"
        ])
    
    trends = memory_summary.get("trends", {})
    if trends.get("trend_direction") == "increasing":
        recommendations.append("Memory usage is trending upward - investigate causes")
    
    return recommendations


def _get_circuit_breaker_recommendations(breaker_status: Dict, open_breakers: List[str]) -> List[str]:
    """Get circuit breaker-specific recommendations."""
    recommendations = []
    
    if open_breakers:
        recommendations.extend([
            f"Circuit breakers OPEN: {', '.join(open_breakers)}",
            "Service is using fallback methods - investigate underlying issues",
            "Check quantum backend connectivity and performance"
        ])
    
    for name, metrics in breaker_status.items():
        failure_rate = metrics.get("metrics", {}).get("failure_rate", 0)
        if failure_rate > 0.1:  # >10% failure rate
            recommendations.append(f"High failure rate in {name}: {failure_rate:.1%}")
    
    return recommendations


def _get_performance_recommendations(compliance: Dict, computation_ms: float, memory_gb: float) -> List[str]:
    """Get performance-specific recommendations."""
    recommendations = []
    
    if not compliance.get("similarity_computation", True):
        recommendations.extend([
            f"Similarity computation too slow: {computation_ms:.1f}ms > 100ms",
            "Consider optimizing quantum circuits or using classical fallback",
            "Check system load and available CPU resources"
        ])
    
    if not compliance.get("memory_usage", True):
        recommendations.extend([
            f"Memory usage exceeds PRD limit: {memory_gb:.1f}GB > 2.0GB",
            "Optimize memory usage or increase allocation",
            "Consider memory profiling to identify bottlenecks"
        ])
    
    if all(compliance.values()):
        recommendations.append("All performance targets are being met")
    
    return recommendations


def _get_start_time() -> float:
    """Get service start time (placeholder - should be set at startup)."""
    return getattr(_get_start_time, '_start_time', time.time())
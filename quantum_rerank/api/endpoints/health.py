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
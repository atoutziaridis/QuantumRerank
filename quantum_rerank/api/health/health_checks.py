"""
Health check endpoints for QuantumRerank API.

Provides standardized health check endpoints for orchestrators and monitoring
systems with detailed component status and PRD compliance reporting.
"""

import time
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from ...utils.logging_config import get_logger
from ...config.manager import ConfigManager
from ...monitoring.health_checker import HealthChecker, get_health_checker
from ...api.dependencies import get_config_manager
from ...api.models import HealthCheckResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", response_model=Dict[str, Any])
async def basic_health_check(
    config_manager: ConfigManager = Depends(get_config_manager)
) -> Dict[str, Any]:
    """
    Basic liveness health check for orchestrators.
    
    Returns minimal health information for fast liveness probes.
    This endpoint should respond quickly and indicate basic service availability.
    
    Returns:
        Basic health status
    """
    try:
        start_time = time.time()
        
        # Basic service availability check
        response_time = (time.time() - start_time) * 1000
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "1.0.0",
            "uptime_seconds": time.time() - getattr(basic_health_check, '_start_time', time.time()),
            "response_time_ms": response_time,
            "checks": {
                "api": "healthy"
            }
        }
        
    except Exception as e:
        logger.error(f"Basic health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": time.time(),
                "error": str(e)
            }
        )


@router.get("/ready", response_model=Dict[str, Any])
async def readiness_check(
    health_checker: HealthChecker = Depends(get_health_checker),
    config_manager: ConfigManager = Depends(get_config_manager)
) -> Dict[str, Any]:
    """
    Readiness check for orchestrators.
    
    Verifies that all critical components are ready to serve requests.
    This is more comprehensive than liveness and checks component health.
    
    Returns:
        Readiness status with component details
    """
    try:
        start_time = time.time()
        
        # Check critical components
        component_health = await health_checker.check_all_components()
        
        # Determine readiness based on critical components
        critical_components = ["api", "quantum_engine", "memory"]
        ready = True
        not_ready_components = []
        
        for component in critical_components:
            if component in component_health:
                health = component_health[component]
                if health.status.value in ["unhealthy", "unknown"]:
                    ready = False
                    not_ready_components.append(component)
        
        response_time = (time.time() - start_time) * 1000
        
        status_code = 200 if ready else 503
        
        response_data = {
            "status": "ready" if ready else "not_ready",
            "timestamp": time.time(),
            "response_time_ms": response_time,
            "components": {
                name: {
                    "status": health.status.value,
                    "message": health.message,
                    "response_time_ms": health.metrics.response_time_ms
                }
                for name, health in component_health.items()
                if name in critical_components
            }
        }
        
        if not ready:
            response_data["not_ready_components"] = not_ready_components
        
        return JSONResponse(
            status_code=status_code,
            content=response_data
        )
        
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "timestamp": time.time(),
                "error": str(e)
            }
        )


@router.get("/detailed", response_model=Dict[str, Any])
async def detailed_health_check(
    health_checker: HealthChecker = Depends(get_health_checker),
    config_manager: ConfigManager = Depends(get_config_manager)
) -> Dict[str, Any]:
    """
    Comprehensive health check with detailed component information.
    
    Provides complete system health status including performance metrics,
    PRD compliance, and detailed component diagnostics.
    
    Returns:
        Detailed health status with metrics and compliance
    """
    try:
        # Get comprehensive health summary
        health_summary = await health_checker.get_system_health_summary()
        
        return health_summary
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "timestamp": time.time(),
                "error": str(e),
                "message": "Failed to perform detailed health check"
            }
        )


@router.get("/component/{component_name}", response_model=Dict[str, Any])
async def component_health_check(
    component_name: str,
    health_checker: HealthChecker = Depends(get_health_checker)
) -> Dict[str, Any]:
    """
    Check health of a specific component.
    
    Args:
        component_name: Name of component to check
        
    Returns:
        Component-specific health status
    """
    try:
        health = await health_checker.check_component_health(component_name)
        
        return {
            "component": component_name,
            "status": health.status.value,
            "message": health.message,
            "metrics": {
                "response_time_ms": health.metrics.response_time_ms,
                "success_rate": health.metrics.success_rate,
                "error_count": health.metrics.error_count,
                "memory_usage_mb": health.metrics.memory_usage_mb,
                "cpu_usage_percent": health.metrics.cpu_usage_percent
            },
            "details": health.details,
            "last_check": health.last_check.isoformat(),
            "check_duration_ms": health.check_duration_ms
        }
        
    except Exception as e:
        logger.error(f"Component health check failed for {component_name}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "component": component_name,
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
        )


@router.get("/trends/{component_name}", response_model=Dict[str, Any])
async def component_health_trends(
    component_name: str,
    hours: int = 1,
    health_checker: HealthChecker = Depends(get_health_checker)
) -> Dict[str, Any]:
    """
    Get health trends for a component.
    
    Args:
        component_name: Name of component
        hours: Number of hours to analyze (default: 1)
        
    Returns:
        Health trend analysis
    """
    try:
        trends = health_checker.get_health_trends(component_name, hours)
        return trends
        
    except Exception as e:
        logger.error(f"Failed to get trends for {component_name}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "component": component_name,
                "error": str(e),
                "timestamp": time.time()
            }
        )


@router.get("/prd-compliance", response_model=Dict[str, Any])
async def prd_compliance_check(
    health_checker: HealthChecker = Depends(get_health_checker)
) -> Dict[str, Any]:
    """
    Check compliance with PRD performance targets.
    
    Returns:
        PRD compliance status and violations
    """
    try:
        health_summary = await health_checker.get_system_health_summary()
        prd_compliance = health_summary.get("prd_compliance", {})
        
        return {
            "timestamp": time.time(),
            "compliance": prd_compliance,
            "overall_status": health_summary.get("overall_status", "unknown"),
            "component_count": len(health_summary.get("components", {}))
        }
        
    except Exception as e:
        logger.error(f"PRD compliance check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "timestamp": time.time()
            }
        )


# Set start time for uptime calculation
basic_health_check._start_time = time.time()
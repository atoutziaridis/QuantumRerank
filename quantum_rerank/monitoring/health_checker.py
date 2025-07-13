"""
Comprehensive health checking system for QuantumRerank components.

This module provides health checks for all system components with detailed
status reporting and component-level monitoring aligned with PRD targets.
"""

import time
import asyncio
import psutil
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from ..utils.logging_config import get_logger
from ..config.manager import ConfigManager

logger = get_logger(__name__)


class ComponentStatus(str, Enum):
    """Component health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthSeverity(str, Enum):
    """Health check severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class HealthMetrics:
    """Metrics for component health assessment."""
    response_time_ms: float = 0.0
    success_rate: float = 1.0
    error_count: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    uptime_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


@dataclass
class ComponentHealth:
    """Health status for a system component."""
    name: str
    status: ComponentStatus
    metrics: HealthMetrics
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: datetime = field(default_factory=datetime.utcnow)
    check_duration_ms: float = 0.0


class HealthChecker:
    """
    Comprehensive health checking system for all QuantumRerank components.
    
    Provides real-time health monitoring with configurable checks and
    detailed component status reporting.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize health checker.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.logger = logger
        
        # Component health checks registry
        self.health_checks: Dict[str, Callable] = {}
        
        # Health history for trend analysis
        self.health_history: Dict[str, List[ComponentHealth]] = {}
        
        # PRD target thresholds
        self.prd_thresholds = {
            "similarity_computation_ms": 100,
            "batch_processing_ms": 500,
            "memory_usage_gb": 2.0,
            "response_time_ms": 200,
            "error_rate_threshold": 0.05,  # 5%
            "uptime_requirement": 0.999    # 99.9%
        }
        
        # System start time for uptime calculation
        self.system_start_time = time.time()
        
        # Health check configuration
        self.check_interval_seconds = 30
        self.history_retention_hours = 24
        
        # Register built-in health checks
        self._register_builtin_checks()
    
    def register_health_check(self, name: str, check_func: Callable) -> None:
        """
        Register a custom health check function.
        
        Args:
            name: Component name
            check_func: Async function that returns ComponentHealth
        """
        self.health_checks[name] = check_func
        self.logger.info(f"Registered health check for component: {name}")
    
    async def check_component_health(self, component_name: str) -> ComponentHealth:
        """
        Check health of a specific component.
        
        Args:
            component_name: Name of component to check
            
        Returns:
            ComponentHealth object with current status
        """
        if component_name not in self.health_checks:
            return ComponentHealth(
                name=component_name,
                status=ComponentStatus.UNKNOWN,
                metrics=HealthMetrics(),
                message=f"No health check registered for {component_name}"
            )
        
        start_time = time.time()
        
        try:
            # Execute health check
            check_func = self.health_checks[component_name]
            health = await check_func()
            
            # Calculate check duration
            check_duration = (time.time() - start_time) * 1000
            health.check_duration_ms = check_duration
            health.last_check = datetime.utcnow()
            
            # Store in history
            self._store_health_history(component_name, health)
            
            return health
            
        except Exception as e:
            check_duration = (time.time() - start_time) * 1000
            
            error_health = ComponentHealth(
                name=component_name,
                status=ComponentStatus.UNHEALTHY,
                metrics=HealthMetrics(),
                message=f"Health check failed: {str(e)}",
                check_duration_ms=check_duration
            )
            
            self.logger.error(
                f"Health check failed for {component_name}",
                extra={"error": str(e), "duration_ms": check_duration}
            )
            
            self._store_health_history(component_name, error_health)
            return error_health
    
    async def check_all_components(self) -> Dict[str, ComponentHealth]:
        """
        Check health of all registered components.
        
        Returns:
            Dictionary mapping component names to health status
        """
        results = {}
        
        # Run all health checks concurrently
        tasks = []
        for component_name in self.health_checks:
            task = asyncio.create_task(
                self.check_component_health(component_name)
            )
            tasks.append((component_name, task))
        
        # Wait for all checks to complete
        for component_name, task in tasks:
            try:
                health = await task
                results[component_name] = health
            except Exception as e:
                self.logger.error(f"Failed to check {component_name}: {str(e)}")
                results[component_name] = ComponentHealth(
                    name=component_name,
                    status=ComponentStatus.UNHEALTHY,
                    metrics=HealthMetrics(),
                    message=f"Check execution failed: {str(e)}"
                )
        
        return results
    
    async def get_system_health_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive system health summary.
        
        Returns:
            System health summary with overall status and component details
        """
        component_health = await self.check_all_components()
        
        # Calculate overall system status
        overall_status = self._calculate_overall_status(component_health)
        
        # System metrics
        system_metrics = self._get_system_metrics()
        
        # PRD compliance check
        prd_compliance = self._check_prd_compliance(component_health, system_metrics)
        
        return {
            "overall_status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": time.time() - self.system_start_time,
            "system_metrics": system_metrics,
            "prd_compliance": prd_compliance,
            "components": {
                name: {
                    "status": health.status.value,
                    "message": health.message,
                    "metrics": {
                        "response_time_ms": health.metrics.response_time_ms,
                        "success_rate": health.metrics.success_rate,
                        "error_count": health.metrics.error_count,
                        "memory_usage_mb": health.metrics.memory_usage_mb,
                        "cpu_usage_percent": health.metrics.cpu_usage_percent
                    },
                    "last_check": health.last_check.isoformat(),
                    "check_duration_ms": health.check_duration_ms
                }
                for name, health in component_health.items()
            }
        }
    
    def _register_builtin_checks(self) -> None:
        """Register built-in health checks for core components."""
        
        # API health check
        async def api_health_check() -> ComponentHealth:
            """Check API server health."""
            try:
                # Basic API responsiveness check
                start_time = time.time()
                
                # Simulate API check (in real implementation, make actual request)
                await asyncio.sleep(0.001)  # Minimal delay
                
                response_time = (time.time() - start_time) * 1000
                
                metrics = HealthMetrics(
                    response_time_ms=response_time,
                    success_rate=1.0,
                    uptime_seconds=time.time() - self.system_start_time
                )
                
                status = ComponentStatus.HEALTHY
                if response_time > self.prd_thresholds["response_time_ms"]:
                    status = ComponentStatus.DEGRADED
                
                return ComponentHealth(
                    name="api",
                    status=status,
                    metrics=metrics,
                    message="API server operational"
                )
                
            except Exception as e:
                return ComponentHealth(
                    name="api",
                    status=ComponentStatus.UNHEALTHY,
                    metrics=HealthMetrics(),
                    message=f"API server error: {str(e)}"
                )
        
        # Quantum engine health check
        async def quantum_engine_health_check() -> ComponentHealth:
            """Check quantum similarity engine health."""
            try:
                from ..core.quantum.similarity_engine import QuantumSimilarityEngine
                
                # Test quantum engine availability
                start_time = time.time()
                
                # Basic engine initialization check
                engine = QuantumSimilarityEngine()
                
                computation_time = (time.time() - start_time) * 1000
                
                metrics = HealthMetrics(
                    response_time_ms=computation_time,
                    success_rate=1.0
                )
                
                status = ComponentStatus.HEALTHY
                if computation_time > self.prd_thresholds["similarity_computation_ms"]:
                    status = ComponentStatus.DEGRADED
                
                return ComponentHealth(
                    name="quantum_engine",
                    status=status,
                    metrics=metrics,
                    message="Quantum similarity engine operational",
                    details={
                        "n_qubits": getattr(engine, 'n_qubits', 4),
                        "backend": getattr(engine, 'backend', 'default')
                    }
                )
                
            except Exception as e:
                return ComponentHealth(
                    name="quantum_engine",
                    status=ComponentStatus.UNHEALTHY,
                    metrics=HealthMetrics(),
                    message=f"Quantum engine error: {str(e)}"
                )
        
        # Memory health check
        async def memory_health_check() -> ComponentHealth:
            """Check system memory usage."""
            try:
                memory_info = psutil.virtual_memory()
                process = psutil.Process()
                process_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                metrics = HealthMetrics(
                    memory_usage_mb=process_memory,
                    cpu_usage_percent=process.cpu_percent()
                )
                
                # Check against PRD targets
                memory_gb = process_memory / 1024
                status = ComponentStatus.HEALTHY
                
                if memory_gb > self.prd_thresholds["memory_usage_gb"]:
                    status = ComponentStatus.DEGRADED
                
                if memory_gb > self.prd_thresholds["memory_usage_gb"] * 1.5:
                    status = ComponentStatus.UNHEALTHY
                
                return ComponentHealth(
                    name="memory",
                    status=status,
                    metrics=metrics,
                    message=f"Memory usage: {memory_gb:.2f}GB",
                    details={
                        "total_system_memory_gb": memory_info.total / 1024 / 1024 / 1024,
                        "system_memory_usage_percent": memory_info.percent,
                        "process_memory_gb": memory_gb
                    }
                )
                
            except Exception as e:
                return ComponentHealth(
                    name="memory",
                    status=ComponentStatus.UNHEALTHY,
                    metrics=HealthMetrics(),
                    message=f"Memory check error: {str(e)}"
                )
        
        # Register all built-in checks
        self.register_health_check("api", api_health_check)
        self.register_health_check("quantum_engine", quantum_engine_health_check)
        self.register_health_check("memory", memory_health_check)
    
    def _calculate_overall_status(self, component_health: Dict[str, ComponentHealth]) -> str:
        """Calculate overall system status based on component health."""
        if not component_health:
            return ComponentStatus.UNKNOWN.value
        
        statuses = [health.status for health in component_health.values()]
        
        # If any component is unhealthy, system is unhealthy
        if ComponentStatus.UNHEALTHY in statuses:
            return ComponentStatus.UNHEALTHY.value
        
        # If any component is degraded, system is degraded
        if ComponentStatus.DEGRADED in statuses:
            return ComponentStatus.DEGRADED.value
        
        # All components healthy
        return ComponentStatus.HEALTHY.value
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics."""
        try:
            memory_info = psutil.virtual_memory()
            cpu_info = psutil.cpu_percent(interval=1)
            disk_info = psutil.disk_usage('/')
            
            return {
                "memory": {
                    "total_gb": memory_info.total / 1024 / 1024 / 1024,
                    "available_gb": memory_info.available / 1024 / 1024 / 1024,
                    "usage_percent": memory_info.percent
                },
                "cpu": {
                    "usage_percent": cpu_info,
                    "core_count": psutil.cpu_count()
                },
                "disk": {
                    "total_gb": disk_info.total / 1024 / 1024 / 1024,
                    "free_gb": disk_info.free / 1024 / 1024 / 1024,
                    "usage_percent": (disk_info.used / disk_info.total) * 100
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {str(e)}")
            return {}
    
    def _check_prd_compliance(
        self, 
        component_health: Dict[str, ComponentHealth],
        system_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check compliance with PRD performance targets."""
        compliance = {
            "overall_compliant": True,
            "violations": [],
            "targets": self.prd_thresholds.copy()
        }
        
        # Check component response times
        for name, health in component_health.items():
            if health.metrics.response_time_ms > self.prd_thresholds.get("response_time_ms", 200):
                compliance["violations"].append({
                    "component": name,
                    "metric": "response_time_ms",
                    "actual": health.metrics.response_time_ms,
                    "target": self.prd_thresholds["response_time_ms"]
                })
                compliance["overall_compliant"] = False
        
        # Check memory usage
        if system_metrics.get("memory", {}).get("usage_percent", 0) > 80:
            compliance["violations"].append({
                "component": "system",
                "metric": "memory_usage_percent",
                "actual": system_metrics["memory"]["usage_percent"],
                "target": 80
            })
            compliance["overall_compliant"] = False
        
        return compliance
    
    def _store_health_history(self, component_name: str, health: ComponentHealth) -> None:
        """Store health check result in history."""
        if component_name not in self.health_history:
            self.health_history[component_name] = []
        
        self.health_history[component_name].append(health)
        
        # Clean up old history entries
        cutoff_time = datetime.utcnow() - timedelta(hours=self.history_retention_hours)
        self.health_history[component_name] = [
            h for h in self.health_history[component_name]
            if h.last_check > cutoff_time
        ]
    
    def get_health_trends(self, component_name: str, hours: int = 1) -> Dict[str, Any]:
        """
        Get health trends for a component over time.
        
        Args:
            component_name: Component name
            hours: Number of hours to analyze
            
        Returns:
            Health trend analysis
        """
        if component_name not in self.health_history:
            return {"error": f"No history available for {component_name}"}
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        relevant_history = [
            h for h in self.health_history[component_name]
            if h.last_check > cutoff_time
        ]
        
        if not relevant_history:
            return {"error": f"No recent history for {component_name}"}
        
        # Calculate trends
        response_times = [h.metrics.response_time_ms for h in relevant_history]
        success_rates = [h.metrics.success_rate for h in relevant_history]
        
        return {
            "component": component_name,
            "time_window_hours": hours,
            "sample_count": len(relevant_history),
            "response_time": {
                "avg_ms": sum(response_times) / len(response_times),
                "min_ms": min(response_times),
                "max_ms": max(response_times)
            },
            "success_rate": {
                "avg": sum(success_rates) / len(success_rates),
                "min": min(success_rates),
                "max": max(success_rates)
            },
            "status_distribution": {
                status.value: sum(1 for h in relevant_history if h.status == status)
                for status in ComponentStatus
            }
        }


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker(config_manager: ConfigManager) -> HealthChecker:
    """
    Get or create global health checker instance.
    
    Args:
        config_manager: Configuration manager
        
    Returns:
        HealthChecker instance
    """
    global _health_checker
    
    if _health_checker is None:
        _health_checker = HealthChecker(config_manager)
    
    return _health_checker
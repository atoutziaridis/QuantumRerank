"""
Health service for QuantumRerank API.

This service provides health checks, status monitoring, and system diagnostics
for the API service and its components.
"""

import time
import psutil
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum

from ...core.rag_reranker import QuantumRAGReranker
from ...config.manager import ConfigManager
from ...monitoring.performance_monitor import PerformanceMonitor
from ...utils.logging_config import get_logger

logger = get_logger(__name__)


class ComponentStatus(str, Enum):
    """Component health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class HealthService:
    """
    Service for monitoring API and component health.
    
    Provides comprehensive health checks, system monitoring, and diagnostics
    for all service components.
    """
    
    def __init__(
        self,
        quantum_reranker: QuantumRAGReranker,
        config_manager: ConfigManager,
        performance_monitor: PerformanceMonitor
    ):
        """
        Initialize health service.
        
        Args:
            quantum_reranker: Quantum reranker instance
            config_manager: Configuration manager
            performance_monitor: Performance monitor
        """
        self.quantum_reranker = quantum_reranker
        self.config_manager = config_manager
        self.performance_monitor = performance_monitor
        self.logger = logger
        
        # Service startup time
        self.startup_time = time.time()
        
        # Health check history
        self.health_history: List[Dict[str, Any]] = []
        self.max_history_entries = 100
        
        # Component health cache
        self.component_health_cache = {}
        self.cache_ttl = 30  # seconds
        self.last_check_time = 0
    
    async def get_basic_health(self) -> Dict[str, Any]:
        """
        Get basic health status.
        
        Returns:
            Basic health information
        """
        try:
            # Check core components quickly
            components = await self._check_core_components()
            
            # Determine overall status
            overall_status = self._determine_overall_status(components)
            
            health_data = {
                "status": overall_status,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "version": "1.0.0",  # From configuration
                "uptime_seconds": time.time() - self.startup_time,
                "components": {name: status["status"] for name, status in components.items()}
            }
            
            self.logger.info(
                "Basic health check completed",
                extra={"health_status": overall_status}
            )
            
            return health_data
            
        except Exception as e:
            self.logger.error(f"Basic health check failed: {e}")
            return {
                "status": ComponentStatus.CRITICAL,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "error": str(e),
                "uptime_seconds": time.time() - self.startup_time
            }
    
    async def get_detailed_health(self) -> Dict[str, Any]:
        """
        Get detailed health status with comprehensive checks.
        
        Returns:
            Detailed health information
        """
        try:
            # Perform comprehensive component checks
            components = await self._check_all_components()
            
            # Get performance metrics
            performance_metrics = await self._get_performance_metrics()
            
            # Get resource usage
            resource_usage = await self._get_resource_usage()
            
            # Determine overall status
            overall_status = self._determine_overall_status(components)
            
            # Create detailed response
            detailed_health = {
                "status": overall_status,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "version": "1.0.0",
                "uptime_seconds": time.time() - self.startup_time,
                "components": components,
                "performance_metrics": performance_metrics,
                "resource_usage": resource_usage,
                "system_info": await self._get_system_info(),
                "health_summary": self._generate_health_summary(components)
            }
            
            # Store in history
            self._store_health_check(detailed_health)
            
            self.logger.info(
                "Detailed health check completed",
                extra={
                    "health_status": overall_status,
                    "component_count": len(components)
                }
            )
            
            return detailed_health
            
        except Exception as e:
            self.logger.error(f"Detailed health check failed: {e}")
            return {
                "status": ComponentStatus.CRITICAL,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "error": str(e),
                "uptime_seconds": time.time() - self.startup_time
            }
    
    async def _check_core_components(self) -> Dict[str, Dict[str, Any]]:
        """Check status of core components."""
        components = {}
        
        # Check quantum reranker
        components["quantum_reranker"] = await self._check_quantum_reranker()
        
        # Check configuration manager
        components["config_manager"] = await self._check_config_manager()
        
        # Check performance monitor
        components["performance_monitor"] = await self._check_performance_monitor()
        
        return components
    
    async def _check_all_components(self) -> Dict[str, Dict[str, Any]]:
        """Check status of all components with detailed information."""
        # Use cache if available and fresh
        current_time = time.time()
        if (current_time - self.last_check_time) < self.cache_ttl and self.component_health_cache:
            return self.component_health_cache
        
        components = {}
        
        # Check quantum reranker with details
        components["quantum_reranker"] = await self._check_quantum_reranker_detailed()
        
        # Check configuration manager with details
        components["config_manager"] = await self._check_config_manager_detailed()
        
        # Check performance monitor with details
        components["performance_monitor"] = await self._check_performance_monitor_detailed()
        
        # Check system dependencies
        components["system_dependencies"] = await self._check_system_dependencies()
        
        # Check memory usage
        components["memory"] = await self._check_memory_usage()
        
        # Check disk usage
        components["disk"] = await self._check_disk_usage()
        
        # Update cache
        self.component_health_cache = components
        self.last_check_time = current_time
        
        return components
    
    async def _check_quantum_reranker(self) -> Dict[str, Any]:
        """Basic quantum reranker health check."""
        try:
            if self.quantum_reranker is None:
                return {
                    "status": ComponentStatus.CRITICAL,
                    "message": "Quantum reranker not initialized"
                }
            
            # Quick functionality test
            test_result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.quantum_reranker.compute_similarity,
                "test",
                "test",
                "classical"
            )
            
            if isinstance(test_result, (int, float)) and 0 <= test_result <= 1:
                return {
                    "status": ComponentStatus.HEALTHY,
                    "message": "Quantum reranker operational"
                }
            else:
                return {
                    "status": ComponentStatus.WARNING,
                    "message": "Quantum reranker returning unexpected results"
                }
                
        except Exception as e:
            return {
                "status": ComponentStatus.CRITICAL,
                "message": f"Quantum reranker error: {str(e)}"
            }
    
    async def _check_quantum_reranker_detailed(self) -> Dict[str, Any]:
        """Detailed quantum reranker health check."""
        basic_check = await self._check_quantum_reranker()
        
        try:
            # Get performance stats
            stats = self.quantum_reranker.get_performance_stats()
            
            # Test different methods
            method_tests = {}
            for method in ["classical", "quantum", "hybrid"]:
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.quantum_reranker.compute_similarity,
                        "quantum",
                        "computing",
                        method
                    )
                    method_tests[method] = {
                        "status": "working",
                        "result": result
                    }
                except Exception as e:
                    method_tests[method] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            basic_check.update({
                "performance_stats": stats,
                "method_tests": method_tests,
                "embedding_model": getattr(self.quantum_reranker, 'model_name', 'unknown'),
                "device": getattr(self.quantum_reranker, 'device', 'unknown')
            })
            
        except Exception as e:
            basic_check["detailed_check_error"] = str(e)
        
        return basic_check
    
    async def _check_config_manager(self) -> Dict[str, Any]:
        """Basic configuration manager health check."""
        try:
            if self.config_manager is None:
                return {
                    "status": ComponentStatus.CRITICAL,
                    "message": "Configuration manager not initialized"
                }
            
            # Test configuration access
            config = self.config_manager.get_config()
            if config:
                return {
                    "status": ComponentStatus.HEALTHY,
                    "message": "Configuration manager operational"
                }
            else:
                return {
                    "status": ComponentStatus.WARNING,
                    "message": "Configuration manager returns empty config"
                }
                
        except Exception as e:
            return {
                "status": ComponentStatus.CRITICAL,
                "message": f"Configuration manager error: {str(e)}"
            }
    
    async def _check_config_manager_detailed(self) -> Dict[str, Any]:
        """Detailed configuration manager health check."""
        basic_check = await self._check_config_manager()
        
        try:
            config = self.config_manager.get_config()
            
            basic_check.update({
                "config_sections": list(config.__dict__.keys()) if config else [],
                "watching_enabled": hasattr(self.config_manager, '_watching') and self.config_manager._watching,
                "last_reload": getattr(self.config_manager, '_last_reload_time', 'unknown')
            })
            
        except Exception as e:
            basic_check["detailed_check_error"] = str(e)
        
        return basic_check
    
    async def _check_performance_monitor(self) -> Dict[str, Any]:
        """Basic performance monitor health check."""
        try:
            if self.performance_monitor is None:
                return {
                    "status": ComponentStatus.CRITICAL,
                    "message": "Performance monitor not initialized"
                }
            
            # Test metrics access
            metrics = self.performance_monitor.get_current_metrics()
            if isinstance(metrics, dict):
                return {
                    "status": ComponentStatus.HEALTHY,
                    "message": "Performance monitor operational"
                }
            else:
                return {
                    "status": ComponentStatus.WARNING,
                    "message": "Performance monitor not returning metrics"
                }
                
        except Exception as e:
            return {
                "status": ComponentStatus.CRITICAL,
                "message": f"Performance monitor error: {str(e)}"
            }
    
    async def _check_performance_monitor_detailed(self) -> Dict[str, Any]:
        """Detailed performance monitor health check."""
        basic_check = await self._check_performance_monitor()
        
        try:
            metrics = self.performance_monitor.get_current_metrics()
            
            basic_check.update({
                "current_metrics": metrics,
                "metrics_count": len(metrics) if isinstance(metrics, dict) else 0,
                "monitoring_active": hasattr(self.performance_monitor, '_monitoring_active')
            })
            
        except Exception as e:
            basic_check["detailed_check_error"] = str(e)
        
        return basic_check
    
    async def _check_system_dependencies(self) -> Dict[str, Any]:
        """Check system dependencies and libraries."""
        try:
            dependencies = {}
            
            # Check Python libraries
            try:
                import torch
                dependencies["torch"] = {
                    "status": "available",
                    "version": torch.__version__,
                    "cuda_available": torch.cuda.is_available()
                }
            except ImportError:
                dependencies["torch"] = {"status": "missing"}
            
            try:
                import qiskit
                dependencies["qiskit"] = {
                    "status": "available",
                    "version": qiskit.__version__
                }
            except ImportError:
                dependencies["qiskit"] = {"status": "missing"}
            
            try:
                import sentence_transformers
                dependencies["sentence_transformers"] = {
                    "status": "available",
                    "version": sentence_transformers.__version__
                }
            except ImportError:
                dependencies["sentence_transformers"] = {"status": "missing"}
            
            # Determine overall dependency status
            missing_deps = [name for name, info in dependencies.items() if info["status"] == "missing"]
            
            if not missing_deps:
                status = ComponentStatus.HEALTHY
                message = "All dependencies available"
            elif len(missing_deps) < len(dependencies) / 2:
                status = ComponentStatus.WARNING
                message = f"Some dependencies missing: {missing_deps}"
            else:
                status = ComponentStatus.CRITICAL
                message = f"Critical dependencies missing: {missing_deps}"
            
            return {
                "status": status,
                "message": message,
                "dependencies": dependencies
            }
            
        except Exception as e:
            return {
                "status": ComponentStatus.CRITICAL,
                "message": f"Dependency check error: {str(e)}"
            }
    
    async def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage status."""
        try:
            memory = psutil.virtual_memory()
            
            # Determine status based on usage
            if memory.percent < 70:
                status = ComponentStatus.HEALTHY
                message = "Memory usage normal"
            elif memory.percent < 85:
                status = ComponentStatus.WARNING
                message = "Memory usage elevated"
            else:
                status = ComponentStatus.CRITICAL
                message = "Memory usage critical"
            
            return {
                "status": status,
                "message": message,
                "usage_percent": memory.percent,
                "available_gb": memory.available / (1024**3),
                "total_gb": memory.total / (1024**3),
                "used_gb": memory.used / (1024**3)
            }
            
        except Exception as e:
            return {
                "status": ComponentStatus.UNKNOWN,
                "message": f"Memory check error: {str(e)}"
            }
    
    async def _check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage status."""
        try:
            disk = psutil.disk_usage('/')
            
            # Determine status based on usage
            usage_percent = (disk.used / disk.total) * 100
            
            if usage_percent < 80:
                status = ComponentStatus.HEALTHY
                message = "Disk usage normal"
            elif usage_percent < 90:
                status = ComponentStatus.WARNING
                message = "Disk usage elevated"
            else:
                status = ComponentStatus.CRITICAL
                message = "Disk usage critical"
            
            return {
                "status": status,
                "message": message,
                "usage_percent": usage_percent,
                "free_gb": disk.free / (1024**3),
                "total_gb": disk.total / (1024**3),
                "used_gb": disk.used / (1024**3)
            }
            
        except Exception as e:
            return {
                "status": ComponentStatus.UNKNOWN,
                "message": f"Disk check error: {str(e)}"
            }
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        try:
            if self.performance_monitor:
                return self.performance_monitor.get_current_metrics()
            else:
                return {}
        except Exception as e:
            self.logger.warning(f"Failed to get performance metrics: {e}")
            return {"error": str(e)}
    
    async def _get_resource_usage(self) -> Dict[str, Any]:
        """Get system resource usage."""
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_percent": cpu_usage,
                "memory_percent": memory.percent,
                "disk_percent": (disk.used / disk.total) * 100,
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        try:
            return {
                "platform": psutil.os.name,
                "cpu_count": psutil.cpu_count(),
                "boot_time": psutil.boot_time(),
                "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _determine_overall_status(self, components: Dict[str, Dict[str, Any]]) -> str:
        """Determine overall health status from component statuses."""
        statuses = [comp.get("status", ComponentStatus.UNKNOWN) for comp in components.values()]
        
        if any(status == ComponentStatus.CRITICAL for status in statuses):
            return ComponentStatus.CRITICAL
        elif any(status == ComponentStatus.WARNING for status in statuses):
            return ComponentStatus.WARNING
        elif all(status == ComponentStatus.HEALTHY for status in statuses):
            return ComponentStatus.HEALTHY
        else:
            return ComponentStatus.UNKNOWN
    
    def _generate_health_summary(self, components: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate health summary from component data."""
        status_counts = {}
        for comp in components.values():
            status = comp.get("status", ComponentStatus.UNKNOWN)
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_components": len(components),
            "healthy_components": status_counts.get(ComponentStatus.HEALTHY, 0),
            "warning_components": status_counts.get(ComponentStatus.WARNING, 0),
            "critical_components": status_counts.get(ComponentStatus.CRITICAL, 0),
            "unknown_components": status_counts.get(ComponentStatus.UNKNOWN, 0)
        }
    
    def _store_health_check(self, health_data: Dict[str, Any]) -> None:
        """Store health check in history."""
        self.health_history.append({
            "timestamp": health_data["timestamp"],
            "status": health_data["status"],
            "component_summary": health_data.get("health_summary", {})
        })
        
        # Keep only recent entries
        if len(self.health_history) > self.max_history_entries:
            self.health_history = self.health_history[-self.max_history_entries:]
    
    def get_health_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get health check history.
        
        Args:
            hours: Number of hours of history to return
            
        Returns:
            List of health check entries
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        cutoff_timestamp = cutoff_time.isoformat() + "Z"
        
        return [
            entry for entry in self.health_history
            if entry["timestamp"] >= cutoff_timestamp
        ]
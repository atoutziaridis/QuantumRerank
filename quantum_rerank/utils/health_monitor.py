"""
Health monitoring and diagnostics for QuantumRerank.

This module provides comprehensive health checks, system diagnostics,
and performance monitoring to ensure robust operation.

Implements PRD Section 6.1: Technical Risks and Mitigation through health monitoring.
"""

import time
import psutil
import threading
import gc
from typing import Dict, List, Optional, Any, Callable, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from .logging_config import get_logger
from .exceptions import QuantumRerankException, PerformanceError, ErrorContext

logger = get_logger("health_monitor")


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check definition."""
    name: str
    description: str
    check_func: Callable[[], bool]
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    interval_s: float = 60.0
    timeout_s: float = 10.0
    enabled: bool = True
    
    # Status tracking
    last_run: Optional[datetime] = None
    last_status: HealthStatus = HealthStatus.UNKNOWN
    last_value: Optional[float] = None
    failure_count: int = 0
    consecutive_failures: int = 0


@dataclass
class SystemHealth:
    """Overall system health status."""
    overall_status: HealthStatus = HealthStatus.UNKNOWN
    last_updated: Optional[datetime] = None
    
    # Component health
    component_status: Dict[str, HealthStatus] = field(default_factory=dict)
    check_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # System metrics
    memory_usage_mb: float = 0.0
    cpu_usage_pct: float = 0.0
    active_threads: int = 0
    
    # Performance metrics
    avg_response_time_ms: float = 0.0
    error_rate_pct: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "overall_status": self.overall_status.value,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "component_status": {k: v.value for k, v in self.component_status.items()},
            "check_results": self.check_results,
            "system_metrics": {
                "memory_usage_mb": self.memory_usage_mb,
                "cpu_usage_pct": self.cpu_usage_pct,
                "active_threads": self.active_threads
            },
            "performance_metrics": {
                "avg_response_time_ms": self.avg_response_time_ms,
                "error_rate_pct": self.error_rate_pct
            }
        }


class HealthMonitor:
    """
    Comprehensive health monitoring system for QuantumRerank.
    
    Monitors system health, component availability, performance metrics,
    and provides diagnostics for troubleshooting.
    """
    
    def __init__(self, monitoring_interval_s: float = 30.0):
        """
        Initialize health monitor.
        
        Args:
            monitoring_interval_s: Interval between health checks
        """
        self.monitoring_interval_s = monitoring_interval_s
        self.health_checks: Dict[str, HealthCheck] = {}
        self.current_health = SystemHealth()
        
        # Monitoring control
        self._monitoring = False
        self._monitor_thread = None
        self._lock = threading.Lock()
        
        # Performance tracking
        self._response_times: List[float] = []
        self._error_counts: Dict[str, int] = {}
        self._operation_counts: Dict[str, int] = {}
        
        # Register default health checks
        self._register_default_checks()
        
        logger.info("Health monitor initialized")
    
    def _register_default_checks(self):
        """Register default system health checks."""
        # Memory usage check
        self.register_check(HealthCheck(
            name="memory_usage",
            description="System memory usage",
            check_func=self._check_memory_usage,
            warning_threshold=1024.0,  # 1GB
            critical_threshold=2048.0,  # 2GB (PRD limit)
            interval_s=30.0
        ))
        
        # CPU usage check
        self.register_check(HealthCheck(
            name="cpu_usage",
            description="CPU utilization",
            check_func=self._check_cpu_usage,
            warning_threshold=70.0,
            critical_threshold=90.0,
            interval_s=30.0
        ))
        
        # Response time check
        self.register_check(HealthCheck(
            name="response_time",
            description="Average response time",
            check_func=self._check_response_time,
            warning_threshold=200.0,  # 200ms
            critical_threshold=500.0,  # 500ms (PRD limit)
            interval_s=60.0
        ))
        
        # Error rate check
        self.register_check(HealthCheck(
            name="error_rate",
            description="System error rate",
            check_func=self._check_error_rate,
            warning_threshold=5.0,  # 5%
            critical_threshold=10.0,  # 10%
            interval_s=60.0
        ))
        
        # Component availability checks
        self.register_check(HealthCheck(
            name="embedding_processor",
            description="Embedding processor availability",
            check_func=self._check_embedding_processor,
            interval_s=120.0
        ))
        
        self.register_check(HealthCheck(
            name="quantum_engine",
            description="Quantum similarity engine availability",
            check_func=self._check_quantum_engine,
            interval_s=120.0
        ))
    
    def register_check(self, health_check: HealthCheck):
        """
        Register a new health check.
        
        Args:
            health_check: Health check to register
        """
        with self._lock:
            self.health_checks[health_check.name] = health_check
        
        logger.info(f"Registered health check: {health_check.name}")
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._monitoring:
            logger.warning("Health monitoring already running")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                self.run_health_checks()
                time.sleep(self.monitoring_interval_s)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval_s)
    
    def run_health_checks(self) -> SystemHealth:
        """
        Run all enabled health checks and update system health.
        
        Returns:
            Current system health status
        """
        logger.debug("Running health checks")
        
        with self._lock:
            current_time = datetime.now()
            
            # Run individual checks
            for check in self.health_checks.values():
                if not check.enabled:
                    continue
                
                # Check if it's time to run this check
                if (check.last_run is None or 
                    (current_time - check.last_run).total_seconds() >= check.interval_s):
                    
                    self._run_single_check(check, current_time)
            
            # Update overall system health
            self._update_system_health(current_time)
        
        return self.current_health
    
    def _run_single_check(self, check: HealthCheck, current_time: datetime):
        """Run a single health check."""
        try:
            start_time = time.time()
            
            # Run the check function
            result = check.check_func()
            check_duration = time.time() - start_time
            
            # Handle boolean result
            if isinstance(result, bool):
                check.last_value = None
                check.last_status = HealthStatus.HEALTHY if result else HealthStatus.CRITICAL
            
            # Handle numeric result with thresholds
            elif isinstance(result, (int, float)):
                check.last_value = float(result)
                
                if check.critical_threshold and result >= check.critical_threshold:
                    check.last_status = HealthStatus.CRITICAL
                elif check.warning_threshold and result >= check.warning_threshold:
                    check.last_status = HealthStatus.WARNING
                else:
                    check.last_status = HealthStatus.HEALTHY
            
            else:
                check.last_status = HealthStatus.UNKNOWN
                logger.warning(f"Health check {check.name} returned unexpected type: {type(result)}")
            
            # Update check metadata
            check.last_run = current_time
            
            if check.last_status == HealthStatus.HEALTHY:
                check.consecutive_failures = 0
            else:
                check.failure_count += 1
                check.consecutive_failures += 1
            
            # Store check result
            self.current_health.check_results[check.name] = {
                "status": check.last_status.value,
                "value": check.last_value,
                "duration_ms": check_duration * 1000,
                "timestamp": current_time.isoformat(),
                "failure_count": check.failure_count,
                "consecutive_failures": check.consecutive_failures
            }
            
            logger.debug(f"Health check {check.name}: {check.last_status.value} "
                        f"(value: {check.last_value})")
        
        except Exception as e:
            logger.error(f"Health check {check.name} failed: {e}")
            check.last_status = HealthStatus.CRITICAL
            check.failure_count += 1
            check.consecutive_failures += 1
            check.last_run = current_time
            
            self.current_health.check_results[check.name] = {
                "status": HealthStatus.CRITICAL.value,
                "error": str(e),
                "timestamp": current_time.isoformat(),
                "failure_count": check.failure_count,
                "consecutive_failures": check.consecutive_failures
            }
    
    def _update_system_health(self, current_time: datetime):
        """Update overall system health based on individual checks."""
        # Determine overall status
        critical_count = 0
        warning_count = 0
        healthy_count = 0
        
        component_status = {}
        
        for check_name, check in self.health_checks.items():
            if not check.enabled:
                continue
            
            component_status[check_name] = check.last_status
            
            if check.last_status == HealthStatus.CRITICAL:
                critical_count += 1
            elif check.last_status == HealthStatus.WARNING:
                warning_count += 1
            elif check.last_status == HealthStatus.HEALTHY:
                healthy_count += 1
        
        # Determine overall status
        if critical_count > 0:
            overall_status = HealthStatus.CRITICAL
        elif warning_count > 0:
            overall_status = HealthStatus.WARNING
        elif healthy_count > 0:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        # Update system health
        self.current_health.overall_status = overall_status
        self.current_health.last_updated = current_time
        self.current_health.component_status = component_status
        
        # Update system metrics
        self.current_health.memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
        self.current_health.cpu_usage_pct = psutil.cpu_percent()
        self.current_health.active_threads = threading.active_count()
        
        # Update performance metrics
        if self._response_times:
            self.current_health.avg_response_time_ms = sum(self._response_times) / len(self._response_times)
        
        total_operations = sum(self._operation_counts.values())
        total_errors = sum(self._error_counts.values())
        if total_operations > 0:
            self.current_health.error_rate_pct = (total_errors / total_operations) * 100
    
    def get_health_status(self) -> SystemHealth:
        """
        Get current system health status.
        
        Returns:
            Current system health
        """
        return self.current_health
    
    def record_operation(self, operation_name: str, duration_ms: float, success: bool):
        """
        Record operation metrics for health monitoring.
        
        Args:
            operation_name: Name of the operation
            duration_ms: Operation duration in milliseconds
            success: Whether operation succeeded
        """
        with self._lock:
            # Track response times (keep last 1000)
            self._response_times.append(duration_ms)
            if len(self._response_times) > 1000:
                self._response_times.pop(0)
            
            # Track operation counts
            self._operation_counts[operation_name] = self._operation_counts.get(operation_name, 0) + 1
            
            # Track error counts
            if not success:
                self._error_counts[operation_name] = self._error_counts.get(operation_name, 0) + 1
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive system diagnostics.
        
        Returns:
            Detailed diagnostic information
        """
        # System information
        memory_info = psutil.virtual_memory()
        process = psutil.Process()
        
        diagnostics = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": memory_info.total / (1024**3),
                "memory_available_gb": memory_info.available / (1024**3),
                "memory_percent": memory_info.percent,
                "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}"
            },
            "process_info": {
                "pid": process.pid,
                "memory_rss_mb": process.memory_info().rss / 1024 / 1024,
                "memory_vms_mb": process.memory_info().vms / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "create_time": datetime.fromtimestamp(process.create_time()).isoformat()
            },
            "health_status": self.current_health.to_dict(),
            "performance_metrics": {
                "total_operations": sum(self._operation_counts.values()),
                "total_errors": sum(self._error_counts.values()),
                "operation_breakdown": self._operation_counts.copy(),
                "error_breakdown": self._error_counts.copy(),
                "avg_response_time_ms": self.current_health.avg_response_time_ms,
                "error_rate_pct": self.current_health.error_rate_pct
            }
        }
        
        # Garbage collection info
        gc_stats = gc.get_stats()
        if gc_stats:
            diagnostics["gc_info"] = {
                "collections": gc_stats,
                "unreachable_objects": len(gc.garbage),
                "threshold": gc.get_threshold()
            }
        
        return diagnostics
    
    # Health check implementations
    
    def _check_memory_usage(self) -> float:
        """Check current memory usage."""
        return psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    def _check_cpu_usage(self) -> float:
        """Check current CPU usage."""
        return psutil.cpu_percent(interval=1.0)
    
    def _check_response_time(self) -> float:
        """Check average response time."""
        if not self._response_times:
            return 0.0
        return sum(self._response_times) / len(self._response_times)
    
    def _check_error_rate(self) -> float:
        """Check current error rate."""
        total_operations = sum(self._operation_counts.values())
        total_errors = sum(self._error_counts.values())
        
        if total_operations == 0:
            return 0.0
        
        return (total_errors / total_operations) * 100
    
    def _check_embedding_processor(self) -> bool:
        """Check embedding processor availability."""
        try:
            from ..core.embeddings import EmbeddingProcessor
            processor = EmbeddingProcessor()
            
            # Simple test encoding
            test_result = processor.encode_texts(["test"])
            return len(test_result) > 0 and len(test_result[0]) > 0
        
        except Exception as e:
            logger.warning(f"Embedding processor check failed: {e}")
            return False
    
    def _check_quantum_engine(self) -> bool:
        """Check quantum similarity engine availability."""
        try:
            from ..core.quantum_similarity_engine import QuantumSimilarityEngine
            engine = QuantumSimilarityEngine()
            
            # Simple test computation
            test_result = engine.compute_similarity("test1", "test2")
            return test_result[0] is not None
        
        except Exception as e:
            logger.warning(f"Quantum engine check failed: {e}")
            return False


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor


def start_health_monitoring():
    """Start global health monitoring."""
    monitor = get_health_monitor()
    monitor.start_monitoring()


def stop_health_monitoring():
    """Stop global health monitoring."""
    global _health_monitor
    if _health_monitor:
        _health_monitor.stop_monitoring()


def get_system_health() -> Dict[str, Any]:
    """Get current system health status."""
    monitor = get_health_monitor()
    return monitor.get_health_status().to_dict()


def record_operation_metric(operation_name: str, duration_ms: float, success: bool):
    """Record operation metrics for health monitoring."""
    monitor = get_health_monitor()
    monitor.record_operation(operation_name, duration_ms, success)
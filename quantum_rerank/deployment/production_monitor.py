"""
Production Monitoring for Edge Deployment

Provides monitoring, metrics collection, and alerting capabilities for
production deployment of quantum-inspired RAG systems.
"""

import time
import json
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels for monitoring."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MonitoringConfig:
    """Configuration for production monitoring."""
    enable_metrics: bool = True
    enable_alerting: bool = True
    enable_health_checks: bool = True
    metrics_interval_seconds: int = 30
    health_check_interval_seconds: int = 10
    alert_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                "memory_usage_percent": 80.0,
                "cpu_usage_percent": 80.0,
                "latency_ms": 1000.0,
                "error_rate_percent": 5.0
            }


@dataclass
class SystemMetrics:
    """System metrics for monitoring."""
    timestamp: float
    memory_usage_mb: float
    memory_usage_percent: float
    cpu_usage_percent: float
    disk_usage_percent: float
    network_io_bytes: int
    gpu_usage_percent: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    timestamp: float
    requests_per_second: float
    average_latency_ms: float
    error_rate_percent: float
    active_connections: int
    cache_hit_rate: float
    compression_ratio: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


@dataclass
class Alert:
    """Alert definition."""
    alert_id: str
    timestamp: float
    severity: str  # "warning", "critical", "info"
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return asdict(self)


class ProductionMonitor:
    """Production monitoring system for edge deployments."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.system_metrics_history: List[SystemMetrics] = []
        self.app_metrics_history: List[ApplicationMetrics] = []
        self.alerts: List[Alert] = []
        self.monitoring_thread: Optional[threading.Thread] = None
        self.health_check_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.health_status = "healthy"
        self.custom_metrics = {}
        
        logger.info("Production monitor initialized")
    
    def start_monitoring(self):
        """Start monitoring threads."""
        if self.is_running:
            return
        
        self.is_running = True
        
        if self.config.enable_metrics:
            self.monitoring_thread = threading.Thread(target=self._metrics_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logger.info("Metrics monitoring started")
        
        if self.config.enable_health_checks:
            self.health_check_thread = threading.Thread(target=self._health_check_loop)
            self.health_check_thread.daemon = True
            self.health_check_thread.start()
            logger.info("Health check monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring threads."""
        self.is_running = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5)
        
        logger.info("Production monitoring stopped")
    
    def _metrics_loop(self):
        """Main metrics collection loop."""
        while self.is_running:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics_history.append(system_metrics)
                
                # Collect application metrics
                app_metrics = self._collect_application_metrics()
                self.app_metrics_history.append(app_metrics)
                
                # Check for alerts
                if self.config.enable_alerting:
                    self._check_alerts(system_metrics, app_metrics)
                
                # Trim history to prevent memory bloat
                self._trim_history()
                
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")
            
            time.sleep(self.config.metrics_interval_seconds)
    
    def _health_check_loop(self):
        """Health check loop."""
        while self.is_running:
            try:
                health_status = self._perform_health_check()
                self.health_status = health_status
                
                if health_status != "healthy":
                    self._create_alert(
                        "health_check_failed",
                        "critical",
                        f"Health check failed: {health_status}",
                        "health_status",
                        0.0,
                        1.0
                    )
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                self.health_status = "unhealthy"
            
            time.sleep(self.config.health_check_interval_seconds)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level metrics."""
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_usage_mb = memory.used / 1024 / 1024
        memory_usage_percent = memory.percent
        
        # CPU metrics
        cpu_usage_percent = psutil.cpu_percent(interval=1)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_usage_percent = disk.percent
        
        # Network metrics
        network = psutil.net_io_counters()
        network_io_bytes = network.bytes_sent + network.bytes_recv
        
        # GPU metrics (if available)
        gpu_usage_percent = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated()
                gpu_memory_total = torch.cuda.max_memory_allocated()
                if gpu_memory_total > 0:
                    gpu_usage_percent = (gpu_memory_used / gpu_memory_total) * 100
        except:
            pass
        
        return SystemMetrics(
            timestamp=time.time(),
            memory_usage_mb=memory_usage_mb,
            memory_usage_percent=memory_usage_percent,
            cpu_usage_percent=cpu_usage_percent,
            disk_usage_percent=disk_usage_percent,
            network_io_bytes=network_io_bytes,
            gpu_usage_percent=gpu_usage_percent
        )
    
    def _collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-specific metrics."""
        # Mock application metrics (in real implementation, these would come from the application)
        return ApplicationMetrics(
            timestamp=time.time(),
            requests_per_second=self.custom_metrics.get("requests_per_second", 10.0),
            average_latency_ms=self.custom_metrics.get("average_latency_ms", 50.0),
            error_rate_percent=self.custom_metrics.get("error_rate_percent", 0.1),
            active_connections=self.custom_metrics.get("active_connections", 5),
            cache_hit_rate=self.custom_metrics.get("cache_hit_rate", 0.95),
            compression_ratio=self.custom_metrics.get("compression_ratio", 8.0)
        )
    
    def _check_alerts(self, system_metrics: SystemMetrics, app_metrics: ApplicationMetrics):
        """Check for alert conditions."""
        thresholds = self.config.alert_thresholds
        
        # System metric alerts
        if system_metrics.memory_usage_percent > thresholds["memory_usage_percent"]:
            self._create_alert(
                "high_memory_usage",
                "warning",
                f"Memory usage is {system_metrics.memory_usage_percent:.1f}%",
                "memory_usage_percent",
                system_metrics.memory_usage_percent,
                thresholds["memory_usage_percent"]
            )
        
        if system_metrics.cpu_usage_percent > thresholds["cpu_usage_percent"]:
            self._create_alert(
                "high_cpu_usage",
                "warning",
                f"CPU usage is {system_metrics.cpu_usage_percent:.1f}%",
                "cpu_usage_percent",
                system_metrics.cpu_usage_percent,
                thresholds["cpu_usage_percent"]
            )
        
        # Application metric alerts
        if app_metrics.average_latency_ms > thresholds["latency_ms"]:
            self._create_alert(
                "high_latency",
                "critical",
                f"Average latency is {app_metrics.average_latency_ms:.1f}ms",
                "latency_ms",
                app_metrics.average_latency_ms,
                thresholds["latency_ms"]
            )
        
        if app_metrics.error_rate_percent > thresholds["error_rate_percent"]:
            self._create_alert(
                "high_error_rate",
                "critical",
                f"Error rate is {app_metrics.error_rate_percent:.1f}%",
                "error_rate_percent",
                app_metrics.error_rate_percent,
                thresholds["error_rate_percent"]
            )
    
    def _create_alert(self, alert_id: str, severity: str, message: str, 
                     metric_name: str, metric_value: float, threshold: float):
        """Create a new alert."""
        alert = Alert(
            alert_id=alert_id,
            timestamp=time.time(),
            severity=severity,
            message=message,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold
        )
        
        self.alerts.append(alert)
        logger.warning(f"Alert created: {alert.severity} - {alert.message}")
    
    def _perform_health_check(self) -> str:
        """Perform application health check."""
        try:
            # Check if system resources are available
            if psutil.virtual_memory().percent > 95:
                return "unhealthy - high memory usage"
            
            if psutil.cpu_percent(interval=1) > 95:
                return "unhealthy - high CPU usage"
            
            # Check disk space
            if psutil.disk_usage('/').percent > 95:
                return "unhealthy - low disk space"
            
            # Additional health checks can be added here
            return "healthy"
            
        except Exception as e:
            return f"unhealthy - health check error: {str(e)}"
    
    def _trim_history(self):
        """Trim metric history to prevent memory bloat."""
        max_history_size = 1000
        
        if len(self.system_metrics_history) > max_history_size:
            self.system_metrics_history = self.system_metrics_history[-max_history_size:]
        
        if len(self.app_metrics_history) > max_history_size:
            self.app_metrics_history = self.app_metrics_history[-max_history_size:]
        
        if len(self.alerts) > max_history_size:
            self.alerts = self.alerts[-max_history_size:]
    
    def update_custom_metric(self, metric_name: str, value: float):
        """Update a custom application metric."""
        self.custom_metrics[metric_name] = value
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system and application metrics."""
        return {
            "system_metrics": self.system_metrics_history[-1].to_dict() if self.system_metrics_history else {},
            "app_metrics": self.app_metrics_history[-1].to_dict() if self.app_metrics_history else {},
            "health_status": self.health_status,
            "custom_metrics": self.custom_metrics
        }
    
    def get_metrics_history(self, last_n: int = 100) -> Dict[str, Any]:
        """Get metrics history."""
        return {
            "system_metrics": [m.to_dict() for m in self.system_metrics_history[-last_n:]],
            "app_metrics": [m.to_dict() for m in self.app_metrics_history[-last_n:]],
            "alerts": [a.to_dict() for a in self.alerts[-last_n:]]
        }
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts from the last hour."""
        current_time = time.time()
        one_hour_ago = current_time - 3600
        
        active_alerts = [
            alert.to_dict() for alert in self.alerts
            if alert.timestamp > one_hour_ago
        ]
        
        return active_alerts
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        metrics_data = {
            "export_timestamp": time.time(),
            "system_metrics": [m.to_dict() for m in self.system_metrics_history],
            "app_metrics": [m.to_dict() for m in self.app_metrics_history],
            "alerts": [a.to_dict() for a in self.alerts],
            "health_status": self.health_status,
            "custom_metrics": self.custom_metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for monitoring dashboard."""
        if not self.system_metrics_history or not self.app_metrics_history:
            return {"status": "no_data"}
        
        # Recent metrics
        recent_system = self.system_metrics_history[-10:]
        recent_app = self.app_metrics_history[-10:]
        
        # Calculate averages
        avg_cpu = sum(m.cpu_usage_percent for m in recent_system) / len(recent_system)
        avg_memory = sum(m.memory_usage_percent for m in recent_system) / len(recent_system)
        avg_latency = sum(m.average_latency_ms for m in recent_app) / len(recent_app)
        avg_error_rate = sum(m.error_rate_percent for m in recent_app) / len(recent_app)
        
        # Get active alerts
        active_alerts = self.get_active_alerts()
        
        return {
            "status": "active",
            "health_status": self.health_status,
            "current_metrics": {
                "cpu_usage_percent": avg_cpu,
                "memory_usage_percent": avg_memory,
                "average_latency_ms": avg_latency,
                "error_rate_percent": avg_error_rate
            },
            "active_alerts": active_alerts,
            "alert_counts": {
                "critical": len([a for a in active_alerts if a["severity"] == "critical"]),
                "warning": len([a for a in active_alerts if a["severity"] == "warning"]),
                "info": len([a for a in active_alerts if a["severity"] == "info"])
            },
            "system_status": {
                "monitoring_active": self.is_running,
                "metrics_collected": len(self.system_metrics_history),
                "uptime_seconds": time.time() - (self.system_metrics_history[0].timestamp if self.system_metrics_history else time.time())
            }
        }


# Global monitor instance
_global_monitor: Optional[ProductionMonitor] = None


def get_monitor() -> Optional[ProductionMonitor]:
    """Get the global monitor instance."""
    return _global_monitor


def initialize_monitor(config: MonitoringConfig) -> ProductionMonitor:
    """Initialize the global monitor."""
    global _global_monitor
    _global_monitor = ProductionMonitor(config)
    return _global_monitor


def start_global_monitoring():
    """Start the global monitor."""
    if _global_monitor:
        _global_monitor.start_monitoring()


def stop_global_monitoring():
    """Stop the global monitor."""
    if _global_monitor:
        _global_monitor.stop_monitoring()
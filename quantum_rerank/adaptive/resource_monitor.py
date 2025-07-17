"""
Resource Monitor for Adaptive Compression

Monitors system resources in real-time to enable adaptive compression
and optimization based on current resource availability.
"""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ResourceThreshold(Enum):
    """Resource threshold levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_percent: float
    network_io_mb: float
    gpu_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


@dataclass
class ResourceAlert:
    """Resource alert information."""
    timestamp: float
    resource_type: str
    threshold: ResourceThreshold
    current_value: float
    threshold_value: float
    message: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return asdict(self)


class ResourceMonitor:
    """Real-time resource monitoring for adaptive optimization."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.metrics_history: List[ResourceMetrics] = []
        self.alerts: List[ResourceAlert] = []
        self.alert_callbacks: List[Callable] = []
        
        # Default thresholds
        self.thresholds = {
            "cpu_percent": {
                ResourceThreshold.LOW: 25.0,
                ResourceThreshold.MEDIUM: 50.0,
                ResourceThreshold.HIGH: 75.0,
                ResourceThreshold.CRITICAL: 90.0
            },
            "memory_percent": {
                ResourceThreshold.LOW: 30.0,
                ResourceThreshold.MEDIUM: 60.0,
                ResourceThreshold.HIGH: 80.0,
                ResourceThreshold.CRITICAL: 95.0
            },
            "disk_percent": {
                ResourceThreshold.LOW: 50.0,
                ResourceThreshold.MEDIUM: 70.0,
                ResourceThreshold.HIGH: 85.0,
                ResourceThreshold.CRITICAL: 95.0
            },
            "gpu_percent": {
                ResourceThreshold.LOW: 30.0,
                ResourceThreshold.MEDIUM: 60.0,
                ResourceThreshold.HIGH: 80.0,
                ResourceThreshold.CRITICAL: 95.0
            }
        }
        
        logger.info(f"Resource Monitor initialized with {monitoring_interval}s interval")
    
    def add_alert_callback(self, callback: Callable[[ResourceAlert], None]):
        """Add callback for resource alerts."""
        self.alert_callbacks.append(callback)
    
    def set_threshold(self, resource_type: str, threshold: ResourceThreshold, value: float):
        """Set custom threshold for a resource type."""
        if resource_type not in self.thresholds:
            self.thresholds[resource_type] = {}
        
        self.thresholds[resource_type][threshold] = value
        logger.info(f"Threshold updated: {resource_type} {threshold.value} = {value}")
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check thresholds and generate alerts
                self._check_thresholds(metrics)
                
                # Trim history to prevent memory bloat
                self._trim_history()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_mb = memory.used / 1024 / 1024
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # Network metrics
        network = psutil.net_io_counters()
        network_io_mb = (network.bytes_sent + network.bytes_recv) / 1024 / 1024
        
        # GPU metrics (if available)
        gpu_percent = None
        gpu_memory_mb = None
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated()
                gpu_memory_total = torch.cuda.max_memory_allocated()
                
                if gpu_memory_total > 0:
                    gpu_percent = (gpu_memory_used / gpu_memory_total) * 100
                    gpu_memory_mb = gpu_memory_used / 1024 / 1024
        except:
            pass
        
        return ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_mb=memory_mb,
            disk_percent=disk_percent,
            network_io_mb=network_io_mb,
            gpu_percent=gpu_percent,
            gpu_memory_mb=gpu_memory_mb
        )
    
    def _check_thresholds(self, metrics: ResourceMetrics):
        """Check metrics against thresholds and generate alerts."""
        checks = [
            ("cpu_percent", metrics.cpu_percent),
            ("memory_percent", metrics.memory_percent),
            ("disk_percent", metrics.disk_percent)
        ]
        
        if metrics.gpu_percent is not None:
            checks.append(("gpu_percent", metrics.gpu_percent))
        
        for resource_type, current_value in checks:
            if resource_type not in self.thresholds:
                continue
            
            # Find the highest threshold exceeded
            exceeded_threshold = None
            for threshold in [ResourceThreshold.CRITICAL, ResourceThreshold.HIGH, 
                            ResourceThreshold.MEDIUM, ResourceThreshold.LOW]:
                if current_value >= self.thresholds[resource_type][threshold]:
                    exceeded_threshold = threshold
                    break
            
            if exceeded_threshold:
                # Check if we need to generate an alert
                if self._should_generate_alert(resource_type, exceeded_threshold):
                    alert = ResourceAlert(
                        timestamp=time.time(),
                        resource_type=resource_type,
                        threshold=exceeded_threshold,
                        current_value=current_value,
                        threshold_value=self.thresholds[resource_type][exceeded_threshold],
                        message=f"{resource_type} usage is {exceeded_threshold.value}: {current_value:.1f}%"
                    )
                    
                    self.alerts.append(alert)
                    
                    # Trigger callbacks
                    for callback in self.alert_callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            logger.error(f"Error in alert callback: {e}")
                    
                    logger.warning(f"Resource alert: {alert.message}")
    
    def _should_generate_alert(self, resource_type: str, threshold: ResourceThreshold) -> bool:
        """Check if we should generate an alert to avoid spam."""
        # Check if we've generated a similar alert recently
        recent_alerts = [a for a in self.alerts[-10:] 
                        if a.resource_type == resource_type and 
                        a.threshold == threshold and 
                        time.time() - a.timestamp < 60]  # Within last minute
        
        return len(recent_alerts) == 0
    
    def _trim_history(self):
        """Trim metrics history to prevent memory bloat."""
        max_history = 3600  # Keep 1 hour of data at 1s intervals
        
        if len(self.metrics_history) > max_history:
            self.metrics_history = self.metrics_history[-max_history:]
        
        # Trim alerts too
        max_alerts = 100
        if len(self.alerts) > max_alerts:
            self.alerts = self.alerts[-max_alerts:]
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get the most recent metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_history(self, duration_seconds: int = 300) -> List[ResourceMetrics]:
        """Get metrics history for specified duration."""
        if not self.metrics_history:
            return []
        
        cutoff_time = time.time() - duration_seconds
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_resource_utilization(self) -> Dict[str, Any]:
        """Get current resource utilization summary."""
        current = self.get_current_metrics()
        if not current:
            return {"status": "no_data"}
        
        return {
            "timestamp": current.timestamp,
            "cpu": {
                "percent": current.cpu_percent,
                "threshold": self._get_current_threshold("cpu_percent", current.cpu_percent)
            },
            "memory": {
                "percent": current.memory_percent,
                "mb": current.memory_mb,
                "threshold": self._get_current_threshold("memory_percent", current.memory_percent)
            },
            "disk": {
                "percent": current.disk_percent,
                "threshold": self._get_current_threshold("disk_percent", current.disk_percent)
            },
            "network": {
                "io_mb": current.network_io_mb
            },
            "gpu": {
                "percent": current.gpu_percent,
                "memory_mb": current.gpu_memory_mb,
                "threshold": self._get_current_threshold("gpu_percent", current.gpu_percent) if current.gpu_percent else None
            }
        }
    
    def _get_current_threshold(self, resource_type: str, current_value: float) -> Optional[str]:
        """Get current threshold level for a resource."""
        if resource_type not in self.thresholds:
            return None
        
        for threshold in [ResourceThreshold.CRITICAL, ResourceThreshold.HIGH, 
                        ResourceThreshold.MEDIUM, ResourceThreshold.LOW]:
            if current_value >= self.thresholds[resource_type][threshold]:
                return threshold.value
        
        return None
    
    def get_resource_availability(self) -> Dict[str, float]:
        """Get resource availability scores (0-1, higher is better)."""
        current = self.get_current_metrics()
        if not current:
            return {}
        
        availability = {
            "cpu": max(0.0, (100.0 - current.cpu_percent) / 100.0),
            "memory": max(0.0, (100.0 - current.memory_percent) / 100.0),
            "disk": max(0.0, (100.0 - current.disk_percent) / 100.0)
        }
        
        if current.gpu_percent is not None:
            availability["gpu"] = max(0.0, (100.0 - current.gpu_percent) / 100.0)
        
        return availability
    
    def get_recent_alerts(self, duration_seconds: int = 300) -> List[ResourceAlert]:
        """Get recent alerts within specified duration."""
        cutoff_time = time.time() - duration_seconds
        return [a for a in self.alerts if a.timestamp >= cutoff_time]
    
    def calculate_resource_pressure(self) -> Dict[str, Any]:
        """Calculate resource pressure metrics."""
        recent_metrics = self.get_metrics_history(duration_seconds=60)
        
        if len(recent_metrics) < 2:
            return {"status": "insufficient_data"}
        
        # Calculate average utilization
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_disk = sum(m.disk_percent for m in recent_metrics) / len(recent_metrics)
        
        # Calculate pressure score (0-1, higher is more pressure)
        pressure_score = (avg_cpu + avg_memory + avg_disk) / 300.0
        
        # Calculate trends
        cpu_trend = self._calculate_trend([m.cpu_percent for m in recent_metrics[-10:]])
        memory_trend = self._calculate_trend([m.memory_percent for m in recent_metrics[-10:]])
        
        return {
            "pressure_score": min(1.0, pressure_score),
            "average_utilization": {
                "cpu": avg_cpu,
                "memory": avg_memory,
                "disk": avg_disk
            },
            "trends": {
                "cpu": "increasing" if cpu_trend > 0.5 else "decreasing" if cpu_trend < -0.5 else "stable",
                "memory": "increasing" if memory_trend > 0.5 else "decreasing" if memory_trend < -0.5 else "stable"
            },
            "pressure_level": (
                "high" if pressure_score > 0.8 else
                "medium" if pressure_score > 0.5 else
                "low"
            )
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in resource usage."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression
        n = len(values)
        x = list(range(n))
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        if n * sum_x2 - sum_x ** 2 == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        return slope
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            "is_monitoring": self.is_monitoring,
            "monitoring_interval": self.monitoring_interval,
            "metrics_collected": len(self.metrics_history),
            "alerts_generated": len(self.alerts),
            "alert_callbacks": len(self.alert_callbacks),
            "thresholds": {
                resource_type: {threshold.value: value for threshold, value in thresholds.items()}
                for resource_type, thresholds in self.thresholds.items()
            }
        }
    
    def export_metrics(self, filepath: str, duration_seconds: int = 3600):
        """Export metrics to file."""
        metrics_data = {
            "export_timestamp": time.time(),
            "monitoring_stats": self.get_monitoring_stats(),
            "recent_metrics": [m.to_dict() for m in self.get_metrics_history(duration_seconds)],
            "recent_alerts": [a.to_dict() for a in self.get_recent_alerts(duration_seconds)],
            "resource_utilization": self.get_resource_utilization(),
            "resource_pressure": self.calculate_resource_pressure()
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")


# Utility functions
def create_resource_monitor(interval: float = 1.0) -> ResourceMonitor:
    """Create and configure resource monitor."""
    monitor = ResourceMonitor(interval)
    
    # Add default alert callback
    def log_alert(alert: ResourceAlert):
        logger.warning(f"Resource Alert: {alert.message}")
    
    monitor.add_alert_callback(log_alert)
    
    return monitor


def get_system_capabilities() -> Dict[str, Any]:
    """Get system capabilities and limits."""
    try:
        # CPU info
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # Memory info
        memory = psutil.virtual_memory()
        
        # Disk info
        disk = psutil.disk_usage('/')
        
        # GPU info
        gpu_available = False
        gpu_memory = None
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        except:
            pass
        
        return {
            "cpu": {
                "count": cpu_count,
                "frequency_mhz": cpu_freq.current if cpu_freq else None,
                "max_frequency_mhz": cpu_freq.max if cpu_freq else None
            },
            "memory": {
                "total_mb": memory.total / 1024 / 1024,
                "available_mb": memory.available / 1024 / 1024
            },
            "disk": {
                "total_gb": disk.total / 1024 / 1024 / 1024,
                "free_gb": disk.free / 1024 / 1024 / 1024
            },
            "gpu": {
                "available": gpu_available,
                "memory_mb": gpu_memory
            }
        }
    
    except Exception as e:
        logger.error(f"Error getting system capabilities: {e}")
        return {"error": str(e)}
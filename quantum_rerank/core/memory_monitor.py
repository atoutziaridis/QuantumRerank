"""
Memory monitoring for production stability.

Tracks memory usage and provides warnings when approaching limits
to prevent out-of-memory errors in production.
"""

import psutil
import gc
import time
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryThresholds:
    """Memory usage thresholds."""
    warning_threshold: float = 0.8     # 80% of limit
    critical_threshold: float = 0.9    # 90% of limit
    max_memory_gb: float = 2.0         # PRD: <2GB limit


@dataclass
class MemorySnapshot:
    """Point-in-time memory usage snapshot."""
    timestamp: datetime
    memory_mb: float
    memory_percent: float
    available_mb: float
    process_memory_mb: float
    process_memory_percent: float
    gc_collections: int


class MemoryMonitor:
    """
    Production memory monitoring with alerting and automatic cleanup.
    
    Tracks memory usage against PRD limits and provides warnings when
    approaching memory limits to prevent production issues.
    """
    
    def __init__(self, thresholds: Optional[MemoryThresholds] = None):
        """
        Initialize memory monitor.
        
        Args:
            thresholds: Memory threshold configuration
        """
        self.thresholds = thresholds or MemoryThresholds()
        self.process = psutil.Process()
        self.snapshots: List[MemorySnapshot] = []
        self.max_snapshots = 100  # Keep last 100 snapshots
        
        # Warning state tracking
        self.last_warning_time: Optional[datetime] = None
        self.warning_cooldown = timedelta(minutes=5)  # Don't spam warnings
        
        # GC tracking
        self.last_gc_count = sum(gc.get_count())
        
        logger.info("Memory monitor initialized", extra={
            "max_memory_gb": self.thresholds.max_memory_gb,
            "warning_threshold": self.thresholds.warning_threshold,
            "critical_threshold": self.thresholds.critical_threshold
        })
    
    def check_memory_usage(self) -> Dict[str, any]:
        """
        Check current memory usage and return detailed metrics.
        
        Returns:
            Dictionary with memory usage information
        """
        try:
            # System memory info
            system_memory = psutil.virtual_memory()
            
            # Process memory info
            process_memory = self.process.memory_info()
            process_percent = self.process.memory_percent()
            
            # Calculate limits
            max_memory_bytes = self.thresholds.max_memory_gb * 1024 * 1024 * 1024
            memory_mb = process_memory.rss / 1024 / 1024
            available_mb = system_memory.available / 1024 / 1024
            
            # GC info
            current_gc_count = sum(gc.get_count())
            gc_collections = current_gc_count - self.last_gc_count
            
            # Create usage snapshot
            usage = {
                "memory_mb": round(memory_mb, 2),
                "memory_percent": round(process_percent, 2),
                "memory_limit_gb": self.thresholds.max_memory_gb,
                "memory_usage_ratio": round(memory_mb / (self.thresholds.max_memory_gb * 1024), 3),
                "available_mb": round(available_mb, 2),
                "system_memory_total_gb": round(system_memory.total / 1024 / 1024 / 1024, 2),
                "system_memory_percent": system_memory.percent,
                "gc_collections_since_last_check": gc_collections,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            
            # Store snapshot
            snapshot = MemorySnapshot(
                timestamp=datetime.utcnow(),
                memory_mb=memory_mb,
                memory_percent=process_percent,
                available_mb=available_mb,
                process_memory_mb=memory_mb,
                process_memory_percent=process_percent,
                gc_collections=gc_collections
            )
            
            self.snapshots.append(snapshot)
            
            # Keep only recent snapshots
            if len(self.snapshots) > self.max_snapshots:
                self.snapshots = self.snapshots[-self.max_snapshots:]
            
            # Update GC count
            self.last_gc_count = current_gc_count
            
            # Check for warnings
            self._check_memory_warnings(usage, memory_mb, max_memory_bytes)
            
            return usage
            
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
    
    def _check_memory_warnings(self, usage: Dict, memory_mb: float, max_memory_bytes: float):
        """Check if memory warnings should be issued."""
        memory_ratio = memory_mb / (self.thresholds.max_memory_gb * 1024)
        current_time = datetime.utcnow()
        
        # Only warn if cooldown period has passed
        should_warn = (
            self.last_warning_time is None or 
            current_time - self.last_warning_time > self.warning_cooldown
        )
        
        if memory_ratio >= self.thresholds.critical_threshold:
            if should_warn:
                logger.critical(
                    "CRITICAL: Memory usage exceeds critical threshold",
                    extra={
                        "memory_mb": memory_mb,
                        "memory_percent": usage["memory_percent"],
                        "threshold": self.thresholds.critical_threshold,
                        "limit_gb": self.thresholds.max_memory_gb,
                        "suggested_action": "immediate_gc_or_restart"
                    }
                )
                self.last_warning_time = current_time
            
            # Force garbage collection at critical levels
            self._force_garbage_collection()
            
        elif memory_ratio >= self.thresholds.warning_threshold:
            if should_warn:
                logger.warning(
                    "High memory usage detected",
                    extra={
                        "memory_mb": memory_mb,
                        "memory_percent": usage["memory_percent"],
                        "threshold": self.thresholds.warning_threshold,
                        "limit_gb": self.thresholds.max_memory_gb,
                        "suggested_action": "monitor_closely"
                    }
                )
                self.last_warning_time = current_time
    
    def _force_garbage_collection(self):
        """Force garbage collection to free memory."""
        try:
            logger.info("Forcing garbage collection due to high memory usage")
            
            # Get memory before GC
            before_memory = self.process.memory_info().rss / 1024 / 1024
            
            # Force GC
            collected = gc.collect()
            
            # Get memory after GC
            after_memory = self.process.memory_info().rss / 1024 / 1024
            freed_mb = before_memory - after_memory
            
            logger.info(
                "Garbage collection completed",
                extra={
                    "objects_collected": collected,
                    "memory_before_mb": round(before_memory, 2),
                    "memory_after_mb": round(after_memory, 2),
                    "memory_freed_mb": round(freed_mb, 2)
                }
            )
            
        except Exception as e:
            logger.error(f"Error during forced garbage collection: {e}")
    
    def is_memory_available(self) -> bool:
        """
        Check if we have memory available for new requests.
        
        Returns:
            True if memory is available, False if approaching limits
        """
        try:
            process_memory = self.process.memory_info()
            max_memory_bytes = self.thresholds.max_memory_gb * 1024 * 1024 * 1024
            
            # Use critical threshold for availability check
            return process_memory.rss < max_memory_bytes * self.thresholds.critical_threshold
            
        except Exception as e:
            logger.error(f"Error checking memory availability: {e}")
            return False  # Assume not available on error
    
    def get_memory_trends(self) -> Dict[str, any]:
        """
        Get memory usage trends over time.
        
        Returns:
            Dictionary with trend analysis
        """
        if len(self.snapshots) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        try:
            recent_snapshots = self.snapshots[-10:]  # Last 10 snapshots
            
            # Calculate trends
            memory_values = [s.memory_mb for s in recent_snapshots]
            time_values = [(s.timestamp - recent_snapshots[0].timestamp).total_seconds() 
                          for s in recent_snapshots]
            
            # Simple linear trend calculation
            if len(memory_values) >= 2:
                avg_memory = sum(memory_values) / len(memory_values)
                memory_trend = (memory_values[-1] - memory_values[0]) / max(time_values[-1], 1)
                
                trend_direction = "increasing" if memory_trend > 0.1 else "decreasing" if memory_trend < -0.1 else "stable"
                
                return {
                    "current_memory_mb": round(memory_values[-1], 2),
                    "average_memory_mb": round(avg_memory, 2),
                    "memory_trend_mb_per_second": round(memory_trend, 4),
                    "trend_direction": trend_direction,
                    "samples_count": len(recent_snapshots),
                    "time_span_seconds": round(time_values[-1], 2),
                    "max_memory_mb": round(max(memory_values), 2),
                    "min_memory_mb": round(min(memory_values), 2)
                }
            else:
                return {"error": "Insufficient samples for trend calculation"}
                
        except Exception as e:
            logger.error(f"Error calculating memory trends: {e}")
            return {"error": str(e)}
    
    def get_memory_summary(self) -> Dict[str, any]:
        """
        Get comprehensive memory summary.
        
        Returns:
            Dictionary with complete memory status
        """
        current_usage = self.check_memory_usage()
        trends = self.get_memory_trends()
        
        return {
            "current_usage": current_usage,
            "trends": trends,
            "is_memory_available": self.is_memory_available(),
            "thresholds": {
                "warning_threshold": self.thresholds.warning_threshold,
                "critical_threshold": self.thresholds.critical_threshold,
                "max_memory_gb": self.thresholds.max_memory_gb
            },
            "monitoring_info": {
                "snapshots_count": len(self.snapshots),
                "last_warning_time": self.last_warning_time.isoformat() + "Z" if self.last_warning_time else None,
                "warning_cooldown_minutes": self.warning_cooldown.total_seconds() / 60
            }
        }


# Global memory monitor instance
memory_monitor = MemoryMonitor()


def get_memory_status() -> Dict[str, any]:
    """
    Get current memory status (convenience function).
    
    Returns:
        Memory status dictionary
    """
    return memory_monitor.get_memory_summary()


def check_memory_health() -> bool:
    """
    Quick memory health check.
    
    Returns:
        True if memory usage is healthy, False if concerning
    """
    return memory_monitor.is_memory_available()
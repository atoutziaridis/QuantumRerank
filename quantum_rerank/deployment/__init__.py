"""
Production Deployment Module for Edge-Ready RAG Systems.

This module provides comprehensive production deployment capabilities for
edge-deployed quantum-inspired RAG systems, including packaging, monitoring,
and lifecycle management.

Components:
- Edge deployment framework
- Production monitoring and observability
- Update and rollback mechanisms
- Performance optimization and tuning
"""

__version__ = "1.0.0"

from .edge_deployment import (
    EdgeDeployment,
    DeploymentConfig,
    DeploymentTarget
)

from .production_monitor import (
    ProductionMonitor,
    MonitoringConfig,
    AlertLevel
)

from .lifecycle_manager import (
    LifecycleManager,
    UpdateConfig,
    RollbackStrategy
)

__all__ = [
    "EdgeDeployment",
    "DeploymentConfig",
    "DeploymentTarget",
    "ProductionMonitor",
    "MonitoringConfig",
    "AlertLevel",
    "LifecycleManager",
    "UpdateConfig",
    "RollbackStrategy"
]
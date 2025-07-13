"""
Configuration integration with QuantumRerank components.

This module provides automatic configuration injection, setting change propagation,
and component reconfiguration handling for seamless configuration management.
"""

import threading
from typing import Dict, Any, Optional, Type, Callable, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

from .schemas import QuantumRerankConfigSchema
from .manager import ConfigManager, get_config_manager
from ..utils import get_logger


class Configurable(ABC):
    """Abstract base class for configurable components."""
    
    @abstractmethod
    def configure(self, config: QuantumRerankConfigSchema) -> None:
        """Configure component with new configuration."""
        pass
    
    @abstractmethod
    def validate_config(self, config: QuantumRerankConfigSchema) -> bool:
        """Validate configuration for this component."""
        pass
    
    def get_config_requirements(self) -> List[str]:
        """Get list of required configuration sections."""
        return []


@dataclass
class ComponentRegistration:
    """Registration information for a configurable component."""
    component: Configurable
    name: str
    config_sections: List[str]
    auto_reconfigure: bool = True
    critical: bool = False  # If True, configuration failures will raise exceptions


class ConfigurationIntegrator:
    """
    Configuration integration manager for QuantumRerank components.
    
    Provides automatic configuration injection and change propagation
    to registered components throughout the system.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager or get_config_manager()
        self.logger = get_logger(__name__)
        
        # Component registry
        self._components: Dict[str, ComponentRegistration] = {}
        self._components_lock = threading.RLock()
        
        # Configuration change tracking
        self._last_config_hash: Optional[str] = None
        
        # Register for configuration changes
        self.config_manager.register_change_callback(self._handle_config_change)
    
    def register_component(
        self,
        component: Configurable,
        name: str,
        config_sections: Optional[List[str]] = None,
        auto_reconfigure: bool = True,
        critical: bool = False
    ) -> None:
        """Register a component for configuration management."""
        with self._components_lock:
            # Use component's requirements if not specified
            if config_sections is None:
                config_sections = component.get_config_requirements()
            
            registration = ComponentRegistration(
                component=component,
                name=name,
                config_sections=config_sections,
                auto_reconfigure=auto_reconfigure,
                critical=critical
            )
            
            self._components[name] = registration
            
            # Apply current configuration to new component
            try:
                current_config = self.config_manager.get_config()
                if component.validate_config(current_config):
                    component.configure(current_config)
                    self.logger.info(f"Component '{name}' registered and configured successfully")
                else:
                    self.logger.warning(f"Component '{name}' registered but failed configuration validation")
            
            except Exception as e:
                self.logger.error(f"Failed to configure component '{name}' during registration: {str(e)}")
                if critical:
                    raise
    
    def unregister_component(self, name: str) -> bool:
        """Unregister a component from configuration management."""
        with self._components_lock:
            if name in self._components:
                del self._components[name]
                self.logger.info(f"Component '{name}' unregistered")
                return True
            return False
    
    def get_registered_components(self) -> List[str]:
        """Get list of registered component names."""
        with self._components_lock:
            return list(self._components.keys())
    
    def reconfigure_component(self, name: str, force: bool = False) -> bool:
        """Manually reconfigure a specific component."""
        with self._components_lock:
            if name not in self._components:
                self.logger.warning(f"Component '{name}' not registered")
                return False
            
            registration = self._components[name]
            
            try:
                current_config = self.config_manager.get_config()
                
                # Validate configuration if not forced
                if not force and not registration.component.validate_config(current_config):
                    self.logger.warning(f"Configuration validation failed for component '{name}'")
                    return False
                
                # Apply configuration
                registration.component.configure(current_config)
                self.logger.info(f"Component '{name}' reconfigured successfully")
                return True
            
            except Exception as e:
                self.logger.error(f"Failed to reconfigure component '{name}': {str(e)}")
                if registration.critical:
                    raise
                return False
    
    def reconfigure_all_components(self, force: bool = False) -> Dict[str, bool]:
        """Reconfigure all registered components."""
        results = {}
        
        with self._components_lock:
            for name in self._components:
                results[name] = self.reconfigure_component(name, force=force)
        
        return results
    
    def _handle_config_change(self, new_config: QuantumRerankConfigSchema) -> None:
        """Handle configuration change events."""
        self.logger.info("Configuration change detected, updating components")
        
        with self._components_lock:
            for name, registration in self._components.items():
                if not registration.auto_reconfigure:
                    continue
                
                try:
                    # Check if this component's configuration sections changed
                    if self._component_config_changed(registration, new_config):
                        self.reconfigure_component(name)
                
                except Exception as e:
                    self.logger.error(f"Failed to handle configuration change for component '{name}': {str(e)}")
    
    def _component_config_changed(
        self, 
        registration: ComponentRegistration, 
        new_config: QuantumRerankConfigSchema
    ) -> bool:
        """Check if component's relevant configuration sections changed."""
        # For now, assume any configuration change affects all components
        # In the future, this could be optimized to check specific sections
        return True
    
    def validate_all_components(self) -> Dict[str, bool]:
        """Validate current configuration against all registered components."""
        results = {}
        current_config = self.config_manager.get_config()
        
        with self._components_lock:
            for name, registration in self._components.items():
                try:
                    results[name] = registration.component.validate_config(current_config)
                except Exception as e:
                    self.logger.error(f"Validation failed for component '{name}': {str(e)}")
                    results[name] = False
        
        return results
    
    def get_component_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all registered components."""
        status = {}
        
        with self._components_lock:
            for name, registration in self._components.items():
                try:
                    current_config = self.config_manager.get_config()
                    is_valid = registration.component.validate_config(current_config)
                    
                    status[name] = {
                        "registered": True,
                        "auto_reconfigure": registration.auto_reconfigure,
                        "critical": registration.critical,
                        "config_sections": registration.config_sections,
                        "config_valid": is_valid
                    }
                
                except Exception as e:
                    status[name] = {
                        "registered": True,
                        "error": str(e),
                        "config_valid": False
                    }
        
        return status


# Component-specific configuration helpers

class QuantumEngineConfig:
    """Configuration helper for quantum engine components."""
    
    @staticmethod
    def extract_quantum_config(config: QuantumRerankConfigSchema) -> Dict[str, Any]:
        """Extract quantum-specific configuration."""
        return {
            "n_qubits": config.quantum.n_qubits,
            "max_circuit_depth": config.quantum.max_circuit_depth,
            "shots": config.quantum.shots,
            "simulator_method": config.quantum.simulator_method,
            "quantum_backends": config.quantum.quantum_backends,
            "enable_optimization": config.quantum.enable_optimization
        }


class MLModelConfig:
    """Configuration helper for ML model components."""
    
    @staticmethod
    def extract_ml_config(config: QuantumRerankConfigSchema) -> Dict[str, Any]:
        """Extract ML-specific configuration."""
        return {
            "embedding_model": config.ml.embedding_model,
            "embedding_dim": config.ml.embedding_dim,
            "batch_size": config.ml.batch_size,
            "max_sequence_length": config.ml.max_sequence_length,
            "use_quantum_compression": config.ml.use_quantum_compression,
            "compressed_dim": config.ml.compressed_dim,
            "parameter_prediction": config.ml.parameter_prediction
        }


class PerformanceConfig:
    """Configuration helper for performance monitoring components."""
    
    @staticmethod
    def extract_performance_config(config: QuantumRerankConfigSchema) -> Dict[str, Any]:
        """Extract performance-specific configuration."""
        return {
            "similarity_timeout_ms": config.performance.similarity_timeout_ms,
            "batch_timeout_ms": config.performance.batch_timeout_ms,
            "max_memory_gb": config.performance.max_memory_gb,
            "cache_size": config.performance.cache_size,
            "enable_caching": config.performance.enable_caching,
            "max_concurrent_requests": config.performance.max_concurrent_requests
        }


class APIConfig:
    """Configuration helper for API service components."""
    
    @staticmethod
    def extract_api_config(config: QuantumRerankConfigSchema) -> Dict[str, Any]:
        """Extract API-specific configuration."""
        return {
            "host": config.api.host,
            "port": config.api.port,
            "workers": config.api.workers,
            "rate_limit": config.api.rate_limit,
            "enable_auth": config.api.enable_auth,
            "cors_enabled": config.api.cors_enabled,
            "cors_origins": config.api.cors_origins,
            "request_timeout_s": config.api.request_timeout_s
        }


# Global configuration integrator instance
_global_integrator: Optional[ConfigurationIntegrator] = None


def get_configuration_integrator(config_manager: Optional[ConfigManager] = None) -> ConfigurationIntegrator:
    """Get or create global configuration integrator instance."""
    global _global_integrator
    
    if _global_integrator is None:
        _global_integrator = ConfigurationIntegrator(config_manager)
    
    return _global_integrator


def register_configurable_component(
    component: Configurable,
    name: str,
    config_sections: Optional[List[str]] = None,
    auto_reconfigure: bool = True,
    critical: bool = False
) -> None:
    """Register a component with the global configuration integrator."""
    integrator = get_configuration_integrator()
    integrator.register_component(
        component=component,
        name=name,
        config_sections=config_sections,
        auto_reconfigure=auto_reconfigure,
        critical=critical
    )


__all__ = [
    "Configurable",
    "ComponentRegistration",
    "ConfigurationIntegrator",
    "QuantumEngineConfig",
    "MLModelConfig", 
    "PerformanceConfig",
    "APIConfig",
    "get_configuration_integrator",
    "register_configurable_component"
]
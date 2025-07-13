"""
Configuration management system for QuantumRerank.

This module provides comprehensive configuration management including schemas,
validation, environment-specific configurations, dynamic management, and utilities.
"""

# Import legacy configuration classes for backward compatibility
from .settings import (
    QuantumConfig, ModelConfig, PerformanceConfig, 
    APIConfig, LoggingConfig, DEFAULT_CONFIG
)

# Import new configuration management system
from .schemas import (
    QuantumRerankConfigSchema, ValidationResult,
    QuantumConfigSchema, MLConfigSchema, PerformanceConfigSchema,
    APIConfigSchema, MonitoringConfigSchema,
    LogLevel, Environment, SimulatorMethod,
    validate_config_file
)

from .environments import (
    get_development_config, get_testing_config,
    get_staging_config, get_production_config,
    get_config_for_environment, get_environment_defaults,
    validate_environment_configs
)

from .manager import (
    ConfigManager, ConfigChangeEvent, ConfigHistory,
    get_config_manager, get_current_config, reload_config
)

from .integration import (
    Configurable, ConfigurationIntegrator,
    QuantumEngineConfig, MLModelConfig, PerformanceConfig as PerfConfig,
    APIConfig as APIConfigHelper,
    get_configuration_integrator, register_configurable_component
)

from .utils import (
    ConfigDiff, ConfigGenerator, ConfigComparator,
    ConfigMigrator, ConfigBackup, create_config_template
)

__all__ = [
    # Legacy compatibility
    "QuantumConfig", "ModelConfig", "PerformanceConfig", 
    "APIConfig", "LoggingConfig", "DEFAULT_CONFIG",
    
    # Configuration schemas
    "QuantumRerankConfigSchema", "ValidationResult",
    "QuantumConfigSchema", "MLConfigSchema", "PerformanceConfigSchema",
    "APIConfigSchema", "MonitoringConfigSchema",
    "LogLevel", "Environment", "SimulatorMethod",
    "validate_config_file",
    
    # Environment configurations
    "get_development_config", "get_testing_config",
    "get_staging_config", "get_production_config", 
    "get_config_for_environment", "get_environment_defaults",
    "validate_environment_configs",
    
    # Configuration management
    "ConfigManager", "ConfigChangeEvent", "ConfigHistory",
    "get_config_manager", "get_current_config", "reload_config",
    
    # Component integration
    "Configurable", "ConfigurationIntegrator",
    "QuantumEngineConfig", "MLModelConfig", "PerfConfig",
    "APIConfigHelper", "get_configuration_integrator",
    "register_configurable_component",
    
    # Configuration utilities
    "ConfigDiff", "ConfigGenerator", "ConfigComparator",
    "ConfigMigrator", "ConfigBackup", "create_config_template"
]
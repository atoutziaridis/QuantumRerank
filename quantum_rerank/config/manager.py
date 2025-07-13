"""
Configuration management system for QuantumRerank.

This module provides dynamic configuration loading, validation, hot-reload
capabilities, and configuration change management with audit logging.
"""

import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
import yaml
import json
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .schemas import QuantumRerankConfigSchema, ValidationResult, validate_config_file
from .environments import get_config_for_environment, get_development_config
from ..utils import get_logger


@dataclass
class ConfigChangeEvent:
    """Configuration change event for audit logging."""
    timestamp: datetime = field(default_factory=datetime.now)
    config_path: str = ""
    change_type: str = ""  # "reload", "update", "rollback"
    old_config_hash: Optional[str] = None
    new_config_hash: Optional[str] = None
    validation_result: Optional[ValidationResult] = None
    user: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class ConfigHistory:
    """Configuration history for rollback functionality."""
    configs: List[QuantumRerankConfigSchema] = field(default_factory=list)
    events: List[ConfigChangeEvent] = field(default_factory=list)
    max_history: int = 10
    
    def add_config(self, config: QuantumRerankConfigSchema, event: ConfigChangeEvent) -> None:
        """Add configuration to history."""
        self.configs.append(config)
        self.events.append(event)
        
        # Trim history if needed
        if len(self.configs) > self.max_history:
            self.configs = self.configs[-self.max_history:]
            self.events = self.events[-self.max_history:]
    
    def get_previous_config(self, steps_back: int = 1) -> Optional[QuantumRerankConfigSchema]:
        """Get previous configuration for rollback."""
        if len(self.configs) >= steps_back + 1:
            return self.configs[-(steps_back + 1)]
        return None


class ConfigFileWatcher(FileSystemEventHandler):
    """File system watcher for configuration hot-reload."""
    
    def __init__(self, config_manager: 'ConfigManager'):
        self.config_manager = config_manager
        self.logger = get_logger(__name__)
    
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory and event.src_path == str(self.config_manager.config_file_path):
            self.logger.info(f"Configuration file modified: {event.src_path}")
            try:
                time.sleep(0.1)  # Brief delay to ensure file write is complete
                self.config_manager.reload_config()
            except Exception as e:
                self.logger.error(f"Failed to reload configuration: {str(e)}")


class ConfigManager:
    """
    Comprehensive configuration management system.
    
    Features:
    - Configuration loading and validation
    - Hot-reload capabilities
    - Configuration versioning and rollback
    - Change impact analysis
    - Audit logging
    """
    
    def __init__(
        self,
        config_file_path: Optional[Union[str, Path]] = None,
        environment: Optional[str] = None,
        enable_hot_reload: bool = True,
        validation_strict: bool = True
    ):
        self.logger = get_logger(__name__)
        self.config_file_path = Path(config_file_path) if config_file_path else None
        self.environment = environment or os.getenv("QUANTUM_RERANK_ENV", "development")
        self.enable_hot_reload = enable_hot_reload
        self.validation_strict = validation_strict
        
        # Current configuration
        self._config: Optional[QuantumRerankConfigSchema] = None
        self._config_lock = threading.RLock()
        
        # Configuration history and change tracking
        self.history = ConfigHistory()
        
        # Hot-reload setup
        self.file_observer: Optional[Observer] = None
        self.change_callbacks: List[Callable[[QuantumRerankConfigSchema], None]] = []
        
        # Load initial configuration
        self._load_initial_config()
        
        # Setup file watching if enabled
        if self.enable_hot_reload and self.config_file_path:
            self._setup_file_watcher()
    
    def _load_initial_config(self) -> None:
        """Load initial configuration from file or environment defaults."""
        try:
            if self.config_file_path and self.config_file_path.exists():
                self.logger.info(f"Loading configuration from file: {self.config_file_path}")
                self._config = self._load_config_from_file(self.config_file_path)
            else:
                self.logger.info(f"Loading default configuration for environment: {self.environment}")
                self._config = get_config_for_environment(self.environment)
            
            # Validate loaded configuration
            validation_result = self._config.validate()
            if not validation_result.is_valid and self.validation_strict:
                raise ValueError(f"Configuration validation failed: {validation_result.errors}")
            
            # Log validation warnings
            for warning in validation_result.warnings:
                self.logger.warning(f"Configuration warning: {warning}")
            
            # Add to history
            event = ConfigChangeEvent(
                config_path=str(self.config_file_path) if self.config_file_path else "environment",
                change_type="initial_load",
                validation_result=validation_result
            )
            self.history.add_config(self._config, event)
            
        except Exception as e:
            self.logger.error(f"Failed to load initial configuration: {str(e)}")
            # Fallback to development config
            self._config = get_development_config()
            self.logger.warning("Using fallback development configuration")
    
    def _load_config_from_file(self, file_path: Path) -> QuantumRerankConfigSchema:
        """Load configuration from file."""
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            return QuantumRerankConfigSchema.from_yaml_file(file_path)
        elif file_path.suffix.lower() == '.json':
            return QuantumRerankConfigSchema.from_json_file(file_path)
        else:
            raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
    
    def _setup_file_watcher(self) -> None:
        """Setup file system watcher for hot-reload."""
        if not self.config_file_path:
            return
        
        try:
            self.file_observer = Observer()
            event_handler = ConfigFileWatcher(self)
            self.file_observer.schedule(
                event_handler,
                str(self.config_file_path.parent),
                recursive=False
            )
            self.file_observer.start()
            self.logger.info(f"Configuration file watcher started for: {self.config_file_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to setup file watcher: {str(e)}")
            self.enable_hot_reload = False
    
    def get_config(self) -> QuantumRerankConfigSchema:
        """Get current configuration (thread-safe)."""
        with self._config_lock:
            if self._config is None:
                raise RuntimeError("Configuration not loaded")
            return self._config
    
    def reload_config(self) -> ValidationResult:
        """Reload configuration from file."""
        if not self.config_file_path or not self.config_file_path.exists():
            self.logger.warning("No configuration file to reload")
            return ValidationResult(is_valid=False, errors=["No configuration file available"])
        
        try:
            # Validate before loading
            validation_result = validate_config_file(self.config_file_path)
            if not validation_result.is_valid and self.validation_strict:
                self.logger.error(f"Configuration validation failed: {validation_result.errors}")
                return validation_result
            
            # Load new configuration
            new_config = self._load_config_from_file(self.config_file_path)
            
            with self._config_lock:
                old_config = self._config
                self._config = new_config
            
            # Create change event
            event = ConfigChangeEvent(
                config_path=str(self.config_file_path),
                change_type="reload",
                validation_result=validation_result
            )
            self.history.add_config(new_config, event)
            
            # Notify change callbacks
            self._notify_config_change(new_config)
            
            self.logger.info("Configuration reloaded successfully")
            return validation_result
        
        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {str(e)}")
            return ValidationResult(is_valid=False, errors=[str(e)])
    
    def update_config(
        self,
        config_updates: Dict[str, Any],
        user: Optional[str] = None,
        reason: Optional[str] = None
    ) -> ValidationResult:
        """Update configuration with partial changes."""
        try:
            with self._config_lock:
                # Create updated configuration
                current_dict = self._config.to_dict()
                updated_dict = self._deep_update(current_dict, config_updates)
                new_config = QuantumRerankConfigSchema.from_dict(updated_dict)
                
                # Validate updated configuration
                validation_result = new_config.validate()
                if not validation_result.is_valid and self.validation_strict:
                    return validation_result
                
                # Apply update
                old_config = self._config
                self._config = new_config
            
            # Create change event
            event = ConfigChangeEvent(
                config_path=str(self.config_file_path) if self.config_file_path else "runtime",
                change_type="update",
                validation_result=validation_result,
                user=user,
                reason=reason
            )
            self.history.add_config(new_config, event)
            
            # Notify change callbacks
            self._notify_config_change(new_config)
            
            self.logger.info(f"Configuration updated by {user or 'system'}: {reason or 'no reason provided'}")
            return validation_result
        
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {str(e)}")
            return ValidationResult(is_valid=False, errors=[str(e)])
    
    def rollback_config(self, steps_back: int = 1) -> bool:
        """Rollback configuration to previous version."""
        try:
            previous_config = self.history.get_previous_config(steps_back)
            if not previous_config:
                self.logger.warning(f"No configuration available {steps_back} steps back")
                return False
            
            with self._config_lock:
                self._config = previous_config
            
            # Create rollback event
            event = ConfigChangeEvent(
                config_path="rollback",
                change_type="rollback",
                reason=f"Rolled back {steps_back} steps"
            )
            self.history.add_config(previous_config, event)
            
            # Notify change callbacks
            self._notify_config_change(previous_config)
            
            self.logger.info(f"Configuration rolled back {steps_back} steps")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to rollback configuration: {str(e)}")
            return False
    
    def register_change_callback(self, callback: Callable[[QuantumRerankConfigSchema], None]) -> None:
        """Register callback for configuration changes."""
        self.change_callbacks.append(callback)
    
    def _notify_config_change(self, new_config: QuantumRerankConfigSchema) -> None:
        """Notify all registered callbacks of configuration change."""
        for callback in self.change_callbacks:
            try:
                callback(new_config)
            except Exception as e:
                self.logger.error(f"Configuration change callback failed: {str(e)}")
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Deep update dictionary with nested values."""
        result = base_dict.copy()
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._deep_update(result[key], value)
            else:
                result[key] = value
        return result
    
    def save_config_to_file(self, file_path: Optional[Union[str, Path]] = None) -> bool:
        """Save current configuration to file."""
        try:
            target_path = Path(file_path) if file_path else self.config_file_path
            if not target_path:
                raise ValueError("No file path specified for saving configuration")
            
            with self._config_lock:
                config = self._config
            
            # Ensure directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save based on file extension
            if target_path.suffix.lower() in ['.yaml', '.yml']:
                config.to_yaml_file(target_path)
            elif target_path.suffix.lower() == '.json':
                config.to_json_file(target_path)
            else:
                raise ValueError(f"Unsupported file format: {target_path.suffix}")
            
            self.logger.info(f"Configuration saved to: {target_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {str(e)}")
            return False
    
    def get_config_history(self) -> List[ConfigChangeEvent]:
        """Get configuration change history."""
        return self.history.events.copy()
    
    def get_config_diff(self, steps_back: int = 1) -> Optional[Dict[str, Any]]:
        """Get configuration differences from previous version."""
        try:
            current_config = self.get_config()
            previous_config = self.history.get_previous_config(steps_back)
            
            if not previous_config:
                return None
            
            current_dict = current_config.to_dict()
            previous_dict = previous_config.to_dict()
            
            return self._compute_dict_diff(previous_dict, current_dict)
        
        except Exception as e:
            self.logger.error(f"Failed to compute configuration diff: {str(e)}")
            return None
    
    def _compute_dict_diff(self, old_dict: Dict[str, Any], new_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Compute differences between two dictionaries."""
        diff = {}
        
        # Find added and modified keys
        for key, new_value in new_dict.items():
            if key not in old_dict:
                diff[f"+{key}"] = new_value
            elif old_dict[key] != new_value:
                if isinstance(new_value, dict) and isinstance(old_dict[key], dict):
                    nested_diff = self._compute_dict_diff(old_dict[key], new_value)
                    if nested_diff:
                        diff[key] = nested_diff
                else:
                    diff[f"~{key}"] = {"old": old_dict[key], "new": new_value}
        
        # Find removed keys
        for key in old_dict:
            if key not in new_dict:
                diff[f"-{key}"] = old_dict[key]
        
        return diff
    
    def stop(self) -> None:
        """Stop configuration manager and cleanup resources."""
        if self.file_observer:
            self.file_observer.stop()
            self.file_observer.join()
            self.logger.info("Configuration file watcher stopped")


# Global configuration manager instance
_global_config_manager: Optional[ConfigManager] = None


def get_config_manager(
    config_file_path: Optional[Union[str, Path]] = None,
    environment: Optional[str] = None,
    **kwargs
) -> ConfigManager:
    """Get or create global configuration manager instance."""
    global _global_config_manager
    
    if _global_config_manager is None:
        _global_config_manager = ConfigManager(
            config_file_path=config_file_path,
            environment=environment,
            **kwargs
        )
    
    return _global_config_manager


def get_current_config() -> QuantumRerankConfigSchema:
    """Get current configuration from global manager."""
    manager = get_config_manager()
    return manager.get_config()


def reload_config() -> ValidationResult:
    """Reload configuration from global manager."""
    manager = get_config_manager()
    return manager.reload_config()


__all__ = [
    "ConfigChangeEvent",
    "ConfigHistory", 
    "ConfigFileWatcher",
    "ConfigManager",
    "get_config_manager",
    "get_current_config",
    "reload_config"
]
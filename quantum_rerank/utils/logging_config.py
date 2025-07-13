"""
Centralized logging configuration for QuantumRerank.

This module provides structured logging with JSON formatting, multiple output
destinations, and component-specific loggers for comprehensive debugging and monitoring.

Implements PRD Section 6.1: Technical Risks and Mitigation through proper logging.
"""

import logging
import logging.config
import json
import sys
import os
from pathlib import Path
from typing import Dict, Optional, Any, Union
from datetime import datetime
import threading
from dataclasses import dataclass, field


@dataclass
class LogConfig:
    """Configuration for logging system."""
    # Log levels
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    
    # Output configuration
    enable_console: bool = True
    enable_file: bool = True
    log_dir: str = "logs"
    log_file: str = "quantum_rerank.log"
    
    # Formatting
    enable_json_format: bool = True
    include_performance_metrics: bool = True
    include_component_info: bool = True
    
    # Rotation
    max_file_size_mb: int = 100
    backup_count: int = 5
    
    # Component loggers
    component_levels: Dict[str, str] = field(default_factory=lambda: {
        "quantum_rerank.core": "DEBUG",
        "quantum_rerank.ml": "DEBUG", 
        "quantum_rerank.retrieval": "INFO",
        "quantum_rerank.benchmarks": "INFO",
        "quantum_rerank.api": "INFO"
    })


class QuantumRerankFormatter(logging.Formatter):
    """
    Custom formatter for QuantumRerank with structured logging support.
    
    Provides JSON formatting with quantum-specific context information.
    """
    
    def __init__(self, include_json: bool = True, include_performance: bool = True):
        self.include_json = include_json
        self.include_performance = include_performance
        
        if include_json:
            super().__init__()
        else:
            super().__init__(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
    
    def format(self, record):
        """Format log record with quantum-specific information."""
        if not self.include_json:
            return super().format(record)
        
        # Base log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "thread_name": record.threadName
        }
        
        # Add component information
        if hasattr(record, 'component'):
            log_entry['component'] = record.component
        
        # Add operation context
        if hasattr(record, 'operation'):
            log_entry['operation'] = record.operation
        
        # Add performance metrics if enabled
        if self.include_performance and hasattr(record, 'performance'):
            log_entry['performance'] = record.performance
        
        # Add quantum-specific context
        if hasattr(record, 'quantum_context'):
            log_entry['quantum_context'] = record.quantum_context
        
        # Add error context
        if record.exc_info:
            log_entry['exception'] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName', 'processName',
                          'process', 'exc_info', 'exc_text', 'stack_info', 'getMessage']:
                if not key.startswith('_') and key not in log_entry:
                    log_entry[key] = value
        
        return json.dumps(log_entry, default=str, separators=(',', ':'))


class PerformanceLogFilter(logging.Filter):
    """Filter for performance-sensitive logging."""
    
    def __init__(self, max_message_length: int = 1000):
        super().__init__()
        self.max_message_length = max_message_length
    
    def filter(self, record):
        """Filter log records for performance."""
        # Truncate long messages
        if len(record.getMessage()) > self.max_message_length:
            record.msg = record.getMessage()[:self.max_message_length] + "... [truncated]"
            record.args = ()
        
        return True


class ComponentLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds component-specific context.
    
    Automatically adds component name and operation context to log records.
    """
    
    def __init__(self, logger, component_name: str):
        self.component_name = component_name
        super().__init__(logger, {"component": component_name})
    
    def process(self, msg, kwargs):
        """Add component context to log records."""
        extra = kwargs.get('extra', {})
        extra['component'] = self.component_name
        kwargs['extra'] = extra
        return msg, kwargs
    
    def log_operation(self, level, operation_name: str, message: str, 
                     performance_data: Optional[Dict] = None, **kwargs):
        """Log with operation context."""
        extra = kwargs.get('extra', {})
        extra.update({
            'operation': operation_name,
            'component': self.component_name
        })
        
        if performance_data:
            extra['performance'] = performance_data
        
        kwargs['extra'] = extra
        self.log(level, message, **kwargs)
    
    def log_quantum_operation(self, level, operation_name: str, message: str,
                            circuit_info: Optional[Dict] = None,
                            performance_data: Optional[Dict] = None, **kwargs):
        """Log quantum-specific operations."""
        extra = kwargs.get('extra', {})
        extra.update({
            'operation': operation_name,
            'component': self.component_name
        })
        
        if circuit_info:
            extra['quantum_context'] = circuit_info
        
        if performance_data:
            extra['performance'] = performance_data
        
        kwargs['extra'] = extra
        self.log(level, message, **kwargs)


class LoggingConfigManager:
    """
    Manager for centralized logging configuration.
    
    Handles setup, configuration updates, and component logger creation.
    """
    
    def __init__(self, config: Optional[LogConfig] = None):
        """
        Initialize logging configuration manager.
        
        Args:
            config: Logging configuration (uses defaults if None)
        """
        self.config = config or LogConfig()
        self._loggers: Dict[str, ComponentLoggerAdapter] = {}
        self._configured = False
        self._lock = threading.Lock()
    
    def setup_logging(self):
        """Set up centralized logging configuration."""
        with self._lock:
            if self._configured:
                return
            
            # Create log directory
            if self.config.enable_file:
                log_dir = Path(self.config.log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
            
            # Configure root logger
            self._configure_root_logger()
            
            # Configure component loggers
            self._configure_component_loggers()
            
            self._configured = True
            
            # Log successful configuration
            logger = self.get_logger("logging_config")
            logger.info("Logging system configured successfully", extra={
                "config": {
                    "console_level": self.config.console_level,
                    "file_level": self.config.file_level,
                    "json_format": self.config.enable_json_format,
                    "performance_metrics": self.config.include_performance_metrics
                }
            })
    
    def _configure_root_logger(self):
        """Configure the root logger with handlers."""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if self.config.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, self.config.console_level.upper()))
            
            if self.config.enable_json_format:
                console_formatter = QuantumRerankFormatter(
                    include_json=True,
                    include_performance=self.config.include_performance_metrics
                )
            else:
                console_formatter = QuantumRerankFormatter(include_json=False)
            
            console_handler.setFormatter(console_formatter)
            console_handler.addFilter(PerformanceLogFilter())
            root_logger.addHandler(console_handler)
        
        # File handler
        if self.config.enable_file:
            from logging.handlers import RotatingFileHandler
            
            log_file_path = Path(self.config.log_dir) / self.config.log_file
            file_handler = RotatingFileHandler(
                log_file_path,
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            file_handler.setLevel(getattr(logging, self.config.file_level.upper()))
            
            file_formatter = QuantumRerankFormatter(
                include_json=True,
                include_performance=self.config.include_performance_metrics
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
    
    def _configure_component_loggers(self):
        """Configure component-specific loggers."""
        for component, level in self.config.component_levels.items():
            logger = logging.getLogger(component)
            logger.setLevel(getattr(logging, level.upper()))
    
    def get_logger(self, component_name: str) -> ComponentLoggerAdapter:
        """
        Get a component-specific logger.
        
        Args:
            component_name: Name of the component
            
        Returns:
            ComponentLoggerAdapter for the component
        """
        if not self._configured:
            self.setup_logging()
        
        if component_name not in self._loggers:
            base_logger = logging.getLogger(f"quantum_rerank.{component_name}")
            self._loggers[component_name] = ComponentLoggerAdapter(base_logger, component_name)
        
        return self._loggers[component_name]
    
    def update_log_level(self, component_name: str, level: str):
        """
        Update log level for a specific component.
        
        Args:
            component_name: Component to update
            level: New log level
        """
        logger_name = f"quantum_rerank.{component_name}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level.upper()))
        
        self.config.component_levels[logger_name] = level
    
    def get_log_stats(self) -> Dict[str, Any]:
        """
        Get logging system statistics.
        
        Returns:
            Dictionary with logging statistics
        """
        stats = {
            "configured": self._configured,
            "active_loggers": len(self._loggers),
            "log_levels": self.config.component_levels.copy()
        }
        
        # Add file information if file logging is enabled
        if self.config.enable_file:
            log_file_path = Path(self.config.log_dir) / self.config.log_file
            if log_file_path.exists():
                stats["log_file_size_mb"] = log_file_path.stat().st_size / (1024 * 1024)
        
        return stats


# Global logging manager instance
_logging_manager: Optional[LoggingConfigManager] = None


def setup_logging(config: Optional[LogConfig] = None):
    """
    Set up global logging configuration.
    
    Args:
        config: Logging configuration (uses defaults if None)
    """
    global _logging_manager
    _logging_manager = LoggingConfigManager(config)
    _logging_manager.setup_logging()


def get_logger(component_name: str) -> ComponentLoggerAdapter:
    """
    Get a logger for a specific component.
    
    Args:
        component_name: Name of the component
        
    Returns:
        ComponentLoggerAdapter for the component
    """
    global _logging_manager
    if _logging_manager is None:
        setup_logging()
    
    return _logging_manager.get_logger(component_name)


def configure_logging_from_dict(config_dict: Dict[str, Any]):
    """
    Configure logging from dictionary configuration.
    
    Args:
        config_dict: Dictionary with logging configuration
    """
    config = LogConfig(**config_dict)
    setup_logging(config)


def get_logging_stats() -> Dict[str, Any]:
    """
    Get logging system statistics.
    
    Returns:
        Dictionary with logging statistics
    """
    global _logging_manager
    if _logging_manager is None:
        return {"error": "Logging not configured"}
    
    return _logging_manager.get_log_stats()
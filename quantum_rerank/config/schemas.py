"""
Configuration schemas and validation for QuantumRerank system.

This module provides comprehensive validation for all configuration categories
as specified in the PRD, ensuring system constraints and performance targets.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import yaml
import json
from pathlib import Path


class LogLevel(Enum):
    """Supported logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Environment(Enum):
    """Supported deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class SimulatorMethod(Enum):
    """Supported quantum simulator methods."""
    STATEVECTOR = "statevector"
    QASM = "qasm"
    STABILIZER = "stabilizer"
    EXTENDED_STABILIZER = "extended_stabilizer"


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, message: str) -> None:
        """Add validation error."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add validation warning."""
        self.warnings.append(message)


@dataclass
class QuantumConfigSchema:
    """Quantum configuration schema with PRD constraint validation."""
    n_qubits: int = 4
    max_circuit_depth: int = 15
    shots: int = 1024
    simulator_method: str = "statevector"
    quantum_backends: List[str] = field(default_factory=lambda: ["aer_simulator", "qasm_simulator"])
    enable_optimization: bool = True
    noise_model: Optional[str] = None
    
    def validate(self) -> ValidationResult:
        """Validate quantum configuration against PRD constraints."""
        result = ValidationResult(is_valid=True)
        
        # PRD: 2-4 qubits maximum for classical simulation
        if not (2 <= self.n_qubits <= 4):
            result.add_error(f"n_qubits must be between 2-4 (PRD constraint), got {self.n_qubits}")
        
        # PRD: ≤15 gate depth for performance
        if self.max_circuit_depth > 15:
            result.add_error(f"max_circuit_depth must be ≤15 (PRD constraint), got {self.max_circuit_depth}")
        
        if self.max_circuit_depth < 1:
            result.add_error(f"max_circuit_depth must be positive, got {self.max_circuit_depth}")
        
        # Validate shots count
        if self.shots < 1:
            result.add_error(f"shots must be positive, got {self.shots}")
        
        # Validate simulator method
        try:
            SimulatorMethod(self.simulator_method)
        except ValueError:
            valid_methods = [m.value for m in SimulatorMethod]
            result.add_error(f"Invalid simulator_method '{self.simulator_method}'. Valid options: {valid_methods}")
        
        # Validate quantum backends
        if not self.quantum_backends:
            result.add_warning("No quantum backends specified, using defaults")
        
        return result


@dataclass
class MLConfigSchema:
    """ML configuration schema for embedding and training."""
    embedding_model: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    embedding_dim: int = 768
    batch_size: int = 50
    max_sequence_length: int = 512
    fallback_models: List[str] = field(default_factory=lambda: [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2"
    ])
    use_quantum_compression: bool = True
    compressed_dim: Optional[int] = 256
    parameter_prediction: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_dims": [512, 256],
        "dropout_rate": 0.1,
        "learning_rate": 0.001,
        "activation": "relu"
    })
    
    def validate(self) -> ValidationResult:
        """Validate ML configuration."""
        result = ValidationResult(is_valid=True)
        
        # PRD: 50-100 document batch size
        if not (1 <= self.batch_size <= 100):
            result.add_warning(f"batch_size {self.batch_size} outside recommended range 50-100")
        
        # Validate embedding dimensions
        if self.embedding_dim <= 0:
            result.add_error(f"embedding_dim must be positive, got {self.embedding_dim}")
        
        # Validate sequence length
        if self.max_sequence_length <= 0:
            result.add_error(f"max_sequence_length must be positive, got {self.max_sequence_length}")
        
        # Validate compressed dimensions
        if self.use_quantum_compression and self.compressed_dim:
            if self.compressed_dim >= self.embedding_dim:
                result.add_warning(f"compressed_dim {self.compressed_dim} >= embedding_dim {self.embedding_dim}")
        
        # Validate parameter prediction config
        if isinstance(self.parameter_prediction, dict):
            if "learning_rate" in self.parameter_prediction:
                lr = self.parameter_prediction["learning_rate"]
                if not (1e-6 <= lr <= 1.0):
                    result.add_warning(f"learning_rate {lr} outside typical range [1e-6, 1.0]")
        
        return result


@dataclass
class PerformanceConfigSchema:
    """Performance configuration schema with PRD targets."""
    similarity_timeout_ms: int = 100  # PRD target
    batch_timeout_ms: int = 500      # PRD target
    max_memory_gb: float = 2.0       # PRD target
    cache_size: int = 1000
    enable_caching: bool = True
    max_concurrent_requests: int = 10
    cpu_usage_threshold: float = 0.8
    memory_usage_threshold: float = 0.9
    quantum_computation_timeout_s: int = 30
    model_inference_timeout_s: int = 10
    
    def validate(self) -> ValidationResult:
        """Validate performance configuration against PRD targets."""
        result = ValidationResult(is_valid=True)
        
        # PRD: <100ms similarity computation
        if self.similarity_timeout_ms > 100:
            result.add_warning(f"similarity_timeout_ms {self.similarity_timeout_ms} exceeds PRD target of 100ms")
        
        # PRD: <500ms batch processing
        if self.batch_timeout_ms > 500:
            result.add_warning(f"batch_timeout_ms {self.batch_timeout_ms} exceeds PRD target of 500ms")
        
        # PRD: <2GB memory usage
        if self.max_memory_gb > 2.0:
            result.add_warning(f"max_memory_gb {self.max_memory_gb} exceeds PRD target of 2.0GB")
        
        # Validate threshold values
        if not (0.0 <= self.cpu_usage_threshold <= 1.0):
            result.add_error(f"cpu_usage_threshold must be between 0-1, got {self.cpu_usage_threshold}")
        
        if not (0.0 <= self.memory_usage_threshold <= 1.0):
            result.add_error(f"memory_usage_threshold must be between 0-1, got {self.memory_usage_threshold}")
        
        # Validate timeout values
        if self.quantum_computation_timeout_s <= 0:
            result.add_error(f"quantum_computation_timeout_s must be positive, got {self.quantum_computation_timeout_s}")
        
        if self.model_inference_timeout_s <= 0:
            result.add_error(f"model_inference_timeout_s must be positive, got {self.model_inference_timeout_s}")
        
        return result


@dataclass
class APIConfigSchema:
    """API configuration schema for service deployment."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    rate_limit: str = "100/minute"
    enable_auth: bool = False
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    request_timeout_s: int = 30
    max_request_size_mb: int = 10
    
    def validate(self) -> ValidationResult:
        """Validate API configuration."""
        result = ValidationResult(is_valid=True)
        
        # Validate port range
        if not (1 <= self.port <= 65535):
            result.add_error(f"port must be between 1-65535, got {self.port}")
        
        # Validate workers count
        if self.workers <= 0:
            result.add_error(f"workers must be positive, got {self.workers}")
        
        # Validate timeout
        if self.request_timeout_s <= 0:
            result.add_error(f"request_timeout_s must be positive, got {self.request_timeout_s}")
        
        # Validate request size
        if self.max_request_size_mb <= 0:
            result.add_error(f"max_request_size_mb must be positive, got {self.max_request_size_mb}")
        
        return result


@dataclass
class MonitoringConfigSchema:
    """Monitoring and logging configuration schema."""
    log_level: str = "INFO"
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_interval_s: int = 30
    enable_tracing: bool = False
    log_format: str = "json"
    log_file: Optional[str] = None
    max_log_file_size_mb: int = 100
    log_retention_days: int = 7
    
    def validate(self) -> ValidationResult:
        """Validate monitoring configuration."""
        result = ValidationResult(is_valid=True)
        
        # Validate log level
        try:
            LogLevel(self.log_level.upper())
        except ValueError:
            valid_levels = [level.value for level in LogLevel]
            result.add_error(f"Invalid log_level '{self.log_level}'. Valid options: {valid_levels}")
        
        # Validate metrics port
        if not (1 <= self.metrics_port <= 65535):
            result.add_error(f"metrics_port must be between 1-65535, got {self.metrics_port}")
        
        # Validate health check interval
        if self.health_check_interval_s <= 0:
            result.add_error(f"health_check_interval_s must be positive, got {self.health_check_interval_s}")
        
        # Validate log file settings
        if self.max_log_file_size_mb <= 0:
            result.add_error(f"max_log_file_size_mb must be positive, got {self.max_log_file_size_mb}")
        
        if self.log_retention_days <= 0:
            result.add_error(f"log_retention_days must be positive, got {self.log_retention_days}")
        
        return result


@dataclass
class QuantumRerankConfigSchema:
    """Complete QuantumRerank configuration schema."""
    quantum: QuantumConfigSchema = field(default_factory=QuantumConfigSchema)
    ml: MLConfigSchema = field(default_factory=MLConfigSchema)
    performance: PerformanceConfigSchema = field(default_factory=PerformanceConfigSchema)
    api: APIConfigSchema = field(default_factory=APIConfigSchema)
    monitoring: MonitoringConfigSchema = field(default_factory=MonitoringConfigSchema)
    environment: str = "development"
    version: str = "1.0.0"
    
    def validate(self) -> ValidationResult:
        """Validate complete configuration."""
        result = ValidationResult(is_valid=True)
        
        # Validate environment
        try:
            Environment(self.environment)
        except ValueError:
            valid_envs = [env.value for env in Environment]
            result.add_error(f"Invalid environment '{self.environment}'. Valid options: {valid_envs}")
        
        # Validate all subsections
        for attr_name in ["quantum", "ml", "performance", "api", "monitoring"]:
            subsection = getattr(self, attr_name)
            if hasattr(subsection, 'validate'):
                sub_result = subsection.validate()
                result.errors.extend([f"{attr_name}.{error}" for error in sub_result.errors])
                result.warnings.extend([f"{attr_name}.{warning}" for warning in sub_result.warnings])
                if not sub_result.is_valid:
                    result.is_valid = False
        
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QuantumRerankConfigSchema':
        """Create configuration from dictionary."""
        quantum_config = QuantumConfigSchema(**config_dict.get("quantum", {}))
        ml_config = MLConfigSchema(**config_dict.get("ml", {}))
        performance_config = PerformanceConfigSchema(**config_dict.get("performance", {}))
        api_config = APIConfigSchema(**config_dict.get("api", {}))
        monitoring_config = MonitoringConfigSchema(**config_dict.get("monitoring", {}))
        
        return cls(
            quantum=quantum_config,
            ml=ml_config,
            performance=performance_config,
            api=api_config,
            monitoring=monitoring_config,
            environment=config_dict.get("environment", "development"),
            version=config_dict.get("version", "1.0.0")
        )
    
    @classmethod
    def from_yaml_file(cls, file_path: Union[str, Path]) -> 'QuantumRerankConfigSchema':
        """Load configuration from YAML file."""
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json_file(cls, file_path: Union[str, Path]) -> 'QuantumRerankConfigSchema':
        """Load configuration from JSON file."""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "quantum": self.quantum.__dict__,
            "ml": self.ml.__dict__,
            "performance": self.performance.__dict__,
            "api": self.api.__dict__,
            "monitoring": self.monitoring.__dict__,
            "environment": self.environment,
            "version": self.version
        }
    
    def to_yaml_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(file_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def to_json_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def validate_config_file(file_path: Union[str, Path]) -> ValidationResult:
    """Validate configuration file without loading into schema."""
    try:
        file_path = Path(file_path)
        if file_path.suffix.lower() == '.yaml' or file_path.suffix.lower() == '.yml':
            config = QuantumRerankConfigSchema.from_yaml_file(file_path)
        elif file_path.suffix.lower() == '.json':
            config = QuantumRerankConfigSchema.from_json_file(file_path)
        else:
            result = ValidationResult(is_valid=False)
            result.add_error(f"Unsupported file format: {file_path.suffix}")
            return result
        
        return config.validate()
    
    except Exception as e:
        result = ValidationResult(is_valid=False)
        result.add_error(f"Failed to load config file: {str(e)}")
        return result


__all__ = [
    "LogLevel",
    "Environment", 
    "SimulatorMethod",
    "ValidationResult",
    "QuantumConfigSchema",
    "MLConfigSchema",
    "PerformanceConfigSchema",
    "APIConfigSchema",
    "MonitoringConfigSchema",
    "QuantumRerankConfigSchema",
    "validate_config_file"
]
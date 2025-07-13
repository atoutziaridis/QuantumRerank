"""
Environment-specific configurations for QuantumRerank system.

This module provides pre-configured settings for different deployment environments
optimized for development, testing, staging, and production use cases.
"""

from typing import Dict, Any
from .schemas import QuantumRerankConfigSchema, Environment


def get_development_config() -> QuantumRerankConfigSchema:
    """Development environment configuration with debugging enabled."""
    return QuantumRerankConfigSchema.from_dict({
        "environment": "development",
        "version": "1.0.0-dev",
        "quantum": {
            "n_qubits": 2,  # Minimal for fast iteration
            "max_circuit_depth": 5,  # Reduced for speed
            "shots": 100,  # Lower shots for faster execution
            "simulator_method": "statevector",
            "enable_optimization": False,  # Disable for predictable behavior
            "quantum_backends": ["aer_simulator"]
        },
        "ml": {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",  # Faster model
            "embedding_dim": 384,  # Smaller for development
            "batch_size": 10,  # Small batches for testing
            "max_sequence_length": 256,  # Shorter sequences
            "use_quantum_compression": False,  # Disable for simplicity
            "parameter_prediction": {
                "hidden_dims": [128, 64],  # Smaller network
                "dropout_rate": 0.0,  # No dropout for debugging
                "learning_rate": 0.01,  # Higher learning rate
                "activation": "relu"
            }
        },
        "performance": {
            "similarity_timeout_ms": 1000,  # Relaxed timeouts
            "batch_timeout_ms": 5000,
            "max_memory_gb": 4.0,  # More memory for debugging
            "cache_size": 100,
            "enable_caching": True,
            "max_concurrent_requests": 2,  # Limited concurrency
            "quantum_computation_timeout_s": 60,
            "model_inference_timeout_s": 30
        },
        "api": {
            "host": "127.0.0.1",  # Local only
            "port": 8000,
            "workers": 1,  # Single worker for debugging
            "rate_limit": "1000/minute",  # Generous rate limiting
            "enable_auth": False,
            "cors_enabled": True,
            "request_timeout_s": 60,
            "max_request_size_mb": 50
        },
        "monitoring": {
            "log_level": "DEBUG",  # Verbose logging
            "enable_metrics": True,
            "metrics_port": 9090,
            "health_check_interval_s": 60,
            "enable_tracing": True,  # Enable for debugging
            "log_format": "json",
            "log_file": "logs/quantum_rerank_dev.log",
            "max_log_file_size_mb": 10,
            "log_retention_days": 3
        }
    })


def get_testing_config() -> QuantumRerankConfigSchema:
    """Testing environment configuration for reproducible test runs."""
    return QuantumRerankConfigSchema.from_dict({
        "environment": "testing",
        "version": "1.0.0-test",
        "quantum": {
            "n_qubits": 3,  # Medium complexity for thorough testing
            "max_circuit_depth": 10,
            "shots": 512,  # Deterministic results
            "simulator_method": "statevector",
            "enable_optimization": True,
            "quantum_backends": ["aer_simulator", "qasm_simulator"]
        },
        "ml": {
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
            "embedding_dim": 768,
            "batch_size": 25,  # Medium batch size
            "max_sequence_length": 512,
            "use_quantum_compression": True,
            "compressed_dim": 256,
            "parameter_prediction": {
                "hidden_dims": [256, 128],
                "dropout_rate": 0.1,
                "learning_rate": 0.001,
                "activation": "relu"
            }
        },
        "performance": {
            "similarity_timeout_ms": 200,  # Slightly relaxed for testing
            "batch_timeout_ms": 1000,
            "max_memory_gb": 3.0,
            "cache_size": 500,
            "enable_caching": True,
            "max_concurrent_requests": 5,
            "quantum_computation_timeout_s": 45,
            "model_inference_timeout_s": 20
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8001,  # Different port to avoid conflicts
            "workers": 2,
            "rate_limit": "500/minute",
            "enable_auth": False,  # Simplified for testing
            "cors_enabled": True,
            "request_timeout_s": 45,
            "max_request_size_mb": 25
        },
        "monitoring": {
            "log_level": "INFO",
            "enable_metrics": True,
            "metrics_port": 9091,
            "health_check_interval_s": 30,
            "enable_tracing": False,  # Reduce overhead
            "log_format": "json",
            "log_file": "logs/quantum_rerank_test.log",
            "max_log_file_size_mb": 25,
            "log_retention_days": 7
        }
    })


def get_staging_config() -> QuantumRerankConfigSchema:
    """Staging environment configuration similar to production with safety constraints."""
    return QuantumRerankConfigSchema.from_dict({
        "environment": "staging",
        "version": "1.0.0-staging",
        "quantum": {
            "n_qubits": 4,  # Full PRD specification
            "max_circuit_depth": 15,
            "shots": 1024,
            "simulator_method": "statevector",
            "enable_optimization": True,
            "quantum_backends": ["aer_simulator", "qasm_simulator"]
        },
        "ml": {
            "embedding_model": "sentence-transformers/multi-qa-mpnet-base-dot-v1",  # PRD recommended
            "embedding_dim": 768,
            "batch_size": 50,  # PRD specification
            "max_sequence_length": 512,
            "use_quantum_compression": True,
            "compressed_dim": 256,
            "parameter_prediction": {
                "hidden_dims": [512, 256],
                "dropout_rate": 0.1,
                "learning_rate": 0.001,
                "activation": "relu"
            }
        },
        "performance": {
            "similarity_timeout_ms": 120,  # Slightly above PRD for safety
            "batch_timeout_ms": 600,
            "max_memory_gb": 2.5,  # Slightly above PRD for safety
            "cache_size": 800,
            "enable_caching": True,
            "max_concurrent_requests": 8,
            "quantum_computation_timeout_s": 35,
            "model_inference_timeout_s": 15
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 4,
            "rate_limit": "200/minute",
            "enable_auth": True,  # Enable security for staging
            "cors_enabled": True,
            "cors_origins": ["https://staging.example.com"],  # Restricted origins
            "request_timeout_s": 35,
            "max_request_size_mb": 15
        },
        "monitoring": {
            "log_level": "INFO",
            "enable_metrics": True,
            "metrics_port": 9090,
            "health_check_interval_s": 30,
            "enable_tracing": True,
            "log_format": "json",
            "log_file": "/var/log/quantum_rerank/staging.log",
            "max_log_file_size_mb": 100,
            "log_retention_days": 14
        }
    })


def get_production_config() -> QuantumRerankConfigSchema:
    """Production environment configuration optimized for performance and reliability."""
    return QuantumRerankConfigSchema.from_dict({
        "environment": "production",
        "version": "1.0.0",
        "quantum": {
            "n_qubits": 4,  # PRD specification
            "max_circuit_depth": 15,  # PRD specification
            "shots": 1024,
            "simulator_method": "statevector",
            "enable_optimization": True,
            "quantum_backends": ["aer_simulator", "qasm_simulator"]
        },
        "ml": {
            "embedding_model": "sentence-transformers/multi-qa-mpnet-base-dot-v1",  # PRD recommended
            "embedding_dim": 768,
            "batch_size": 75,  # Optimized within PRD range
            "max_sequence_length": 512,
            "use_quantum_compression": True,
            "compressed_dim": 256,
            "parameter_prediction": {
                "hidden_dims": [512, 256],
                "dropout_rate": 0.1,
                "learning_rate": 0.001,
                "activation": "relu"
            }
        },
        "performance": {
            "similarity_timeout_ms": 100,  # PRD target
            "batch_timeout_ms": 500,  # PRD target
            "max_memory_gb": 2.0,  # PRD target
            "cache_size": 1000,
            "enable_caching": True,
            "max_concurrent_requests": 10,
            "cpu_usage_threshold": 0.8,
            "memory_usage_threshold": 0.85,  # Slightly below limit
            "quantum_computation_timeout_s": 30,
            "model_inference_timeout_s": 10
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 8,  # Higher for production load
            "rate_limit": "100/minute",  # PRD specification
            "enable_auth": True,
            "cors_enabled": True,
            "cors_origins": ["https://app.example.com", "https://api.example.com"],
            "request_timeout_s": 30,
            "max_request_size_mb": 10
        },
        "monitoring": {
            "log_level": "WARNING",  # Reduce log volume in production
            "enable_metrics": True,
            "metrics_port": 9090,
            "health_check_interval_s": 15,  # Frequent health checks
            "enable_tracing": False,  # Reduce overhead
            "log_format": "json",
            "log_file": "/var/log/quantum_rerank/production.log",
            "max_log_file_size_mb": 500,
            "log_retention_days": 30
        }
    })


def get_config_for_environment(environment: str) -> QuantumRerankConfigSchema:
    """Get configuration for specified environment."""
    env_configs = {
        Environment.DEVELOPMENT.value: get_development_config,
        Environment.TESTING.value: get_testing_config,
        Environment.STAGING.value: get_staging_config,
        Environment.PRODUCTION.value: get_production_config
    }
    
    if environment not in env_configs:
        raise ValueError(f"Unknown environment: {environment}. Valid options: {list(env_configs.keys())}")
    
    return env_configs[environment]()


def get_environment_defaults() -> Dict[str, Any]:
    """Get default configuration values for all environments."""
    return {
        "development": get_development_config().to_dict(),
        "testing": get_testing_config().to_dict(),
        "staging": get_staging_config().to_dict(),
        "production": get_production_config().to_dict()
    }


def validate_environment_configs() -> Dict[str, bool]:
    """Validate all environment configurations."""
    results = {}
    
    for env_name in [e.value for e in Environment]:
        try:
            config = get_config_for_environment(env_name)
            validation_result = config.validate()
            results[env_name] = validation_result.is_valid
            
            if not validation_result.is_valid:
                print(f"Validation errors for {env_name}:")
                for error in validation_result.errors:
                    print(f"  ERROR: {error}")
                for warning in validation_result.warnings:
                    print(f"  WARNING: {warning}")
        
        except Exception as e:
            results[env_name] = False
            print(f"Failed to validate {env_name}: {str(e)}")
    
    return results


__all__ = [
    "get_development_config",
    "get_testing_config", 
    "get_staging_config",
    "get_production_config",
    "get_config_for_environment",
    "get_environment_defaults",
    "validate_environment_configs"
]
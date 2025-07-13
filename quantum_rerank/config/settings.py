"""
Configuration settings for QuantumRerank system.

Based on PRD specifications and quantum-inspired embedding research.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class QuantumConfig:
    """Quantum computation configuration based on PRD constraints."""
    
    # PRD: 2-4 qubits maximum for classical simulation
    n_qubits: int = 4
    
    # PRD: ≤15 gate depth for performance
    max_circuit_depth: int = 15
    
    # Standard quantum measurement configuration
    shots: int = 1024
    
    # Classical simulation method for quantum circuits
    simulator_method: str = 'statevector'
    
    # Quantum backends to try in order
    quantum_backends: List[str] = None
    
    def __post_init__(self):
        if self.quantum_backends is None:
            self.quantum_backends = ['aer_simulator', 'qasm_simulator']


@dataclass
class ModelConfig:
    """Model configuration optimized for quantum-inspired similarity."""
    
    # Based on documentation: multi-qa-mpnet-base-dot-v1 recommended
    embedding_model: str = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
    
    # 768-dimensional embeddings for optimal quantum-classical balance
    embedding_dim: int = 768
    
    # PRD: 50-100 document batch size
    batch_size: int = 50
    
    # Standard sequence length for efficiency
    max_sequence_length: int = 512
    
    # Alternative models for fallback
    fallback_models: List[str] = None
    
    # Quantum-inspired compression settings
    use_quantum_compression: bool = True
    compressed_dim: Optional[int] = 256
    
    def __post_init__(self):
        if self.fallback_models is None:
            self.fallback_models = [
                'sentence-transformers/all-MiniLM-L6-v2',
                'sentence-transformers/all-mpnet-base-v2'
            ]


@dataclass  
class PerformanceConfig:
    """Performance targets from PRD specifications."""
    
    # PRD: <500ms reranking latency
    max_latency_ms: int = 500
    
    # PRD: <2GB memory for 100 documents
    max_memory_gb: float = 2.0
    
    # PRD: <100ms similarity computation
    similarity_computation_ms: int = 100
    
    # PRD: 10-20% accuracy improvement target
    target_accuracy_improvement: float = 0.15
    
    # Additional performance settings
    max_concurrent_requests: int = 10
    cache_size_mb: int = 256
    
    # Timeout settings
    quantum_computation_timeout_s: int = 30
    model_inference_timeout_s: int = 10
    
    # Resource monitoring thresholds
    cpu_usage_threshold: float = 0.8
    memory_usage_threshold: float = 0.9


@dataclass
class APIConfig:
    """API service configuration."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Rate limiting
    rate_limit_requests_per_minute: int = 100
    
    # CORS settings
    cors_origins: List[str] = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]


@dataclass
class LoggingConfig:
    """Logging configuration."""
    
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
    log_file: Optional[str] = "quantum_rerank.log"
    max_file_size: str = "100 MB"
    backup_count: int = 3


# Default configuration instance
DEFAULT_CONFIG = {
    "quantum": QuantumConfig(),
    "model": ModelConfig(), 
    "performance": PerformanceConfig(),
    "api": APIConfig(),
    "logging": LoggingConfig()
}
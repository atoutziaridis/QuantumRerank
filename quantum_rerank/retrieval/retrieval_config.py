"""
Configuration classes for the retrieval module.

Provides centralized configuration for FAISS indices, retrieval pipelines,
and optimization settings.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Any


class OptimizationLevel(Enum):
    """Optimization level for retrieval performance."""
    ACCURACY = "accuracy"      # Prioritize accuracy over speed
    BALANCED = "balanced"      # Balance between accuracy and speed  
    SPEED = "speed"           # Prioritize speed over accuracy
    MEMORY = "memory"         # Minimize memory usage


@dataclass
class RetrievalConfig:
    """
    Master configuration for the retrieval system.
    
    Combines settings for FAISS indexing, two-stage retrieval,
    and performance optimization.
    """
    # FAISS Configuration
    index_type: str = "Flat"
    embedding_dim: int = 768
    normalize_embeddings: bool = True
    distance_metric: str = "cosine"
    
    # Index-specific parameters
    faiss_params: Dict[str, Any] = field(default_factory=lambda: {
        "nlist": 100,      # For IVF
        "nprobe": 10,      # For IVF search
        "M": 32,           # For HNSW
        "ef_construction": 200,  # For HNSW
        "ef_search": 50,   # For HNSW search
        "nbits": 768       # For LSH
    })
    
    # Two-stage retrieval settings
    initial_retrieval_k: int = 100  # Number of candidates from FAISS
    final_top_k: int = 10          # Number of results after reranking
    
    # Quantum reranking settings
    reranking_method: str = "hybrid"  # classical, quantum, or hybrid
    quantum_n_qubits: int = 4
    quantum_n_layers: int = 2
    enable_caching: bool = True
    cache_size: int = 1000
    
    # Performance optimization
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    use_gpu: bool = False
    batch_size: int = 32
    num_threads: int = 4
    
    # Persistence settings
    index_save_path: Optional[str] = None
    auto_save: bool = False
    save_interval: int = 1000  # Save after every N documents
    
    # Monitoring and logging
    enable_monitoring: bool = True
    log_level: str = "INFO"
    collect_detailed_metrics: bool = False
    
    def get_optimization_presets(self) -> Dict[str, Any]:
        """
        Get preset configurations based on optimization level.
        
        Returns configuration adjustments for the selected optimization level.
        """
        if self.optimization_level == OptimizationLevel.ACCURACY:
            return {
                "index_type": "Flat",
                "initial_retrieval_k": 200,
                "reranking_method": "hybrid",
                "faiss_params": {
                    "nprobe": 20  # More thorough IVF search
                }
            }
        elif self.optimization_level == OptimizationLevel.SPEED:
            return {
                "index_type": "IVF",
                "initial_retrieval_k": 50,
                "reranking_method": "classical",
                "faiss_params": {
                    "nlist": 256,
                    "nprobe": 5
                },
                "batch_size": 64
            }
        elif self.optimization_level == OptimizationLevel.MEMORY:
            return {
                "index_type": "LSH",
                "initial_retrieval_k": 50,
                "enable_caching": False,
                "faiss_params": {
                    "nbits": 256  # Reduced bits for LSH
                }
            }
        else:  # BALANCED
            return {
                "index_type": "IVF",
                "initial_retrieval_k": 100,
                "reranking_method": "hybrid",
                "faiss_params": {
                    "nlist": 100,
                    "nprobe": 10
                }
            }
    
    def apply_optimization_presets(self):
        """Apply optimization presets to current configuration."""
        presets = self.get_optimization_presets()
        
        for key, value in presets.items():
            if key == "faiss_params":
                self.faiss_params.update(value)
            else:
                setattr(self, key, value)
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate dimensions
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        
        # Validate retrieval parameters
        if self.initial_retrieval_k <= 0:
            raise ValueError("initial_retrieval_k must be positive")
        
        if self.final_top_k <= 0:
            raise ValueError("final_top_k must be positive")
        
        if self.final_top_k > self.initial_retrieval_k:
            raise ValueError("final_top_k cannot exceed initial_retrieval_k")
        
        # Validate quantum parameters
        if self.quantum_n_qubits not in [2, 3, 4]:
            raise ValueError("quantum_n_qubits must be 2, 3, or 4")
        
        if self.quantum_n_layers <= 0 or self.quantum_n_layers > 5:
            raise ValueError("quantum_n_layers must be between 1 and 5")
        
        # Validate index type
        valid_index_types = ["Flat", "IVF", "HNSW", "LSH"]
        if self.index_type not in valid_index_types:
            raise ValueError(f"index_type must be one of {valid_index_types}")
        
        # Validate distance metric
        if self.distance_metric not in ["cosine", "l2"]:
            raise ValueError("distance_metric must be 'cosine' or 'l2'")
        
        # Validate reranking method
        if self.reranking_method not in ["classical", "quantum", "hybrid"]:
            raise ValueError("reranking_method must be 'classical', 'quantum', or 'hybrid'")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "index_type": self.index_type,
            "embedding_dim": self.embedding_dim,
            "normalize_embeddings": self.normalize_embeddings,
            "distance_metric": self.distance_metric,
            "faiss_params": self.faiss_params,
            "initial_retrieval_k": self.initial_retrieval_k,
            "final_top_k": self.final_top_k,
            "reranking_method": self.reranking_method,
            "quantum_n_qubits": self.quantum_n_qubits,
            "quantum_n_layers": self.quantum_n_layers,
            "enable_caching": self.enable_caching,
            "cache_size": self.cache_size,
            "optimization_level": self.optimization_level.value,
            "use_gpu": self.use_gpu,
            "batch_size": self.batch_size,
            "num_threads": self.num_threads,
            "index_save_path": self.index_save_path,
            "auto_save": self.auto_save,
            "save_interval": self.save_interval,
            "enable_monitoring": self.enable_monitoring,
            "log_level": self.log_level,
            "collect_detailed_metrics": self.collect_detailed_metrics
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RetrievalConfig":
        """Create configuration from dictionary."""
        # Handle enum conversion
        if "optimization_level" in config_dict:
            config_dict["optimization_level"] = OptimizationLevel(config_dict["optimization_level"])
        
        return cls(**config_dict)
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        self.validate()
        
        # Apply optimization presets if not manually configured
        if hasattr(self, '_apply_presets'):
            self.apply_optimization_presets()
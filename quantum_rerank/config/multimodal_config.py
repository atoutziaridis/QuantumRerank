"""
Multimodal Medical Configuration for QuantumRerank.

This module provides configuration classes for multimodal medical data processing,
including text and clinical data integration with quantum compression.

Based on:
- QMMR-01 task requirements
- Research insights from QPMeL and quantum-inspired embeddings
- PRD performance constraints (<100ms, <2GB memory)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class MultimodalMedicalConfig:
    """Configuration for multimodal medical data processing."""
    
    # Text processing (preserve existing from EmbeddingProcessor)
    text_encoder: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    text_dim: int = 768
    
    # Clinical data processing (new - using Bio_ClinicalBERT)
    clinical_encoder: str = "emilyalsentzer/Bio_ClinicalBERT"
    clinical_dim: int = 768
    
    # Quantum compression (leverage existing QuantumCompressionHead)
    target_quantum_dim: int = 256  # 2^8 amplitudes for 8-qubit states
    compression_ratio: float = 6.0  # (768+768)/256 = 6:1 compression
    
    # Performance constraints (PRD compliance)
    max_latency_ms: float = 100.0  # <100ms per similarity computation
    batch_size: int = 50  # Batch processing for efficiency
    max_memory_gb: float = 2.0  # <2GB memory usage
    
    # Medical domain specific
    medical_abbreviation_expansion: bool = True
    clinical_entity_extraction: bool = True
    enable_medical_preprocessing: bool = True
    
    # Fallback behavior
    graceful_degradation: bool = True  # Continue with single modal if one fails
    missing_modality_handling: str = "zero_padding"  # "zero_padding", "skip", "error"
    
    # Caching for performance
    enable_embedding_cache: bool = True
    cache_size: int = 1000
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate compression ratio
        total_input_dim = self.text_dim + self.clinical_dim
        actual_compression_ratio = total_input_dim / self.target_quantum_dim
        
        if abs(actual_compression_ratio - self.compression_ratio) > 0.1:
            logger.warning(f"Compression ratio mismatch: expected {self.compression_ratio}, "
                         f"actual {actual_compression_ratio:.2f}")
            self.compression_ratio = actual_compression_ratio
        
        # Validate quantum dimension is power of 2
        if not (self.target_quantum_dim & (self.target_quantum_dim - 1)) == 0:
            raise ValueError(f"target_quantum_dim must be power of 2, got {self.target_quantum_dim}")
        
        # Validate missing modality handling
        valid_handlers = ["zero_padding", "skip", "error"]
        if self.missing_modality_handling not in valid_handlers:
            raise ValueError(f"missing_modality_handling must be one of {valid_handlers}")
        
        logger.info(f"MultimodalMedicalConfig initialized: "
                   f"text_dim={self.text_dim}, clinical_dim={self.clinical_dim}, "
                   f"target_quantum_dim={self.target_quantum_dim}, "
                   f"compression_ratio={self.compression_ratio:.2f}")


@dataclass 
class ClinicalDataConfig:
    """Configuration for clinical data processing."""
    
    # Clinical encoder settings
    encoder_model: str = "emilyalsentzer/Bio_ClinicalBERT"
    max_sequence_length: int = 512
    
    # Supported clinical data types
    supported_data_types: List[str] = field(default_factory=lambda: [
        "demographics", "vitals", "lab_results", "medications", 
        "procedures", "diagnoses", "symptoms", "allergies"
    ])
    
    # Data preprocessing
    normalize_values: bool = True
    handle_missing_values: str = "zero"  # "zero", "mean", "median", "skip"
    
    # Medical entity extraction
    extract_medical_entities: bool = True
    entity_types: List[str] = field(default_factory=lambda: [
        "DISEASE", "SYMPTOM", "MEDICATION", "PROCEDURE", "ANATOMY"
    ])
    
    # Clinical abbreviation expansion
    expand_abbreviations: bool = True
    custom_abbreviations: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate clinical data configuration."""
        valid_handlers = ["zero", "mean", "median", "skip"]
        if self.handle_missing_values not in valid_handlers:
            raise ValueError(f"handle_missing_values must be one of {valid_handlers}")
        
        logger.info(f"ClinicalDataConfig initialized: "
                   f"encoder={self.encoder_model}, "
                   f"supported_types={len(self.supported_data_types)}")


@dataclass
class QuantumFusionConfig:
    """Configuration for quantum-inspired multimodal fusion."""
    
    # Fusion strategy
    fusion_method: str = "parallel_compression"  # "parallel_compression", "sequential", "attention"
    
    # Quantum compression per modality
    text_compression_ratio: float = 6.0  # 768 -> 128
    clinical_compression_ratio: float = 6.0  # 768 -> 128
    
    # Quantum circuit constraints (from PRD)
    max_qubits: int = 4
    max_circuit_depth: int = 15
    
    # Fusion weights
    modality_weights: Dict[str, float] = field(default_factory=lambda: {
        "text": 0.6,
        "clinical": 0.4
    })
    
    # Quantum-inspired operations
    enable_entanglement: bool = True
    use_polar_encoding: bool = True  # From QPMeL research
    enable_quantum_noise: bool = False  # For training robustness
    
    def __post_init__(self):
        """Validate quantum fusion configuration."""
        # Validate fusion method
        valid_methods = ["parallel_compression", "sequential", "attention"]
        if self.fusion_method not in valid_methods:
            raise ValueError(f"fusion_method must be one of {valid_methods}")
        
        # Validate modality weights sum to 1
        total_weight = sum(self.modality_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Modality weights sum to {total_weight:.3f}, normalizing...")
            for key in self.modality_weights:
                self.modality_weights[key] /= total_weight
        
        # Validate quantum constraints
        if self.max_qubits > 8:  # Practical limit for classical simulation
            logger.warning(f"max_qubits={self.max_qubits} may be too large for classical simulation")
        
        if self.max_circuit_depth > 20:  # Practical limit for noise and performance
            logger.warning(f"max_circuit_depth={self.max_circuit_depth} may impact performance")
        
        logger.info(f"QuantumFusionConfig initialized: "
                   f"method={self.fusion_method}, "
                   f"weights={self.modality_weights}")


@dataclass
class MedicalDomainConfig:
    """Configuration for medical domain processing."""
    
    # Medical abbreviation expansion
    abbreviation_expansion: bool = True
    custom_abbreviations: Dict[str, str] = field(default_factory=dict)
    
    # Medical entity extraction
    entity_extraction: bool = True
    entity_model: str = "en_core_sci_sm"  # scispaCy model
    
    # Domain classification
    domain_classification: bool = True
    supported_domains: List[str] = field(default_factory=lambda: [
        "cardiology", "diabetes", "respiratory", "neurology", "oncology", "general"
    ])
    
    # Medical preprocessing
    normalize_medical_terms: bool = True
    expand_medical_acronyms: bool = True
    
    # Clinical context
    preserve_clinical_context: bool = True
    context_window_size: int = 5  # Sentences
    
    def __post_init__(self):
        """Validate medical domain configuration."""
        if len(self.supported_domains) == 0:
            raise ValueError("supported_domains cannot be empty")
        
        if self.context_window_size < 1:
            raise ValueError("context_window_size must be positive")
        
        logger.info(f"MedicalDomainConfig initialized: "
                   f"domains={len(self.supported_domains)}, "
                   f"abbreviations={self.abbreviation_expansion}")


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    
    # PRD constraints
    max_similarity_latency_ms: float = 100.0
    max_batch_latency_ms: float = 500.0  # For 50 documents
    max_memory_usage_gb: float = 2.0
    
    # Optimization settings
    enable_caching: bool = True
    cache_size: int = 1000
    enable_batching: bool = True
    batch_size: int = 32
    
    # Parallel processing
    enable_parallel_processing: bool = True
    max_workers: int = 4
    
    # Memory optimization
    enable_memory_optimization: bool = True
    garbage_collection_threshold: int = 100  # Operations before GC
    
    # Monitoring
    enable_performance_monitoring: bool = True
    log_performance_metrics: bool = True
    
    def __post_init__(self):
        """Validate performance configuration."""
        if self.max_similarity_latency_ms <= 0:
            raise ValueError("max_similarity_latency_ms must be positive")
        
        if self.max_batch_latency_ms <= 0:
            raise ValueError("max_batch_latency_ms must be positive")
        
        if self.max_memory_usage_gb <= 0:
            raise ValueError("max_memory_usage_gb must be positive")
        
        logger.info(f"PerformanceConfig initialized: "
                   f"similarity_latency={self.max_similarity_latency_ms}ms, "
                   f"batch_latency={self.max_batch_latency_ms}ms, "
                   f"memory={self.max_memory_usage_gb}GB")


# Convenience function for creating complete multimodal config
def create_multimodal_config(
    text_encoder: Optional[str] = None,
    clinical_encoder: Optional[str] = None,
    target_quantum_dim: Optional[int] = None,
    max_latency_ms: Optional[float] = None,
    custom_settings: Optional[Dict[str, Any]] = None
) -> MultimodalMedicalConfig:
    """
    Create MultimodalMedicalConfig with optional overrides.
    
    Args:
        text_encoder: Text encoder model name
        clinical_encoder: Clinical encoder model name
        target_quantum_dim: Target quantum dimension
        max_latency_ms: Maximum latency constraint
        custom_settings: Additional custom settings
    
    Returns:
        Configured MultimodalMedicalConfig
    """
    config_kwargs = {}
    
    if text_encoder:
        config_kwargs["text_encoder"] = text_encoder
    
    if clinical_encoder:
        config_kwargs["clinical_encoder"] = clinical_encoder
    
    if target_quantum_dim:
        config_kwargs["target_quantum_dim"] = target_quantum_dim
    
    if max_latency_ms:
        config_kwargs["max_latency_ms"] = max_latency_ms
    
    if custom_settings:
        config_kwargs.update(custom_settings)
    
    return MultimodalMedicalConfig(**config_kwargs)


# Default configurations for common use cases
DEFAULT_MULTIMODAL_CONFIG = MultimodalMedicalConfig()

FAST_MULTIMODAL_CONFIG = MultimodalMedicalConfig(
    max_latency_ms=50.0,
    batch_size=32,
    target_quantum_dim=128,  # Smaller for speed
    enable_embedding_cache=True
)

ACCURATE_MULTIMODAL_CONFIG = MultimodalMedicalConfig(
    max_latency_ms=200.0,  # Allow more time for accuracy
    batch_size=16,
    target_quantum_dim=512,  # Larger for accuracy
    clinical_entity_extraction=True,
    medical_abbreviation_expansion=True
)

MEMORY_OPTIMIZED_CONFIG = MultimodalMedicalConfig(
    max_latency_ms=100.0,
    batch_size=8,
    target_quantum_dim=128,
    enable_embedding_cache=False,  # Save memory
    missing_modality_handling="skip"
)
"""
Configuration settings for multimodal medical evaluation and optimization.

Provides comprehensive configuration for QMMR-05 evaluation framework including
performance targets, clinical validation parameters, and statistical testing settings.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class MultimodalMedicalEvaluationConfig:
    """Configuration for comprehensive multimodal medical evaluation."""
    
    # Dataset configuration
    min_multimodal_queries: int = 200
    min_documents_per_query: int = 100
    test_set_size: int = 1000
    validation_split: float = 0.2
    
    # Evaluation metrics
    primary_metrics: List[str] = field(default_factory=lambda: [
        'ndcg_at_10', 'map', 'mrr', 'precision_at_5', 'recall_at_20'
    ])
    
    # Medical-specific metrics
    medical_metrics: List[str] = field(default_factory=lambda: [
        'clinical_relevance', 'diagnostic_accuracy', 'safety_assessment',
        'treatment_recommendation_quality', 'workflow_integration_score'
    ])
    
    # Quantum-specific metrics
    quantum_metrics: List[str] = field(default_factory=lambda: [
        'quantum_advantage_score', 'entanglement_utilization', 'uncertainty_quality',
        'circuit_efficiency', 'fidelity_preservation'
    ])
    
    # Performance constraints (from PRD and task specifications)
    max_similarity_latency_ms: float = 150.0  # Increased for multimodal
    max_batch_latency_ms: float = 1000.0  # Increased for image processing
    max_memory_usage_gb: float = 4.0  # Increased for multimodal
    min_throughput_qps: float = 100.0  # Queries per second
    target_availability: float = 0.999  # 99.9% uptime
    
    # Statistical testing parameters
    significance_level: float = 0.05
    effect_size_threshold: float = 0.1  # Minimum meaningful improvement
    bootstrap_samples: int = 1000
    confidence_interval: float = 0.95
    
    # Medical validation parameters
    clinical_expert_validation: bool = True
    safety_assessment: bool = True
    privacy_compliance_check: bool = True
    regulatory_compliance_check: bool = True
    
    # Quantum evaluation parameters
    min_quantum_advantage: float = 0.05  # 5% improvement threshold
    max_circuit_depth: int = 15
    max_qubits: int = 4
    min_entanglement_score: float = 0.1
    
    # Optimization parameters
    optimization_iterations: int = 10
    optimization_timeout_minutes: int = 30
    memory_optimization: bool = True
    circuit_optimization: bool = True
    
    # Complexity assessment levels
    complexity_levels: List[str] = field(default_factory=lambda: [
        'simple', 'moderate', 'complex', 'very_complex'
    ])
    
    # Medical scenario types
    medical_scenarios: List[str] = field(default_factory=lambda: [
        'diagnostic_inquiry', 'treatment_recommendation', 'imaging_interpretation',
        'clinical_correlation', 'emergency_assessment', 'follow_up_care'
    ])
    
    # Noise injection parameters for robustness testing
    ocr_error_rate: float = 0.05
    missing_data_rate: float = 0.1
    image_artifact_probability: float = 0.15
    text_noise_level: float = 0.02
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.min_multimodal_queries < 50:
            raise ValueError("Minimum queries must be at least 50 for reliable evaluation")
        
        if self.test_set_size < self.min_multimodal_queries:
            raise ValueError("Test set size must be >= min_multimodal_queries")
        
        if not 0 < self.validation_split < 1:
            raise ValueError("Validation split must be between 0 and 1")
        
        if self.significance_level <= 0 or self.significance_level >= 1:
            raise ValueError("Significance level must be between 0 and 1")
        
        if self.min_quantum_advantage <= 0:
            raise ValueError("Quantum advantage threshold must be positive")


@dataclass
class DatasetGenerationConfig:
    """Configuration for multimodal medical dataset generation."""
    
    # Medical data sources
    medical_text_templates: int = 100
    clinical_data_templates: int = 50
    medical_image_templates: int = 75
    
    # Modality distribution
    text_only_ratio: float = 0.2
    image_only_ratio: float = 0.1
    text_image_ratio: float = 0.3
    text_clinical_ratio: float = 0.2
    all_modalities_ratio: float = 0.2
    
    # Medical specialties distribution
    specialty_distribution: Dict[str, float] = field(default_factory=lambda: {
        'radiology': 0.3,
        'cardiology': 0.2,
        'pulmonology': 0.15,
        'neurology': 0.15,
        'emergency_medicine': 0.1,
        'general_medicine': 0.1
    })
    
    # Image modality distribution
    image_modality_distribution: Dict[str, float] = field(default_factory=lambda: {
        'XR': 0.4,  # X-ray
        'CT': 0.25,  # CT scan
        'MR': 0.2,   # MRI
        'US': 0.1,   # Ultrasound
        'MG': 0.05   # Mammography
    })
    
    # Ground truth annotation quality
    annotation_quality_threshold: float = 0.8
    inter_annotator_agreement_threshold: float = 0.7
    clinical_expert_review: bool = True
    
    def __post_init__(self):
        """Validate dataset generation configuration."""
        modality_ratios = [
            self.text_only_ratio, self.image_only_ratio, self.text_image_ratio,
            self.text_clinical_ratio, self.all_modalities_ratio
        ]
        
        if abs(sum(modality_ratios) - 1.0) > 0.01:
            raise ValueError("Modality ratios must sum to 1.0")
        
        if abs(sum(self.specialty_distribution.values()) - 1.0) > 0.01:
            raise ValueError("Specialty distribution must sum to 1.0")
        
        if abs(sum(self.image_modality_distribution.values()) - 1.0) > 0.01:
            raise ValueError("Image modality distribution must sum to 1.0")


@dataclass
class QuantumAdvantageConfig:
    """Configuration for quantum advantage assessment."""
    
    # Classical baseline systems
    classical_baselines: List[str] = field(default_factory=lambda: [
        'bm25', 'bert', 'clip', 'multimodal_transformer', 'dense_retrieval'
    ])
    
    # Quantum advantage metrics
    advantage_metrics: List[str] = field(default_factory=lambda: [
        'accuracy_improvement', 'latency_efficiency', 'memory_efficiency',
        'robustness_improvement', 'uncertainty_quality'
    ])
    
    # Statistical testing
    statistical_tests: List[str] = field(default_factory=lambda: [
        'paired_t_test', 'wilcoxon_signed_rank', 'bootstrap_ci', 'effect_size'
    ])
    
    # Quantum-specific assessments
    entanglement_assessment: bool = True
    quantum_fidelity_assessment: bool = True
    circuit_efficiency_assessment: bool = True
    
    # Advantage thresholds
    min_accuracy_improvement: float = 0.02  # 2% minimum
    min_statistical_significance: float = 0.05
    min_effect_size: float = 0.1


@dataclass
class ClinicalValidationConfig:
    """Configuration for clinical validation framework."""
    
    # Clinical expert panel
    expert_panel_size: int = 5
    clinical_specialties: List[str] = field(default_factory=lambda: [
        'radiology', 'emergency_medicine', 'internal_medicine', 'cardiology'
    ])
    
    # Safety assessment parameters
    safety_threshold: float = 0.95
    adverse_event_tolerance: float = 0.01
    clinical_workflow_disruption_threshold: float = 0.1
    
    # Privacy and compliance
    hipaa_compliance_required: bool = True
    gdpr_compliance_required: bool = True
    phi_detection_threshold: float = 0.99
    
    # Regulatory requirements
    fda_guidance_compliance: bool = True
    clinical_evidence_quality: str = "high"
    validation_study_design: str = "prospective"
    
    # Clinical utility metrics
    diagnostic_accuracy_threshold: float = 0.9
    treatment_recommendation_quality_threshold: float = 0.85
    workflow_integration_score_threshold: float = 0.8
    time_efficiency_improvement_threshold: float = 0.1  # 10% improvement


@dataclass
class PerformanceOptimizationConfig:
    """Configuration for performance optimization."""
    
    # Optimization targets
    target_latency_ms: float = 100.0  # Target below max constraint
    target_memory_gb: float = 2.0     # Target below max constraint
    target_throughput_qps: float = 150.0  # Target above min constraint
    
    # Optimization strategies
    quantum_circuit_optimization: bool = True
    embedding_compression_optimization: bool = True
    batch_processing_optimization: bool = True
    caching_optimization: bool = True
    parallelization_optimization: bool = True
    
    # Memory optimization
    gradient_checkpointing: bool = True
    model_pruning: bool = True
    quantization: bool = True
    
    # Circuit optimization
    gate_fusion: bool = True
    circuit_compilation: bool = True
    noise_adaptive_compilation: bool = True
    
    # Caching configuration
    embedding_cache_size: int = 10000
    circuit_cache_size: int = 1000
    result_cache_ttl_minutes: int = 60
    
    # Parallelization
    max_worker_threads: int = 8
    batch_processing_workers: int = 4
    async_processing: bool = True


@dataclass
class ComprehensiveEvaluationConfig:
    """Master configuration combining all evaluation components."""
    
    def __init__(
        self,
        evaluation_config: Optional[MultimodalMedicalEvaluationConfig] = None,
        dataset_config: Optional[DatasetGenerationConfig] = None,
        quantum_config: Optional[QuantumAdvantageConfig] = None,
        clinical_config: Optional[ClinicalValidationConfig] = None,
        optimization_config: Optional[PerformanceOptimizationConfig] = None
    ):
        self.evaluation = evaluation_config or MultimodalMedicalEvaluationConfig()
        self.dataset = dataset_config or DatasetGenerationConfig()
        self.quantum_advantage = quantum_config or QuantumAdvantageConfig()
        self.clinical_validation = clinical_config or ClinicalValidationConfig()
        self.performance_optimization = optimization_config or PerformanceOptimizationConfig()
    
    def validate_configuration(self) -> bool:
        """Validate the complete configuration for consistency."""
        try:
            # Check consistency between evaluation and dataset configs
            if self.evaluation.test_set_size < self.evaluation.min_multimodal_queries:
                return False
            
            # Check quantum advantage thresholds are achievable
            if self.quantum_advantage.min_accuracy_improvement > self.evaluation.min_quantum_advantage:
                return False
            
            # Check performance targets are consistent
            if self.optimization_config.target_latency_ms > self.evaluation.max_similarity_latency_ms:
                return False
            
            return True
            
        except Exception:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'evaluation': self.evaluation.__dict__,
            'dataset': self.dataset.__dict__,
            'quantum_advantage': self.quantum_advantage.__dict__,
            'clinical_validation': self.clinical_validation.__dict__,
            'performance_optimization': self.performance_optimization.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ComprehensiveEvaluationConfig':
        """Create configuration from dictionary."""
        return cls(
            evaluation_config=MultimodalMedicalEvaluationConfig(**config_dict.get('evaluation', {})),
            dataset_config=DatasetGenerationConfig(**config_dict.get('dataset', {})),
            quantum_config=QuantumAdvantageConfig(**config_dict.get('quantum_advantage', {})),
            clinical_config=ClinicalValidationConfig(**config_dict.get('clinical_validation', {})),
            optimization_config=PerformanceOptimizationConfig(**config_dict.get('performance_optimization', {}))
        )


# Default configuration instance
DEFAULT_EVALUATION_CONFIG = ComprehensiveEvaluationConfig()
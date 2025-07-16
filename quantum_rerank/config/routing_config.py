"""
Configuration for Quantum-Classical Routing System.

This module provides configuration classes for the routing system components
including complexity assessment, routing decisions, and A/B testing.

Based on:
- QMMR-02 task specification
- Performance requirements (<50ms assessment, <500ms total)
- Medical domain requirements
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class RoutingMethod(Enum):
    """Available routing methods."""
    CLASSICAL = "classical"
    QUANTUM = "quantum"
    HYBRID = "hybrid"


class ABTestStrategy(Enum):
    """A/B testing strategies."""
    RANDOM = "random"
    COMPLEXITY_BASED = "complexity_based"
    PERFORMANCE_BASED = "performance_based"
    USER_BASED = "user_based"


@dataclass
class RoutingConfig:
    """Configuration for routing decision engine."""
    
    # Routing thresholds
    quantum_threshold: float = 0.6  # Route to quantum if complexity > 0.6
    classical_threshold: float = 0.4  # Route to classical if complexity < 0.4
    hybrid_threshold_range: tuple = (0.4, 0.6)  # Hybrid range
    
    # Adaptive routing parameters
    enable_adaptive_routing: bool = True
    adaptation_learning_rate: float = 0.1
    adaptation_window_size: int = 100  # Recent decisions to consider
    
    # Confidence thresholds
    min_routing_confidence: float = 0.7  # Minimum confidence for routing
    uncertainty_threshold: float = 0.8  # High uncertainty threshold
    
    # Performance-based routing
    enable_performance_routing: bool = True
    latency_threshold_ms: float = 100.0  # Max latency for quantum routing
    memory_threshold_mb: float = 512.0  # Max memory for quantum routing
    
    # Domain-specific routing rules
    enable_domain_specific_rules: bool = True
    emergency_medicine_quantum_preference: bool = True
    complex_diagnosis_quantum_preference: bool = True
    simple_query_classical_preference: bool = True
    
    # Fallback behavior
    fallback_method: RoutingMethod = RoutingMethod.CLASSICAL
    enable_fallback_on_error: bool = True
    
    # Monitoring and logging
    enable_routing_metrics: bool = True
    log_routing_decisions: bool = True
    metrics_collection_interval: int = 60  # seconds
    
    def __post_init__(self):
        """Validate configuration."""
        if self.quantum_threshold <= self.classical_threshold:
            raise ValueError("quantum_threshold must be greater than classical_threshold")
        
        if not (0 <= self.classical_threshold <= 1):
            raise ValueError("classical_threshold must be between 0 and 1")
        
        if not (0 <= self.quantum_threshold <= 1):
            raise ValueError("quantum_threshold must be between 0 and 1")


@dataclass
class HybridPipelineConfig:
    """Configuration for hybrid quantum-classical pipeline."""
    
    # Component configurations
    routing_config: RoutingConfig = field(default_factory=RoutingConfig)
    
    # Pipeline performance targets
    max_total_latency_ms: float = 500.0  # PRD requirement
    max_memory_usage_mb: float = 2048.0  # PRD requirement
    
    # Reranking parameters
    default_top_k: int = 10
    max_candidates: int = 100
    enable_batch_processing: bool = True
    batch_size: int = 50
    
    # Classical reranker settings
    classical_reranker_type: str = "sentence_bert"  # or "bm25", "cross_encoder"
    classical_model_name: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    
    # Quantum reranker settings
    quantum_n_qubits: int = 4
    quantum_n_layers: int = 2
    quantum_enable_multimodal: bool = True
    
    # Hybrid combination settings
    hybrid_weight_classical: float = 0.3
    hybrid_weight_quantum: float = 0.7
    hybrid_combination_method: str = "weighted_average"  # or "rank_fusion", "ensemble"
    
    # Caching settings
    enable_result_caching: bool = True
    cache_size: int = 1000
    cache_ttl_seconds: int = 3600
    
    # Monitoring settings
    enable_performance_monitoring: bool = True
    enable_accuracy_monitoring: bool = True
    monitoring_sample_rate: float = 0.1  # 10% sampling
    
    # A/B testing integration
    enable_ab_testing: bool = False
    ab_test_sample_rate: float = 0.2  # 20% in A/B tests
    
    def __post_init__(self):
        """Validate configuration."""
        if self.hybrid_weight_classical + self.hybrid_weight_quantum != 1.0:
            raise ValueError("Hybrid weights must sum to 1.0")
        
        if not (0 < self.monitoring_sample_rate <= 1.0):
            raise ValueError("monitoring_sample_rate must be between 0 and 1")


@dataclass
class ABTestConfig:
    """Configuration for A/B testing framework."""
    
    # Test identification
    test_name: str
    test_description: str = ""
    
    # Test strategy
    strategy: ABTestStrategy = ABTestStrategy.RANDOM
    
    # Test groups
    control_method: RoutingMethod = RoutingMethod.CLASSICAL
    treatment_method: RoutingMethod = RoutingMethod.QUANTUM
    
    # Sample allocation
    control_allocation: float = 0.5
    treatment_allocation: float = 0.5
    
    # Test parameters
    sample_size: int = 1000
    min_sample_size: int = 100
    max_duration_hours: int = 168  # 1 week
    
    # Statistical testing
    significance_level: float = 0.05
    power: float = 0.8
    minimum_detectable_effect: float = 0.05
    
    # Complexity-based testing (if strategy is COMPLEXITY_BASED)
    complexity_threshold: float = 0.5
    complexity_bands: List[tuple] = field(default_factory=lambda: [(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)])
    
    # Performance-based testing (if strategy is PERFORMANCE_BASED)
    latency_threshold_ms: float = 100.0
    accuracy_threshold: float = 0.8
    
    # User-based testing (if strategy is USER_BASED)
    user_segments: List[str] = field(default_factory=lambda: ["all"])
    
    # Monitoring and early stopping
    enable_early_stopping: bool = True
    early_stopping_confidence: float = 0.95
    monitoring_frequency_hours: int = 24
    
    # Metrics to track
    primary_metric: str = "accuracy"  # or "latency", "user_satisfaction"
    secondary_metrics: List[str] = field(default_factory=lambda: ["latency", "memory_usage"])
    
    def __post_init__(self):
        """Validate configuration."""
        if abs(self.control_allocation + self.treatment_allocation - 1.0) > 0.01:
            raise ValueError("Control and treatment allocations must sum to 1.0")
        
        if self.sample_size < self.min_sample_size:
            raise ValueError("sample_size must be >= min_sample_size")
        
        if not (0 < self.significance_level < 1):
            raise ValueError("significance_level must be between 0 and 1")


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring and optimization."""
    
    # Latency targets
    max_complexity_assessment_ms: float = 50.0
    max_routing_decision_ms: float = 10.0
    max_classical_reranking_ms: float = 200.0
    max_quantum_reranking_ms: float = 300.0
    max_total_pipeline_ms: float = 500.0
    
    # Memory targets
    max_complexity_assessment_mb: float = 100.0
    max_routing_decision_mb: float = 50.0
    max_classical_reranking_mb: float = 1000.0
    max_quantum_reranking_mb: float = 1500.0
    max_total_pipeline_mb: float = 2048.0
    
    # Throughput targets
    min_queries_per_second: float = 10.0
    target_queries_per_second: float = 50.0
    
    # Quality targets
    min_routing_accuracy: float = 0.8
    min_reranking_accuracy: float = 0.85
    
    # Monitoring settings
    enable_real_time_monitoring: bool = True
    monitoring_window_seconds: int = 60
    alert_threshold_multiplier: float = 1.5  # Alert if exceeding target by this factor
    
    # Circuit breaker settings
    enable_circuit_breaker: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_reset_timeout_seconds: int = 60
    
    # Auto-scaling settings
    enable_auto_scaling: bool = True
    scale_up_threshold: float = 0.8  # Scale up if utilization > 80%
    scale_down_threshold: float = 0.3  # Scale down if utilization < 30%
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_total_pipeline_ms < (
            self.max_complexity_assessment_ms + 
            self.max_routing_decision_ms + 
            max(self.max_classical_reranking_ms, self.max_quantum_reranking_ms)
        ):
            raise ValueError("Total pipeline time must be >= sum of component times")


@dataclass
class MedicalDomainConfig:
    """Configuration for medical domain-specific routing."""
    
    # Domain-specific thresholds
    domain_complexity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "emergency_medicine": 0.8,
        "critical_care": 0.9,
        "diagnostic_imaging": 0.7,
        "pathology": 0.8,
        "cardiology": 0.6,
        "oncology": 0.7,
        "neurology": 0.8,
        "general_medicine": 0.5
    })
    
    # Specialty routing preferences
    quantum_preferred_specialties: List[str] = field(default_factory=lambda: [
        "emergency_medicine",
        "critical_care", 
        "diagnostic_imaging",
        "pathology",
        "complex_diagnosis"
    ])
    
    classical_preferred_specialties: List[str] = field(default_factory=lambda: [
        "general_medicine",
        "routine_checkup",
        "medication_refill",
        "administrative"
    ])
    
    # Clinical complexity factors
    enable_clinical_complexity_assessment: bool = True
    clinical_data_weight: float = 0.4
    text_weight: float = 0.6
    
    # Medical terminology handling
    enable_medical_terminology_boost: bool = True
    terminology_complexity_multiplier: float = 1.2
    
    # Uncertainty handling in medical context
    enable_medical_uncertainty_routing: bool = True
    diagnostic_uncertainty_threshold: float = 0.7
    
    # Integration with medical knowledge bases
    enable_medical_knowledge_integration: bool = True
    medical_knowledge_sources: List[str] = field(default_factory=lambda: [
        "mesh",
        "snomed",
        "icd10"
    ])
    
    def __post_init__(self):
        """Validate configuration."""
        if not (0 <= self.clinical_data_weight <= 1):
            raise ValueError("clinical_data_weight must be between 0 and 1")
        
        if not (0 <= self.text_weight <= 1):
            raise ValueError("text_weight must be between 0 and 1")
        
        if abs(self.clinical_data_weight + self.text_weight - 1.0) > 0.01:
            raise ValueError("clinical_data_weight and text_weight must sum to 1.0")


@dataclass
class QuantumAdvantageConfig:
    """Configuration for quantum advantage estimation."""
    
    # Quantum advantage factors
    entanglement_weight: float = 0.3
    superposition_weight: float = 0.25
    interference_weight: float = 0.25
    noise_resilience_weight: float = 0.2
    
    # Quantum advantage thresholds
    min_advantage_threshold: float = 0.1  # Minimum advantage to use quantum
    significant_advantage_threshold: float = 0.3  # Significant quantum advantage
    
    # Quantum circuit constraints
    max_qubits: int = 4
    max_circuit_depth: int = 15
    max_gate_count: int = 100
    
    # Quantum simulation settings
    enable_quantum_simulation: bool = True
    simulation_shots: int = 1000
    noise_model: Optional[str] = None  # "ibmq_qasm_simulator" or None
    
    # Advantage estimation method
    estimation_method: str = "heuristic"  # or "simulation", "analytical"
    
    def __post_init__(self):
        """Validate configuration."""
        total_weight = (
            self.entanglement_weight + 
            self.superposition_weight + 
            self.interference_weight + 
            self.noise_resilience_weight
        )
        
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError("Quantum advantage weights must sum to 1.0")
        
        if not (0 <= self.min_advantage_threshold <= 1):
            raise ValueError("min_advantage_threshold must be between 0 and 1")


# Default configurations for common use cases
DEFAULT_ROUTING_CONFIG = RoutingConfig()
DEFAULT_HYBRID_PIPELINE_CONFIG = HybridPipelineConfig()
DEFAULT_PERFORMANCE_CONFIG = PerformanceConfig()
DEFAULT_MEDICAL_DOMAIN_CONFIG = MedicalDomainConfig()
DEFAULT_QUANTUM_ADVANTAGE_CONFIG = QuantumAdvantageConfig()

# Development/testing configurations
DEVELOPMENT_ROUTING_CONFIG = RoutingConfig(
    quantum_threshold=0.5,
    classical_threshold=0.3,
    enable_adaptive_routing=True,
    log_routing_decisions=True
)

TESTING_HYBRID_PIPELINE_CONFIG = HybridPipelineConfig(
    max_total_latency_ms=1000.0,  # More lenient for testing
    enable_performance_monitoring=True,
    enable_ab_testing=True,
    ab_test_sample_rate=0.5
)

# Production configurations
PRODUCTION_ROUTING_CONFIG = RoutingConfig(
    quantum_threshold=0.7,
    classical_threshold=0.4,
    enable_adaptive_routing=True,
    enable_performance_routing=True,
    log_routing_decisions=False  # Reduce logging overhead
)

PRODUCTION_HYBRID_PIPELINE_CONFIG = HybridPipelineConfig(
    max_total_latency_ms=500.0,
    enable_performance_monitoring=True,
    enable_result_caching=True,
    cache_size=5000,
    enable_ab_testing=False  # Disable in production unless needed
)
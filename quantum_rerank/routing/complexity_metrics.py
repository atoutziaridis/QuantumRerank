"""
Complexity Metrics for Quantum-Classical Routing.

This module defines the data structures and metrics used to assess
query and document complexity for routing decisions.

Based on:
- QMMR-02 task specification
- Quantum research insights on complexity assessment
- Medical domain complexity characteristics
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class ComplexityDimension(Enum):
    """Dimensions of complexity for routing assessment."""
    MULTIMODAL = "multimodal"
    NOISE = "noise"
    UNCERTAINTY = "uncertainty"
    MEDICAL_DOMAIN = "medical_domain"
    OVERALL = "overall"


@dataclass
class ComplexityMetrics:
    """Comprehensive complexity metrics for medical queries and documents."""
    
    # Multimodal complexity indicators
    modality_count: int = 0
    modality_diversity: float = 0.0  # Entropy of modality distribution
    cross_modal_dependencies: float = 0.0  # Correlation between modalities
    text_clinical_correlation: float = 0.0  # Specific text-clinical correlation
    
    # Noise indicators
    ocr_error_probability: float = 0.0  # Estimated OCR error rate
    abbreviation_density: float = 0.0  # Fraction of abbreviated terms
    missing_data_ratio: float = 0.0  # Fraction of missing clinical data
    typo_probability: float = 0.0  # Estimated typo rate
    
    # Uncertainty markers
    term_ambiguity_score: float = 0.0  # Ambiguous medical terms
    conflicting_information: float = 0.0  # Contradictory data points
    diagnostic_uncertainty: float = 0.0  # Uncertainty in diagnosis
    confidence_variance: float = 0.0  # Variance in confidence scores
    
    # Medical domain complexity
    medical_terminology_density: float = 0.0  # Specialized term frequency
    clinical_correlation_complexity: float = 0.0  # Inter-variable complexity
    domain_specificity: float = 0.0  # Domain-specific content ratio
    semantic_depth: float = 0.0  # Semantic relationship complexity
    
    # Quantum-inspired complexity metrics
    quantum_entanglement_potential: float = 0.0  # Cross-modal entanglement benefit
    interference_complexity: float = 0.0  # Quantum interference potential
    superposition_benefit: float = 0.0  # Quantum superposition advantage
    
    # Overall complexity score (0.0 to 1.0)
    overall_complexity: float = 0.0
    
    # Metadata
    assessment_confidence: float = 0.0  # Confidence in complexity assessment
    processing_time_ms: float = 0.0  # Time taken for assessment
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary for serialization."""
        return {
            'modality_count': float(self.modality_count),
            'modality_diversity': self.modality_diversity,
            'cross_modal_dependencies': self.cross_modal_dependencies,
            'text_clinical_correlation': self.text_clinical_correlation,
            'ocr_error_probability': self.ocr_error_probability,
            'abbreviation_density': self.abbreviation_density,
            'missing_data_ratio': self.missing_data_ratio,
            'typo_probability': self.typo_probability,
            'term_ambiguity_score': self.term_ambiguity_score,
            'conflicting_information': self.conflicting_information,
            'diagnostic_uncertainty': self.diagnostic_uncertainty,
            'confidence_variance': self.confidence_variance,
            'medical_terminology_density': self.medical_terminology_density,
            'clinical_correlation_complexity': self.clinical_correlation_complexity,
            'domain_specificity': self.domain_specificity,
            'semantic_depth': self.semantic_depth,
            'quantum_entanglement_potential': self.quantum_entanglement_potential,
            'interference_complexity': self.interference_complexity,
            'superposition_benefit': self.superposition_benefit,
            'overall_complexity': self.overall_complexity,
            'assessment_confidence': self.assessment_confidence,
            'processing_time_ms': self.processing_time_ms
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'ComplexityMetrics':
        """Create ComplexityMetrics from dictionary."""
        return cls(
            modality_count=int(data.get('modality_count', 0)),
            modality_diversity=data.get('modality_diversity', 0.0),
            cross_modal_dependencies=data.get('cross_modal_dependencies', 0.0),
            text_clinical_correlation=data.get('text_clinical_correlation', 0.0),
            ocr_error_probability=data.get('ocr_error_probability', 0.0),
            abbreviation_density=data.get('abbreviation_density', 0.0),
            missing_data_ratio=data.get('missing_data_ratio', 0.0),
            typo_probability=data.get('typo_probability', 0.0),
            term_ambiguity_score=data.get('term_ambiguity_score', 0.0),
            conflicting_information=data.get('conflicting_information', 0.0),
            diagnostic_uncertainty=data.get('diagnostic_uncertainty', 0.0),
            confidence_variance=data.get('confidence_variance', 0.0),
            medical_terminology_density=data.get('medical_terminology_density', 0.0),
            clinical_correlation_complexity=data.get('clinical_correlation_complexity', 0.0),
            domain_specificity=data.get('domain_specificity', 0.0),
            semantic_depth=data.get('semantic_depth', 0.0),
            quantum_entanglement_potential=data.get('quantum_entanglement_potential', 0.0),
            interference_complexity=data.get('interference_complexity', 0.0),
            superposition_benefit=data.get('superposition_benefit', 0.0),
            overall_complexity=data.get('overall_complexity', 0.0),
            assessment_confidence=data.get('assessment_confidence', 0.0),
            processing_time_ms=data.get('processing_time_ms', 0.0)
        )


@dataclass
class ComplexityAssessmentResult:
    """Result of complexity assessment for routing decisions."""
    
    # Query and candidate complexities
    query_complexity: ComplexityMetrics
    candidate_complexities: List[ComplexityMetrics]
    
    # Overall assessment
    overall_complexity: ComplexityMetrics
    
    # Routing recommendation
    routing_recommendation: str  # "classical", "quantum", "hybrid"
    routing_confidence: float  # Confidence in routing decision
    
    # Performance metrics
    assessment_time_ms: float
    success: bool
    error_message: Optional[str] = None
    
    # Detailed analysis
    complexity_breakdown: Dict[ComplexityDimension, float] = None
    quantum_advantage_score: float = 0.0  # Estimated quantum advantage
    
    def __post_init__(self):
        """Initialize complexity breakdown if not provided."""
        if self.complexity_breakdown is None:
            self.complexity_breakdown = self._compute_complexity_breakdown()
    
    def _compute_complexity_breakdown(self) -> Dict[ComplexityDimension, float]:
        """Compute complexity breakdown by dimension."""
        overall = self.overall_complexity
        
        # Multimodal complexity
        multimodal_score = (
            overall.modality_count / 3.0 * 0.3 +  # Max 3 modalities
            overall.modality_diversity * 0.4 +
            overall.cross_modal_dependencies * 0.3
        )
        
        # Noise complexity
        noise_score = (
            overall.ocr_error_probability * 0.3 +
            overall.abbreviation_density * 0.3 +
            overall.missing_data_ratio * 0.2 +
            overall.typo_probability * 0.2
        )
        
        # Uncertainty complexity
        uncertainty_score = (
            overall.term_ambiguity_score * 0.3 +
            overall.conflicting_information * 0.3 +
            overall.diagnostic_uncertainty * 0.2 +
            overall.confidence_variance * 0.2
        )
        
        # Medical domain complexity
        medical_score = (
            overall.medical_terminology_density * 0.3 +
            overall.clinical_correlation_complexity * 0.3 +
            overall.domain_specificity * 0.2 +
            overall.semantic_depth * 0.2
        )
        
        return {
            ComplexityDimension.MULTIMODAL: min(multimodal_score, 1.0),
            ComplexityDimension.NOISE: min(noise_score, 1.0),
            ComplexityDimension.UNCERTAINTY: min(uncertainty_score, 1.0),
            ComplexityDimension.MEDICAL_DOMAIN: min(medical_score, 1.0),
            ComplexityDimension.OVERALL: overall.overall_complexity
        }


@dataclass
class ComplexityAssessmentConfig:
    """Configuration for complexity assessment engine."""
    
    # Assessment weights for different dimensions
    multimodal_weight: float = 0.25
    noise_weight: float = 0.25
    uncertainty_weight: float = 0.25
    medical_domain_weight: float = 0.25
    
    # Quantum-inspired assessment parameters
    enable_quantum_metrics: bool = True
    quantum_entanglement_threshold: float = 0.5
    interference_analysis_depth: int = 3
    
    # Performance constraints
    max_assessment_time_ms: float = 50.0  # Part of 100ms total budget
    enable_caching: bool = True
    cache_size: int = 1000
    
    # Thresholds for complexity classification
    low_complexity_threshold: float = 0.3
    medium_complexity_threshold: float = 0.6
    high_complexity_threshold: float = 0.8
    
    # Noise simulation parameters
    enable_noise_simulation: bool = True
    ocr_error_simulation_rate: float = 0.02
    abbreviation_expansion_rate: float = 0.3
    
    # Uncertainty quantification
    enable_uncertainty_quantification: bool = True
    uncertainty_confidence_threshold: float = 0.7
    
    # Medical domain parameters
    medical_terminology_threshold: float = 0.1
    clinical_correlation_threshold: float = 0.5
    domain_specificity_threshold: float = 0.6
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Ensure weights sum to 1.0
        total_weight = (
            self.multimodal_weight + 
            self.noise_weight + 
            self.uncertainty_weight + 
            self.medical_domain_weight
        )
        
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Complexity assessment weights must sum to 1.0, got {total_weight}")
        
        # Validate thresholds
        if not (0 <= self.low_complexity_threshold <= 
                self.medium_complexity_threshold <= 
                self.high_complexity_threshold <= 1.0):
            raise ValueError("Complexity thresholds must be in ascending order between 0 and 1")


@dataclass
class ComplexityTrend:
    """Tracks complexity trends over time for optimization."""
    
    timestamp: float
    complexity_metrics: ComplexityMetrics
    routing_decision: str
    routing_accuracy: float
    processing_time_ms: float
    
    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp == 0:
            import time
            self.timestamp = time.time()


class ComplexityAggregator:
    """Aggregates complexity metrics across multiple queries/documents."""
    
    def __init__(self):
        self.metrics_history: List[ComplexityMetrics] = []
        self.trends: List[ComplexityTrend] = []
    
    def add_metrics(self, metrics: ComplexityMetrics):
        """Add complexity metrics to aggregator."""
        self.metrics_history.append(metrics)
    
    def get_average_complexity(self) -> ComplexityMetrics:
        """Compute average complexity across all metrics."""
        if not self.metrics_history:
            return ComplexityMetrics()
        
        # Compute averages across all metrics
        avg_metrics = ComplexityMetrics()
        n = len(self.metrics_history)
        
        for metrics in self.metrics_history:
            avg_metrics.modality_count += metrics.modality_count
            avg_metrics.modality_diversity += metrics.modality_diversity
            avg_metrics.cross_modal_dependencies += metrics.cross_modal_dependencies
            avg_metrics.text_clinical_correlation += metrics.text_clinical_correlation
            avg_metrics.ocr_error_probability += metrics.ocr_error_probability
            avg_metrics.abbreviation_density += metrics.abbreviation_density
            avg_metrics.missing_data_ratio += metrics.missing_data_ratio
            avg_metrics.typo_probability += metrics.typo_probability
            avg_metrics.term_ambiguity_score += metrics.term_ambiguity_score
            avg_metrics.conflicting_information += metrics.conflicting_information
            avg_metrics.diagnostic_uncertainty += metrics.diagnostic_uncertainty
            avg_metrics.confidence_variance += metrics.confidence_variance
            avg_metrics.medical_terminology_density += metrics.medical_terminology_density
            avg_metrics.clinical_correlation_complexity += metrics.clinical_correlation_complexity
            avg_metrics.domain_specificity += metrics.domain_specificity
            avg_metrics.semantic_depth += metrics.semantic_depth
            avg_metrics.quantum_entanglement_potential += metrics.quantum_entanglement_potential
            avg_metrics.interference_complexity += metrics.interference_complexity
            avg_metrics.superposition_benefit += metrics.superposition_benefit
            avg_metrics.overall_complexity += metrics.overall_complexity
            avg_metrics.assessment_confidence += metrics.assessment_confidence
            avg_metrics.processing_time_ms += metrics.processing_time_ms
        
        # Divide by count for averages
        avg_metrics.modality_count = int(avg_metrics.modality_count / n)
        avg_metrics.modality_diversity /= n
        avg_metrics.cross_modal_dependencies /= n
        avg_metrics.text_clinical_correlation /= n
        avg_metrics.ocr_error_probability /= n
        avg_metrics.abbreviation_density /= n
        avg_metrics.missing_data_ratio /= n
        avg_metrics.typo_probability /= n
        avg_metrics.term_ambiguity_score /= n
        avg_metrics.conflicting_information /= n
        avg_metrics.diagnostic_uncertainty /= n
        avg_metrics.confidence_variance /= n
        avg_metrics.medical_terminology_density /= n
        avg_metrics.clinical_correlation_complexity /= n
        avg_metrics.domain_specificity /= n
        avg_metrics.semantic_depth /= n
        avg_metrics.quantum_entanglement_potential /= n
        avg_metrics.interference_complexity /= n
        avg_metrics.superposition_benefit /= n
        avg_metrics.overall_complexity /= n
        avg_metrics.assessment_confidence /= n
        avg_metrics.processing_time_ms /= n
        
        return avg_metrics
    
    def get_complexity_distribution(self) -> Dict[str, float]:
        """Get distribution of complexity levels."""
        if not self.metrics_history:
            return {'low': 0.0, 'medium': 0.0, 'high': 0.0}
        
        low_count = sum(1 for m in self.metrics_history if m.overall_complexity < 0.3)
        medium_count = sum(1 for m in self.metrics_history if 0.3 <= m.overall_complexity < 0.6)
        high_count = sum(1 for m in self.metrics_history if m.overall_complexity >= 0.6)
        
        total = len(self.metrics_history)
        
        return {
            'low': low_count / total,
            'medium': medium_count / total,
            'high': high_count / total
        }
    
    def get_complexity_trends(self) -> List[ComplexityTrend]:
        """Get complexity trends over time."""
        return self.trends.copy()
    
    def add_trend(self, trend: ComplexityTrend):
        """Add complexity trend data."""
        self.trends.append(trend)
    
    def clear_history(self):
        """Clear complexity history."""
        self.metrics_history.clear()
        self.trends.clear()
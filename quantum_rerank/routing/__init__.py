"""
Quantum-Classical Routing System for QuantumRerank.

This module implements intelligent routing between classical and quantum rerankers
based on complexity assessment and performance optimization.

Key Components:
- ComplexityAssessmentEngine: Analyzes multimodal query complexity
- RoutingDecisionEngine: Makes intelligent routing decisions
- HybridPipeline: Unified reranking system
- ABTestingFramework: Performance testing and optimization

Based on:
- QMMR-02 task specification
- Quantum research insights for complexity assessment
- Industry-standard evaluation framework
"""

from .complexity_assessment_engine import (
    ComplexityAssessmentEngine,
    MultimodalComplexityAssessor,
    NoiseIndicatorAssessor,
    UncertaintyAssessor,
    MedicalDomainComplexityAssessor,
    ComplexityAssessmentResult
)

from .routing_decision_engine import (
    RoutingDecisionEngine,
    RoutingDecision,
    RoutingMethod,
    RoutingConfig
)

from .hybrid_pipeline import (
    HybridQuantumClassicalPipeline,
    HybridRerankingResult,
    HybridPipelineConfig
)

from .ab_testing_framework import (
    ABTestingFramework,
    ABTest,
    ABTestConfig,
    ABTestAnalysis
)

from .complexity_metrics import (
    ComplexityMetrics,
    ComplexityAssessmentConfig
)

__all__ = [
    'ComplexityAssessmentEngine',
    'MultimodalComplexityAssessor',
    'NoiseIndicatorAssessor', 
    'UncertaintyAssessor',
    'MedicalDomainComplexityAssessor',
    'ComplexityAssessmentResult',
    'RoutingDecisionEngine',
    'RoutingDecision',
    'RoutingMethod',
    'RoutingConfig',
    'HybridQuantumClassicalPipeline',
    'HybridRerankingResult',
    'HybridPipelineConfig',
    'ABTestingFramework',
    'ABTest',
    'ABTestConfig',
    'ABTestAnalysis',
    'ComplexityMetrics',
    'ComplexityAssessmentConfig'
]
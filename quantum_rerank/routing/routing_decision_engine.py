"""
Routing Decision Engine for Quantum-Classical Pipeline.

This module implements intelligent routing decisions between classical and quantum rerankers
based on complexity assessment, performance constraints, and domain-specific rules.

Based on:
- QMMR-02 task specification
- Quantum research insights for context-aware routing
- Performance requirements and A/B testing support
"""

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque
from enum import Enum

from .complexity_metrics import ComplexityAssessmentResult, ComplexityMetrics
from ..config.routing_config import RoutingConfig, RoutingMethod, MedicalDomainConfig
from ..evaluation.medical_relevance import MedicalRelevanceJudgments

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Result of routing decision with detailed reasoning."""
    
    # Primary routing decision
    method: RoutingMethod
    confidence: float
    
    # Complexity analysis
    complexity_score: float
    complexity_factors: Dict[str, float]
    
    # Performance considerations
    estimated_latency_ms: float
    estimated_memory_mb: float
    
    # Reasoning and metadata
    reasoning: str
    decision_time_ms: float
    
    # Domain-specific factors
    medical_domain: Optional[str] = None
    domain_confidence: float = 0.0
    
    # Quantum advantage estimation
    quantum_advantage_score: float = 0.0
    quantum_advantage_factors: Dict[str, float] = None
    
    # Fallback information
    fallback_method: Optional[RoutingMethod] = None
    fallback_reason: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.quantum_advantage_factors is None:
            self.quantum_advantage_factors = {}


class AdaptiveThresholdManager:
    """Manages adaptive threshold adjustment based on performance feedback."""
    
    def __init__(self, config: RoutingConfig):
        self.config = config
        
        # Current thresholds
        self.quantum_threshold = config.quantum_threshold
        self.classical_threshold = config.classical_threshold
        
        # Performance tracking
        self.recent_decisions = deque(maxlen=config.adaptation_window_size)
        self.performance_history = deque(maxlen=config.adaptation_window_size)
        
        # Adaptation parameters
        self.learning_rate = config.adaptation_learning_rate
        self.adaptation_enabled = config.enable_adaptive_routing
        
        logger.debug("AdaptiveThresholdManager initialized")
    
    def get_current_thresholds(self) -> Tuple[float, float]:
        """Get current quantum and classical thresholds."""
        return self.quantum_threshold, self.classical_threshold
    
    def record_decision(self, complexity_score: float, method: RoutingMethod, 
                       performance_score: float):
        """Record routing decision and performance for adaptation."""
        decision_record = {
            'complexity_score': complexity_score,
            'method': method,
            'performance_score': performance_score,
            'timestamp': time.time()
        }
        
        self.recent_decisions.append(decision_record)
        self.performance_history.append(performance_score)
        
        # Trigger adaptation if enabled
        if self.adaptation_enabled and len(self.recent_decisions) >= 20:
            self._adapt_thresholds()
    
    def _adapt_thresholds(self):
        """Adapt thresholds based on recent performance."""
        if len(self.recent_decisions) < 20:
            return
        
        # Analyze performance by routing method
        quantum_performance = []
        classical_performance = []
        
        for decision in self.recent_decisions:
            if decision['method'] == RoutingMethod.QUANTUM:
                quantum_performance.append(decision['performance_score'])
            elif decision['method'] == RoutingMethod.CLASSICAL:
                classical_performance.append(decision['performance_score'])
        
        # Calculate average performance
        if quantum_performance and classical_performance:
            avg_quantum_perf = np.mean(quantum_performance)
            avg_classical_perf = np.mean(classical_performance)
            
            # Adjust thresholds based on relative performance
            if avg_quantum_perf > avg_classical_perf:
                # Quantum performing better, lower threshold
                adjustment = self.learning_rate * (avg_quantum_perf - avg_classical_perf)
                self.quantum_threshold = max(0.1, self.quantum_threshold - adjustment)
            else:
                # Classical performing better, raise threshold
                adjustment = self.learning_rate * (avg_classical_perf - avg_quantum_perf)
                self.quantum_threshold = min(0.9, self.quantum_threshold + adjustment)
        
        logger.debug(f"Adapted thresholds: quantum={self.quantum_threshold:.3f}, classical={self.classical_threshold:.3f}")
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get statistics about threshold adaptation."""
        if not self.recent_decisions:
            return {'adaptation_enabled': self.adaptation_enabled, 'decisions_count': 0}
        
        quantum_decisions = sum(1 for d in self.recent_decisions if d['method'] == RoutingMethod.QUANTUM)
        classical_decisions = sum(1 for d in self.recent_decisions if d['method'] == RoutingMethod.CLASSICAL)
        
        return {
            'adaptation_enabled': self.adaptation_enabled,
            'decisions_count': len(self.recent_decisions),
            'quantum_decisions': quantum_decisions,
            'classical_decisions': classical_decisions,
            'current_quantum_threshold': self.quantum_threshold,
            'current_classical_threshold': self.classical_threshold,
            'avg_performance': np.mean(self.performance_history) if self.performance_history else 0.0
        }


class DomainSpecificRuleEngine:
    """Applies domain-specific routing rules for medical queries."""
    
    def __init__(self, config: RoutingConfig, medical_config: MedicalDomainConfig = None):
        self.config = config
        self.medical_config = medical_config or MedicalDomainConfig()
        self.medical_relevance = MedicalRelevanceJudgments()
        
        # Domain-specific thresholds
        self.domain_thresholds = self.medical_config.domain_complexity_thresholds
        
        logger.debug("DomainSpecificRuleEngine initialized")
    
    def apply_domain_rules(self, complexity_result: ComplexityAssessmentResult, 
                          base_method: RoutingMethod) -> Tuple[RoutingMethod, str]:
        """Apply domain-specific routing rules."""
        if not self.config.enable_domain_specific_rules:
            return base_method, "domain_rules_disabled"
        
        # Classify medical domain
        domain = self._classify_domain(complexity_result)
        domain_confidence = complexity_result.query_complexity.domain_specificity
        
        # Apply emergency medicine rule
        if (self.config.emergency_medicine_quantum_preference and 
            domain == 'emergency_medicine' and domain_confidence > 0.7):
            return RoutingMethod.QUANTUM, "emergency_medicine_quantum_preference"
        
        # Apply complex diagnosis rule
        if (self.config.complex_diagnosis_quantum_preference and 
            complexity_result.query_complexity.diagnostic_uncertainty > 0.7):
            return RoutingMethod.QUANTUM, "complex_diagnosis_quantum_preference"
        
        # Apply simple query rule
        if (self.config.simple_query_classical_preference and 
            complexity_result.overall_complexity.overall_complexity < 0.3 and
            complexity_result.query_complexity.modality_count == 1):
            return RoutingMethod.CLASSICAL, "simple_query_classical_preference"
        
        # Apply domain-specific thresholds
        if domain in self.domain_thresholds:
            domain_threshold = self.domain_thresholds[domain]
            if complexity_result.overall_complexity.overall_complexity >= domain_threshold:
                return RoutingMethod.QUANTUM, f"domain_threshold_{domain}"
        
        # Check quantum-preferred specialties
        if domain in self.medical_config.quantum_preferred_specialties:
            return RoutingMethod.QUANTUM, f"quantum_preferred_specialty_{domain}"
        
        # Check classical-preferred specialties
        if domain in self.medical_config.classical_preferred_specialties:
            return RoutingMethod.CLASSICAL, f"classical_preferred_specialty_{domain}"
        
        return base_method, "no_domain_rule_applied"
    
    def _classify_domain(self, complexity_result: ComplexityAssessmentResult) -> str:
        """Classify medical domain from complexity result."""
        # Use medical relevance system if available
        if hasattr(complexity_result, 'medical_domain') and complexity_result.medical_domain:
            return complexity_result.medical_domain
        
        # Fallback to general classification
        complexity_score = complexity_result.overall_complexity.overall_complexity
        
        if complexity_score > 0.8:
            return 'emergency_medicine'
        elif complexity_score > 0.6:
            return 'complex_diagnosis'
        else:
            return 'general_medicine'


class PerformanceBasedRouter:
    """Routes based on performance constraints and resource availability."""
    
    def __init__(self, config: RoutingConfig):
        self.config = config
        
        # Performance tracking
        self.latency_history = deque(maxlen=100)
        self.memory_history = deque(maxlen=100)
        
        # Resource monitoring
        self.current_system_load = 0.0
        self.available_memory_mb = 2048.0
        
        logger.debug("PerformanceBasedRouter initialized")
    
    def check_performance_constraints(self, complexity_result: ComplexityAssessmentResult,
                                    proposed_method: RoutingMethod) -> Tuple[bool, str]:
        """Check if proposed method meets performance constraints."""
        if not self.config.enable_performance_routing:
            return True, "performance_routing_disabled"
        
        # Estimate resource requirements
        estimated_latency = self._estimate_latency(complexity_result, proposed_method)
        estimated_memory = self._estimate_memory(complexity_result, proposed_method)
        
        # Check latency constraint
        if estimated_latency > self.config.latency_threshold_ms:
            return False, f"latency_exceeded_{estimated_latency:.1f}ms"
        
        # Check memory constraint
        if estimated_memory > self.config.memory_threshold_mb:
            return False, f"memory_exceeded_{estimated_memory:.1f}mb"
        
        # Check system load
        if self.current_system_load > 0.8 and proposed_method == RoutingMethod.QUANTUM:
            return False, "system_load_high"
        
        return True, "performance_constraints_met"
    
    def _estimate_latency(self, complexity_result: ComplexityAssessmentResult,
                         method: RoutingMethod) -> float:
        """Estimate latency for routing method."""
        base_latency = {
            RoutingMethod.CLASSICAL: 150.0,
            RoutingMethod.QUANTUM: 250.0,
            RoutingMethod.HYBRID: 300.0
        }
        
        # Adjust based on complexity
        complexity_factor = complexity_result.overall_complexity.overall_complexity
        latency_adjustment = complexity_factor * 100.0
        
        return base_latency[method] + latency_adjustment
    
    def _estimate_memory(self, complexity_result: ComplexityAssessmentResult,
                        method: RoutingMethod) -> float:
        """Estimate memory usage for routing method."""
        base_memory = {
            RoutingMethod.CLASSICAL: 512.0,
            RoutingMethod.QUANTUM: 1024.0,
            RoutingMethod.HYBRID: 1536.0
        }
        
        # Adjust based on query complexity
        modality_factor = complexity_result.query_complexity.modality_count
        memory_adjustment = modality_factor * 200.0
        
        return base_memory[method] + memory_adjustment
    
    def update_performance_stats(self, method: RoutingMethod, actual_latency: float,
                               actual_memory: float):
        """Update performance statistics."""
        self.latency_history.append(actual_latency)
        self.memory_history.append(actual_memory)
        
        # Update system load estimate
        if self.latency_history:
            avg_latency = np.mean(self.latency_history)
            self.current_system_load = min(avg_latency / 500.0, 1.0)  # Normalize to [0,1]


class QuantumAdvantageEstimator:
    """Estimates quantum advantage for routing decisions."""
    
    def __init__(self, config: RoutingConfig):
        self.config = config
        
        # Quantum advantage factors
        self.advantage_factors = {
            'entanglement_potential': 0.3,
            'superposition_benefit': 0.25,
            'interference_complexity': 0.25,
            'noise_resilience': 0.2
        }
        
        logger.debug("QuantumAdvantageEstimator initialized")
    
    def estimate_quantum_advantage(self, complexity_result: ComplexityAssessmentResult) -> Tuple[float, Dict[str, float]]:
        """Estimate quantum advantage score and factor breakdown."""
        complexity = complexity_result.overall_complexity
        
        # Calculate individual advantage factors
        factors = {}
        
        # Entanglement potential (multimodal dependencies)
        factors['entanglement_potential'] = complexity.cross_modal_dependencies
        
        # Superposition benefit (uncertainty handling)
        factors['superposition_benefit'] = (
            complexity.diagnostic_uncertainty * 0.4 +
            complexity.term_ambiguity_score * 0.3 +
            complexity.conflicting_information * 0.3
        )
        
        # Interference complexity (semantic depth)
        factors['interference_complexity'] = (
            complexity.semantic_depth * 0.5 +
            complexity.clinical_correlation_complexity * 0.5
        )
        
        # Noise resilience (error handling)
        factors['noise_resilience'] = (
            complexity.ocr_error_probability * 0.3 +
            complexity.abbreviation_density * 0.3 +
            complexity.missing_data_ratio * 0.4
        )
        
        # Calculate weighted advantage score
        advantage_score = sum(
            self.advantage_factors[factor] * value
            for factor, value in factors.items()
        )
        
        return advantage_score, factors
    
    def should_use_quantum(self, advantage_score: float, complexity_score: float) -> bool:
        """Determine if quantum method should be used based on advantage."""
        # Quantum advantage threshold
        min_advantage = 0.3
        
        # Use quantum if advantage is high or complexity is very high
        return (advantage_score >= min_advantage or 
                complexity_score >= 0.8)


class RoutingDecisionEngine:
    """
    Intelligent routing decision engine for quantum-classical pipeline.
    
    Makes routing decisions based on complexity assessment, performance constraints,
    domain-specific rules, and quantum advantage estimation.
    """
    
    def __init__(self, config: RoutingConfig = None):
        self.config = config or RoutingConfig()
        
        # Initialize components
        self.adaptive_manager = AdaptiveThresholdManager(self.config)
        self.domain_rules = DomainSpecificRuleEngine(self.config)
        self.performance_router = PerformanceBasedRouter(self.config)
        self.quantum_estimator = QuantumAdvantageEstimator(self.config)
        
        # Routing statistics
        self.routing_stats = {
            'total_routings': 0,
            'quantum_routings': 0,
            'classical_routings': 0,
            'hybrid_routings': 0,
            'avg_decision_time_ms': 0.0,
            'routing_accuracy': 0.0
        }
        
        logger.info("RoutingDecisionEngine initialized")
    
    def route_query(self, complexity_result: ComplexityAssessmentResult) -> RoutingDecision:
        """
        Make routing decision based on complexity assessment.
        
        Args:
            complexity_result: Result of complexity assessment
            
        Returns:
            RoutingDecision with method and detailed reasoning
        """
        start_time = time.time()
        
        try:
            # Get current adaptive thresholds
            quantum_threshold, classical_threshold = self.adaptive_manager.get_current_thresholds()
            
            # Base routing decision
            base_method, base_confidence = self._make_base_routing_decision(
                complexity_result, quantum_threshold, classical_threshold
            )
            
            # Apply domain-specific rules
            domain_method, domain_reason = self.domain_rules.apply_domain_rules(
                complexity_result, base_method
            )
            
            # Check performance constraints
            performance_ok, performance_reason = self.performance_router.check_performance_constraints(
                complexity_result, domain_method
            )
            
            # Apply fallback if performance constraints not met
            final_method = domain_method if performance_ok else self.config.fallback_method
            fallback_reason = None if performance_ok else performance_reason
            
            # Estimate quantum advantage
            quantum_advantage, advantage_factors = self.quantum_estimator.estimate_quantum_advantage(
                complexity_result
            )
            
            # Estimate performance metrics
            estimated_latency = self.performance_router._estimate_latency(
                complexity_result, final_method
            )
            estimated_memory = self.performance_router._estimate_memory(
                complexity_result, final_method
            )
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                complexity_result, base_method, domain_method, final_method,
                domain_reason, performance_reason, quantum_advantage
            )
            
            # Create decision
            decision_time = (time.time() - start_time) * 1000
            
            decision = RoutingDecision(
                method=final_method,
                confidence=base_confidence,
                complexity_score=complexity_result.overall_complexity.overall_complexity,
                complexity_factors=complexity_result.complexity_breakdown,
                estimated_latency_ms=estimated_latency,
                estimated_memory_mb=estimated_memory,
                reasoning=reasoning,
                decision_time_ms=decision_time,
                quantum_advantage_score=quantum_advantage,
                quantum_advantage_factors=advantage_factors,
                fallback_method=self.config.fallback_method if not performance_ok else None,
                fallback_reason=fallback_reason
            )
            
            # Update statistics
            self._update_routing_stats(final_method, decision_time)
            
            return decision
            
        except Exception as e:
            decision_time = (time.time() - start_time) * 1000
            logger.error(f"Routing decision failed: {e}")
            
            # Return fallback decision
            return RoutingDecision(
                method=self.config.fallback_method,
                confidence=0.5,
                complexity_score=0.5,
                complexity_factors={},
                estimated_latency_ms=200.0,
                estimated_memory_mb=512.0,
                reasoning=f"Error in routing decision: {str(e)}",
                decision_time_ms=decision_time,
                fallback_reason=str(e)
            )
    
    def _make_base_routing_decision(self, complexity_result: ComplexityAssessmentResult,
                                   quantum_threshold: float, classical_threshold: float) -> Tuple[RoutingMethod, float]:
        """Make base routing decision based on complexity thresholds."""
        overall_complexity = complexity_result.overall_complexity.overall_complexity
        
        if overall_complexity >= quantum_threshold:
            confidence = min(overall_complexity, 1.0)
            return RoutingMethod.QUANTUM, confidence
        elif overall_complexity <= classical_threshold:
            confidence = 1.0 - overall_complexity
            return RoutingMethod.CLASSICAL, confidence
        else:
            # Hybrid approach for medium complexity
            confidence = 0.5
            return RoutingMethod.HYBRID, confidence
    
    def _generate_reasoning(self, complexity_result: ComplexityAssessmentResult,
                          base_method: RoutingMethod, domain_method: RoutingMethod,
                          final_method: RoutingMethod, domain_reason: str,
                          performance_reason: str, quantum_advantage: float) -> str:
        """Generate human-readable reasoning for routing decision."""
        reasoning_parts = []
        
        # Base complexity reasoning
        complexity_score = complexity_result.overall_complexity.overall_complexity
        reasoning_parts.append(f"Complexity score: {complexity_score:.3f}")
        
        # Base method reasoning
        if base_method == RoutingMethod.QUANTUM:
            reasoning_parts.append("High complexity suggests quantum reranking")
        elif base_method == RoutingMethod.CLASSICAL:
            reasoning_parts.append("Low complexity suggests classical reranking")
        else:
            reasoning_parts.append("Medium complexity suggests hybrid approach")
        
        # Domain-specific reasoning
        if domain_method != base_method:
            reasoning_parts.append(f"Domain rule applied: {domain_reason}")
        
        # Performance reasoning
        if final_method != domain_method:
            reasoning_parts.append(f"Performance constraint: {performance_reason}")
        
        # Quantum advantage reasoning
        if quantum_advantage > 0.3:
            reasoning_parts.append(f"High quantum advantage: {quantum_advantage:.3f}")
        
        return "; ".join(reasoning_parts)
    
    def _update_routing_stats(self, method: RoutingMethod, decision_time: float):
        """Update routing statistics."""
        self.routing_stats['total_routings'] += 1
        
        if method == RoutingMethod.QUANTUM:
            self.routing_stats['quantum_routings'] += 1
        elif method == RoutingMethod.CLASSICAL:
            self.routing_stats['classical_routings'] += 1
        else:
            self.routing_stats['hybrid_routings'] += 1
        
        # Update average decision time
        n = self.routing_stats['total_routings']
        current_avg = self.routing_stats['avg_decision_time_ms']
        self.routing_stats['avg_decision_time_ms'] = (
            (current_avg * (n - 1) + decision_time) / n
        )
    
    def record_routing_performance(self, decision: RoutingDecision, 
                                 actual_latency: float, actual_memory: float,
                                 accuracy_score: float):
        """Record actual performance for adaptive learning."""
        # Update performance router
        self.performance_router.update_performance_stats(
            decision.method, actual_latency, actual_memory
        )
        
        # Update adaptive thresholds
        self.adaptive_manager.record_decision(
            decision.complexity_score, decision.method, accuracy_score
        )
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""
        stats = self.routing_stats.copy()
        
        # Add distribution percentages
        total = stats['total_routings']
        if total > 0:
            stats['quantum_percentage'] = stats['quantum_routings'] / total * 100
            stats['classical_percentage'] = stats['classical_routings'] / total * 100
            stats['hybrid_percentage'] = stats['hybrid_routings'] / total * 100
        
        # Add adaptive threshold stats
        stats['adaptive_stats'] = self.adaptive_manager.get_adaptation_stats()
        
        return stats
    
    def optimize_for_performance(self):
        """Optimize routing engine for performance."""
        # Clear performance history
        self.adaptive_manager.recent_decisions.clear()
        self.adaptive_manager.performance_history.clear()
        
        # Reset statistics
        self.routing_stats = {
            'total_routings': 0,
            'quantum_routings': 0,
            'classical_routings': 0,
            'hybrid_routings': 0,
            'avg_decision_time_ms': 0.0,
            'routing_accuracy': 0.0
        }
        
        logger.info("RoutingDecisionEngine optimized for performance")
    
    def update_configuration(self, new_config: RoutingConfig):
        """Update routing configuration."""
        self.config = new_config
        
        # Update components
        self.adaptive_manager.config = new_config
        self.domain_rules.config = new_config
        self.performance_router.config = new_config
        
        logger.info("RoutingDecisionEngine configuration updated")
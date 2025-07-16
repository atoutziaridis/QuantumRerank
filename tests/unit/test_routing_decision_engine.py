"""
Unit tests for RoutingDecisionEngine.

Tests intelligent routing decisions, adaptive thresholds, 
domain-specific rules, and performance-based routing.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from quantum_rerank.routing.routing_decision_engine import (
    RoutingDecisionEngine,
    AdaptiveThresholdManager,
    DomainSpecificRuleEngine,
    PerformanceBasedRouter,
    QuantumAdvantageEstimator,
    RoutingDecision,
    RoutingMethod
)
from quantum_rerank.routing.complexity_metrics import (
    ComplexityMetrics,
    ComplexityAssessmentResult
)
from quantum_rerank.config.routing_config import RoutingConfig, MedicalDomainConfig


class TestAdaptiveThresholdManager:
    """Test adaptive threshold management."""
    
    def test_adaptive_manager_initialization(self):
        """Test adaptive threshold manager initialization."""
        config = RoutingConfig(quantum_threshold=0.7, classical_threshold=0.3)
        manager = AdaptiveThresholdManager(config)
        
        assert manager.quantum_threshold == 0.7
        assert manager.classical_threshold == 0.3
        assert manager.adaptation_enabled is True
        assert len(manager.recent_decisions) == 0
    
    def test_get_current_thresholds(self):
        """Test getting current thresholds."""
        config = RoutingConfig(quantum_threshold=0.6, classical_threshold=0.4)
        manager = AdaptiveThresholdManager(config)
        
        quantum_threshold, classical_threshold = manager.get_current_thresholds()
        
        assert quantum_threshold == 0.6
        assert classical_threshold == 0.4
    
    def test_record_decision(self):
        """Test recording routing decisions."""
        config = RoutingConfig()
        manager = AdaptiveThresholdManager(config)
        
        # Record some decisions
        manager.record_decision(0.8, RoutingMethod.QUANTUM, 0.9)
        manager.record_decision(0.2, RoutingMethod.CLASSICAL, 0.85)
        
        assert len(manager.recent_decisions) == 2
        assert len(manager.performance_history) == 2
    
    def test_threshold_adaptation_quantum_better(self):
        """Test threshold adaptation when quantum performs better."""
        config = RoutingConfig(adaptation_learning_rate=0.1)
        manager = AdaptiveThresholdManager(config)
        
        original_threshold = manager.quantum_threshold
        
        # Record quantum performing better
        for i in range(25):
            if i % 2 == 0:
                manager.record_decision(0.8, RoutingMethod.QUANTUM, 0.9)
            else:
                manager.record_decision(0.2, RoutingMethod.CLASSICAL, 0.7)
        
        # Threshold should be lowered (quantum gets used more)
        assert manager.quantum_threshold <= original_threshold
    
    def test_threshold_adaptation_classical_better(self):
        """Test threshold adaptation when classical performs better."""
        config = RoutingConfig(adaptation_learning_rate=0.1)
        manager = AdaptiveThresholdManager(config)
        
        original_threshold = manager.quantum_threshold
        
        # Record classical performing better
        for i in range(25):
            if i % 2 == 0:
                manager.record_decision(0.8, RoutingMethod.QUANTUM, 0.7)
            else:
                manager.record_decision(0.2, RoutingMethod.CLASSICAL, 0.9)
        
        # Threshold should be raised (quantum gets used less)
        assert manager.quantum_threshold >= original_threshold
    
    def test_adaptation_disabled(self):
        """Test behavior when adaptation is disabled."""
        config = RoutingConfig(enable_adaptive_routing=False)
        manager = AdaptiveThresholdManager(config)
        
        original_threshold = manager.quantum_threshold
        
        # Record decisions
        for i in range(25):
            manager.record_decision(0.8, RoutingMethod.QUANTUM, 0.9)
        
        # Threshold should not change
        assert manager.quantum_threshold == original_threshold
    
    def test_get_adaptation_stats(self):
        """Test getting adaptation statistics."""
        config = RoutingConfig()
        manager = AdaptiveThresholdManager(config)
        
        # Record some decisions
        manager.record_decision(0.8, RoutingMethod.QUANTUM, 0.9)
        manager.record_decision(0.2, RoutingMethod.CLASSICAL, 0.8)
        
        stats = manager.get_adaptation_stats()
        
        assert 'adaptation_enabled' in stats
        assert 'decisions_count' in stats
        assert 'quantum_decisions' in stats
        assert 'classical_decisions' in stats
        assert stats['decisions_count'] == 2
        assert stats['quantum_decisions'] == 1
        assert stats['classical_decisions'] == 1


class TestDomainSpecificRuleEngine:
    """Test domain-specific routing rules."""
    
    def test_domain_rules_initialization(self):
        """Test domain rules engine initialization."""
        config = RoutingConfig()
        engine = DomainSpecificRuleEngine(config)
        
        assert engine.config is not None
        assert engine.medical_config is not None
        assert engine.domain_thresholds is not None
    
    def test_apply_emergency_medicine_rule(self):
        """Test emergency medicine routing rule."""
        config = RoutingConfig(emergency_medicine_quantum_preference=True)
        engine = DomainSpecificRuleEngine(config)
        
        # Mock complexity result for emergency medicine
        complexity_result = Mock()
        complexity_result.query_complexity = Mock()
        complexity_result.query_complexity.domain_specificity = 0.8
        complexity_result.overall_complexity = Mock()
        complexity_result.overall_complexity.overall_complexity = 0.5
        
        # Mock domain classification
        with patch.object(engine, '_classify_domain', return_value='emergency_medicine'):
            method, reason = engine.apply_domain_rules(complexity_result, RoutingMethod.CLASSICAL)
            
            assert method == RoutingMethod.QUANTUM
            assert 'emergency_medicine' in reason
    
    def test_apply_complex_diagnosis_rule(self):
        """Test complex diagnosis routing rule."""
        config = RoutingConfig(complex_diagnosis_quantum_preference=True)
        engine = DomainSpecificRuleEngine(config)
        
        # Mock complexity result with high diagnostic uncertainty
        complexity_result = Mock()
        complexity_result.query_complexity = Mock()
        complexity_result.query_complexity.diagnostic_uncertainty = 0.8
        complexity_result.overall_complexity = Mock()
        complexity_result.overall_complexity.overall_complexity = 0.5
        
        method, reason = engine.apply_domain_rules(complexity_result, RoutingMethod.CLASSICAL)
        
        assert method == RoutingMethod.QUANTUM
        assert 'complex_diagnosis' in reason
    
    def test_apply_simple_query_rule(self):
        """Test simple query routing rule."""
        config = RoutingConfig(simple_query_classical_preference=True)
        engine = DomainSpecificRuleEngine(config)
        
        # Mock complexity result for simple query
        complexity_result = Mock()
        complexity_result.query_complexity = Mock()
        complexity_result.query_complexity.modality_count = 1
        complexity_result.overall_complexity = Mock()
        complexity_result.overall_complexity.overall_complexity = 0.2
        
        method, reason = engine.apply_domain_rules(complexity_result, RoutingMethod.QUANTUM)
        
        assert method == RoutingMethod.CLASSICAL
        assert 'simple_query' in reason
    
    def test_domain_rules_disabled(self):
        """Test behavior when domain rules are disabled."""
        config = RoutingConfig(enable_domain_specific_rules=False)
        engine = DomainSpecificRuleEngine(config)
        
        complexity_result = Mock()
        base_method = RoutingMethod.QUANTUM
        
        method, reason = engine.apply_domain_rules(complexity_result, base_method)
        
        assert method == base_method
        assert reason == "domain_rules_disabled"
    
    def test_classify_domain(self):
        """Test domain classification."""
        config = RoutingConfig()
        engine = DomainSpecificRuleEngine(config)
        
        # High complexity -> emergency medicine
        complexity_result = Mock()
        complexity_result.overall_complexity = Mock()
        complexity_result.overall_complexity.overall_complexity = 0.9
        
        domain = engine._classify_domain(complexity_result)
        assert domain == 'emergency_medicine'
        
        # Medium complexity -> complex diagnosis
        complexity_result.overall_complexity.overall_complexity = 0.7
        domain = engine._classify_domain(complexity_result)
        assert domain == 'complex_diagnosis'
        
        # Low complexity -> general medicine
        complexity_result.overall_complexity.overall_complexity = 0.3
        domain = engine._classify_domain(complexity_result)
        assert domain == 'general_medicine'


class TestPerformanceBasedRouter:
    """Test performance-based routing."""
    
    def test_performance_router_initialization(self):
        """Test performance router initialization."""
        config = RoutingConfig()
        router = PerformanceBasedRouter(config)
        
        assert router.config is not None
        assert len(router.latency_history) == 0
        assert len(router.memory_history) == 0
    
    def test_check_performance_constraints_pass(self):
        """Test performance constraints check that passes."""
        config = RoutingConfig(latency_threshold_ms=200.0, memory_threshold_mb=1000.0)
        router = PerformanceBasedRouter(config)
        
        # Mock complexity result
        complexity_result = Mock()
        complexity_result.overall_complexity = Mock()
        complexity_result.overall_complexity.overall_complexity = 0.5
        complexity_result.query_complexity = Mock()
        complexity_result.query_complexity.modality_count = 1
        
        passes, reason = router.check_performance_constraints(complexity_result, RoutingMethod.CLASSICAL)
        
        assert passes is True
        assert reason == "performance_constraints_met"
    
    def test_check_performance_constraints_fail_latency(self):
        """Test performance constraints check that fails on latency."""
        config = RoutingConfig(latency_threshold_ms=50.0)  # Very low threshold
        router = PerformanceBasedRouter(config)
        
        # Mock high complexity result
        complexity_result = Mock()
        complexity_result.overall_complexity = Mock()
        complexity_result.overall_complexity.overall_complexity = 0.9
        complexity_result.query_complexity = Mock()
        complexity_result.query_complexity.modality_count = 3
        
        passes, reason = router.check_performance_constraints(complexity_result, RoutingMethod.QUANTUM)
        
        assert passes is False
        assert 'latency_exceeded' in reason
    
    def test_check_performance_constraints_fail_memory(self):
        """Test performance constraints check that fails on memory."""
        config = RoutingConfig(memory_threshold_mb=100.0)  # Very low threshold
        router = PerformanceBasedRouter(config)
        
        # Mock high complexity result
        complexity_result = Mock()
        complexity_result.overall_complexity = Mock()
        complexity_result.overall_complexity.overall_complexity = 0.9
        complexity_result.query_complexity = Mock()
        complexity_result.query_complexity.modality_count = 3
        
        passes, reason = router.check_performance_constraints(complexity_result, RoutingMethod.QUANTUM)
        
        assert passes is False
        assert 'memory_exceeded' in reason
    
    def test_check_performance_constraints_high_load(self):
        """Test performance constraints check with high system load."""
        config = RoutingConfig()
        router = PerformanceBasedRouter(config)
        router.current_system_load = 0.9  # High load
        
        complexity_result = Mock()
        complexity_result.overall_complexity = Mock()
        complexity_result.overall_complexity.overall_complexity = 0.5
        complexity_result.query_complexity = Mock()
        complexity_result.query_complexity.modality_count = 1
        
        passes, reason = router.check_performance_constraints(complexity_result, RoutingMethod.QUANTUM)
        
        assert passes is False
        assert reason == "system_load_high"
    
    def test_performance_routing_disabled(self):
        """Test behavior when performance routing is disabled."""
        config = RoutingConfig(enable_performance_routing=False)
        router = PerformanceBasedRouter(config)
        
        complexity_result = Mock()
        passes, reason = router.check_performance_constraints(complexity_result, RoutingMethod.QUANTUM)
        
        assert passes is True
        assert reason == "performance_routing_disabled"
    
    def test_update_performance_stats(self):
        """Test performance statistics update."""
        config = RoutingConfig()
        router = PerformanceBasedRouter(config)
        
        # Update with some performance data
        router.update_performance_stats(RoutingMethod.QUANTUM, 150.0, 800.0)
        router.update_performance_stats(RoutingMethod.CLASSICAL, 100.0, 400.0)
        
        assert len(router.latency_history) == 2
        assert len(router.memory_history) == 2
        assert router.current_system_load > 0.0


class TestQuantumAdvantageEstimator:
    """Test quantum advantage estimation."""
    
    def test_quantum_advantage_estimator_initialization(self):
        """Test quantum advantage estimator initialization."""
        config = RoutingConfig()
        estimator = QuantumAdvantageEstimator(config)
        
        assert estimator.config is not None
        assert estimator.advantage_factors is not None
        assert 'entanglement_potential' in estimator.advantage_factors
    
    def test_estimate_quantum_advantage_low(self):
        """Test quantum advantage estimation with low advantage."""
        config = RoutingConfig()
        estimator = QuantumAdvantageEstimator(config)
        
        # Mock complexity result with low quantum advantage
        complexity_result = Mock()
        complexity_result.overall_complexity = Mock()
        complexity_result.overall_complexity.cross_modal_dependencies = 0.1
        complexity_result.overall_complexity.diagnostic_uncertainty = 0.1
        complexity_result.overall_complexity.term_ambiguity_score = 0.1
        complexity_result.overall_complexity.conflicting_information = 0.1
        complexity_result.overall_complexity.semantic_depth = 0.1
        complexity_result.overall_complexity.clinical_correlation_complexity = 0.1
        complexity_result.overall_complexity.ocr_error_probability = 0.1
        complexity_result.overall_complexity.abbreviation_density = 0.1
        complexity_result.overall_complexity.missing_data_ratio = 0.1
        
        advantage_score, factors = estimator.estimate_quantum_advantage(complexity_result)
        
        assert 0.0 <= advantage_score <= 1.0
        assert advantage_score < 0.5  # Should be low
        assert 'entanglement_potential' in factors
        assert 'superposition_benefit' in factors
    
    def test_estimate_quantum_advantage_high(self):
        """Test quantum advantage estimation with high advantage."""
        config = RoutingConfig()
        estimator = QuantumAdvantageEstimator(config)
        
        # Mock complexity result with high quantum advantage
        complexity_result = Mock()
        complexity_result.overall_complexity = Mock()
        complexity_result.overall_complexity.cross_modal_dependencies = 0.8
        complexity_result.overall_complexity.diagnostic_uncertainty = 0.8
        complexity_result.overall_complexity.term_ambiguity_score = 0.8
        complexity_result.overall_complexity.conflicting_information = 0.8
        complexity_result.overall_complexity.semantic_depth = 0.8
        complexity_result.overall_complexity.clinical_correlation_complexity = 0.8
        complexity_result.overall_complexity.ocr_error_probability = 0.8
        complexity_result.overall_complexity.abbreviation_density = 0.8
        complexity_result.overall_complexity.missing_data_ratio = 0.8
        
        advantage_score, factors = estimator.estimate_quantum_advantage(complexity_result)
        
        assert 0.0 <= advantage_score <= 1.0
        assert advantage_score > 0.5  # Should be high
        assert all(factor >= 0.0 for factor in factors.values())
    
    def test_should_use_quantum_high_advantage(self):
        """Test quantum usage decision with high advantage."""
        config = RoutingConfig()
        estimator = QuantumAdvantageEstimator(config)
        
        should_use = estimator.should_use_quantum(0.5, 0.5)
        assert should_use is True
    
    def test_should_use_quantum_high_complexity(self):
        """Test quantum usage decision with high complexity."""
        config = RoutingConfig()
        estimator = QuantumAdvantageEstimator(config)
        
        should_use = estimator.should_use_quantum(0.2, 0.9)
        assert should_use is True
    
    def test_should_use_quantum_low_both(self):
        """Test quantum usage decision with low advantage and complexity."""
        config = RoutingConfig()
        estimator = QuantumAdvantageEstimator(config)
        
        should_use = estimator.should_use_quantum(0.1, 0.3)
        assert should_use is False


class TestRoutingDecisionEngine:
    """Test the main routing decision engine."""
    
    def test_routing_engine_initialization(self):
        """Test routing engine initialization."""
        engine = RoutingDecisionEngine()
        
        assert engine.config is not None
        assert engine.adaptive_manager is not None
        assert engine.domain_rules is not None
        assert engine.performance_router is not None
        assert engine.quantum_estimator is not None
        assert engine.routing_stats['total_routings'] == 0
    
    def test_route_query_low_complexity(self):
        """Test routing decision for low complexity query."""
        engine = RoutingDecisionEngine()
        
        # Mock low complexity result
        complexity_result = Mock()
        complexity_result.overall_complexity = Mock()
        complexity_result.overall_complexity.overall_complexity = 0.2
        complexity_result.complexity_breakdown = {}
        complexity_result.query_complexity = Mock()
        complexity_result.query_complexity.modality_count = 1
        complexity_result.query_complexity.diagnostic_uncertainty = 0.1
        
        decision = engine.route_query(complexity_result)
        
        assert isinstance(decision, RoutingDecision)
        assert decision.method == RoutingMethod.CLASSICAL
        assert decision.confidence > 0.5
        assert decision.complexity_score == 0.2
        assert decision.decision_time_ms > 0
    
    def test_route_query_high_complexity(self):
        """Test routing decision for high complexity query."""
        engine = RoutingDecisionEngine()
        
        # Mock high complexity result
        complexity_result = Mock()
        complexity_result.overall_complexity = Mock()
        complexity_result.overall_complexity.overall_complexity = 0.8
        complexity_result.complexity_breakdown = {}
        complexity_result.query_complexity = Mock()
        complexity_result.query_complexity.modality_count = 2
        complexity_result.query_complexity.diagnostic_uncertainty = 0.7
        
        decision = engine.route_query(complexity_result)
        
        assert isinstance(decision, RoutingDecision)
        assert decision.method == RoutingMethod.QUANTUM
        assert decision.confidence > 0.5
        assert decision.complexity_score == 0.8
        assert decision.decision_time_ms > 0
    
    def test_route_query_medium_complexity(self):
        """Test routing decision for medium complexity query."""
        engine = RoutingDecisionEngine()
        
        # Mock medium complexity result
        complexity_result = Mock()
        complexity_result.overall_complexity = Mock()
        complexity_result.overall_complexity.overall_complexity = 0.5
        complexity_result.complexity_breakdown = {}
        complexity_result.query_complexity = Mock()
        complexity_result.query_complexity.modality_count = 1
        complexity_result.query_complexity.diagnostic_uncertainty = 0.4
        
        decision = engine.route_query(complexity_result)
        
        assert isinstance(decision, RoutingDecision)
        assert decision.method == RoutingMethod.HYBRID
        assert decision.confidence == 0.5
        assert decision.complexity_score == 0.5
    
    def test_route_query_with_domain_override(self):
        """Test routing decision with domain-specific override."""
        config = RoutingConfig(emergency_medicine_quantum_preference=True)
        engine = RoutingDecisionEngine(config)
        
        # Mock complexity result that would normally go to classical
        complexity_result = Mock()
        complexity_result.overall_complexity = Mock()
        complexity_result.overall_complexity.overall_complexity = 0.3
        complexity_result.complexity_breakdown = {}
        complexity_result.query_complexity = Mock()
        complexity_result.query_complexity.modality_count = 1
        complexity_result.query_complexity.diagnostic_uncertainty = 0.2
        complexity_result.query_complexity.domain_specificity = 0.8
        
        # Mock domain classification
        with patch.object(engine.domain_rules, '_classify_domain', return_value='emergency_medicine'):
            decision = engine.route_query(complexity_result)
            
            assert decision.method == RoutingMethod.QUANTUM
            assert 'emergency_medicine' in decision.reasoning
    
    def test_route_query_with_performance_fallback(self):
        """Test routing decision with performance fallback."""
        config = RoutingConfig(latency_threshold_ms=50.0)  # Very low threshold
        engine = RoutingDecisionEngine(config)
        
        # Mock high complexity result that would normally go to quantum
        complexity_result = Mock()
        complexity_result.overall_complexity = Mock()
        complexity_result.overall_complexity.overall_complexity = 0.8
        complexity_result.complexity_breakdown = {}
        complexity_result.query_complexity = Mock()
        complexity_result.query_complexity.modality_count = 3
        complexity_result.query_complexity.diagnostic_uncertainty = 0.7
        
        decision = engine.route_query(complexity_result)
        
        # Should fallback to classical due to performance constraints
        assert decision.method == RoutingMethod.CLASSICAL
        assert decision.fallback_reason is not None
        assert 'latency_exceeded' in decision.fallback_reason
    
    def test_record_routing_performance(self):
        """Test recording routing performance for adaptive learning."""
        engine = RoutingDecisionEngine()
        
        # Create a mock decision
        decision = RoutingDecision(
            method=RoutingMethod.QUANTUM,
            confidence=0.8,
            complexity_score=0.7,
            complexity_factors={},
            estimated_latency_ms=200.0,
            estimated_memory_mb=1000.0,
            reasoning="test decision",
            decision_time_ms=10.0
        )
        
        # Record performance
        engine.record_routing_performance(decision, 180.0, 900.0, 0.85)
        
        # Check that adaptive manager received the data
        assert len(engine.adaptive_manager.recent_decisions) == 1
        assert len(engine.adaptive_manager.performance_history) == 1
    
    def test_get_routing_stats(self):
        """Test getting routing statistics."""
        engine = RoutingDecisionEngine()
        
        # Mock some routing decisions
        for i in range(5):
            complexity_result = Mock()
            complexity_result.overall_complexity = Mock()
            complexity_result.overall_complexity.overall_complexity = 0.3 + i * 0.2
            complexity_result.complexity_breakdown = {}
            complexity_result.query_complexity = Mock()
            complexity_result.query_complexity.modality_count = 1
            complexity_result.query_complexity.diagnostic_uncertainty = 0.1
            
            engine.route_query(complexity_result)
        
        stats = engine.get_routing_stats()
        
        assert 'total_routings' in stats
        assert 'quantum_routings' in stats
        assert 'classical_routings' in stats
        assert 'hybrid_routings' in stats
        assert 'avg_decision_time_ms' in stats
        assert 'adaptive_stats' in stats
        assert stats['total_routings'] == 5
        assert stats['avg_decision_time_ms'] > 0
    
    def test_optimize_for_performance(self):
        """Test performance optimization."""
        engine = RoutingDecisionEngine()
        
        # Make some routing decisions
        complexity_result = Mock()
        complexity_result.overall_complexity = Mock()
        complexity_result.overall_complexity.overall_complexity = 0.5
        complexity_result.complexity_breakdown = {}
        complexity_result.query_complexity = Mock()
        complexity_result.query_complexity.modality_count = 1
        complexity_result.query_complexity.diagnostic_uncertainty = 0.3
        
        engine.route_query(complexity_result)
        
        # Optimize for performance
        engine.optimize_for_performance()
        
        # Statistics should be reset
        assert engine.routing_stats['total_routings'] == 0
        assert engine.routing_stats['avg_decision_time_ms'] == 0.0
        assert len(engine.adaptive_manager.recent_decisions) == 0
    
    def test_update_configuration(self):
        """Test configuration update."""
        engine = RoutingDecisionEngine()
        
        # Create new configuration
        new_config = RoutingConfig(
            quantum_threshold=0.8,
            classical_threshold=0.2,
            enable_adaptive_routing=False
        )
        
        # Update configuration
        engine.update_configuration(new_config)
        
        assert engine.config.quantum_threshold == 0.8
        assert engine.config.classical_threshold == 0.2
        assert engine.config.enable_adaptive_routing is False
    
    def test_error_handling(self):
        """Test error handling in routing decisions."""
        engine = RoutingDecisionEngine()
        
        # Mock an error in adaptive manager
        with patch.object(engine.adaptive_manager, 'get_current_thresholds', side_effect=Exception("Test error")):
            complexity_result = Mock()
            complexity_result.overall_complexity = Mock()
            complexity_result.overall_complexity.overall_complexity = 0.5
            complexity_result.complexity_breakdown = {}
            
            decision = engine.route_query(complexity_result)
            
            assert decision.method == RoutingMethod.CLASSICAL  # Fallback
            assert decision.fallback_reason == "Test error"
    
    def test_generate_reasoning(self):
        """Test reasoning generation."""
        engine = RoutingDecisionEngine()
        
        # Mock complexity result
        complexity_result = Mock()
        complexity_result.overall_complexity = Mock()
        complexity_result.overall_complexity.overall_complexity = 0.7
        
        reasoning = engine._generate_reasoning(
            complexity_result,
            RoutingMethod.QUANTUM,
            RoutingMethod.QUANTUM,
            RoutingMethod.CLASSICAL,
            "performance_constraint",
            "latency_exceeded",
            0.4
        )
        
        assert isinstance(reasoning, str)
        assert "0.7" in reasoning
        assert "performance_constraint" in reasoning
        assert "latency_exceeded" in reasoning


if __name__ == '__main__':
    pytest.main([__file__])
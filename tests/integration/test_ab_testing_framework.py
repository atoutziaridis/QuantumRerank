"""
Integration tests for A/B Testing Framework.

Tests the complete A/B testing functionality including test creation,
execution, statistical analysis, and routing optimization.
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from quantum_rerank.routing.ab_testing_framework import (
    ABTestingFramework,
    ABTest,
    ABTestResult,
    ABTestMetrics,
    ABTestAnalysis,
    StatisticalAnalyzer,
    ABTestStatus
)
from quantum_rerank.routing.complexity_metrics import ComplexityAssessmentResult, ComplexityMetrics
from quantum_rerank.routing.routing_decision_engine import RoutingMethod, RoutingDecision
from quantum_rerank.config.routing_config import ABTestConfig, ABTestStrategy


class TestABTestingFramework:
    """Integration tests for the A/B testing framework."""
    
    def test_framework_initialization(self):
        """Test framework initialization."""
        framework = ABTestingFramework()
        
        assert len(framework.active_tests) == 0
        assert len(framework.completed_tests) == 0
        assert framework.test_stats['total_tests'] == 0
        assert framework.test_stats['active_tests'] == 0
        assert framework.test_stats['completed_tests'] == 0
    
    def test_create_and_start_test(self):
        """Test creating and starting an A/B test."""
        framework = ABTestingFramework()
        
        config = ABTestConfig(
            test_name="Classical vs Quantum Routing",
            test_description="Compare classical and quantum routing performance",
            control_method=RoutingMethod.CLASSICAL,
            treatment_method=RoutingMethod.QUANTUM,
            sample_size=100,
            strategy=ABTestStrategy.RANDOM
        )
        
        # Create test
        test = framework.create_routing_test(config)
        
        assert isinstance(test, ABTest)
        assert test.config.test_name == "Classical vs Quantum Routing"
        assert test.status == ABTestStatus.PLANNING
        assert framework.test_stats['total_tests'] == 1
        
        # Start test
        success = framework.start_test(test)
        
        assert success is True
        assert test.status == ABTestStatus.ACTIVE
        assert test.test_id in framework.active_tests
        assert framework.test_stats['active_tests'] == 1
    
    def test_routing_decision_from_active_test(self):
        """Test getting routing decisions from active A/B tests."""
        framework = ABTestingFramework()
        
        config = ABTestConfig(
            test_name="Test Routing",
            control_method=RoutingMethod.CLASSICAL,
            treatment_method=RoutingMethod.QUANTUM,
            sample_size=50,
            strategy=ABTestStrategy.RANDOM
        )
        
        test = framework.create_routing_test(config)
        framework.start_test(test)
        
        # Create mock complexity result
        complexity_result = Mock()
        complexity_result.overall_complexity = Mock()
        complexity_result.overall_complexity.overall_complexity = 0.5
        complexity_result.query_complexity = Mock()
        complexity_result.query_complexity.modality_count = 1
        
        # Get routing decision
        decision = framework.get_routing_decision(complexity_result)
        
        # Should get a decision from the active test
        assert decision is not None
        assert isinstance(decision, RoutingDecision)
        assert decision.method in [RoutingMethod.CLASSICAL, RoutingMethod.QUANTUM]
        assert "A/B test" in decision.reasoning
    
    def test_complexity_based_test_filtering(self):
        """Test complexity-based test filtering."""
        framework = ABTestingFramework()
        
        config = ABTestConfig(
            test_name="Complexity Based Test",
            control_method=RoutingMethod.CLASSICAL,
            treatment_method=RoutingMethod.QUANTUM,
            sample_size=50,
            strategy=ABTestStrategy.COMPLEXITY_BASED,
            complexity_bands=[(0.4, 0.8)]  # Only apply to medium complexity
        )
        
        test = framework.create_routing_test(config)
        framework.start_test(test)
        
        # Test with low complexity (should not apply)
        low_complexity = Mock()
        low_complexity.overall_complexity = Mock()
        low_complexity.overall_complexity.overall_complexity = 0.2
        
        decision_low = framework.get_routing_decision(low_complexity)
        assert decision_low is None  # Should not apply
        
        # Test with medium complexity (should apply)
        medium_complexity = Mock()
        medium_complexity.overall_complexity = Mock()
        medium_complexity.overall_complexity.overall_complexity = 0.6
        medium_complexity.query_complexity = Mock()
        medium_complexity.query_complexity.modality_count = 1
        
        decision_medium = framework.get_routing_decision(medium_complexity)
        assert decision_medium is not None  # Should apply
    
    def test_performance_based_test_filtering(self):
        """Test performance-based test filtering."""
        framework = ABTestingFramework()
        
        config = ABTestConfig(
            test_name="Performance Based Test",
            control_method=RoutingMethod.CLASSICAL,
            treatment_method=RoutingMethod.QUANTUM,
            sample_size=50,
            strategy=ABTestStrategy.PERFORMANCE_BASED
        )
        
        test = framework.create_routing_test(config)
        framework.start_test(test)
        
        # Test with single modality, low complexity (should not apply)
        simple_complexity = Mock()
        simple_complexity.overall_complexity = Mock()
        simple_complexity.overall_complexity.overall_complexity = 0.3
        simple_complexity.query_complexity = Mock()
        simple_complexity.query_complexity.modality_count = 1
        
        decision_simple = framework.get_routing_decision(simple_complexity)
        assert decision_simple is None
        
        # Test with multimodal, high complexity (should apply)
        complex_complexity = Mock()
        complex_complexity.overall_complexity = Mock()
        complex_complexity.overall_complexity.overall_complexity = 0.7
        complex_complexity.query_complexity = Mock()
        complex_complexity.query_complexity.modality_count = 3
        
        decision_complex = framework.get_routing_decision(complex_complexity)
        assert decision_complex is not None
    
    def test_record_test_outcomes(self):
        """Test recording test outcomes."""
        framework = ABTestingFramework()
        
        config = ABTestConfig(
            test_name="Outcome Recording Test",
            control_method=RoutingMethod.CLASSICAL,
            treatment_method=RoutingMethod.QUANTUM,
            sample_size=10
        )
        
        test = framework.create_routing_test(config)
        framework.start_test(test)
        
        # Simulate test decisions and outcomes
        for i in range(5):
            complexity_result = Mock()
            complexity_result.overall_complexity = Mock()
            complexity_result.overall_complexity.overall_complexity = 0.5
            complexity_result.query_complexity = Mock()
            complexity_result.query_complexity.modality_count = 1
            
            decision = framework.get_routing_decision(complexity_result)
            if decision:
                # Record the decision
                test.record_decision(decision, f"user_{i}")
                
                # Record outcome
                framework.record_test_outcome(
                    test.test_id, f"user_{i}",
                    accuracy=0.8 + (i * 0.02),  # Varying accuracy
                    latency_ms=150.0 + (i * 10),  # Varying latency
                    memory_mb=800.0 + (i * 50),   # Varying memory
                    user_satisfaction=0.7 + (i * 0.05)
                )
        
        # Check metrics collection
        metrics = test.metrics_collector.get_test_metrics(test.test_id)
        assert len(metrics) > 0
        
        # Should have control and/or treatment groups
        for group, group_metrics in metrics.items():
            assert group in ['control', 'treatment']
            if group_metrics.accuracy_scores:
                assert len(group_metrics.accuracy_scores) > 0
                assert all(0.0 <= score <= 1.0 for score in group_metrics.accuracy_scores)
    
    def test_early_stopping_detection(self):
        """Test early stopping detection."""
        framework = ABTestingFramework()
        
        config = ABTestConfig(
            test_name="Early Stopping Test",
            control_method=RoutingMethod.CLASSICAL,
            treatment_method=RoutingMethod.QUANTUM,
            sample_size=100,
            min_sample_size=20,
            enable_early_stopping=True
        )
        
        test = framework.create_routing_test(config)
        framework.start_test(test)
        
        # Simulate clear performance difference
        for i in range(25):
            complexity_result = Mock()
            complexity_result.overall_complexity = Mock()
            complexity_result.overall_complexity.overall_complexity = 0.5
            complexity_result.query_complexity = Mock()
            complexity_result.query_complexity.modality_count = 1
            
            decision = framework.get_routing_decision(complexity_result)
            if decision:
                test.record_decision(decision, f"user_{i}")
                
                # Treatment performs significantly better
                if decision.method == RoutingMethod.QUANTUM:
                    accuracy = 0.9 + np.random.normal(0, 0.05)  # High accuracy
                else:
                    accuracy = 0.7 + np.random.normal(0, 0.05)  # Lower accuracy
                
                framework.record_test_outcome(
                    test.test_id, f"user_{i}",
                    accuracy=max(0.0, min(1.0, accuracy)),
                    latency_ms=150.0,
                    memory_mb=800.0
                )
        
        # Check for early stopping
        framework.check_early_stopping()
        
        # If significant difference is detected, test might be stopped
        # (This depends on statistical significance and effect size)
        stats = framework.get_framework_stats()
        assert stats['total_tests'] == 1
    
    def test_analyze_completed_test(self):
        """Test analysis of completed A/B test."""
        framework = ABTestingFramework()
        
        config = ABTestConfig(
            test_name="Analysis Test",
            control_method=RoutingMethod.CLASSICAL,
            treatment_method=RoutingMethod.QUANTUM,
            sample_size=20,
            primary_metric="accuracy"
        )
        
        test = framework.create_routing_test(config)
        framework.start_test(test)
        
        # Simulate test with known outcomes
        control_scores = [0.75, 0.78, 0.76, 0.77, 0.74, 0.79, 0.76, 0.78, 0.75, 0.77]
        treatment_scores = [0.82, 0.85, 0.83, 0.84, 0.81, 0.86, 0.83, 0.85, 0.82, 0.84]
        
        user_id = 0
        
        # Add control group results
        for score in control_scores:
            complexity_result = Mock()
            complexity_result.overall_complexity = Mock()
            complexity_result.overall_complexity.overall_complexity = 0.5
            complexity_result.query_complexity = Mock()
            complexity_result.query_complexity.modality_count = 1
            
            # Force control group assignment
            with patch.object(test, '_assign_group', return_value='control'):
                decision = test.get_routing_decision(complexity_result)
                test.record_decision(decision, f"user_{user_id}")
                framework.record_test_outcome(
                    test.test_id, f"user_{user_id}",
                    accuracy=score, latency_ms=150.0, memory_mb=800.0
                )
                user_id += 1
        
        # Add treatment group results
        for score in treatment_scores:
            complexity_result = Mock()
            complexity_result.overall_complexity = Mock()
            complexity_result.overall_complexity.overall_complexity = 0.5
            complexity_result.query_complexity = Mock()
            complexity_result.query_complexity.modality_count = 1
            
            # Force treatment group assignment
            with patch.object(test, '_assign_group', return_value='treatment'):
                decision = test.get_routing_decision(complexity_result)
                test.record_decision(decision, f"user_{user_id}")
                framework.record_test_outcome(
                    test.test_id, f"user_{user_id}",
                    accuracy=score, latency_ms=200.0, memory_mb=1200.0
                )
                user_id += 1
        
        # Stop test and analyze
        framework.stop_test(test.test_id, "manual_stop_for_analysis")
        
        analysis = framework.analyze_test_results(test.test_id)
        
        assert isinstance(analysis, ABTestAnalysis)
        assert analysis.test_name == "Analysis Test"
        assert analysis.winner in ['control', 'treatment', 'inconclusive']
        assert 0.0 <= analysis.confidence_level <= 1.0
        assert len(analysis.recommendations) > 0
        assert len(analysis.insights) >= 0
        
        # Control and treatment metrics should be available
        assert 'sample_count' in analysis.control_metrics
        assert 'sample_count' in analysis.treatment_metrics
        assert analysis.control_metrics['sample_count'] == len(control_scores)
        assert analysis.treatment_metrics['sample_count'] == len(treatment_scores)
    
    def test_export_test_results(self):
        """Test exporting test results for external analysis."""
        framework = ABTestingFramework()
        
        config = ABTestConfig(
            test_name="Export Test",
            control_method=RoutingMethod.CLASSICAL,
            treatment_method=RoutingMethod.QUANTUM,
            sample_size=5
        )
        
        test = framework.create_routing_test(config)
        framework.start_test(test)
        
        # Add some test data
        for i in range(3):
            complexity_result = Mock()
            complexity_result.overall_complexity = Mock()
            complexity_result.overall_complexity.overall_complexity = 0.5
            complexity_result.query_complexity = Mock()
            complexity_result.query_complexity.modality_count = 1
            
            decision = framework.get_routing_decision(complexity_result)
            if decision:
                test.record_decision(decision, f"user_{i}")
                framework.record_test_outcome(
                    test.test_id, f"user_{i}",
                    accuracy=0.8, latency_ms=150.0, memory_mb=800.0
                )
        
        # Export results
        export_data = framework.export_test_results(test.test_id)
        
        assert 'test_config' in export_data
        assert 'test_status' in export_data
        assert 'raw_results' in export_data
        assert 'export_timestamp' in export_data
        
        # Check test config
        test_config = export_data['test_config']
        assert test_config['test_name'] == "Export Test"
        assert test_config['control_method'] == RoutingMethod.CLASSICAL.value
        assert test_config['treatment_method'] == RoutingMethod.QUANTUM.value
        
        # Check raw results
        raw_results = export_data['raw_results']
        assert len(raw_results) > 0
        
        for result in raw_results:
            assert 'user_id' in result
            assert 'group' in result
            assert 'routing_method' in result
            assert 'accuracy' in result
            assert 'latency_ms' in result
            assert 'memory_mb' in result
    
    def test_cleanup_old_tests(self):
        """Test cleanup of old completed tests."""
        framework = ABTestingFramework()
        
        # Create and complete some tests
        for i in range(3):
            config = ABTestConfig(
                test_name=f"Old Test {i}",
                control_method=RoutingMethod.CLASSICAL,
                treatment_method=RoutingMethod.QUANTUM,
                sample_size=5
            )
            
            test = framework.create_routing_test(config)
            framework.start_test(test)
            framework.stop_test(test.test_id, "completed")
            
            # Simulate old test by setting old end time
            test.end_time = time.time() - (35 * 24 * 3600)  # 35 days ago
        
        assert len(framework.completed_tests) == 3
        
        # Cleanup tests older than 30 days
        framework.cleanup_old_tests(days_old=30)
        
        # All tests should be cleaned up
        assert len(framework.completed_tests) == 0
    
    def test_no_active_tests_returns_none(self):
        """Test that no routing decision is returned when no tests are active."""
        framework = ABTestingFramework()
        
        complexity_result = Mock()
        decision = framework.get_routing_decision(complexity_result)
        
        assert decision is None
        assert not framework.is_active()
    
    def test_framework_stats(self):
        """Test framework statistics retrieval."""
        framework = ABTestingFramework()
        
        # Create some tests
        config1 = ABTestConfig(
            test_name="Active Test",
            control_method=RoutingMethod.CLASSICAL,
            treatment_method=RoutingMethod.QUANTUM,
            sample_size=10
        )
        
        config2 = ABTestConfig(
            test_name="Completed Test",
            control_method=RoutingMethod.CLASSICAL,
            treatment_method=RoutingMethod.HYBRID,
            sample_size=10
        )
        
        test1 = framework.create_routing_test(config1)
        test2 = framework.create_routing_test(config2)
        
        framework.start_test(test1)
        framework.start_test(test2)
        framework.stop_test(test2.test_id, "completed")
        
        stats = framework.get_framework_stats()
        
        assert stats['total_tests'] == 2
        assert stats['active_tests'] == 1
        assert stats['completed_tests'] == 1
        assert len(stats['active_test_ids']) == 1
        assert len(stats['completed_test_ids']) == 1
        assert test1.test_id in stats['active_test_ids']
        assert test2.test_id in stats['completed_test_ids']


class TestStatisticalAnalyzer:
    """Integration tests for statistical analysis."""
    
    def test_basic_analysis_without_scipy(self):
        """Test basic analysis when scipy is not available."""
        analyzer = StatisticalAnalyzer()
        
        # Create mock metrics
        control_metrics = ABTestMetrics()
        control_metrics.accuracy_scores = [0.75, 0.76, 0.74, 0.77, 0.75]
        
        treatment_metrics = ABTestMetrics()
        treatment_metrics.accuracy_scores = [0.82, 0.83, 0.81, 0.84, 0.82]
        
        # Mock scipy unavailable
        with patch('quantum_rerank.routing.ab_testing_framework.SCIPY_AVAILABLE', False):
            results = analyzer.analyze_test(control_metrics, treatment_metrics)
            
            assert 'primary_metric' in results
            assert 'control_mean' in results['primary_metric']
            assert 'treatment_mean' in results['primary_metric']
            assert 'difference' in results['primary_metric']
            assert 'percent_change' in results['primary_metric']
            assert 'note' in results
    
    def test_statistical_analysis_with_scipy(self):
        """Test statistical analysis with scipy available."""
        analyzer = StatisticalAnalyzer()
        
        # Create metrics with clear difference
        control_metrics = ABTestMetrics()
        control_metrics.accuracy_scores = [0.70, 0.72, 0.71, 0.73, 0.70, 0.72, 0.71]
        control_metrics.latency_measurements = [200, 210, 205, 215, 200, 210, 205]
        control_metrics.memory_usage_measurements = [800, 820, 810, 830, 800, 820, 810]
        
        treatment_metrics = ABTestMetrics()
        treatment_metrics.accuracy_scores = [0.85, 0.87, 0.86, 0.88, 0.85, 0.87, 0.86]
        treatment_metrics.latency_measurements = [180, 185, 182, 188, 180, 185, 182]
        treatment_metrics.memory_usage_measurements = [900, 920, 910, 930, 900, 920, 910]
        
        # Mock scipy available
        with patch('quantum_rerank.routing.ab_testing_framework.SCIPY_AVAILABLE', True):
            results = analyzer.analyze_test(control_metrics, treatment_metrics)
            
            assert 'primary_metric' in results
            primary = results['primary_metric']
            assert 'p_value' in primary
            assert 't_statistic' in primary
            assert 'significant' in primary
            assert 'effect_size' in primary
            
            assert 'secondary_metrics' in results
            assert len(results['secondary_metrics']) > 0


class TestABTest:
    """Integration tests for individual A/B test."""
    
    def test_test_lifecycle(self):
        """Test complete A/B test lifecycle."""
        config = ABTestConfig(
            test_name="Lifecycle Test",
            control_method=RoutingMethod.CLASSICAL,
            treatment_method=RoutingMethod.QUANTUM,
            sample_size=10
        )
        
        test = ABTest(config)
        
        # Initial state
        assert test.status == ABTestStatus.PLANNING
        assert test.start_time is None
        
        # Start test
        success = test.start()
        assert success is True
        assert test.status == ABTestStatus.ACTIVE
        assert test.start_time is not None
        
        # Cannot start again
        success = test.start()
        assert success is False
        
        # Process some queries
        for i in range(5):
            complexity_result = Mock()
            complexity_result.overall_complexity = Mock()
            complexity_result.overall_complexity.overall_complexity = 0.5
            complexity_result.query_complexity = Mock()
            complexity_result.query_complexity.modality_count = 1
            
            if test.should_apply(complexity_result):
                decision = test.get_routing_decision(complexity_result)
                assert decision is not None
                test.record_decision(decision, f"user_{i}")
                test.record_outcome(f"user_{i}", 0.8, 150.0, 800.0, 0.7)
        
        # Check status
        status = test.get_current_status()
        assert 'test_id' in status
        assert 'test_name' in status
        assert 'status' in status
        assert 'control_samples' in status
        assert 'treatment_samples' in status
        assert 'progress' in status
    
    def test_sample_size_limit(self):
        """Test that test stops applying when sample size is reached."""
        config = ABTestConfig(
            test_name="Sample Size Test",
            control_method=RoutingMethod.CLASSICAL,
            treatment_method=RoutingMethod.QUANTUM,
            sample_size=3  # Small sample size
        )
        
        test = ABTest(config)
        test.start()
        
        complexity_result = Mock()
        complexity_result.overall_complexity = Mock()
        complexity_result.overall_complexity.overall_complexity = 0.5
        complexity_result.query_complexity = Mock()
        complexity_result.query_complexity.modality_count = 1
        
        # Fill up sample size
        applications = 0
        for i in range(10):  # Try more than sample size
            if test.should_apply(complexity_result):
                decision = test.get_routing_decision(complexity_result)
                test.record_decision(decision, f"user_{i}")
                applications += 1
        
        # Should not exceed sample size
        total_samples = test.control_samples + test.treatment_samples
        assert total_samples <= config.sample_size
    
    def test_early_stopping_conditions(self):
        """Test early stopping condition detection."""
        config = ABTestConfig(
            test_name="Early Stopping Test",
            control_method=RoutingMethod.CLASSICAL,
            treatment_method=RoutingMethod.QUANTUM,
            sample_size=100,
            min_sample_size=10,
            enable_early_stopping=True
        )
        
        test = ABTest(config)
        test.start()
        
        # Add insufficient samples
        for i in range(5):
            complexity_result = Mock()
            complexity_result.overall_complexity = Mock()
            complexity_result.overall_complexity.overall_complexity = 0.5
            complexity_result.query_complexity = Mock()
            complexity_result.query_complexity.modality_count = 1
            
            if test.should_apply(complexity_result):
                decision = test.get_routing_decision(complexity_result)
                test.record_decision(decision, f"user_{i}")
                test.record_outcome(f"user_{i}", 0.8, 150.0, 800.0)
        
        # Should not stop early with insufficient samples
        should_stop, reason = test.should_stop_early()
        assert should_stop is False
        assert reason == "insufficient_samples"


if __name__ == '__main__':
    pytest.main([__file__])
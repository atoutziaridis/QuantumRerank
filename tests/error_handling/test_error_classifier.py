"""
Test suite for error classification system.

Tests the multi-dimensional error classification, pattern detection,
and error taxonomy functionality.
"""

import pytest
import time
from unittest.mock import Mock, patch

from quantum_rerank.error_handling.error_classifier import (
    ErrorClassifier, ErrorSeverity, ErrorCategory, ErrorClassification,
    ErrorPattern, ErrorPatternType, ErrorMetrics
)
from quantum_rerank.utils.exceptions import (
    QuantumCircuitError, SimilarityComputationError, PerformanceError
)


class TestErrorClassifier:
    """Test cases for ErrorClassifier."""
    
    @pytest.fixture
    def classifier(self):
        """Create error classifier instance."""
        return ErrorClassifier()
    
    @pytest.fixture
    def sample_context(self):
        """Create sample context for testing."""
        return {
            "component": "quantum_engine",
            "operation": "fidelity_computation",
            "cpu_usage": 50.0,
            "memory_usage": 60.0,
            "environment": "development"
        }
    
    def test_classify_quantum_circuit_error(self, classifier, sample_context):
        """Test classification of quantum circuit errors."""
        error = QuantumCircuitError("Circuit compilation failed")
        
        classification = classifier.classify_error(error, sample_context)
        
        assert classification.category == ErrorCategory.QUANTUM_COMPUTATION
        assert classification.severity in [ErrorSeverity.MEDIUM, ErrorSeverity.HIGH]
        assert classification.recoverable is True
        assert classification.fallback_available is True
        assert classification.confidence > 0.5
        assert "quantum" in " ".join(classification.suggested_actions).lower()
    
    def test_classify_similarity_computation_error(self, classifier):
        """Test classification of similarity computation errors."""
        error = SimilarityComputationError("Similarity computation timeout")
        context = {
            "component": "similarity_engine",
            "operation": "cosine_similarity",
            "timeout": True
        }
        
        classification = classifier.classify_error(error, context)
        
        assert classification.category == ErrorCategory.CLASSICAL_COMPUTATION
        assert classification.recoverable is True
        assert "similarity" in classification.error_type.lower()
    
    def test_classify_performance_error(self, classifier):
        """Test classification of performance errors."""
        error = PerformanceError("Latency threshold exceeded")
        context = {
            "component": "performance_monitor",
            "operation": "latency_check",
            "actual_latency": 500.0,
            "target_latency": 100.0
        }
        
        classification = classifier.classify_error(error, context)
        
        assert classification.category == ErrorCategory.PERFORMANCE_DEGRADATION
        assert classification.severity == ErrorSeverity.MEDIUM
        assert "performance" in " ".join(classification.suggested_actions).lower()
    
    def test_severity_adjustment_by_context(self, classifier):
        """Test severity adjustment based on context."""
        error = Exception("Test error")
        
        # High resource usage should escalate severity
        high_load_context = {
            "component": "test",
            "operation": "test",
            "cpu_usage": 95.0,
            "memory_usage": 98.0,
            "environment": "production",
            "critical_operation": True
        }
        
        low_load_context = {
            "component": "test",
            "operation": "test",
            "cpu_usage": 20.0,
            "memory_usage": 30.0,
            "environment": "development"
        }
        
        high_load_classification = classifier.classify_error(error, high_load_context)
        low_load_classification = classifier.classify_error(error, low_load_context)
        
        # High load should result in higher severity
        severity_levels = [ErrorSeverity.INFO, ErrorSeverity.LOW, 
                          ErrorSeverity.MEDIUM, ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
        
        high_load_index = severity_levels.index(high_load_classification.severity)
        low_load_index = severity_levels.index(low_load_classification.severity)
        
        assert high_load_index >= low_load_index
    
    def test_error_pattern_detection(self, classifier):
        """Test error pattern detection."""
        # Simulate frequency spike pattern
        for i in range(15):
            error_metrics = ErrorMetrics(
                error_type="TestError",
                timestamp=time.time() - (i * 60),  # Errors every minute
                component="test_component",
                severity=ErrorSeverity.MEDIUM
            )
            classifier.error_history.append(error_metrics)
        
        patterns = classifier.detect_error_patterns()
        
        # Should detect frequency spike
        frequency_patterns = [p for p in patterns if p.pattern_type == ErrorPatternType.FREQUENCY_SPIKE]
        assert len(frequency_patterns) > 0
        
        pattern = frequency_patterns[0]
        assert pattern.frequency > 10  # More than 10 errors per hour
        assert "test_component" in pattern.affected_components
    
    def test_error_statistics(self, classifier):
        """Test error statistics calculation."""
        # Add some test errors
        test_errors = [
            ErrorMetrics("Error1", time.time(), "comp1", ErrorSeverity.LOW),
            ErrorMetrics("Error2", time.time(), "comp2", ErrorSeverity.HIGH),
            ErrorMetrics("Error1", time.time(), "comp1", ErrorSeverity.MEDIUM),
        ]
        
        for error in test_errors:
            classifier.error_history.append(error)
        
        stats = classifier.get_error_statistics(time_window_s=3600)
        
        assert stats["total_errors"] == len(test_errors)
        assert stats["error_rate_per_hour"] > 0
        assert "severity_distribution" in stats
        assert "component_distribution" in stats
    
    def test_error_prediction(self, classifier):
        """Test error likelihood prediction."""
        # Add some historical patterns
        for i in range(10):
            error_metrics = ErrorMetrics(
                error_type="QuantumCircuitError",
                timestamp=time.time() - (i * 300),  # Every 5 minutes
                component="quantum_engine",
                severity=ErrorSeverity.MEDIUM
            )
            classifier.error_history.append(error_metrics)
        
        # Create a pattern
        from quantum_rerank.error_handling.error_classifier import ErrorPattern
        pattern = ErrorPattern(
            pattern_type=ErrorPatternType.PERIODIC_FAILURE,
            affected_components=["quantum_engine"],
            frequency=12.0,  # 12 per hour
            severity_trend="stable",
            first_occurrence=time.time() - 3000,
            last_occurrence=time.time() - 300,
            pattern_confidence=0.8,
            prediction={
                "error_type": "QuantumCircuitError",
                "likelihood": 0.3,
                "next_occurrence": time.time() + 300
            }
        )
        
        classifier.detected_patterns["test_pattern"] = pattern
        
        predictions = classifier.predict_error_likelihood(
            "quantum_engine", 
            "fidelity_computation",
            {"cpu_usage": 85}  # High CPU should increase likelihood
        )
        
        assert "QuantumCircuitError" in predictions
        assert predictions["QuantumCircuitError"] > 0.3  # Should be adjusted upward
    
    def test_classification_confidence(self, classifier):
        """Test classification confidence calculation."""
        # Error with clear quantum indicators
        quantum_error = QuantumCircuitError("Quantum circuit compilation failed")
        quantum_context = {
            "component": "quantum_engine",
            "operation": "circuit_compilation",
            "circuit_depth": 15,
            "n_qubits": 4
        }
        
        # Generic error with minimal context
        generic_error = Exception("Something went wrong")
        generic_context = {"component": "unknown"}
        
        quantum_classification = classifier.classify_error(quantum_error, quantum_context)
        generic_classification = classifier.classify_error(generic_error, generic_context)
        
        # Quantum error should have higher confidence
        assert quantum_classification.confidence > generic_classification.confidence
        assert quantum_classification.confidence > 0.7
    
    def test_related_components_identification(self, classifier):
        """Test identification of related components."""
        error = QuantumCircuitError("Circuit failed")
        context = {
            "component": "quantum_engine",
            "operation": "fidelity_computation"
        }
        
        classification = classifier.classify_error(error, context)
        
        # Should identify related components for quantum errors
        assert len(classification.related_components) > 0
        # Classical similarity should be related to quantum engine
        related_components_str = " ".join(classification.related_components)
        assert any(comp in related_components_str for comp in ["classical", "embedding", "similarity"])


class TestErrorPatternDetection:
    """Test cases for error pattern detection."""
    
    @pytest.fixture
    def classifier(self):
        """Create error classifier instance."""
        return ErrorClassifier()
    
    def test_cascading_failure_detection(self, classifier):
        """Test detection of cascading failure patterns."""
        # Simulate cascading failures across components
        base_time = time.time()
        
        # Errors spreading across components over time
        cascade_errors = [
            ErrorMetrics("Error1", base_time, "component_a", ErrorSeverity.HIGH),
            ErrorMetrics("Error2", base_time + 60, "component_b", ErrorSeverity.HIGH),
            ErrorMetrics("Error3", base_time + 120, "component_c", ErrorSeverity.HIGH),
            ErrorMetrics("Error4", base_time + 180, "component_d", ErrorSeverity.HIGH),
        ]
        
        for error in cascade_errors:
            classifier.error_history.append(error)
        
        patterns = classifier.detect_error_patterns()
        
        # Should detect cascading failure
        cascade_patterns = [p for p in patterns if p.pattern_type == ErrorPatternType.CASCADING_FAILURE]
        assert len(cascade_patterns) > 0
        
        pattern = cascade_patterns[0]
        assert len(pattern.affected_components) >= 3
        assert pattern.pattern_confidence > 0.8
    
    def test_periodic_failure_detection(self, classifier):
        """Test detection of periodic failure patterns."""
        # Simulate periodic failures every 10 minutes
        base_time = time.time()
        interval = 600  # 10 minutes
        
        periodic_errors = [
            ErrorMetrics("PeriodicError", base_time - i * interval, "periodic_component", ErrorSeverity.MEDIUM)
            for i in range(6)  # 6 periodic errors
        ]
        
        for error in periodic_errors:
            classifier.error_history.append(error)
        
        patterns = classifier.detect_error_patterns()
        
        # Should detect periodic pattern
        periodic_patterns = [p for p in patterns if p.pattern_type == ErrorPatternType.PERIODIC_FAILURE]
        assert len(periodic_patterns) > 0
        
        pattern = periodic_patterns[0]
        assert pattern.prediction is not None
        assert "next_occurrence" in pattern.prediction
        assert "interval_seconds" in pattern.prediction
        assert abs(pattern.prediction["interval_seconds"] - interval) < 60  # Within 1 minute tolerance
    
    def test_gradual_degradation_detection(self, classifier):
        """Test detection of gradual degradation patterns."""
        # Simulate gradual severity increase
        base_time = time.time()
        severities = [ErrorSeverity.LOW, ErrorSeverity.LOW, ErrorSeverity.MEDIUM, 
                     ErrorSeverity.MEDIUM, ErrorSeverity.HIGH, ErrorSeverity.HIGH,
                     ErrorSeverity.CRITICAL, ErrorSeverity.CRITICAL]
        
        degradation_errors = [
            ErrorMetrics("DegradingError", base_time - i * 300, "degrading_component", severity)
            for i, severity in enumerate(reversed(severities))
        ]
        
        for error in degradation_errors:
            classifier.error_history.append(error)
        
        patterns = classifier.detect_error_patterns()
        
        # Should detect gradual degradation
        degradation_patterns = [p for p in patterns if p.pattern_type == ErrorPatternType.GRADUAL_DEGRADATION]
        assert len(degradation_patterns) > 0
        
        pattern = degradation_patterns[0]
        assert pattern.severity_trend == "increasing"
        assert pattern.prediction is not None
        assert "degradation_rate" in pattern.prediction
        assert pattern.prediction["degradation_rate"] > 0.5  # Strong positive correlation


if __name__ == "__main__":
    pytest.main([__file__])
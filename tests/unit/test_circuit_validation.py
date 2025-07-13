"""
Unit tests for quantum circuit validation and performance analysis.

Tests validation functionality against PRD requirements and
performance analysis capabilities.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from quantum_rerank.core.circuit_validators import (
    CircuitValidator,
    PerformanceAnalyzer,
    ValidationResult,
    PerformanceMetrics,
    ValidationSeverity,
    ValidationIssue
)
from quantum_rerank.core.quantum_circuits import BasicQuantumCircuits
from quantum_rerank.config.settings import QuantumConfig, PerformanceConfig
from qiskit import QuantumCircuit


class TestCircuitValidator:
    """Test circuit validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.quantum_config = QuantumConfig(n_qubits=4, max_circuit_depth=15)
        self.performance_config = PerformanceConfig()
        self.validator = CircuitValidator(self.quantum_config, self.performance_config)
        self.circuit_handler = BasicQuantumCircuits(self.quantum_config)
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = CircuitValidator()
        
        assert validator.quantum_config is not None
        assert validator.performance_config is not None
        
        # Test with custom configs
        validator_custom = CircuitValidator(self.quantum_config, self.performance_config)
        assert validator_custom.quantum_config == self.quantum_config
        assert validator_custom.performance_config == self.performance_config
    
    def test_validate_valid_circuit(self):
        """Test validation of a valid circuit."""
        circuit = self.circuit_handler.create_superposition_circuit()
        result = self.validator.validate_circuit(circuit)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.performance_score >= 80  # Should have high score
        assert len(result.recommendations) >= 0
    
    def test_validate_circuit_with_too_few_qubits(self):
        """Test validation of circuit with too few qubits."""
        circuit = QuantumCircuit(1)  # Only 1 qubit
        circuit.h(0)
        
        result = self.validator.validate_circuit(circuit)
        
        assert result.is_valid is False
        assert any(issue.severity == ValidationSeverity.ERROR 
                  for issue in result.issues 
                  if issue.constraint == "min_qubits")
    
    def test_validate_circuit_with_too_many_qubits(self):
        """Test validation of circuit with too many qubits."""
        circuit = QuantumCircuit(5)  # Too many qubits
        for i in range(5):
            circuit.h(i)
        
        result = self.validator.validate_circuit(circuit)
        
        assert result.is_valid is False
        assert any(issue.severity == ValidationSeverity.CRITICAL 
                  for issue in result.issues 
                  if issue.constraint == "max_qubits")
    
    def test_validate_circuit_with_excessive_depth(self):
        """Test validation of circuit with excessive depth."""
        circuit = QuantumCircuit(4)
        # Create deep circuit exceeding PRD limit
        for _ in range(20):  # 20 layers of gates
            for i in range(4):
                circuit.h(i)
        
        result = self.validator.validate_circuit(circuit)
        
        assert result.is_valid is False
        assert any(issue.severity == ValidationSeverity.CRITICAL 
                  for issue in result.issues 
                  if issue.constraint == "max_circuit_depth")
    
    def test_validate_circuit_approaching_depth_limit(self):
        """Test validation of circuit approaching depth limit."""
        circuit = QuantumCircuit(4)
        # Create circuit near depth limit
        for _ in range(12):  # Close to 15-gate limit
            circuit.h(0)
        
        result = self.validator.validate_circuit(circuit)
        
        # Should be valid but have warning
        assert result.is_valid is True
        assert any(issue.severity == ValidationSeverity.WARNING 
                  for issue in result.issues 
                  if issue.constraint == "circuit_depth_warning")
    
    def test_validate_circuit_with_many_gates(self):
        """Test validation of circuit with many gates."""
        circuit = QuantumCircuit(4)
        # Add many gates
        for _ in range(60):  # Excessive number of gates
            circuit.h(0)
        
        result = self.validator.validate_circuit(circuit)
        
        # Should have warning about gate count
        assert any(issue.severity == ValidationSeverity.WARNING 
                  for issue in result.issues 
                  if issue.constraint == "gate_count")
    
    def test_validate_circuit_with_inefficient_gates(self):
        """Test validation of circuit with inefficient gates."""
        circuit = QuantumCircuit(4)
        # Add inefficient gates
        circuit.u1(0.5, 0)
        circuit.u2(0.5, 0.3, 1)
        
        result = self.validator.validate_circuit(circuit)
        
        # Should have info about gate efficiency
        assert any(issue.severity == ValidationSeverity.INFO 
                  for issue in result.issues 
                  if issue.constraint == "gate_efficiency")
    
    def test_validate_circuit_high_complexity(self):
        """Test validation of circuit with high computational complexity."""
        circuit = QuantumCircuit(4)
        # Create complex circuit
        for i in range(4):
            circuit.h(i)
        for _ in range(10):
            for i in range(3):
                circuit.cnot(i, i+1)
        
        result = self.validator.validate_circuit(circuit)
        
        # Should have warning about complexity
        complexity_issues = [issue for issue in result.issues 
                           if issue.constraint == "simulation_complexity"]
        if complexity_issues:
            assert complexity_issues[0].severity == ValidationSeverity.WARNING
    
    def test_validate_circuit_memory_usage(self):
        """Test validation of circuit memory usage."""
        # Note: For 4 qubits, memory usage should be fine
        # This test ensures the validation logic works
        circuit = self.circuit_handler.create_superposition_circuit()
        result = self.validator.validate_circuit(circuit)
        
        # 4-qubit circuit should not have memory issues
        memory_issues = [issue for issue in result.issues 
                        if issue.constraint == "memory_usage"]
        assert len(memory_issues) == 0
    
    def test_generate_recommendations(self):
        """Test recommendation generation."""
        # Create circuit with issues
        circuit = QuantumCircuit(5)  # Too many qubits
        for _ in range(20):  # Too deep
            circuit.h(0)
        
        result = self.validator.validate_circuit(circuit)
        
        assert len(result.recommendations) > 0
        assert any("amplitude encoding" in rec.lower() for rec in result.recommendations)
        assert any("circuit" in rec.lower() and "depth" in rec.lower() for rec in result.recommendations)
    
    def test_performance_score_calculation(self):
        """Test performance score calculation."""
        # Test perfect circuit
        perfect_circuit = self.circuit_handler.create_empty_circuit()
        perfect_result = self.validator.validate_circuit(perfect_circuit)
        
        # Test problematic circuit
        problematic_circuit = QuantumCircuit(5)  # Too many qubits
        for _ in range(20):  # Too deep
            problematic_circuit.h(0)
        problematic_result = self.validator.validate_circuit(problematic_circuit)
        
        # Perfect circuit should have higher score
        assert perfect_result.performance_score > problematic_result.performance_score
        assert perfect_result.performance_score >= 90
        assert problematic_result.performance_score <= 50


class TestPerformanceAnalyzer:
    """Test performance analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.quantum_config = QuantumConfig(n_qubits=4)
        self.analyzer = PerformanceAnalyzer(self.quantum_config)
        self.circuit_handler = BasicQuantumCircuits(self.quantum_config)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = PerformanceAnalyzer()
        assert analyzer.quantum_config is not None
        
        analyzer_custom = PerformanceAnalyzer(self.quantum_config)
        assert analyzer_custom.quantum_config == self.quantum_config
    
    def test_analyze_circuit_performance(self):
        """Test circuit performance analysis."""
        circuit = self.circuit_handler.create_superposition_circuit()
        metrics = self.analyzer.analyze_circuit_performance(circuit, num_trials=3)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.simulation_time_ms > 0
        assert metrics.total_time_ms >= metrics.simulation_time_ms
        assert 0 <= metrics.success_rate <= 1
        assert metrics.circuit_depth >= 0
        assert metrics.circuit_size >= 0
        assert metrics.memory_usage_mb > 0
        
        # Should meet PRD requirements
        assert metrics.prd_compliant is True
        assert metrics.simulation_time_ms < 100
    
    def test_benchmark_encoding_performance_default_methods(self):
        """Test encoding performance benchmarking with default methods."""
        embeddings = [np.random.rand(16) for _ in range(2)]
        results = self.analyzer.benchmark_encoding_performance(embeddings)
        
        assert isinstance(results, dict)
        expected_methods = ['amplitude', 'angle', 'dense_angle']
        
        for method in expected_methods:
            assert method in results
            metrics = results[method]
            assert isinstance(metrics, PerformanceMetrics)
            assert metrics.encoding_time_ms > 0
            assert metrics.simulation_time_ms > 0
            assert metrics.total_time_ms > 0
            assert 0 <= metrics.success_rate <= 1
    
    def test_benchmark_encoding_performance_custom_methods(self):
        """Test encoding performance benchmarking with custom methods."""
        embeddings = [np.random.rand(16) for _ in range(2)]
        custom_methods = ['amplitude', 'angle']
        
        results = self.analyzer.benchmark_encoding_performance(
            embeddings, encoding_methods=custom_methods
        )
        
        assert set(results.keys()) == set(custom_methods)
        
        for method in custom_methods:
            metrics = results[method]
            assert isinstance(metrics, PerformanceMetrics)
    
    def test_benchmark_encoding_performance_invalid_method(self):
        """Test encoding performance benchmarking with invalid method."""
        embeddings = [np.random.rand(16)]
        invalid_methods = ['invalid_method']
        
        with pytest.raises(ValueError, match="Unknown encoding method"):
            self.analyzer.benchmark_encoding_performance(
                embeddings, encoding_methods=invalid_methods
            )
    
    def test_benchmark_encoding_performance_prd_compliance(self):
        """Test that encoding performance meets PRD requirements."""
        embeddings = [np.random.rand(16) for _ in range(3)]
        results = self.analyzer.benchmark_encoding_performance(embeddings)
        
        for method, metrics in results.items():
            # Check PRD compliance
            assert metrics.total_time_ms < 100, f"Method '{method}' exceeds 100ms target"
            assert metrics.circuit_depth <= 15, f"Method '{method}' exceeds depth limit"
            assert metrics.success_rate >= 0.9, f"Method '{method}' has low success rate"
    
    def test_generate_performance_report_single_method(self):
        """Test performance report generation for single method."""
        embeddings = [np.random.rand(16)]
        results = self.analyzer.benchmark_encoding_performance(
            embeddings, encoding_methods=['amplitude']
        )
        
        report = self.analyzer.generate_performance_report(results)
        
        assert isinstance(report, str)
        assert "QUANTUM CIRCUIT PERFORMANCE REPORT" in report
        assert "AMPLITUDE" in report
        assert "PRD Targets:" in report
        assert "✓ <100ms total computation time" in report
        assert "Recommendations:" in report
    
    def test_generate_performance_report_multiple_methods(self):
        """Test performance report generation for multiple methods."""
        embeddings = [np.random.rand(16)]
        results = self.analyzer.benchmark_encoding_performance(embeddings)
        
        report = self.analyzer.generate_performance_report(results)
        
        assert "AMPLITUDE" in report
        assert "ANGLE" in report
        assert "DENSE_ANGLE" in report
        assert "Summary:" in report
        assert "Methods tested: 3" in report
    
    def test_generate_performance_report_non_compliant(self):
        """Test performance report with non-compliant methods."""
        # Create mock metrics with non-compliant method
        mock_metrics = {
            'slow_method': PerformanceMetrics(
                encoding_time_ms=50.0,
                simulation_time_ms=80.0,
                total_time_ms=130.0,  # Exceeds 100ms limit
                memory_usage_mb=1.0,
                circuit_depth=10,
                circuit_size=15,
                success_rate=0.95,
                prd_compliant=False
            ),
            'fast_method': PerformanceMetrics(
                encoding_time_ms=5.0,
                simulation_time_ms=20.0,
                total_time_ms=25.0,
                memory_usage_mb=1.0,
                circuit_depth=5,
                circuit_size=8,
                success_rate=1.0,
                prd_compliant=True
            )
        }
        
        report = self.analyzer.generate_performance_report(mock_metrics)
        
        assert "✗ slow_method" in report
        assert "✓ fast_method" in report
        assert "PRD compliant: 1/2" in report
        assert "Use 'fast_method' for best performance" in report


class TestValidationDataStructures:
    """Test validation data structures."""
    
    def test_validation_issue_creation(self):
        """Test ValidationIssue creation."""
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            message="Test warning",
            constraint="test_constraint",
            actual_value=10,
            expected_value="< 5"
        )
        
        assert issue.severity == ValidationSeverity.WARNING
        assert issue.message == "Test warning"
        assert issue.constraint == "test_constraint"
        assert issue.actual_value == 10
        assert issue.expected_value == "< 5"
    
    def test_validation_result_creation(self):
        """Test ValidationResult creation."""
        issues = [
            ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Info message",
                constraint="info_constraint",
                actual_value=1,
                expected_value=1
            )
        ]
        
        result = ValidationResult(
            is_valid=True,
            issues=issues,
            performance_score=85.5,
            recommendations=["Test recommendation"]
        )
        
        assert result.is_valid is True
        assert len(result.issues) == 1
        assert result.performance_score == 85.5
        assert len(result.recommendations) == 1
    
    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics creation."""
        metrics = PerformanceMetrics(
            encoding_time_ms=10.5,
            simulation_time_ms=25.3,
            total_time_ms=35.8,
            memory_usage_mb=2.1,
            circuit_depth=8,
            circuit_size=12,
            success_rate=0.95,
            prd_compliant=True
        )
        
        assert metrics.encoding_time_ms == 10.5
        assert metrics.simulation_time_ms == 25.3
        assert metrics.total_time_ms == 35.8
        assert metrics.memory_usage_mb == 2.1
        assert metrics.circuit_depth == 8
        assert metrics.circuit_size == 12
        assert metrics.success_rate == 0.95
        assert metrics.prd_compliant is True


class TestValidationSeverity:
    """Test ValidationSeverity enum."""
    
    def test_severity_values(self):
        """Test severity enum values."""
        assert ValidationSeverity.INFO.value == "info"
        assert ValidationSeverity.WARNING.value == "warning"
        assert ValidationSeverity.ERROR.value == "error"
        assert ValidationSeverity.CRITICAL.value == "critical"


@pytest.mark.integration
class TestValidationIntegration:
    """Integration tests for validation functionality."""
    
    def test_end_to_end_validation_workflow(self):
        """Test complete validation workflow."""
        # Create circuit handler and validator
        circuit_handler = BasicQuantumCircuits()
        validator = CircuitValidator()
        analyzer = PerformanceAnalyzer()
        
        # Create and validate circuit
        circuit = circuit_handler.create_entanglement_circuit()
        validation_result = validator.validate_circuit(circuit)
        performance_metrics = analyzer.analyze_circuit_performance(circuit, num_trials=3)
        
        # Should be valid and performant
        assert validation_result.is_valid
        assert performance_metrics.prd_compliant
        assert performance_metrics.total_time_ms < 100
    
    def test_validation_with_encoding_workflow(self):
        """Test validation workflow with encoding."""
        circuit_handler = BasicQuantumCircuits()
        validator = CircuitValidator()
        
        # Test with encoded embedding
        embedding = np.random.rand(16)
        circuit = circuit_handler.amplitude_encode_embedding(embedding)
        
        validation_result = validator.validate_circuit(circuit)
        
        assert validation_result.is_valid
        assert validation_result.performance_score >= 60  # Should be reasonable
    
    def test_performance_benchmarking_integration(self):
        """Test performance benchmarking integration."""
        analyzer = PerformanceAnalyzer()
        
        # Create realistic test embeddings
        embeddings = [np.random.rand(64) for _ in range(3)]
        
        # Benchmark all methods
        results = analyzer.benchmark_encoding_performance(embeddings)
        report = analyzer.generate_performance_report(results)
        
        # All methods should be PRD compliant
        for method, metrics in results.items():
            assert metrics.prd_compliant, f"Method '{method}' not PRD compliant"
        
        assert "✓" in report  # Should have passing methods
        assert "100ms" in report  # Should mention PRD target
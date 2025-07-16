"""
Unit tests for uncertainty quantification module.

Tests the quantum uncertainty quantification implementation
for QMMR-03 task requirements.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from quantum_rerank.core.uncertainty_quantification import (
    UncertaintyMetrics,
    QuantumFidelityUncertaintyQuantifier,
    MultimodalUncertaintyQuantifier
)


class TestUncertaintyMetrics:
    """Test cases for UncertaintyMetrics dataclass."""
    
    def test_initialization(self):
        """Test proper initialization of UncertaintyMetrics."""
        metrics = UncertaintyMetrics()
        
        assert metrics.quantum_variance == 0.0
        assert metrics.measurement_uncertainty == 0.0
        assert metrics.confidence_intervals == {}
        assert metrics.uncertainty_validation == {}
    
    def test_total_uncertainty_calculation(self):
        """Test total uncertainty calculation."""
        metrics = UncertaintyMetrics(
            quantum_variance=0.01,
            measurement_uncertainty=0.02,
            coherence_uncertainty=0.015,
            entanglement_uncertainty=0.01,
            statistical_variance=0.005,
            shot_noise_uncertainty=0.008
        )
        
        total_uncertainty = metrics.get_total_uncertainty()
        
        # Verify reasonable range
        assert 0 <= total_uncertainty <= 1
        assert total_uncertainty > 0  # Should be non-zero with non-zero components
    
    def test_uncertainty_bounds(self):
        """Test that uncertainty values are properly bounded."""
        metrics = UncertaintyMetrics(
            quantum_variance=1.5,  # Over bounds
            measurement_uncertainty=-0.1  # Under bounds
        )
        
        total_uncertainty = metrics.get_total_uncertainty()
        
        # Should still be bounded despite invalid inputs
        assert 0 <= total_uncertainty <= 1


class TestQuantumFidelityUncertaintyQuantifier:
    """Test cases for QuantumFidelityUncertaintyQuantifier."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.quantifier = QuantumFidelityUncertaintyQuantifier(
            confidence_levels=[0.95, 0.99]
        )
        
        self.sample_measurement_data = {
            'fidelity': 0.85,
            'shots': 1024,
            'counts': {'0': 900, '1': 124},
            'circuit_depth': 10,
            'entanglement_measure': 0.7
        }
    
    def test_initialization(self):
        """Test proper initialization."""
        assert self.quantifier is not None
        assert self.quantifier.confidence_levels == [0.95, 0.99]
        assert self.quantifier.bootstrap_samples == 1000
        assert self.quantifier.measurement_efficiency == 0.95
    
    def test_quantify_uncertainty_basic(self):
        """Test basic uncertainty quantification."""
        metrics = self.quantifier.quantify_uncertainty(self.sample_measurement_data)
        
        # Verify structure
        assert isinstance(metrics, UncertaintyMetrics)
        assert 0 <= metrics.quantum_variance <= 1
        assert 0 <= metrics.statistical_variance <= 1
        assert len(metrics.confidence_intervals) == 2  # 95% and 99%
        assert metrics.uncertainty_validation is not None
    
    def test_quantum_variance_computation(self):
        """Test quantum variance computation."""
        fidelity = 0.8
        circuit_depth = 12
        
        quantum_var = self.quantifier._compute_quantum_variance(fidelity, circuit_depth)
        
        # Verify bounds and behavior
        assert 0 <= quantum_var <= 1
        # Deeper circuits should have higher variance
        shallow_var = self.quantifier._compute_quantum_variance(fidelity, 5)
        assert quantum_var >= shallow_var
    
    def test_measurement_uncertainty(self):
        """Test measurement uncertainty computation."""
        fidelity = 0.9
        efficiency = 0.95
        
        meas_uncertainty = self.quantifier._compute_measurement_uncertainty(
            fidelity, efficiency
        )
        
        # Verify bounds
        assert 0 <= meas_uncertainty <= 1
        
        # Lower efficiency should increase uncertainty
        low_eff_uncertainty = self.quantifier._compute_measurement_uncertainty(
            fidelity, 0.8
        )
        assert low_eff_uncertainty >= meas_uncertainty
    
    def test_statistical_variance(self):
        """Test statistical variance computation."""
        fidelity = 0.75
        shots = 1024
        
        stat_var = self.quantifier._compute_statistical_variance(fidelity, shots)
        
        # Verify bounds
        assert 0 <= stat_var <= 1
        
        # More shots should reduce variance
        high_shots_var = self.quantifier._compute_statistical_variance(fidelity, 4096)
        assert high_shots_var <= stat_var
    
    def test_confidence_intervals(self):
        """Test confidence interval computation."""
        fidelity = 0.8
        shots = 1024
        counts = {'0': 850, '1': 174}
        
        ci_dict = self.quantifier._compute_confidence_intervals(
            fidelity, shots, counts
        )
        
        # Verify structure
        assert 'ci_95' in ci_dict
        assert 'ci_99' in ci_dict
        
        for ci_level, ci_data in ci_dict.items():
            # Check normal approximation intervals
            assert 'normal_lower' in ci_data
            assert 'normal_upper' in ci_data
            assert 'normal_width' in ci_data
            
            # Check ordering
            assert ci_data['normal_lower'] <= fidelity <= ci_data['normal_upper']
            assert ci_data['normal_width'] >= 0
            
            # Check Wilson intervals
            assert 'wilson_lower' in ci_data
            assert 'wilson_upper' in ci_data
            assert ci_data['wilson_lower'] <= ci_data['wilson_upper']
    
    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap confidence interval computation."""
        counts = {'0': 800, '1': 224}
        shots = 1024
        
        with patch('numpy.random.binomial') as mock_binomial:
            # Mock bootstrap samples
            mock_binomial.return_value = 820
            
            bootstrap_ci = self.quantifier._bootstrap_confidence_intervals(
                counts, shots
            )
            
            # Verify structure
            assert len(bootstrap_ci) == 2  # 95% and 99%
            
            for ci_level, ci_data in bootstrap_ci.items():
                assert 'bootstrap_lower' in ci_data
                assert 'bootstrap_upper' in ci_data
                assert 'bootstrap_width' in ci_data
                assert ci_data['bootstrap_lower'] <= ci_data['bootstrap_upper']
    
    def test_systematic_uncertainty(self):
        """Test systematic uncertainty computation."""
        circuit_depth = 15
        entanglement = 0.8
        
        systematic = self.quantifier._compute_systematic_uncertainty(
            circuit_depth, entanglement
        )
        
        # Verify bounds
        assert 0 <= systematic <= 1
        
        # Higher complexity should increase systematic uncertainty
        low_systematic = self.quantifier._compute_systematic_uncertainty(5, 0.2)
        assert systematic >= low_systematic
    
    def test_uncertainty_validation(self):
        """Test uncertainty metrics validation."""
        # Valid metrics
        valid_metrics = UncertaintyMetrics(
            quantum_variance=0.02,
            statistical_variance=0.01,
            systematic_uncertainty=0.03,
            random_uncertainty=0.015,
            confidence_intervals={
                'ci_95': {
                    'normal_lower': 0.75,
                    'normal_upper': 0.85,
                    'wilson_lower': 0.76,
                    'wilson_upper': 0.84
                }
            }
        )
        
        validation = self.quantifier._validate_uncertainty_metrics(valid_metrics)
        
        # Should pass all validations
        assert validation['total_uncertainty_reasonable']
        assert validation['quantum_variance_valid']
        assert validation['statistical_variance_valid']
        assert validation['confidence_intervals_valid']
        assert validation['overall_valid']
    
    def test_propagate_uncertainty_mean(self):
        """Test uncertainty propagation for mean operation."""
        measurements = [
            UncertaintyMetrics(quantum_variance=0.01, statistical_variance=0.005),
            UncertaintyMetrics(quantum_variance=0.015, statistical_variance=0.008),
            UncertaintyMetrics(quantum_variance=0.012, statistical_variance=0.006)
        ]
        
        combined = self.quantifier.propagate_uncertainty(measurements, 'mean')
        
        # Verify bounds
        assert 0 <= combined.quantum_variance <= 1
        assert 0 <= combined.statistical_variance <= 1
        
        # Mean should reduce uncertainty
        avg_quantum = np.mean([m.quantum_variance for m in measurements])
        assert combined.quantum_variance <= avg_quantum
    
    def test_propagate_uncertainty_sum(self):
        """Test uncertainty propagation for sum operation."""
        measurements = [
            UncertaintyMetrics(quantum_variance=0.01, statistical_variance=0.005),
            UncertaintyMetrics(quantum_variance=0.01, statistical_variance=0.005)
        ]
        
        combined = self.quantifier.propagate_uncertainty(measurements, 'sum')
        
        # For sum, uncertainties add in quadrature
        expected_quantum = np.sqrt(0.01**2 + 0.01**2)
        expected_stat = np.sqrt(0.005**2 + 0.005**2)
        
        assert abs(combined.quantum_variance - expected_quantum) < 1e-10
        assert abs(combined.statistical_variance - expected_stat) < 1e-10
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty measurement data
        empty_metrics = self.quantifier.quantify_uncertainty({})
        assert isinstance(empty_metrics, UncertaintyMetrics)
        
        # Zero fidelity
        zero_data = {'fidelity': 0.0, 'shots': 1024, 'circuit_depth': 0}
        zero_metrics = self.quantifier.quantify_uncertainty(zero_data)
        assert isinstance(zero_metrics, UncertaintyMetrics)
        
        # Perfect fidelity
        perfect_data = {'fidelity': 1.0, 'shots': 1024, 'circuit_depth': 5}
        perfect_metrics = self.quantifier.quantify_uncertainty(perfect_data)
        assert isinstance(perfect_metrics, UncertaintyMetrics)


class TestMultimodalUncertaintyQuantifier:
    """Test cases for MultimodalUncertaintyQuantifier."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.quantifier = MultimodalUncertaintyQuantifier()
        
        self.sample_cross_modal_fidelities = {
            'text_clinical': 0.85,
            'text_image': 0.75,
            'clinical_image': 0.8
        }
        
        self.sample_metadata = {
            'shots': 1024,
            'circuit_depth': 12,
            'entanglement_measure': 0.7
        }
    
    def test_initialization(self):
        """Test proper initialization."""
        assert self.quantifier is not None
        assert hasattr(self.quantifier, 'fidelity_quantifier')
        assert isinstance(self.quantifier.fidelity_quantifier, 
                         QuantumFidelityUncertaintyQuantifier)
    
    def test_quantify_multimodal_uncertainty(self):
        """Test multimodal uncertainty quantification."""
        overall_fidelity = 0.82
        
        uncertainty_results = self.quantifier.quantify_multimodal_uncertainty(
            overall_fidelity,
            self.sample_cross_modal_fidelities,
            self.sample_metadata
        )
        
        # Verify structure
        assert 'overall' in uncertainty_results
        assert 'text_clinical' in uncertainty_results
        assert 'text_image' in uncertainty_results
        assert 'clinical_image' in uncertainty_results
        
        # Verify all are UncertaintyMetrics
        for key, metrics in uncertainty_results.items():
            assert isinstance(metrics, UncertaintyMetrics)
    
    def test_assess_uncertainty_quality(self):
        """Test uncertainty quality assessment."""
        # Create sample uncertainty metrics
        metrics = UncertaintyMetrics(
            quantum_variance=0.01,
            statistical_variance=0.005,
            confidence_intervals={
                'ci_95': {'wilson_width': 0.1},
                'ci_99': {'wilson_width': 0.15}
            },
            uncertainty_validation={
                'total_uncertainty_reasonable': True,
                'quantum_variance_valid': True,
                'statistical_variance_valid': False  # One failure
            }
        )
        
        quality_scores = self.quantifier.assess_uncertainty_quality(metrics)
        
        # Verify structure
        assert 'precision_quality' in quality_scores
        assert 'confidence_quality' in quality_scores
        assert 'validation_quality' in quality_scores
        assert 'overall_quality' in quality_scores
        
        # Verify ranges
        for score in quality_scores.values():
            assert 0 <= score <= 1
        
        # Validation quality should reflect the 2/3 success rate
        assert abs(quality_scores['validation_quality'] - 2/3) < 0.01
    
    def test_quality_assessment_edge_cases(self):
        """Test quality assessment with edge cases."""
        # Perfect metrics
        perfect_metrics = UncertaintyMetrics(
            quantum_variance=0.0,
            statistical_variance=0.0,
            confidence_intervals={
                'ci_95': {'wilson_width': 0.0}
            },
            uncertainty_validation={'all_good': True}
        )
        
        perfect_quality = self.quantifier.assess_uncertainty_quality(perfect_metrics)
        
        # Should have high quality scores
        assert perfect_quality['precision_quality'] == 1.0
        assert perfect_quality['confidence_quality'] == 1.0
        assert perfect_quality['validation_quality'] == 1.0
        
        # Poor metrics
        poor_metrics = UncertaintyMetrics(
            quantum_variance=1.0,
            statistical_variance=1.0,
            confidence_intervals={
                'ci_95': {'wilson_width': 1.0}
            },
            uncertainty_validation={'all_bad': False}
        )
        
        poor_quality = self.quantifier.assess_uncertainty_quality(poor_metrics)
        
        # Should have low quality scores
        assert poor_quality['precision_quality'] == 0.0
        assert poor_quality['confidence_quality'] == 0.0
        assert poor_quality['validation_quality'] == 0.0
    
    def test_missing_data_handling(self):
        """Test handling of missing or incomplete data."""
        # Missing cross-modal fidelities
        uncertainty_results = self.quantifier.quantify_multimodal_uncertainty(
            0.8, {}, self.sample_metadata
        )
        
        # Should still have overall result
        assert 'overall' in uncertainty_results
        assert len(uncertainty_results) == 1  # Only overall, no cross-modal
        
        # Missing metadata
        uncertainty_results = self.quantifier.quantify_multimodal_uncertainty(
            0.8, self.sample_cross_modal_fidelities, {}
        )
        
        # Should still work with defaults
        assert 'overall' in uncertainty_results
        assert len(uncertainty_results) == 4  # Overall + 3 cross-modal


if __name__ == '__main__':
    pytest.main([__file__])
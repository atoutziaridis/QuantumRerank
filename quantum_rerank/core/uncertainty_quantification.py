"""
Uncertainty Quantification for Quantum Multimodal Similarity.

This module implements uncertainty quantification from quantum measurements
and statistical analysis as specified in QMMR-03 task.

Based on:
- QMMR-03 uncertainty quantification requirements
- Quantum measurement statistics
- Confidence intervals from quantum fidelity
- Statistical analysis for quantum uncertainty
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyMetrics:
    """Container for quantum uncertainty measurements."""
    
    # Quantum uncertainty measures
    quantum_variance: float = 0.0
    measurement_uncertainty: float = 0.0
    coherence_uncertainty: float = 0.0
    entanglement_uncertainty: float = 0.0
    
    # Statistical uncertainty measures
    statistical_variance: float = 0.0
    shot_noise_uncertainty: float = 0.0
    
    # Confidence intervals
    confidence_intervals: Dict[str, Dict[str, float]] = None
    
    # Uncertainty sources breakdown
    systematic_uncertainty: float = 0.0
    random_uncertainty: float = 0.0
    
    # Validation metrics
    uncertainty_validation: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.confidence_intervals is None:
            self.confidence_intervals = {}
        if self.uncertainty_validation is None:
            self.uncertainty_validation = {}
    
    def get_total_uncertainty(self) -> float:
        """Calculate total uncertainty from all sources."""
        # Combine quantum and statistical uncertainties
        quantum_total = np.sqrt(
            self.quantum_variance + 
            self.measurement_uncertainty**2 + 
            self.coherence_uncertainty**2 + 
            self.entanglement_uncertainty**2
        )
        
        statistical_total = np.sqrt(
            self.statistical_variance + 
            self.shot_noise_uncertainty**2
        )
        
        # Combine in quadrature
        total_uncertainty = np.sqrt(quantum_total**2 + statistical_total**2)
        
        return float(np.clip(total_uncertainty, 0, 1))


class UncertaintyQuantifier(ABC):
    """Abstract base class for uncertainty quantification methods."""
    
    @abstractmethod
    def quantify_uncertainty(self, 
                           measurement_data: Dict[str, any],
                           context: Optional[Dict] = None) -> UncertaintyMetrics:
        """Quantify uncertainty from measurement data."""
        pass


class QuantumFidelityUncertaintyQuantifier(UncertaintyQuantifier):
    """
    Uncertainty quantification for quantum fidelity measurements.
    
    Implements comprehensive uncertainty analysis for quantum multimodal
    similarity measurements including quantum and statistical sources.
    """
    
    def __init__(self, confidence_levels: List[float] = None):
        """
        Initialize quantum fidelity uncertainty quantifier.
        
        Args:
            confidence_levels: List of confidence levels for intervals (e.g., [0.95, 0.99])
        """
        self.confidence_levels = confidence_levels or [0.90, 0.95, 0.99]
        self.bootstrap_samples = 1000  # For bootstrap confidence intervals
        
        # Physical constants and parameters
        self.planck_constant = 1.0  # Normalized
        self.measurement_efficiency = 0.95  # Typical quantum measurement efficiency
        
        logger.info(f"QuantumFidelityUncertaintyQuantifier initialized with "
                   f"confidence levels: {self.confidence_levels}")
    
    def quantify_uncertainty(self, 
                           measurement_data: Dict[str, any],
                           context: Optional[Dict] = None) -> UncertaintyMetrics:
        """
        Quantify uncertainty from quantum fidelity measurement data.
        
        Args:
            measurement_data: Dictionary containing measurement results
            context: Additional context for uncertainty calculation
            
        Returns:
            UncertaintyMetrics with comprehensive uncertainty analysis
        """
        try:
            # Extract measurement data
            fidelity = measurement_data.get('fidelity', 0.0)
            shots = measurement_data.get('shots', 1024)
            counts = measurement_data.get('counts', {})
            circuit_depth = measurement_data.get('circuit_depth', 0)
            entanglement_measure = measurement_data.get('entanglement_measure', 0.0)
            
            # Initialize metrics
            metrics = UncertaintyMetrics()
            
            # Quantum uncertainty components
            metrics.quantum_variance = self._compute_quantum_variance(fidelity, circuit_depth)
            metrics.measurement_uncertainty = self._compute_measurement_uncertainty(
                fidelity, self.measurement_efficiency
            )
            metrics.coherence_uncertainty = self._compute_coherence_uncertainty(
                fidelity, circuit_depth
            )
            metrics.entanglement_uncertainty = self._compute_entanglement_uncertainty(
                entanglement_measure, circuit_depth
            )
            
            # Statistical uncertainty components
            metrics.statistical_variance = self._compute_statistical_variance(fidelity, shots)
            metrics.shot_noise_uncertainty = self._compute_shot_noise_uncertainty(fidelity, shots)
            
            # Confidence intervals
            metrics.confidence_intervals = self._compute_confidence_intervals(
                fidelity, shots, counts
            )
            
            # Uncertainty source breakdown
            metrics.systematic_uncertainty = self._compute_systematic_uncertainty(
                circuit_depth, entanglement_measure
            )
            metrics.random_uncertainty = self._compute_random_uncertainty(fidelity, shots)
            
            # Validation
            metrics.uncertainty_validation = self._validate_uncertainty_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Uncertainty quantification failed: {e}")
            return UncertaintyMetrics()
    
    def _compute_quantum_variance(self, fidelity: float, circuit_depth: int) -> float:
        """Compute quantum variance based on quantum mechanics principles."""
        # Quantum variance increases with circuit depth due to gate errors
        depth_factor = 1 + 0.01 * circuit_depth  # 1% error per gate layer
        
        # Quantum Fisher information scaling
        quantum_fisher_info = 4 * fidelity * (1 - fidelity)  # For qubit systems
        
        if quantum_fisher_info > 0:
            quantum_variance = depth_factor / quantum_fisher_info
        else:
            quantum_variance = 1.0  # Maximum uncertainty
        
        return float(np.clip(quantum_variance, 0, 1))
    
    def _compute_measurement_uncertainty(self, fidelity: float, efficiency: float) -> float:
        """Compute uncertainty from quantum measurement process."""
        # Measurement uncertainty due to finite detection efficiency
        detection_uncertainty = (1 - efficiency) * fidelity
        
        # Heisenberg uncertainty contribution
        heisenberg_uncertainty = np.sqrt(fidelity * (1 - fidelity)) / 2
        
        # Combine uncertainties
        measurement_uncertainty = np.sqrt(
            detection_uncertainty**2 + heisenberg_uncertainty**2
        )
        
        return float(np.clip(measurement_uncertainty, 0, 1))
    
    def _compute_coherence_uncertainty(self, fidelity: float, circuit_depth: int) -> float:
        """Compute uncertainty from quantum decoherence effects."""
        # Decoherence increases with circuit depth
        decoherence_rate = 0.005 * circuit_depth  # 0.5% decoherence per layer
        
        # Coherence uncertainty affects high-fidelity measurements more
        coherence_uncertainty = decoherence_rate * fidelity
        
        return float(np.clip(coherence_uncertainty, 0, 1))
    
    def _compute_entanglement_uncertainty(self, entanglement: float, circuit_depth: int) -> float:
        """Compute uncertainty from entanglement degradation."""
        # Entanglement uncertainty depends on entanglement strength and circuit complexity
        if entanglement > 0:
            # More entangled states are more fragile
            fragility_factor = entanglement * (1 + 0.02 * circuit_depth)
            entanglement_uncertainty = fragility_factor * 0.1  # Scale factor
        else:
            entanglement_uncertainty = 0.0
        
        return float(np.clip(entanglement_uncertainty, 0, 1))
    
    def _compute_statistical_variance(self, fidelity: float, shots: int) -> float:
        """Compute statistical variance from finite sampling."""
        # Binomial variance for measurement outcomes
        p = (fidelity + 1) / 2  # Convert fidelity to probability
        binomial_variance = p * (1 - p) / shots
        
        # Convert back to fidelity scale
        statistical_variance = 4 * binomial_variance  # Factor of 4 for fidelity scaling
        
        return float(np.clip(statistical_variance, 0, 1))
    
    def _compute_shot_noise_uncertainty(self, fidelity: float, shots: int) -> float:
        """Compute shot noise uncertainty."""
        # Standard shot noise scaling
        shot_noise = 1 / np.sqrt(shots)
        
        # Scale by fidelity sensitivity
        fidelity_sensitivity = 2 * np.sqrt(fidelity * (1 - fidelity))
        
        shot_noise_uncertainty = shot_noise * fidelity_sensitivity
        
        return float(np.clip(shot_noise_uncertainty, 0, 1))
    
    def _compute_confidence_intervals(self, 
                                    fidelity: float, 
                                    shots: int,
                                    counts: Dict[str, int]) -> Dict[str, Dict[str, float]]:
        """Compute confidence intervals for fidelity measurement."""
        confidence_intervals = {}
        
        # Convert fidelity to probability scale for statistics
        p = (fidelity + 1) / 2
        
        for conf_level in self.confidence_levels:
            # Normal approximation confidence interval
            z_score = stats.norm.ppf((1 + conf_level) / 2)
            se = np.sqrt(p * (1 - p) / shots)
            
            # Confidence interval in probability scale
            p_lower = max(0, p - z_score * se)
            p_upper = min(1, p + z_score * se)
            
            # Convert back to fidelity scale
            f_lower = max(0, 2 * p_lower - 1)
            f_upper = min(1, 2 * p_upper - 1)
            
            # Wilson score interval (more accurate for small samples)
            wilson_center = (p + z_score**2 / (2 * shots)) / (1 + z_score**2 / shots)
            wilson_width = z_score * np.sqrt(p * (1 - p) / shots + z_score**2 / (4 * shots**2)) / (1 + z_score**2 / shots)
            
            w_lower = max(0, 2 * (wilson_center - wilson_width) - 1)
            w_upper = min(1, 2 * (wilson_center + wilson_width) - 1)
            
            confidence_intervals[f'ci_{int(conf_level*100)}'] = {
                'normal_lower': f_lower,
                'normal_upper': f_upper,
                'normal_width': f_upper - f_lower,
                'wilson_lower': w_lower,
                'wilson_upper': w_upper,
                'wilson_width': w_upper - w_lower
            }
        
        # Bootstrap confidence intervals if count data available
        if counts and sum(counts.values()) == shots:
            bootstrap_intervals = self._bootstrap_confidence_intervals(counts, shots)
            
            for conf_level, interval in bootstrap_intervals.items():
                if conf_level in confidence_intervals:
                    confidence_intervals[conf_level].update(interval)
        
        return confidence_intervals
    
    def _bootstrap_confidence_intervals(self, 
                                      counts: Dict[str, int], 
                                      shots: int) -> Dict[str, Dict[str, float]]:
        """Compute bootstrap confidence intervals."""
        # Extract measurement results
        success_count = counts.get('0', 0)  # Assuming '0' is success outcome
        
        bootstrap_fidelities = []
        
        # Generate bootstrap samples
        for _ in range(self.bootstrap_samples):
            # Resample with replacement
            bootstrap_successes = np.random.binomial(shots, success_count / shots)
            bootstrap_p = bootstrap_successes / shots
            bootstrap_fidelity = 2 * bootstrap_p - 1  # Convert to fidelity scale
            bootstrap_fidelities.append(bootstrap_fidelity)
        
        bootstrap_fidelities = np.array(bootstrap_fidelities)
        
        # Compute percentile-based confidence intervals
        bootstrap_intervals = {}
        
        for conf_level in self.confidence_levels:
            alpha = 1 - conf_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            b_lower = np.percentile(bootstrap_fidelities, lower_percentile)
            b_upper = np.percentile(bootstrap_fidelities, upper_percentile)
            
            bootstrap_intervals[f'ci_{int(conf_level*100)}'] = {
                'bootstrap_lower': float(np.clip(b_lower, 0, 1)),
                'bootstrap_upper': float(np.clip(b_upper, 0, 1)),
                'bootstrap_width': float(b_upper - b_lower)
            }
        
        return bootstrap_intervals
    
    def _compute_systematic_uncertainty(self, circuit_depth: int, entanglement: float) -> float:
        """Compute systematic uncertainty from circuit and hardware limitations."""
        # Circuit depth contributes to systematic errors
        depth_systematic = 0.005 * circuit_depth  # 0.5% per layer
        
        # Entanglement operations are more prone to systematic errors
        entanglement_systematic = 0.02 * entanglement  # 2% for full entanglement
        
        # Hardware calibration errors
        calibration_systematic = 0.01  # 1% baseline systematic error
        
        total_systematic = np.sqrt(
            depth_systematic**2 + 
            entanglement_systematic**2 + 
            calibration_systematic**2
        )
        
        return float(np.clip(total_systematic, 0, 1))
    
    def _compute_random_uncertainty(self, fidelity: float, shots: int) -> float:
        """Compute random uncertainty from measurement statistics."""
        # Random uncertainty is primarily from shot noise
        shot_noise = 1 / np.sqrt(shots)
        
        # Scale by measurement sensitivity
        sensitivity = 2 * np.sqrt(fidelity * (1 - fidelity))
        
        random_uncertainty = shot_noise * sensitivity
        
        return float(np.clip(random_uncertainty, 0, 1))
    
    def _validate_uncertainty_metrics(self, metrics: UncertaintyMetrics) -> Dict[str, bool]:
        """Validate uncertainty metrics for consistency and physical bounds."""
        validation = {}
        
        # Check if total uncertainty is reasonable
        total_uncertainty = metrics.get_total_uncertainty()
        validation['total_uncertainty_reasonable'] = 0 <= total_uncertainty <= 1
        
        # Check if quantum variance is within bounds
        validation['quantum_variance_valid'] = 0 <= metrics.quantum_variance <= 1
        
        # Check if statistical variance is consistent
        validation['statistical_variance_valid'] = 0 <= metrics.statistical_variance <= 1
        
        # Check if confidence intervals are properly ordered
        ci_valid = True
        for ci_level, ci_data in metrics.confidence_intervals.items():
            for method in ['normal', 'wilson', 'bootstrap']:
                lower_key = f'{method}_lower'
                upper_key = f'{method}_upper'
                if lower_key in ci_data and upper_key in ci_data:
                    if ci_data[lower_key] > ci_data[upper_key]:
                        ci_valid = False
        
        validation['confidence_intervals_valid'] = ci_valid
        
        # Check if systematic and random uncertainties are reasonable
        validation['systematic_uncertainty_valid'] = 0 <= metrics.systematic_uncertainty <= 0.5
        validation['random_uncertainty_valid'] = 0 <= metrics.random_uncertainty <= 1
        
        # Overall validation
        validation['overall_valid'] = all(validation.values())
        
        return validation
    
    def propagate_uncertainty(self, 
                            measurements: List[UncertaintyMetrics],
                            operation: str = 'mean') -> UncertaintyMetrics:
        """
        Propagate uncertainty through mathematical operations.
        
        Args:
            measurements: List of uncertainty metrics to combine
            operation: Type of operation ('mean', 'sum', 'product')
            
        Returns:
            Combined uncertainty metrics
        """
        if not measurements:
            return UncertaintyMetrics()
        
        if len(measurements) == 1:
            return measurements[0]
        
        combined = UncertaintyMetrics()
        
        if operation == 'mean':
            # For mean, uncertainties combine in quadrature and scale by 1/sqrt(n)
            n = len(measurements)
            
            # Combine quantum uncertainties
            quantum_vars = [m.quantum_variance for m in measurements]
            combined.quantum_variance = np.mean(quantum_vars) / np.sqrt(n)
            
            # Combine statistical uncertainties
            stat_vars = [m.statistical_variance for m in measurements]
            combined.statistical_variance = np.mean(stat_vars) / n
            
        elif operation == 'sum':
            # For sum, uncertainties add in quadrature
            quantum_vars = [m.quantum_variance for m in measurements]
            combined.quantum_variance = np.sqrt(np.sum(np.array(quantum_vars)**2))
            
            stat_vars = [m.statistical_variance for m in measurements]
            combined.statistical_variance = np.sqrt(np.sum(np.array(stat_vars)**2))
            
        elif operation == 'product':
            # For product, relative uncertainties add in quadrature
            # This is an approximation for small uncertainties
            total_uncertainties = [m.get_total_uncertainty() for m in measurements]
            combined_relative = np.sqrt(np.sum(np.array(total_uncertainties)**2))
            
            # Distribute back to quantum and statistical components proportionally
            avg_quantum_frac = np.mean([m.quantum_variance / m.get_total_uncertainty() 
                                      for m in measurements if m.get_total_uncertainty() > 0])
            
            combined.quantum_variance = combined_relative * avg_quantum_frac
            combined.statistical_variance = combined_relative * (1 - avg_quantum_frac)
        
        return combined


class MultimodalUncertaintyQuantifier:
    """
    Uncertainty quantification for multimodal quantum similarity measurements.
    
    Handles uncertainty propagation across different modalities and
    quantum entanglement effects on uncertainty.
    """
    
    def __init__(self):
        self.fidelity_quantifier = QuantumFidelityUncertaintyQuantifier()
        
        logger.info("MultimodalUncertaintyQuantifier initialized")
    
    def quantify_multimodal_uncertainty(self,
                                      overall_fidelity: float,
                                      cross_modal_fidelities: Dict[str, float],
                                      measurement_metadata: Dict[str, any]) -> Dict[str, UncertaintyMetrics]:
        """
        Quantify uncertainty for multimodal quantum measurements.
        
        Args:
            overall_fidelity: Overall multimodal fidelity
            cross_modal_fidelities: Individual cross-modal fidelities
            measurement_metadata: Metadata from quantum measurements
            
        Returns:
            Dictionary of uncertainty metrics for each component
        """
        uncertainty_results = {}
        
        # Overall uncertainty
        overall_data = {
            'fidelity': overall_fidelity,
            **measurement_metadata
        }
        uncertainty_results['overall'] = self.fidelity_quantifier.quantify_uncertainty(overall_data)
        
        # Cross-modal uncertainties
        for modality_pair, fidelity in cross_modal_fidelities.items():
            modal_data = {
                'fidelity': fidelity,
                'shots': measurement_metadata.get('shots', 1024),
                'circuit_depth': measurement_metadata.get('circuit_depth', 0),
                'entanglement_measure': measurement_metadata.get('entanglement_measure', 0.0)
            }
            uncertainty_results[modality_pair] = self.fidelity_quantifier.quantify_uncertainty(modal_data)
        
        return uncertainty_results
    
    def assess_uncertainty_quality(self, uncertainty_metrics: UncertaintyMetrics) -> Dict[str, float]:
        """
        Assess the quality of uncertainty quantification.
        
        Args:
            uncertainty_metrics: Uncertainty metrics to assess
            
        Returns:
            Quality assessment scores
        """
        quality_scores = {}
        
        # Precision quality (smaller uncertainties are better)
        total_uncertainty = uncertainty_metrics.get_total_uncertainty()
        quality_scores['precision_quality'] = max(0, 1 - total_uncertainty)
        
        # Confidence quality (narrow confidence intervals are better)
        if uncertainty_metrics.confidence_intervals:
            avg_ci_width = np.mean([
                ci_data.get('wilson_width', 1.0)
                for ci_data in uncertainty_metrics.confidence_intervals.values()
            ])
            quality_scores['confidence_quality'] = max(0, 1 - avg_ci_width)
        else:
            quality_scores['confidence_quality'] = 0.0
        
        # Validation quality (all validations passing is better)
        if uncertainty_metrics.uncertainty_validation:
            valid_count = sum(uncertainty_metrics.uncertainty_validation.values())
            total_count = len(uncertainty_metrics.uncertainty_validation)
            quality_scores['validation_quality'] = valid_count / total_count if total_count > 0 else 0
        else:
            quality_scores['validation_quality'] = 0.0
        
        # Overall quality score
        quality_scores['overall_quality'] = np.mean(list(quality_scores.values()))
        
        return quality_scores
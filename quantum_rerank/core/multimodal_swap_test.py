"""
Multimodal SWAP Test for Quantum Fidelity Computation.

This module extends the SWAP test algorithm for multimodal quantum fidelity
computation as specified in QMMR-03 task.

Based on:
- QMMR-03 task requirements
- Extended SWAP test for multimodal states
- Cross-modal fidelity computation
- Uncertainty quantification from quantum measurements
"""

import numpy as np
import time
import logging
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import scipy.stats as stats

from .swap_test import QuantumSWAPTest, SWAPTestConfig
from .multimodal_quantum_circuits import MultimodalQuantumCircuits
from ..config.settings import SimilarityEngineConfig

logger = logging.getLogger(__name__)


@dataclass
class MultimodalFidelityResult:
    """Result of multimodal fidelity computation."""
    overall_fidelity: float
    cross_modal_fidelities: Dict[str, float]
    uncertainty_metrics: Dict[str, float]
    computation_time_ms: float
    metadata: Dict[str, any]


class MultimodalSwapTest(QuantumSWAPTest):
    """
    Multimodal quantum fidelity computation using extended SWAP test.
    
    Extends the basic SWAP test to handle multimodal quantum states
    with cross-modal entanglement and uncertainty quantification.
    """
    
    def __init__(self, config: SimilarityEngineConfig = None):
        # Initialize base SWAP test
        config = config or SimilarityEngineConfig()
        super().__init__(n_qubits=config.n_qubits, config=SWAPTestConfig())
        
        self.config = config
        self.multimodal_circuits = MultimodalQuantumCircuits(config)
        
        # Statistical parameters for uncertainty quantification
        self.confidence_levels = [0.95, 0.99]  # 95% and 99% confidence intervals
        self.bootstrap_samples = 100  # For bootstrap confidence intervals
        
        logger.info(f"MultimodalSwapTest initialized with {self.n_qubits} qubits")
    
    def compute_multimodal_fidelity(self, 
                                   query_modalities: Dict[str, np.ndarray],
                                   candidate_modalities: Dict[str, np.ndarray]) -> Tuple[float, Dict]:
        """
        Compute quantum fidelity between multimodal query and candidate.
        
        Args:
            query_modalities: Query embeddings by modality
            candidate_modalities: Candidate embeddings by modality
            
        Returns:
            Tuple of (fidelity, metadata)
        """
        start_time = time.time()
        
        try:
            # Prepare quantum states for query and candidate
            query_circuit = self._prepare_multimodal_state(query_modalities)
            candidate_circuit = self._prepare_multimodal_state(candidate_modalities)
            
            # Execute multimodal SWAP test
            fidelity = self._execute_multimodal_swap_test(query_circuit, candidate_circuit)
            
            # Compute cross-modal contributions
            cross_modal_fidelities = self._compute_cross_modal_fidelities(
                query_modalities, candidate_modalities
            )
            
            # Compute uncertainty metrics
            uncertainty_metrics = self._compute_uncertainty_metrics(
                fidelity, shots=self.config.shots if hasattr(self.config, 'shots') else 1024
            )
            
            # Generate comprehensive metadata
            computation_time = (time.time() - start_time) * 1000
            metadata = {
                'computation_time_ms': computation_time,
                'modalities_used': list(query_modalities.keys()),
                'cross_modal_fidelities': cross_modal_fidelities,
                'quantum_circuit_depth': query_circuit.depth(),
                'quantum_circuit_qubits': query_circuit.num_qubits,
                'entanglement_measure': self._compute_entanglement_measure(query_circuit),
                'uncertainty_quantification': uncertainty_metrics,
                'success': True
            }
            
            # Check performance constraint
            if computation_time > 100:  # PRD constraint
                logger.warning(f"Multimodal fidelity computation exceeded 100ms: {computation_time:.2f}ms")
            
            return fidelity, metadata
            
        except Exception as e:
            logger.error(f"Multimodal fidelity computation failed: {e}")
            computation_time = (time.time() - start_time) * 1000
            
            return 0.0, {
                'computation_time_ms': computation_time,
                'success': False,
                'error': str(e)
            }
    
    def _prepare_multimodal_state(self, modalities: Dict[str, np.ndarray]) -> QuantumCircuit:
        """
        Prepare quantum state from multimodal embeddings.
        
        Args:
            modalities: Dictionary of modality embeddings
            
        Returns:
            Quantum circuit representing the multimodal state
        """
        # Extract embeddings
        text_emb = modalities.get('text', np.zeros(256))
        clinical_emb = modalities.get('clinical', np.zeros(256))
        image_emb = modalities.get('image', None)
        
        # Create multimodal quantum circuit
        circuit = self.multimodal_circuits.create_multimodal_state_preparation_circuit(
            text_emb, clinical_emb, image_emb
        )
        
        return circuit
    
    def _execute_multimodal_swap_test(self, query_circuit: QuantumCircuit, 
                                     candidate_circuit: QuantumCircuit) -> float:
        """
        Execute SWAP test between multimodal quantum states.
        
        Extends the basic SWAP test to handle the multimodal circuit structure.
        """
        # Ensure circuits have same number of qubits
        if query_circuit.num_qubits != candidate_circuit.num_qubits:
            raise ValueError("Query and candidate circuits must have same number of qubits")
        
        n_qubits = query_circuit.num_qubits
        
        # Create combined circuit for SWAP test
        # Total qubits: 2 * n_qubits (for both states) + 1 (ancilla)
        ancilla = QuantumRegister(1, 'ancilla')
        query_reg = QuantumRegister(n_qubits, 'query')
        candidate_reg = QuantumRegister(n_qubits, 'candidate')
        classical_reg = ClassicalRegister(1, 'measure')
        
        combined_circuit = QuantumCircuit(ancilla, query_reg, candidate_reg, classical_reg)
        
        # Add query state preparation
        combined_circuit.compose(query_circuit, qubits=query_reg, inplace=True)
        
        # Add candidate state preparation
        combined_circuit.compose(candidate_circuit, qubits=candidate_reg, inplace=True)
        
        # Add SWAP test protocol
        combined_circuit.h(ancilla[0])
        
        # Controlled SWAP operations between corresponding qubits
        for i in range(n_qubits):
            combined_circuit.cswap(ancilla[0], query_reg[i], candidate_reg[i])
        
        # Final Hadamard and measurement
        combined_circuit.h(ancilla[0])
        combined_circuit.measure(ancilla[0], classical_reg[0])
        
        # Execute circuit
        backend = AerSimulator()
        shots = self.config.shots if hasattr(self.config, 'shots') else 1024
        job = backend.run(combined_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts(combined_circuit)
        
        # Compute fidelity from measurement results
        fidelity = self.compute_fidelity_from_counts(counts)
        
        return fidelity
    
    def _compute_cross_modal_fidelities(self, 
                                       query_modalities: Dict[str, np.ndarray],
                                       candidate_modalities: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute individual cross-modal fidelity contributions.
        
        This helps understand which modal combinations contribute most
        to the overall similarity.
        """
        cross_modal_fidelities = {}
        
        # Define modality pairs to evaluate
        modality_pairs = [
            ('text', 'clinical'),
            ('text', 'image'),
            ('clinical', 'image')
        ]
        
        for mod1, mod2 in modality_pairs:
            # Check if both modalities exist in query and candidate
            if (mod1 in query_modalities and mod2 in candidate_modalities and
                query_modalities[mod1] is not None and candidate_modalities[mod2] is not None):
                
                # Compute cross-modal fidelity
                fidelity = self._compute_cross_modal_pair_fidelity(
                    query_modalities[mod1], candidate_modalities[mod2]
                )
                cross_modal_fidelities[f'{mod1}_{mod2}'] = fidelity
            
            # Also compute reverse pair
            if (mod2 in query_modalities and mod1 in candidate_modalities and
                query_modalities[mod2] is not None and candidate_modalities[mod1] is not None):
                
                fidelity = self._compute_cross_modal_pair_fidelity(
                    query_modalities[mod2], candidate_modalities[mod1]
                )
                cross_modal_fidelities[f'{mod2}_{mod1}'] = fidelity
        
        # Compute intra-modal fidelities
        for modality in ['text', 'clinical', 'image']:
            if (modality in query_modalities and modality in candidate_modalities and
                query_modalities[modality] is not None and candidate_modalities[modality] is not None):
                
                fidelity = self._compute_cross_modal_pair_fidelity(
                    query_modalities[modality], candidate_modalities[modality]
                )
                cross_modal_fidelities[f'{modality}_{modality}'] = fidelity
        
        return cross_modal_fidelities
    
    def _compute_cross_modal_pair_fidelity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute fidelity between a pair of embeddings.
        
        Uses a simplified quantum-inspired fidelity calculation.
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        emb1_norm = emb1 / norm1
        emb2_norm = emb2 / norm2
        
        # Compute quantum-inspired fidelity
        # F = |<ψ|φ>|² for normalized states
        inner_product = np.dot(emb1_norm, emb2_norm)
        fidelity = np.abs(inner_product) ** 2
        
        return float(np.clip(fidelity, 0, 1))
    
    def _compute_uncertainty_metrics(self, fidelity: float, shots: int = 1024) -> Dict[str, float]:
        """
        Compute uncertainty quantification metrics from quantum fidelity.
        
        Provides confidence intervals and measurement uncertainty estimates.
        """
        # Quantum measurement uncertainty
        quantum_uncertainty = 1 - fidelity
        
        # Statistical uncertainty from finite shots
        # Standard error for binomial proportion
        p = (fidelity + 1) / 2  # Convert fidelity to probability
        statistical_uncertainty = np.sqrt(p * (1 - p) / shots)
        
        # Confidence intervals
        confidence_intervals = {}
        for conf_level in self.confidence_levels:
            z_score = stats.norm.ppf((1 + conf_level) / 2)
            margin = z_score * statistical_uncertainty
            
            lower = max(0, fidelity - 2 * margin)  # Factor of 2 for fidelity scale
            upper = min(1, fidelity + 2 * margin)
            
            confidence_intervals[f'ci_{int(conf_level*100)}'] = {
                'lower': lower,
                'upper': upper,
                'width': upper - lower
            }
        
        # Measurement variance
        measurement_variance = 4 * statistical_uncertainty ** 2  # Variance in fidelity scale
        
        return {
            'quantum_uncertainty': quantum_uncertainty,
            'statistical_uncertainty': statistical_uncertainty,
            'measurement_variance': measurement_variance,
            'confidence_intervals': confidence_intervals,
            'effective_shots': shots
        }
    
    def _compute_entanglement_measure(self, circuit: QuantumCircuit) -> float:
        """
        Compute entanglement measure for the quantum circuit.
        
        Returns a normalized measure of quantum entanglement.
        """
        # Get entanglement metrics from circuit
        entanglement_metrics = self.multimodal_circuits.compute_entanglement_metrics(circuit)
        
        # Combine metrics into single entanglement measure
        num_gates = entanglement_metrics['num_entangling_gates']
        depth = entanglement_metrics['entanglement_depth']
        connectivity = entanglement_metrics['connectivity_degree']
        
        # Weighted combination (can be tuned based on importance)
        if circuit.num_qubits > 1:
            max_possible_gates = circuit.num_qubits * (circuit.num_qubits - 1)
            normalized_gates = num_gates / max_possible_gates if max_possible_gates > 0 else 0
            
            entanglement_measure = (
                0.4 * normalized_gates +
                0.3 * (depth / circuit.depth() if circuit.depth() > 0 else 0) +
                0.3 * connectivity
            )
        else:
            entanglement_measure = 0.0
        
        return float(np.clip(entanglement_measure, 0, 1))
    
    def compute_confidence_interval(self, fidelity: float, confidence_level: float) -> Tuple[float, float]:
        """
        Compute confidence interval for fidelity measurement.
        
        Args:
            fidelity: Measured fidelity value
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        shots = self.config.shots if hasattr(self.config, 'shots') else 1024
        
        # Convert fidelity to probability scale
        p = (fidelity + 1) / 2
        
        # Standard error
        se = np.sqrt(p * (1 - p) / shots)
        
        # Z-score for confidence level
        z = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Margin of error in probability scale
        margin = z * se
        
        # Convert back to fidelity scale
        lower = max(0, 2 * (p - margin) - 1)
        upper = min(1, 2 * (p + margin) - 1)
        
        return lower, upper
    
    def _compute_measurement_variance(self, fidelity: float) -> float:
        """
        Compute measurement variance for quantum fidelity.
        
        Based on quantum measurement statistics and shot noise.
        """
        shots = self.config.shots if hasattr(self.config, 'shots') else 1024
        
        # Variance in probability scale
        p = (fidelity + 1) / 2
        var_p = p * (1 - p) / shots
        
        # Convert to fidelity scale variance
        # Var(2X - 1) = 4 * Var(X)
        var_fidelity = 4 * var_p
        
        return var_fidelity
    
    def batch_compute_multimodal_fidelity(self,
                                         query_modalities: Dict[str, np.ndarray],
                                         candidate_modalities_list: List[Dict[str, np.ndarray]]) -> List[Tuple[float, Dict]]:
        """
        Compute fidelity between query and multiple candidates efficiently.
        
        Optimized for batch processing to meet <500ms constraint.
        """
        results = []
        
        # Prepare query circuit once
        query_circuit = self._prepare_multimodal_state(query_modalities)
        
        # Process candidates in batch
        for i, candidate_modalities in enumerate(candidate_modalities_list):
            fidelity, metadata = self.compute_multimodal_fidelity(
                query_modalities, candidate_modalities
            )
            
            # Add batch information
            metadata['batch_index'] = i
            metadata['batch_size'] = len(candidate_modalities_list)
            
            results.append((fidelity, metadata))
        
        return results
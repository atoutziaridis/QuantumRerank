"""
Quantum Entanglement Metrics for Multimodal Similarity Assessment.

This module provides utilities for measuring and analyzing quantum entanglement
in multimodal quantum circuits as specified in QMMR-03 task.

Based on:
- QMMR-03 quantum entanglement requirements
- Cross-modal relationship measurement
- Quantum advantage assessment
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, entropy
from qiskit.quantum_info.operators import SparsePauliOp

logger = logging.getLogger(__name__)


@dataclass
class EntanglementMetrics:
    """Container for quantum entanglement measurements."""
    
    # Bipartite entanglement measures
    von_neumann_entropy: float = 0.0
    linear_entropy: float = 0.0
    concurrence: float = 0.0
    
    # Multipartite entanglement measures
    three_tangle: float = 0.0
    global_entanglement: float = 0.0
    
    # Circuit-based measures
    entangling_gate_count: int = 0
    entanglement_depth: int = 0
    connectivity_measure: float = 0.0
    
    # Cross-modal specific measures
    text_clinical_entanglement: float = 0.0
    text_image_entanglement: float = 0.0
    clinical_image_entanglement: float = 0.0
    
    # Quantum advantage indicators
    quantum_coherence: float = 0.0
    superposition_measure: float = 0.0
    
    def get_overall_entanglement_score(self) -> float:
        """Calculate overall entanglement score from individual measures."""
        # Weighted combination of different entanglement measures
        weights = {
            'von_neumann': 0.3,
            'concurrence': 0.2,
            'global': 0.2,
            'cross_modal': 0.2,
            'coherence': 0.1
        }
        
        cross_modal_avg = np.mean([
            self.text_clinical_entanglement,
            self.text_image_entanglement,
            self.clinical_image_entanglement
        ])
        
        overall_score = (
            weights['von_neumann'] * self.von_neumann_entropy +
            weights['concurrence'] * self.concurrence +
            weights['global'] * self.global_entanglement +
            weights['cross_modal'] * cross_modal_avg +
            weights['coherence'] * self.quantum_coherence
        )
        
        return np.clip(overall_score, 0, 1)


class QuantumEntanglementAnalyzer:
    """
    Analyzes quantum entanglement in multimodal quantum circuits.
    
    Provides comprehensive entanglement measurement and analysis tools
    for assessing quantum advantages in multimodal similarity computation.
    """
    
    def __init__(self):
        self.analysis_cache = {}  # Cache for expensive computations
        self.cache_enabled = True
        
        logger.info("QuantumEntanglementAnalyzer initialized")
    
    def analyze_circuit_entanglement(self, circuit: QuantumCircuit) -> EntanglementMetrics:
        """
        Comprehensive entanglement analysis of quantum circuit.
        
        Args:
            circuit: Quantum circuit to analyze
            
        Returns:
            EntanglementMetrics with detailed entanglement measures
        """
        try:
            # Check cache
            circuit_hash = self._get_circuit_hash(circuit)
            if self.cache_enabled and circuit_hash in self.analysis_cache:
                return self.analysis_cache[circuit_hash]
            
            metrics = EntanglementMetrics()
            
            # Get statevector if possible
            try:
                statevector = Statevector.from_instruction(circuit)
                metrics = self._analyze_statevector_entanglement(statevector, circuit.num_qubits)
            except Exception as e:
                logger.warning(f"Could not compute statevector entanglement: {e}")
                # Fall back to circuit-based analysis
                metrics = self._analyze_circuit_structure(circuit)
            
            # Add circuit-based measures
            circuit_metrics = self._analyze_circuit_structure(circuit)
            metrics.entangling_gate_count = circuit_metrics.entangling_gate_count
            metrics.entanglement_depth = circuit_metrics.entanglement_depth
            metrics.connectivity_measure = circuit_metrics.connectivity_measure
            
            # Cache result
            if self.cache_enabled:
                self.analysis_cache[circuit_hash] = metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Entanglement analysis failed: {e}")
            return EntanglementMetrics()  # Return empty metrics
    
    def _analyze_statevector_entanglement(self, statevector: Statevector, num_qubits: int) -> EntanglementMetrics:
        """
        Analyze entanglement using quantum state information.
        
        Args:
            statevector: Quantum statevector
            num_qubits: Number of qubits in the system
            
        Returns:
            EntanglementMetrics from statevector analysis
        """
        metrics = EntanglementMetrics()
        
        # Bipartite entanglement measures
        if num_qubits >= 2:
            # Text-Clinical entanglement (qubits 0-1)
            metrics.text_clinical_entanglement = self._compute_bipartite_entanglement(
                statevector, [0], list(range(1, num_qubits))
            )
            
            # Von Neumann entropy for first bipartition
            metrics.von_neumann_entropy = self._compute_von_neumann_entropy(
                statevector, [0]
            )
            
            # Linear entropy
            metrics.linear_entropy = self._compute_linear_entropy(
                statevector, [0]
            )
            
            # Concurrence for two-qubit case
            if num_qubits == 2:
                metrics.concurrence = self._compute_concurrence(statevector)
        
        # Three-qubit entanglement measures
        if num_qubits >= 3:
            # Text-Image entanglement (qubits 0-2)
            metrics.text_image_entanglement = self._compute_bipartite_entanglement(
                statevector, [0], [2]
            )
            
            # Clinical-Image entanglement (qubits 1-2)
            metrics.clinical_image_entanglement = self._compute_bipartite_entanglement(
                statevector, [1], [2]
            )
            
            # Three-tangle for genuine tripartite entanglement
            metrics.three_tangle = self._compute_three_tangle(statevector)
        
        # Global entanglement measure
        metrics.global_entanglement = self._compute_global_entanglement(statevector)
        
        # Quantum coherence
        metrics.quantum_coherence = self._compute_quantum_coherence(statevector)
        
        # Superposition measure
        metrics.superposition_measure = self._compute_superposition_measure(statevector)
        
        return metrics
    
    def _analyze_circuit_structure(self, circuit: QuantumCircuit) -> EntanglementMetrics:
        """
        Analyze entanglement from circuit structure.
        
        Args:
            circuit: Quantum circuit to analyze
            
        Returns:
            EntanglementMetrics from circuit structure
        """
        metrics = EntanglementMetrics()
        
        # Count entangling gates
        entangling_gates = ['cx', 'cy', 'cz', 'crx', 'cry', 'crz', 'ccx', 'cswap', 'cu1', 'cu3']
        
        entangling_positions = []
        qubit_connections = set()
        
        for instruction, qargs, _ in circuit.data:
            if instruction.name in entangling_gates:
                metrics.entangling_gate_count += 1
                
                # Track position for depth calculation
                entangling_positions.append(circuit.data.index((instruction, qargs, _)))
                
                # Track qubit connections
                if len(qargs) >= 2:
                    qubit_pair = tuple(sorted([qargs[0]._index, qargs[1]._index]))
                    qubit_connections.add(qubit_pair)
        
        # Calculate entanglement depth
        if entangling_positions:
            unique_layers = set()
            for pos in entangling_positions:
                # Approximate layer by position
                layer = pos // max(1, circuit.num_qubits)
                unique_layers.add(layer)
            metrics.entanglement_depth = len(unique_layers)
        
        # Calculate connectivity measure
        if circuit.num_qubits > 1:
            max_connections = circuit.num_qubits * (circuit.num_qubits - 1) // 2
            metrics.connectivity_measure = len(qubit_connections) / max_connections
        
        # Estimate cross-modal entanglement from connections
        if circuit.num_qubits >= 2:
            # Text-Clinical (0-1)
            if (0, 1) in qubit_connections:
                metrics.text_clinical_entanglement = 0.8  # High if directly connected
            
            # Text-Image (0-2)
            if circuit.num_qubits >= 3 and (0, 2) in qubit_connections:
                metrics.text_image_entanglement = 0.8
            
            # Clinical-Image (1-2)
            if circuit.num_qubits >= 3 and (1, 2) in qubit_connections:
                metrics.clinical_image_entanglement = 0.8
        
        return metrics
    
    def _compute_bipartite_entanglement(self, statevector: Statevector, 
                                      subsystem_a: List[int], 
                                      subsystem_b: List[int]) -> float:
        """Compute bipartite entanglement between two subsystems."""
        try:
            # Get reduced density matrix for subsystem A
            rho_a = partial_trace(statevector, subsystem_b)
            
            # Compute von Neumann entropy
            entropy_value = entropy(rho_a, base=2)
            
            # Normalize to [0, 1]
            max_entropy = min(len(subsystem_a), len(subsystem_b))
            normalized_entropy = entropy_value / max_entropy if max_entropy > 0 else 0
            
            return min(normalized_entropy, 1.0)
            
        except Exception as e:
            logger.warning(f"Bipartite entanglement computation failed: {e}")
            return 0.0
    
    def _compute_von_neumann_entropy(self, statevector: Statevector, subsystem: List[int]) -> float:
        """Compute von Neumann entropy of a subsystem."""
        try:
            # Get reduced density matrix
            other_qubits = [i for i in range(statevector.num_qubits) if i not in subsystem]
            if not other_qubits:
                return 0.0  # No entanglement for single system
            
            rho = partial_trace(statevector, other_qubits)
            entropy_value = entropy(rho, base=2)
            
            # Normalize by maximum possible entropy
            max_entropy = len(subsystem)
            return entropy_value / max_entropy if max_entropy > 0 else 0
            
        except Exception as e:
            logger.warning(f"Von Neumann entropy computation failed: {e}")
            return 0.0
    
    def _compute_linear_entropy(self, statevector: Statevector, subsystem: List[int]) -> float:
        """Compute linear entropy of a subsystem."""
        try:
            # Get reduced density matrix
            other_qubits = [i for i in range(statevector.num_qubits) if i not in subsystem]
            if not other_qubits:
                return 0.0
            
            rho = partial_trace(statevector, other_qubits)
            
            # Linear entropy = 1 - Tr(ρ²)
            rho_squared = rho @ rho
            trace_rho_squared = np.trace(rho_squared.data)
            linear_entropy = 1 - np.real(trace_rho_squared)
            
            # Normalize by maximum possible linear entropy
            d = 2 ** len(subsystem)
            max_linear_entropy = (d - 1) / d
            
            return linear_entropy / max_linear_entropy if max_linear_entropy > 0 else 0
            
        except Exception as e:
            logger.warning(f"Linear entropy computation failed: {e}")
            return 0.0
    
    def _compute_concurrence(self, statevector: Statevector) -> float:
        """Compute concurrence for two-qubit systems."""
        try:
            if statevector.num_qubits != 2:
                return 0.0
            
            # Get statevector as array
            psi = statevector.data
            
            # Compute concurrence using the formula for two qubits
            # C = 2|α₀₀α₁₁ - α₀₁α₁₀|
            alpha_00 = psi[0]  # |00⟩
            alpha_01 = psi[1]  # |01⟩
            alpha_10 = psi[2]  # |10⟩
            alpha_11 = psi[3]  # |11⟩
            
            concurrence = 2 * abs(alpha_00 * alpha_11 - alpha_01 * alpha_10)
            
            return min(concurrence, 1.0)
            
        except Exception as e:
            logger.warning(f"Concurrence computation failed: {e}")
            return 0.0
    
    def _compute_three_tangle(self, statevector: Statevector) -> float:
        """Compute three-tangle for three-qubit systems."""
        try:
            if statevector.num_qubits != 3:
                return 0.0
            
            # Simplified three-tangle calculation
            # For a more precise calculation, would need to compute residual entanglement
            psi = statevector.data
            
            # Check for GHZ-like states (high three-tangle)
            # |000⟩ + |111⟩ type states have high three-tangle
            ghz_component = abs(psi[0] * psi[7])  # |000⟩ and |111⟩ components
            
            return min(2 * ghz_component, 1.0)
            
        except Exception as e:
            logger.warning(f"Three-tangle computation failed: {e}")
            return 0.0
    
    def _compute_global_entanglement(self, statevector: Statevector) -> float:
        """Compute global entanglement measure."""
        try:
            # Global entanglement based on total correlation
            n_qubits = statevector.num_qubits
            
            if n_qubits <= 1:
                return 0.0
            
            # Average bipartite entanglement across all possible bipartitions
            total_entanglement = 0.0
            count = 0
            
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    entanglement = self._compute_bipartite_entanglement(
                        statevector, [i], [j]
                    )
                    total_entanglement += entanglement
                    count += 1
            
            return total_entanglement / count if count > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Global entanglement computation failed: {e}")
            return 0.0
    
    def _compute_quantum_coherence(self, statevector: Statevector) -> float:
        """Compute quantum coherence measure."""
        try:
            # Coherence based on off-diagonal elements in computational basis
            psi = statevector.data
            n = len(psi)
            
            # Sum of absolute values of off-diagonal density matrix elements
            coherence = 0.0
            for i in range(n):
                for j in range(n):
                    if i != j:
                        coherence += abs(psi[i] * np.conj(psi[j]))
            
            # Normalize
            max_coherence = n * (n - 1) / 2
            return coherence / max_coherence if max_coherence > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Quantum coherence computation failed: {e}")
            return 0.0
    
    def _compute_superposition_measure(self, statevector: Statevector) -> float:
        """Compute superposition measure."""
        try:
            # Measure based on how far the state is from being classical
            psi = statevector.data
            
            # Classical states have only one non-zero amplitude
            # Superposition measure = 1 - max(|αᵢ|²)
            probabilities = np.abs(psi) ** 2
            max_probability = np.max(probabilities)
            
            superposition = 1 - max_probability
            
            return superposition
            
        except Exception as e:
            logger.warning(f"Superposition measure computation failed: {e}")
            return 0.0
    
    def _get_circuit_hash(self, circuit: QuantumCircuit) -> str:
        """Generate hash for circuit caching."""
        import hashlib
        
        # Create string representation of circuit
        circuit_str = f"{circuit.num_qubits}_{circuit.depth()}_{circuit.size()}"
        
        # Add gate sequence
        gate_sequence = "_".join([
            f"{instr.name}_{len(qargs)}" 
            for instr, qargs, _ in circuit.data
        ])
        
        full_str = f"{circuit_str}_{gate_sequence}"
        return hashlib.md5(full_str.encode()).hexdigest()
    
    def compare_entanglement_methods(self, circuit: QuantumCircuit) -> Dict[str, float]:
        """
        Compare different entanglement measures for the same circuit.
        
        Useful for understanding which measures are most relevant
        for multimodal quantum similarity assessment.
        """
        metrics = self.analyze_circuit_entanglement(circuit)
        
        comparison = {
            'von_neumann_entropy': metrics.von_neumann_entropy,
            'linear_entropy': metrics.linear_entropy,
            'concurrence': metrics.concurrence,
            'three_tangle': metrics.three_tangle,
            'global_entanglement': metrics.global_entanglement,
            'quantum_coherence': metrics.quantum_coherence,
            'superposition_measure': metrics.superposition_measure,
            'overall_score': metrics.get_overall_entanglement_score()
        }
        
        return comparison
    
    def clear_cache(self):
        """Clear analysis cache."""
        self.analysis_cache.clear()
        logger.info("Entanglement analysis cache cleared")
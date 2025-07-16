"""
Multimodal Quantum Circuit Generator for Medical Data.

This module implements quantum circuits for multimodal medical data processing
with entanglement-based fusion as specified in QMMR-03 task.

Based on:
- QMMR-03 task requirements
- PRD circuit constraints (≤15 gate depth, ≤4 qubits)
- Quantum entanglement for cross-modal relationships
- Quantum-inspired compression techniques
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Union
import logging
from dataclasses import dataclass

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector

from ..config.settings import SimilarityEngineConfig

logger = logging.getLogger(__name__)


@dataclass
class MultimodalQuantumState:
    """Quantum state representation for multimodal medical data."""
    
    # Quantum state components
    text_amplitude: np.ndarray
    clinical_amplitude: np.ndarray
    image_amplitude: Optional[np.ndarray] = None
    
    # Entanglement structure
    cross_modal_entanglement: Dict[str, float] = None
    uncertainty_superposition: float = 0.0
    
    # Quantum circuit metadata
    circuit_depth: int = 0
    qubit_count: int = 0
    gate_operations: List[str] = None
    
    def __post_init__(self):
        if self.cross_modal_entanglement is None:
            self.cross_modal_entanglement = {}
        if self.gate_operations is None:
            self.gate_operations = []


class MultimodalQuantumCircuits:
    """
    Generate quantum circuits for multimodal medical similarity computation.
    
    Implements quantum entanglement-based fusion for cross-modal relationships
    while respecting PRD circuit constraints.
    """
    
    def __init__(self, config: SimilarityEngineConfig = None):
        self.config = config or SimilarityEngineConfig()
        self.max_qubits = min(4, self.config.n_qubits)  # PRD constraint
        self.max_depth = 15  # PRD constraint
        
        # Quantum circuit optimization parameters
        self.entanglement_depth = 2  # Number of entangling layers
        self.rotation_precision = 3  # Decimal places for angle precision
        
        logger.info(f"MultimodalQuantumCircuits initialized: "
                   f"{self.max_qubits} qubits, max depth {self.max_depth}")
    
    def create_multimodal_state_preparation_circuit(self, 
                                                   text_emb: np.ndarray,
                                                   clinical_emb: np.ndarray,
                                                   image_emb: Optional[np.ndarray] = None) -> QuantumCircuit:
        """
        Create quantum circuit for multimodal state preparation with entanglement.
        
        Args:
            text_emb: Text embedding vector
            clinical_emb: Clinical data embedding vector
            image_emb: Optional image embedding vector
            
        Returns:
            Quantum circuit with multimodal state preparation
        """
        # Determine number of qubits based on available modalities
        n_modalities = 2 if image_emb is None else 3
        n_qubits = min(self.max_qubits, n_modalities + 1)  # +1 for ancilla if needed
        
        # Initialize quantum circuit
        circuit = QuantumCircuit(n_qubits, name="multimodal_state_prep")
        
        # Prepare individual modal states
        text_angles = self._compute_rotation_angles(text_emb, target_qubits=1)
        clinical_angles = self._compute_rotation_angles(clinical_emb, target_qubits=1)
        
        # Text modality preparation (qubit 0)
        self._apply_rotation_gates(circuit, 0, text_angles[:2])
        
        # Clinical modality preparation (qubit 1)
        self._apply_rotation_gates(circuit, 1, clinical_angles[:2])
        
        # Create entanglement between modalities
        circuit.cx(0, 1)  # Text-Clinical entanglement
        
        # Add controlled rotation for cross-modal correlation
        correlation_angle = self._compute_cross_modal_correlation(text_emb, clinical_emb)
        circuit.cry(correlation_angle, 0, 1)
        
        # Add image modality if present and enough qubits
        if image_emb is not None and n_qubits >= 3:
            image_angles = self._compute_rotation_angles(image_emb, target_qubits=1)
            self._apply_rotation_gates(circuit, 2, image_angles[:2])
            
            # Create three-way entanglement
            circuit.cx(1, 2)  # Clinical-Image entanglement
            circuit.cx(0, 2)  # Text-Image entanglement
            
            # Add three-qubit entangling gate for stronger correlation
            if n_qubits >= 3:
                circuit.ccx(0, 1, 2)  # Toffoli gate
        
        # Add uncertainty encoding if we have an ancilla qubit
        if n_qubits == self.max_qubits and n_qubits > n_modalities:
            ancilla_idx = n_qubits - 1
            uncertainty = self._compute_uncertainty_measure(text_emb, clinical_emb, image_emb)
            circuit.ry(uncertainty * np.pi, ancilla_idx)
            
            # Entangle ancilla with data qubits
            for i in range(n_modalities):
                circuit.cx(ancilla_idx, i)
        
        # Validate circuit constraints
        if circuit.depth() > self.max_depth:
            logger.warning(f"Circuit depth {circuit.depth()} exceeds maximum {self.max_depth}")
            # Apply circuit optimization
            circuit = self._optimize_circuit_depth(circuit)
        
        return circuit
    
    def create_parameterized_multimodal_circuit(self, n_modalities: int = 2) -> Tuple[QuantumCircuit, ParameterVector]:
        """
        Create parameterized quantum circuit for multimodal data.
        
        Args:
            n_modalities: Number of modalities (2 or 3)
            
        Returns:
            Tuple of (circuit, parameter_vector)
        """
        n_qubits = min(self.max_qubits, n_modalities)
        
        # Create parameter vector
        n_params_per_qubit = 3  # θ, φ, λ for full rotation
        n_params = n_qubits * n_params_per_qubit + n_qubits * (n_qubits - 1) // 2  # + entanglement params
        params = ParameterVector('θ', n_params)
        
        # Initialize circuit
        circuit = QuantumCircuit(n_qubits, name="param_multimodal")
        
        param_idx = 0
        
        # Single-qubit rotations for each modality
        for q in range(n_qubits):
            circuit.ry(params[param_idx], q)
            circuit.rz(params[param_idx + 1], q)
            circuit.ry(params[param_idx + 2], q)
            param_idx += 3
        
        # Entangling gates between modalities
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                circuit.cx(i, j)
                circuit.crz(params[param_idx], i, j)
                param_idx += 1
        
        return circuit, params
    
    def _compute_rotation_angles(self, embedding: np.ndarray, target_qubits: int) -> List[float]:
        """
        Compute rotation angles for quantum state preparation from embeddings.
        
        Uses quantum-inspired compression to map high-dimensional embeddings
        to quantum rotation angles.
        """
        # Compress embedding to required dimensions
        compressed_dim = 2 ** target_qubits
        compressed_emb = self._compress_embedding(embedding, compressed_dim)
        
        # Normalize for quantum state preparation
        norm = np.linalg.norm(compressed_emb)
        if norm > 0:
            normalized_emb = compressed_emb / norm
        else:
            normalized_emb = np.zeros_like(compressed_emb)
        
        # Convert to rotation angles using Bloch sphere parameterization
        angles = []
        for i in range(min(len(normalized_emb), 4)):  # Limit to prevent circuit explosion
            # Map to [0, π] for θ angles and [0, 2π] for φ angles
            theta = 2 * np.arccos(np.clip(np.abs(normalized_emb[i]), 0, 1))
            phi = np.angle(normalized_emb[i] if isinstance(normalized_emb[i], complex) else 0)
            
            angles.append(round(theta, self.rotation_precision))
            angles.append(round(phi, self.rotation_precision))
        
        return angles
    
    def _compress_embedding(self, embedding: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Compress high-dimensional embedding to target dimension.
        
        Uses quantum-inspired techniques for efficient compression.
        """
        current_dim = len(embedding)
        
        if current_dim <= target_dim:
            # Pad with zeros if needed
            return np.pad(embedding, (0, target_dim - current_dim), mode='constant')
        
        # Quantum-inspired compression using principal components
        # For now, use simple averaging chunks (can be improved with learned compression)
        chunk_size = current_dim // target_dim
        compressed = np.zeros(target_dim)
        
        for i in range(target_dim):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < target_dim - 1 else current_dim
            compressed[i] = np.mean(embedding[start_idx:end_idx])
        
        return compressed
    
    def _apply_rotation_gates(self, circuit: QuantumCircuit, qubit: int, angles: List[float]):
        """Apply rotation gates to prepare quantum state."""
        if len(angles) >= 1:
            circuit.ry(angles[0], qubit)
        if len(angles) >= 2:
            circuit.rz(angles[1], qubit)
    
    def _compute_cross_modal_correlation(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute correlation angle between two modalities."""
        # Compute cosine similarity
        dot_product = np.dot(emb1, emb2)
        norms = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        
        if norms > 0:
            cosine_sim = dot_product / norms
            # Map to rotation angle [0, π]
            angle = np.arccos(np.clip(cosine_sim, -1, 1))
        else:
            angle = np.pi / 2  # Orthogonal if one embedding is zero
        
        return round(angle, self.rotation_precision)
    
    def _compute_uncertainty_measure(self, text_emb: np.ndarray, 
                                   clinical_emb: np.ndarray,
                                   image_emb: Optional[np.ndarray]) -> float:
        """Compute uncertainty measure for multimodal data."""
        # Calculate variance across modalities as uncertainty proxy
        embeddings = [text_emb, clinical_emb]
        if image_emb is not None:
            embeddings.append(image_emb)
        
        # Compute pairwise distances
        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                distances.append(dist)
        
        # Normalize uncertainty to [0, 1]
        if distances:
            uncertainty = np.std(distances) / (np.mean(distances) + 1e-8)
            uncertainty = np.clip(uncertainty, 0, 1)
        else:
            uncertainty = 0.0
        
        return uncertainty
    
    def _optimize_circuit_depth(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Optimize circuit to reduce depth while preserving functionality.
        
        Uses circuit optimization techniques to meet PRD constraints.
        """
        try:
            from qiskit.transpiler import PassManager
            from qiskit.transpiler.passes import (
                Optimize1qGates,
                CXCancellation,
                CommutativeCancellation
            )
            
            # Create optimization pass manager
            pm = PassManager([
                Optimize1qGates(),
                CXCancellation(),
                CommutativeCancellation()
            ])
            
            optimized_circuit = pm.run(circuit)
            
            if optimized_circuit.depth() <= self.max_depth:
                logger.info(f"Circuit optimized: {circuit.depth()} -> {optimized_circuit.depth()}")
                return optimized_circuit
            else:
                logger.warning("Circuit optimization insufficient, applying aggressive reduction")
                return self._aggressive_circuit_reduction(circuit)
                
        except Exception as e:
            logger.error(f"Circuit optimization failed: {e}")
            return circuit
    
    def _aggressive_circuit_reduction(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Aggressively reduce circuit depth by simplifying operations."""
        # Create new circuit with essential operations only
        reduced_circuit = QuantumCircuit(circuit.num_qubits, name=circuit.name + "_reduced")
        
        # Keep only essential gates
        essential_gate_types = ['ry', 'cx', 'cry']
        gate_count = 0
        
        for instruction, qargs, cargs in circuit.data:
            if instruction.name in essential_gate_types and gate_count < self.max_depth - 2:
                reduced_circuit.append(instruction, qargs, cargs)
                gate_count += 1
        
        return reduced_circuit
    
    def compute_entanglement_metrics(self, circuit: QuantumCircuit) -> Dict[str, float]:
        """
        Compute entanglement metrics for the quantum circuit.
        
        Returns various measures of quantum entanglement useful for
        assessing cross-modal relationships.
        """
        metrics = {
            'num_entangling_gates': 0,
            'entanglement_depth': 0,
            'connectivity_degree': 0.0
        }
        
        # Count entangling gates
        entangling_gates = ['cx', 'cy', 'cz', 'crx', 'cry', 'crz', 'ccx']
        entangling_layers = []
        
        for instruction, qargs, cargs in circuit.data:
            if instruction.name in entangling_gates:
                metrics['num_entangling_gates'] += 1
                # Track which layer this gate appears in
                layer = circuit.depth() - circuit.decompose().depth()
                entangling_layers.append(layer)
        
        # Calculate entanglement depth
        if entangling_layers:
            metrics['entanglement_depth'] = len(set(entangling_layers))
        
        # Calculate connectivity degree (how interconnected qubits are)
        if circuit.num_qubits > 1:
            possible_connections = circuit.num_qubits * (circuit.num_qubits - 1) / 2
            actual_connections = len(set((min(q), max(q)) for instruction, qargs, _ in circuit.data
                                       if len(qargs) == 2 and instruction.name in entangling_gates
                                       for q in [(qargs[0]._index, qargs[1]._index)]))
            metrics['connectivity_degree'] = actual_connections / possible_connections if possible_connections > 0 else 0
        
        return metrics
    
    def validate_circuit_constraints(self, circuit: QuantumCircuit) -> Dict[str, bool]:
        """
        Validate circuit against PRD constraints.
        
        Returns validation results for various constraints.
        """
        validation = {
            'depth_ok': circuit.depth() <= self.max_depth,
            'qubits_ok': circuit.num_qubits <= self.max_qubits,
            'depth_value': circuit.depth(),
            'qubits_value': circuit.num_qubits,
            'gate_count': circuit.size()
        }
        
        # Additional validation for multimodal requirements
        entanglement_metrics = self.compute_entanglement_metrics(circuit)
        validation['has_entanglement'] = entanglement_metrics['num_entangling_gates'] > 0
        validation['entanglement_metrics'] = entanglement_metrics
        
        return validation
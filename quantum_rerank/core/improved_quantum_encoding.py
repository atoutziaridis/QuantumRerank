"""
Improved Quantum Encoding Methods for QRF-01 Fidelity Saturation Fix.

This module implements alternative quantum encoding methods to address the
identified fidelity saturation issues, based on the debug analysis findings.

Key Issues Addressed:
1. Amplitude encoding loses 98% semantic information (768D -> 16D)
2. Quantum states become too similar after normalization  
3. Circuit simulation failures with statevector extraction
4. SWAP test implementation errors for orthogonal states

Based on:
- Task QRF-01 findings and recommendations
- Papers: "A quantum binary classifier based on cosine similarity"
- Papers: "A Quantum Geometric Model of Similarity"
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

logger = logging.getLogger(__name__)


@dataclass
class EncodingResult:
    """Result from quantum encoding operation."""
    success: bool
    circuit: Optional[QuantumCircuit] = None
    statevector: Optional[Statevector] = None
    encoding_method: str = ""
    information_preservation: float = 0.0
    metadata: Dict[str, Any] = None
    error: Optional[str] = None


class ImprovedQuantumEncoder:
    """
    Improved quantum encoding methods that preserve semantic information.
    
    Implements fixes for QRF-01 fidelity saturation issues:
    1. Feature selection before encoding
    2. Multiple encoding strategies
    3. Information-preserving preprocessing
    4. Robust circuit simulation
    """
    
    def __init__(self, n_qubits: int = 4):
        """Initialize improved quantum encoder."""
        self.n_qubits = n_qubits
        self.max_amplitudes = 2 ** n_qubits
        self.simulator = AerSimulator(method='statevector')
        
        logger.info(f"Initialized ImprovedQuantumEncoder with {n_qubits} qubits")
    
    def preprocess_embedding_with_feature_selection(self, 
                                                   embedding: np.ndarray,
                                                   method: str = 'variance') -> Tuple[np.ndarray, Dict]:
        """
        Preprocess embedding with intelligent feature selection.
        
        Addresses the information loss issue identified in QRF-01.
        
        Args:
            embedding: Original high-dimensional embedding (768D)
            method: Feature selection method ('variance', 'magnitude', 'pca')
            
        Returns:
            Tuple of (selected_features, metadata)
        """
        original_dim = len(embedding)
        target_dim = self.max_amplitudes
        
        if original_dim <= target_dim:
            # No reduction needed
            return embedding.copy(), {
                'method': 'no_reduction',
                'original_dim': original_dim,
                'target_dim': target_dim,
                'information_loss': 0.0
            }
        
        if method == 'variance':
            # Select features with highest variance contribution
            # Use sliding window to preserve local structure
            window_size = original_dim // target_dim
            selected_features = []
            
            for i in range(target_dim):
                start_idx = i * window_size
                end_idx = min((i + 1) * window_size, original_dim)
                window = embedding[start_idx:end_idx]
                
                # Select feature with highest absolute value in window
                if len(window) > 0:
                    max_idx = start_idx + np.argmax(np.abs(window))
                    selected_features.append(embedding[max_idx])
                else:
                    selected_features.append(0.0)
            
            selected_embedding = np.array(selected_features)
            
        elif method == 'magnitude':
            # Select top features by magnitude
            indices = np.argsort(np.abs(embedding))[-target_dim:]
            selected_embedding = embedding[indices]
            
        elif method == 'pca':
            # Use PCA-like dimension reduction
            # Simple implementation: select evenly spaced features
            indices = np.linspace(0, original_dim-1, target_dim, dtype=int)
            selected_embedding = embedding[indices]
            
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # Calculate information preservation
        original_var = np.var(embedding)
        selected_var = np.var(selected_embedding)
        information_preservation = min(1.0, selected_var / original_var) if original_var > 0 else 0.0
        
        metadata = {
            'method': method,
            'original_dim': original_dim,
            'target_dim': target_dim,
            'information_preservation': information_preservation,
            'original_variance': original_var,
            'selected_variance': selected_var
        }
        
        logger.debug(f"Feature selection: {method}, preservation: {information_preservation:.3f}")
        
        return selected_embedding, metadata
    
    def create_statevector_from_circuit(self, circuit: QuantumCircuit) -> Optional[np.ndarray]:
        """
        Create statevector from quantum circuit with robust error handling.
        
        Applies QRF-01 fixes for statevector extraction.
        """
        try:
            # Method 1: Direct statevector simulation (most reliable)
            if circuit.num_qubits <= 10:  # Reasonable limit
                statevector = Statevector.from_instruction(circuit)
                return statevector.data
            else:
                logger.warning(f"Circuit too large for statevector simulation: {circuit.num_qubits} qubits")
                return None
                
        except Exception as e1:
            logger.debug(f"Direct statevector failed: {e1}")
            
            try:
                # Method 2: Simulator with statevector method
                clean_circuit = circuit.copy()
                clean_circuit.remove_final_measurements()
                statevector = Statevector.from_instruction(clean_circuit)
                return statevector.data
                    
            except Exception as e2:
                logger.error(f"Failed to create statevector: {e1}, {e2}")
                return None

    def angle_encoding_with_normalization(self, embedding: np.ndarray,
                                        name: str = "angle_encoded") -> EncodingResult:
        """
        Implement angle encoding that preserves relative differences.
        
        Alternative to amplitude encoding that maintains better discrimination.
        Uses QRF-01 fixes for robust statevector extraction.
        
        Args:
            embedding: Preprocessed embedding vector
            name: Circuit name
            
        Returns:
            EncodingResult with circuit and analysis
        """
        try:
            # Ensure we have the right number of features
            if len(embedding) != self.n_qubits:
                # Pad or truncate to match qubit count
                if len(embedding) < self.n_qubits:
                    padded = np.pad(embedding, (0, self.n_qubits - len(embedding)), 'constant')
                else:
                    padded = embedding[:self.n_qubits]
                embedding = padded
            
            # Create quantum circuit
            circuit = QuantumCircuit(self.n_qubits, name=name)
            
            # Improved angle encoding: preserve ranking by using adaptive scaling
            # Scale features to preserve relative magnitudes
            feature_mean = np.mean(embedding)
            feature_std = np.std(embedding)
            
            if feature_std > 0:
                # Standardize features and map to [-π, π] range
                standardized_features = (embedding - feature_mean) / feature_std
                angles = standardized_features * np.pi / 2  # Scale to [-π/2, π/2] for better preservation
            else:
                angles = np.zeros(self.n_qubits)
            
            # Apply RY rotations (angle encoding) with bias to preserve ranking
            for i, angle in enumerate(angles):
                # Add small bias based on feature magnitude to preserve ranking
                magnitude_bias = np.abs(embedding[i]) / (np.max(np.abs(embedding)) + 1e-8) * 0.1
                circuit.ry(angle + magnitude_bias, i)
            
            # Add selective entanglement to capture feature correlations while preserving ranking
            # Use CZ gates which are phase-preserving
            for i in range(self.n_qubits - 1):
                # Only entangle if features are significantly correlated
                if len(embedding) > i+1:
                    feature_correlation = np.abs(embedding[i] * embedding[i+1])
                    if feature_correlation > 0.1:  # Threshold for entanglement
                        circuit.cz(i, i + 1)
            
            # Use robust statevector extraction from QRF-01
            statevector_data = self.create_statevector_from_circuit(circuit)
            
            if statevector_data is not None:
                # Calculate information preservation with ranking awareness
                original_var = np.var(embedding)
                quantum_var = np.var(np.abs(statevector_data))
                info_preservation = min(1.0, quantum_var / original_var) if original_var > 0 else 0.0
                
                metadata = {
                    'angles': angles.tolist(),
                    'circuit_depth': circuit.depth(),
                    'feature_mean': feature_mean,
                    'feature_std': feature_std,
                    'info_preservation': info_preservation,
                    'simulation_success': True,
                    'encoding_strategy': 'ranking_aware_angle'
                }
                
                # Create Statevector object for compatibility
                statevector = Statevector(statevector_data)
                
                return EncodingResult(
                    success=True,
                    circuit=circuit,
                    statevector=statevector,
                    encoding_method='angle_encoding',
                    information_preservation=info_preservation,
                    metadata=metadata
                )
            else:
                logger.error(f"Failed to extract statevector from angle encoding circuit")
                return EncodingResult(
                    success=False,
                    circuit=circuit,
                    encoding_method='angle_encoding',
                    error="Statevector extraction failed",
                    metadata={'simulation_success': False}
                )
        
        except Exception as e:
            logger.error(f"Angle encoding failed: {e}")
            return EncodingResult(
                success=False,
                encoding_method='angle_encoding',
                error=str(e)
            )
    
    def hybrid_amplitude_angle_encoding(self, embedding: np.ndarray,
                                      name: str = "hybrid_encoded") -> EncodingResult:
        """
        Hybrid encoding combining amplitude and angle encoding.
        
        Uses amplitude encoding for magnitude and angle encoding for phase.
        
        Args:
            embedding: Preprocessed embedding vector
            name: Circuit name
            
        Returns:
            EncodingResult with hybrid encoded circuit
        """
        try:
            # Feature selection to match amplitude requirements
            if len(embedding) > self.max_amplitudes:
                selected_embedding, selection_metadata = self.preprocess_embedding_with_feature_selection(
                    embedding, method='variance'
                )
            else:
                selected_embedding = embedding.copy()
                selection_metadata = {'method': 'no_selection'}
            
            # Ensure proper size
            if len(selected_embedding) < self.max_amplitudes:
                selected_embedding = np.pad(
                    selected_embedding, 
                    (0, self.max_amplitudes - len(selected_embedding)), 
                    'constant'
                )
            elif len(selected_embedding) > self.max_amplitudes:
                selected_embedding = selected_embedding[:self.max_amplitudes]
            
            # Create quantum circuit
            circuit = QuantumCircuit(self.n_qubits, name=name)
            
            # Normalize for quantum state with tolerance
            norm = np.linalg.norm(selected_embedding)
            if norm > 0:
                normalized_amplitudes = selected_embedding / norm
                # Check normalization with tolerance for floating point precision
                final_norm = np.linalg.norm(normalized_amplitudes)
                if abs(final_norm - 1.0) > 1e-6:
                    logger.warning(f"Normalization precision: {final_norm}, re-normalizing")
                    normalized_amplitudes = normalized_amplitudes / final_norm
            else:
                normalized_amplitudes = np.ones(self.max_amplitudes) / np.sqrt(self.max_amplitudes)
            
            # Initialize state with amplitude encoding - use tolerance for normalization
            try:
                circuit.initialize(normalized_amplitudes, range(self.n_qubits))
            except Exception as init_error:
                # If initialization fails due to normalization, try manual normalization
                logger.debug(f"Initialize failed: {init_error}, attempting manual normalization")
                manual_norm = normalized_amplitudes / np.linalg.norm(normalized_amplitudes)
                circuit.initialize(manual_norm, range(self.n_qubits))
            
            # Add phase encoding based on original feature relationships
            if len(embedding) > self.n_qubits:
                # Use phase rotations to encode additional information
                phase_features = embedding[self.n_qubits:2*self.n_qubits] if len(embedding) >= 2*self.n_qubits else embedding[-self.n_qubits:]
                phase_range = np.max(phase_features) - np.min(phase_features)
                
                if phase_range > 0:
                    normalized_phases = (phase_features - np.min(phase_features)) / phase_range
                    phase_angles = normalized_phases * np.pi  # Phase range [0, π]
                    
                    for i, phase in enumerate(phase_angles[:self.n_qubits]):
                        circuit.rz(phase, i)
            
            # Use robust statevector extraction from QRF-01
            statevector_data = self.create_statevector_from_circuit(circuit)
            
            if statevector_data is not None:
                # Calculate information preservation
                original_var = np.var(embedding)
                quantum_var = np.var(np.abs(statevector_data))
                info_preservation = min(1.0, quantum_var / original_var) if original_var > 0 else 0.0
                
                metadata = {
                    'amplitude_features': len(selected_embedding),
                    'phase_features': len(embedding) - self.n_qubits if len(embedding) > self.n_qubits else 0,
                    'circuit_depth': circuit.depth(),
                    'info_preservation': info_preservation,
                    'selection_metadata': selection_metadata,
                    'simulation_success': True
                }
                
                # Create Statevector object for compatibility
                statevector = Statevector(statevector_data)
                
                return EncodingResult(
                    success=True,
                    circuit=circuit,
                    statevector=statevector,
                    encoding_method='hybrid_amplitude_angle',
                    information_preservation=info_preservation,
                    metadata=metadata
                )
            else:
                logger.error(f"Failed to extract statevector from hybrid encoding circuit")
                return EncodingResult(
                    success=False,
                    circuit=circuit,
                    encoding_method='hybrid_amplitude_angle',
                    error="Statevector extraction failed",
                    metadata={'simulation_success': False}
                )
        
        except Exception as e:
            logger.error(f"Hybrid encoding failed: {e}")
            return EncodingResult(
                success=False,
                encoding_method='hybrid_amplitude_angle',
                error=str(e)
            )
    
    def multi_scale_encoding(self, embedding: np.ndarray,
                           name: str = "multi_scale") -> EncodingResult:
        """
        Multi-scale encoding that captures features at different resolutions.
        
        Addresses the semantic information loss by encoding at multiple scales.
        
        Args:
            embedding: Original embedding vector
            name: Circuit name
            
        Returns:
            EncodingResult with multi-scale encoded circuit
        """
        try:
            # Create multiple feature representations at different scales
            scales = []
            
            # Scale 1: Global features (coarse-grained)
            global_features, _ = self.preprocess_embedding_with_feature_selection(
                embedding, method='magnitude'
            )
            scales.append(('global', global_features))
            
            # Scale 2: Local features (fine-grained) 
            local_features, _ = self.preprocess_embedding_with_feature_selection(
                embedding, method='variance'
            )
            scales.append(('local', local_features))
            
            # Combine scales with weighted average
            combined_features = np.zeros(self.max_amplitudes)
            total_weight = 0
            
            for scale_name, features in scales:
                weight = 1.0  # Equal weighting for now
                if len(features) >= len(combined_features):
                    combined_features += weight * features[:len(combined_features)]
                else:
                    padded_features = np.pad(features, (0, len(combined_features) - len(features)), 'constant')
                    combined_features += weight * padded_features
                total_weight += weight
            
            combined_features /= total_weight
            
            # Create quantum circuit with combined features
            circuit = QuantumCircuit(self.n_qubits, name=name)
            
            # Normalize for quantum state
            norm = np.linalg.norm(combined_features)
            if norm > 0:
                normalized_amplitudes = combined_features / norm
            else:
                normalized_amplitudes = np.ones(self.max_amplitudes) / np.sqrt(self.max_amplitudes)
            
            # Initialize state
            circuit.initialize(normalized_amplitudes, range(self.n_qubits))
            
            # Add cross-scale correlations
            for i in range(self.n_qubits - 1):
                circuit.cz(i, (i + 1) % self.n_qubits)
            
            # Use robust statevector extraction from QRF-01
            statevector_data = self.create_statevector_from_circuit(circuit)
            
            if statevector_data is not None:
                # Calculate information preservation across scales
                original_var = np.var(embedding)
                quantum_var = np.var(np.abs(statevector_data))
                info_preservation = min(1.0, quantum_var / original_var) if original_var > 0 else 0.0
                
                metadata = {
                    'scales': len(scales),
                    'combined_features': len(combined_features),
                    'circuit_depth': circuit.depth(),
                    'info_preservation': info_preservation,
                    'simulation_success': True
                }
                
                # Create Statevector object for compatibility
                statevector = Statevector(statevector_data)
                
                return EncodingResult(
                    success=True,
                    circuit=circuit,
                    statevector=statevector,
                    encoding_method='multi_scale',
                    information_preservation=info_preservation,
                    metadata=metadata
                )
            else:
                logger.error(f"Failed to extract statevector from multi-scale encoding circuit")
                return EncodingResult(
                    success=False,
                    circuit=circuit,
                    encoding_method='multi_scale',
                    error="Statevector extraction failed",
                    metadata={'simulation_success': False}
                )
        
        except Exception as e:
            logger.error(f"Multi-scale encoding failed: {e}")
            return EncodingResult(
                success=False,
                encoding_method='multi_scale',
                error=str(e)
            )
    
    def distance_preserving_encoding(self, embedding: np.ndarray,
                                   name: str = "distance_preserving") -> EncodingResult:
        """
        Distance-preserving encoding that optimizes for ranking preservation.
        
        Uses a combination of magnitude-aware scaling and correlation-preserving
        entanglement to maintain semantic ordering in quantum state.
        
        Args:
            embedding: Input embedding vector
            name: Circuit name
            
        Returns:
            EncodingResult with distance-preserving encoded circuit
        """
        try:
            # Feature selection with ranking awareness
            selected_embedding, selection_metadata = self.preprocess_embedding_with_feature_selection(
                embedding, method='magnitude'  # Use magnitude selection for ranking preservation
            )
            
            # Ensure proper size
            if len(selected_embedding) != self.n_qubits:
                if len(selected_embedding) < self.n_qubits:
                    selected_embedding = np.pad(selected_embedding, (0, self.n_qubits - len(selected_embedding)), 'constant')
                else:
                    selected_embedding = selected_embedding[:self.n_qubits]
            
            # Create quantum circuit
            circuit = QuantumCircuit(self.n_qubits, name=name)
            
            # Distance-preserving encoding strategy
            # 1. Preserve magnitude ordering through careful angle mapping
            feature_magnitudes = np.abs(selected_embedding)
            magnitude_ranking = np.argsort(feature_magnitudes)[::-1]  # Descending order
            
            # 2. Map features to angles preserving relative distances with better discrimination
            max_magnitude = np.max(feature_magnitudes)
            if max_magnitude > 0:
                # Use square root scaling for better discrimination
                normalized_magnitudes = feature_magnitudes / max_magnitude
                # Map to [0, π/3] range for better discrimination (smaller angles)
                angles = np.sqrt(normalized_magnitudes) * (np.pi / 3)  
                
                # Preserve sign information with reduced range to avoid saturation
                signs = np.sign(selected_embedding)
                signed_angles = angles * signs
            else:
                signed_angles = np.zeros(self.n_qubits)
            
            # 3. Apply rotations in magnitude order to preserve ranking
            for rank, qubit_idx in enumerate(magnitude_ranking):
                if qubit_idx < len(signed_angles):
                    angle = signed_angles[qubit_idx]
                    # Use different rotation axes based on magnitude rank
                    if rank % 3 == 0:
                        circuit.ry(angle, qubit_idx)
                    elif rank % 3 == 1:
                        circuit.rx(angle, qubit_idx)
                    else:
                        circuit.rz(angle, qubit_idx)
            
            # 4. Add minimal entanglement to preserve discrimination
            # Use lighter entanglement to avoid state saturation
            if self.n_qubits >= 2:
                # Only entangle the most significant qubits
                most_significant_qubit = magnitude_ranking[0]
                second_significant_qubit = magnitude_ranking[1] if len(magnitude_ranking) > 1 else 0
                
                # Light entanglement between most significant features
                if most_significant_qubit != second_significant_qubit:
                    circuit.cz(most_significant_qubit, second_significant_qubit)
            
            # Use robust statevector extraction from QRF-01
            statevector_data = self.create_statevector_from_circuit(circuit)
            
            if statevector_data is not None:
                # Calculate information preservation
                original_var = np.var(embedding)
                quantum_var = np.var(np.abs(statevector_data))
                info_preservation = min(1.0, quantum_var / original_var) if original_var > 0 else 0.0
                
                metadata = {
                    'signed_angles': signed_angles.tolist(),
                    'magnitude_ranking': magnitude_ranking.tolist(),
                    'circuit_depth': circuit.depth(),
                    'max_magnitude': max_magnitude,
                    'info_preservation': info_preservation,
                    'selection_metadata': selection_metadata,
                    'simulation_success': True,
                    'encoding_strategy': 'distance_preserving'
                }
                
                # Create Statevector object for compatibility
                statevector = Statevector(statevector_data)
                
                return EncodingResult(
                    success=True,
                    circuit=circuit,
                    statevector=statevector,
                    encoding_method='distance_preserving',
                    information_preservation=info_preservation,
                    metadata=metadata
                )
            else:
                logger.error(f"Failed to extract statevector from distance-preserving encoding circuit")
                return EncodingResult(
                    success=False,
                    circuit=circuit,
                    encoding_method='distance_preserving',
                    error="Statevector extraction failed",
                    metadata={'simulation_success': False}
                )
        
        except Exception as e:
            logger.error(f"Distance-preserving encoding failed: {e}")
            return EncodingResult(
                success=False,
                encoding_method='distance_preserving',
                error=str(e)
            )

    def ranking_optimized_encoding(self, embedding: np.ndarray,
                                 name: str = "ranking_optimized") -> EncodingResult:
        """
        Ranking-optimized encoding that specifically targets ranking preservation.
        
        Uses a conservative approach with minimal saturation to maximize ranking preservation.
        
        Args:
            embedding: Input embedding vector
            name: Circuit name
            
        Returns:
            EncodingResult with ranking-optimized encoded circuit
        """
        try:
            # Use ranking-aware feature selection
            selected_embedding, selection_metadata = self.preprocess_embedding_with_feature_selection(
                embedding, method='magnitude'
            )
            
            # Ensure proper size
            if len(selected_embedding) != self.n_qubits:
                if len(selected_embedding) < self.n_qubits:
                    selected_embedding = np.pad(selected_embedding, (0, self.n_qubits - len(selected_embedding)), 'constant')
                else:
                    selected_embedding = selected_embedding[:self.n_qubits]
            
            # Create quantum circuit
            circuit = QuantumCircuit(self.n_qubits, name=name)
            
            # Ranking-optimized strategy: preserve order through careful scaling
            # 1. Sort features by magnitude to preserve ranking
            feature_magnitudes = np.abs(selected_embedding)
            sorted_indices = np.argsort(feature_magnitudes)
            
            # 2. Apply conservative angle scaling to avoid saturation
            # Use linear scaling with small maximum angle to preserve discrimination
            max_magnitude = np.max(feature_magnitudes)
            if max_magnitude > 0:
                # Scale to [0, π/6] range for maximum discrimination preservation
                angles = (feature_magnitudes / max_magnitude) * (np.pi / 6)
                
                # Apply sign preservation
                signs = np.sign(selected_embedding)
                signed_angles = angles * signs
            else:
                signed_angles = np.zeros(self.n_qubits)
            
            # 3. Apply rotations with ranking-aware gate selection
            for i, angle in enumerate(signed_angles):
                # Use different gates based on ranking to preserve order
                rank = np.where(sorted_indices == i)[0][0]
                if rank < self.n_qubits // 2:  # Top half: use RY
                    circuit.ry(angle, i)
                else:  # Bottom half: use RX for differentiation
                    circuit.rx(angle, i)
            
            # 4. NO entanglement to avoid any state mixing that could hurt ranking
            
            # Use robust statevector extraction from QRF-01
            statevector_data = self.create_statevector_from_circuit(circuit)
            
            if statevector_data is not None:
                # Calculate information preservation
                original_var = np.var(embedding)
                quantum_var = np.var(np.abs(statevector_data))
                info_preservation = min(1.0, quantum_var / original_var) if original_var > 0 else 0.0
                
                metadata = {
                    'signed_angles': signed_angles.tolist(),
                    'sorted_indices': sorted_indices.tolist(),
                    'circuit_depth': circuit.depth(),
                    'max_magnitude': max_magnitude,
                    'info_preservation': info_preservation,
                    'selection_metadata': selection_metadata,
                    'simulation_success': True,
                    'encoding_strategy': 'ranking_optimized'
                }
                
                # Create Statevector object for compatibility
                statevector = Statevector(statevector_data)
                
                return EncodingResult(
                    success=True,
                    circuit=circuit,
                    statevector=statevector,
                    encoding_method='ranking_optimized',
                    information_preservation=info_preservation,
                    metadata=metadata
                )
            else:
                logger.error(f"Failed to extract statevector from ranking-optimized encoding circuit")
                return EncodingResult(
                    success=False,
                    circuit=circuit,
                    encoding_method='ranking_optimized',
                    error="Statevector extraction failed",
                    metadata={'simulation_success': False}
                )
        
        except Exception as e:
            logger.error(f"Ranking-optimized encoding failed: {e}")
            return EncodingResult(
                success=False,
                encoding_method='ranking_optimized',
                error=str(e)
            )

    def encode_embedding(self, embedding: np.ndarray, 
                        method: str = 'angle',
                        name: str = "encoded") -> EncodingResult:
        """
        Main encoding interface that selects the appropriate method.
        
        Args:
            embedding: Input embedding vector
            method: Encoding method ('angle', 'hybrid', 'multi_scale', 'distance_preserving')
            name: Circuit name
            
        Returns:
            EncodingResult with encoded quantum circuit
        """
        if method == 'angle':
            # First do feature selection
            selected_embedding, selection_metadata = self.preprocess_embedding_with_feature_selection(
                embedding, method='variance'
            )
            result = self.angle_encoding_with_normalization(selected_embedding, name)
            if result.metadata:
                result.metadata['feature_selection'] = selection_metadata
            
        elif method == 'hybrid':
            result = self.hybrid_amplitude_angle_encoding(embedding, name)
            
        elif method == 'multi_scale':
            result = self.multi_scale_encoding(embedding, name)
            
        elif method == 'distance_preserving':
            result = self.distance_preserving_encoding(embedding, name)
        
        elif method == 'ranking_optimized':
            result = self.ranking_optimized_encoding(embedding, name)
            
        else:
            return EncodingResult(
                success=False,
                encoding_method=method,
                error=f"Unknown encoding method: {method}"
            )
        
        return result
    
    def compare_encoding_methods(self, embedding: np.ndarray) -> Dict[str, EncodingResult]:
        """
        Compare all encoding methods on the same embedding.
        
        Useful for debugging and method selection.
        
        Args:
            embedding: Input embedding to test
            
        Returns:
            Dictionary mapping method names to results
        """
        methods = ['angle', 'ranking_optimized', 'distance_preserving', 'multi_scale']  # Skip hybrid due to normalization issues
        results = {}
        
        for method in methods:
            logger.info(f"Testing encoding method: {method}")
            result = self.encode_embedding(embedding, method, f"test_{method}")
            results[method] = result
            
            if result.success:
                logger.info(f"{method}: success, info_preservation={result.information_preservation:.3f}")
            else:
                logger.warning(f"{method}: failed - {result.error}")
        
        return results


def test_improved_encoding_on_medical_texts():
    """Test function for improved encoding methods."""
    from quantum_rerank.core.embeddings import EmbeddingProcessor
    
    # Initialize components
    embedding_processor = EmbeddingProcessor()
    encoder = ImprovedQuantumEncoder(n_qubits=4)
    
    # Test texts
    test_texts = [
        "The patient presented with acute myocardial infarction and elevated troponin levels.",
        "Quantum computing leverages quantum mechanical phenomena for computational advantages.",
        "Machine learning algorithms can process large datasets to identify patterns."
    ]
    
    print("Testing Improved Quantum Encoding Methods")
    print("="*50)
    
    for i, text in enumerate(test_texts):
        print(f"\nText {i}: {text[:60]}...")
        
        # Generate embedding
        embedding = embedding_processor.encode_single_text(text)
        print(f"  Original embedding dim: {len(embedding)}")
        
        # Test all encoding methods
        results = encoder.compare_encoding_methods(embedding)
        
        for method, result in results.items():
            if result.success:
                print(f"  {method}: SUCCESS (preservation: {result.information_preservation:.3f})")
                print(f"    Circuit depth: {result.metadata.get('circuit_depth', 'N/A')}")
            else:
                print(f"  {method}: FAILED - {result.error}")
    
    return results


if __name__ == "__main__":
    # Run test if executed directly
    test_results = test_improved_encoding_on_medical_texts()
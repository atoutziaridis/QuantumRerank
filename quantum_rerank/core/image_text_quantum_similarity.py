"""
Image-Text Quantum Similarity Engine.

This module implements quantum similarity computation for medical image-text pairs
using quantum geometric model approaches and cross-modal entanglement
as specified in QMMR-04 task.

Based on:
- QMMR-04 task requirements  
- Quantum geometric model of similarity research
- Cross-modal quantum entanglement for medical data
- Image-text fusion with attention mechanisms
"""

import numpy as np
import torch
import torch.nn as nn
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

from .medical_image_processor import MedicalImageProcessor, MedicalImageProcessingResult
from .multimodal_quantum_circuits import MultimodalQuantumCircuits
from .quantum_entanglement_metrics import QuantumEntanglementAnalyzer
from .uncertainty_quantification import MultimodalUncertaintyQuantifier
from ..config.settings import SimilarityEngineConfig
from ..config.medical_image_config import MedicalImageConfig, CrossModalAttentionConfig

logger = logging.getLogger(__name__)


@dataclass
class ImageTextSimilarityResult:
    """Result of image-text quantum similarity computation."""
    
    # Core similarity
    similarity_score: float = 0.0
    
    # Processing metadata
    computation_time_ms: float = 0.0
    image_processing_time_ms: float = 0.0
    text_processing_time_ms: float = 0.0
    
    # Image metadata
    image_format: str = 'unknown'
    image_modality: str = 'unknown'
    image_quality_score: float = 0.0
    
    # Quantum processing
    quantum_circuit_depth: int = 0
    quantum_circuit_qubits: int = 0
    quantum_fidelity: float = 0.0
    entanglement_measure: float = 0.0
    
    # Cross-modal analysis
    cross_modal_attention_weights: Dict[str, float] = None
    modality_contributions: Dict[str, float] = None
    
    # Uncertainty quantification
    uncertainty_metrics: Dict[str, Any] = None
    confidence_intervals: Dict[str, Dict[str, float]] = None
    
    # Quality indicators
    processing_success: bool = False
    quantum_advantage_score: float = 0.0
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.cross_modal_attention_weights is None:
            self.cross_modal_attention_weights = {}
        if self.modality_contributions is None:
            self.modality_contributions = {}
        if self.uncertainty_metrics is None:
            self.uncertainty_metrics = {}
        if self.confidence_intervals is None:
            self.confidence_intervals = {}
        if self.warnings is None:
            self.warnings = []


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for image-text quantum similarity.
    
    Implements learnable attention weights for optimal fusion of visual
    and textual information in quantum similarity computation.
    """
    
    def __init__(self, config: CrossModalAttentionConfig = None):
        super().__init__()
        self.config = config or CrossModalAttentionConfig()
        
        # Attention networks
        self.image_attention = nn.Linear(
            self.config.image_attention_dim, 
            self.config.image_attention_dim
        )
        self.text_attention = nn.Linear(
            self.config.text_attention_dim, 
            self.config.text_attention_dim
        )
        
        # Combined attention
        self.combined_attention = nn.Linear(
            self.config.combined_attention_dim, 
            2  # Image and text weights
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(self.config.attention_dropout)
        
        # Store attention weights for analysis
        self.latest_attention_weights = None
        
        logger.info(f"CrossModalAttention initialized with {self.config.attention_type} attention")
    
    def forward(self, image_embedding: torch.Tensor, text_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-modal attention to image and text embeddings.
        
        Args:
            image_embedding: Image embedding tensor
            text_embedding: Text embedding tensor
            
        Returns:
            Tuple of (attended_image_embedding, attended_text_embedding)
        """
        # Ensure tensors are 2D (batch_size, embedding_dim)
        if len(image_embedding.shape) == 1:
            image_embedding = image_embedding.unsqueeze(0)
        if len(text_embedding.shape) == 1:
            text_embedding = text_embedding.unsqueeze(0)
        
        # Compute individual attention representations
        image_attn = torch.tanh(self.image_attention(image_embedding))
        text_attn = torch.tanh(self.text_attention(text_embedding))
        
        # Apply dropout
        image_attn = self.dropout(image_attn)
        text_attn = self.dropout(text_attn)
        
        # Combine features for attention weight computation
        if self.config.fusion_method == 'concatenate':
            # Ensure dimensions match for concatenation
            min_dim = min(image_attn.size(-1), text_attn.size(-1))
            image_attn_resized = image_attn[:, :min_dim]
            text_attn_resized = text_attn[:, :min_dim]
            combined_features = torch.cat([image_attn_resized, text_attn_resized], dim=-1)
        elif self.config.fusion_method == 'element_wise':
            min_dim = min(image_attn.size(-1), text_attn.size(-1))
            image_attn_resized = image_attn[:, :min_dim]
            text_attn_resized = text_attn[:, :min_dim]
            combined_features = image_attn_resized * text_attn_resized
        else:  # weighted_sum
            min_dim = min(image_attn.size(-1), text_attn.size(-1))
            image_attn_resized = image_attn[:, :min_dim]
            text_attn_resized = text_attn[:, :min_dim]
            combined_features = 0.5 * image_attn_resized + 0.5 * text_attn_resized
        
        # Compute attention weights
        attention_logits = self.combined_attention(combined_features)
        attention_weights = torch.softmax(attention_logits, dim=-1)
        
        # Store for analysis
        self.latest_attention_weights = attention_weights.detach().cpu().numpy()
        
        # Apply attention weights
        image_weight = attention_weights[:, 0:1]
        text_weight = attention_weights[:, 1:2]
        
        attended_image = image_embedding * image_weight
        attended_text = text_embedding * text_weight
        
        return attended_image, attended_text
    
    def get_attention_weights(self) -> Dict[str, float]:
        """Get latest attention weights for interpretability."""
        if self.latest_attention_weights is None:
            return {'image': 0.5, 'text': 0.5}
        
        # Average across batch dimension
        avg_weights = np.mean(self.latest_attention_weights, axis=0)
        
        return {
            'image': float(avg_weights[0]),
            'text': float(avg_weights[1])
        }


class ImageTextQuantumCircuits:
    """
    Quantum circuit generation for image-text similarity computation.
    
    Creates quantum circuits that encode image-text relationships through
    cross-modal entanglement and quantum superposition.
    """
    
    def __init__(self, config: SimilarityEngineConfig):
        self.config = config
        self.max_qubits = min(4, config.n_qubits)  # PRD constraint
        self.max_depth = 15  # PRD constraint
        
        # Circuit optimization parameters
        self.rotation_precision = 3
        self.entanglement_layers = 2
        
        logger.info(f"ImageTextQuantumCircuits initialized: {self.max_qubits} qubits, max depth {self.max_depth}")
    
    def create_image_text_circuit(self, image_embedding: np.ndarray, text_embedding: np.ndarray) -> QuantumCircuit:
        """
        Create quantum circuit for image-text similarity computation.
        
        Args:
            image_embedding: Image embedding vector
            text_embedding: Text embedding vector
            
        Returns:
            Quantum circuit encoding image-text relationships
        """
        # Use 2 qubits minimum for image-text pairing
        n_qubits = min(self.max_qubits, 2)
        
        # Initialize quantum circuit
        circuit = QuantumCircuit(n_qubits, name="image_text_similarity")
        
        # Encode image information (qubit 0)
        image_angles = self._compute_rotation_angles(image_embedding, target_qubits=1)
        if len(image_angles) >= 2:
            circuit.ry(image_angles[0], 0)
            circuit.rz(image_angles[1], 0)
        
        # Encode text information (qubit 1)
        text_angles = self._compute_rotation_angles(text_embedding, target_qubits=1)
        if len(text_angles) >= 2:
            circuit.ry(text_angles[0], 1)
            circuit.rz(text_angles[1], 1)
        
        # Create cross-modal entanglement
        circuit.cx(0, 1)  # Image-text entanglement
        
        # Add cross-modal correlation
        correlation_angle = self._compute_cross_modal_correlation(image_embedding, text_embedding)
        circuit.cry(correlation_angle, 0, 1)
        
        # Add additional entangling layers if circuit depth allows
        current_depth = circuit.depth()
        if current_depth < self.max_depth - 4 and n_qubits >= 2:
            # Second entanglement layer
            circuit.ry(correlation_angle / 2, 0)
            circuit.ry(correlation_angle / 2, 1)
            circuit.cx(1, 0)  # Reverse entanglement
        
        # Add measurements
        circuit.add_register(ClassicalRegister(n_qubits, 'c'))
        circuit.measure_all()
        
        # Validate circuit constraints
        if circuit.depth() > self.max_depth:
            logger.warning(f"Circuit depth {circuit.depth()} exceeds maximum {self.max_depth}")
            circuit = self._optimize_circuit_depth(circuit)
        
        return circuit
    
    def _compute_rotation_angles(self, embedding: np.ndarray, target_qubits: int) -> List[float]:
        """Compute rotation angles from embedding vector."""
        # Compress embedding to required dimensions
        compressed_dim = 2 * target_qubits  # θ and φ for each qubit
        compressed_emb = self._compress_embedding(embedding, compressed_dim)
        
        # Normalize for quantum state preparation
        norm = np.linalg.norm(compressed_emb)
        if norm > 0:
            normalized_emb = compressed_emb / norm
        else:
            normalized_emb = np.zeros_like(compressed_emb)
        
        # Convert to rotation angles
        angles = []
        for i in range(min(len(normalized_emb), compressed_dim)):
            # Map to [0, π] for θ angles and [0, 2π] for φ angles  
            if i % 2 == 0:  # θ angle
                theta = np.arccos(np.clip(np.abs(normalized_emb[i]), 0, 1))
                angles.append(round(theta, self.rotation_precision))
            else:  # φ angle
                phi = np.angle(normalized_emb[i] if np.iscomplexobj(normalized_emb) else 0)
                angles.append(round(phi, self.rotation_precision))
        
        return angles
    
    def _compress_embedding(self, embedding: np.ndarray, target_dim: int) -> np.ndarray:
        """Compress embedding to target dimension."""
        current_dim = len(embedding)
        
        if current_dim <= target_dim:
            # Pad with zeros if needed
            return np.pad(embedding, (0, target_dim - current_dim), mode='constant')
        
        # Simple chunking compression (could be improved with PCA)
        chunk_size = current_dim // target_dim
        compressed = np.zeros(target_dim)
        
        for i in range(target_dim):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < target_dim - 1 else current_dim
            compressed[i] = np.mean(embedding[start_idx:end_idx])
        
        return compressed
    
    def _compute_cross_modal_correlation(self, image_emb: np.ndarray, text_emb: np.ndarray) -> float:
        """Compute correlation angle between image and text embeddings."""
        # Compute cosine similarity
        dot_product = np.dot(image_emb, text_emb)
        norms = np.linalg.norm(image_emb) * np.linalg.norm(text_emb)
        
        if norms > 0:
            cosine_sim = dot_product / norms
            # Map to rotation angle [0, π]
            angle = np.arccos(np.clip(cosine_sim, -1, 1))
        else:
            angle = np.pi / 2  # Orthogonal if one embedding is zero
        
        return round(angle, self.rotation_precision)
    
    def _optimize_circuit_depth(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Optimize circuit to reduce depth while preserving functionality."""
        try:
            from qiskit.transpiler import PassManager
            from qiskit.transpiler.passes import Optimize1qGates, CXCancellation
            
            # Create optimization pass manager
            pm = PassManager([
                Optimize1qGates(),
                CXCancellation()
            ])
            
            optimized_circuit = pm.run(circuit)
            
            if optimized_circuit.depth() <= self.max_depth:
                logger.info(f"Circuit optimized: {circuit.depth()} -> {optimized_circuit.depth()}")
                return optimized_circuit
            else:
                logger.warning("Circuit optimization insufficient")
                return self._aggressive_circuit_reduction(circuit)
        
        except Exception as e:
            logger.error(f"Circuit optimization failed: {e}")
            return circuit
    
    def _aggressive_circuit_reduction(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Aggressively reduce circuit depth by keeping only essential operations."""
        # Create simplified circuit with essential gates only
        simplified_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits, name=circuit.name + "_reduced")
        
        essential_gates = ['ry', 'rz', 'cx']
        gate_count = 0
        
        for instruction, qargs, cargs in circuit.data:
            if instruction.name in essential_gates and gate_count < self.max_depth - 2:
                simplified_circuit.append(instruction, qargs, cargs)
                gate_count += 1
        
        # Add final measurements
        if circuit.num_clbits > 0:
            simplified_circuit.measure_all()
        
        return simplified_circuit


class ImageTextQuantumSimilarity:
    """
    Quantum similarity computation for medical image-text pairs.
    
    Implements comprehensive quantum similarity using quantum geometric model
    approaches, cross-modal attention, and uncertainty quantification.
    """
    
    def __init__(self, config: SimilarityEngineConfig = None):
        """
        Initialize image-text quantum similarity engine.
        
        Args:
            config: Similarity engine configuration
        """
        self.config = config or SimilarityEngineConfig()
        
        # Initialize components
        self.image_processor = MedicalImageProcessor()
        self.quantum_circuits = ImageTextQuantumCircuits(self.config)
        self.cross_modal_attention = CrossModalAttention()
        self.entanglement_analyzer = QuantumEntanglementAnalyzer()
        self.uncertainty_quantifier = MultimodalUncertaintyQuantifier()
        
        # Performance monitoring
        self.similarity_stats = {
            'total_computations': 0,
            'avg_computation_time_ms': 0.0,
            'avg_image_processing_time_ms': 0.0,
            'avg_quantum_time_ms': 0.0,
            'successful_computations': 0,
            'quantum_advantage_cases': 0
        }
        
        logger.info("ImageTextQuantumSimilarity initialized")
    
    def compute_image_text_similarity(self, 
                                    image_data: Union[str, np.ndarray, Any],
                                    text_embedding: np.ndarray,
                                    text_metadata: Optional[Dict[str, Any]] = None) -> ImageTextSimilarityResult:
        """
        Compute quantum similarity between medical image and text.
        
        Args:
            image_data: Image file path, numpy array, or PIL Image
            text_embedding: Text embedding vector
            text_metadata: Optional metadata about the text
            
        Returns:
            ImageTextSimilarityResult with comprehensive analysis
        """
        start_time = time.time()
        result = ImageTextSimilarityResult()
        
        try:
            # Process image
            image_start = time.time()
            image_result = self.image_processor.process_medical_image(image_data)
            result.image_processing_time_ms = (time.time() - image_start) * 1000
            
            if not image_result.processing_success:
                result.error_message = f"Image processing failed: {image_result.error_message}"
                return result
            
            # Extract image information
            image_embedding = image_result.embedding
            result.image_format = image_result.image_format
            result.image_modality = image_result.modality
            result.image_quality_score = image_result.image_quality_score
            
            # Validate embeddings
            if image_embedding is None or text_embedding is None:
                result.error_message = "Missing embeddings for similarity computation"
                return result
            
            # Convert to tensors for attention mechanism
            image_tensor = torch.tensor(image_embedding, dtype=torch.float32)
            text_tensor = torch.tensor(text_embedding, dtype=torch.float32)
            
            # Apply cross-modal attention
            with torch.no_grad():
                attended_image, attended_text = self.cross_modal_attention(image_tensor, text_tensor)
                attended_image_np = attended_image.squeeze().cpu().numpy()
                attended_text_np = attended_text.squeeze().cpu().numpy()
            
            result.cross_modal_attention_weights = self.cross_modal_attention.get_attention_weights()
            
            # Create quantum circuit for similarity computation
            quantum_start = time.time()
            similarity_circuit = self.quantum_circuits.create_image_text_circuit(
                attended_image_np, attended_text_np
            )
            
            # Execute quantum similarity computation
            quantum_fidelity = self._execute_quantum_similarity(similarity_circuit)
            result.quantum_fidelity = quantum_fidelity
            result.similarity_score = quantum_fidelity
            
            # Analyze quantum circuit
            result.quantum_circuit_depth = similarity_circuit.depth()
            result.quantum_circuit_qubits = similarity_circuit.num_qubits
            
            # Compute entanglement measures
            entanglement_metrics = self.entanglement_analyzer.analyze_circuit_entanglement(similarity_circuit)
            result.entanglement_measure = entanglement_metrics.get_overall_entanglement_score()
            
            quantum_elapsed = (time.time() - quantum_start) * 1000
            
            # Uncertainty quantification
            uncertainty_data = {
                'fidelity': quantum_fidelity,
                'shots': self.config.shots if hasattr(self.config, 'shots') else 1024,
                'circuit_depth': result.quantum_circuit_depth,
                'entanglement_measure': result.entanglement_measure
            }
            
            uncertainty_results = self.uncertainty_quantifier.quantify_multimodal_uncertainty(
                quantum_fidelity,
                {'image_text': quantum_fidelity},
                uncertainty_data
            )
            
            if 'overall' in uncertainty_results:
                result.uncertainty_metrics = {
                    'total_uncertainty': uncertainty_results['overall'].get_total_uncertainty(),
                    'quantum_uncertainty': uncertainty_results['overall'].quantum_variance,
                    'statistical_uncertainty': uncertainty_results['overall'].statistical_variance
                }
                result.confidence_intervals = uncertainty_results['overall'].confidence_intervals
            
            # Assess quantum advantage
            result.quantum_advantage_score = self._assess_quantum_advantage(
                quantum_fidelity, entanglement_metrics, result.cross_modal_attention_weights
            )
            
            # Calculate modality contributions
            result.modality_contributions = self._calculate_modality_contributions(
                image_embedding, text_embedding, result.cross_modal_attention_weights
            )
            
            # Final timing and success
            total_time = (time.time() - start_time) * 1000
            result.computation_time_ms = total_time
            result.text_processing_time_ms = total_time - result.image_processing_time_ms - quantum_elapsed
            result.processing_success = True
            
            # Update statistics
            self._update_similarity_stats(result)
            
            # Check performance constraint
            if total_time > 150:  # PRD constraint from task spec
                result.warnings.append(f"Computation time {total_time:.0f}ms exceeds 150ms constraint")
            
            return result
            
        except Exception as e:
            # Error handling
            total_time = (time.time() - start_time) * 1000
            result.computation_time_ms = total_time
            result.error_message = str(e)
            result.processing_success = False
            
            logger.error(f"Image-text similarity computation failed: {e}")
            return result
    
    def _execute_quantum_similarity(self, circuit: QuantumCircuit) -> float:
        """Execute quantum circuit and compute similarity from measurement statistics."""
        try:
            # Execute circuit using Qiskit Aer simulator
            backend = AerSimulator()
            shots = self.config.shots if hasattr(self.config, 'shots') else 1024
            
            # Run the circuit
            job = backend.run(circuit, shots=shots)
            result = job.result()
            counts = result.get_counts(circuit)
            
            # Compute fidelity from measurement statistics
            total_shots = sum(counts.values())
            
            # For quantum similarity, we measure the overlap between quantum states
            # Higher probability of measuring |00⟩ indicates higher similarity
            zero_state_key = '0' * circuit.num_qubits
            prob_zero = counts.get(zero_state_key, 0) / total_shots
            
            # Convert probability to fidelity score
            # Quantum fidelity ranges from 0 to 1
            fidelity = 2 * prob_zero - 1  # Map [0.5, 1] -> [0, 1]
            fidelity = max(0, fidelity)  # Ensure non-negative
            
            return float(fidelity)
            
        except Exception as e:
            logger.error(f"Quantum similarity execution failed: {e}")
            return 0.0
    
    def _assess_quantum_advantage(self, 
                                 fidelity: float, 
                                 entanglement_metrics, 
                                 attention_weights: Dict[str, float]) -> float:
        """Assess quantum advantage in image-text similarity computation."""
        try:
            # Factor 1: Entanglement strength indicates quantum advantage
            entanglement_advantage = entanglement_metrics.get_overall_entanglement_score()
            
            # Factor 2: Cross-modal attention balance (neither modality dominates completely)
            attention_balance = 1 - abs(attention_weights.get('image', 0.5) - attention_weights.get('text', 0.5))
            
            # Factor 3: Fidelity in the quantum regime (high values indicate quantum processing)
            quantum_regime_score = fidelity if fidelity > 0.5 else 0.0
            
            # Factor 4: Circuit complexity utilization
            circuit_utilization = min(1.0, entanglement_metrics.entangling_gate_count / 4.0)  # Max 4 gates
            
            # Weighted combination
            quantum_advantage = (
                0.4 * entanglement_advantage +
                0.25 * attention_balance +
                0.2 * quantum_regime_score +
                0.15 * circuit_utilization
            )
            
            return float(np.clip(quantum_advantage, 0, 1))
            
        except Exception as e:
            logger.warning(f"Quantum advantage assessment failed: {e}")
            return 0.0
    
    def _calculate_modality_contributions(self, 
                                        image_embedding: np.ndarray,
                                        text_embedding: np.ndarray,
                                        attention_weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate individual modality contributions to similarity."""
        try:
            # Embedding magnitudes (information content)
            image_magnitude = np.linalg.norm(image_embedding)
            text_magnitude = np.linalg.norm(text_embedding)
            
            # Attention weights (learned importance)
            image_attention = attention_weights.get('image', 0.5)
            text_attention = attention_weights.get('text', 0.5)
            
            # Combined contribution scores
            total_magnitude = image_magnitude + text_magnitude
            if total_magnitude > 0:
                image_contribution = (image_magnitude / total_magnitude) * image_attention
                text_contribution = (text_magnitude / total_magnitude) * text_attention
                
                # Normalize to sum to 1
                total_contribution = image_contribution + text_contribution
                if total_contribution > 0:
                    image_contribution /= total_contribution
                    text_contribution /= total_contribution
            else:
                image_contribution = 0.5
                text_contribution = 0.5
            
            return {
                'image': float(image_contribution),
                'text': float(text_contribution)
            }
            
        except Exception as e:
            logger.warning(f"Modality contribution calculation failed: {e}")
            return {'image': 0.5, 'text': 0.5}
    
    def _update_similarity_stats(self, result: ImageTextSimilarityResult):
        """Update similarity computation statistics."""
        self.similarity_stats['total_computations'] += 1
        
        if result.processing_success:
            self.similarity_stats['successful_computations'] += 1
            
            # Update average computation time
            n = self.similarity_stats['successful_computations']
            current_avg = self.similarity_stats['avg_computation_time_ms']
            self.similarity_stats['avg_computation_time_ms'] = (
                (current_avg * (n - 1) + result.computation_time_ms) / n
            )
            
            # Update average image processing time
            current_img_avg = self.similarity_stats['avg_image_processing_time_ms']
            self.similarity_stats['avg_image_processing_time_ms'] = (
                (current_img_avg * (n - 1) + result.image_processing_time_ms) / n
            )
            
            # Track quantum advantage cases
            if result.quantum_advantage_score > 0.6:  # Threshold for significant advantage
                self.similarity_stats['quantum_advantage_cases'] += 1
    
    def get_similarity_stats(self) -> Dict[str, Any]:
        """Get comprehensive similarity computation statistics."""
        stats = self.similarity_stats.copy()
        
        # Calculate derived metrics
        if stats['total_computations'] > 0:
            stats['success_rate'] = stats['successful_computations'] / stats['total_computations']
            stats['quantum_advantage_rate'] = stats['quantum_advantage_cases'] / stats['total_computations']
        else:
            stats['success_rate'] = 0.0
            stats['quantum_advantage_rate'] = 0.0
        
        # Add component statistics
        stats['image_processor_stats'] = self.image_processor.get_processing_stats()
        
        return stats
    
    def batch_compute_similarities(self,
                                 image_data_list: List[Union[str, np.ndarray, Any]],
                                 text_embeddings: List[np.ndarray]) -> List[ImageTextSimilarityResult]:
        """
        Batch compute image-text similarities for improved performance.
        
        Args:
            image_data_list: List of image data
            text_embeddings: List of text embeddings
            
        Returns:
            List of ImageTextSimilarityResult objects
        """
        if len(image_data_list) != len(text_embeddings):
            raise ValueError("Image data list and text embeddings must have same length")
        
        results = []
        start_time = time.time()
        
        for i, (image_data, text_embedding) in enumerate(zip(image_data_list, text_embeddings)):
            result = self.compute_image_text_similarity(image_data, text_embedding)
            result.warnings.extend([f"Batch item {i}"])
            results.append(result)
        
        # Check batch performance constraint
        total_time = (time.time() - start_time) * 1000
        batch_size = len(image_data_list)
        
        if total_time > 1000:  # 1 second constraint for batch from task spec
            logger.warning(f"Batch processing time {total_time:.0f}ms exceeds 1000ms for {batch_size} items")
        
        # Add batch timing to all results
        for result in results:
            result.warnings.append(f"Batch total time: {total_time:.0f}ms for {batch_size} items")
        
        return results
    
    def clear_caches(self):
        """Clear all caches for memory optimization."""
        self.image_processor.clear_cache()
        self.entanglement_analyzer.clear_cache()
        logger.info("ImageTextQuantumSimilarity caches cleared")
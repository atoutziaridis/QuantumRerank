"""
Quantum State Analysis and Debugging Tools for QRF-01.

This module provides comprehensive tools for analyzing quantum state preparation,
amplitude encoding, and SWAP test implementation to debug fidelity saturation issues.

Based on:
- Task QRF-01: Debug Quantum Fidelity Saturation Issue
- Papers: "A quantum binary classifier based on cosine similarity"
- Papers: "A Quantum Geometric Model of Similarity" 
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
# import matplotlib.pyplot as plt  # Optional for visualization

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from quantum_rerank.core.embeddings import EmbeddingProcessor
from quantum_rerank.core.quantum_embedding_bridge import QuantumEmbeddingBridge
from quantum_rerank.core.swap_test import QuantumSWAPTest

logger = logging.getLogger(__name__)


@dataclass
class StateAnalysisResult:
    """Results from quantum state analysis."""
    text: str
    original_embedding: np.ndarray
    processed_embedding: np.ndarray
    quantum_amplitudes: np.ndarray
    state_norm: float
    information_loss: float
    encoding_method: str
    metadata: Dict[str, Any]


@dataclass
class FidelityDebugResult:
    """Results from fidelity debugging analysis."""
    text1: str
    text2: str
    classical_similarity: float
    theoretical_fidelity: float
    swap_test_fidelity: float
    amplitude_overlap: float
    state_distance: float
    discrimination_score: float
    issue_identified: List[str]


class QuantumStateAnalyzer:
    """
    Comprehensive analysis and debugging tools for quantum state preparation.
    
    Designed to identify and fix quantum fidelity saturation issues in QRF-01.
    """
    
    def __init__(self, n_qubits: int = 4):
        """Initialize quantum state analyzer."""
        self.n_qubits = n_qubits
        self.max_amplitudes = 2 ** n_qubits
        
        # Initialize components
        self.embedding_processor = EmbeddingProcessor()
        self.quantum_bridge = QuantumEmbeddingBridge(n_qubits=n_qubits)
        self.swap_test = QuantumSWAPTest(n_qubits=n_qubits)
        
        logger.info(f"Initialized QuantumStateAnalyzer with {n_qubits} qubits")
    
    def analyze_quantum_state_preparation(self, text: str, 
                                        encoding_method: str = 'amplitude') -> StateAnalysisResult:
        """
        Analyze quantum state preparation for a single text.
        
        Identifies issues in embedding -> quantum state conversion pipeline.
        """
        logger.info(f"Analyzing quantum state preparation for: '{text[:50]}...'")
        
        # Step 1: Generate original embedding
        original_embedding = self.embedding_processor.encode_single_text(text)
        
        # Step 2: Preprocess for quantum
        processed_embeddings, preprocessing_metadata = self.embedding_processor.preprocess_for_quantum(
            np.array([original_embedding]), self.n_qubits
        )
        processed_embedding = processed_embeddings[0]
        
        # Step 3: Convert to quantum circuit and extract amplitudes
        bridge_result = self.quantum_bridge.text_to_quantum_circuit(text, encoding_method)
        
        if bridge_result.success and bridge_result.statevector is not None:
            quantum_amplitudes = bridge_result.statevector.data
            state_norm = np.linalg.norm(quantum_amplitudes)
        else:
            quantum_amplitudes = np.zeros(self.max_amplitudes, dtype=complex)
            state_norm = 0.0
        
        # Step 4: Calculate information metrics
        information_loss = self._calculate_information_loss(
            original_embedding, processed_embedding, quantum_amplitudes
        )
        
        # Step 5: Compile analysis metadata
        metadata = {
            'original_dim': len(original_embedding),
            'processed_dim': len(processed_embedding),
            'quantum_amplitudes_count': len(quantum_amplitudes),
            'preprocessing_metadata': preprocessing_metadata,
            'bridge_success': bridge_result.success,
            'bridge_metadata': bridge_result.metadata if bridge_result.metadata else {},
            'amplitude_statistics': self._compute_amplitude_statistics(quantum_amplitudes),
            'encoding_issues': self._identify_encoding_issues(
                original_embedding, processed_embedding, quantum_amplitudes
            )
        }
        
        return StateAnalysisResult(
            text=text,
            original_embedding=original_embedding,
            processed_embedding=processed_embedding,
            quantum_amplitudes=quantum_amplitudes,
            state_norm=state_norm,
            information_loss=information_loss,
            encoding_method=encoding_method,
            metadata=metadata
        )
    
    def debug_fidelity_computation(self, text1: str, text2: str,
                                 encoding_method: str = 'amplitude') -> FidelityDebugResult:
        """
        Debug fidelity computation between two texts.
        
        Identifies issues causing fidelity saturation.
        """
        logger.info(f"Debugging fidelity computation between texts")
        
        # Step 1: Analyze both quantum states
        state1_analysis = self.analyze_quantum_state_preparation(text1, encoding_method)
        state2_analysis = self.analyze_quantum_state_preparation(text2, encoding_method)
        
        # Step 2: Compute classical similarity baseline
        classical_similarity = self.embedding_processor.compute_classical_similarity(
            state1_analysis.original_embedding, state2_analysis.original_embedding
        )
        
        # Step 3: Compute theoretical fidelity from quantum amplitudes
        theoretical_fidelity = self._compute_theoretical_fidelity(
            state1_analysis.quantum_amplitudes, state2_analysis.quantum_amplitudes
        )
        
        # Step 4: Compute SWAP test fidelity
        swap_test_fidelity = 0.0
        if state1_analysis.metadata['bridge_success'] and state2_analysis.metadata['bridge_success']:
            # Get quantum circuits from bridge results
            circuit1 = self.quantum_bridge.text_to_quantum_circuit(text1, encoding_method).circuit
            circuit2 = self.quantum_bridge.text_to_quantum_circuit(text2, encoding_method).circuit
            
            if circuit1 and circuit2:
                swap_fidelity, swap_metadata = self.swap_test.compute_fidelity(circuit1, circuit2)
                swap_test_fidelity = swap_fidelity
        
        # Step 5: Compute amplitude overlap directly
        amplitude_overlap = np.abs(np.vdot(
            state1_analysis.quantum_amplitudes, state2_analysis.quantum_amplitudes
        ))
        
        # Step 6: Compute state distance measures
        state_distance = np.linalg.norm(
            state1_analysis.quantum_amplitudes - state2_analysis.quantum_amplitudes
        )
        
        # Step 7: Calculate discrimination score
        discrimination_score = abs(theoretical_fidelity - 1.0)  # How far from perfect fidelity
        
        # Step 8: Identify specific issues
        issues = self._identify_fidelity_issues(
            classical_similarity, theoretical_fidelity, swap_test_fidelity,
            amplitude_overlap, state_distance, discrimination_score
        )
        
        return FidelityDebugResult(
            text1=text1,
            text2=text2,
            classical_similarity=classical_similarity,
            theoretical_fidelity=theoretical_fidelity,
            swap_test_fidelity=swap_test_fidelity,
            amplitude_overlap=amplitude_overlap,
            state_distance=state_distance,
            discrimination_score=discrimination_score,
            issue_identified=issues
        )
    
    def validate_swap_test_with_known_states(self) -> Dict[str, Any]:
        """
        Validate SWAP test implementation with known quantum states.
        
        Tests fundamental SWAP test correctness.
        """
        logger.info("Validating SWAP test with known quantum states")
        
        results = {}
        
        # Test 1: Identical states (should give fidelity = 1.0)
        identical_circuit = QuantumCircuit(self.n_qubits)
        identical_circuit.h(0)  # Simple superposition
        
        fidelity_identical, metadata = self.swap_test.compute_fidelity(
            identical_circuit, identical_circuit
        )
        
        results['identical_states'] = {
            'fidelity': fidelity_identical,
            'expected': 1.0,
            'error': abs(fidelity_identical - 1.0),
            'pass': abs(fidelity_identical - 1.0) < 0.1,
            'metadata': metadata
        }
        
        # Test 2: Orthogonal states (should give fidelity = 0.0)
        zero_state = QuantumCircuit(self.n_qubits)  # |00...0⟩
        one_state = QuantumCircuit(self.n_qubits)
        one_state.x(0)  # |10...0⟩
        
        fidelity_orthogonal, metadata = self.swap_test.compute_fidelity(
            zero_state, one_state
        )
        
        results['orthogonal_states'] = {
            'fidelity': fidelity_orthogonal,
            'expected': 0.0,
            'error': abs(fidelity_orthogonal - 0.0),
            'pass': abs(fidelity_orthogonal - 0.0) < 0.1,
            'metadata': metadata
        }
        
        # Test 3: Partially overlapping states
        partial1 = QuantumCircuit(self.n_qubits)
        partial1.h(0)  # |+⟩⊗|00...0⟩
        
        partial2 = QuantumCircuit(self.n_qubits)
        partial2.ry(np.pi/3, 0)  # Different rotation
        
        fidelity_partial, metadata = self.swap_test.compute_fidelity(
            partial1, partial2
        )
        
        # Calculate expected fidelity for these specific states
        state1_sv = Statevector.from_instruction(partial1)
        state2_sv = Statevector.from_instruction(partial2)
        expected_partial = float(np.abs(state1_sv.inner(state2_sv))**2)
        
        results['partial_overlap'] = {
            'fidelity': fidelity_partial,
            'expected': expected_partial,
            'error': abs(fidelity_partial - expected_partial),
            'pass': abs(fidelity_partial - expected_partial) < 0.1,
            'metadata': metadata
        }
        
        # Overall validation
        all_tests_pass = all(result.get('pass', False) for result in results.values())
        results['overall_validation'] = {
            'pass': all_tests_pass,
            'total_tests': len(results) - 1,  # Exclude this summary
            'passed_tests': sum(1 for r in results.values() if r.get('pass', False))
        }
        
        logger.info(f"SWAP test validation: {'PASS' if all_tests_pass else 'FAIL'}")
        return results
    
    def comprehensive_fidelity_saturation_analysis(self, 
                                                 test_texts: List[str]) -> Dict[str, Any]:
        """
        Comprehensive analysis of fidelity saturation across multiple text pairs.
        
        Primary debugging function for QRF-01.
        """
        logger.info(f"Running comprehensive fidelity saturation analysis with {len(test_texts)} texts")
        
        results = {
            'individual_states': [],
            'pairwise_fidelities': [],
            'classical_similarities': [],
            'quantum_fidelities': [],
            'discrimination_analysis': {},
            'encoding_analysis': {},
            'recommendations': []
        }
        
        # Step 1: Analyze individual state preparations
        for text in test_texts:
            state_analysis = self.analyze_quantum_state_preparation(text)
            results['individual_states'].append({
                'text': text[:100],  # Truncate for logging
                'information_loss': state_analysis.information_loss,
                'state_norm': state_analysis.state_norm,
                'encoding_issues': state_analysis.metadata.get('encoding_issues', []),
                'amplitude_stats': state_analysis.metadata.get('amplitude_statistics', {})
            })
        
        # Step 2: Analyze all pairwise fidelities
        for i, text1 in enumerate(test_texts):
            for j, text2 in enumerate(test_texts):
                if i <= j:  # Only compute upper triangle + diagonal
                    fidelity_debug = self.debug_fidelity_computation(text1, text2)
                    
                    results['pairwise_fidelities'].append({
                        'pair': (i, j),
                        'text1_snippet': text1[:50],
                        'text2_snippet': text2[:50],
                        'classical_similarity': fidelity_debug.classical_similarity,
                        'theoretical_fidelity': fidelity_debug.theoretical_fidelity,
                        'swap_test_fidelity': fidelity_debug.swap_test_fidelity,
                        'discrimination_score': fidelity_debug.discrimination_score,
                        'issues': fidelity_debug.issue_identified
                    })
                    
                    results['classical_similarities'].append(fidelity_debug.classical_similarity)
                    results['quantum_fidelities'].append(fidelity_debug.theoretical_fidelity)
        
        # Step 3: Discrimination analysis
        classical_range = max(results['classical_similarities']) - min(results['classical_similarities'])
        quantum_range = max(results['quantum_fidelities']) - min(results['quantum_fidelities'])
        
        results['discrimination_analysis'] = {
            'classical_similarity_range': classical_range,
            'quantum_fidelity_range': quantum_range,
            'discrimination_ratio': quantum_range / classical_range if classical_range > 0 else 0,
            'quantum_discrimination_adequate': quantum_range > 0.1,
            'avg_classical_similarity': np.mean(results['classical_similarities']),
            'avg_quantum_fidelity': np.mean(results['quantum_fidelities']),
            'quantum_saturation_detected': quantum_range < 0.01
        }
        
        # Step 4: Encoding analysis across all states
        all_information_losses = [state['information_loss'] for state in results['individual_states']]
        all_encoding_issues = []
        for state in results['individual_states']:
            all_encoding_issues.extend(state['encoding_issues'])
        
        results['encoding_analysis'] = {
            'avg_information_loss': np.mean(all_information_losses),
            'max_information_loss': np.max(all_information_losses),
            'common_encoding_issues': list(set(all_encoding_issues)),
            'information_loss_critical': np.mean(all_information_losses) > 0.9
        }
        
        # Step 5: Generate recommendations
        results['recommendations'] = self._generate_fix_recommendations(results)
        
        logger.info("Comprehensive fidelity saturation analysis complete")
        return results
    
    def _calculate_information_loss(self, original_embedding: np.ndarray,
                                  processed_embedding: np.ndarray,
                                  quantum_amplitudes: np.ndarray) -> float:
        """Calculate information loss in quantum encoding pipeline."""
        # Calculate loss from dimension reduction (768D -> 16D)
        original_dim = len(original_embedding)
        processed_dim = len(processed_embedding)
        quantum_dim = len(quantum_amplitudes)
        
        # Dimension reduction loss
        if original_dim > processed_dim:
            dimension_loss = 1.0 - (processed_dim / original_dim)
        else:
            dimension_loss = 0.0
        
        # Variance preservation loss
        original_var = np.var(original_embedding)
        processed_var = np.var(processed_embedding.real if processed_embedding.dtype == complex else processed_embedding)
        quantum_var = np.var(np.abs(quantum_amplitudes))
        
        if original_var > 0:
            variance_loss = 1.0 - min(1.0, quantum_var / original_var)
        else:
            variance_loss = 1.0
        
        # Combined information loss (weighted average)
        total_loss = 0.7 * dimension_loss + 0.3 * variance_loss
        
        return float(min(1.0, max(0.0, total_loss)))
    
    def _compute_amplitude_statistics(self, amplitudes: np.ndarray) -> Dict[str, float]:
        """Compute statistics about quantum state amplitudes."""
        abs_amplitudes = np.abs(amplitudes)
        
        return {
            'mean_amplitude': float(np.mean(abs_amplitudes)),
            'max_amplitude': float(np.max(abs_amplitudes)),
            'min_amplitude': float(np.min(abs_amplitudes)),
            'amplitude_variance': float(np.var(abs_amplitudes)),
            'amplitude_entropy': float(-np.sum(abs_amplitudes**2 * np.log(abs_amplitudes**2 + 1e-10))),
            'effective_rank': float(1.0 / np.sum(abs_amplitudes**4)),  # Participation ratio
            'norm': float(np.linalg.norm(amplitudes))
        }
    
    def _identify_encoding_issues(self, original_embedding: np.ndarray,
                                processed_embedding: np.ndarray,
                                quantum_amplitudes: np.ndarray) -> List[str]:
        """Identify specific issues in quantum encoding pipeline."""
        issues = []
        
        # Check for dimension mismatch
        if len(processed_embedding) != len(quantum_amplitudes):
            issues.append("dimension_mismatch")
        
        # Check for excessive truncation
        if len(original_embedding) > len(processed_embedding):
            truncation_ratio = len(processed_embedding) / len(original_embedding)
            if truncation_ratio < 0.1:
                issues.append("excessive_truncation")
        
        # Check for amplitude uniformity (indicates loss of discrimination)
        abs_amplitudes = np.abs(quantum_amplitudes)
        amplitude_variance = np.var(abs_amplitudes)
        if amplitude_variance < 1e-6:
            issues.append("amplitude_uniformity")
        
        # Check for normalization issues
        norm = np.linalg.norm(quantum_amplitudes)
        if abs(norm - 1.0) > 0.1:
            issues.append("normalization_error")
        
        # Check for information concentration
        max_amplitude = np.max(abs_amplitudes)
        if max_amplitude > 0.9:
            issues.append("information_concentration")
        
        return issues
    
    def _compute_theoretical_fidelity(self, amplitudes1: np.ndarray,
                                    amplitudes2: np.ndarray) -> float:
        """Compute theoretical fidelity between quantum state amplitudes."""
        # Ensure both are normalized
        norm1 = np.linalg.norm(amplitudes1)
        norm2 = np.linalg.norm(amplitudes2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        normed_amp1 = amplitudes1 / norm1
        normed_amp2 = amplitudes2 / norm2
        
        # Fidelity is squared overlap
        overlap = np.vdot(normed_amp1, normed_amp2)
        fidelity = float(np.abs(overlap)**2)
        
        return fidelity
    
    def _identify_fidelity_issues(self, classical_similarity: float,
                                theoretical_fidelity: float,
                                swap_test_fidelity: float,
                                amplitude_overlap: float,
                                state_distance: float,
                                discrimination_score: float) -> List[str]:
        """Identify specific issues causing fidelity saturation."""
        issues = []
        
        # Check for quantum fidelity saturation
        if theoretical_fidelity > 0.995:
            issues.append("quantum_fidelity_saturation")
        
        # Check for loss of discrimination
        if discrimination_score < 0.005:  # Very close to 1.0
            issues.append("discrimination_loss")
        
        # Check for classical vs quantum mismatch
        if abs(classical_similarity - theoretical_fidelity) > 0.1:
            issues.append("classical_quantum_mismatch")
        
        # Check for SWAP test accuracy
        if abs(theoretical_fidelity - swap_test_fidelity) > 0.1:
            issues.append("swap_test_inaccuracy")
        
        # Check for amplitude encoding problems
        if amplitude_overlap > 0.999:
            issues.append("amplitude_encoding_collapse")
        
        # Check for insufficient state separation
        if state_distance < 0.01:
            issues.append("insufficient_state_separation")
        
        return issues
    
    def _generate_fix_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations to fix identified issues."""
        recommendations = []
        
        # Check quantum discrimination
        if analysis_results['discrimination_analysis']['quantum_saturation_detected']:
            recommendations.append("CRITICAL: Implement alternative amplitude encoding method")
            recommendations.append("Increase qubit count to preserve more semantic information")
            recommendations.append("Implement feature selection before quantum encoding")
        
        # Check information loss
        if analysis_results['encoding_analysis']['information_loss_critical']:
            recommendations.append("Implement PCA dimensionality reduction before quantum encoding")
            recommendations.append("Use angle encoding instead of amplitude encoding")
            recommendations.append("Implement hybrid classical-quantum similarity scoring")
        
        # Check common encoding issues
        common_issues = analysis_results['encoding_analysis']['common_encoding_issues']
        if 'amplitude_uniformity' in common_issues:
            recommendations.append("Fix amplitude encoding to preserve embedding variance")
        
        if 'excessive_truncation' in common_issues:
            recommendations.append("Reduce embedding dimensions before quantum processing")
        
        if 'normalization_error' in common_issues:
            recommendations.append("Fix quantum state normalization in preprocessing")
        
        # SWAP test specific recommendations
        pairwise_issues = []
        for pair in analysis_results['pairwise_fidelities']:
            pairwise_issues.extend(pair['issues'])
        
        if 'swap_test_inaccuracy' in pairwise_issues:
            recommendations.append("Validate and fix SWAP test circuit implementation")
        
        if not recommendations:
            recommendations.append("No critical issues detected - investigate measurement precision")
        
        return recommendations


def run_qrf01_debug_analysis(test_texts: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Main function to run QRF-01 debugging analysis.
    
    Can be called from other modules or run standalone.
    """
    if test_texts is None:
        # Default test texts with known different semantic content
        test_texts = [
            "Quantum mechanics describes the behavior of matter and energy at atomic scales.",
            "The patient presented with acute myocardial infarction and elevated troponin levels.",
            "Machine learning algorithms can classify images using convolutional neural networks.",
            "Today is a sunny day and I am feeling happy about going to the park.",
            "Diabetes mellitus type 2 is characterized by insulin resistance and hyperglycemia."
        ]
    
    analyzer = QuantumStateAnalyzer(n_qubits=4)
    
    # Run comprehensive analysis
    results = analyzer.comprehensive_fidelity_saturation_analysis(test_texts)
    
    # Add SWAP test validation
    results['swap_test_validation'] = analyzer.validate_swap_test_with_known_states()
    
    # Print summary
    print("="*50)
    print("QRF-01 FIDELITY SATURATION DEBUG ANALYSIS")
    print("="*50)
    
    print(f"\nQuantum Discrimination Analysis:")
    disc = results['discrimination_analysis']
    print(f"  - Classical similarity range: {disc['classical_similarity_range']:.4f}")
    print(f"  - Quantum fidelity range: {disc['quantum_fidelity_range']:.4f}")
    print(f"  - Discrimination ratio: {disc['discrimination_ratio']:.4f}")
    print(f"  - Quantum saturation detected: {disc['quantum_saturation_detected']}")
    
    print(f"\nEncoding Analysis:")
    enc = results['encoding_analysis']
    print(f"  - Average information loss: {enc['avg_information_loss']:.4f}")
    print(f"  - Common issues: {enc['common_encoding_issues']}")
    
    print(f"\nSWAP Test Validation:")
    swap_val = results['swap_test_validation']['overall_validation']
    print(f"  - Overall validation: {'PASS' if swap_val['pass'] else 'FAIL'}")
    print(f"  - Tests passed: {swap_val['passed_tests']}/{swap_val['total_tests']}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    return results


if __name__ == "__main__":
    # Run the main debug analysis
    debug_results = run_qrf01_debug_analysis()
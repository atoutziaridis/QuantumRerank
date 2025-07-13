"""
Bridge between classical embeddings and quantum circuits for QuantumRerank.

This module implements the integration specified in PRD Section 5.2,
connecting SentenceTransformer embeddings to quantum circuits for 
quantum-inspired similarity computation.

Based on:
- PRD Section 5.2: Integration with RAG Pipeline
- Research: Quantum binary classifier based on cosine similarity
- Research: Quantum-inspired embeddings projection and similarity metrics
"""

import numpy as np
import time
import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .embeddings import EmbeddingProcessor, EmbeddingConfig
from .quantum_circuits import BasicQuantumCircuits, CircuitResult
from ..config.settings import QuantumConfig

logger = logging.getLogger(__name__)


@dataclass
class BridgeResult:
    """Result container for quantum embedding bridge operations."""
    success: bool
    text: str
    circuit: Optional[QuantumCircuit] = None
    statevector: Optional[Statevector] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass 
class SimilarityResult:
    """Result container for similarity computations."""
    classical_cosine: float
    quantum_fidelity: float
    quantum_amplitude_overlap: Optional[float] = None
    computation_time_ms: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class QuantumEmbeddingBridge:
    """
    Bridge between classical embeddings and quantum circuits.
    
    Implements the integration specified in PRD Section 5.2.
    Converts text to quantum circuits and computes quantum-inspired similarities.
    """
    
    def __init__(self, n_qubits: int = 4, 
                 embedding_config: Optional[EmbeddingConfig] = None,
                 quantum_config: Optional[QuantumConfig] = None):
        """
        Initialize the quantum embedding bridge.
        
        Args:
            n_qubits: Number of qubits for quantum circuits
            embedding_config: Configuration for embedding processor
            quantum_config: Configuration for quantum circuits
        """
        self.n_qubits = n_qubits
        
        # Initialize embedding processor and quantum circuits
        self.embedding_processor = EmbeddingProcessor(embedding_config)
        self.quantum_circuits = BasicQuantumCircuits(quantum_config)
        
        # Validate compatibility
        if self.quantum_circuits.n_qubits != n_qubits:
            logger.warning(f"Quantum circuit qubits ({self.quantum_circuits.n_qubits}) != bridge qubits ({n_qubits})")
            self.n_qubits = self.quantum_circuits.n_qubits
        
        logger.info(f"Initialized QuantumEmbeddingBridge with {self.n_qubits} qubits")
    
    def text_to_quantum_circuit(self, text: str, 
                               encoding_method: str = 'amplitude') -> BridgeResult:
        """
        Convert text directly to quantum circuit.
        
        Full pipeline: text -> embedding -> quantum circuit
        
        Args:
            text: Input text string
            encoding_method: Quantum encoding method ('amplitude', 'angle', 'dense_angle')
            
        Returns:
            BridgeResult with circuit and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Generate embedding
            embedding = self.embedding_processor.encode_single_text(text)
            
            # Step 2: Preprocess for quantum
            processed_embeddings, preprocessing_metadata = self.embedding_processor.preprocess_for_quantum(
                np.array([embedding]), self.n_qubits
            )
            processed_embedding = processed_embeddings[0]
            
            # Step 3: Create quantum circuit based on encoding method
            if encoding_method == 'amplitude':
                quantum_circuit = self.quantum_circuits.amplitude_encode_embedding(
                    processed_embedding, name=f"text_amplitude_{hash(text) % 10000}"
                )
            elif encoding_method == 'angle':
                quantum_circuit = self.quantum_circuits.angle_encode_embedding(
                    processed_embedding, name=f"text_angle_{hash(text) % 10000}"
                )
            elif encoding_method == 'dense_angle':
                quantum_circuit = self.quantum_circuits.dense_angle_encoding(
                    processed_embedding, name=f"text_dense_{hash(text) % 10000}"
                )
            else:
                raise ValueError(f"Unknown encoding method: {encoding_method}")
            
            # Step 4: Simulate circuit to get statevector
            sim_result = self.quantum_circuits.simulate_circuit(quantum_circuit)
            
            processing_time = time.time() - start_time
            
            # Compile metadata
            metadata = {
                'text_length': len(text),
                'original_embedding_dim': len(embedding),
                'processed_embedding_dim': len(processed_embedding),
                'encoding_method': encoding_method,
                'quantum_circuit_depth': quantum_circuit.depth(),
                'quantum_circuit_size': quantum_circuit.size(),
                'preprocessing_metadata': preprocessing_metadata,
                'simulation_success': sim_result.success,
                'total_processing_time_ms': processing_time * 1000,
                'prd_compliant': quantum_circuit.depth() <= 15  # PRD constraint
            }
            
            if sim_result.success:
                metadata.update(sim_result.metadata)
                
                return BridgeResult(
                    success=True,
                    text=text,
                    circuit=quantum_circuit,
                    statevector=sim_result.statevector,
                    metadata=metadata
                )
            else:
                return BridgeResult(
                    success=False,
                    text=text,
                    circuit=quantum_circuit,
                    metadata=metadata,
                    error=sim_result.error
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Text to quantum circuit conversion failed: {str(e)}"
            logger.error(error_msg)
            
            return BridgeResult(
                success=False,
                text=text,
                metadata={
                    'text_length': len(text),
                    'total_processing_time_ms': processing_time * 1000,
                    'error': error_msg
                },
                error=error_msg
            )
    
    def batch_texts_to_circuits(self, texts: List[str], 
                               encoding_method: str = 'amplitude') -> List[BridgeResult]:
        """
        Convert batch of texts to quantum circuits efficiently.
        
        Args:
            texts: List of input texts
            encoding_method: Quantum encoding method
            
        Returns:
            List of BridgeResult objects
        """
        start_time = time.time()
        logger.info(f"Converting {len(texts)} texts to quantum circuits using {encoding_method} encoding")
        
        try:
            # Batch encode all texts
            embeddings = self.embedding_processor.encode_texts(texts)
            
            # Preprocess for quantum
            processed_embeddings, batch_metadata = self.embedding_processor.preprocess_for_quantum(
                embeddings, self.n_qubits
            )
            
            results = []
            for i, (text, processed_embedding) in enumerate(zip(texts, processed_embeddings)):
                
                # Create quantum circuit
                try:
                    if encoding_method == 'amplitude':
                        circuit = self.quantum_circuits.amplitude_encode_embedding(
                            processed_embedding, name=f"batch_{i}_amplitude"
                        )
                    elif encoding_method == 'angle':
                        circuit = self.quantum_circuits.angle_encode_embedding(
                            processed_embedding, name=f"batch_{i}_angle"
                        )
                    elif encoding_method == 'dense_angle':
                        circuit = self.quantum_circuits.dense_angle_encoding(
                            processed_embedding, name=f"batch_{i}_dense"
                        )
                    else:
                        raise ValueError(f"Unknown encoding method: {encoding_method}")
                    
                    # Simulate circuit
                    sim_result = self.quantum_circuits.simulate_circuit(circuit)
                    
                    # Individual metadata
                    metadata = {
                        'batch_index': i,
                        'text_length': len(text),
                        'encoding_method': encoding_method,
                        'quantum_circuit_depth': circuit.depth(),
                        'quantum_circuit_size': circuit.size(),
                        'batch_metadata': batch_metadata,
                        'simulation_success': sim_result.success,
                        'prd_compliant': circuit.depth() <= 15
                    }
                    
                    if sim_result.success:
                        metadata.update(sim_result.metadata)
                        
                        results.append(BridgeResult(
                            success=True,
                            text=text,
                            circuit=circuit,
                            statevector=sim_result.statevector,
                            metadata=metadata
                        ))
                    else:
                        results.append(BridgeResult(
                            success=False,
                            text=text,
                            circuit=circuit,
                            metadata=metadata,
                            error=sim_result.error
                        ))
                        
                except Exception as e:
                    error_msg = f"Failed to process text {i}: {str(e)}"
                    logger.error(error_msg)
                    
                    results.append(BridgeResult(
                        success=False,
                        text=text,
                        metadata={
                            'batch_index': i,
                            'text_length': len(text),
                            'error': error_msg
                        },
                        error=error_msg
                    ))
            
            total_time = time.time() - start_time
            success_count = sum(1 for r in results if r.success)
            
            logger.info(f"Batch conversion complete: {success_count}/{len(texts)} successful in {total_time:.2f}s")
            
            return results
            
        except Exception as e:
            error_msg = f"Batch text to circuit conversion failed: {str(e)}"
            logger.error(error_msg)
            
            # Return error results for all texts
            return [
                BridgeResult(
                    success=False,
                    text=text,
                    error=error_msg
                ) for text in texts
            ]
    
    def compute_quantum_similarity(self, text1: str, text2: str, 
                                  encoding_method: str = 'amplitude') -> SimilarityResult:
        """
        Compute both classical and quantum similarities between two texts.
        
        Args:
            text1, text2: Input text strings
            encoding_method: Quantum encoding method
            
        Returns:
            SimilarityResult with multiple similarity metrics
        """
        start_time = time.time()
        
        try:
            # Generate embeddings
            embeddings = self.embedding_processor.encode_texts([text1, text2])
            emb1, emb2 = embeddings[0], embeddings[1]
            
            # Classical cosine similarity
            classical_cosine = self.embedding_processor.compute_classical_similarity(emb1, emb2)
            
            # Quantum fidelity similarity
            quantum_fidelity = self.embedding_processor.compute_fidelity_similarity(emb1, emb2)
            
            # Convert to quantum circuits
            result1 = self.text_to_quantum_circuit(text1, encoding_method)
            result2 = self.text_to_quantum_circuit(text2, encoding_method)
            
            quantum_amplitude_overlap = None
            if result1.success and result2.success:
                # Compute quantum amplitude overlap
                state1 = result1.statevector.data
                state2 = result2.statevector.data
                quantum_amplitude_overlap = float(np.abs(np.vdot(state1, state2)))
            
            computation_time = time.time() - start_time
            
            metadata = {
                'text1_length': len(text1),
                'text2_length': len(text2),
                'encoding_method': encoding_method,
                'quantum_circuit1_success': result1.success,
                'quantum_circuit2_success': result2.success,
                'computation_time_ms': computation_time * 1000
            }
            
            return SimilarityResult(
                classical_cosine=classical_cosine,
                quantum_fidelity=quantum_fidelity,
                quantum_amplitude_overlap=quantum_amplitude_overlap,
                computation_time_ms=computation_time * 1000,
                metadata=metadata
            )
            
        except Exception as e:
            computation_time = time.time() - start_time
            logger.error(f"Quantum similarity computation failed: {str(e)}")
            
            # Return with NaN values to indicate failure
            return SimilarityResult(
                classical_cosine=float('nan'),
                quantum_fidelity=float('nan'),
                quantum_amplitude_overlap=None,
                computation_time_ms=computation_time * 1000,
                metadata={'error': str(e)}
            )
    
    def benchmark_bridge_performance(self, test_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Benchmark bridge performance against PRD targets.
        
        Args:
            test_texts: Optional test texts (uses default if None)
            
        Returns:
            Performance benchmark results
        """
        if test_texts is None:
            test_texts = [
                "Quantum computing leverages quantum mechanical phenomena",
                "Machine learning enables computers to learn from data", 
                "Information retrieval finds relevant documents in collections",
                "Natural language processing understands human language"
            ]
        
        logger.info(f"Benchmarking bridge performance with {len(test_texts)} texts")
        
        results = {}
        encoding_methods = ['amplitude', 'angle', 'dense_angle']
        
        for method in encoding_methods:
            method_results = {
                'single_conversion_times_ms': [],
                'batch_conversion_time_ms': 0,
                'similarity_computation_times_ms': [],
                'success_rates': [],
                'circuit_depths': [],
                'prd_compliance': []
            }
            
            # Test single text conversions
            for text in test_texts:
                result = self.text_to_quantum_circuit(text, method)
                
                if result.metadata:
                    method_results['single_conversion_times_ms'].append(
                        result.metadata.get('total_processing_time_ms', 0)
                    )
                    method_results['success_rates'].append(result.success)
                    
                    if result.circuit:
                        method_results['circuit_depths'].append(result.circuit.depth())
                        method_results['prd_compliance'].append(
                            result.metadata.get('prd_compliant', False)
                        )
            
            # Test batch conversion
            start_time = time.time()
            batch_results = self.batch_texts_to_circuits(test_texts, method)
            batch_time = time.time() - start_time
            method_results['batch_conversion_time_ms'] = batch_time * 1000
            
            # Test similarity computations
            for i in range(len(test_texts)):
                for j in range(i+1, len(test_texts)):
                    sim_result = self.compute_quantum_similarity(
                        test_texts[i], test_texts[j], method
                    )
                    method_results['similarity_computation_times_ms'].append(
                        sim_result.computation_time_ms
                    )
            
            # Calculate summary statistics
            results[method] = {
                'avg_single_conversion_ms': np.mean(method_results['single_conversion_times_ms']),
                'batch_conversion_ms': method_results['batch_conversion_time_ms'],
                'avg_similarity_computation_ms': np.mean(method_results['similarity_computation_times_ms']),
                'success_rate': np.mean(method_results['success_rates']),
                'avg_circuit_depth': np.mean(method_results['circuit_depths']),
                'prd_compliance_rate': np.mean(method_results['prd_compliance']),
                'prd_targets': {
                    'similarity_under_100ms': np.mean(method_results['similarity_computation_times_ms']) < 100,
                    'circuit_depth_under_15': np.mean(method_results['circuit_depths']) <= 15,
                    'high_success_rate': np.mean(method_results['success_rates']) > 0.9
                }
            }
        
        # Overall summary
        all_similarity_times = []
        all_success_rates = []
        for method_data in results.values():
            all_similarity_times.append(method_data['avg_similarity_computation_ms'])
            all_success_rates.append(method_data['success_rate'])
        
        results['summary'] = {
            'overall_avg_similarity_ms': np.mean(all_similarity_times),
            'overall_success_rate': np.mean(all_success_rates),
            'prd_compliance': {
                'similarity_target_met': np.mean(all_similarity_times) < 100,
                'overall_success_high': np.mean(all_success_rates) > 0.9
            },
            'benchmark_timestamp': time.time()
        }
        
        logger.info(f"Bridge benchmark complete: {results['summary']['overall_avg_similarity_ms']:.2f}ms avg similarity")
        
        return results
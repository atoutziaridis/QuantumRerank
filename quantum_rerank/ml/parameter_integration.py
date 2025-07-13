"""
Parameter Integration Pipeline.

This module provides the complete pipeline for embedding-to-circuit conversion,
integrating all components for end-to-end processing from text to quantum circuits.

Based on:
- Task 05 specifications for complete integration
- PRD Section 3.1: Core Algorithms - End-to-End Pipeline
"""

import torch
import numpy as np
import time
import logging
from typing import List, Tuple, Dict, Optional
from qiskit import QuantumCircuit

from .parameter_predictor import QuantumParameterPredictor, ParameterPredictorConfig
from .parameterized_circuits import ParameterizedQuantumCircuits
from ..core.embeddings import EmbeddingProcessor

logger = logging.getLogger(__name__)


class EmbeddingToCircuitPipeline:
    """
    Complete pipeline: embedding -> parameters -> quantum circuit.
    
    Integrates all components for end-to-end processing from text/embeddings
    to parameterized quantum circuits that can be used for similarity computation.
    """
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        """
        Initialize the complete pipeline.
        
        Args:
            n_qubits: Number of qubits for quantum circuits (PRD: 2-4)
            n_layers: Number of circuit layers
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Initialize components
        self.embedding_processor = EmbeddingProcessor()
        
        config = ParameterPredictorConfig(
            embedding_dim=self.embedding_processor.config.embedding_dim,
            n_qubits=n_qubits,
            n_layers=n_layers
        )
        self.parameter_predictor = QuantumParameterPredictor(config)
        self.circuit_builder = ParameterizedQuantumCircuits(n_qubits, n_layers)
        
        logger.info(f"EmbeddingToCircuitPipeline initialized: {n_qubits} qubits, {n_layers} layers")
    
    def text_to_parameterized_circuit(self, text: str) -> Tuple[QuantumCircuit, Dict]:
        """
        Convert text to parameterized quantum circuit.
        
        Complete pipeline: text -> embedding -> parameters -> circuit
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (quantum_circuit, metadata)
        """
        start_time = time.time()
        
        try:
            # Step 1: Generate embedding
            embedding = self.embedding_processor.encode_single_text(text)
            embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0)
            
            # Step 2: Predict quantum parameters
            with torch.no_grad():
                parameters = self.parameter_predictor(embedding_tensor)
            
            # Step 3: Create parameterized quantum circuit
            circuit = self.circuit_builder.create_parameterized_circuit(
                parameters, batch_index=0, circuit_name=f"circuit_{hash(text) % 10000}"
            )
            
            # Collect metadata
            total_time = time.time() - start_time
            metadata = {
                'text': text,
                'text_length': len(text),
                'embedding_dim': len(embedding),
                'total_parameters': sum(p.numel() for p in parameters.values()),
                'circuit_depth': circuit.depth(),
                'circuit_size': circuit.size(),
                'processing_time_ms': total_time * 1000,
                'prd_compliant': {
                    'circuit_depth_ok': circuit.depth() <= 15,
                    'processing_time_ok': total_time * 1000 < 100,  # Target <100ms
                    'parameter_validation': self.parameter_predictor.validate_parameters(parameters)
                },
                'success': True
            }
            
            return circuit, metadata
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"Text to circuit conversion failed: {e}")
            
            # Return empty circuit and error metadata
            empty_circuit = QuantumCircuit(self.n_qubits, name="failed_circuit")
            metadata = {
                'text': text,
                'success': False,
                'error': str(e),
                'processing_time_ms': error_time * 1000
            }
            
            return empty_circuit, metadata
    
    def batch_text_to_circuits(self, texts: List[str]) -> List[Tuple[str, QuantumCircuit, Dict]]:
        """
        Convert batch of texts to parameterized circuits.
        
        Efficient batch processing version that optimizes embedding and parameter prediction.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of (text, circuit, metadata) tuples
        """
        start_time = time.time()
        
        try:
            # Step 1: Batch encode embeddings
            embeddings = self.embedding_processor.encode_texts(texts)
            embedding_tensor = torch.FloatTensor(embeddings)
            
            # Step 2: Batch predict parameters
            with torch.no_grad():
                parameters = self.parameter_predictor(embedding_tensor)
            
            # Step 3: Create circuits
            circuit_names = [f"batch_circuit_{i}_{hash(text) % 1000}" 
                           for i, text in enumerate(texts)]
            circuits = self.circuit_builder.create_batch_circuits(parameters, circuit_names)
            
            # Step 4: Combine results with metadata
            batch_time = time.time() - start_time
            results = []
            
            for i, (text, circuit) in enumerate(zip(texts, circuits)):
                metadata = {
                    'batch_index': i,
                    'text': text,
                    'text_length': len(text),
                    'circuit_depth': circuit.depth(),
                    'circuit_size': circuit.size(),
                    'batch_processing_time_ms': batch_time * 1000,
                    'per_item_time_ms': (batch_time / len(texts)) * 1000,
                    'success': True
                }
                results.append((text, circuit, metadata))
            
            logger.info(f"Batch processed {len(texts)} texts in {batch_time*1000:.2f}ms")
            return results
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"Batch text to circuit conversion failed: {e}")
            
            # Return failed results for all texts
            results = []
            for i, text in enumerate(texts):
                empty_circuit = QuantumCircuit(self.n_qubits, name=f"failed_circuit_{i}")
                metadata = {
                    'batch_index': i,
                    'text': text,
                    'success': False,
                    'error': str(e),
                    'processing_time_ms': error_time * 1000
                }
                results.append((text, empty_circuit, metadata))
            
            return results
    
    def embedding_to_parameterized_circuit(self, 
                                         embedding: np.ndarray,
                                         include_embedding_encoding: bool = False) -> Tuple[QuantumCircuit, Dict]:
        """
        Convert embedding directly to parameterized quantum circuit.
        
        Args:
            embedding: Input embedding vector
            include_embedding_encoding: Whether to include amplitude encoding of embedding
            
        Returns:
            Tuple of (quantum_circuit, metadata)
        """
        start_time = time.time()
        
        try:
            # Convert to tensor
            embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0)
            
            # Predict parameters
            with torch.no_grad():
                parameters = self.parameter_predictor(embedding_tensor)
            
            # Create circuit
            if include_embedding_encoding:
                # Create hybrid circuit with embedding encoding + parameterization
                circuit = self.circuit_builder.create_embedding_parameterized_circuit(
                    embedding, parameters, batch_index=0
                )
            else:
                # Create purely parameterized circuit
                circuit = self.circuit_builder.create_parameterized_circuit(
                    parameters, batch_index=0
                )
            
            # Metadata
            processing_time = time.time() - start_time
            metadata = {
                'embedding_dim': len(embedding),
                'include_embedding_encoding': include_embedding_encoding,
                'circuit_depth': circuit.depth(),
                'circuit_size': circuit.size(),
                'processing_time_ms': processing_time * 1000,
                'success': True
            }
            
            return circuit, metadata
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"Embedding to circuit conversion failed: {e}")
            
            empty_circuit = QuantumCircuit(self.n_qubits)
            metadata = {
                'embedding_dim': len(embedding) if embedding is not None else 0,
                'success': False,
                'error': str(e),
                'processing_time_ms': error_time * 1000
            }
            
            return empty_circuit, metadata
    
    def compute_similarity_via_circuits(self, 
                                      text1: str, 
                                      text2: str) -> Tuple[float, Dict]:
        """
        Compute similarity between two texts via parameterized quantum circuits.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Tuple of (similarity_score, metadata)
        """
        start_time = time.time()
        
        try:
            # Convert texts to circuits
            circuit1, meta1 = self.text_to_parameterized_circuit(text1)
            circuit2, meta2 = self.text_to_parameterized_circuit(text2)
            
            if not meta1['success'] or not meta2['success']:
                return 0.0, {
                    'success': False,
                    'error': 'Circuit creation failed',
                    'text1_metadata': meta1,
                    'text2_metadata': meta2
                }
            
            # Compute circuit fidelity
            fidelity, fidelity_meta = self.circuit_builder.compute_circuit_fidelity(
                circuit1, circuit2
            )
            
            # Combine metadata
            total_time = time.time() - start_time
            metadata = {
                'text1': text1,
                'text2': text2,
                'similarity_score': fidelity,
                'text1_metadata': meta1,
                'text2_metadata': meta2,
                'fidelity_metadata': fidelity_meta,
                'total_time_ms': total_time * 1000,
                'success': True
            }
            
            return fidelity, metadata
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"Similarity computation failed: {e}")
            
            return 0.0, {
                'text1': text1,
                'text2': text2,
                'success': False,
                'error': str(e),
                'total_time_ms': error_time * 1000
            }
    
    def benchmark_pipeline_performance(self, test_texts: List[str] = None) -> Dict:
        """
        Benchmark the complete pipeline performance.
        
        Args:
            test_texts: Optional test texts, will use default if None
            
        Returns:
            Performance benchmark results
        """
        if test_texts is None:
            test_texts = [
                "quantum computing applications",
                "machine learning algorithms", 
                "information retrieval systems",
                "natural language processing"
            ]
        
        results = {
            'single_processing': {},
            'batch_processing': {},
            'similarity_computation': {},
            'performance_summary': {}
        }
        
        # Benchmark single text processing
        single_times = []
        for text in test_texts[:2]:  # Test first 2
            start_time = time.time()
            circuit, metadata = self.text_to_parameterized_circuit(text)
            processing_time = time.time() - start_time
            
            single_times.append({
                'text': text,
                'time_ms': processing_time * 1000,
                'success': metadata['success'],
                'circuit_depth': metadata.get('circuit_depth', 0)
            })
        
        results['single_processing'] = {
            'times': single_times,
            'avg_time_ms': np.mean([t['time_ms'] for t in single_times]),
            'max_time_ms': np.max([t['time_ms'] for t in single_times])
        }
        
        # Benchmark batch processing
        start_time = time.time()
        batch_results = self.batch_text_to_circuits(test_texts)
        batch_time = time.time() - start_time
        
        results['batch_processing'] = {
            'total_time_ms': batch_time * 1000,
            'per_item_time_ms': (batch_time / len(test_texts)) * 1000,
            'successful_conversions': sum(1 for _, _, meta in batch_results if meta['success']),
            'efficiency_vs_single': (batch_time / len(test_texts)) / results['single_processing']['avg_time_ms'] * 1000
        }
        
        # Benchmark similarity computation
        if len(test_texts) >= 2:
            start_time = time.time()
            similarity, sim_metadata = self.compute_similarity_via_circuits(
                test_texts[0], test_texts[1]
            )
            similarity_time = time.time() - start_time
            
            results['similarity_computation'] = {
                'time_ms': similarity_time * 1000,
                'similarity_score': similarity,
                'success': sim_metadata['success']
            }
        
        # Parameter validation
        try:
            embeddings = self.embedding_processor.encode_texts(test_texts[:2])
            embedding_tensor = torch.FloatTensor(embeddings)
            
            with torch.no_grad():
                parameters = self.parameter_predictor(embedding_tensor)
                param_validation = self.circuit_builder.validate_circuit_parameters(parameters)
                
            results['parameter_validation'] = param_validation
            
        except Exception as e:
            logger.warning(f"Parameter validation failed: {e}")
            results['parameter_validation'] = {'error': str(e)}
        
        # Performance summary
        results['performance_summary'] = {
            'avg_single_processing_ms': results['single_processing']['avg_time_ms'],
            'batch_efficiency_gain': 1.0 / results['batch_processing']['efficiency_vs_single'],
            'similarity_computation_ms': results['similarity_computation'].get('time_ms', 0),
            'prd_compliance': {
                'single_processing_under_100ms': results['single_processing']['avg_time_ms'] < 100,
                'batch_processing_efficient': results['batch_processing']['efficiency_vs_single'] < 0.8,
                'all_parameters_valid': results['parameter_validation'].get('overall', {}).get('valid', False)
            }
        }
        
        logger.info("Pipeline benchmark completed")
        return results
    
    def get_pipeline_info(self) -> Dict:
        """Get information about the pipeline configuration."""
        return {
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'embedding_dim': self.embedding_processor.config.embedding_dim,
            'model_name': self.embedding_processor.config.model_name,
            'parameter_predictor_config': self.parameter_predictor.config,
            'total_quantum_parameters': self.parameter_predictor.total_params,
            'parameters_per_layer': self.parameter_predictor.params_per_layer
        }
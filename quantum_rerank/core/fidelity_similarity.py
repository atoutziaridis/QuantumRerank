"""
Fidelity Similarity Engine - Integration of SWAP Test with Embedding Processing.

This module combines SWAP test quantum fidelity computation with embedding processing
to provide a complete text-to-similarity pipeline for quantum-inspired reranking.

Implements the complete pipeline:
text -> embedding -> quantum circuit -> SWAP test -> fidelity similarity
"""

import time
import logging
from typing import List, Tuple, Dict

from .swap_test import QuantumSWAPTest
from .quantum_embedding_bridge import QuantumEmbeddingBridge

logger = logging.getLogger(__name__)


class FidelitySimilarityEngine:
    """
    Combines SWAP test with embedding processing for similarity computation.
    
    Implements the complete text-to-similarity pipeline as specified in PRD:
    1. Text encoding using SentenceTransformer
    2. Quantum preprocessing and circuit creation
    3. SWAP test for fidelity computation
    4. Similarity scoring and metadata collection
    """
    
    def __init__(self, n_qubits: int = 4):
        """
        Initialize fidelity similarity engine.
        
        Args:
            n_qubits: Number of qubits for quantum circuits (PRD: 2-4 qubits)
        """
        self.n_qubits = n_qubits
        self.swap_test = QuantumSWAPTest(n_qubits)
        self.embedding_bridge = QuantumEmbeddingBridge(n_qubits)
        
        logger.info(f"Fidelity similarity engine initialized with {n_qubits} qubits")
    
    def compute_text_similarity(self, text1: str, text2: str, 
                               encoding_method: str = 'amplitude') -> Tuple[float, Dict]:
        """
        Compute quantum fidelity-based similarity between two texts.
        
        Complete pipeline: text -> embedding -> quantum circuit -> fidelity
        
        Args:
            text1: First text for comparison
            text2: Second text for comparison
            encoding_method: Quantum encoding method ('amplitude', 'angle', 'dense_angle')
            
        Returns:
            Tuple of (similarity_score, metadata)
        """
        start_time = time.time()
        
        try:
            # Convert texts to quantum circuits
            circuit1_result = self.embedding_bridge.text_to_quantum_circuit(
                text1, encoding_method=encoding_method
            )
            circuit2_result = self.embedding_bridge.text_to_quantum_circuit(
                text2, encoding_method=encoding_method
            )
            
            # Check if circuit creation was successful
            if not circuit1_result.success or not circuit2_result.success:
                error_msg = f"Circuit creation failed: {circuit1_result.error or circuit2_result.error}"
                logger.error(error_msg)
                return 0.0, {
                    'success': False,
                    'error': error_msg,
                    'total_time_ms': (time.time() - start_time) * 1000
                }
            
            # Compute fidelity using SWAP test
            fidelity, fidelity_metadata = self.swap_test.compute_fidelity(
                circuit1_result.circuit, circuit2_result.circuit
            )
            
            # Combine metadata
            total_time = time.time() - start_time
            combined_metadata = {
                'text1': text1,
                'text2': text2,
                'encoding_method': encoding_method,
                'text1_metadata': circuit1_result.metadata,
                'text2_metadata': circuit2_result.metadata,
                'fidelity_metadata': fidelity_metadata,
                'similarity_score': fidelity,
                'total_time_ms': total_time * 1000,
                'success': True,
                'prd_compliant': {
                    'similarity_under_100ms': total_time * 1000 < 100,
                    'circuits_valid': circuit1_result.success and circuit2_result.success,
                    'fidelity_computation_success': fidelity_metadata.get('success', False)
                }
            }
            
            # Check PRD performance target
            if total_time * 1000 > 100:
                logger.warning(f"Text similarity computation took {total_time*1000:.2f}ms, exceeds PRD target")
            
            return fidelity, combined_metadata
            
        except Exception as e:
            error_msg = f"Text similarity computation failed: {e}"
            logger.error(error_msg)
            return 0.0, {
                'success': False,
                'error': error_msg,
                'total_time_ms': (time.time() - start_time) * 1000
            }
    
    def compute_query_similarities(self, query: str, 
                                 candidates: List[str],
                                 encoding_method: str = 'amplitude') -> List[Tuple[str, float, Dict]]:
        """
        Compute similarities between query and candidate texts.
        
        Supports PRD reranking use case with batch processing optimization.
        
        Args:
            query: Query text
            candidates: List of candidate texts for comparison
            encoding_method: Quantum encoding method
            
        Returns:
            List of (candidate_text, similarity_score, metadata) tuples
        """
        start_time = time.time()
        
        logger.info(f"Computing similarities for query against {len(candidates)} candidates")
        
        try:
            # Convert query to quantum circuit once
            query_result = self.embedding_bridge.text_to_quantum_circuit(
                query, encoding_method=encoding_method
            )
            
            if not query_result.success:
                error_msg = f"Query circuit creation failed: {query_result.error}"
                logger.error(error_msg)
                # Return failed results for all candidates
                return [(candidate, 0.0, {
                    'success': False,
                    'error': error_msg,
                    'candidate_index': i
                }) for i, candidate in enumerate(candidates)]
            
            # Convert all candidates to circuits (batch processing)
            candidate_results = self.embedding_bridge.batch_texts_to_circuits(
                candidates, encoding_method=encoding_method
            )
            
            # Extract successful circuits for batch fidelity computation
            valid_circuits = []
            candidate_mapping = []  # Maps valid circuit index to original candidate index
            
            for i, result in enumerate(candidate_results):
                if result.success:
                    valid_circuits.append(result.circuit)
                    candidate_mapping.append(i)
            
            # Batch compute fidelities for valid circuits
            if valid_circuits:
                fidelity_results = self.swap_test.batch_compute_fidelity(
                    query_result.circuit, valid_circuits
                )
            else:
                fidelity_results = []
            
            # Combine results with original candidate order
            final_results = []
            fidelity_index = 0
            
            for i, (candidate, candidate_result) in enumerate(zip(candidates, candidate_results)):
                if candidate_result.success and fidelity_index < len(fidelity_results):
                    # Use computed fidelity
                    fidelity, fidelity_metadata = fidelity_results[fidelity_index]
                    fidelity_index += 1
                    
                    combined_metadata = {
                        'candidate_index': i,
                        'query_metadata': query_result.metadata,
                        'candidate_metadata': candidate_result.metadata,
                        'fidelity_metadata': fidelity_metadata,
                        'success': True
                    }
                else:
                    # Failed candidate
                    fidelity = 0.0
                    combined_metadata = {
                        'candidate_index': i,
                        'success': False,
                        'error': candidate_result.error if not candidate_result.success else "Unknown error"
                    }
                
                final_results.append((candidate, fidelity, combined_metadata))
            
            # Add batch processing metadata
            total_time = time.time() - start_time
            batch_metadata = {
                'total_candidates': len(candidates),
                'successful_candidates': len(valid_circuits),
                'batch_processing_time_ms': total_time * 1000,
                'avg_time_per_candidate_ms': (total_time / len(candidates)) * 1000 if candidates else 0,
                'prd_compliant': {
                    'batch_under_500ms': total_time * 1000 < 500,  # PRD target for reranking
                    'avg_under_100ms': (total_time / len(candidates)) * 1000 < 100 if candidates else True
                }
            }
            
            # Add batch metadata to all results
            for i, (candidate, similarity, metadata) in enumerate(final_results):
                metadata.update({'batch_metadata': batch_metadata})
                final_results[i] = (candidate, similarity, metadata)
            
            logger.info(f"Batch similarity computation completed: {total_time*1000:.2f}ms for {len(candidates)} candidates")
            
            return final_results
            
        except Exception as e:
            error_msg = f"Batch similarity computation failed: {e}"
            logger.error(error_msg)
            return [(candidate, 0.0, {
                'success': False,
                'error': error_msg,
                'candidate_index': i,
                'total_time_ms': (time.time() - start_time) * 1000
            }) for i, candidate in enumerate(candidates)]
    
    def rank_candidates_by_similarity(self, query: str, 
                                    candidates: List[str],
                                    encoding_method: str = 'amplitude',
                                    top_k: int = None) -> List[Tuple[str, float, Dict]]:
        """
        Rank candidates by quantum fidelity similarity to query.
        
        Implements core reranking functionality as specified in PRD.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            encoding_method: Quantum encoding method
            top_k: Return only top K results (None for all)
            
        Returns:
            List of (candidate, similarity, metadata) sorted by similarity (descending)
        """
        start_time = time.time()
        
        # Compute similarities
        similarity_results = self.compute_query_similarities(
            query, candidates, encoding_method
        )
        
        # Sort by similarity score (descending)
        ranked_results = sorted(
            similarity_results, 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Apply top_k filtering if specified
        if top_k is not None:
            ranked_results = ranked_results[:top_k]
        
        # Add ranking metadata
        total_time = time.time() - start_time
        for i, (candidate, similarity, metadata) in enumerate(ranked_results):
            metadata.update({
                'rank': i + 1,
                'ranking_metadata': {
                    'total_ranking_time_ms': total_time * 1000,
                    'total_candidates_ranked': len(candidates),
                    'top_k_requested': top_k,
                    'encoding_method': encoding_method
                }
            })
            ranked_results[i] = (candidate, similarity, metadata)
        
        logger.info(f"Ranking completed: {len(ranked_results)} results in {total_time*1000:.2f}ms")
        
        return ranked_results
    
    def benchmark_similarity_engine(self, test_texts: List[str] = None) -> Dict:
        """
        Comprehensive benchmarking of the fidelity similarity engine.
        
        Tests performance across different encoding methods and batch sizes.
        """
        if test_texts is None:
            test_texts = [
                "quantum computing algorithms",
                "machine learning models", 
                "artificial intelligence systems",
                "natural language processing"
            ]
        
        results = {
            'encoding_methods': {},
            'batch_sizes': {},
            'summary': {}
        }
        
        logger.info("Starting similarity engine benchmark")
        
        # Test different encoding methods
        for encoding_method in ['amplitude', 'angle', 'dense_angle']:
            start_time = time.time()
            
            try:
                # Test pairwise similarities
                similarity, metadata = self.compute_text_similarity(
                    test_texts[0], test_texts[1], encoding_method
                )
                
                # Test batch processing
                batch_results = self.compute_query_similarities(
                    test_texts[0], test_texts[1:], encoding_method
                )
                
                execution_time = time.time() - start_time
                
                results['encoding_methods'][encoding_method] = {
                    'pairwise_similarity': similarity,
                    'pairwise_metadata': metadata,
                    'batch_results_count': len(batch_results),
                    'batch_success_rate': sum(1 for _, _, meta in batch_results if meta.get('success', False)) / len(batch_results),
                    'total_time_ms': execution_time * 1000,
                    'success': True
                }
                
            except Exception as e:
                results['encoding_methods'][encoding_method] = {
                    'success': False,
                    'error': str(e),
                    'total_time_ms': (time.time() - start_time) * 1000
                }
        
        # Test batch processing with different sizes
        batch_sizes = [2, 4, len(test_texts)]
        for batch_size in batch_sizes:
            if batch_size <= len(test_texts):
                start_time = time.time()
                
                try:
                    batch_candidates = test_texts[1:batch_size+1] if batch_size < len(test_texts) else test_texts[1:]
                    batch_results = self.compute_query_similarities(
                        test_texts[0], batch_candidates
                    )
                    
                    execution_time = time.time() - start_time
                    
                    results['batch_sizes'][batch_size] = {
                        'candidates_processed': len(batch_candidates),
                        'successful_computations': sum(1 for _, _, meta in batch_results if meta.get('success', False)),
                        'total_time_ms': execution_time * 1000,
                        'avg_time_per_candidate_ms': (execution_time / len(batch_candidates)) * 1000,
                        'success': True
                    }
                    
                except Exception as e:
                    results['batch_sizes'][batch_size] = {
                        'success': False,
                        'error': str(e),
                        'total_time_ms': (time.time() - start_time) * 1000
                    }
        
        # Generate summary
        successful_encodings = [method for method, result in results['encoding_methods'].items() 
                               if result.get('success', False)]
        
        if successful_encodings:
            avg_encoding_time = np.mean([
                results['encoding_methods'][method]['total_time_ms'] 
                for method in successful_encodings
            ])
            
            successful_batches = [size for size, result in results['batch_sizes'].items() 
                                 if result.get('success', False)]
            
            avg_batch_time = np.mean([
                results['batch_sizes'][size]['avg_time_per_candidate_ms'] 
                for size in successful_batches
            ]) if successful_batches else 0
            
            results['summary'] = {
                'successful_encoding_methods': len(successful_encodings),
                'avg_encoding_time_ms': avg_encoding_time,
                'successful_batch_sizes': len(successful_batches),
                'avg_batch_time_per_candidate_ms': avg_batch_time,
                'prd_compliance': {
                    'similarity_target_met': avg_encoding_time < 100,
                    'batch_efficiency_good': avg_batch_time < 100
                },
                'benchmark_timestamp': time.time()
            }
        else:
            results['summary'] = {
                'successful_encoding_methods': 0,
                'all_methods_failed': True,
                'benchmark_timestamp': time.time()
            }
        
        logger.info("Similarity engine benchmark completed")
        return results
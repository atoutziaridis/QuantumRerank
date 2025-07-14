"""
Core Quantum Similarity Engine.

This module implements the core quantum similarity engine that combines all previous 
components into a working similarity computation system, as specified in PRD Section 5.1.

Based on:
- PRD Section 5.1: Quantum-Inspired Similarity Engine
- PRD Section 5.2: Integration with Existing RAG Pipeline
- PRD Section 4.3: Performance Targets (<100ms per similarity pair)
- Research papers: Quantum similarity metrics and hybrid algorithms
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Union
import logging
import time
import hashlib
from dataclasses import dataclass
from enum import Enum

from .embeddings import EmbeddingProcessor
from .swap_test import QuantumSWAPTest
from ..ml.parameter_predictor import QuantumParameterPredictor, ParameterPredictorConfig
from ..ml.parameterized_circuits import ParameterizedQuantumCircuits

logger = logging.getLogger(__name__)


class SimilarityMethod(Enum):
    """Available similarity computation methods."""
    QUANTUM_FIDELITY = "quantum_fidelity"
    CLASSICAL_COSINE = "classical_cosine"
    HYBRID_WEIGHTED = "hybrid_weighted"


@dataclass
class SimilarityEngineConfig:
    """Configuration for quantum similarity engine."""
    n_qubits: int = 4
    n_layers: int = 2
    similarity_method: SimilarityMethod = SimilarityMethod.HYBRID_WEIGHTED
    hybrid_weights: Dict[str, float] = None  # Default: {"quantum": 0.7, "classical": 0.3}
    enable_caching: bool = True
    max_cache_size: int = 1000
    performance_monitoring: bool = True
    # Adaptive improvements
    adaptive_weighting: bool = True  # Dynamically adjust quantum/classical weights
    confidence_threshold: float = 0.8  # Confidence threshold for method selection
    use_ensemble: bool = True  # Use ensemble of multiple similarity methods
    
    def __post_init__(self):
        if self.hybrid_weights is None:
            self.hybrid_weights = {"quantum": 0.7, "classical": 0.3}


class QuantumSimilarityEngine:
    """
    Core quantum similarity engine integrating all components.
    
    Implements PRD Section 5.1 with performance optimization and caching.
    Combines classical embeddings, quantum parameter prediction, and SWAP test
    fidelity computation to provide multiple similarity computation methods.
    """
    
    def __init__(self, config: SimilarityEngineConfig = None):
        self.config = config or SimilarityEngineConfig()
        
        # Initialize all components
        self._initialize_components()
        
        # Performance monitoring
        self.performance_stats = {
            'total_comparisons': 0,
            'avg_computation_time_ms': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Caching system
        self._similarity_cache = {} if self.config.enable_caching else None
        
        logger.info(f"Quantum similarity engine initialized: {self.config.similarity_method.value}")
    
    def _initialize_components(self):
        """Initialize all required components."""
        # Embedding processor
        self.embedding_processor = EmbeddingProcessor()
        
        # Quantum components
        self.swap_test = QuantumSWAPTest(self.config.n_qubits)
        
        # Parameter prediction
        predictor_config = ParameterPredictorConfig(
            embedding_dim=self.embedding_processor.config.embedding_dim,
            n_qubits=self.config.n_qubits,
            n_layers=self.config.n_layers
        )
        self.parameter_predictor = QuantumParameterPredictor(predictor_config)
        
        # Circuit building
        self.circuit_builder = ParameterizedQuantumCircuits(
            self.config.n_qubits, self.config.n_layers
        )
        
        logger.debug("All similarity engine components initialized")
    
    def compute_similarity(self, 
                         text1: str, 
                         text2: str,
                         method: Optional[SimilarityMethod] = None) -> Tuple[float, Dict]:
        """
        Compute similarity between two texts using specified method.
        
        Args:
            text1, text2: Input texts
            method: Override default similarity method
            
        Returns:
            Tuple of (similarity_score, metadata)
        """
        start_time = time.time()
        
        method = method or self.config.similarity_method
        
        # Check cache first
        cache_key = self._get_cache_key(text1, text2, method)
        if self._similarity_cache and cache_key in self._similarity_cache:
            self.performance_stats['cache_hits'] += 1
            cached_result = self._similarity_cache[cache_key]
            cached_result[1]['cache_hit'] = True
            return cached_result
        
        self.performance_stats['cache_misses'] += 1
        
        try:
            # Compute similarity based on method
            if method == SimilarityMethod.CLASSICAL_COSINE:
                similarity, metadata = self._compute_classical_similarity(text1, text2)
            elif method == SimilarityMethod.QUANTUM_FIDELITY:
                similarity, metadata = self._compute_quantum_similarity(text1, text2)
            elif method == SimilarityMethod.HYBRID_WEIGHTED:
                similarity, metadata = self._compute_hybrid_similarity(text1, text2)
            else:
                raise ValueError(f"Unknown similarity method: {method}")
            
            # Add timing information
            computation_time = time.time() - start_time
            metadata.update({
                'computation_time_ms': computation_time * 1000,
                'method': method.value,
                'cache_hit': False
            })
            
            # Update performance statistics
            self._update_performance_stats(computation_time)
            
            # Cache result
            if self._similarity_cache:
                self._cache_similarity(cache_key, similarity, metadata)
            
            return similarity, metadata
            
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            error_metadata = {
                'error': str(e),
                'method': method.value,
                'computation_time_ms': (time.time() - start_time) * 1000,
                'success': False
            }
            return 0.0, error_metadata
    
    def _compute_classical_similarity(self, text1: str, text2: str) -> Tuple[float, Dict]:
        """Compute classical cosine similarity."""
        # Generate embeddings
        embeddings = self.embedding_processor.encode_texts([text1, text2])
        
        # Compute cosine similarity
        similarity = self.embedding_processor.compute_classical_similarity(
            embeddings[0], embeddings[1]
        )
        
        metadata = {
            'method_details': 'cosine_similarity',
            'embedding_dim': len(embeddings[0]),
            'success': True
        }
        
        return float(similarity), metadata
    
    def _compute_quantum_similarity(self, text1: str, text2: str) -> Tuple[float, Dict]:
        """Compute quantum fidelity-based similarity."""
        # Generate embeddings
        embeddings = self.embedding_processor.encode_texts([text1, text2])
        embedding_tensor = torch.FloatTensor(embeddings)
        
        # Predict quantum parameters
        with torch.no_grad():
            parameters = self.parameter_predictor(embedding_tensor)
        
        # Create quantum circuits
        circuits = self.circuit_builder.create_batch_circuits(parameters)
        circuit1, circuit2 = circuits[0], circuits[1]
        
        # Compute fidelity
        fidelity, fidelity_metadata = self.swap_test.compute_fidelity(circuit1, circuit2)
        
        metadata = {
            'method_details': 'quantum_fidelity_swap_test',
            'circuit1_depth': circuit1.depth(),
            'circuit2_depth': circuit2.depth(),
            'fidelity_metadata': fidelity_metadata,
            'success': fidelity_metadata.get('success', False)
        }
        
        return fidelity, metadata
    
    def _compute_hybrid_similarity(self, text1: str, text2: str) -> Tuple[float, Dict]:
        """Compute adaptive weighted combination of classical and quantum similarities."""
        # Compute both similarities
        classical_sim, classical_meta = self._compute_classical_similarity(text1, text2)
        quantum_sim, quantum_meta = self._compute_quantum_similarity(text1, text2)
        
        # Adaptive weighting based on confidence and agreement
        if self.config.adaptive_weighting:
            weights = self._compute_adaptive_weights(classical_sim, quantum_sim, classical_meta, quantum_meta)
        else:
            weights = self.config.hybrid_weights
        
        # Ensemble approach if enabled
        if self.config.use_ensemble:
            hybrid_similarity = self._compute_ensemble_similarity(classical_sim, quantum_sim, weights)
        else:
            hybrid_similarity = (
                weights["classical"] * classical_sim + 
                weights["quantum"] * quantum_sim
            )
        
        metadata = {
            'method_details': 'adaptive_hybrid' if self.config.adaptive_weighting else 'hybrid_weighted',
            'classical_similarity': classical_sim,
            'quantum_similarity': quantum_sim,
            'weights': weights,
            'classical_metadata': classical_meta,
            'quantum_metadata': quantum_meta,
            'success': classical_meta.get('success', False) and quantum_meta.get('success', False)
        }
        
        return float(hybrid_similarity), metadata
    
    def _compute_adaptive_weights(self, classical_sim: float, quantum_sim: float, 
                                classical_meta: Dict, quantum_meta: Dict) -> Dict[str, float]:
        """Dynamically compute weights based on method agreement and confidence."""
        
        # Calculate agreement between methods
        agreement = 1.0 - abs(classical_sim - quantum_sim)
        
        # Check method reliability
        quantum_reliable = quantum_meta.get('success', False)
        classical_reliable = classical_meta.get('success', False)
        
        # Base weights
        quantum_weight = self.config.hybrid_weights["quantum"]
        classical_weight = self.config.hybrid_weights["classical"]
        
        # Adjust based on reliability
        if not quantum_reliable and classical_reliable:
            # Quantum failed, rely on classical
            quantum_weight = 0.1
            classical_weight = 0.9
        elif quantum_reliable and not classical_reliable:
            # Classical failed, rely on quantum
            quantum_weight = 0.9
            classical_weight = 0.1
        elif agreement > self.config.confidence_threshold:
            # High agreement - trust quantum more
            quantum_weight = min(0.8, quantum_weight + 0.1)
            classical_weight = 1.0 - quantum_weight
        elif agreement < 0.5:
            # Low agreement - be more conservative, use more classical
            quantum_weight = max(0.3, quantum_weight - 0.2)
            classical_weight = 1.0 - quantum_weight
        
        return {"quantum": quantum_weight, "classical": classical_weight}
    
    def _compute_ensemble_similarity(self, classical_sim: float, quantum_sim: float, 
                                   weights: Dict[str, float]) -> float:
        """Compute ensemble similarity with multiple combination strategies."""
        
        # Strategy 1: Weighted average (original)
        weighted_avg = weights["classical"] * classical_sim + weights["quantum"] * quantum_sim
        
        # Strategy 2: Geometric mean for better handling of extreme values
        if classical_sim > 0 and quantum_sim > 0:
            geometric_mean = (classical_sim ** weights["classical"]) * (quantum_sim ** weights["quantum"])
        else:
            geometric_mean = weighted_avg
        
        # Strategy 3: Max similarity when methods agree, weighted when they disagree
        agreement = 1.0 - abs(classical_sim - quantum_sim)
        if agreement > self.config.confidence_threshold:
            consensus_sim = max(classical_sim, quantum_sim)
        else:
            consensus_sim = weighted_avg
        
        # Final ensemble: weighted combination of strategies
        ensemble_sim = (
            0.5 * weighted_avg +
            0.3 * geometric_mean +
            0.2 * consensus_sim
        )
        
        return ensemble_sim
    
    def compute_similarities_batch(self, 
                                 query: str,
                                 candidates: List[str],
                                 method: Optional[SimilarityMethod] = None) -> List[Tuple[str, float, Dict]]:
        """
        Compute similarities between query and multiple candidates efficiently.
        
        Supports PRD reranking use case with batch optimization.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            method: Similarity method to use
            
        Returns:
            List of (candidate_text, similarity_score, metadata) tuples
        """
        start_time = time.time()
        method = method or self.config.similarity_method
        
        logger.info(f"Computing similarities for {len(candidates)} candidates using {method.value}")
        
        try:
            if method == SimilarityMethod.CLASSICAL_COSINE:
                results = self._batch_classical_similarities(query, candidates)
            elif method == SimilarityMethod.QUANTUM_FIDELITY:
                results = self._batch_quantum_similarities(query, candidates)
            elif method == SimilarityMethod.HYBRID_WEIGHTED:
                results = self._batch_hybrid_similarities(query, candidates)
            else:
                raise ValueError(f"Unknown similarity method: {method}")
            
            # Add batch timing to all results
            batch_time = time.time() - start_time
            for i, (candidate, similarity, metadata) in enumerate(results):
                metadata.update({
                    'batch_total_time_ms': batch_time * 1000,
                    'batch_per_item_ms': (batch_time / len(candidates)) * 1000
                })
                results[i] = (candidate, similarity, metadata)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch similarity computation failed: {e}")
            # Return error results for all candidates
            error_metadata = {
                'error': str(e),
                'method': method.value,
                'success': False
            }
            return [(candidate, 0.0, error_metadata.copy()) for candidate in candidates]
    
    def _batch_classical_similarities(self, 
                                    query: str, 
                                    candidates: List[str]) -> List[Tuple[str, float, Dict]]:
        """Efficient batch classical similarity computation."""
        # Batch encode all texts
        all_texts = [query] + candidates
        embeddings = self.embedding_processor.encode_texts(all_texts)
        
        query_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]
        
        results = []
        for i, (candidate, candidate_embedding) in enumerate(zip(candidates, candidate_embeddings)):
            similarity = self.embedding_processor.compute_classical_similarity(
                query_embedding, candidate_embedding
            )
            
            metadata = {
                'method': 'classical_cosine',
                'batch_index': i,
                'embedding_dim': len(candidate_embedding),
                'success': True
            }
            
            results.append((candidate, float(similarity), metadata))
        
        return results
    
    def _batch_quantum_similarities(self, 
                                   query: str,
                                   candidates: List[str]) -> List[Tuple[str, float, Dict]]:
        """Efficient batch quantum similarity computation."""
        # Batch encode all texts
        all_texts = [query] + candidates
        embeddings = self.embedding_processor.encode_texts(all_texts)
        embedding_tensor = torch.FloatTensor(embeddings)
        
        # Batch predict parameters
        with torch.no_grad():
            parameters = self.parameter_predictor(embedding_tensor)
        
        # Create circuits
        circuits = self.circuit_builder.create_batch_circuits(parameters)
        query_circuit = circuits[0]
        candidate_circuits = circuits[1:]
        
        # Batch compute fidelities
        fidelity_results = self.swap_test.batch_compute_fidelity(query_circuit, candidate_circuits)
        
        results = []
        for i, (candidate, (fidelity, fidelity_metadata)) in enumerate(zip(candidates, fidelity_results)):
            metadata = {
                'method': 'quantum_fidelity',
                'batch_index': i,
                'fidelity_metadata': fidelity_metadata,
                'circuit_depth': candidate_circuits[i].depth(),
                'success': fidelity_metadata.get('success', False)
            }
            
            results.append((candidate, fidelity, metadata))
        
        return results
    
    def _batch_hybrid_similarities(self, 
                                 query: str,
                                 candidates: List[str]) -> List[Tuple[str, float, Dict]]:
        """Efficient batch hybrid similarity computation."""
        # Compute both classical and quantum similarities in batch
        classical_results = self._batch_classical_similarities(query, candidates)
        quantum_results = self._batch_quantum_similarities(query, candidates)
        
        results = []
        weights = self.config.hybrid_weights
        
        for (cand_c, classical_sim, classical_meta), (cand_q, quantum_sim, quantum_meta) in zip(classical_results, quantum_results):
            assert cand_c == cand_q  # Sanity check
            
            # Weighted combination
            hybrid_similarity = (
                weights["classical"] * classical_sim + 
                weights["quantum"] * quantum_sim
            )
            
            metadata = {
                'method': 'hybrid_weighted',
                'classical_similarity': classical_sim,
                'quantum_similarity': quantum_sim,
                'weights': weights,
                'classical_metadata': classical_meta,
                'quantum_metadata': quantum_meta,
                'success': classical_meta.get('success', False) and quantum_meta.get('success', False)
            }
            
            results.append((cand_c, float(hybrid_similarity), metadata))
        
        return results
    
    def rerank_candidates(self, 
                        query: str,
                        candidates: List[str],
                        top_k: Optional[int] = None,
                        method: Optional[SimilarityMethod] = None) -> List[Tuple[str, float, Dict]]:
        """
        Rerank candidates based on quantum-inspired similarity.
        
        Main interface for RAG reranking as specified in PRD Section 5.2.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of top results to return
            method: Similarity method to use
            
        Returns:
            List of (candidate, similarity, metadata) sorted by similarity
        """
        # Compute similarities
        similarities = self.compute_similarities_batch(query, candidates, method)
        
        # Sort by similarity (descending)
        ranked_results = sorted(similarities, key=lambda x: x[1], reverse=True)
        
        # Apply top-k filtering
        if top_k is not None:
            ranked_results = ranked_results[:top_k]
        
        # Update ranks in metadata
        for i, (candidate, similarity, metadata) in enumerate(ranked_results):
            metadata['final_rank'] = i + 1
        
        logger.info(f"Reranked {len(candidates)} candidates, returning top {len(ranked_results)}")
        
        return ranked_results
    
    def _get_cache_key(self, text1: str, text2: str, method: SimilarityMethod) -> str:
        """Generate cache key for similarity computation."""
        content = f"{text1}|||{text2}|||{method.value}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _cache_similarity(self, cache_key: str, similarity: float, metadata: Dict):
        """Cache similarity result with size management."""
        if len(self._similarity_cache) >= self.config.max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._similarity_cache))
            del self._similarity_cache[oldest_key]
        
        self._similarity_cache[cache_key] = (similarity, metadata.copy())
    
    def _update_performance_stats(self, computation_time: float):
        """Update performance statistics."""
        self.performance_stats['total_comparisons'] += 1
        
        # Update rolling average
        n = self.performance_stats['total_comparisons']
        current_avg = self.performance_stats['avg_computation_time_ms']
        new_time_ms = computation_time * 1000
        
        self.performance_stats['avg_computation_time_ms'] = (
            (current_avg * (n - 1) + new_time_ms) / n
        )
    
    def get_performance_report(self) -> Dict:
        """
        Generate comprehensive performance report.
        
        Returns metrics aligned with PRD performance targets.
        """
        stats = self.performance_stats.copy()
        
        # Add cache statistics
        if self._similarity_cache is not None:
            total_requests = stats['cache_hits'] + stats['cache_misses']
            cache_hit_rate = stats['cache_hits'] / total_requests if total_requests > 0 else 0
            
            stats.update({
                'cache_hit_rate': cache_hit_rate,
                'cache_size': len(self._similarity_cache),
                'cache_max_size': self.config.max_cache_size
            })
        
        # Performance analysis
        meets_prd_target = stats['avg_computation_time_ms'] < 100  # PRD: <100ms per pair
        
        stats.update({
            'meets_prd_latency_target': meets_prd_target,
            'prd_target_ms': 100,
            'performance_ratio': stats['avg_computation_time_ms'] / 100
        })
        
        return stats
    
    def benchmark_similarity_methods(self, test_texts: Optional[List[str]] = None) -> Dict:
        """
        Benchmark all similarity methods for comparison.
        
        Returns performance comparison across methods.
        """
        if test_texts is None:
            test_texts = [
                "quantum computing applications in machine learning",
                "classical algorithms for information retrieval",
                "hybrid quantum-classical optimization methods",
                "natural language processing with transformers"
            ]
        
        methods = [
            SimilarityMethod.CLASSICAL_COSINE,
            SimilarityMethod.QUANTUM_FIDELITY,
            SimilarityMethod.HYBRID_WEIGHTED
        ]
        
        benchmark_results = {}
        
        for method in methods:
            method_results = {
                'pairwise_times': [],
                'batch_times': [],
                'similarities': []
            }
            
            # Test pairwise similarities
            for i in range(len(test_texts)):
                for j in range(i + 1, len(test_texts)):
                    start_time = time.time()
                    similarity, metadata = self.compute_similarity(
                        test_texts[i], test_texts[j], method
                    )
                    computation_time = time.time() - start_time
                    
                    method_results['pairwise_times'].append(computation_time * 1000)
                    method_results['similarities'].append(similarity)
            
            # Test batch processing
            query = test_texts[0]
            candidates = test_texts[1:]
            
            start_time = time.time()
            batch_results = self.compute_similarities_batch(query, candidates, method)
            batch_time = time.time() - start_time
            
            method_results['batch_times'].append(batch_time * 1000)
            
            # Summary statistics
            method_results['avg_pairwise_time_ms'] = np.mean(method_results['pairwise_times'])
            method_results['max_pairwise_time_ms'] = np.max(method_results['pairwise_times'])
            method_results['avg_similarity'] = np.mean(method_results['similarities'])
            method_results['batch_total_time_ms'] = batch_time * 1000
            method_results['batch_per_item_ms'] = (batch_time / len(candidates)) * 1000
            method_results['meets_prd_target'] = method_results['max_pairwise_time_ms'] < 100
            
            benchmark_results[method.value] = method_results
            
            logger.info(f"{method.value}: avg={method_results['avg_pairwise_time_ms']:.2f}ms, "
                       f"max={method_results['max_pairwise_time_ms']:.2f}ms")
        
        return benchmark_results
    
    def clear_cache(self):
        """Clear similarity cache."""
        if self._similarity_cache:
            self._similarity_cache.clear()
            logger.info("Similarity cache cleared")
    
    def get_cache_statistics(self) -> Dict:
        """Get detailed cache statistics."""
        if not self._similarity_cache:
            return {'caching_enabled': False}
        
        total_requests = self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']
        hit_rate = self.performance_stats['cache_hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'caching_enabled': True,
            'cache_size': len(self._similarity_cache),
            'max_cache_size': self.config.max_cache_size,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }
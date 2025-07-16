"""
Hybrid Quantum-Classical Pipeline for Intelligent Reranking.

This module implements the unified pipeline that combines classical and quantum rerankers
with intelligent routing based on complexity assessment and performance constraints.

Based on:
- QMMR-02 task specification
- Quantum-classical hybrid architecture
- Performance requirements (<500ms batch processing)
- Medical domain optimization
"""

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

from .complexity_assessment_engine import ComplexityAssessmentEngine
from .routing_decision_engine import RoutingDecisionEngine, RoutingDecision
from .complexity_metrics import ComplexityAssessmentResult, ComplexityMetrics
from ..config.routing_config import HybridPipelineConfig, RoutingMethod
from ..core.quantum_similarity_engine import QuantumSimilarityEngine, SimilarityMethod
from ..core.multimodal_embedding_processor import MultimodalEmbeddingProcessor
from ..evaluation.industry_standard_evaluation import IndustryStandardEvaluator

logger = logging.getLogger(__name__)


@dataclass
class RerankingResult:
    """Result of reranking operation."""
    
    # Reranked candidates with scores
    reranked_candidates: List[Tuple[Dict[str, Any], float, Dict[str, Any]]]
    
    # Method used for reranking
    method: RoutingMethod
    
    # Performance metrics
    processing_time_ms: float
    memory_usage_mb: float
    
    # Quality metrics
    quantum_advantage_score: float = 0.0
    confidence_score: float = 0.0
    
    # Metadata
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class HybridRerankingResult:
    """Result of hybrid reranking with comprehensive metadata."""
    
    # Final reranked candidates
    reranked_candidates: List[Tuple[Dict[str, Any], float, Dict[str, Any]]]
    
    # Routing information
    routing_decision: RoutingDecision
    complexity_assessment: ComplexityAssessmentResult
    
    # Performance metrics
    processing_time_ms: float
    memory_usage_mb: float
    
    # Method-specific results
    classical_result: Optional[RerankingResult] = None
    quantum_result: Optional[RerankingResult] = None
    
    # Quality metrics
    quantum_advantage_score: float = 0.0
    final_confidence_score: float = 0.0
    
    # Pipeline metadata
    pipeline_version: str = "1.0"
    success: bool = True
    error_message: Optional[str] = None


class ClassicalReranker:
    """Classical reranking implementation using sentence transformers."""
    
    def __init__(self, config: HybridPipelineConfig):
        self.config = config
        self.multimodal_processor = MultimodalEmbeddingProcessor()
        
        # Initialize classical models based on config
        self.reranker_type = config.classical_reranker_type
        self.model_name = config.classical_model_name
        
        # Performance tracking
        self.processing_stats = {
            'total_queries': 0,
            'avg_processing_time_ms': 0.0,
            'avg_memory_usage_mb': 0.0
        }
        
        logger.debug(f"ClassicalReranker initialized with {self.reranker_type}")
    
    def rerank(self, query: Dict[str, Any], candidates: List[Dict[str, Any]], 
              top_k: int = 10) -> RerankingResult:
        """Execute classical reranking."""
        start_time = time.time()
        
        try:
            # Process query
            if 'text' in query:
                # Text-based reranking
                similarities = self._compute_text_similarities(query, candidates)
            else:
                # Multimodal reranking
                similarities = self._compute_multimodal_similarities(query, candidates)
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Take top-k results
            reranked_candidates = similarities[:top_k]
            
            # Performance tracking
            processing_time = (time.time() - start_time) * 1000
            memory_usage = self._estimate_memory_usage(len(candidates))
            
            self._update_processing_stats(processing_time, memory_usage)
            
            return RerankingResult(
                reranked_candidates=reranked_candidates,
                method=RoutingMethod.CLASSICAL,
                processing_time_ms=processing_time,
                memory_usage_mb=memory_usage,
                confidence_score=self._compute_confidence_score(similarities),
                success=True
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Classical reranking failed: {e}")
            
            return RerankingResult(
                reranked_candidates=[],
                method=RoutingMethod.CLASSICAL,
                processing_time_ms=processing_time,
                memory_usage_mb=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _compute_text_similarities(self, query: Dict[str, Any], 
                                 candidates: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float, Dict[str, Any]]]:
        """Compute text-based similarities."""
        query_text = query.get('text', '')
        results = []
        
        # Use multimodal processor for text encoding
        query_result = self.multimodal_processor.encode_multimodal({'text': query_text})
        
        for candidate in candidates:
            candidate_text = candidate.get('text', '')
            candidate_result = self.multimodal_processor.encode_multimodal({'text': candidate_text})
            
            # Compute cosine similarity
            if query_result.text_embedding is not None and candidate_result.text_embedding is not None:
                similarity = np.dot(query_result.text_embedding, candidate_result.text_embedding) / (
                    np.linalg.norm(query_result.text_embedding) * np.linalg.norm(candidate_result.text_embedding)
                )
            else:
                similarity = 0.0
            
            metadata = {
                'similarity_type': 'text_cosine',
                'query_modalities': query_result.modalities_used,
                'candidate_modalities': candidate_result.modalities_used
            }
            
            results.append((candidate, float(similarity), metadata))
        
        return results
    
    def _compute_multimodal_similarities(self, query: Dict[str, Any], 
                                       candidates: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float, Dict[str, Any]]]:
        """Compute multimodal similarities."""
        results = []
        
        # Process query
        query_result = self.multimodal_processor.encode_multimodal(query)
        
        for candidate in candidates:
            candidate_result = self.multimodal_processor.encode_multimodal(candidate)
            
            # Compute fused similarity
            if query_result.fused_embedding is not None and candidate_result.fused_embedding is not None:
                similarity = np.dot(query_result.fused_embedding, candidate_result.fused_embedding) / (
                    np.linalg.norm(query_result.fused_embedding) * np.linalg.norm(candidate_result.fused_embedding)
                )
            else:
                # Fallback to text similarity
                if query_result.text_embedding is not None and candidate_result.text_embedding is not None:
                    similarity = np.dot(query_result.text_embedding, candidate_result.text_embedding) / (
                        np.linalg.norm(query_result.text_embedding) * np.linalg.norm(candidate_result.text_embedding)
                    )
                else:
                    similarity = 0.0
            
            metadata = {
                'similarity_type': 'multimodal_fused',
                'query_modalities': query_result.modalities_used,
                'candidate_modalities': candidate_result.modalities_used,
                'compression_ratio': self.multimodal_processor.quantum_compressor.get_compression_ratio() if self.multimodal_processor.quantum_compressor else 0
            }
            
            results.append((candidate, float(similarity), metadata))
        
        return results
    
    def _estimate_memory_usage(self, num_candidates: int) -> float:
        """Estimate memory usage for classical reranking."""
        # Base memory for models
        base_memory = 512.0  # MB
        
        # Additional memory per candidate
        memory_per_candidate = 2.0  # MB
        
        return base_memory + (num_candidates * memory_per_candidate)
    
    def _compute_confidence_score(self, similarities: List[Tuple[Dict[str, Any], float, Dict[str, Any]]]) -> float:
        """Compute confidence score for classical reranking."""
        if not similarities:
            return 0.0
        
        scores = [sim[1] for sim in similarities]
        
        # Confidence based on score distribution
        if len(scores) > 1:
            score_std = np.std(scores)
            score_mean = np.mean(scores)
            
            # Higher confidence when scores are well-separated
            confidence = min(score_std * 2.0, 1.0)
        else:
            confidence = scores[0]
        
        return confidence
    
    def _update_processing_stats(self, processing_time: float, memory_usage: float):
        """Update processing statistics."""
        self.processing_stats['total_queries'] += 1
        n = self.processing_stats['total_queries']
        
        # Update averages
        current_time_avg = self.processing_stats['avg_processing_time_ms']
        self.processing_stats['avg_processing_time_ms'] = (
            (current_time_avg * (n - 1) + processing_time) / n
        )
        
        current_memory_avg = self.processing_stats['avg_memory_usage_mb']
        self.processing_stats['avg_memory_usage_mb'] = (
            (current_memory_avg * (n - 1) + memory_usage) / n
        )


class QuantumReranker:
    """Quantum reranking implementation using quantum similarity engine."""
    
    def __init__(self, config: HybridPipelineConfig):
        self.config = config
        
        # Initialize quantum similarity engine
        from ..core.quantum_similarity_engine import SimilarityEngineConfig
        
        quantum_config = SimilarityEngineConfig(
            n_qubits=config.quantum_n_qubits,
            n_layers=config.quantum_n_layers,
            enable_multimodal=config.quantum_enable_multimodal,
            similarity_method=SimilarityMethod.MULTIMODAL_QUANTUM if config.quantum_enable_multimodal else SimilarityMethod.QUANTUM_FIDELITY
        )
        
        self.quantum_engine = QuantumSimilarityEngine(quantum_config)
        
        # Performance tracking
        self.processing_stats = {
            'total_queries': 0,
            'avg_processing_time_ms': 0.0,
            'avg_memory_usage_mb': 0.0
        }
        
        logger.debug(f"QuantumReranker initialized with {config.quantum_n_qubits} qubits")
    
    def rerank(self, query: Dict[str, Any], candidates: List[Dict[str, Any]], 
              top_k: int = 10) -> RerankingResult:
        """Execute quantum reranking."""
        start_time = time.time()
        
        try:
            # Use quantum similarity engine
            if self.config.quantum_enable_multimodal:
                similarities = self._compute_multimodal_quantum_similarities(query, candidates)
            else:
                similarities = self._compute_quantum_similarities(query, candidates)
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Take top-k results
            reranked_candidates = similarities[:top_k]
            
            # Performance tracking
            processing_time = (time.time() - start_time) * 1000
            memory_usage = self._estimate_memory_usage(len(candidates))
            
            self._update_processing_stats(processing_time, memory_usage)
            
            return RerankingResult(
                reranked_candidates=reranked_candidates,
                method=RoutingMethod.QUANTUM,
                processing_time_ms=processing_time,
                memory_usage_mb=memory_usage,
                quantum_advantage_score=self._compute_quantum_advantage_score(similarities),
                confidence_score=self._compute_confidence_score(similarities),
                success=True
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Quantum reranking failed: {e}")
            
            return RerankingResult(
                reranked_candidates=[],
                method=RoutingMethod.QUANTUM,
                processing_time_ms=processing_time,
                memory_usage_mb=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _compute_multimodal_quantum_similarities(self, query: Dict[str, Any], 
                                               candidates: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float, Dict[str, Any]]]:
        """Compute multimodal quantum similarities."""
        results = []
        
        for candidate in candidates:
            try:
                similarity, metadata = self.quantum_engine.compute_multimodal_similarity(query, candidate)
                results.append((candidate, similarity, metadata))
            except Exception as e:
                logger.warning(f"Failed to compute quantum similarity for candidate: {e}")
                results.append((candidate, 0.0, {'error': str(e)}))
        
        return results
    
    def _compute_quantum_similarities(self, query: Dict[str, Any], 
                                    candidates: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float, Dict[str, Any]]]:
        """Compute quantum similarities for text-only queries."""
        results = []
        
        query_text = query.get('text', '')
        candidate_texts = [c.get('text', '') for c in candidates]
        
        # Batch compute similarities
        batch_results = self.quantum_engine.compute_similarities_batch(
            query_text, candidate_texts, method=SimilarityMethod.QUANTUM_FIDELITY
        )
        
        for (candidate, similarity, metadata), original_candidate in zip(batch_results, candidates):
            results.append((original_candidate, similarity, metadata))
        
        return results
    
    def _estimate_memory_usage(self, num_candidates: int) -> float:
        """Estimate memory usage for quantum reranking."""
        # Base memory for quantum engine
        base_memory = 1024.0  # MB
        
        # Additional memory per candidate (quantum circuits)
        memory_per_candidate = 5.0  # MB
        
        return base_memory + (num_candidates * memory_per_candidate)
    
    def _compute_quantum_advantage_score(self, similarities: List[Tuple[Dict[str, Any], float, Dict[str, Any]]]) -> float:
        """Compute quantum advantage score."""
        if not similarities:
            return 0.0
        
        # Extract quantum-specific metadata
        quantum_features = []
        
        for _, similarity, metadata in similarities:
            if 'quantum_advantage_score' in metadata:
                quantum_features.append(metadata['quantum_advantage_score'])
            elif 'fidelity_metadata' in metadata:
                # Extract fidelity information
                fidelity_info = metadata['fidelity_metadata']
                if 'quantum_advantage' in fidelity_info:
                    quantum_features.append(fidelity_info['quantum_advantage'])
        
        # Average quantum advantage
        return np.mean(quantum_features) if quantum_features else 0.5
    
    def _compute_confidence_score(self, similarities: List[Tuple[Dict[str, Any], float, Dict[str, Any]]]) -> float:
        """Compute confidence score for quantum reranking."""
        if not similarities:
            return 0.0
        
        scores = [sim[1] for sim in similarities]
        
        # Confidence based on quantum fidelity distribution
        if len(scores) > 1:
            score_entropy = -np.sum([s * np.log2(s + 1e-10) for s in scores if s > 0])
            normalized_entropy = score_entropy / np.log2(len(scores))
            
            # Higher confidence with lower entropy (more decisive)
            confidence = 1.0 - normalized_entropy
        else:
            confidence = scores[0]
        
        return confidence
    
    def _update_processing_stats(self, processing_time: float, memory_usage: float):
        """Update processing statistics."""
        self.processing_stats['total_queries'] += 1
        n = self.processing_stats['total_queries']
        
        # Update averages
        current_time_avg = self.processing_stats['avg_processing_time_ms']
        self.processing_stats['avg_processing_time_ms'] = (
            (current_time_avg * (n - 1) + processing_time) / n
        )
        
        current_memory_avg = self.processing_stats['avg_memory_usage_mb']
        self.processing_stats['avg_memory_usage_mb'] = (
            (current_memory_avg * (n - 1) + memory_usage) / n
        )


class HybridQuantumClassicalPipeline:
    """
    Unified pipeline combining classical and quantum reranking with intelligent routing.
    
    This is the main interface for the hybrid reranking system, implementing
    the complete QMMR-02 functionality.
    """
    
    def __init__(self, config: HybridPipelineConfig = None):
        self.config = config or HybridPipelineConfig()
        
        # Initialize components
        self.complexity_engine = ComplexityAssessmentEngine()
        self.routing_engine = RoutingDecisionEngine(self.config.routing_config)
        self.classical_reranker = ClassicalReranker(self.config)
        self.quantum_reranker = QuantumReranker(self.config)
        
        # Performance monitoring
        self.pipeline_stats = {
            'total_queries': 0,
            'quantum_queries': 0,
            'classical_queries': 0,
            'hybrid_queries': 0,
            'avg_processing_time_ms': 0.0,
            'avg_memory_usage_mb': 0.0,
            'routing_accuracy': 0.0
        }
        
        # Result caching
        self._result_cache = {} if config.enable_result_caching else None
        
        logger.info(f"HybridQuantumClassicalPipeline initialized")
    
    def rerank(self, query: Dict[str, Any], candidates: List[Dict[str, Any]], 
              top_k: int = None) -> HybridRerankingResult:
        """
        Perform hybrid reranking with intelligent routing.
        
        Args:
            query: Multimodal query data
            candidates: List of candidate documents
            top_k: Number of top results to return
            
        Returns:
            HybridRerankingResult with routing decision and results
        """
        start_time = time.time()
        top_k = top_k or self.config.default_top_k
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(query, candidates, top_k)
            if self._result_cache and cache_key in self._result_cache:
                cached_result = self._result_cache[cache_key]
                cached_result.processing_time_ms = (time.time() - start_time) * 1000
                return cached_result
            
            # Assess complexity
            complexity_result = self.complexity_engine.assess_complexity(query, candidates)
            
            # Make routing decision
            routing_decision = self.routing_engine.route_query(complexity_result)
            
            # Execute reranking based on routing decision
            if routing_decision.method == RoutingMethod.CLASSICAL:
                reranking_result = self._execute_classical_reranking(query, candidates, top_k)
                classical_result = reranking_result
                quantum_result = None
            elif routing_decision.method == RoutingMethod.QUANTUM:
                reranking_result = self._execute_quantum_reranking(query, candidates, top_k)
                classical_result = None
                quantum_result = reranking_result
            else:  # HYBRID
                reranking_result = self._execute_hybrid_reranking(query, candidates, top_k)
                classical_result = reranking_result.classical_result if hasattr(reranking_result, 'classical_result') else None
                quantum_result = reranking_result.quantum_result if hasattr(reranking_result, 'quantum_result') else None
            
            # Performance tracking
            processing_time = (time.time() - start_time) * 1000
            memory_usage = routing_decision.estimated_memory_mb
            
            self._update_pipeline_stats(routing_decision.method, processing_time, memory_usage)
            
            # Create result
            result = HybridRerankingResult(
                reranked_candidates=reranking_result.reranked_candidates,
                routing_decision=routing_decision,
                complexity_assessment=complexity_result,
                processing_time_ms=processing_time,
                memory_usage_mb=memory_usage,
                classical_result=classical_result,
                quantum_result=quantum_result,
                quantum_advantage_score=routing_decision.quantum_advantage_score,
                final_confidence_score=reranking_result.confidence_score,
                success=True
            )
            
            # Cache result
            if self._result_cache:
                self._cache_result(cache_key, result)
            
            # Record performance for adaptive learning
            self.routing_engine.record_routing_performance(
                routing_decision, processing_time, memory_usage, reranking_result.confidence_score
            )
            
            return result
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Hybrid reranking failed: {e}")
            
            # Return error result
            return HybridRerankingResult(
                reranked_candidates=[],
                routing_decision=RoutingDecision(
                    method=RoutingMethod.CLASSICAL,
                    confidence=0.0,
                    complexity_score=0.0,
                    complexity_factors={},
                    estimated_latency_ms=processing_time,
                    estimated_memory_mb=0.0,
                    reasoning=f"Error: {str(e)}",
                    decision_time_ms=processing_time
                ),
                complexity_assessment=ComplexityAssessmentResult(
                    query_complexity=ComplexityMetrics(),
                    candidate_complexities=[],
                    overall_complexity=ComplexityMetrics(),
                    routing_recommendation="classical",
                    routing_confidence=0.0,
                    assessment_time_ms=processing_time,
                    success=False,
                    error_message=str(e)
                ),
                processing_time_ms=processing_time,
                memory_usage_mb=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _execute_classical_reranking(self, query: Dict[str, Any], candidates: List[Dict[str, Any]], 
                                   top_k: int) -> RerankingResult:
        """Execute classical reranking."""
        return self.classical_reranker.rerank(query, candidates, top_k)
    
    def _execute_quantum_reranking(self, query: Dict[str, Any], candidates: List[Dict[str, Any]], 
                                 top_k: int) -> RerankingResult:
        """Execute quantum reranking."""
        return self.quantum_reranker.rerank(query, candidates, top_k)
    
    def _execute_hybrid_reranking(self, query: Dict[str, Any], candidates: List[Dict[str, Any]], 
                                top_k: int) -> RerankingResult:
        """Execute hybrid reranking combining classical and quantum methods."""
        # Execute both methods
        classical_result = self.classical_reranker.rerank(query, candidates, top_k * 2)  # Get more for fusion
        quantum_result = self.quantum_reranker.rerank(query, candidates, top_k * 2)
        
        # Combine results based on configuration
        if self.config.hybrid_combination_method == "weighted_average":
            combined_results = self._weighted_average_fusion(classical_result, quantum_result)
        elif self.config.hybrid_combination_method == "rank_fusion":
            combined_results = self._rank_fusion(classical_result, quantum_result)
        else:  # ensemble
            combined_results = self._ensemble_fusion(classical_result, quantum_result)
        
        # Take top-k from combined results
        combined_results.sort(key=lambda x: x[1], reverse=True)
        final_candidates = combined_results[:top_k]
        
        # Create hybrid result
        processing_time = classical_result.processing_time_ms + quantum_result.processing_time_ms
        memory_usage = classical_result.memory_usage_mb + quantum_result.memory_usage_mb
        
        return RerankingResult(
            reranked_candidates=final_candidates,
            method=RoutingMethod.HYBRID,
            processing_time_ms=processing_time,
            memory_usage_mb=memory_usage,
            quantum_advantage_score=quantum_result.quantum_advantage_score,
            confidence_score=(classical_result.confidence_score + quantum_result.confidence_score) / 2,
            success=classical_result.success and quantum_result.success
        )
    
    def _weighted_average_fusion(self, classical_result: RerankingResult, 
                               quantum_result: RerankingResult) -> List[Tuple[Dict[str, Any], float, Dict[str, Any]]]:
        """Combine results using weighted average."""
        # Create score dictionaries
        classical_scores = {id(candidate): score for candidate, score, _ in classical_result.reranked_candidates}
        quantum_scores = {id(candidate): score for candidate, score, _ in quantum_result.reranked_candidates}
        
        # Combine scores
        combined_results = []
        all_candidates = set(classical_scores.keys()) | set(quantum_scores.keys())
        
        for candidate_id in all_candidates:
            # Find original candidate
            candidate = None
            for cand, _, _ in classical_result.reranked_candidates:
                if id(cand) == candidate_id:
                    candidate = cand
                    break
            if candidate is None:
                for cand, _, _ in quantum_result.reranked_candidates:
                    if id(cand) == candidate_id:
                        candidate = cand
                        break
            
            if candidate is not None:
                classical_score = classical_scores.get(candidate_id, 0.0)
                quantum_score = quantum_scores.get(candidate_id, 0.0)
                
                # Weighted combination
                combined_score = (
                    self.config.hybrid_weight_classical * classical_score +
                    self.config.hybrid_weight_quantum * quantum_score
                )
                
                metadata = {
                    'fusion_method': 'weighted_average',
                    'classical_score': classical_score,
                    'quantum_score': quantum_score,
                    'classical_weight': self.config.hybrid_weight_classical,
                    'quantum_weight': self.config.hybrid_weight_quantum
                }
                
                combined_results.append((candidate, combined_score, metadata))
        
        return combined_results
    
    def _rank_fusion(self, classical_result: RerankingResult, 
                    quantum_result: RerankingResult) -> List[Tuple[Dict[str, Any], float, Dict[str, Any]]]:
        """Combine results using rank fusion."""
        # Create rank dictionaries
        classical_ranks = {id(candidate): rank for rank, (candidate, _, _) in enumerate(classical_result.reranked_candidates)}
        quantum_ranks = {id(candidate): rank for rank, (candidate, _, _) in enumerate(quantum_result.reranked_candidates)}
        
        # Combine ranks
        combined_results = []
        all_candidates = set(classical_ranks.keys()) | set(quantum_ranks.keys())
        
        for candidate_id in all_candidates:
            # Find original candidate
            candidate = None
            for cand, _, _ in classical_result.reranked_candidates:
                if id(cand) == candidate_id:
                    candidate = cand
                    break
            if candidate is None:
                for cand, _, _ in quantum_result.reranked_candidates:
                    if id(cand) == candidate_id:
                        candidate = cand
                        break
            
            if candidate is not None:
                classical_rank = classical_ranks.get(candidate_id, len(classical_result.reranked_candidates))
                quantum_rank = quantum_ranks.get(candidate_id, len(quantum_result.reranked_candidates))
                
                # Reciprocal rank fusion
                combined_score = (
                    self.config.hybrid_weight_classical / (classical_rank + 1) +
                    self.config.hybrid_weight_quantum / (quantum_rank + 1)
                )
                
                metadata = {
                    'fusion_method': 'rank_fusion',
                    'classical_rank': classical_rank,
                    'quantum_rank': quantum_rank
                }
                
                combined_results.append((candidate, combined_score, metadata))
        
        return combined_results
    
    def _ensemble_fusion(self, classical_result: RerankingResult, 
                        quantum_result: RerankingResult) -> List[Tuple[Dict[str, Any], float, Dict[str, Any]]]:
        """Combine results using ensemble method."""
        # Use weighted average as base
        weighted_results = self._weighted_average_fusion(classical_result, quantum_result)
        
        # Apply confidence-based weighting
        for i, (candidate, score, metadata) in enumerate(weighted_results):
            confidence_factor = (classical_result.confidence_score + quantum_result.confidence_score) / 2
            adjusted_score = score * confidence_factor
            
            metadata.update({
                'fusion_method': 'ensemble',
                'confidence_factor': confidence_factor,
                'original_score': score
            })
            
            weighted_results[i] = (candidate, adjusted_score, metadata)
        
        return weighted_results
    
    def _get_cache_key(self, query: Dict[str, Any], candidates: List[Dict[str, Any]], top_k: int) -> str:
        """Generate cache key for result caching."""
        import hashlib
        
        # Create deterministic hash
        query_str = str(sorted(query.items()))
        candidates_str = str([sorted(c.items()) for c in candidates[:10]])  # Limit for performance
        
        content = f"{query_str}|||{candidates_str}|||{top_k}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _cache_result(self, cache_key: str, result: HybridRerankingResult):
        """Cache reranking result."""
        if len(self._result_cache) >= self.config.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._result_cache))
            del self._result_cache[oldest_key]
        
        self._result_cache[cache_key] = result
    
    def _update_pipeline_stats(self, method: RoutingMethod, processing_time: float, memory_usage: float):
        """Update pipeline statistics."""
        self.pipeline_stats['total_queries'] += 1
        
        if method == RoutingMethod.QUANTUM:
            self.pipeline_stats['quantum_queries'] += 1
        elif method == RoutingMethod.CLASSICAL:
            self.pipeline_stats['classical_queries'] += 1
        else:
            self.pipeline_stats['hybrid_queries'] += 1
        
        # Update averages
        n = self.pipeline_stats['total_queries']
        current_time_avg = self.pipeline_stats['avg_processing_time_ms']
        self.pipeline_stats['avg_processing_time_ms'] = (
            (current_time_avg * (n - 1) + processing_time) / n
        )
        
        current_memory_avg = self.pipeline_stats['avg_memory_usage_mb']
        self.pipeline_stats['avg_memory_usage_mb'] = (
            (current_memory_avg * (n - 1) + memory_usage) / n
        )
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        stats = self.pipeline_stats.copy()
        
        # Add distribution percentages
        total = stats['total_queries']
        if total > 0:
            stats['quantum_percentage'] = stats['quantum_queries'] / total * 100
            stats['classical_percentage'] = stats['classical_queries'] / total * 100
            stats['hybrid_percentage'] = stats['hybrid_queries'] / total * 100
        
        # Add component statistics
        stats['complexity_assessment_stats'] = self.complexity_engine.get_assessment_stats()
        stats['routing_stats'] = self.routing_engine.get_routing_stats()
        
        # Add performance analysis
        stats['meets_latency_target'] = stats['avg_processing_time_ms'] < self.config.max_total_latency_ms
        stats['meets_memory_target'] = stats['avg_memory_usage_mb'] < self.config.max_memory_usage_mb
        
        return stats
    
    def clear_cache(self):
        """Clear result cache."""
        if self._result_cache:
            self._result_cache.clear()
            logger.info("Pipeline result cache cleared")
    
    def optimize_for_performance(self):
        """Optimize pipeline for performance."""
        # Clear caches
        self.clear_cache()
        self.complexity_engine.clear_cache()
        
        # Optimize components
        self.complexity_engine.optimize_for_performance()
        self.routing_engine.optimize_for_performance()
        
        # Reset statistics
        self.pipeline_stats = {
            'total_queries': 0,
            'quantum_queries': 0,
            'classical_queries': 0,
            'hybrid_queries': 0,
            'avg_processing_time_ms': 0.0,
            'avg_memory_usage_mb': 0.0,
            'routing_accuracy': 0.0
        }
        
        logger.info("HybridQuantumClassicalPipeline optimized for performance")
    
    def benchmark_pipeline(self, test_queries: List[Dict[str, Any]], 
                         test_candidates: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Benchmark pipeline performance."""
        if len(test_queries) != len(test_candidates):
            raise ValueError("Number of queries must match number of candidate lists")
        
        start_time = time.time()
        results = []
        
        for query, candidates in zip(test_queries, test_candidates):
            result = self.rerank(query, candidates)
            results.append(result)
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_results = [r for r in results if r.success]
        processing_times = [r.processing_time_ms for r in successful_results]
        memory_usages = [r.memory_usage_mb for r in successful_results]
        
        routing_distribution = {
            'classical': sum(1 for r in successful_results if r.routing_decision.method == RoutingMethod.CLASSICAL),
            'quantum': sum(1 for r in successful_results if r.routing_decision.method == RoutingMethod.QUANTUM),
            'hybrid': sum(1 for r in successful_results if r.routing_decision.method == RoutingMethod.HYBRID)
        }
        
        benchmark_results = {
            'total_queries': len(test_queries),
            'successful_queries': len(successful_results),
            'success_rate': len(successful_results) / len(test_queries),
            'total_time_seconds': total_time,
            'avg_processing_time_ms': np.mean(processing_times) if processing_times else 0,
            'max_processing_time_ms': np.max(processing_times) if processing_times else 0,
            'avg_memory_usage_mb': np.mean(memory_usages) if memory_usages else 0,
            'max_memory_usage_mb': np.max(memory_usages) if memory_usages else 0,
            'routing_distribution': routing_distribution,
            'meets_latency_target': all(t < self.config.max_total_latency_ms for t in processing_times),
            'meets_memory_target': all(m < self.config.max_memory_usage_mb for m in memory_usages)
        }
        
        return benchmark_results
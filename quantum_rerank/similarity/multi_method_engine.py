"""
Multi-Method Similarity Engine implementation.

This module provides a unified similarity engine supporting classical, quantum,
and hybrid similarity methods with intelligent method selection based on performance
requirements and accuracy targets.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..quantum.fidelity_engine import QuantumFidelityEngine
from ..core import QuantumSimilarityEngine
from ..config import Configurable, QuantumRerankConfigSchema, get_current_config
from ..utils import get_logger, QuantumRerankException, with_recovery
from .method_selector import MethodSelector, MethodSelectionContext
from .optimizer import PerformanceOptimizer
from .aggregator import ResultAggregator
from .performance_monitor import SimilarityPerformanceMonitor


class SimilarityMethod(Enum):
    """Available similarity computation methods."""
    CLASSICAL_FAST = "classical_fast"
    CLASSICAL_ACCURATE = "classical_accurate"
    QUANTUM_PRECISE = "quantum_precise"
    QUANTUM_APPROXIMATE = "quantum_approximate"
    HYBRID_BALANCED = "hybrid_balanced"
    HYBRID_BATCH = "hybrid_batch"
    ENSEMBLE = "ensemble"


@dataclass
class SimilarityRequirements:
    """Requirements for similarity computation."""
    min_accuracy: float = 0.9
    max_latency_ms: float = 100.0
    batch_size: int = 1
    allow_approximation: bool = True
    force_method: Optional[str] = None
    require_consistency: bool = False
    confidence_threshold: float = 0.8


@dataclass
class SimilarityResult:
    """Result of similarity computation."""
    query_id: str
    candidate_scores: List[float]
    candidate_ids: List[str]
    method_used: str
    computation_time_ms: float
    accuracy_estimate: float
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def get_top_k(self, k: int) -> List[Tuple[str, float]]:
        """Get top-k candidates by score."""
        sorted_pairs = sorted(
            zip(self.candidate_ids, self.candidate_scores),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_pairs[:k]


@dataclass
class ConsensusResult:
    """Result of multi-method consensus computation."""
    query_id: str
    consensus_scores: List[float]
    candidate_ids: List[str]
    method_scores: Dict[str, List[float]]
    consensus_confidence: float
    agreement_score: float
    computation_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ClassicalSimilarity:
    """Classical similarity computation methods."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def compute_cosine_similarity(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity."""
        # Normalize vectors
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        if len(candidate_embeddings.shape) == 1:
            candidate_embeddings = candidate_embeddings.reshape(1, -1)
        
        candidate_norms = candidate_embeddings / (
            np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8
        )
        
        # Compute dot product
        similarities = np.dot(candidate_norms, query_norm)
        
        return similarities
    
    def compute_dot_product_similarity(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute dot product similarity."""
        if len(candidate_embeddings.shape) == 1:
            candidate_embeddings = candidate_embeddings.reshape(1, -1)
        
        similarities = np.dot(candidate_embeddings, query_embedding)
        
        # Normalize to [0, 1] range
        min_sim = np.min(similarities)
        max_sim = np.max(similarities)
        
        if max_sim > min_sim:
            similarities = (similarities - min_sim) / (max_sim - min_sim)
        
        return similarities
    
    def compute_euclidean_similarity(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute Euclidean distance-based similarity."""
        if len(candidate_embeddings.shape) == 1:
            candidate_embeddings = candidate_embeddings.reshape(1, -1)
        
        # Compute Euclidean distances
        distances = np.linalg.norm(
            candidate_embeddings - query_embedding.reshape(1, -1),
            axis=1
        )
        
        # Convert to similarity (inverse distance)
        similarities = 1.0 / (1.0 + distances)
        
        return similarities


class HybridSimilarity:
    """Hybrid quantum-classical similarity computation."""
    
    def __init__(
        self,
        quantum_weight: float = 0.5,
        classical_weight: float = 0.5
    ):
        self.quantum_weight = quantum_weight
        self.classical_weight = classical_weight
        self.logger = get_logger(__name__)
        
        # Initialize component methods
        self.classical_similarity = ClassicalSimilarity()
        self.quantum_fidelity_engine = QuantumFidelityEngine()
    
    def compute_hybrid_similarity(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        use_quantum: bool = True
    ) -> np.ndarray:
        """Compute weighted hybrid similarity."""
        # Classical component
        classical_scores = self.classical_similarity.compute_cosine_similarity(
            query_embedding, candidate_embeddings
        )
        
        if not use_quantum:
            return classical_scores
        
        # Quantum component (selective for performance)
        quantum_scores = self._compute_quantum_scores_batch(
            query_embedding, candidate_embeddings
        )
        
        # Weighted combination
        hybrid_scores = (
            self.classical_weight * classical_scores +
            self.quantum_weight * quantum_scores
        )
        
        return hybrid_scores
    
    def _compute_quantum_scores_batch(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute quantum similarity scores for batch."""
        if len(candidate_embeddings.shape) == 1:
            candidate_embeddings = candidate_embeddings.reshape(1, -1)
        
        quantum_scores = []
        
        # Compute quantum fidelity for each candidate
        for candidate in candidate_embeddings:
            result = self.quantum_fidelity_engine.compute_fidelity(
                query_embedding, candidate,
                method="approximate_classical"  # Fast approximation
            )
            quantum_scores.append(result.fidelity)
        
        return np.array(quantum_scores)


class MultiMethodSimilarityEngine(Configurable):
    """
    Unified similarity engine supporting multiple computation methods.
    
    This engine provides intelligent method selection, performance optimization,
    and result aggregation across classical, quantum, and hybrid similarity methods.
    """
    
    def __init__(self, config: Optional[QuantumRerankConfigSchema] = None):
        self.config = config or get_current_config()
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.method_selector = MethodSelector()
        self.performance_optimizer = PerformanceOptimizer()
        self.result_aggregator = ResultAggregator()
        self.performance_monitor = SimilarityPerformanceMonitor()
        
        # Initialize similarity methods
        self.methods = self._initialize_methods()
        
        # Thread pool for parallel computation
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Method performance cache
        self.method_performance_cache = {}
        self._cache_lock = threading.RLock()
        
        self.logger.info("Initialized MultiMethodSimilarityEngine with {} methods".format(
            len(self.methods)
        ))
    
    def _initialize_methods(self) -> Dict[str, Any]:
        """Initialize all available similarity methods."""
        methods = {}
        
        # Classical methods
        classical_sim = ClassicalSimilarity()
        methods[SimilarityMethod.CLASSICAL_FAST.value] = {
            "compute": lambda q, c: classical_sim.compute_cosine_similarity(q, c),
            "performance": {"latency_ms": 15, "accuracy": 0.85},
            "batch_optimized": True
        }
        methods[SimilarityMethod.CLASSICAL_ACCURATE.value] = {
            "compute": lambda q, c: classical_sim.compute_euclidean_similarity(q, c),
            "performance": {"latency_ms": 25, "accuracy": 0.88},
            "batch_optimized": True
        }
        
        # Quantum methods
        quantum_engine = QuantumFidelityEngine()
        methods[SimilarityMethod.QUANTUM_PRECISE.value] = {
            "compute": lambda q, c: self._compute_quantum_precise(q, c, quantum_engine),
            "performance": {"latency_ms": 95, "accuracy": 0.99},
            "batch_optimized": False
        }
        methods[SimilarityMethod.QUANTUM_APPROXIMATE.value] = {
            "compute": lambda q, c: self._compute_quantum_approximate(q, c, quantum_engine),
            "performance": {"latency_ms": 45, "accuracy": 0.93},
            "batch_optimized": True
        }
        
        # Hybrid methods
        hybrid_sim = HybridSimilarity()
        methods[SimilarityMethod.HYBRID_BALANCED.value] = {
            "compute": lambda q, c: hybrid_sim.compute_hybrid_similarity(q, c, use_quantum=True),
            "performance": {"latency_ms": 60, "accuracy": 0.93},
            "batch_optimized": True
        }
        methods[SimilarityMethod.HYBRID_BATCH.value] = {
            "compute": lambda q, c: hybrid_sim.compute_hybrid_similarity(q, c, use_quantum=False),
            "performance": {"latency_ms": 20, "accuracy": 0.90},
            "batch_optimized": True
        }
        
        return methods
    
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: Union[np.ndarray, List[np.ndarray]],
        requirements: Optional[SimilarityRequirements] = None,
        query_id: Optional[str] = None
    ) -> SimilarityResult:
        """
        Compute similarity using optimal method selection.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Candidate embedding vectors
            requirements: Similarity computation requirements
            query_id: Optional query identifier
            
        Returns:
            SimilarityResult with scores and metadata
        """
        start_time = time.time()
        requirements = requirements or SimilarityRequirements()
        query_id = query_id or f"query_{int(time.time() * 1000)}"
        
        # Convert to numpy array if needed
        if isinstance(candidate_embeddings, list):
            candidate_embeddings = np.array(candidate_embeddings)
        
        # Ensure 2D array
        if len(candidate_embeddings.shape) == 1:
            candidate_embeddings = candidate_embeddings.reshape(1, -1)
        
        # Create selection context
        context = MethodSelectionContext(
            query_size=len(query_embedding),
            batch_size=len(candidate_embeddings),
            embedding_dim=query_embedding.shape[0],
            accuracy_requirement=requirements.min_accuracy,
            latency_requirement_ms=requirements.max_latency_ms
        )
        
        # Select optimal method
        if requirements.force_method:
            selected_method = requirements.force_method
        else:
            selected_method = self.method_selector.select_method(
                context, self.method_performance_cache
            )
        
        # Log method selection
        self.logger.debug(f"Selected method: {selected_method} for {len(candidate_embeddings)} candidates")
        
        # Compute similarity with monitoring
        with self.performance_monitor.measure(selected_method):
            try:
                # Execute computation
                if selected_method == SimilarityMethod.ENSEMBLE.value:
                    result = self._compute_ensemble_similarity(
                        query_embedding, candidate_embeddings, requirements, query_id
                    )
                else:
                    scores = self._execute_method(
                        selected_method, query_embedding, candidate_embeddings
                    )
                    
                    # Create result
                    computation_time = (time.time() - start_time) * 1000
                    method_info = self.methods.get(selected_method, {})
                    
                    result = SimilarityResult(
                        query_id=query_id,
                        candidate_scores=scores.tolist(),
                        candidate_ids=[f"candidate_{i}" for i in range(len(scores))],
                        method_used=selected_method,
                        computation_time_ms=computation_time,
                        accuracy_estimate=method_info.get("performance", {}).get("accuracy", 0.9),
                        confidence_score=self._compute_confidence_score(scores),
                        metadata={
                            "batch_size": len(candidate_embeddings),
                            "embedding_dim": query_embedding.shape[0],
                            "method_performance": method_info.get("performance", {})
                        }
                    )
                
                # Update performance cache
                self._update_performance_cache(selected_method, result)
                
                # Optimize if needed
                if result.computation_time_ms > requirements.max_latency_ms:
                    self.performance_optimizer.optimize_method(
                        selected_method, result, requirements
                    )
                
                return result
                
            except Exception as e:
                self.logger.error(f"Similarity computation failed with {selected_method}: {e}")
                # Fallback to classical method
                return self._compute_fallback_similarity(
                    query_embedding, candidate_embeddings, query_id, str(e)
                )
    
    def compute_multi_method_consensus(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: Union[np.ndarray, List[np.ndarray]],
        query_id: Optional[str] = None,
        methods_to_use: Optional[List[str]] = None
    ) -> ConsensusResult:
        """
        Compute consensus similarity from multiple methods.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Candidate embedding vectors
            query_id: Optional query identifier
            methods_to_use: Specific methods to include in consensus
            
        Returns:
            ConsensusResult with aggregated scores
        """
        start_time = time.time()
        query_id = query_id or f"query_{int(time.time() * 1000)}"
        
        # Convert to numpy array if needed
        if isinstance(candidate_embeddings, list):
            candidate_embeddings = np.array(candidate_embeddings)
        
        if len(candidate_embeddings.shape) == 1:
            candidate_embeddings = candidate_embeddings.reshape(1, -1)
        
        # Determine methods to use
        if methods_to_use is None:
            # Select diverse methods for consensus
            methods_to_use = [
                SimilarityMethod.CLASSICAL_FAST.value,
                SimilarityMethod.QUANTUM_APPROXIMATE.value,
                SimilarityMethod.HYBRID_BALANCED.value
            ]
        
        # Compute with each method in parallel
        method_results = {}
        futures = {}
        
        for method in methods_to_use:
            if method in self.methods:
                future = self.executor.submit(
                    self._execute_method,
                    method,
                    query_embedding,
                    candidate_embeddings
                )
                futures[future] = method
        
        # Collect results
        for future in as_completed(futures):
            method = futures[future]
            try:
                scores = future.result()
                method_results[method] = scores
            except Exception as e:
                self.logger.warning(f"Method {method} failed in consensus: {e}")
        
        # Aggregate results
        if not method_results:
            # No methods succeeded, use fallback
            return self._create_fallback_consensus(
                query_embedding, candidate_embeddings, query_id
            )
        
        consensus_scores = self.result_aggregator.compute_consensus(
            method_results, strategy="weighted_average"
        )
        
        # Compute agreement metrics
        agreement_score = self._compute_method_agreement(method_results)
        confidence = self._compute_consensus_confidence(method_results, agreement_score)
        
        computation_time = (time.time() - start_time) * 1000
        
        return ConsensusResult(
            query_id=query_id,
            consensus_scores=consensus_scores.tolist(),
            candidate_ids=[f"candidate_{i}" for i in range(len(consensus_scores))],
            method_scores={k: v.tolist() for k, v in method_results.items()},
            consensus_confidence=confidence,
            agreement_score=agreement_score,
            computation_time_ms=computation_time,
            metadata={
                "methods_used": list(method_results.keys()),
                "batch_size": len(candidate_embeddings)
            }
        )
    
    def _execute_method(
        self,
        method_name: str,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray
    ) -> np.ndarray:
        """Execute specific similarity method."""
        if method_name not in self.methods:
            raise ValueError(f"Unknown method: {method_name}")
        
        method = self.methods[method_name]
        compute_func = method["compute"]
        
        # Execute computation
        scores = compute_func(query_embedding, candidate_embeddings)
        
        # Ensure numpy array
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)
        
        # Ensure scores are in [0, 1] range
        scores = np.clip(scores, 0.0, 1.0)
        
        return scores
    
    def _compute_quantum_precise(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        quantum_engine: QuantumFidelityEngine
    ) -> np.ndarray:
        """Compute precise quantum similarity."""
        if len(candidate_embeddings.shape) == 1:
            candidate_embeddings = candidate_embeddings.reshape(1, -1)
        
        # Compute quantum fidelity for each candidate
        results = quantum_engine.compute_batch_fidelity(
            query_embedding,
            [candidate for candidate in candidate_embeddings],
            method="swap_test"
        )
        
        scores = np.array([r.fidelity for r in results])
        return scores
    
    def _compute_quantum_approximate(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        quantum_engine: QuantumFidelityEngine
    ) -> np.ndarray:
        """Compute approximate quantum similarity."""
        if len(candidate_embeddings.shape) == 1:
            candidate_embeddings = candidate_embeddings.reshape(1, -1)
        
        # Use faster approximate method
        results = quantum_engine.compute_batch_fidelity(
            query_embedding,
            [candidate for candidate in candidate_embeddings],
            method="approximate_classical"
        )
        
        scores = np.array([r.fidelity for r in results])
        return scores
    
    def _compute_ensemble_similarity(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        requirements: SimilarityRequirements,
        query_id: str
    ) -> SimilarityResult:
        """Compute ensemble similarity from multiple methods."""
        # Use consensus computation
        consensus = self.compute_multi_method_consensus(
            query_embedding, candidate_embeddings, query_id
        )
        
        # Convert to SimilarityResult
        return SimilarityResult(
            query_id=query_id,
            candidate_scores=consensus.consensus_scores,
            candidate_ids=consensus.candidate_ids,
            method_used=SimilarityMethod.ENSEMBLE.value,
            computation_time_ms=consensus.computation_time_ms,
            accuracy_estimate=0.95,  # High accuracy for ensemble
            confidence_score=consensus.consensus_confidence,
            metadata=consensus.metadata
        )
    
    def _compute_fallback_similarity(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        query_id: str,
        error_msg: str
    ) -> SimilarityResult:
        """Compute fallback similarity when primary method fails."""
        self.logger.warning(f"Using fallback classical similarity due to: {error_msg}")
        
        classical_sim = ClassicalSimilarity()
        scores = classical_sim.compute_cosine_similarity(
            query_embedding, candidate_embeddings
        )
        
        return SimilarityResult(
            query_id=query_id,
            candidate_scores=scores.tolist(),
            candidate_ids=[f"candidate_{i}" for i in range(len(scores))],
            method_used="fallback_classical",
            computation_time_ms=0.0,  # Not measured for fallback
            accuracy_estimate=0.85,
            confidence_score=0.7,  # Lower confidence for fallback
            warnings=[f"Fallback method used due to: {error_msg}"]
        )
    
    def _create_fallback_consensus(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        query_id: str
    ) -> ConsensusResult:
        """Create fallback consensus result."""
        classical_sim = ClassicalSimilarity()
        scores = classical_sim.compute_cosine_similarity(
            query_embedding, candidate_embeddings
        )
        
        return ConsensusResult(
            query_id=query_id,
            consensus_scores=scores.tolist(),
            candidate_ids=[f"candidate_{i}" for i in range(len(scores))],
            method_scores={"fallback_classical": scores.tolist()},
            consensus_confidence=0.5,
            agreement_score=1.0,  # Single method
            computation_time_ms=0.0
        )
    
    def _compute_confidence_score(self, scores: np.ndarray) -> float:
        """Compute confidence score for similarity results."""
        if len(scores) == 0:
            return 0.0
        
        # Confidence based on score distribution
        score_std = np.std(scores)
        score_range = np.max(scores) - np.min(scores)
        
        # Higher confidence when scores are well-separated
        if score_range > 0:
            confidence = min(1.0, score_std / score_range * 2)
        else:
            confidence = 0.5
        
        return float(confidence)
    
    def _compute_method_agreement(self, method_results: Dict[str, np.ndarray]) -> float:
        """Compute agreement score between different methods."""
        if len(method_results) <= 1:
            return 1.0
        
        # Compute pairwise correlations
        correlations = []
        methods = list(method_results.keys())
        
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                scores1 = method_results[methods[i]]
                scores2 = method_results[methods[j]]
                
                if len(scores1) > 1 and len(scores2) > 1:
                    corr = np.corrcoef(scores1, scores2)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        if correlations:
            return float(np.mean(correlations))
        else:
            return 0.5
    
    def _compute_consensus_confidence(
        self,
        method_results: Dict[str, np.ndarray],
        agreement_score: float
    ) -> float:
        """Compute confidence in consensus results."""
        # Base confidence on agreement and number of methods
        num_methods = len(method_results)
        
        if num_methods == 0:
            return 0.0
        
        # Higher confidence with more methods and better agreement
        method_factor = min(1.0, num_methods / 3.0)
        confidence = agreement_score * method_factor
        
        return float(confidence)
    
    def _update_performance_cache(self, method: str, result: SimilarityResult) -> None:
        """Update method performance cache."""
        with self._cache_lock:
            if method not in self.method_performance_cache:
                self.method_performance_cache[method] = {
                    "latency_ms": [],
                    "accuracy_estimates": [],
                    "batch_sizes": []
                }
            
            cache = self.method_performance_cache[method]
            cache["latency_ms"].append(result.computation_time_ms)
            cache["accuracy_estimates"].append(result.accuracy_estimate)
            cache["batch_sizes"].append(len(result.candidate_scores))
            
            # Keep last 100 measurements
            for key in cache:
                if len(cache[key]) > 100:
                    cache[key] = cache[key][-100:]
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            "method_statistics": {},
            "overall_statistics": self.performance_monitor.get_overall_statistics()
        }
        
        # Get per-method statistics
        for method_name in self.methods:
            method_stats = self.performance_monitor.get_method_statistics(method_name)
            if method_stats:
                stats["method_statistics"][method_name] = method_stats
        
        # Add cache statistics
        with self._cache_lock:
            for method, cache in self.method_performance_cache.items():
                if cache["latency_ms"]:
                    stats["method_statistics"][method] = {
                        "avg_latency_ms": np.mean(cache["latency_ms"]),
                        "max_latency_ms": np.max(cache["latency_ms"]),
                        "avg_accuracy": np.mean(cache["accuracy_estimates"]),
                        "avg_batch_size": np.mean(cache["batch_sizes"])
                    }
        
        return stats
    
    def configure(self, config: QuantumRerankConfigSchema) -> None:
        """Configure engine with new configuration."""
        self.config = config
        
        # Reconfigure components
        if hasattr(self.method_selector, 'configure'):
            self.method_selector.configure(config)
        
        # Update method parameters based on config
        if hasattr(config, 'performance'):
            perf_config = config.performance
            # Update method selection criteria based on performance targets
            self.method_selector.update_performance_targets(
                latency_ms=perf_config.similarity_timeout_ms,
                memory_gb=perf_config.max_memory_gb
            )
    
    def validate_config(self, config: QuantumRerankConfigSchema) -> bool:
        """Validate configuration for this component."""
        # Check required configuration sections
        if not hasattr(config, 'performance'):
            return False
        
        # Validate performance targets
        perf = config.performance
        if perf.similarity_timeout_ms <= 0 or perf.max_memory_gb <= 0:
            return False
        
        return True
    
    def get_config_requirements(self) -> List[str]:
        """Get list of required configuration sections."""
        return ["performance", "quantum", "ml"]
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


__all__ = [
    "SimilarityMethod",
    "SimilarityRequirements",
    "SimilarityResult",
    "ConsensusResult",
    "MultiMethodSimilarityEngine"
]
"""
High-level RAG reranker interface for quantum similarity engine.

This module provides a simplified interface for integrating quantum-inspired
similarity reranking into RAG (Retrieval-Augmented Generation) systems.

Implements PRD Section 5.2: Integration with Existing RAG Pipeline.
"""

from .quantum_similarity_engine import QuantumSimilarityEngine, SimilarityEngineConfig, SimilarityMethod
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class QuantumRAGReranker:
    """
    High-level interface for RAG system integration.
    
    Implements PRD Section 5.2 integration requirements, providing
    a simple API for reranking candidate documents using quantum-inspired
    similarity metrics.
    
    Attributes:
        config: Configuration for the underlying similarity engine
        similarity_engine: Core quantum similarity computation engine
    """
    
    def __init__(self, config: SimilarityEngineConfig = None):
        """
        Initialize the quantum RAG reranker.
        
        Args:
            config: Optional configuration for similarity engine.
                   Defaults to standard configuration if not provided.
        """
        self.config = config or SimilarityEngineConfig()
        self.similarity_engine = QuantumSimilarityEngine(self.config)
        
        logger.info("QuantumRAGReranker initialized with config: "
                   f"method={self.config.similarity_method.value}, "
                   f"n_qubits={self.config.n_qubits}, "
                   f"caching={self.config.enable_caching}")
    
    def rerank(self, 
               query: str,
               candidates: List[str],
               top_k: int = 10,
               method: str = "hybrid") -> List[Dict]:
        """
        Main reranking interface for RAG systems.
        
        Takes a query and list of candidate documents, computes similarity
        scores using quantum-inspired methods, and returns reranked results.
        
        Args:
            query: Search query or user question
            candidates: List of candidate documents/passages to rerank
            top_k: Number of top results to return (default: 10)
            method: Similarity method - "classical", "quantum", or "hybrid" (default: "hybrid")
            
        Returns:
            List of dictionaries containing reranked results with:
                - text: The candidate text
                - similarity_score: Computed similarity score (0-1)
                - rank: Final rank position (1-based)
                - method: Similarity method used
                - metadata: Additional computation metadata
                
        Example:
            >>> reranker = QuantumRAGReranker()
            >>> results = reranker.rerank(
            ...     "What is quantum computing?",
            ...     ["Quantum computing uses qubits...", "Classical computing..."],
            ...     top_k=5
            ... )
        """
        # Validate inputs
        if not query:
            raise ValueError("Query cannot be empty")
        if not candidates:
            return []
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        
        # Convert method string to enum
        method_map = {
            "classical": SimilarityMethod.CLASSICAL_COSINE,
            "quantum": SimilarityMethod.QUANTUM_FIDELITY,
            "hybrid": SimilarityMethod.HYBRID_WEIGHTED
        }
        
        similarity_method = method_map.get(method.lower())
        if similarity_method is None:
            logger.warning(f"Unknown method '{method}', defaulting to hybrid")
            similarity_method = SimilarityMethod.HYBRID_WEIGHTED
        
        # Log reranking request
        logger.info(f"Reranking {len(candidates)} candidates with method={method}, top_k={top_k}")
        
        # Perform reranking
        ranked_results = self.similarity_engine.rerank_candidates(
            query, candidates, top_k, similarity_method
        )
        
        # Format results for RAG system consumption
        formatted_results = []
        for i, (candidate, similarity, metadata) in enumerate(ranked_results):
            result = {
                'text': candidate,
                'similarity_score': float(similarity),
                'rank': i + 1,
                'method': method,
                'metadata': metadata
            }
            formatted_results.append(result)
        
        logger.info(f"Reranking complete: returned {len(formatted_results)} results")
        
        return formatted_results
    
    def compute_similarity(self, text1: str, text2: str, method: str = "hybrid") -> Dict:
        """
        Compute similarity between two texts.
        
        Simplified interface for direct similarity computation between
        a pair of texts using quantum-inspired methods.
        
        Args:
            text1: First text
            text2: Second text  
            method: Similarity method - "classical", "quantum", or "hybrid"
            
        Returns:
            Dictionary containing:
                - similarity_score: Computed similarity (0-1)
                - method: Method used
                - metadata: Additional computation details
        """
        # Validate inputs
        if not text1 or not text2:
            raise ValueError("Both texts must be non-empty")
        
        method_map = {
            "classical": SimilarityMethod.CLASSICAL_COSINE,
            "quantum": SimilarityMethod.QUANTUM_FIDELITY,
            "hybrid": SimilarityMethod.HYBRID_WEIGHTED
        }
        
        similarity_method = method_map.get(method.lower())
        if similarity_method is None:
            logger.warning(f"Unknown method '{method}', defaulting to hybrid")
            similarity_method = SimilarityMethod.HYBRID_WEIGHTED
        
        # Compute similarity
        similarity, metadata = self.similarity_engine.compute_similarity(
            text1, text2, similarity_method
        )
        
        return {
            'similarity_score': float(similarity),
            'method': method,
            'metadata': metadata
        }
    
    def batch_compute_similarities(self,
                                 query: str,
                                 candidates: List[str],
                                 method: str = "hybrid") -> List[Tuple[str, float, Dict]]:
        """
        Compute similarities between query and multiple candidates.
        
        Efficient batch processing for similarity computation without reranking.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            method: Similarity method
            
        Returns:
            List of tuples (candidate_text, similarity_score, metadata)
        """
        method_map = {
            "classical": SimilarityMethod.CLASSICAL_COSINE,
            "quantum": SimilarityMethod.QUANTUM_FIDELITY,
            "hybrid": SimilarityMethod.HYBRID_WEIGHTED
        }
        
        similarity_method = method_map.get(method.lower(), SimilarityMethod.HYBRID_WEIGHTED)
        
        return self.similarity_engine.compute_similarities_batch(
            query, candidates, similarity_method
        )
    
    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics for monitoring.
        
        Returns comprehensive performance metrics including:
        - Average computation time
        - Cache hit rate
        - PRD compliance status
        - Total comparisons performed
        """
        return self.similarity_engine.get_performance_report()
    
    def benchmark_performance(self, test_texts: Optional[List[str]] = None) -> Dict:
        """
        Run performance benchmarks on all similarity methods.
        
        Args:
            test_texts: Optional custom test texts. Uses defaults if not provided.
            
        Returns:
            Dictionary with benchmark results for each method including:
            - Average/max computation times
            - PRD target compliance
            - Similarity score distributions
        """
        return self.similarity_engine.benchmark_similarity_methods(test_texts)
    
    def clear_cache(self):
        """Clear the similarity cache."""
        self.similarity_engine.clear_cache()
        logger.info("RAG reranker cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return self.similarity_engine.get_cache_statistics()
    
    def update_config(self, **kwargs):
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
                     (n_qubits, n_layers, similarity_method, etc.)
        """
        # Update config attributes
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key}={value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")
        
        # Reinitialize engine with new config
        self.similarity_engine = QuantumSimilarityEngine(self.config)
        logger.info("Similarity engine reinitialized with updated config")
    
    def __repr__(self):
        """String representation of the reranker."""
        return (f"QuantumRAGReranker(method={self.config.similarity_method.value}, "
                f"n_qubits={self.config.n_qubits}, "
                f"caching={self.config.enable_caching})")
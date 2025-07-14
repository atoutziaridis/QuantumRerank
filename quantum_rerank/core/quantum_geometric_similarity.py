"""
Quantum Geometric Similarity Module for QuantumRerank.

Implements subspace-based similarity with sequential projections for asymmetric,
context-aware scoring. Based on "A Quantum Geometric Model of Similarity" research.

Key Features:
- Asymmetric similarity: Sim(A,B) ≠ Sim(B,A)
- Context-dependent scoring: Include session/user context
- Subspace representations: Concepts as higher-dimensional spans
- Sequential projections: Order-sensitive similarity computation

Based on:
- "A Quantum Geometric Model of Similarity" paper
- Quantum subspace projection theory
- QuantumRerank's existing embedding infrastructure
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
from enum import Enum
import time

from .embeddings import EmbeddingProcessor

logger = logging.getLogger(__name__)


@dataclass
class GeometricSimilarityConfig:
    """Configuration for quantum geometric similarity."""
    subspace_dim: int = 32  # Dimension of concept subspaces
    max_subspace_vectors: int = 5  # Max vectors per subspace
    context_weight: float = 0.3  # Weight for context projections
    enable_asymmetry: bool = True  # Enable asymmetric similarity
    enable_caching: bool = True
    max_cache_size: int = 1000
    # Projection sequence settings
    use_context_projections: bool = True
    context_projection_steps: int = 1


class SubspaceProjector:
    """
    Implements quantum-inspired subspace projection operations.
    
    Handles creation of subspaces from embeddings and sequential projection
    operations for similarity computation.
    """
    
    def __init__(self, config: GeometricSimilarityConfig):
        self.config = config
        
    def create_subspace(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Create a subspace from a set of embedding vectors.
        
        Args:
            embeddings: Set of embedding vectors (N x D)
            
        Returns:
            Orthonormal basis for the subspace (K x D) where K <= subspace_dim
        """
        if len(embeddings) == 0:
            raise ValueError("Cannot create subspace from empty embeddings")
        
        # Limit number of vectors
        if len(embeddings) > self.config.max_subspace_vectors:
            # Use k-means or random sampling to select representative vectors
            indices = np.random.choice(
                len(embeddings), 
                self.config.max_subspace_vectors, 
                replace=False
            )
            embeddings = embeddings[indices]
        
        # Perform SVD to get orthonormal basis
        U, s, Vt = np.linalg.svd(embeddings, full_matrices=False)
        
        # Keep only significant singular vectors
        significant_dims = min(
            len(s),
            self.config.subspace_dim,
            np.sum(s > 1e-6)  # Numerical tolerance
        )
        
        # Return orthonormal basis vectors (transposed for easier computation)
        return Vt[:significant_dims, :]
    
    def create_projector_matrix(self, subspace_basis: np.ndarray) -> np.ndarray:
        """
        Create projection matrix for a subspace.
        
        Args:
            subspace_basis: Orthonormal basis vectors (K x D)
            
        Returns:
            Projection matrix P = V^T * V (D x D)
        """
        # P = V^T * V where V is the basis matrix
        return subspace_basis.T @ subspace_basis
    
    def project_vector(self, vector: np.ndarray, projector: np.ndarray) -> np.ndarray:
        """
        Project a vector onto a subspace.
        
        Args:
            vector: Vector to project (D,)
            projector: Projection matrix (D x D)
            
        Returns:
            Projected vector (D,)
        """
        return projector @ vector
    
    def sequential_projection(self, 
                            initial_vector: np.ndarray,
                            projection_sequence: List[np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Apply sequential projections to a vector.
        
        Args:
            initial_vector: Starting vector (D,)
            projection_sequence: List of projection matrices [(D x D), ...]
            
        Returns:
            Tuple of (final_vector, squared_norm)
        """
        current_vector = initial_vector.copy()
        
        # Apply projections in sequence
        for projector in projection_sequence:
            current_vector = self.project_vector(current_vector, projector)
        
        # Return final vector and its squared norm (similarity score)
        squared_norm = np.linalg.norm(current_vector) ** 2
        
        return current_vector, squared_norm


class QuantumGeometricSimilarity:
    """
    Quantum geometric similarity engine with subspace projections.
    
    Implements asymmetric, context-aware similarity based on sequential
    projections in quantum-inspired subspaces.
    """
    
    def __init__(self, config: GeometricSimilarityConfig = None):
        self.config = config or GeometricSimilarityConfig()
        
        # Initialize components
        self.embedding_processor = EmbeddingProcessor()
        self.projector = SubspaceProjector(self.config)
        
        # Caching system
        self._subspace_cache = {} if self.config.enable_caching else None
        self._similarity_cache = {} if self.config.enable_caching else None
        
        # Performance tracking
        self.stats = {
            'total_similarities': 0,
            'avg_computation_time_ms': 0.0,
            'subspace_cache_hits': 0,
            'similarity_cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info(f"Quantum geometric similarity initialized with {self.config.subspace_dim}D subspaces")
    
    def create_concept_subspace(self, 
                              concept_texts: List[str],
                              concept_id: Optional[str] = None) -> np.ndarray:
        """
        Create a subspace representation for a concept from text examples.
        
        Args:
            concept_texts: List of texts representing the concept
            concept_id: Optional identifier for caching
            
        Returns:
            Projection matrix for the concept subspace
        """
        # Check cache
        if concept_id and self._subspace_cache and concept_id in self._subspace_cache:
            self.stats['subspace_cache_hits'] += 1
            return self._subspace_cache[concept_id]
        
        # Generate embeddings
        embeddings = self.embedding_processor.encode_texts(concept_texts)
        embeddings_array = np.array(embeddings)
        
        # Create subspace basis
        subspace_basis = self.projector.create_subspace(embeddings_array)
        
        # Create projection matrix
        projector_matrix = self.projector.create_projector_matrix(subspace_basis)
        
        # Cache if concept_id provided
        if concept_id and self._subspace_cache:
            self._cache_subspace(concept_id, projector_matrix)
        
        return projector_matrix
    
    def compute_asymmetric_similarity(self,
                                    text_a: str,
                                    text_b: str,
                                    context_texts: Optional[List[str]] = None) -> Tuple[float, float, Dict]:
        """
        Compute asymmetric similarity: Sim(A,B) and Sim(B,A).
        
        Args:
            text_a: First text
            text_b: Second text
            context_texts: Optional context texts for context-aware similarity
            
        Returns:
            Tuple of (sim_a_to_b, sim_b_to_a, metadata)
        """
        start_time = time.time()
        
        # Generate embeddings
        embeddings = self.embedding_processor.encode_texts([text_a, text_b])
        embed_a, embed_b = embeddings[0], embeddings[1]
        
        # Create subspaces for A and B
        # For single texts, create subspace from the embedding itself
        subspace_a = self.projector.create_projector_matrix(
            np.array([embed_a])
        )
        subspace_b = self.projector.create_projector_matrix(
            np.array([embed_b])
        )
        
        # Prepare context projections if provided
        context_projectors = []
        if context_texts and self.config.use_context_projections:
            context_embeddings = self.embedding_processor.encode_texts(context_texts)
            context_subspace = self.projector.create_subspace(np.array(context_embeddings))
            context_projector = self.projector.create_projector_matrix(context_subspace)
            context_projectors = [context_projector] * self.config.context_projection_steps
        
        # Create neutral initial state (average of A and B embeddings)
        neutral_state = (embed_a + embed_b) / 2
        neutral_state = neutral_state / np.linalg.norm(neutral_state)
        
        # Compute Sim(A,B): neutral -> [context] -> A -> B
        projection_sequence_a_to_b = context_projectors + [subspace_a, subspace_b]
        _, sim_a_to_b = self.projector.sequential_projection(
            neutral_state, projection_sequence_a_to_b
        )
        
        # Compute Sim(B,A): neutral -> [context] -> B -> A
        projection_sequence_b_to_a = context_projectors + [subspace_b, subspace_a]
        _, sim_b_to_a = self.projector.sequential_projection(
            neutral_state, projection_sequence_b_to_a
        )
        
        # Compute metadata
        computation_time = (time.time() - start_time) * 1000
        metadata = {
            'asymmetric': True,
            'context_used': len(context_texts) if context_texts else 0,
            'projection_steps': len(projection_sequence_a_to_b),
            'computation_time_ms': computation_time,
            'subspace_dims': {
                'A': subspace_a.shape[0],
                'B': subspace_b.shape[0]
            }
        }
        
        # Update stats
        self.stats['total_similarities'] += 1
        self._update_stats(computation_time)
        
        return sim_a_to_b, sim_b_to_a, metadata
    
    def compute_contextual_similarity(self,
                                    query: str,
                                    document: str,
                                    user_context: Optional[List[str]] = None,
                                    session_context: Optional[List[str]] = None) -> Tuple[float, Dict]:
        """
        Compute context-aware similarity for RAG applications.
        
        Args:
            query: Query text
            document: Document text
            user_context: User profile/history texts
            session_context: Current session context texts
            
        Returns:
            Tuple of (similarity_score, metadata)
        """
        start_time = time.time()
        
        # Combine all context
        all_context = []
        if user_context:
            all_context.extend(user_context)
        if session_context:
            all_context.extend(session_context)
        
        # If asymmetry is enabled, compute directional similarity
        if self.config.enable_asymmetry:
            sim_q_to_d, sim_d_to_q, base_metadata = self.compute_asymmetric_similarity(
                query, document, all_context
            )
            
            # Combine asymmetric similarities (query->document direction is primary)
            similarity_score = (
                0.7 * sim_q_to_d +  # Query to document (primary)
                0.3 * sim_d_to_q    # Document to query (secondary)
            )
            
            metadata = base_metadata.copy()
            metadata.update({
                'similarity_q_to_d': sim_q_to_d,
                'similarity_d_to_q': sim_d_to_q,
                'combined_similarity': similarity_score,
                'asymmetric_weights': [0.7, 0.3]
            })
        else:
            # Symmetric similarity (average of both directions)
            sim_q_to_d, sim_d_to_q, base_metadata = self.compute_asymmetric_similarity(
                query, document, all_context
            )
            
            similarity_score = (sim_q_to_d + sim_d_to_q) / 2
            
            metadata = base_metadata.copy()
            metadata.update({
                'similarity_q_to_d': sim_q_to_d,
                'similarity_d_to_q': sim_d_to_q,
                'combined_similarity': similarity_score,
                'symmetric': True
            })
        
        return similarity_score, metadata
    
    def rank_documents_with_context(self,
                                   query: str,
                                   documents: List[str],
                                   user_context: Optional[List[str]] = None,
                                   session_context: Optional[List[str]] = None,
                                   top_k: Optional[int] = None) -> List[Tuple[str, float, Dict]]:
        """
        Rank documents using context-aware geometric similarity.
        
        Args:
            query: Query text
            documents: List of document texts
            user_context: User profile/history texts
            session_context: Current session context texts
            top_k: Return only top K results
            
        Returns:
            List of (document, similarity_score, metadata) sorted by similarity
        """
        start_time = time.time()
        
        # Compute similarities for all documents
        results = []
        for i, document in enumerate(documents):
            similarity, metadata = self.compute_contextual_similarity(
                query, document, user_context, session_context
            )
            metadata['document_index'] = i
            results.append((document, similarity, metadata))
        
        # Sort by similarity (descending)
        ranked_results = sorted(results, key=lambda x: x[1], reverse=True)
        
        # Apply top_k if specified
        if top_k is not None:
            ranked_results = ranked_results[:top_k]
        
        # Add ranking metadata
        total_time = time.time() - start_time
        for i, (document, similarity, metadata) in enumerate(ranked_results):
            metadata.update({
                'final_rank': i + 1,
                'total_ranking_time_ms': total_time * 1000,
                'documents_ranked': len(documents)
            })
            ranked_results[i] = (document, similarity, metadata)
        
        logger.info(f"Ranked {len(documents)} documents in {total_time*1000:.2f}ms")
        
        return ranked_results
    
    def _cache_subspace(self, concept_id: str, projector_matrix: np.ndarray):
        """Cache subspace projector matrix."""
        if len(self._subspace_cache) >= self.config.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._subspace_cache))
            del self._subspace_cache[oldest_key]
        
        self._subspace_cache[concept_id] = projector_matrix.copy()
    
    def _update_stats(self, computation_time_ms: float):
        """Update performance statistics."""
        n = self.stats['total_similarities']
        current_avg = self.stats['avg_computation_time_ms']
        
        self.stats['avg_computation_time_ms'] = (
            (current_avg * (n - 1) + computation_time_ms) / n
        )
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        stats = self.stats.copy()
        
        if self._subspace_cache is not None:
            stats.update({
                'subspace_cache_size': len(self._subspace_cache),
                'subspace_cache_max': self.config.max_cache_size
            })
        
        return stats
    
    def clear_caches(self):
        """Clear all caches."""
        if self._subspace_cache:
            self._subspace_cache.clear()
        if self._similarity_cache:
            self._similarity_cache.clear()
        logger.info("Geometric similarity caches cleared")
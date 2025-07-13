"""
SentenceTransformer Integration and Embedding Processing for QuantumRerank.

This module implements the embedding preprocessing pipeline as specified in the PRD 
and documentation, integrating SentenceTransformers with quantum circuits.

Based on:
- PRD Section 2.2: Implementation Stack - SentenceTransformers
- PRD Section 4.1: System Requirements - Embedding Models
- PRD Section 5.2: Integration with Existing RAG Pipeline
- Documentation: "Recommendation for Pre-trained Text Embedding Mode.md"
- Documentation: "Quantum-Inspired Semantic Reranking with PyTorch_.md"
- Research: Quantum-inspired embeddings and cosine similarity techniques
"""

import numpy as np
import torch
import time
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Union, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from ..config.settings import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding processing."""
    # Based on documentation recommendation: multi-qa-mpnet-base-dot-v1
    model_name: str = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
    embedding_dim: int = 768
    max_sequence_length: int = 512
    batch_size: int = 32
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    normalize_embeddings: bool = True
    # Alternative fallback models
    fallback_models: List[str] = None
    
    def __post_init__(self):
        if self.fallback_models is None:
            self.fallback_models = [
                'sentence-transformers/all-mpnet-base-v2',
                'sentence-transformers/all-MiniLM-L6-v2'
            ]


class EmbeddingProcessor:
    """
    Handles text embedding generation and preprocessing for quantum circuits.
    
    Based on PRD Section 5.2 and documentation recommendations.
    Implements quantum-compatible preprocessing as specified in research papers.
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the embedding processor.
        
        Args:
            config: Embedding configuration (uses default if None)
        """
        self.config = config or EmbeddingConfig()
        
        # Determine device
        if self.config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = self.config.device
        
        # Load SentenceTransformer model with fallback
        self.model = None
        self._load_model()
        
        # Verify embedding dimension matches configuration
        self._verify_embedding_dimension()
        
        logger.info(f"Initialized EmbeddingProcessor: {self.config.model_name} on {self.device}")
    
    def _load_model(self):
        """Load SentenceTransformer model with fallback options."""
        models_to_try = [self.config.model_name] + self.config.fallback_models
        
        for model_name in models_to_try:
            try:
                self.model = SentenceTransformer(model_name, device=self.device)
                self.config.model_name = model_name  # Update config to reflect loaded model
                logger.info(f"Successfully loaded {model_name} on {self.device}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        if self.model is None:
            raise RuntimeError("Failed to load any embedding model")
    
    def _verify_embedding_dimension(self):
        """Verify embedding dimension matches configuration."""
        try:
            test_embedding = self.model.encode(["test"], convert_to_tensor=False)
            actual_dim = len(test_embedding[0])
            
            if actual_dim != self.config.embedding_dim:
                logger.warning(f"Model dimension {actual_dim} != config {self.config.embedding_dim}")
                self.config.embedding_dim = actual_dim
                logger.info(f"Updated embedding dimension to {actual_dim}")
        except Exception as e:
            logger.error(f"Failed to verify embedding dimension: {e}")
            raise
    
    def encode_texts(self, texts: List[str], 
                    batch_size: Optional[int] = None) -> np.ndarray:
        """
        Encode list of texts to embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Override default batch size
            
        Returns:
            Array of embeddings [n_texts, embedding_dim]
        """
        if not texts:
            return np.array([])
        
        batch_size = batch_size or self.config.batch_size
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=False,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=len(texts) > 100
            )
            
            logger.debug(f"Encoded {len(texts)} texts to shape {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise
    
    def encode_single_text(self, text: str) -> np.ndarray:
        """
        Encode single text to embedding.
        
        Args:
            text: Input text string
            
        Returns:
            Embedding vector [embedding_dim]
        """
        embedding = self.encode_texts([text])
        return embedding[0] if len(embedding) > 0 else np.array([])
    
    def preprocess_for_quantum(self, embeddings: np.ndarray, 
                              n_qubits: int = 4) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess embeddings for quantum circuit encoding.
        
        Based on PRD quantum constraints and circuit requirements.
        Implements quantum-compatible normalization and dimensionality adjustment.
        
        Args:
            embeddings: Input embeddings [n_embeddings, embedding_dim]
            n_qubits: Number of qubits for quantum encoding
            
        Returns:
            Tuple of (processed_embeddings, metadata)
        """
        max_amplitudes = 2 ** n_qubits  # Maximum amplitudes for quantum state
        
        processed_embeddings = []
        metadata = {
            'original_dim': embeddings.shape[1] if embeddings.ndim > 1 else len(embeddings),
            'target_amplitudes': max_amplitudes,
            'n_qubits': n_qubits,
            'processing_applied': []
        }
        
        for embedding in embeddings:
            processed_emb = embedding.copy()
            
            # Step 1: Dimensionality adjustment
            if len(processed_emb) > max_amplitudes:
                # Truncate to fit quantum state
                processed_emb = processed_emb[:max_amplitudes]
                metadata['processing_applied'].append('truncation')
            elif len(processed_emb) < max_amplitudes:
                # Pad with zeros
                padding = max_amplitudes - len(processed_emb)
                processed_emb = np.pad(processed_emb, (0, padding), mode='constant')
                metadata['processing_applied'].append('zero_padding')
            
            # Step 2: Ensure unit norm (required for quantum states)
            norm = np.linalg.norm(processed_emb)
            if norm > 0:
                processed_emb = processed_emb / norm
                metadata['processing_applied'].append('normalization')
            
            processed_embeddings.append(processed_emb)
        
        result = np.array(processed_embeddings)
        
        logger.debug(f"Quantum preprocessing: {metadata}")
        return result, metadata
    
    def create_embedding_batches(self, texts: List[str], 
                               batch_size: Optional[int] = None) -> List[Tuple[List[str], np.ndarray]]:
        """
        Create batches of texts and their embeddings for efficient processing.
        
        Supports PRD batch processing requirements (50-100 documents).
        
        Args:
            texts: List of input texts
            batch_size: Batch size override
            
        Returns:
            List of (batch_texts, batch_embeddings) tuples
        """
        batch_size = batch_size or self.config.batch_size
        batches = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.encode_texts(batch_texts)
            batches.append((batch_texts, batch_embeddings))
            
            logger.debug(f"Created batch {len(batches)}: {len(batch_texts)} texts")
        
        return batches
    
    def compute_classical_similarity(self, embedding1: np.ndarray, 
                                   embedding2: np.ndarray) -> float:
        """
        Compute classical cosine similarity for comparison baseline.
        
        Based on quantum-inspired cosine similarity research from papers.
        
        Args:
            embedding1, embedding2: Normalized embedding vectors
            
        Returns:
            Cosine similarity score [0, 1]
        """
        # Ensure embeddings are normalized
        emb1_norm = embedding1 / np.linalg.norm(embedding1)
        emb2_norm = embedding2 / np.linalg.norm(embedding2)
        
        # Compute cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        # Clamp to [0, 1] range
        similarity = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
        
        return float(similarity)
    
    def compute_fidelity_similarity(self, embedding1: np.ndarray, 
                                  embedding2: np.ndarray) -> float:
        """
        Compute quantum-inspired fidelity similarity between embeddings.
        
        Based on research from "Quantum-inspired Embeddings Projection and 
        Similarity Metrics for Representation Learning".
        
        Args:
            embedding1, embedding2: Embedding vectors
            
        Returns:
            Fidelity-based similarity score [0, 1]
        """
        # Normalize embeddings to unit vectors (quantum state requirement)
        emb1_norm = embedding1 / np.linalg.norm(embedding1)
        emb2_norm = embedding2 / np.linalg.norm(embedding2)
        
        # Compute quantum fidelity (squared overlap)
        fidelity = np.abs(np.dot(emb1_norm.conj(), emb2_norm)) ** 2
        
        return float(fidelity)
    
    def benchmark_embedding_performance(self) -> Dict[str, Any]:
        """
        Benchmark embedding performance against PRD targets.
        
        Returns metrics for similarity computation speed, batch processing, etc.
        
        Returns:
            Performance metrics dictionary
        """
        import time
        
        # Test data
        test_texts = [
            "Quantum computing uses quantum mechanics for computation",
            "Machine learning algorithms process data to find patterns",
            "Information retrieval systems find relevant documents",
            "Natural language processing analyzes human language"
        ]
        
        results = {}
        
        # Single text encoding
        start_time = time.time()
        single_embedding = self.encode_single_text(test_texts[0])
        single_time = time.time() - start_time
        
        results['single_encoding_ms'] = single_time * 1000
        
        # Batch encoding
        start_time = time.time()
        batch_embeddings = self.encode_texts(test_texts)
        batch_time = time.time() - start_time
        
        results['batch_encoding_ms'] = batch_time * 1000
        results['batch_per_text_ms'] = (batch_time / len(test_texts)) * 1000
        
        # Quantum preprocessing
        start_time = time.time()
        processed, metadata = self.preprocess_for_quantum(batch_embeddings, n_qubits=4)
        preprocessing_time = time.time() - start_time
        
        results['quantum_preprocessing_ms'] = preprocessing_time * 1000
        
        # Classical similarity (baseline)
        start_time = time.time()
        cosine_sim = self.compute_classical_similarity(batch_embeddings[0], batch_embeddings[1])
        cosine_time = time.time() - start_time
        
        results['classical_similarity_ms'] = cosine_time * 1000
        
        # Fidelity similarity
        start_time = time.time()
        fidelity_sim = self.compute_fidelity_similarity(batch_embeddings[0], batch_embeddings[1])
        fidelity_time = time.time() - start_time
        
        results['fidelity_similarity_ms'] = fidelity_time * 1000
        
        # Memory usage estimation
        import sys
        results['embedding_memory_mb'] = sys.getsizeof(batch_embeddings) / (1024 * 1024)
        
        # PRD compliance check
        results['prd_compliance'] = {
            'single_encoding_under_100ms': results['single_encoding_ms'] < 100,
            'similarity_under_100ms': results['classical_similarity_ms'] < 100,
            'fidelity_under_100ms': results['fidelity_similarity_ms'] < 100,
            'preprocessing_efficient': results['quantum_preprocessing_ms'] < 10
        }
        
        return results
    
    def validate_embedding_quality(self) -> Dict[str, Any]:
        """
        Validate embedding quality and quantum compatibility.
        
        Returns:
            Quality validation results
        """
        test_texts = [
            "quantum computing",
            "classical computing", 
            "machine learning",
            "artificial intelligence"
        ]
        
        embeddings = self.encode_texts(test_texts)
        
        # Check embedding properties
        results = {
            'embedding_dim': embeddings.shape[1],
            'all_finite': np.all(np.isfinite(embeddings)),
            'normalized': np.allclose(np.linalg.norm(embeddings, axis=1), 1.0),
            'embedding_range': {
                'min': float(np.min(embeddings)),
                'max': float(np.max(embeddings)),
                'mean': float(np.mean(embeddings)),
                'std': float(np.std(embeddings))
            }
        }
        
        # Test quantum preprocessing
        processed, metadata = self.preprocess_for_quantum(embeddings, n_qubits=4)
        results['quantum_compatible'] = all([
            np.allclose(np.linalg.norm(emb), 1.0) for emb in processed
        ])
        
        # Test similarity relationships
        cosine_similarities = []
        fidelity_similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                cosine_sim = self.compute_classical_similarity(embeddings[i], embeddings[j])
                fidelity_sim = self.compute_fidelity_similarity(embeddings[i], embeddings[j])
                cosine_similarities.append(cosine_sim)
                fidelity_similarities.append(fidelity_sim)
        
        results['cosine_similarity_stats'] = {
            'mean': float(np.mean(cosine_similarities)),
            'min': float(np.min(cosine_similarities)),
            'max': float(np.max(cosine_similarities)),
            'std': float(np.std(cosine_similarities))
        }
        
        results['fidelity_similarity_stats'] = {
            'mean': float(np.mean(fidelity_similarities)),
            'min': float(np.min(fidelity_similarities)),
            'max': float(np.max(fidelity_similarities)),
            'std': float(np.std(fidelity_similarities))
        }
        
        return results
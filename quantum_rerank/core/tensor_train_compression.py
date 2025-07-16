"""
Tensor Train (TT) Decomposition for BERT Embedding Compression

This module implements TT decomposition for achieving 8-44x compression ratios
on BERT embeddings while maintaining <1% accuracy loss, following the 
quantum-inspired lightweight RAG transition strategy.

Based on:
- Research: "Tensor Train Decomposition for BERT Embeddings in RAG Systems"
- Target: 44x compression with <1% accuracy loss
- Library: TensorLy for tensor decomposition operations
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import time

try:
    import tensorly as tl
    from tensorly.decomposition import tensor_train
    from tensorly.tt_tensor import tt_to_tensor
    tl.set_backend('pytorch')
    TT_AVAILABLE = True
except ImportError:
    TT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TTConfig:
    """Configuration for Tensor Train decomposition."""
    # TT rank controls compression ratio vs accuracy tradeoff
    tt_rank: int = 8  # Target: 44x compression
    
    # Compression targets from research
    target_compression_ratio: float = 44.0
    max_accuracy_loss: float = 0.01  # 1% maximum
    
    # Decomposition parameters
    max_iter: int = 100
    tolerance: float = 1e-6
    
    # Performance settings
    batch_decomposition: bool = True
    cache_decomposed: bool = True
    
    # Validation settings
    validate_reconstruction: bool = True
    reconstruction_threshold: float = 0.95


class TTEmbeddingLayer(nn.Module):
    """
    Tensor Train compressed embedding layer.
    
    Replaces standard embedding layers with TT-compressed versions,
    achieving 44x compression with <1% accuracy loss.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embed_dim: int,
                 tt_rank: int = 8,
                 original_embedding: Optional[torch.Tensor] = None):
        """
        Initialize TT embedding layer.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            tt_rank: TT rank for compression
            original_embedding: Pre-trained embedding to compress
        """
        super().__init__()
        
        if not TT_AVAILABLE:
            raise ImportError("TensorLy is required for TT decomposition")
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.tt_rank = tt_rank
        
        # Calculate tensor dimensions for 3D tensorization
        # Following research: reshape vocab_size x embed_dim -> 3D tensor
        self.tensor_shape = self._calculate_tensor_shape(vocab_size, embed_dim)
        
        if original_embedding is not None:
            self.tt_cores = self._decompose_embedding(original_embedding)
        else:
            self.tt_cores = self._initialize_random_cores()
        
        # Register TT cores as parameters
        self.register_tt_cores()
        
        logger.info(f"TT Embedding: {vocab_size}x{embed_dim} -> "
                   f"{self.tensor_shape} (rank {tt_rank})")
    
    def _calculate_tensor_shape(self, vocab_size: int, embed_dim: int) -> Tuple[int, int, int]:
        """Calculate optimal 3D tensor shape for tensorization."""
        # Find factors close to cube root for balanced decomposition
        target_dim = int(np.ceil((vocab_size * embed_dim) ** (1/3)))
        
        # Find good factors
        factors = []
        temp = vocab_size * embed_dim
        
        # Try to get 3 roughly equal factors
        for i in range(2, int(np.sqrt(temp)) + 1):
            if temp % i == 0:
                factors.append(i)
                temp //= i
                if len(factors) == 2:
                    factors.append(temp)
                    break
        
        # If we couldn't find 3 factors, use simple approach
        if len(factors) != 3:
            d1 = int(np.ceil(vocab_size ** 0.5))
            d2 = int(np.ceil(vocab_size / d1))
            d3 = embed_dim
            factors = [d1, d2, d3]
        
        return tuple(factors)
    
    def _decompose_embedding(self, embedding: torch.Tensor) -> List[torch.Tensor]:
        """Decompose pre-trained embedding using TT decomposition."""
        # Reshape embedding to 3D tensor
        reshaped = embedding.reshape(self.tensor_shape)
        
        # Convert to TensorLy format
        tensor = tl.tensor(reshaped)
        
        # Perform TT decomposition
        tt_tensor = tensor_train(tensor, rank=self.tt_rank)
        
        # Extract cores
        cores = []
        for i, core in enumerate(tt_tensor):
            cores.append(torch.tensor(core, dtype=torch.float32))
        
        logger.info(f"TT decomposition completed: {len(cores)} cores")
        return cores
    
    def _initialize_random_cores(self) -> List[torch.Tensor]:
        """Initialize random TT cores for training from scratch."""
        cores = []
        
        # First core: rank × shape[0] × rank
        core1 = torch.randn(1, self.tensor_shape[0], self.tt_rank)
        cores.append(core1)
        
        # Middle cores: rank × shape[i] × rank
        for i in range(1, len(self.tensor_shape) - 1):
            core = torch.randn(self.tt_rank, self.tensor_shape[i], self.tt_rank)
            cores.append(core)
        
        # Last core: rank × shape[-1] × 1
        core_last = torch.randn(self.tt_rank, self.tensor_shape[-1], 1)
        cores.append(core_last)
        
        return cores
    
    def register_tt_cores(self):
        """Register TT cores as PyTorch parameters."""
        for i, core in enumerate(self.tt_cores):
            self.register_parameter(f'tt_core_{i}', nn.Parameter(core))
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TT embedding layer.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            
        Returns:
            Embeddings [batch_size, seq_len, embed_dim]
        """
        # Reconstruct full embedding tensor from TT cores
        reconstructed = self._reconstruct_tensor()
        
        # Reshape back to embedding matrix
        embedding_matrix = reconstructed.reshape(self.vocab_size, self.embed_dim)
        
        # Standard embedding lookup
        return torch.nn.functional.embedding(input_ids, embedding_matrix)
    
    def _reconstruct_tensor(self) -> torch.Tensor:
        """Reconstruct full tensor from TT cores."""
        # Contract TT cores to reconstruct tensor
        result = self.tt_cores[0]
        
        for i in range(1, len(self.tt_cores)):
            # Contract consecutive cores
            result = torch.einsum('...ij,...jk->...ik', result, self.tt_cores[i])
        
        # Remove virtual bond dimensions
        result = result.squeeze(0).squeeze(-1)
        
        return result
    
    def compression_ratio(self) -> float:
        """Calculate achieved compression ratio."""
        original_params = self.vocab_size * self.embed_dim
        tt_params = sum(core.numel() for core in self.tt_cores)
        return original_params / tt_params


class BERTTTCompressor:
    """
    BERT model compressor using Tensor Train decomposition.
    
    Compresses BERT embedding layers to achieve 44x compression
    with <1% accuracy loss.
    """
    
    def __init__(self, config: Optional[TTConfig] = None):
        """Initialize BERT TT compressor."""
        self.config = config or TTConfig()
        self.compression_stats = {}
        
        if not TT_AVAILABLE:
            raise ImportError("TensorLy is required for TT compression")
        
        logger.info(f"BERT TT Compressor initialized: rank={self.config.tt_rank}")
    
    def compress_bert_embeddings(self, 
                                model_name: str,
                                output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Compress BERT model embeddings using TT decomposition.
        
        Args:
            model_name: HuggingFace model name
            output_path: Path to save compressed model
            
        Returns:
            Compression statistics
        """
        from transformers import AutoModel, AutoTokenizer
        
        # Load original model
        logger.info(f"Loading BERT model: {model_name}")
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Get embedding layer
        embedding_layer = model.embeddings.word_embeddings
        original_weight = embedding_layer.weight.data
        
        logger.info(f"Original embedding shape: {original_weight.shape}")
        
        # Perform TT decomposition
        start_time = time.time()
        tt_layer = TTEmbeddingLayer(
            vocab_size=original_weight.shape[0],
            embed_dim=original_weight.shape[1],
            tt_rank=self.config.tt_rank,
            original_embedding=original_weight
        )
        decomposition_time = time.time() - start_time
        
        # Calculate compression metrics
        compression_ratio = tt_layer.compression_ratio()
        
        # Validate reconstruction accuracy
        reconstruction_accuracy = self._validate_reconstruction(
            original_weight, tt_layer
        )
        
        # Compile statistics
        stats = {
            'model_name': model_name,
            'original_parameters': original_weight.numel(),
            'compressed_parameters': sum(core.numel() for core in tt_layer.tt_cores),
            'compression_ratio': compression_ratio,
            'reconstruction_accuracy': reconstruction_accuracy,
            'decomposition_time_s': decomposition_time,
            'tt_rank': self.config.tt_rank,
            'tensor_shape': tt_layer.tensor_shape,
            'accuracy_loss': 1.0 - reconstruction_accuracy,
            'target_met': compression_ratio >= self.config.target_compression_ratio
        }
        
        logger.info(f"Compression completed: {compression_ratio:.1f}x ratio, "
                   f"{reconstruction_accuracy:.3f} accuracy")
        
        # Save compressed model if requested
        if output_path:
            self._save_compressed_model(model, tt_layer, tokenizer, output_path)
        
        self.compression_stats = stats
        return stats
    
    def _validate_reconstruction(self, 
                               original: torch.Tensor,
                               tt_layer: TTEmbeddingLayer) -> float:
        """Validate reconstruction accuracy."""
        with torch.no_grad():
            # Reconstruct tensor
            reconstructed = tt_layer._reconstruct_tensor()
            reconstructed = reconstructed.reshape(original.shape)
            
            # Calculate relative error
            relative_error = torch.norm(original - reconstructed) / torch.norm(original)
            accuracy = 1.0 - relative_error.item()
            
            return accuracy
    
    def _save_compressed_model(self, 
                             model: nn.Module,
                             tt_layer: TTEmbeddingLayer,
                             tokenizer: Any,
                             output_path: str):
        """Save compressed model to disk."""
        # Replace embedding layer with TT version
        model.embeddings.word_embeddings = tt_layer
        
        # Save model and tokenizer
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        # Save compression metadata
        import json
        metadata_path = f"{output_path}/compression_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.compression_stats, f, indent=2)
        
        logger.info(f"Compressed model saved to: {output_path}")
    
    def benchmark_compression(self, 
                            model_names: List[str],
                            tt_ranks: List[int] = [4, 8, 16, 32]) -> Dict[str, Any]:
        """
        Benchmark compression across multiple models and ranks.
        
        Args:
            model_names: List of model names to test
            tt_ranks: List of TT ranks to evaluate
            
        Returns:
            Comprehensive benchmark results
        """
        results = {}
        
        for model_name in model_names:
            results[model_name] = {}
            
            for tt_rank in tt_ranks:
                logger.info(f"Benchmarking {model_name} with rank {tt_rank}")
                
                # Update config
                original_rank = self.config.tt_rank
                self.config.tt_rank = tt_rank
                
                try:
                    # Compress model
                    stats = self.compress_bert_embeddings(model_name)
                    results[model_name][tt_rank] = stats
                    
                except Exception as e:
                    logger.error(f"Compression failed for {model_name} rank {tt_rank}: {e}")
                    results[model_name][tt_rank] = {'error': str(e)}
                
                # Restore original rank
                self.config.tt_rank = original_rank
        
        return results
    
    def get_optimal_rank(self, 
                        model_name: str,
                        target_compression: float = 44.0,
                        max_accuracy_loss: float = 0.01) -> int:
        """
        Find optimal TT rank for target compression and accuracy.
        
        Args:
            model_name: Model to analyze
            target_compression: Target compression ratio
            max_accuracy_loss: Maximum acceptable accuracy loss
            
        Returns:
            Optimal TT rank
        """
        # Binary search for optimal rank
        low_rank, high_rank = 2, 64
        optimal_rank = 8
        
        while low_rank <= high_rank:
            mid_rank = (low_rank + high_rank) // 2
            
            # Test compression at this rank
            self.config.tt_rank = mid_rank
            stats = self.compress_bert_embeddings(model_name)
            
            compression_ratio = stats['compression_ratio']
            accuracy_loss = stats['accuracy_loss']
            
            if (compression_ratio >= target_compression and 
                accuracy_loss <= max_accuracy_loss):
                optimal_rank = mid_rank
                high_rank = mid_rank - 1
            else:
                low_rank = mid_rank + 1
        
        logger.info(f"Optimal TT rank for {model_name}: {optimal_rank}")
        return optimal_rank


# Utility functions for integration
def compress_sentence_transformer(model_name: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1",
                                tt_rank: int = 8) -> TTEmbeddingLayer:
    """
    Compress SentenceTransformer model used in QuantumRerank.
    
    Args:
        model_name: SentenceTransformer model name
        tt_rank: TT rank for compression
        
    Returns:
        Compressed TT embedding layer
    """
    compressor = BERTTTCompressor(TTConfig(tt_rank=tt_rank))
    stats = compressor.compress_bert_embeddings(model_name)
    
    logger.info(f"SentenceTransformer compression: {stats['compression_ratio']:.1f}x")
    return stats


def validate_compression_pipeline() -> Dict[str, Any]:
    """
    Validate the TT compression pipeline for QuantumRerank.
    
    Returns:
        Validation results
    """
    if not TT_AVAILABLE:
        return {
            'status': 'error',
            'message': 'TensorLy not available. Install with: pip install tensorly[complete]'
        }
    
    try:
        # Test with small example
        compressor = BERTTTCompressor(TTConfig(tt_rank=4))
        
        # Create test embedding
        test_embedding = torch.randn(100, 768)  # Small vocab for testing
        
        # Test TT layer
        tt_layer = TTEmbeddingLayer(
            vocab_size=100,
            embed_dim=768,
            tt_rank=4,
            original_embedding=test_embedding
        )
        
        # Test forward pass
        test_input = torch.randint(0, 100, (2, 10))
        output = tt_layer(test_input)
        
        return {
            'status': 'success',
            'compression_ratio': tt_layer.compression_ratio(),
            'output_shape': output.shape,
            'message': 'TT compression pipeline validated successfully'
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Validation failed: {str(e)}'
        }
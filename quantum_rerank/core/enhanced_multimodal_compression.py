"""
Enhanced Multimodal Compression for Text, Clinical Data, and Medical Images.

This module implements comprehensive quantum-inspired compression for all modalities
including medical images, following the research insights from quantum embeddings
projection and multimodal quantum machine learning as specified in QMMR-04.

Based on:
- QMMR-04 task requirements
- Quantum-inspired embeddings projection research (32x parameter efficiency)
- Three-way fusion (text + clinical + image)
- Medical domain optimization
"""

import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass

from .quantum_compression import QuantumCompressionHead, QuantumCompressionConfig
from ..config.settings import SimilarityEngineConfig
from ..config.medical_image_config import MedicalImageConfig

logger = logging.getLogger(__name__)


@dataclass
class MultimodalCompressionResult:
    """Result of enhanced multimodal compression."""
    
    # Core compressed embeddings
    fused_embedding: Optional[np.ndarray] = None
    
    # Individual compressed modalities
    compressed_text: Optional[np.ndarray] = None
    compressed_clinical: Optional[np.ndarray] = None
    compressed_image: Optional[np.ndarray] = None
    
    # Compression metadata
    compression_ratio: float = 1.0
    total_compression_time_ms: float = 0.0
    
    # Quality metrics
    information_preservation_score: float = 0.0
    modality_balance_score: float = 0.0
    
    # Modality contributions
    modality_weights: Dict[str, float] = None
    modalities_used: List[str] = None
    
    # Success indicators
    compression_success: bool = False
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.modality_weights is None:
            self.modality_weights = {}
        if self.modalities_used is None:
            self.modalities_used = []


class AdaptiveModalityWeighting(nn.Module):
    """
    Adaptive weighting mechanism for balancing different modalities.
    
    Learns optimal weights for combining text, clinical, and image modalities
    based on their information content and relevance.
    """
    
    def __init__(self, 
                 text_dim: int = 85,
                 clinical_dim: int = 85, 
                 image_dim: int = 86):
        super().__init__()
        
        self.text_dim = text_dim
        self.clinical_dim = clinical_dim
        self.image_dim = image_dim
        
        # Modality-specific feature extractors
        self.text_feature_extractor = nn.Linear(text_dim, 32)
        self.clinical_feature_extractor = nn.Linear(clinical_dim, 32)
        self.image_feature_extractor = nn.Linear(image_dim, 32)
        
        # Combined weighting network
        self.weight_network = nn.Sequential(
            nn.Linear(96, 64),  # 3 * 32 = 96
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # Weights for 3 modalities
            nn.Softmax(dim=-1)
        )
        
        # Quality assessment network
        self.quality_network = nn.Sequential(
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        logger.info("AdaptiveModalityWeighting initialized")
    
    def forward(self, 
                text_emb: Optional[torch.Tensor] = None,
                clinical_emb: Optional[torch.Tensor] = None,
                image_emb: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Compute adaptive weights for modality fusion.
        
        Args:
            text_emb: Text embedding tensor
            clinical_emb: Clinical embedding tensor  
            image_emb: Image embedding tensor
            
        Returns:
            Tuple of (weighted_embeddings, modality_weights, quality_score)
        """
        # Extract features from available modalities
        features = []
        embeddings = []
        modality_mask = []
        
        if text_emb is not None:
            text_features = torch.tanh(self.text_feature_extractor(text_emb))
            features.append(text_features)
            embeddings.append(text_emb)
            modality_mask.append(1.0)
        else:
            features.append(torch.zeros(1, 32))
            embeddings.append(torch.zeros(1, self.text_dim))
            modality_mask.append(0.0)
        
        if clinical_emb is not None:
            clinical_features = torch.tanh(self.clinical_feature_extractor(clinical_emb))
            features.append(clinical_features)
            embeddings.append(clinical_emb)
            modality_mask.append(1.0)
        else:
            features.append(torch.zeros(1, 32))
            embeddings.append(torch.zeros(1, self.clinical_dim))
            modality_mask.append(0.0)
        
        if image_emb is not None:
            image_features = torch.tanh(self.image_feature_extractor(image_emb))
            features.append(image_features)
            embeddings.append(image_emb)
            modality_mask.append(1.0)
        else:
            features.append(torch.zeros(1, 32))
            embeddings.append(torch.zeros(1, self.image_dim))
            modality_mask.append(0.0)
        
        # Combine features
        combined_features = torch.cat(features, dim=-1)
        
        # Compute modality weights
        raw_weights = self.weight_network(combined_features)
        
        # Apply modality mask to zero out unavailable modalities
        modality_mask_tensor = torch.tensor(modality_mask, dtype=torch.float32)
        masked_weights = raw_weights * modality_mask_tensor
        
        # Renormalize weights
        weight_sum = torch.sum(masked_weights, dim=-1, keepdim=True)
        normalized_weights = masked_weights / (weight_sum + 1e-8)
        
        # Apply weights to embeddings
        weighted_embeddings = []
        for i, (emb, weight) in enumerate(zip(embeddings, torch.unbind(normalized_weights, dim=-1))):
            weighted_emb = emb * weight.unsqueeze(-1)
            weighted_embeddings.append(weighted_emb)
        
        # Compute quality score
        quality_score = self.quality_network(combined_features).squeeze()
        
        return torch.cat(weighted_embeddings, dim=-1), normalized_weights, quality_score


class EnhancedMultimodalCompression:
    """
    Enhanced quantum compression for text, clinical data, and medical images.
    
    Implements sophisticated multimodal fusion with adaptive weighting,
    quantum-inspired compression, and information preservation optimization.
    """
    
    def __init__(self, config: SimilarityEngineConfig = None):
        """
        Initialize enhanced multimodal compression.
        
        Args:
            config: Similarity engine configuration
        """
        self.config = config or SimilarityEngineConfig()
        
        # Modality dimensions (from research: balanced allocation)
        self.text_dim = 85
        self.clinical_dim = 85  
        self.image_dim = 86
        self.total_intermediate_dim = self.text_dim + self.clinical_dim + self.image_dim  # 256
        
        # Target output dimension
        self.target_output_dim = 256  # Flexible based on requirements
        
        # Initialize individual modality compressors
        self._initialize_modality_compressors()
        
        # Initialize adaptive weighting
        self.adaptive_weighting = AdaptiveModalityWeighting(
            self.text_dim, self.clinical_dim, self.image_dim
        )
        
        # Initialize final fusion compressor
        self._initialize_fusion_compressor()
        
        # Performance monitoring
        self.compression_stats = {
            'total_compressions': 0,
            'avg_compression_time_ms': 0.0,
            'avg_compression_ratio': 0.0,
            'modality_usage_distribution': {'text': 0, 'clinical': 0, 'image': 0},
            'information_preservation_scores': []
        }
        
        logger.info(f"EnhancedMultimodalCompression initialized: "
                   f"{self.total_intermediate_dim} -> {self.target_output_dim}")
    
    def _initialize_modality_compressors(self):
        """Initialize quantum compressors for individual modalities."""
        try:
            # Text compressor (768 -> 85)
            text_config = QuantumCompressionConfig(
                input_dim=768,
                output_dim=self.text_dim,
                compression_stages=3,
                enable_bloch_parameterization=True
            )
            self.text_compressor = QuantumCompressionHead(text_config)
            
            # Clinical compressor (768 -> 85) 
            clinical_config = QuantumCompressionConfig(
                input_dim=768,
                output_dim=self.clinical_dim,
                compression_stages=3,
                enable_bloch_parameterization=True
            )
            self.clinical_compressor = QuantumCompressionHead(clinical_config)
            
            # Image compressor (128 -> 86) - already compressed by image processor
            image_config = QuantumCompressionConfig(
                input_dim=128,
                output_dim=self.image_dim,
                compression_stages=2,
                enable_bloch_parameterization=True
            )
            self.image_compressor = QuantumCompressionHead(image_config)
            
            logger.info("Modality-specific quantum compressors initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize modality compressors: {e}")
            self.text_compressor = None
            self.clinical_compressor = None
            self.image_compressor = None
    
    def _initialize_fusion_compressor(self):
        """Initialize final fusion compressor."""
        try:
            # Final fusion compressor (256 -> 256) - maintains dimension but optimizes representation
            fusion_config = QuantumCompressionConfig(
                input_dim=self.total_intermediate_dim,
                output_dim=self.target_output_dim,
                compression_stages=2,
                enable_bloch_parameterization=True
            )
            self.fusion_compressor = QuantumCompressionHead(fusion_config)
            
            logger.info("Fusion quantum compressor initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize fusion compressor: {e}")
            self.fusion_compressor = None
    
    def fuse_all_modalities(self, modalities: Dict[str, np.ndarray]) -> MultimodalCompressionResult:
        """
        Fuse all available modalities into single quantum state.
        
        Args:
            modalities: Dictionary mapping modality names to embeddings
            
        Returns:
            MultimodalCompressionResult with compressed multimodal embedding
        """
        import time
        start_time = time.time()
        
        result = MultimodalCompressionResult()
        
        try:
            # Track which modalities are available
            available_modalities = list(modalities.keys())
            result.modalities_used = available_modalities
            
            # Update modality usage statistics
            for modality in available_modalities:
                if modality in self.compression_stats['modality_usage_distribution']:
                    self.compression_stats['modality_usage_distribution'][modality] += 1
            
            # Compress individual modalities
            compressed_modalities = self._compress_individual_modalities(modalities, result)
            
            if not compressed_modalities:
                result.error_message = "No valid modalities for compression"
                return result
            
            # Apply adaptive weighting
            weighted_embedding, modality_weights, quality_score = self._apply_adaptive_weighting(
                compressed_modalities
            )
            
            result.modality_weights = {
                'text': float(modality_weights[0]) if len(modality_weights) > 0 else 0.0,
                'clinical': float(modality_weights[1]) if len(modality_weights) > 1 else 0.0,
                'image': float(modality_weights[2]) if len(modality_weights) > 2 else 0.0
            }
            result.modality_balance_score = float(quality_score)
            
            # Final fusion compression
            if self.fusion_compressor:
                with torch.no_grad():
                    final_compressed = self.fusion_compressor(weighted_embedding)
                    result.fused_embedding = final_compressed.squeeze().cpu().numpy()
            else:
                # Fallback: simple concatenation and truncation
                result.fused_embedding = self._fallback_fusion(weighted_embedding.squeeze().cpu().numpy())
            
            # Normalize final embedding
            if result.fused_embedding is not None and np.linalg.norm(result.fused_embedding) > 0:
                result.fused_embedding = result.fused_embedding / np.linalg.norm(result.fused_embedding)
            
            # Calculate compression metrics
            original_total_dim = sum(len(emb) for emb in modalities.values())
            compressed_dim = len(result.fused_embedding) if result.fused_embedding is not None else 0
            result.compression_ratio = original_total_dim / compressed_dim if compressed_dim > 0 else 1.0
            
            # Assess information preservation
            result.information_preservation_score = self._assess_information_preservation(
                modalities, result.fused_embedding, result.modality_weights
            )
            
            # Timing and success
            result.total_compression_time_ms = (time.time() - start_time) * 1000
            result.compression_success = True
            
            # Update statistics
            self._update_compression_stats(result)
            
            return result
            
        except Exception as e:
            # Error handling
            result.total_compression_time_ms = (time.time() - start_time) * 1000
            result.error_message = str(e)
            result.compression_success = False
            
            logger.error(f"Enhanced multimodal compression failed: {e}")
            
            # Fallback: return zero embedding
            result.fused_embedding = np.zeros(self.target_output_dim)
            return result
    
    def _compress_individual_modalities(self, 
                                      modalities: Dict[str, np.ndarray],
                                      result: MultimodalCompressionResult) -> Dict[str, torch.Tensor]:
        """Compress individual modalities using quantum compressors."""
        compressed_modalities = {}
        
        # Compress text modality
        if 'text' in modalities and modalities['text'] is not None:
            try:
                if self.text_compressor:
                    text_tensor = torch.tensor(modalities['text'], dtype=torch.float32)
                    with torch.no_grad():
                        compressed_text = self.text_compressor(text_tensor.unsqueeze(0))
                        compressed_modalities['text'] = compressed_text.squeeze(0)
                        result.compressed_text = compressed_text.squeeze().cpu().numpy()
                else:
                    # Fallback: simple truncation
                    truncated = modalities['text'][:self.text_dim]
                    padded = np.pad(truncated, (0, max(0, self.text_dim - len(truncated))), mode='constant')
                    compressed_modalities['text'] = torch.tensor(padded, dtype=torch.float32)
                    result.compressed_text = padded
            except Exception as e:
                logger.warning(f"Text compression failed: {e}")
        
        # Compress clinical modality
        if 'clinical' in modalities and modalities['clinical'] is not None:
            try:
                if self.clinical_compressor:
                    clinical_tensor = torch.tensor(modalities['clinical'], dtype=torch.float32)
                    with torch.no_grad():
                        compressed_clinical = self.clinical_compressor(clinical_tensor.unsqueeze(0))
                        compressed_modalities['clinical'] = compressed_clinical.squeeze(0)
                        result.compressed_clinical = compressed_clinical.squeeze().cpu().numpy()
                else:
                    # Fallback: simple truncation
                    truncated = modalities['clinical'][:self.clinical_dim]
                    padded = np.pad(truncated, (0, max(0, self.clinical_dim - len(truncated))), mode='constant')
                    compressed_modalities['clinical'] = torch.tensor(padded, dtype=torch.float32)
                    result.compressed_clinical = padded
            except Exception as e:
                logger.warning(f"Clinical compression failed: {e}")
        
        # Compress image modality
        if 'image' in modalities and modalities['image'] is not None:
            try:
                if self.image_compressor:
                    image_tensor = torch.tensor(modalities['image'], dtype=torch.float32)
                    with torch.no_grad():
                        compressed_image = self.image_compressor(image_tensor.unsqueeze(0))
                        compressed_modalities['image'] = compressed_image.squeeze(0)
                        result.compressed_image = compressed_image.squeeze().cpu().numpy()
                else:
                    # Fallback: simple truncation/padding
                    truncated = modalities['image'][:self.image_dim]
                    padded = np.pad(truncated, (0, max(0, self.image_dim - len(truncated))), mode='constant')
                    compressed_modalities['image'] = torch.tensor(padded, dtype=torch.float32)
                    result.compressed_image = padded
            except Exception as e:
                logger.warning(f"Image compression failed: {e}")
        
        return compressed_modalities
    
    def _apply_adaptive_weighting(self, 
                                compressed_modalities: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply adaptive weighting to compressed modalities."""
        try:
            # Prepare modality inputs for adaptive weighting
            text_emb = compressed_modalities.get('text')
            clinical_emb = compressed_modalities.get('clinical')
            image_emb = compressed_modalities.get('image')
            
            # Ensure all tensors have batch dimension
            if text_emb is not None and len(text_emb.shape) == 1:
                text_emb = text_emb.unsqueeze(0)
            if clinical_emb is not None and len(clinical_emb.shape) == 1:
                clinical_emb = clinical_emb.unsqueeze(0)
            if image_emb is not None and len(image_emb.shape) == 1:
                image_emb = image_emb.unsqueeze(0)
            
            # Apply adaptive weighting
            with torch.no_grad():
                weighted_embedding, modality_weights, quality_score = self.adaptive_weighting(
                    text_emb, clinical_emb, image_emb
                )
            
            return weighted_embedding, modality_weights.squeeze(), quality_score
            
        except Exception as e:
            logger.warning(f"Adaptive weighting failed: {e}")
            
            # Fallback: simple concatenation
            available_embeddings = []
            for modality in ['text', 'clinical', 'image']:
                if modality in compressed_modalities:
                    emb = compressed_modalities[modality]
                    if len(emb.shape) == 1:
                        emb = emb.unsqueeze(0)
                    available_embeddings.append(emb)
                else:
                    # Add zero padding
                    if modality == 'text':
                        available_embeddings.append(torch.zeros(1, self.text_dim))
                    elif modality == 'clinical':
                        available_embeddings.append(torch.zeros(1, self.clinical_dim))
                    else:  # image
                        available_embeddings.append(torch.zeros(1, self.image_dim))
            
            weighted_embedding = torch.cat(available_embeddings, dim=-1)
            modality_weights = torch.tensor([1.0/3, 1.0/3, 1.0/3])  # Equal weights
            quality_score = torch.tensor(0.5)  # Default quality
            
            return weighted_embedding, modality_weights, quality_score
    
    def _fallback_fusion(self, combined_embedding: np.ndarray) -> np.ndarray:
        """Fallback fusion method when quantum compression fails."""
        input_dim = len(combined_embedding)
        
        if input_dim <= self.target_output_dim:
            # Pad with zeros if needed
            padding = self.target_output_dim - input_dim
            return np.pad(combined_embedding, (0, padding), mode='constant')
        else:
            # Simple truncation (could be improved with PCA)
            return combined_embedding[:self.target_output_dim]
    
    def _assess_information_preservation(self, 
                                       original_modalities: Dict[str, np.ndarray],
                                       fused_embedding: np.ndarray,
                                       modality_weights: Dict[str, float]) -> float:
        """Assess how well information is preserved during compression."""
        try:
            if fused_embedding is None:
                return 0.0
            
            # Calculate weighted sum of original modality norms
            original_info_content = 0.0
            total_weight = 0.0
            
            for modality, embedding in original_modalities.items():
                if embedding is not None:
                    weight = modality_weights.get(modality, 0.0)
                    info_content = np.linalg.norm(embedding) * weight
                    original_info_content += info_content
                    total_weight += weight
            
            # Normalize by total weight
            if total_weight > 0:
                original_info_content /= total_weight
            
            # Calculate compressed information content
            compressed_info_content = np.linalg.norm(fused_embedding)
            
            # Information preservation ratio
            if original_info_content > 0:
                preservation_ratio = compressed_info_content / original_info_content
                # Clip to reasonable range (compression can sometimes increase norm due to normalization)
                preservation_ratio = min(preservation_ratio, 1.0)
            else:
                preservation_ratio = 0.0
            
            return float(preservation_ratio)
            
        except Exception as e:
            logger.warning(f"Information preservation assessment failed: {e}")
            return 0.5  # Default score
    
    def _update_compression_stats(self, result: MultimodalCompressionResult):
        """Update compression statistics."""
        self.compression_stats['total_compressions'] += 1
        
        if result.compression_success:
            # Update average compression time
            n = self.compression_stats['total_compressions']
            current_avg = self.compression_stats['avg_compression_time_ms']
            self.compression_stats['avg_compression_time_ms'] = (
                (current_avg * (n - 1) + result.total_compression_time_ms) / n
            )
            
            # Update average compression ratio
            current_ratio_avg = self.compression_stats['avg_compression_ratio']
            self.compression_stats['avg_compression_ratio'] = (
                (current_ratio_avg * (n - 1) + result.compression_ratio) / n
            )
            
            # Track information preservation scores
            self.compression_stats['information_preservation_scores'].append(
                result.information_preservation_score
            )
            
            # Keep only recent scores (last 100)
            if len(self.compression_stats['information_preservation_scores']) > 100:
                self.compression_stats['information_preservation_scores'] = \
                    self.compression_stats['information_preservation_scores'][-100:]
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get comprehensive compression statistics."""
        stats = self.compression_stats.copy()
        
        # Calculate derived metrics
        if stats['information_preservation_scores']:
            stats['avg_information_preservation'] = np.mean(stats['information_preservation_scores'])
            stats['min_information_preservation'] = np.min(stats['information_preservation_scores'])
            stats['max_information_preservation'] = np.max(stats['information_preservation_scores'])
        
        # Calculate modality usage percentages
        total_usage = sum(stats['modality_usage_distribution'].values())
        if total_usage > 0:
            stats['modality_usage_percentages'] = {
                modality: count / total_usage
                for modality, count in stats['modality_usage_distribution'].items()
            }
        
        # Add compression efficiency metrics
        stats['compression_efficiency'] = {
            'quantum_compressors_available': all([
                self.text_compressor is not None,
                self.clinical_compressor is not None,
                self.image_compressor is not None,
                self.fusion_compressor is not None
            ]),
            'adaptive_weighting_enabled': self.adaptive_weighting is not None,
            'target_compression_ratio': self.total_intermediate_dim / self.target_output_dim
        }
        
        return stats
    
    def optimize_for_modality_distribution(self, modality_usage_stats: Dict[str, float]):
        """
        Optimize compression based on observed modality usage patterns.
        
        Args:
            modality_usage_stats: Statistics about modality usage frequency
        """
        try:
            # Adjust compression dimensions based on usage frequency
            total_usage = sum(modality_usage_stats.values())
            
            if total_usage > 0:
                # Reallocate dimensions based on usage
                text_usage = modality_usage_stats.get('text', 0) / total_usage
                clinical_usage = modality_usage_stats.get('clinical', 0) / total_usage
                image_usage = modality_usage_stats.get('image', 0) / total_usage
                
                # Adjust dimensions (keeping total at 256)
                base_dim = 80
                extra_dims = 16
                
                self.text_dim = base_dim + int(extra_dims * text_usage)
                self.clinical_dim = base_dim + int(extra_dims * clinical_usage)
                self.image_dim = base_dim + int(extra_dims * image_usage)
                
                # Ensure total doesn't exceed limit
                total_dim = self.text_dim + self.clinical_dim + self.image_dim
                if total_dim > 256:
                    scale_factor = 256 / total_dim
                    self.text_dim = int(self.text_dim * scale_factor)
                    self.clinical_dim = int(self.clinical_dim * scale_factor)
                    self.image_dim = 256 - self.text_dim - self.clinical_dim
                
                logger.info(f"Optimized dimensions: text={self.text_dim}, "
                           f"clinical={self.clinical_dim}, image={self.image_dim}")
        
        except Exception as e:
            logger.warning(f"Modality optimization failed: {e}")
    
    def get_compression_ratio(self) -> float:
        """Get current compression ratio."""
        return self.total_intermediate_dim / self.target_output_dim
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter counts for all compression components."""
        param_counts = {}
        
        if self.text_compressor:
            param_counts['text_compressor'] = sum(p.numel() for p in self.text_compressor.parameters())
        
        if self.clinical_compressor:
            param_counts['clinical_compressor'] = sum(p.numel() for p in self.clinical_compressor.parameters())
        
        if self.image_compressor:
            param_counts['image_compressor'] = sum(p.numel() for p in self.image_compressor.parameters())
        
        if self.fusion_compressor:
            param_counts['fusion_compressor'] = sum(p.numel() for p in self.fusion_compressor.parameters())
        
        param_counts['adaptive_weighting'] = sum(p.numel() for p in self.adaptive_weighting.parameters())
        param_counts['total'] = sum(param_counts.values())
        
        return param_counts
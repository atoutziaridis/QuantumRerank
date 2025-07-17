"""
Multi-Modal Tensor Fusion for Unified Representation.

This module implements quantum-inspired tensor fusion for combining text, image, and
tabular data into unified representations using tensor product operations and
Matrix Product State decompositions.

Based on:
- Tensor Product Multi-Modal fusion from transition plan
- Quantum entanglement-inspired feature fusion
- Unified tensor representation for multiple modalities
- MPS encoding for efficient multi-modal processing

Key Features:
- Tensor product fusion for modality combination
- MPS-based modality encoders for compression
- Unified representation space for all modalities
- Quantum-inspired correlation modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MultiModalFusionConfig:
    """Configuration for multi-modal tensor fusion."""
    text_dim: int = 768
    image_dim: int = 2048
    tabular_dim: int = 100
    unified_dim: int = 512
    bond_dim: int = 32
    num_fusion_layers: int = 3
    dropout: float = 0.1
    use_attention_fusion: bool = True
    temperature: float = 1.0


class MPSModalityEncoder(nn.Module):
    """
    MPS-based encoder for individual modalities.
    
    Compresses modality-specific features using Matrix Product State
    decomposition for efficient representation.
    """
    
    def __init__(self, input_dim: int, output_dim: int, bond_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        
        # Calculate MPS chain length
        self.chain_length = max(3, min(8, int(np.log2(input_dim)) + 1))
        
        # MPS cores for modality encoding
        self.mps_cores = self._build_mps_chain()
        
        # Output projection to unified dimension
        # Calculate output dimension based on final processed dimension
        final_dim = min(self.bond_dim, self.input_dim)
        self.output_projection = nn.Linear(final_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        logger.debug(f"MPS Encoder: {input_dim}→{output_dim}, bond_dim={bond_dim}, "
                    f"chain_length={self.chain_length}")
    
    def _build_mps_chain(self) -> nn.ModuleList:
        """Build MPS chain for modality encoding using linear layers."""
        layers = nn.ModuleList()
        
        # Progressive dimensionality reduction
        current_dim = self.input_dim
        target_dim = min(self.bond_dim, self.input_dim)
        
        # Create a series of linear projections that simulate MPS compression
        while current_dim > target_dim and len(layers) < self.chain_length:
            next_dim = max(target_dim, current_dim // 2)
            layer = nn.Linear(current_dim, next_dim, bias=False)
            layers.append(layer)
            current_dim = next_dim
        
        # Ensure at least one layer exists
        if len(layers) == 0:
            layers.append(nn.Linear(self.input_dim, target_dim, bias=False))
        
        return layers
    
    def _calculate_mps_output_dim(self) -> int:
        """Calculate the output dimension from MPS contraction."""
        return self.bond_dim * len(self.mps_cores)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode modality input using MPS decomposition.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Encoded tensor (batch_size, output_dim)
        """
        batch_size = x.size(0)
        
        # Reshape input for MPS processing
        if x.size(1) != self.input_dim:
            # Adaptive pooling if dimensions don't match
            x = F.adaptive_avg_pool1d(x.unsqueeze(1), self.input_dim).squeeze(1)
        
        # Apply MPS chain processing
        result = x
        
        # Progressive dimensionality reduction through MPS chain
        for layer in self.mps_cores:
            result = F.relu(layer(result))
        
        # Project to unified dimension
        output = self.output_projection(result)
        output = self.layer_norm(output)
        
        return output


class TensorProductFusion(nn.Module):
    """
    Tensor product fusion layer for combining multiple modalities.
    
    Creates unified representations by computing tensor products between
    modality features, capturing cross-modal correlations.
    """
    
    def __init__(self, modality_dims: List[int], output_dim: int):
        super().__init__()
        self.modality_dims = modality_dims
        self.output_dim = output_dim
        self.num_modalities = len(modality_dims)
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.randn(self.num_modalities, self.num_modalities))
        nn.init.eye_(self.fusion_weights)
        
        # Output projection from tensor product space
        total_product_dim = np.prod(modality_dims)
        self.compression_layer = nn.Sequential(
            nn.Linear(total_product_dim, output_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 4, output_dim)
        )
        
    def forward(self, modality_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse modality features using tensor products.
        
        Args:
            modality_features: List of modality tensors [(batch, dim1), (batch, dim2), ...]
            
        Returns:
            Fused tensor (batch_size, output_dim)
        """
        if len(modality_features) != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {len(modality_features)}")
        
        batch_size = modality_features[0].size(0)
        
        # Normalize features
        normalized_features = [F.normalize(feat, p=2, dim=-1) for feat in modality_features]
        
        # Compute weighted tensor product
        if self.num_modalities == 2:
            # Efficient 2-modality tensor product
            feat1, feat2 = normalized_features
            weight = self.fusion_weights[0, 1]
            tensor_product = torch.einsum('bi,bj->bij', feat1, feat2) * weight
            tensor_product = tensor_product.view(batch_size, -1)
        
        elif self.num_modalities == 3:
            # 3-modality tensor product
            feat1, feat2, feat3 = normalized_features
            # Pairwise products with learned weights
            prod12 = torch.einsum('bi,bj->bij', feat1, feat2) * self.fusion_weights[0, 1]
            prod13 = torch.einsum('bi,bj->bij', feat1, feat3) * self.fusion_weights[0, 2]
            prod23 = torch.einsum('bi,bj->bij', feat2, feat3) * self.fusion_weights[1, 2]
            
            # Combine products
            tensor_product = torch.cat([
                prod12.view(batch_size, -1),
                prod13.view(batch_size, -1),
                prod23.view(batch_size, -1)
            ], dim=1)
        
        else:
            # General case for multiple modalities
            tensor_product = normalized_features[0]
            for i in range(1, self.num_modalities):
                weight = self.fusion_weights[0, i]
                tensor_product = torch.einsum('bi,bj->bij', tensor_product, normalized_features[i])
                tensor_product = tensor_product.view(batch_size, -1) * weight
        
        # Compress to output dimension
        fused_output = self.compression_layer(tensor_product)
        
        return fused_output


class QuantumInspiredCrossAttention(nn.Module):
    """
    Quantum-inspired cross-attention for modality fusion.
    
    Uses complex-valued embeddings and quantum coherence principles
    for enhanced cross-modal attention computation.
    """
    
    def __init__(self, query_dim: int, key_dim: int, value_dim: int, output_dim: int):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.output_dim = output_dim
        
        # Complex-valued projections
        self.query_real = nn.Linear(query_dim, output_dim)
        self.query_imag = nn.Linear(query_dim, output_dim)
        self.key_real = nn.Linear(key_dim, output_dim)
        self.key_imag = nn.Linear(key_dim, output_dim)
        self.value_projection = nn.Linear(value_dim, output_dim)
        
        # Coherence parameters
        self.coherence_weight = nn.Parameter(torch.tensor(1.0))
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute quantum-inspired cross-attention.
        
        Args:
            query: Query modality features (batch, seq, query_dim)
            key: Key modality features (batch, seq, key_dim) 
            value: Value modality features (batch, seq, value_dim)
            
        Returns:
            Tuple of (attended_output, coherence_measure)
        """
        # Project to complex space
        q_real = self.query_real(query)
        q_imag = self.query_imag(query)
        k_real = self.key_real(key)
        k_imag = self.key_imag(key)
        v = self.value_projection(value)
        
        # Complex attention scores
        # Real part: q_real * k_real + q_imag * k_imag
        # Imaginary part: q_imag * k_real - q_real * k_imag
        attention_real = torch.matmul(q_real, k_real.transpose(-2, -1)) + \
                        torch.matmul(q_imag, k_imag.transpose(-2, -1))
        attention_imag = torch.matmul(q_imag, k_real.transpose(-2, -1)) - \
                        torch.matmul(q_real, k_imag.transpose(-2, -1))
        
        # Compute attention magnitude (quantum probability)
        attention_magnitude = torch.sqrt(attention_real**2 + attention_imag**2)
        attention_weights = F.softmax(attention_magnitude / np.sqrt(self.output_dim), dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, v)
        
        # Compute quantum coherence measure
        coherence = torch.mean(attention_imag**2) * self.coherence_weight
        
        return output, coherence


class MultiModalTensorFusion(nn.Module):
    """
    Complete multi-modal tensor fusion system.
    
    Combines text, image, and tabular data into unified representations
    using quantum-inspired tensor operations and MPS decompositions.
    """
    
    def __init__(self, config: MultiModalFusionConfig):
        super().__init__()
        self.config = config
        
        # Modality-specific MPS encoders
        self.text_encoder = MPSModalityEncoder(
            config.text_dim, config.unified_dim, config.bond_dim
        )
        self.image_encoder = MPSModalityEncoder(
            config.image_dim, config.unified_dim, config.bond_dim
        )
        self.tabular_encoder = MPSModalityEncoder(
            config.tabular_dim, config.unified_dim, config.bond_dim
        )
        
        # Tensor product fusion
        self.tensor_fusion = TensorProductFusion(
            [config.unified_dim] * 3, config.unified_dim
        )
        
        # Cross-attention fusion (optional)
        if config.use_attention_fusion:
            self.cross_attention = QuantumInspiredCrossAttention(
                config.unified_dim, config.unified_dim, config.unified_dim, config.unified_dim
            )
        
        # Final fusion layers
        self.fusion_layers = nn.ModuleList([
            nn.Linear(config.unified_dim, config.unified_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.unified_dim, config.unified_dim)
        ])
        
        self.layer_norm = nn.LayerNorm(config.unified_dim)
        
        logger.info(f"Initialized Multi-Modal Tensor Fusion: "
                   f"text={config.text_dim}, image={config.image_dim}, "
                   f"tabular={config.tabular_dim} → unified={config.unified_dim}")
    
    def forward(
        self,
        text_features: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        tabular_features: Optional[torch.Tensor] = None,
        fusion_method: str = "tensor_product"
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse multi-modal features into unified representation.
        
        Args:
            text_features: Text embeddings (batch_size, text_dim)
            image_features: Image features (batch_size, image_dim)
            tabular_features: Tabular features (batch_size, tabular_dim)
            fusion_method: Fusion method ("tensor_product", "attention", "hybrid")
            
        Returns:
            Dictionary with fused features and metadata
        """
        batch_size = None
        encoded_modalities = []
        modality_names = []
        
        # Encode available modalities
        if text_features is not None:
            batch_size = text_features.size(0)
            text_encoded = self.text_encoder(text_features)
            encoded_modalities.append(text_encoded)
            modality_names.append("text")
        
        if image_features is not None:
            if batch_size is None:
                batch_size = image_features.size(0)
            image_encoded = self.image_encoder(image_features)
            encoded_modalities.append(image_encoded)
            modality_names.append("image")
        
        if tabular_features is not None:
            if batch_size is None:
                batch_size = tabular_features.size(0)
            tabular_encoded = self.tabular_encoder(tabular_features)
            encoded_modalities.append(tabular_encoded)
            modality_names.append("tabular")
        
        if not encoded_modalities:
            raise ValueError("At least one modality must be provided")
        
        # Pad missing modalities with zeros if needed
        while len(encoded_modalities) < 3:
            zero_modality = torch.zeros(batch_size, self.config.unified_dim, 
                                      device=encoded_modalities[0].device)
            encoded_modalities.append(zero_modality)
            modality_names.append("zero_padded")
        
        # Fusion based on method
        if fusion_method == "tensor_product":
            fused_features = self.tensor_fusion(encoded_modalities)
            coherence_measure = torch.tensor(0.0)
        
        elif fusion_method == "attention" and self.config.use_attention_fusion:
            # Use first modality as query, others as key/value
            query = encoded_modalities[0].unsqueeze(1)  # Add sequence dimension
            key = torch.stack(encoded_modalities[1:], dim=1)  # Stack as sequence
            value = key
            
            fused_features, coherence_measure = self.cross_attention(query, key, value)
            fused_features = fused_features.squeeze(1)  # Remove sequence dimension
        
        elif fusion_method == "hybrid":
            # Combine tensor product and attention
            tensor_fused = self.tensor_fusion(encoded_modalities)
            
            if self.config.use_attention_fusion:
                query = encoded_modalities[0].unsqueeze(1)
                key = torch.stack(encoded_modalities[1:], dim=1)
                attention_fused, coherence_measure = self.cross_attention(query, key, key)
                attention_fused = attention_fused.squeeze(1)
                
                # Weighted combination
                fused_features = 0.6 * tensor_fused + 0.4 * attention_fused
            else:
                fused_features = tensor_fused
                coherence_measure = torch.tensor(0.0)
        
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Apply final fusion layers
        x = fused_features
        for layer in self.fusion_layers:
            x = layer(x)
        
        fused_features = self.layer_norm(x)
        
        return {
            "fused_features": fused_features,
            "modality_features": {
                name: feat for name, feat in zip(modality_names[:len(encoded_modalities)], 
                                               encoded_modalities)
            },
            "coherence_measure": coherence_measure,
            "fusion_method": fusion_method,
            "active_modalities": modality_names[:len([f for f in [text_features, image_features, tabular_features] if f is not None])]
        }
    
    def get_fusion_stats(self) -> Dict[str, any]:
        """Get statistics about the fusion architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        
        # Calculate compression compared to concatenation approach
        concat_dim = self.config.text_dim + self.config.image_dim + self.config.tabular_dim
        concat_projection_params = concat_dim * self.config.unified_dim
        
        compression_ratio = concat_projection_params / total_params if total_params > 0 else 1.0
        
        return {
            "total_parameters": total_params,
            "concatenation_baseline_params": concat_projection_params,
            "compression_ratio": compression_ratio,
            "unified_dimension": self.config.unified_dim,
            "bond_dimension": self.config.bond_dim,
            "modality_dimensions": {
                "text": self.config.text_dim,
                "image": self.config.image_dim,
                "tabular": self.config.tabular_dim
            }
        }


def create_multimodal_fusion(
    text_dim: int = 768,
    image_dim: int = 2048, 
    tabular_dim: int = 100,
    unified_dim: int = 512,
    bond_dim: int = 32
) -> MultiModalTensorFusion:
    """
    Factory function to create multi-modal tensor fusion module.
    
    Args:
        text_dim: Text feature dimension
        image_dim: Image feature dimension
        tabular_dim: Tabular feature dimension
        unified_dim: Unified representation dimension
        bond_dim: MPS bond dimension
        
    Returns:
        Configured MultiModalTensorFusion module
    """
    config = MultiModalFusionConfig(
        text_dim=text_dim,
        image_dim=image_dim,
        tabular_dim=tabular_dim,
        unified_dim=unified_dim,
        bond_dim=bond_dim
    )
    
    return MultiModalTensorFusion(config)
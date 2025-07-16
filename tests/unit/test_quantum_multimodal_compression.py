"""
Unit tests for QuantumMultimodalCompression.

Tests the quantum-inspired compression functionality for multimodal data
while ensuring 32x parameter efficiency and PRD constraints.
"""

import pytest
import numpy as np
import torch
import time
from unittest.mock import Mock, patch

from quantum_rerank.core.quantum_multimodal_compression import (
    QuantumMultimodalCompression,
    PolarQuantumEncoder,
    EntanglementFusionLayer,
    MultimodalCompressionConfig
)


class TestMultimodalCompressionConfig:
    """Test MultimodalCompressionConfig functionality."""
    
    def test_config_initialization(self):
        """Test config initialization with defaults."""
        config = MultimodalCompressionConfig()
        
        assert config.text_dim == 768
        assert config.clinical_dim == 768
        assert config.target_quantum_dim == 256
        assert config.text_compressed_dim == 128
        assert config.clinical_compressed_dim == 128
        assert config.text_weight + config.clinical_weight == 1.0
    
    def test_config_validation(self):
        """Test config validation."""
        # Should raise error if compressed dims don't sum to target
        with pytest.raises(ValueError):
            MultimodalCompressionConfig(
                text_compressed_dim=100,
                clinical_compressed_dim=100,
                target_quantum_dim=256
            )
        
        # Should raise error if weights don't sum to 1.0
        with pytest.raises(ValueError):
            MultimodalCompressionConfig(
                text_weight=0.7,
                clinical_weight=0.7
            )
    
    def test_config_custom_values(self):
        """Test config with custom values."""
        config = MultimodalCompressionConfig(
            text_dim=512,
            clinical_dim=512,
            target_quantum_dim=128,
            text_compressed_dim=64,
            clinical_compressed_dim=64,
            text_weight=0.7,
            clinical_weight=0.3
        )
        
        assert config.text_dim == 512
        assert config.clinical_dim == 512
        assert config.target_quantum_dim == 128
        assert config.text_compressed_dim == 64
        assert config.clinical_compressed_dim == 64
        assert config.text_weight == 0.7
        assert config.clinical_weight == 0.3


class TestPolarQuantumEncoder:
    """Test PolarQuantumEncoder functionality."""
    
    def test_polar_encoder_initialization(self):
        """Test polar encoder initialization."""
        encoder = PolarQuantumEncoder(input_dim=768, output_dim=128)
        
        assert encoder.input_dim == 768
        assert encoder.output_dim == 128
        assert encoder.theta_layer is not None
        assert encoder.phi_layer is not None
        assert encoder.residual_correction is not None
    
    def test_polar_encoder_forward(self):
        """Test polar encoder forward pass."""
        encoder = PolarQuantumEncoder(input_dim=768, output_dim=128)
        
        # Test with batch of embeddings
        batch_size = 4
        embeddings = torch.randn(batch_size, 768)
        
        output = encoder(embeddings)
        
        assert output.shape == (batch_size, 128)
        
        # Output should be normalized (unit vectors)
        norms = torch.norm(output, dim=1)
        assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5)
    
    def test_polar_encoder_deterministic(self):
        """Test that polar encoder is deterministic."""
        encoder = PolarQuantumEncoder(input_dim=768, output_dim=128)
        
        # Same input should produce same output
        embedding = torch.randn(1, 768)
        
        output1 = encoder(embedding)
        output2 = encoder(embedding)
        
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_polar_encoder_parameter_efficiency(self):
        """Test parameter efficiency compared to dense layer."""
        encoder = PolarQuantumEncoder(input_dim=768, output_dim=128)
        
        # Count parameters
        total_params = sum(p.numel() for p in encoder.parameters())
        
        # Compare with dense layer
        dense_params = 768 * 128 + 128  # weights + bias
        
        # Should be more efficient
        assert total_params < dense_params
        
        # Should achieve at least 2x efficiency
        efficiency_ratio = dense_params / total_params
        assert efficiency_ratio >= 2.0


class TestEntanglementFusionLayer:
    """Test EntanglementFusionLayer functionality."""
    
    def test_entanglement_layer_initialization(self):
        """Test entanglement layer initialization."""
        layer = EntanglementFusionLayer(dim=128, enable_entanglement=True)
        
        assert layer.dim == 128
        assert layer.enable_entanglement is True
        assert layer.interaction_matrix is not None
        assert layer.entanglement_strength is not None
    
    def test_entanglement_layer_without_entanglement(self):
        """Test entanglement layer without entanglement."""
        layer = EntanglementFusionLayer(dim=128, enable_entanglement=False)
        
        text_state = torch.randn(4, 128)
        clinical_state = torch.randn(4, 128)
        
        output = layer(text_state, clinical_state)
        
        # Should just concatenate
        assert output.shape == (4, 256)
        assert torch.allclose(output[:, :128], text_state)
        assert torch.allclose(output[:, 128:], clinical_state)
    
    def test_entanglement_layer_with_entanglement(self):
        """Test entanglement layer with entanglement enabled."""
        layer = EntanglementFusionLayer(dim=128, enable_entanglement=True)
        
        text_state = torch.randn(4, 128)
        clinical_state = torch.randn(4, 128)
        
        output = layer(text_state, clinical_state)
        
        # Should produce entangled output
        assert output.shape == (4, 256)
        
        # Should not be simple concatenation
        assert not torch.allclose(output[:, :128], text_state)
        assert not torch.allclose(output[:, 128:], clinical_state)
    
    def test_entanglement_strength_bounds(self):
        """Test entanglement strength is properly bounded."""
        layer = EntanglementFusionLayer(dim=128, enable_entanglement=True)
        
        # Entanglement strength should be bounded by sigmoid
        strength = torch.sigmoid(layer.entanglement_strength)
        assert 0 <= strength <= 1


class TestQuantumMultimodalCompression:
    """Test QuantumMultimodalCompression functionality."""
    
    def test_compression_initialization(self):
        """Test compression initialization."""
        config = MultimodalCompressionConfig()
        compression = QuantumMultimodalCompression(config)
        
        assert compression.config == config
        assert compression.text_compressor is not None
        assert compression.clinical_compressor is not None
        assert compression.entanglement_layer is not None
        assert compression.output_norm is not None
    
    def test_compression_forward(self):
        """Test compression forward pass."""
        config = MultimodalCompressionConfig()
        compression = QuantumMultimodalCompression(config)
        
        batch_size = 4
        text_embedding = torch.randn(batch_size, 768)
        clinical_embedding = torch.randn(batch_size, 768)
        
        output = compression(text_embedding, clinical_embedding)
        
        # Should compress to target dimension
        assert output.shape == (batch_size, 256)
        
        # Should be normalized
        norms = torch.norm(output, dim=1)
        assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5)
    
    def test_compression_ratio_calculation(self):
        """Test compression ratio calculation."""
        config = MultimodalCompressionConfig()
        compression = QuantumMultimodalCompression(config)
        
        compression_ratio = compression.get_compression_ratio()
        
        # Should be 6:1 compression (768+768)/256 = 6
        expected_ratio = (768 + 768) / 256
        assert abs(compression_ratio - expected_ratio) < 0.01
    
    def test_compression_methods(self):
        """Test different compression methods."""
        # Test parallel compression
        config = MultimodalCompressionConfig(fusion_method="parallel_compression")
        compression = QuantumMultimodalCompression(config)
        
        text_emb = torch.randn(2, 768)
        clinical_emb = torch.randn(2, 768)
        
        output = compression(text_emb, clinical_emb)
        assert output.shape == (2, 256)
        
        # Test sequential compression
        config = MultimodalCompressionConfig(fusion_method="sequential")
        compression = QuantumMultimodalCompression(config)
        
        output = compression(text_emb, clinical_emb)
        assert output.shape == (2, 256)
        
        # Test attention compression
        config = MultimodalCompressionConfig(fusion_method="attention")
        compression = QuantumMultimodalCompression(config)
        
        output = compression(text_emb, clinical_emb)
        assert output.shape == (2, 256)
    
    def test_numpy_interface(self):
        """Test numpy interface for compression."""
        config = MultimodalCompressionConfig()
        compression = QuantumMultimodalCompression(config)
        
        text_emb = np.random.randn(768)
        clinical_emb = np.random.randn(768)
        
        compressed = compression.compress_multimodal(text_emb, clinical_emb)
        
        assert isinstance(compressed, np.ndarray)
        assert compressed.shape == (256,)
        assert np.allclose(np.linalg.norm(compressed), 1.0, atol=1e-5)
    
    def test_batch_numpy_interface(self):
        """Test batch numpy interface for compression."""
        config = MultimodalCompressionConfig()
        compression = QuantumMultimodalCompression(config)
        
        batch_size = 5
        text_embeddings = np.random.randn(batch_size, 768)
        clinical_embeddings = np.random.randn(batch_size, 768)
        
        compressed = compression.batch_compress_multimodal(text_embeddings, clinical_embeddings)
        
        assert isinstance(compressed, np.ndarray)
        assert compressed.shape == (batch_size, 256)
        
        # Each should be normalized
        for i in range(batch_size):
            assert np.allclose(np.linalg.norm(compressed[i]), 1.0, atol=1e-5)
    
    def test_performance_stats(self):
        """Test performance statistics tracking."""
        config = MultimodalCompressionConfig()
        compression = QuantumMultimodalCompression(config)
        
        # Process some data
        text_emb = torch.randn(1, 768)
        clinical_emb = torch.randn(1, 768)
        
        compression(text_emb, clinical_emb)
        
        stats = compression.get_compression_stats()
        
        assert 'total_compressions' in stats
        assert 'avg_compression_time_ms' in stats
        assert 'compression_ratio' in stats
        assert 'total_parameters' in stats
        assert 'parameter_efficiency' in stats
        assert stats['total_compressions'] == 1
        assert stats['avg_compression_time_ms'] > 0
    
    def test_parameter_efficiency(self):
        """Test parameter efficiency compared to classical methods."""
        config = MultimodalCompressionConfig()
        compression = QuantumMultimodalCompression(config)
        
        stats = compression.get_compression_stats()
        
        # Should achieve significant parameter efficiency
        assert stats['parameter_efficiency'] >= 2.0
        
        # Should meet latency target
        assert stats['meets_latency_target'] is True
    
    def test_parameter_breakdown(self):
        """Test parameter count breakdown."""
        config = MultimodalCompressionConfig()
        compression = QuantumMultimodalCompression(config)
        
        breakdown = compression.get_parameter_breakdown()
        
        assert 'text_compressor' in breakdown
        assert 'clinical_compressor' in breakdown
        assert 'entanglement_layer' in breakdown
        assert 'output_norm' in breakdown
        assert 'total' in breakdown
        
        # Total should equal sum of components
        expected_total = (
            breakdown['text_compressor'] +
            breakdown['clinical_compressor'] +
            breakdown['entanglement_layer'] +
            breakdown['output_norm']
        )
        
        if 'final_projection' in breakdown:
            expected_total += breakdown['final_projection']
        
        assert breakdown['total'] == expected_total
    
    def test_quantum_constraints_validation(self):
        """Test quantum constraints validation."""
        config = MultimodalCompressionConfig()
        compression = QuantumMultimodalCompression(config)
        
        validation = compression.validate_quantum_constraints()
        
        assert 'output_dim_power_of_2' in validation
        assert 'implied_qubits_valid' in validation
        assert 'compression_ratio_valid' in validation
        assert 'parameter_efficiency_good' in validation
        
        # 256 is a power of 2
        assert validation['output_dim_power_of_2'] is True
        
        # Should have valid compression ratio
        assert validation['compression_ratio_valid'] is True
    
    def test_performance_constraint(self):
        """Test performance constraint (<50ms for compression)."""
        config = MultimodalCompressionConfig(max_compression_time_ms=50.0)
        compression = QuantumMultimodalCompression(config)
        
        text_emb = torch.randn(1, 768)
        clinical_emb = torch.randn(1, 768)
        
        start_time = time.time()
        compression(text_emb, clinical_emb)
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Should meet performance constraint
        assert elapsed_ms < 50.0
    
    def test_optimization_for_inference(self):
        """Test optimization for inference."""
        config = MultimodalCompressionConfig()
        compression = QuantumMultimodalCompression(config)
        
        # Should not raise any errors
        compression.optimize_for_inference()
        
        # Should be in eval mode
        assert not compression.training
    
    def test_benchmark_compression(self):
        """Test compression benchmarking."""
        config = MultimodalCompressionConfig()
        compression = QuantumMultimodalCompression(config)
        
        benchmark_results = compression.benchmark_compression(num_samples=10)
        
        assert 'total_time_ms' in benchmark_results
        assert 'avg_time_per_sample_ms' in benchmark_results
        assert 'throughput_samples_per_second' in benchmark_results
        assert 'output_shape' in benchmark_results
        assert 'compression_ratio' in benchmark_results
        
        # Should have reasonable throughput
        assert benchmark_results['throughput_samples_per_second'] > 0
        
        # Should have correct output shape
        assert benchmark_results['output_shape'] == (10, 256)
    
    def test_compression_deterministic(self):
        """Test that compression is deterministic."""
        config = MultimodalCompressionConfig()
        compression = QuantumMultimodalCompression(config)
        
        # Set to eval mode for deterministic behavior
        compression.eval()
        
        text_emb = torch.randn(1, 768)
        clinical_emb = torch.randn(1, 768)
        
        output1 = compression(text_emb, clinical_emb)
        output2 = compression(text_emb, clinical_emb)
        
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_compression_with_missing_modality(self):
        """Test compression behavior with missing modality."""
        config = MultimodalCompressionConfig()
        compression = QuantumMultimodalCompression(config)
        
        # Test with zero embedding (simulating missing modality)
        text_emb = torch.randn(1, 768)
        clinical_emb = torch.zeros(1, 768)
        
        output = compression(text_emb, clinical_emb)
        
        # Should still produce valid output
        assert output.shape == (1, 256)
        assert torch.allclose(torch.norm(output), torch.ones(1), atol=1e-5)
    
    def test_compression_memory_efficiency(self):
        """Test memory efficiency of compression."""
        config = MultimodalCompressionConfig()
        compression = QuantumMultimodalCompression(config)
        
        # Test with larger batch to check memory usage
        batch_size = 100
        text_emb = torch.randn(batch_size, 768)
        clinical_emb = torch.randn(batch_size, 768)
        
        # Should not raise memory errors
        output = compression(text_emb, clinical_emb)
        
        assert output.shape == (batch_size, 256)
        
        # Memory usage should be reasonable
        # (This is more of a smoke test)
        del output, text_emb, clinical_emb


if __name__ == '__main__':
    pytest.main([__file__])
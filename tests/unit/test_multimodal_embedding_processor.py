"""
Unit tests for MultimodalEmbeddingProcessor.

Tests the core functionality of multimodal embedding processing
while ensuring PRD performance constraints are met.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

from quantum_rerank.core.multimodal_embedding_processor import (
    MultimodalEmbeddingProcessor,
    ClinicalDataProcessor,
    MultimodalEmbeddingResult
)
from quantum_rerank.config.multimodal_config import MultimodalMedicalConfig


class TestClinicalDataProcessor:
    """Test ClinicalDataProcessor functionality."""
    
    def test_clinical_data_processor_initialization(self):
        """Test clinical data processor initialization."""
        processor = ClinicalDataProcessor()
        
        assert processor.config is not None
        assert processor.processing_stats['total_processed'] == 0
        assert processor.processing_stats['avg_processing_time_ms'] == 0.0
        assert processor.processing_stats['error_count'] == 0
    
    def test_process_clinical_data_dict(self):
        """Test processing clinical data from dictionary."""
        processor = ClinicalDataProcessor()
        
        clinical_data = {
            'demographics': {'age': 45, 'gender': 'M'},
            'vitals': {'bp': '140/90', 'hr': 72},
            'lab_results': {'glucose': 120, 'cholesterol': 180}
        }
        
        result = processor.process_clinical_data(clinical_data)
        
        assert result['success'] is True
        assert 'processed_text' in result
        assert 'entities' in result
        assert 'demographics age: 45' in result['processed_text']
        assert 'vitals bp: 140/90' in result['processed_text']
        assert result['processing_time_ms'] > 0
    
    def test_process_clinical_data_string(self):
        """Test processing clinical data from string."""
        processor = ClinicalDataProcessor()
        
        clinical_text = "Patient has HTN and DM with BP 140/90"
        
        result = processor.process_clinical_data(clinical_text)
        
        assert result['success'] is True
        assert 'processed_text' in result
        assert 'HTN' in result['processed_text']
        assert 'DM' in result['processed_text']
    
    def test_encode_clinical_data(self):
        """Test clinical data encoding to embeddings."""
        processor = ClinicalDataProcessor()
        
        clinical_data = {
            'symptoms': ['chest pain', 'shortness of breath'],
            'vitals': {'bp': '140/90', 'hr': 85}
        }
        
        embedding = processor.encode_clinical_data(clinical_data)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)  # Default BERT dimension
        assert np.allclose(np.linalg.norm(embedding), 1.0, atol=1e-5)  # Normalized


class TestMultimodalEmbeddingProcessor:
    """Test MultimodalEmbeddingProcessor functionality."""
    
    def test_multimodal_processor_initialization(self):
        """Test multimodal processor initialization."""
        config = MultimodalMedicalConfig()
        processor = MultimodalEmbeddingProcessor(config)
        
        assert processor.multimodal_config is not None
        assert processor.clinical_processor is not None
        assert processor.multimodal_stats['total_multimodal_processed'] == 0
        assert processor.multimodal_stats['avg_multimodal_time_ms'] == 0.0
    
    def test_encode_multimodal_text_only(self):
        """Test multimodal encoding with text only."""
        processor = MultimodalEmbeddingProcessor()
        
        data = {
            'text': 'patient presents with chest pain'
        }
        
        result = processor.encode_multimodal(data)
        
        assert isinstance(result, MultimodalEmbeddingResult)
        assert result.text_embedding is not None
        assert result.text_embedding.shape == (768,)
        assert result.clinical_embedding is None
        assert result.fused_embedding is not None
        assert result.fused_embedding.shape == (256,)  # Target quantum dimension
        assert 'text' in result.modalities_used
        assert result.processing_time_ms > 0
        assert result.error_message is None
    
    def test_encode_multimodal_clinical_only(self):
        """Test multimodal encoding with clinical data only."""
        processor = MultimodalEmbeddingProcessor()
        
        data = {
            'clinical_data': {
                'age': 45,
                'symptoms': ['chest pain', 'dyspnea'],
                'vitals': {'bp': '140/90', 'hr': 80}
            }
        }
        
        result = processor.encode_multimodal(data)
        
        assert isinstance(result, MultimodalEmbeddingResult)
        assert result.text_embedding is None
        assert result.clinical_embedding is not None
        assert result.clinical_embedding.shape == (768,)
        assert result.fused_embedding is not None
        assert 'clinical' in result.modalities_used
        assert result.processing_time_ms > 0
        assert result.error_message is None
    
    def test_encode_multimodal_both_modalities(self):
        """Test multimodal encoding with both text and clinical data."""
        processor = MultimodalEmbeddingProcessor()
        
        data = {
            'text': 'patient presents with chest pain',
            'clinical_data': {
                'age': 45,
                'symptoms': ['chest pain', 'shortness of breath'],
                'vitals': {'bp': '140/90', 'hr': 85}
            }
        }
        
        result = processor.encode_multimodal(data)
        
        assert isinstance(result, MultimodalEmbeddingResult)
        assert result.text_embedding is not None
        assert result.clinical_embedding is not None
        assert result.fused_embedding is not None
        assert result.fused_embedding.shape == (256,)
        assert 'text' in result.modalities_used
        assert 'clinical' in result.modalities_used
        assert len(result.modalities_used) == 2
        assert result.processing_time_ms > 0
        assert result.error_message is None
    
    def test_encode_multimodal_empty_data(self):
        """Test multimodal encoding with empty data."""
        processor = MultimodalEmbeddingProcessor()
        
        data = {}
        
        result = processor.encode_multimodal(data)
        
        assert isinstance(result, MultimodalEmbeddingResult)
        assert result.text_embedding is None
        assert result.clinical_embedding is None
        assert result.fused_embedding is not None  # Should still create zero embedding
        assert len(result.modalities_used) == 0
        assert result.error_message is not None
    
    def test_encode_multimodal_performance_constraint(self):
        """Test that multimodal encoding meets performance constraints."""
        config = MultimodalMedicalConfig(max_latency_ms=100.0)
        processor = MultimodalEmbeddingProcessor(config)
        
        data = {
            'text': 'patient presents with chest pain and shortness of breath',
            'clinical_data': {
                'age': 45,
                'symptoms': ['chest pain', 'dyspnea', 'fatigue'],
                'vitals': {'bp': '140/90', 'hr': 85, 'rr': 18, 'temp': 98.6}
            }
        }
        
        start_time = time.time()
        result = processor.encode_multimodal(data)
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Should meet PRD constraint
        assert elapsed_ms < 100.0
        assert result.processing_time_ms < 100.0
        assert result.error_message is None
    
    def test_encode_multimodal_batch(self):
        """Test batch multimodal encoding."""
        processor = MultimodalEmbeddingProcessor()
        
        data_batch = [
            {
                'text': 'patient with diabetes',
                'clinical_data': {'age': 55, 'diagnosis': 'T2DM'}
            },
            {
                'text': 'patient with hypertension',
                'clinical_data': {'age': 60, 'diagnosis': 'HTN'}
            },
            {
                'text': 'patient with heart failure',
                'clinical_data': {'age': 70, 'diagnosis': 'CHF'}
            }
        ]
        
        results = processor.encode_multimodal_batch(data_batch)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, MultimodalEmbeddingResult)
            assert result.text_embedding is not None
            assert result.clinical_embedding is not None
            assert result.fused_embedding is not None
            assert 'text' in result.modalities_used
            assert 'clinical' in result.modalities_used
    
    def test_caching_functionality(self):
        """Test embedding caching functionality."""
        config = MultimodalMedicalConfig(enable_embedding_cache=True, cache_size=10)
        processor = MultimodalEmbeddingProcessor(config)
        
        data = {
            'text': 'patient presents with chest pain',
            'clinical_data': {'age': 45, 'symptoms': ['chest pain']}
        }
        
        # First call - should be cache miss
        result1 = processor.encode_multimodal(data)
        assert processor.multimodal_stats['cache_misses'] == 1
        assert processor.multimodal_stats['cache_hits'] == 0
        
        # Second call with same data - should be cache hit
        result2 = processor.encode_multimodal(data)
        assert processor.multimodal_stats['cache_hits'] == 1
        
        # Results should be identical
        assert np.array_equal(result1.fused_embedding, result2.fused_embedding)
    
    def test_get_multimodal_stats(self):
        """Test multimodal statistics retrieval."""
        processor = MultimodalEmbeddingProcessor()
        
        # Process some data
        data = {
            'text': 'patient with chest pain',
            'clinical_data': {'age': 45}
        }
        processor.encode_multimodal(data)
        
        stats = processor.get_multimodal_stats()
        
        assert 'total_multimodal_processed' in stats
        assert 'avg_multimodal_time_ms' in stats
        assert 'full_multimodal_count' in stats
        assert 'clinical_processor_stats' in stats
        assert stats['total_multimodal_processed'] == 1
        assert stats['full_multimodal_count'] == 1
    
    def test_validate_performance(self):
        """Test performance validation."""
        processor = MultimodalEmbeddingProcessor()
        
        # Process some data first
        data = {
            'text': 'patient with chest pain',
            'clinical_data': {'age': 45}
        }
        processor.encode_multimodal(data)
        
        validation_results = processor.validate_performance()
        
        assert 'latency_under_100ms' in validation_results
        assert 'quantum_compression_working' in validation_results
        assert 'clinical_processing_working' in validation_results
        assert isinstance(validation_results['latency_under_100ms'], bool)
        assert isinstance(validation_results['quantum_compression_working'], bool)
    
    def test_memory_optimization(self):
        """Test memory optimization functionality."""
        processor = MultimodalEmbeddingProcessor()
        
        # Should not raise any errors
        processor.optimize_memory()
        
        # Cache should be cleared
        if processor.embedding_cache:
            assert len(processor.embedding_cache) == 0
    
    def test_clear_cache(self):
        """Test cache clearing functionality."""
        config = MultimodalMedicalConfig(enable_embedding_cache=True)
        processor = MultimodalEmbeddingProcessor(config)
        
        # Add some data to cache
        data = {'text': 'test', 'clinical_data': {'age': 45}}
        processor.encode_multimodal(data)
        
        # Clear cache
        processor.clear_cache()
        
        assert len(processor.embedding_cache) == 0
    
    @patch('quantum_rerank.core.multimodal_embedding_processor.ClinicalDataProcessor')
    def test_clinical_processor_error_handling(self, mock_clinical_processor):
        """Test error handling when clinical processor fails."""
        # Mock clinical processor to raise an error
        mock_processor = Mock()
        mock_processor.encode_clinical_data.side_effect = Exception("Clinical processing failed")
        mock_clinical_processor.return_value = mock_processor
        
        processor = MultimodalEmbeddingProcessor()
        processor.clinical_processor = mock_processor
        
        data = {
            'text': 'patient with chest pain',
            'clinical_data': {'age': 45}
        }
        
        result = processor.encode_multimodal(data)
        
        # Should handle error gracefully
        assert result.text_embedding is not None
        assert result.clinical_embedding is not None  # Should be zeros
        assert result.fused_embedding is not None
        assert result.error_message is None  # Should not propagate error
    
    def test_fallback_fusion_when_quantum_compression_fails(self):
        """Test fallback fusion when quantum compression fails."""
        processor = MultimodalEmbeddingProcessor()
        
        # Mock quantum compressor to fail
        with patch.object(processor, 'quantum_compressor', None):
            data = {
                'text': 'patient with chest pain',
                'clinical_data': {'age': 45}
            }
            
            result = processor.encode_multimodal(data)
            
            # Should still produce valid result using fallback
            assert result.fused_embedding is not None
            assert result.fused_embedding.shape == (256,)
            assert result.error_message is None


if __name__ == '__main__':
    pytest.main([__file__])
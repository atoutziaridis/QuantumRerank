"""
Integration tests for Multimodal Quantum Similarity Engine.

Tests the complete multimodal similarity pipeline including medical domain processing,
multimodal embedding fusion, and quantum similarity computation.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

from quantum_rerank.core.quantum_similarity_engine import (
    QuantumSimilarityEngine,
    SimilarityEngineConfig,
    SimilarityMethod
)
from quantum_rerank.config.multimodal_config import MultimodalMedicalConfig


class TestMultimodalSimilarityEngineIntegration:
    """Integration tests for multimodal similarity engine."""
    
    def test_multimodal_similarity_engine_initialization(self):
        """Test initialization of multimodal similarity engine."""
        multimodal_config = MultimodalMedicalConfig(
            enable_embedding_cache=True,
            max_latency_ms=100.0
        )
        
        config = SimilarityEngineConfig(
            enable_multimodal=True,
            multimodal_config=multimodal_config,
            similarity_method=SimilarityMethod.MULTIMODAL_QUANTUM
        )
        
        engine = QuantumSimilarityEngine(config)
        
        assert engine.config.enable_multimodal is True
        assert engine.multimodal_processor is not None
        assert engine.medical_processor is not None
        assert engine.config.similarity_method == SimilarityMethod.MULTIMODAL_QUANTUM
    
    def test_multimodal_text_only_similarity(self):
        """Test multimodal similarity with text-only data."""
        config = SimilarityEngineConfig(
            enable_multimodal=True,
            multimodal_config=MultimodalMedicalConfig()
        )
        
        engine = QuantumSimilarityEngine(config)
        
        query_data = {'text': 'Patient presents with chest pain'}
        candidate_data = {'text': 'Patient has cardiac symptoms'}
        
        similarity, metadata = engine.compute_multimodal_similarity(query_data, candidate_data)
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
        assert metadata['multimodal_processing'] is True
        assert 'text' in metadata['modalities_used']
        assert metadata['total_computation_time_ms'] > 0
    
    def test_multimodal_clinical_only_similarity(self):
        """Test multimodal similarity with clinical data only."""
        config = SimilarityEngineConfig(
            enable_multimodal=True,
            multimodal_config=MultimodalMedicalConfig()
        )
        
        engine = QuantumSimilarityEngine(config)
        
        query_data = {
            'clinical_data': {
                'age': 45,
                'symptoms': ['chest pain', 'shortness of breath'],
                'vitals': {'bp': '140/90', 'hr': 85}
            }
        }
        
        candidate_data = {
            'clinical_data': {
                'age': 50,
                'symptoms': ['chest pain', 'fatigue'],
                'vitals': {'bp': '130/80', 'hr': 72}
            }
        }
        
        similarity, metadata = engine.compute_multimodal_similarity(query_data, candidate_data)
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
        assert metadata['multimodal_processing'] is True
        assert 'clinical' in metadata['modalities_used']
        assert metadata['total_computation_time_ms'] > 0
    
    def test_multimodal_full_similarity(self):
        """Test multimodal similarity with both text and clinical data."""
        config = SimilarityEngineConfig(
            enable_multimodal=True,
            multimodal_config=MultimodalMedicalConfig()
        )
        
        engine = QuantumSimilarityEngine(config)
        
        query_data = {
            'text': 'Patient presents with chest pain and shortness of breath',
            'clinical_data': {
                'age': 45,
                'symptoms': ['chest pain', 'dyspnea'],
                'vitals': {'bp': '140/90', 'hr': 85}
            }
        }
        
        candidate_data = {
            'text': 'Patient has cardiac symptoms with breathing difficulty',
            'clinical_data': {
                'age': 50,
                'symptoms': ['chest pain', 'shortness of breath'],
                'vitals': {'bp': '130/80', 'hr': 72}
            }
        }
        
        similarity, metadata = engine.compute_multimodal_similarity(query_data, candidate_data)
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
        assert metadata['multimodal_processing'] is True
        assert 'text' in metadata['modalities_used']
        assert 'clinical' in metadata['modalities_used']
        assert len(metadata['modalities_used']) == 2
        assert metadata['total_computation_time_ms'] > 0
    
    def test_multimodal_medical_abbreviation_expansion(self):
        """Test that medical abbreviations are properly expanded."""
        config = SimilarityEngineConfig(
            enable_multimodal=True,
            multimodal_config=MultimodalMedicalConfig()
        )
        
        engine = QuantumSimilarityEngine(config)
        
        query_data = {'text': 'Patient has HTN and DM with SOB'}
        candidate_data = {'text': 'Patient has hypertension and diabetes with shortness of breath'}
        
        similarity, metadata = engine.compute_multimodal_similarity(query_data, candidate_data)
        
        # Should have high similarity due to abbreviation expansion
        assert similarity > 0.5
        assert metadata['multimodal_processing'] is True
        assert 'medical_domain' in metadata
    
    def test_multimodal_performance_constraint(self):
        """Test that multimodal processing meets performance constraints."""
        config = SimilarityEngineConfig(
            enable_multimodal=True,
            multimodal_config=MultimodalMedicalConfig(max_latency_ms=100.0)
        )
        
        engine = QuantumSimilarityEngine(config)
        
        query_data = {
            'text': 'Complex medical case with multiple symptoms and conditions',
            'clinical_data': {
                'age': 65,
                'symptoms': ['chest pain', 'dyspnea', 'fatigue', 'weakness'],
                'vitals': {'bp': '160/100', 'hr': 95, 'rr': 22, 'temp': 99.2},
                'medications': ['lisinopril', 'metformin', 'atorvastatin']
            }
        }
        
        candidate_data = {
            'text': 'Patient with cardiovascular and endocrine comorbidities',
            'clinical_data': {
                'age': 70,
                'symptoms': ['chest pain', 'shortness of breath'],
                'vitals': {'bp': '140/90', 'hr': 88, 'rr': 20, 'temp': 98.6},
                'medications': ['amlodipine', 'insulin', 'simvastatin']
            }
        }
        
        start_time = time.time()
        similarity, metadata = engine.compute_multimodal_similarity(query_data, candidate_data)
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Should meet performance constraint
        assert elapsed_ms < 100.0
        assert metadata['total_computation_time_ms'] < 100.0
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
    
    def test_multimodal_batch_processing(self):
        """Test multimodal batch processing through text-based interface."""
        config = SimilarityEngineConfig(
            enable_multimodal=True,
            multimodal_config=MultimodalMedicalConfig()
        )
        
        engine = QuantumSimilarityEngine(config)
        
        # Use text-based batch processing (multimodal processing happens internally)
        query = "Patient with chest pain and diabetes"
        candidates = [
            "Patient has cardiac symptoms with DM",
            "Patient with HTN and MI history",
            "Patient presents with SOB and fatigue"
        ]
        
        results = engine.compute_similarities_batch(
            query, candidates, 
            method=SimilarityMethod.MULTIMODAL_QUANTUM
        )
        
        assert len(results) == 3
        for candidate, similarity, metadata in results:
            assert isinstance(similarity, float)
            assert 0.0 <= similarity <= 1.0
            assert metadata['method'] == 'multimodal_quantum'
            assert 'batch_index' in metadata
    
    def test_multimodal_compression_integration(self):
        """Test that multimodal compression is properly integrated."""
        config = SimilarityEngineConfig(
            enable_multimodal=True,
            multimodal_config=MultimodalMedicalConfig(
                target_quantum_dim=256,
                compression_ratio=6.0
            )
        )
        
        engine = QuantumSimilarityEngine(config)
        
        query_data = {
            'text': 'Patient with complex medical history',
            'clinical_data': {'age': 45, 'symptoms': ['chest pain']}
        }
        
        candidate_data = {
            'text': 'Patient with cardiac symptoms',
            'clinical_data': {'age': 50, 'symptoms': ['chest pain']}
        }
        
        similarity, metadata = engine.compute_multimodal_similarity(query_data, candidate_data)
        
        # Should have compression ratio information
        assert 'compression_ratio' in metadata
        assert metadata['compression_ratio'] > 0
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
    
    def test_multimodal_medical_domain_classification(self):
        """Test medical domain classification integration."""
        config = SimilarityEngineConfig(
            enable_multimodal=True,
            multimodal_config=MultimodalMedicalConfig()
        )
        
        engine = QuantumSimilarityEngine(config)
        
        # Cardiovascular case
        cardio_query = {
            'text': 'Patient with myocardial infarction and chest pain',
            'clinical_data': {'symptoms': ['chest pain', 'dyspnea']}
        }
        
        cardio_candidate = {
            'text': 'Patient with heart attack symptoms',
            'clinical_data': {'symptoms': ['chest pain']}
        }
        
        similarity, metadata = engine.compute_multimodal_similarity(cardio_query, cardio_candidate)
        
        assert 'medical_domain' in metadata
        assert metadata['medical_domain'] in ['cardiovascular', 'cardiology', 'general']
        assert 'query_domain_confidence' in metadata
        assert metadata['query_domain_confidence'] >= 0.0
    
    def test_multimodal_error_handling(self):
        """Test error handling in multimodal processing."""
        config = SimilarityEngineConfig(
            enable_multimodal=True,
            multimodal_config=MultimodalMedicalConfig()
        )
        
        engine = QuantumSimilarityEngine(config)
        
        # Test with empty data
        empty_query = {}
        empty_candidate = {}
        
        similarity, metadata = engine.compute_multimodal_similarity(empty_query, empty_candidate)
        
        # Should handle gracefully
        assert isinstance(similarity, float)
        assert metadata['multimodal_processing'] is True
        assert len(metadata['modalities_used']) == 0
    
    def test_multimodal_fallback_to_classical(self):
        """Test fallback to classical similarity when multimodal fails."""
        config = SimilarityEngineConfig(
            enable_multimodal=True,
            multimodal_config=MultimodalMedicalConfig()
        )
        
        engine = QuantumSimilarityEngine(config)
        
        # Mock multimodal processor to return None for fused embeddings
        with patch.object(engine.multimodal_processor, 'encode_multimodal') as mock_encode:
            mock_result = Mock()
            mock_result.fused_embedding = None
            mock_result.text_embedding = np.random.randn(768)
            mock_result.modalities_used = ['text']
            mock_result.processing_time_ms = 10.0
            mock_encode.return_value = mock_result
            
            query_data = {'text': 'test query'}
            candidate_data = {'text': 'test candidate'}
            
            similarity, metadata = engine.compute_multimodal_similarity(query_data, candidate_data)
            
            # Should still return valid similarity using fallback
            assert isinstance(similarity, float)
            assert 0.0 <= similarity <= 1.0
            assert metadata['multimodal_processing'] is True
    
    def test_multimodal_engine_disabled(self):
        """Test that multimodal methods fail when multimodal is disabled."""
        config = SimilarityEngineConfig(
            enable_multimodal=False  # Disabled
        )
        
        engine = QuantumSimilarityEngine(config)
        
        query_data = {'text': 'test query'}
        candidate_data = {'text': 'test candidate'}
        
        with pytest.raises(ValueError, match="Multimodal processing not enabled"):
            engine.compute_multimodal_similarity(query_data, candidate_data)
    
    def test_multimodal_benchmark_integration(self):
        """Test multimodal method in benchmark suite."""
        config = SimilarityEngineConfig(
            enable_multimodal=True,
            multimodal_config=MultimodalMedicalConfig()
        )
        
        engine = QuantumSimilarityEngine(config)
        
        test_texts = [
            "Patient with chest pain and HTN",
            "Patient has cardiac symptoms with DM",
            "Patient presents with SOB and fatigue",
            "Patient with MI and heart failure"
        ]
        
        benchmark_results = engine.benchmark_similarity_methods(test_texts)
        
        # Should include multimodal quantum method
        assert 'multimodal_quantum' in benchmark_results
        
        multimodal_results = benchmark_results['multimodal_quantum']
        assert 'avg_pairwise_time_ms' in multimodal_results
        assert 'avg_similarity' in multimodal_results
        assert 'meets_prd_target' in multimodal_results
        assert multimodal_results['avg_pairwise_time_ms'] > 0
    
    def test_multimodal_caching_integration(self):
        """Test that multimodal results are properly cached."""
        config = SimilarityEngineConfig(
            enable_multimodal=True,
            enable_caching=True,
            multimodal_config=MultimodalMedicalConfig()
        )
        
        engine = QuantumSimilarityEngine(config)
        
        # Use text-based interface which should use caching
        text1 = "Patient with chest pain and diabetes"
        text2 = "Patient has cardiac symptoms with DM"
        
        # First call - should be cache miss
        similarity1, metadata1 = engine.compute_similarity(
            text1, text2, method=SimilarityMethod.MULTIMODAL_QUANTUM
        )
        
        assert metadata1['cache_hit'] is False
        assert engine.performance_stats['cache_misses'] >= 1
        
        # Second call - should be cache hit
        similarity2, metadata2 = engine.compute_similarity(
            text1, text2, method=SimilarityMethod.MULTIMODAL_QUANTUM
        )
        
        assert metadata2['cache_hit'] is True
        assert engine.performance_stats['cache_hits'] >= 1
        assert similarity1 == similarity2
    
    def test_multimodal_performance_monitoring(self):
        """Test performance monitoring for multimodal processing."""
        config = SimilarityEngineConfig(
            enable_multimodal=True,
            performance_monitoring=True,
            multimodal_config=MultimodalMedicalConfig()
        )
        
        engine = QuantumSimilarityEngine(config)
        
        # Process some multimodal data
        query_data = {
            'text': 'Patient with chest pain',
            'clinical_data': {'age': 45, 'symptoms': ['chest pain']}
        }
        
        candidate_data = {
            'text': 'Patient with cardiac symptoms',
            'clinical_data': {'age': 50, 'symptoms': ['chest pain']}
        }
        
        # Multiple calls to build statistics
        for _ in range(5):
            engine.compute_multimodal_similarity(query_data, candidate_data)
        
        # Check performance report
        report = engine.get_performance_report()
        
        assert 'total_comparisons' in report
        assert 'avg_computation_time_ms' in report
        assert 'meets_prd_latency_target' in report
        assert report['total_comparisons'] >= 5
        assert report['avg_computation_time_ms'] > 0


if __name__ == '__main__':
    pytest.main([__file__])
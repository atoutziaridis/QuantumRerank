"""
Unit tests for MultimodalQuantumSimilarityEngine.

Tests the quantum multimodal similarity engine implementation
for QMMR-03 task requirements.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

from quantum_rerank.core.multimodal_quantum_similarity_engine import MultimodalQuantumSimilarityEngine
from quantum_rerank.config.settings import SimilarityEngineConfig


class TestMultimodalQuantumSimilarityEngine:
    """Test cases for MultimodalQuantumSimilarityEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = SimilarityEngineConfig(n_qubits=3, shots=1024)
        self.engine = MultimodalQuantumSimilarityEngine(self.config)
        
        # Sample multimodal data
        self.sample_query = {
            'text': 'Patient presents with chest pain and shortness of breath',
            'clinical_data': {
                'age': 65,
                'gender': 'M',
                'vitals': {'BP': '140/90', 'HR': 95, 'RR': 18},
                'symptoms': ['chest pain', 'dyspnea'],
                'medications': ['aspirin', 'metoprolol']
            }
        }
        
        self.sample_candidate = {
            'text': 'Elderly male with cardiac symptoms requiring evaluation',
            'clinical_data': {
                'age': 68,
                'gender': 'M',
                'vitals': {'BP': '135/85', 'HR': 88, 'RR': 16},
                'symptoms': ['chest discomfort', 'fatigue'],
                'medications': ['aspirin', 'lisinopril']
            }
        }
    
    def test_initialization(self):
        """Test proper initialization of multimodal engine."""
        assert self.engine is not None
        assert hasattr(self.engine, 'embedding_processor')
        assert hasattr(self.engine, 'swap_test')
        assert hasattr(self.engine, 'complexity_engine')
        assert hasattr(self.engine, 'medical_processor')
        assert self.engine.multimodal_stats['total_computations'] == 0
    
    def test_compute_multimodal_similarity_basic(self):
        """Test basic multimodal similarity computation."""
        with patch.object(self.engine.complexity_engine, 'assess_complexity') as mock_complexity, \
             patch.object(self.engine.medical_processor, 'process_medical_query') as mock_medical, \
             patch.object(self.engine.embedding_processor, 'encode_multimodal') as mock_embed, \
             patch.object(self.engine.swap_test, 'compute_multimodal_fidelity') as mock_fidelity:
            
            # Mock returns
            mock_complexity.return_value = Mock(overall_complexity=Mock(overall_complexity=0.5))
            mock_medical.side_effect = lambda x: x  # Return input unchanged
            mock_embed.return_value = Mock(
                modalities_used=['text', 'clinical'],
                processing_time_ms=25.0
            )
            mock_fidelity.return_value = (0.85, {
                'entanglement_measure': 0.7,
                'cross_modal_fidelities': {'text_clinical': 0.8},
                'uncertainty_quantification': {'quantum_uncertainty': 0.1},
                'modalities_used': ['text', 'clinical'],
                'quantum_circuit_depth': 8
            })
            
            fidelity, metadata = self.engine.compute_multimodal_similarity(
                self.sample_query, self.sample_candidate
            )
            
            # Verify results
            assert 0 <= fidelity <= 1
            assert metadata['success'] is True
            assert 'quantum_advantage_indicators' in metadata
            assert 'total_computation_time_ms' in metadata
            assert metadata['multimodal_processing'] is True
    
    def test_compute_multimodal_similarity_performance_constraint(self):
        """Test that similarity computation meets <100ms constraint."""
        # Mock fast responses
        with patch.object(self.engine.complexity_engine, 'assess_complexity') as mock_complexity, \
             patch.object(self.engine.medical_processor, 'process_medical_query') as mock_medical, \
             patch.object(self.engine.embedding_processor, 'encode_multimodal') as mock_embed, \
             patch.object(self.engine.swap_test, 'compute_multimodal_fidelity') as mock_fidelity:
            
            mock_complexity.return_value = Mock(overall_complexity=Mock(overall_complexity=0.3))
            mock_medical.side_effect = lambda x: x
            mock_embed.return_value = Mock(
                modalities_used=['text', 'clinical'],
                processing_time_ms=15.0
            )
            mock_fidelity.return_value = (0.75, {
                'entanglement_measure': 0.6,
                'cross_modal_fidelities': {},
                'uncertainty_quantification': {},
                'modalities_used': ['text', 'clinical']
            })
            
            start_time = time.time()
            fidelity, metadata = self.engine.compute_multimodal_similarity(
                self.sample_query, self.sample_candidate
            )
            elapsed = (time.time() - start_time) * 1000
            
            # Verify performance constraint
            assert elapsed < 100  # PRD requirement
            assert metadata['total_computation_time_ms'] < 100
    
    def test_batch_compute_multimodal_similarity(self):
        """Test batch processing of multimodal similarity."""
        candidates = [self.sample_candidate, self.sample_candidate.copy()]
        
        with patch.object(self.engine.complexity_engine, 'assess_complexity') as mock_complexity, \
             patch.object(self.engine.medical_processor, 'process_medical_query') as mock_medical, \
             patch.object(self.engine.embedding_processor, 'encode_multimodal') as mock_embed, \
             patch.object(self.engine.swap_test, 'batch_compute_multimodal_fidelity') as mock_batch:
            
            mock_complexity.return_value = Mock(
                candidate_complexities=[
                    Mock(overall_complexity=0.4),
                    Mock(overall_complexity=0.5)
                ]
            )
            mock_medical.side_effect = lambda x: x
            mock_embed.return_value = Mock(
                modalities_used=['text', 'clinical'],
                processing_time_ms=20.0
            )
            mock_batch.return_value = [
                (0.8, {'entanglement_measure': 0.7}),
                (0.75, {'entanglement_measure': 0.6})
            ]
            
            results = self.engine.batch_compute_multimodal_similarity(
                self.sample_query, candidates
            )
            
            # Verify batch results
            assert len(results) == 2
            for i, (fidelity, metadata) in enumerate(results):
                assert 0 <= fidelity <= 1
                assert metadata['batch_index'] == i
                assert metadata['batch_processing'] is True
                assert 'batch_total_time_ms' in metadata
    
    def test_batch_performance_constraint(self):
        """Test that batch processing meets <500ms constraint."""
        # Create larger batch
        candidates = [self.sample_candidate.copy() for _ in range(10)]
        
        with patch.object(self.engine.complexity_engine, 'assess_complexity') as mock_complexity, \
             patch.object(self.engine.medical_processor, 'process_medical_query') as mock_medical, \
             patch.object(self.engine.embedding_processor, 'encode_multimodal') as mock_embed, \
             patch.object(self.engine.swap_test, 'batch_compute_multimodal_fidelity') as mock_batch:
            
            mock_complexity.return_value = Mock(candidate_complexities=[
                Mock(overall_complexity=0.4) for _ in range(10)
            ])
            mock_medical.side_effect = lambda x: x
            mock_embed.return_value = Mock(
                modalities_used=['text', 'clinical'],
                processing_time_ms=10.0
            )
            mock_batch.return_value = [
                (0.7 + i*0.01, {'entanglement_measure': 0.5}) for i in range(10)
            ]
            
            start_time = time.time()
            results = self.engine.batch_compute_multimodal_similarity(
                self.sample_query, candidates
            )
            elapsed = (time.time() - start_time) * 1000
            
            # Verify performance constraint
            assert elapsed < 500  # PRD requirement
            assert all(metadata['batch_total_time_ms'] < 500 
                      for _, metadata in results)
    
    def test_quantum_advantage_assessment(self):
        """Test quantum advantage assessment functionality."""
        fidelity = 0.85
        metadata = {
            'entanglement_measure': 0.8,
            'cross_modal_fidelities': {'text_clinical': 0.9, 'text_image': 0.7},
            'uncertainty_quantification': {
                'quantum_uncertainty': 0.1,
                'confidence_intervals': {'ci_95': {'width': 0.05}}
            },
            'modalities_used': ['text', 'clinical', 'image'],
            'quantum_circuit_depth': 12
        }
        
        advantage_indicators = self.engine._assess_quantum_advantage(fidelity, metadata)
        
        # Verify advantage indicators
        assert 'entanglement_advantage' in advantage_indicators
        assert 'cross_modal_advantage' in advantage_indicators
        assert 'uncertainty_advantage' in advantage_indicators
        assert 'overall_quantum_advantage' in advantage_indicators
        
        # Check ranges
        for key, value in advantage_indicators.items():
            assert 0 <= value <= 1
    
    def test_multimodal_stats_update(self):
        """Test multimodal statistics tracking."""
        initial_count = self.engine.multimodal_stats['total_computations']
        
        metadata = {
            'total_computation_time_ms': 50.0,
            'modalities_used': ['text', 'clinical']
        }
        quantum_advantage = {
            'overall_quantum_advantage': 0.7,
            'cross_modal_advantage': 0.8,
            'uncertainty_advantage': 0.6
        }
        
        self.engine._update_multimodal_stats(metadata, quantum_advantage)
        
        # Verify stats update
        assert self.engine.multimodal_stats['total_computations'] == initial_count + 1
        assert self.engine.multimodal_stats['avg_computation_time_ms'] > 0
        assert self.engine.multimodal_stats['quantum_advantage_cases'] > 0
        assert len(self.engine.multimodal_stats['cross_modal_correlations']) > 0
    
    def test_performance_report(self):
        """Test multimodal performance report generation."""
        # Add some mock statistics
        self.engine.multimodal_stats.update({
            'total_computations': 10,
            'avg_computation_time_ms': 75.0,
            'quantum_advantage_cases': 7,
            'cross_modal_correlations': [0.8, 0.7, 0.9],
            'uncertainty_metrics': [0.9, 0.8, 0.95]
        })
        
        report = self.engine.get_multimodal_performance_report()
        
        # Verify report structure
        assert 'base_performance' in report
        assert 'multimodal_stats' in report
        assert 'quantum_advantage_rate' in report
        assert 'performance_analysis' in report
        
        # Verify calculations
        assert report['quantum_advantage_rate'] == 0.7
        assert 'meets_latency_target' in report['performance_analysis']
    
    def test_optimization(self):
        """Test multimodal engine optimization."""
        # Add some data to cache first
        self.engine.multimodal_stats['total_computations'] = 5
        
        self.engine.optimize_for_multimodal_performance()
        
        # Verify optimization reset stats
        assert self.engine.multimodal_stats['total_computations'] == 0
        assert self.engine.multimodal_stats['avg_computation_time_ms'] == 0.0
    
    def test_validate_quantum_advantages(self):
        """Test quantum advantage validation on test data."""
        test_queries = [self.sample_query]
        test_candidates = [self.sample_candidate]
        
        with patch.object(self.engine, 'compute_multimodal_similarity') as mock_compute:
            mock_compute.return_value = (0.8, {
                'quantum_advantage_indicators': {
                    'overall_quantum_advantage': 0.75,
                    'cross_modal_advantage': 0.8,
                    'uncertainty_advantage': 0.7
                },
                'total_computation_time_ms': 85.0
            })
            
            validation_results = self.engine.validate_quantum_advantages(
                test_queries, test_candidates
            )
            
            # Verify validation results
            assert validation_results['total_tests'] == 1
            assert len(validation_results['quantum_advantages']) == 1
            assert 'summary' in validation_results
            assert 'avg_quantum_advantage' in validation_results['summary']
    
    def test_error_handling(self):
        """Test error handling in multimodal similarity computation."""
        with patch.object(self.engine.complexity_engine, 'assess_complexity') as mock_complexity:
            mock_complexity.side_effect = Exception("Complexity assessment failed")
            
            fidelity, metadata = self.engine.compute_multimodal_similarity(
                self.sample_query, self.sample_candidate
            )
            
            # Verify error handling
            assert fidelity == 0.0
            assert metadata['success'] is False
            assert 'error' in metadata
            assert metadata['multimodal_processing'] is True
    
    def test_missing_modalities(self):
        """Test handling of missing modalities."""
        incomplete_query = {'text': 'Some text'}  # Missing clinical data
        
        with patch.object(self.engine.embedding_processor, 'encode_multimodal') as mock_embed:
            mock_embed.return_value = Mock(
                text_embedding=np.random.rand(768),
                clinical_embedding=None,
                modalities_used=['text'],
                processing_time_ms=20.0
            )
            
            modalities = self.engine._prepare_modality_dict(mock_embed.return_value)
            
            # Verify handling of missing modalities
            assert 'text' in modalities
            assert 'clinical' not in modalities
    
    def test_modality_dict_preparation(self):
        """Test preparation of modality dictionary from embedding results."""
        # Mock embedding result
        embedding_result = Mock()
        embedding_result.text_embedding = np.random.rand(256)
        embedding_result.clinical_embedding = np.random.rand(256)
        
        modalities = self.engine._prepare_modality_dict(embedding_result)
        
        # Verify modality dictionary
        assert 'text' in modalities
        assert 'clinical' in modalities
        assert isinstance(modalities['text'], np.ndarray)
        assert isinstance(modalities['clinical'], np.ndarray)


if __name__ == '__main__':
    pytest.main([__file__])
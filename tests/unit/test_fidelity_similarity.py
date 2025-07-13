"""
Unit tests for fidelity similarity engine.

Tests the FidelitySimilarityEngine class and related functionality for
Task 04: SWAP Test Implementation - Integration with Embedding Bridge.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock

from qiskit import QuantumCircuit

from quantum_rerank.core.fidelity_similarity import FidelitySimilarityEngine
from quantum_rerank.core.quantum_embedding_bridge import BridgeResult
from quantum_rerank.core.swap_test import QuantumSWAPTest


class TestFidelitySimilarityEngine:
    """Test FidelitySimilarityEngine class."""
    
    @pytest.fixture
    def mock_swap_test(self):
        """Mock QuantumSWAPTest for testing."""
        mock_swap = Mock(spec=QuantumSWAPTest)
        mock_swap.compute_fidelity.return_value = (
            0.8, 
            {'success': True, 'execution_time_ms': 50.0}
        )
        mock_swap.batch_compute_fidelity.return_value = [
            (0.9, {'success': True, 'execution_time_ms': 45.0}),
            (0.7, {'success': True, 'execution_time_ms': 55.0}),
            (0.6, {'success': True, 'execution_time_ms': 50.0})
        ]
        return mock_swap
    
    @pytest.fixture  
    def mock_embedding_bridge(self):
        """Mock QuantumEmbeddingBridge for testing."""
        mock_bridge = Mock()
        
        # Mock successful circuit creation
        mock_circuit = Mock(spec=QuantumCircuit)
        mock_result = BridgeResult(
            success=True,
            text="test text",
            circuit=mock_circuit,
            metadata={'total_processing_time_ms': 30.0}
        )
        mock_bridge.text_to_quantum_circuit.return_value = mock_result
        mock_bridge.batch_texts_to_circuits.return_value = [mock_result] * 3
        
        return mock_bridge
    
    @pytest.fixture
    def engine_with_mocks(self, mock_swap_test, mock_embedding_bridge):
        """FidelitySimilarityEngine with mocked dependencies."""
        with patch('quantum_rerank.core.fidelity_similarity.QuantumSWAPTest', return_value=mock_swap_test):
            with patch('quantum_rerank.core.fidelity_similarity.QuantumEmbeddingBridge', return_value=mock_embedding_bridge):
                engine = FidelitySimilarityEngine(n_qubits=3)
                engine.swap_test = mock_swap_test
                engine.embedding_bridge = mock_embedding_bridge
                return engine, mock_swap_test, mock_embedding_bridge
    
    def test_init_default(self):
        """Test engine initialization with default parameters."""
        with patch('quantum_rerank.core.fidelity_similarity.QuantumSWAPTest'):
            with patch('quantum_rerank.core.fidelity_similarity.QuantumEmbeddingBridge'):
                engine = FidelitySimilarityEngine()
                assert engine.n_qubits == 4  # Default value
    
    def test_init_custom_qubits(self):
        """Test engine initialization with custom qubit count."""
        with patch('quantum_rerank.core.fidelity_similarity.QuantumSWAPTest'):
            with patch('quantum_rerank.core.fidelity_similarity.QuantumEmbeddingBridge'):
                engine = FidelitySimilarityEngine(n_qubits=3)
                assert engine.n_qubits == 3
    
    def test_compute_text_similarity_success(self, engine_with_mocks):
        """Test successful text similarity computation."""
        engine, mock_swap, mock_bridge = engine_with_mocks
        
        text1 = "quantum computing"
        text2 = "machine learning"
        
        similarity, metadata = engine.compute_text_similarity(text1, text2)
        
        # Verify similarity computation
        assert similarity == 0.8  # From mock SWAP test
        assert metadata['success'] is True
        assert metadata['text1'] == text1
        assert metadata['text2'] == text2
        assert metadata['encoding_method'] == 'amplitude'
        assert metadata['similarity_score'] == 0.8
        assert 'total_time_ms' in metadata
        assert 'prd_compliant' in metadata
        
        # Verify method calls
        assert mock_bridge.text_to_quantum_circuit.call_count == 2
        mock_swap.compute_fidelity.assert_called_once()
    
    def test_compute_text_similarity_custom_encoding(self, engine_with_mocks):
        """Test text similarity with custom encoding method."""
        engine, mock_swap, mock_bridge = engine_with_mocks
        
        similarity, metadata = engine.compute_text_similarity(
            "text1", "text2", encoding_method='angle'
        )
        
        assert metadata['encoding_method'] == 'angle'
        
        # Verify encoding method was passed to bridge
        calls = mock_bridge.text_to_quantum_circuit.call_args_list
        assert len(calls) == 2
        for call in calls:
            assert call[1]['encoding_method'] == 'angle'
    
    def test_compute_text_similarity_circuit_failure(self, engine_with_mocks):
        """Test text similarity computation with circuit creation failure."""
        engine, mock_swap, mock_bridge = engine_with_mocks
        
        # Mock circuit creation failure
        failed_result = BridgeResult(
            success=False,
            text="failed text",
            error="Circuit creation failed"
        )
        mock_bridge.text_to_quantum_circuit.return_value = failed_result
        
        similarity, metadata = engine.compute_text_similarity("text1", "text2")
        
        assert similarity == 0.0
        assert metadata['success'] is False
        assert "Circuit creation failed" in metadata['error']
        
        # SWAP test should not be called
        mock_swap.compute_fidelity.assert_not_called()
    
    def test_compute_text_similarity_exception(self, engine_with_mocks):
        """Test text similarity computation with exception."""
        engine, mock_swap, mock_bridge = engine_with_mocks
        
        # Mock bridge to raise exception
        mock_bridge.text_to_quantum_circuit.side_effect = Exception("Bridge failed")
        
        similarity, metadata = engine.compute_text_similarity("text1", "text2")
        
        assert similarity == 0.0
        assert metadata['success'] is False
        assert "Text similarity computation failed" in metadata['error']
    
    def test_compute_query_similarities_success(self, engine_with_mocks):
        """Test successful query similarity computation."""
        engine, mock_swap, mock_bridge = engine_with_mocks
        
        query = "quantum computing"
        candidates = ["machine learning", "artificial intelligence", "data science"]
        
        results = engine.compute_query_similarities(query, candidates)
        
        assert len(results) == 3
        
        # Check individual results
        for i, (candidate, similarity, metadata) in enumerate(results):
            assert candidate == candidates[i]
            assert similarity > 0  # From mock SWAP test
            assert metadata['success'] is True
            assert metadata['candidate_index'] == i
            assert 'query_metadata' in metadata
            assert 'candidate_metadata' in metadata
            assert 'fidelity_metadata' in metadata
            assert 'batch_metadata' in metadata
        
        # Verify method calls
        mock_bridge.text_to_quantum_circuit.assert_called_once()  # Query circuit
        mock_bridge.batch_texts_to_circuits.assert_called_once()
        mock_swap.batch_compute_fidelity.assert_called_once()
    
    def test_compute_query_similarities_query_failure(self, engine_with_mocks):
        """Test query similarities with query circuit creation failure."""
        engine, mock_swap, mock_bridge = engine_with_mocks
        
        # Mock query circuit creation failure
        failed_query_result = BridgeResult(
            success=False,
            text="failed query",
            error="Query circuit failed"
        )
        mock_bridge.text_to_quantum_circuit.return_value = failed_query_result
        
        candidates = ["candidate1", "candidate2"]
        results = engine.compute_query_similarities("failed query", candidates)
        
        assert len(results) == 2
        for candidate, similarity, metadata in results:
            assert similarity == 0.0
            assert metadata['success'] is False
            assert "Query circuit creation failed" in metadata['error']
    
    def test_compute_query_similarities_partial_failures(self, engine_with_mocks):
        """Test query similarities with some candidate failures."""
        engine, mock_swap, mock_bridge = engine_with_mocks
        
        # Mock mixed success/failure for candidates
        successful_result = BridgeResult(
            success=True,
            text="success",
            circuit=Mock(spec=QuantumCircuit),
            metadata={}
        )
        failed_result = BridgeResult(
            success=False,
            text="failed",
            error="Failed to create circuit"
        )
        
        mock_bridge.batch_texts_to_circuits.return_value = [
            successful_result, failed_result, successful_result
        ]
        
        # Only successful circuits should be processed by SWAP test
        mock_swap.batch_compute_fidelity.return_value = [
            (0.8, {}), (0.6, {})  # Only 2 results for 2 successful circuits
        ]
        
        candidates = ["success1", "failed", "success2"]
        results = engine.compute_query_similarities("query", candidates)
        
        assert len(results) == 3
        assert results[0][1] == 0.8  # successful
        assert results[1][1] == 0.0  # failed
        assert results[2][1] == 0.6  # successful
        
        assert results[0][2]['success'] is True
        assert results[1][2]['success'] is False
        assert results[2][2]['success'] is True
    
    def test_compute_query_similarities_exception(self, engine_with_mocks):
        """Test query similarities computation with exception."""
        engine, mock_swap, mock_bridge = engine_with_mocks
        
        # Mock bridge to raise exception
        mock_bridge.text_to_quantum_circuit.side_effect = Exception("Bridge failed")
        
        candidates = ["candidate1", "candidate2"]
        results = engine.compute_query_similarities("query", candidates)
        
        assert len(results) == 2
        for candidate, similarity, metadata in results:
            assert similarity == 0.0
            assert metadata['success'] is False
            assert "Batch similarity computation failed" in metadata['error']
    
    def test_rank_candidates_by_similarity(self, engine_with_mocks):
        """Test candidate ranking by similarity."""
        engine, mock_swap, mock_bridge = engine_with_mocks
        
        # Mock similarity results with different scores
        with patch.object(engine, 'compute_query_similarities') as mock_compute:
            mock_compute.return_value = [
                ("candidate1", 0.6, {'success': True}),
                ("candidate2", 0.9, {'success': True}),
                ("candidate3", 0.3, {'success': True})
            ]
            
            candidates = ["candidate1", "candidate2", "candidate3"]
            ranked_results = engine.rank_candidates_by_similarity("query", candidates)
            
            # Should be sorted by similarity (descending)
            assert len(ranked_results) == 3
            assert ranked_results[0][0] == "candidate2"  # 0.9 similarity
            assert ranked_results[0][1] == 0.9
            assert ranked_results[1][0] == "candidate1"  # 0.6 similarity
            assert ranked_results[2][0] == "candidate3"  # 0.3 similarity
            
            # Check ranking metadata
            for i, (candidate, similarity, metadata) in enumerate(ranked_results):
                assert metadata['rank'] == i + 1
                assert 'ranking_metadata' in metadata
    
    def test_rank_candidates_by_similarity_top_k(self, engine_with_mocks):
        """Test candidate ranking with top-K filtering."""
        engine, mock_swap, mock_bridge = engine_with_mocks
        
        with patch.object(engine, 'compute_query_similarities') as mock_compute:
            mock_compute.return_value = [
                ("candidate1", 0.6, {'success': True}),
                ("candidate2", 0.9, {'success': True}),
                ("candidate3", 0.3, {'success': True}),
                ("candidate4", 0.8, {'success': True})
            ]
            
            candidates = ["candidate1", "candidate2", "candidate3", "candidate4"]
            ranked_results = engine.rank_candidates_by_similarity(
                "query", candidates, top_k=2
            )
            
            # Should return only top 2 results
            assert len(ranked_results) == 2
            assert ranked_results[0][0] == "candidate2"  # 0.9 similarity
            assert ranked_results[1][0] == "candidate4"  # 0.8 similarity
    
    def test_benchmark_similarity_engine_default_texts(self, engine_with_mocks):
        """Test similarity engine benchmarking with default texts."""
        engine, mock_swap, mock_bridge = engine_with_mocks
        
        # Mock successful operations for all encoding methods
        with patch.object(engine, 'compute_text_similarity') as mock_text_sim:
            with patch.object(engine, 'compute_query_similarities') as mock_query_sim:
                
                mock_text_sim.return_value = (0.8, {'success': True, 'total_time_ms': 60.0})
                mock_query_sim.return_value = [
                    ("candidate1", 0.7, {'success': True}),
                    ("candidate2", 0.6, {'success': True}),
                    ("candidate3", 0.5, {'success': True})
                ]
                
                results = engine.benchmark_similarity_engine()
        
        # Check structure
        assert 'encoding_methods' in results
        assert 'batch_sizes' in results
        assert 'summary' in results
        
        # Check encoding methods were tested
        for method in ['amplitude', 'angle', 'dense_angle']:
            assert method in results['encoding_methods']
            method_result = results['encoding_methods'][method]
            assert 'pairwise_similarity' in method_result
            assert 'batch_results_count' in method_result
            assert 'success' in method_result
        
        # Check batch sizes were tested
        for batch_size in [2, 4]:
            if batch_size in results['batch_sizes']:
                batch_result = results['batch_sizes'][batch_size]
                assert 'candidates_processed' in batch_result
                assert 'total_time_ms' in batch_result
    
    def test_benchmark_similarity_engine_custom_texts(self, engine_with_mocks):
        """Test similarity engine benchmarking with custom texts."""
        engine, mock_swap, mock_bridge = engine_with_mocks
        
        custom_texts = ["custom1", "custom2"]
        
        with patch.object(engine, 'compute_text_similarity') as mock_text_sim:
            with patch.object(engine, 'compute_query_similarities') as mock_query_sim:
                
                mock_text_sim.return_value = (0.8, {'success': True, 'total_time_ms': 60.0})
                mock_query_sim.return_value = [("custom2", 0.7, {'success': True})]
                
                results = engine.benchmark_similarity_engine(custom_texts)
        
        # Should have called methods with custom texts
        assert mock_text_sim.call_count == 3  # One per encoding method
        assert mock_query_sim.call_count == 6  # 3 encoding methods × 2 batch sizes
    
    def test_benchmark_similarity_engine_with_failures(self, engine_with_mocks):
        """Test similarity engine benchmarking with some failures."""
        engine, mock_swap, mock_bridge = engine_with_mocks
        
        with patch.object(engine, 'compute_text_similarity') as mock_text_sim:
            with patch.object(engine, 'compute_query_similarities') as mock_query_sim:
                
                # Mock failures for some encoding methods
                mock_text_sim.side_effect = [
                    (0.8, {'success': True, 'total_time_ms': 60.0}),  # amplitude success
                    Exception("Angle encoding failed"),  # angle failure  
                    (0.7, {'success': True, 'total_time_ms': 70.0})   # dense_angle success
                ]
                
                mock_query_sim.return_value = [("candidate", 0.7, {'success': True})]
                
                results = engine.benchmark_similarity_engine()
        
        # Check that failures were recorded
        assert results['encoding_methods']['amplitude']['success'] is True
        assert results['encoding_methods']['angle']['success'] is False
        assert 'error' in results['encoding_methods']['angle']
        assert results['encoding_methods']['dense_angle']['success'] is True
        
        # Summary should reflect mixed results
        summary = results['summary']
        assert summary['successful_encoding_methods'] == 2  # amplitude and dense_angle


class TestFidelitySimilarityEngineIntegration:
    """Integration tests for fidelity similarity engine."""
    
    def test_end_to_end_similarity_computation(self):
        """Test end-to-end similarity computation with mocked components."""
        # Create engine with mocked dependencies but test the full flow
        with patch('quantum_rerank.core.fidelity_similarity.QuantumSWAPTest') as mock_swap_class:
            with patch('quantum_rerank.core.fidelity_similarity.QuantumEmbeddingBridge') as mock_bridge_class:
                
                # Setup mocks
                mock_swap = Mock()
                mock_bridge = Mock()
                mock_swap_class.return_value = mock_swap
                mock_bridge_class.return_value = mock_bridge
                
                # Mock successful circuit creation
                mock_circuit = Mock(spec=QuantumCircuit)
                successful_result = BridgeResult(
                    success=True,
                    text="test",
                    circuit=mock_circuit,
                    metadata={'total_processing_time_ms': 40.0}
                )
                mock_bridge.text_to_quantum_circuit.return_value = successful_result
                
                # Mock successful fidelity computation
                mock_swap.compute_fidelity.return_value = (
                    0.85, 
                    {'success': True, 'execution_time_ms': 45.0}
                )
                
                # Create engine and test
                engine = FidelitySimilarityEngine(n_qubits=3)
                
                similarity, metadata = engine.compute_text_similarity(
                    "quantum computing", "machine learning"
                )
                
                # Verify end-to-end flow
                assert similarity == 0.85
                assert metadata['success'] is True
                assert metadata['encoding_method'] == 'amplitude'
                
                # Verify all components were called
                mock_bridge.text_to_quantum_circuit.assert_called()
                mock_swap.compute_fidelity.assert_called_once()
    
    def test_performance_under_prd_targets(self):
        """Test that performance stays within PRD targets."""
        with patch('quantum_rerank.core.fidelity_similarity.QuantumSWAPTest') as mock_swap_class:
            with patch('quantum_rerank.core.fidelity_similarity.QuantumEmbeddingBridge') as mock_bridge_class:
                
                # Setup fast mocks (under PRD targets)
                mock_swap = Mock()
                mock_bridge = Mock()
                mock_swap_class.return_value = mock_swap
                mock_bridge_class.return_value = mock_bridge
                
                # Mock fast circuit creation (< 50ms each)
                mock_circuit = Mock(spec=QuantumCircuit)
                fast_result = BridgeResult(
                    success=True,
                    text="test",
                    circuit=mock_circuit,
                    metadata={'total_processing_time_ms': 30.0}
                )
                mock_bridge.text_to_quantum_circuit.return_value = fast_result
                
                # Mock fast fidelity computation (< 50ms)
                mock_swap.compute_fidelity.return_value = (
                    0.75, 
                    {'success': True, 'execution_time_ms': 40.0}
                )
                
                engine = FidelitySimilarityEngine(n_qubits=3)
                
                # Time the actual computation
                start_time = time.time()
                similarity, metadata = engine.compute_text_similarity("text1", "text2")
                total_time = (time.time() - start_time) * 1000
                
                # Verify PRD compliance
                assert total_time < 200  # Should be much faster with mocks
                assert metadata['prd_compliant']['similarity_under_100ms']
                assert metadata['success'] is True
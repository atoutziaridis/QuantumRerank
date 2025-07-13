"""
Unit tests for quantum embedding bridge functionality.

Tests the QuantumEmbeddingBridge class and related functionality for
Task 03: SentenceTransformer Integration and Embedding Processing.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantum_rerank.core.quantum_embedding_bridge import (
    QuantumEmbeddingBridge, BridgeResult, SimilarityResult
)
from quantum_rerank.core.embeddings import EmbeddingProcessor, EmbeddingConfig
from quantum_rerank.core.quantum_circuits import BasicQuantumCircuits, CircuitResult


class TestBridgeResult:
    """Test BridgeResult dataclass."""
    
    def test_bridge_result_creation(self):
        """Test BridgeResult creation with various parameters."""
        # Minimal result
        result = BridgeResult(success=True, text="test")
        assert result.success is True
        assert result.text == "test"
        assert result.circuit is None
        assert result.error is None
        
        # Full result
        mock_circuit = Mock(spec=QuantumCircuit)
        mock_statevector = Mock(spec=Statevector)
        metadata = {"test": "data"}
        
        result = BridgeResult(
            success=True,
            text="test",
            circuit=mock_circuit,
            statevector=mock_statevector,
            metadata=metadata
        )
        
        assert result.circuit is mock_circuit
        assert result.statevector is mock_statevector
        assert result.metadata == metadata


class TestSimilarityResult:
    """Test SimilarityResult dataclass."""
    
    def test_similarity_result_creation(self):
        """Test SimilarityResult creation."""
        result = SimilarityResult(
            classical_cosine=0.8,
            quantum_fidelity=0.75,
            quantum_amplitude_overlap=0.9,
            computation_time_ms=50.0
        )
        
        assert result.classical_cosine == 0.8
        assert result.quantum_fidelity == 0.75
        assert result.quantum_amplitude_overlap == 0.9
        assert result.computation_time_ms == 50.0


class TestQuantumEmbeddingBridge:
    """Test QuantumEmbeddingBridge class."""
    
    @pytest.fixture
    def mock_embedding_processor(self):
        """Mock EmbeddingProcessor for testing."""
        mock_processor = Mock(spec=EmbeddingProcessor)
        mock_processor.encode_single_text.return_value = np.random.rand(768)
        mock_processor.encode_texts.return_value = np.random.rand(2, 768)
        mock_processor.preprocess_for_quantum.return_value = (
            np.random.rand(1, 16),
            {'processing_applied': ['normalization']}
        )
        mock_processor.compute_classical_similarity.return_value = 0.8
        mock_processor.compute_fidelity_similarity.return_value = 0.75
        return mock_processor
    
    @pytest.fixture
    def mock_quantum_circuits(self):
        """Mock BasicQuantumCircuits for testing."""
        mock_circuits = Mock(spec=BasicQuantumCircuits)
        mock_circuits.n_qubits = 4
        
        # Create a simple mock circuit
        mock_circuit = Mock(spec=QuantumCircuit)
        mock_circuit.name = "test_circuit"
        mock_circuit.depth.return_value = 5
        mock_circuit.size.return_value = 10
        
        mock_circuits.amplitude_encode_embedding.return_value = mock_circuit
        mock_circuits.angle_encode_embedding.return_value = mock_circuit
        mock_circuits.dense_angle_encoding.return_value = mock_circuit
        
        # Mock simulation result
        mock_statevector = Mock(spec=Statevector)
        mock_statevector.data = np.random.rand(16) + 1j * np.random.rand(16)
        
        mock_sim_result = CircuitResult(
            success=True,
            statevector=mock_statevector,
            metadata={'simulation_time_ms': 10.0}
        )
        mock_circuits.simulate_circuit.return_value = mock_sim_result
        
        return mock_circuits
    
    @pytest.fixture
    def bridge_with_mocks(self, mock_embedding_processor, mock_quantum_circuits):
        """QuantumEmbeddingBridge with mocked dependencies."""
        with patch('quantum_rerank.core.quantum_embedding_bridge.EmbeddingProcessor', return_value=mock_embedding_processor):
            with patch('quantum_rerank.core.quantum_embedding_bridge.BasicQuantumCircuits', return_value=mock_quantum_circuits):
                bridge = QuantumEmbeddingBridge(n_qubits=4)
                bridge.embedding_processor = mock_embedding_processor
                bridge.quantum_circuits = mock_quantum_circuits
                return bridge, mock_embedding_processor, mock_quantum_circuits
    
    def test_init_default(self):
        """Test bridge initialization with default parameters."""
        with patch('quantum_rerank.core.quantum_embedding_bridge.EmbeddingProcessor'):
            with patch('quantum_rerank.core.quantum_embedding_bridge.BasicQuantumCircuits') as mock_qc:
                mock_qc.return_value.n_qubits = 4
                bridge = QuantumEmbeddingBridge()
                assert bridge.n_qubits == 4
    
    def test_init_custom_qubits(self):
        """Test bridge initialization with custom qubit count."""
        with patch('quantum_rerank.core.quantum_embedding_bridge.EmbeddingProcessor'):
            with patch('quantum_rerank.core.quantum_embedding_bridge.BasicQuantumCircuits') as mock_qc:
                mock_qc.return_value.n_qubits = 3
                bridge = QuantumEmbeddingBridge(n_qubits=3)
                assert bridge.n_qubits == 3
    
    def test_init_qubit_mismatch_warning(self):
        """Test warning when quantum circuit qubits don't match bridge qubits."""
        with patch('quantum_rerank.core.quantum_embedding_bridge.EmbeddingProcessor'):
            with patch('quantum_rerank.core.quantum_embedding_bridge.BasicQuantumCircuits') as mock_qc:
                mock_qc.return_value.n_qubits = 2  # Different from requested 4
                
                with patch('quantum_rerank.core.quantum_embedding_bridge.logger') as mock_logger:
                    bridge = QuantumEmbeddingBridge(n_qubits=4)
                    assert bridge.n_qubits == 2  # Should use quantum circuit value
                    mock_logger.warning.assert_called_once()
    
    def test_text_to_quantum_circuit_amplitude_success(self, bridge_with_mocks):
        """Test successful text to quantum circuit conversion with amplitude encoding."""
        bridge, mock_processor, mock_circuits = bridge_with_mocks
        
        test_text = "quantum computing test"
        result = bridge.text_to_quantum_circuit(test_text, encoding_method='amplitude')
        
        assert result.success is True
        assert result.text == test_text
        assert result.circuit is not None
        assert result.statevector is not None
        assert result.metadata is not None
        
        # Verify method calls
        mock_processor.encode_single_text.assert_called_once_with(test_text)
        mock_processor.preprocess_for_quantum.assert_called_once()
        mock_circuits.amplitude_encode_embedding.assert_called_once()
        mock_circuits.simulate_circuit.assert_called_once()
    
    def test_text_to_quantum_circuit_angle_encoding(self, bridge_with_mocks):
        """Test text to quantum circuit conversion with angle encoding."""
        bridge, mock_processor, mock_circuits = bridge_with_mocks
        
        test_text = "test text"
        result = bridge.text_to_quantum_circuit(test_text, encoding_method='angle')
        
        assert result.success is True
        mock_circuits.angle_encode_embedding.assert_called_once()
    
    def test_text_to_quantum_circuit_dense_angle_encoding(self, bridge_with_mocks):
        """Test text to quantum circuit conversion with dense angle encoding."""
        bridge, mock_processor, mock_circuits = bridge_with_mocks
        
        test_text = "test text"
        result = bridge.text_to_quantum_circuit(test_text, encoding_method='dense_angle')
        
        assert result.success is True
        mock_circuits.dense_angle_encoding.assert_called_once()
    
    def test_text_to_quantum_circuit_invalid_encoding(self, bridge_with_mocks):
        """Test text to quantum circuit conversion with invalid encoding method."""
        bridge, mock_processor, mock_circuits = bridge_with_mocks
        
        test_text = "test text"
        result = bridge.text_to_quantum_circuit(test_text, encoding_method='invalid')
        
        assert result.success is False
        assert "Unknown encoding method" in result.error
    
    def test_text_to_quantum_circuit_simulation_failure(self, bridge_with_mocks):
        """Test handling of simulation failure."""
        bridge, mock_processor, mock_circuits = bridge_with_mocks
        
        # Mock simulation failure
        mock_circuits.simulate_circuit.return_value = CircuitResult(
            success=False,
            error="Simulation failed"
        )
        
        test_text = "test text"
        result = bridge.text_to_quantum_circuit(test_text)
        
        assert result.success is False
        assert result.error == "Simulation failed"
    
    def test_text_to_quantum_circuit_embedding_failure(self, bridge_with_mocks):
        """Test handling of embedding failure."""
        bridge, mock_processor, mock_circuits = bridge_with_mocks
        
        # Mock embedding failure
        mock_processor.encode_single_text.side_effect = Exception("Embedding failed")
        
        test_text = "test text"
        result = bridge.text_to_quantum_circuit(test_text)
        
        assert result.success is False
        assert "Text to quantum circuit conversion failed" in result.error
    
    def test_text_to_quantum_circuit_metadata(self, bridge_with_mocks):
        """Test metadata generation in text to circuit conversion."""
        bridge, mock_processor, mock_circuits = bridge_with_mocks
        
        test_text = "test"
        result = bridge.text_to_quantum_circuit(test_text)
        
        assert result.success is True
        metadata = result.metadata
        
        # Check required metadata fields
        assert 'text_length' in metadata
        assert 'original_embedding_dim' in metadata
        assert 'processed_embedding_dim' in metadata
        assert 'encoding_method' in metadata
        assert 'quantum_circuit_depth' in metadata
        assert 'quantum_circuit_size' in metadata
        assert 'preprocessing_metadata' in metadata
        assert 'simulation_success' in metadata
        assert 'total_processing_time_ms' in metadata
        assert 'prd_compliant' in metadata
        
        assert metadata['text_length'] == len(test_text)
        assert metadata['encoding_method'] == 'amplitude'
    
    def test_batch_texts_to_circuits_success(self, bridge_with_mocks):
        """Test successful batch text to circuit conversion."""
        bridge, mock_processor, mock_circuits = bridge_with_mocks
        
        test_texts = ["text1", "text2", "text3"]
        mock_processor.encode_texts.return_value = np.random.rand(3, 768)
        mock_processor.preprocess_for_quantum.return_value = (
            np.random.rand(3, 16),
            {'processing_applied': ['normalization']}
        )
        
        results = bridge.batch_texts_to_circuits(test_texts)
        
        assert len(results) == 3
        assert all(result.success for result in results)
        assert all(result.text in test_texts for result in results)
        
        # Verify batch processing calls
        mock_processor.encode_texts.assert_called_once_with(test_texts)
        mock_processor.preprocess_for_quantum.assert_called_once()
        assert mock_circuits.amplitude_encode_embedding.call_count == 3
        assert mock_circuits.simulate_circuit.call_count == 3
    
    def test_batch_texts_to_circuits_partial_failure(self, bridge_with_mocks):
        """Test batch processing with some failures."""
        bridge, mock_processor, mock_circuits = bridge_with_mocks
        
        test_texts = ["text1", "text2"]
        mock_processor.encode_texts.return_value = np.random.rand(2, 768)
        mock_processor.preprocess_for_quantum.return_value = (
            np.random.rand(2, 16),
            {'processing_applied': ['normalization']}
        )
        
        # Mock one circuit creation failure
        mock_circuits.amplitude_encode_embedding.side_effect = [
            Mock(spec=QuantumCircuit),  # Success
            Exception("Circuit creation failed")  # Failure
        ]
        
        results = bridge.batch_texts_to_circuits(test_texts)
        
        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False
        assert "Failed to process text 1" in results[1].error
    
    def test_batch_texts_to_circuits_encoding_failure(self, bridge_with_mocks):
        """Test batch processing with encoding failure."""
        bridge, mock_processor, mock_circuits = bridge_with_mocks
        
        test_texts = ["text1", "text2"]
        mock_processor.encode_texts.side_effect = Exception("Batch encoding failed")
        
        results = bridge.batch_texts_to_circuits(test_texts)
        
        assert len(results) == 2
        assert all(not result.success for result in results)
        assert all("Batch text to circuit conversion failed" in result.error for result in results)
    
    def test_compute_quantum_similarity_success(self, bridge_with_mocks):
        """Test successful quantum similarity computation."""
        bridge, mock_processor, mock_circuits = bridge_with_mocks
        
        text1, text2 = "quantum text", "classical text"
        
        # Mock two different statevectors
        state1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        state2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
        
        mock_statevector1 = Mock(spec=Statevector)
        mock_statevector1.data = state1
        mock_statevector2 = Mock(spec=Statevector)
        mock_statevector2.data = state2
        
        # Mock text to circuit results
        with patch.object(bridge, 'text_to_quantum_circuit') as mock_text_to_circuit:
            mock_text_to_circuit.side_effect = [
                BridgeResult(success=True, text=text1, statevector=mock_statevector1),
                BridgeResult(success=True, text=text2, statevector=mock_statevector2)
            ]
            
            result = bridge.compute_quantum_similarity(text1, text2)
        
        assert not np.isnan(result.classical_cosine)
        assert not np.isnan(result.quantum_fidelity)
        assert result.quantum_amplitude_overlap is not None
        assert result.computation_time_ms > 0
        assert result.metadata is not None
        
        # Verify similarity computations were called
        mock_processor.compute_classical_similarity.assert_called_once()
        mock_processor.compute_fidelity_similarity.assert_called_once()
    
    def test_compute_quantum_similarity_circuit_failure(self, bridge_with_mocks):
        """Test quantum similarity computation with circuit failures."""
        bridge, mock_processor, mock_circuits = bridge_with_mocks
        
        text1, text2 = "text1", "text2"
        
        # Mock circuit failures
        with patch.object(bridge, 'text_to_quantum_circuit') as mock_text_to_circuit:
            mock_text_to_circuit.side_effect = [
                BridgeResult(success=False, text=text1, error="Circuit failed"),
                BridgeResult(success=False, text=text2, error="Circuit failed")
            ]
            
            result = bridge.compute_quantum_similarity(text1, text2)
        
        # Should still have classical similarities
        assert not np.isnan(result.classical_cosine)
        assert not np.isnan(result.quantum_fidelity)
        # But no quantum amplitude overlap
        assert result.quantum_amplitude_overlap is None
    
    def test_compute_quantum_similarity_exception(self, bridge_with_mocks):
        """Test quantum similarity computation with exception."""
        bridge, mock_processor, mock_circuits = bridge_with_mocks
        
        # Mock embedding failure
        mock_processor.encode_texts.side_effect = Exception("Encoding failed")
        
        result = bridge.compute_quantum_similarity("text1", "text2")
        
        assert np.isnan(result.classical_cosine)
        assert np.isnan(result.quantum_fidelity)
        assert result.quantum_amplitude_overlap is None
        assert 'error' in result.metadata
    
    def test_benchmark_bridge_performance_default_texts(self, bridge_with_mocks):
        """Test bridge performance benchmarking with default texts."""
        bridge, mock_processor, mock_circuits = bridge_with_mocks
        
        # Mock successful conversions and similarities
        with patch.object(bridge, 'text_to_quantum_circuit') as mock_convert:
            with patch.object(bridge, 'batch_texts_to_circuits') as mock_batch:
                with patch.object(bridge, 'compute_quantum_similarity') as mock_similarity:
                    
                    # Mock return values
                    mock_convert.return_value = BridgeResult(
                        success=True, text="test", 
                        metadata={'total_processing_time_ms': 50.0},
                        circuit=Mock(depth=lambda: 5)
                    )
                    mock_batch.return_value = [mock_convert.return_value] * 4
                    mock_similarity.return_value = SimilarityResult(
                        classical_cosine=0.8, quantum_fidelity=0.7,
                        computation_time_ms=25.0
                    )
                    
                    results = bridge.benchmark_bridge_performance()
        
        # Check structure
        assert 'amplitude' in results
        assert 'angle' in results  
        assert 'dense_angle' in results
        assert 'summary' in results
        
        # Check method-specific results
        for method in ['amplitude', 'angle', 'dense_angle']:
            method_result = results[method]
            assert 'avg_single_conversion_ms' in method_result
            assert 'batch_conversion_ms' in method_result
            assert 'avg_similarity_computation_ms' in method_result
            assert 'success_rate' in method_result
            assert 'prd_targets' in method_result
    
    def test_benchmark_bridge_performance_custom_texts(self, bridge_with_mocks):
        """Test bridge performance benchmarking with custom texts."""
        bridge, mock_processor, mock_circuits = bridge_with_mocks
        
        custom_texts = ["custom1", "custom2"]
        
        with patch.object(bridge, 'text_to_quantum_circuit') as mock_convert:
            with patch.object(bridge, 'batch_texts_to_circuits') as mock_batch:
                with patch.object(bridge, 'compute_quantum_similarity') as mock_similarity:
                    
                    mock_convert.return_value = BridgeResult(
                        success=True, text="test",
                        metadata={'total_processing_time_ms': 30.0},
                        circuit=Mock(depth=lambda: 3)
                    )
                    mock_batch.return_value = [mock_convert.return_value] * 2
                    mock_similarity.return_value = SimilarityResult(
                        classical_cosine=0.9, quantum_fidelity=0.85,
                        computation_time_ms=15.0
                    )
                    
                    results = bridge.benchmark_bridge_performance(custom_texts)
        
        # Verify custom texts were used
        assert mock_batch.call_count == 3  # Called for each encoding method
        # Check that all batch calls used the custom texts
        for call in mock_batch.call_args_list:
            assert call[0][0] == custom_texts
    
    def test_benchmark_bridge_performance_summary(self, bridge_with_mocks):
        """Test benchmark summary generation."""
        bridge, mock_processor, mock_circuits = bridge_with_mocks
        
        with patch.object(bridge, 'text_to_quantum_circuit') as mock_convert:
            with patch.object(bridge, 'batch_texts_to_circuits') as mock_batch:
                with patch.object(bridge, 'compute_quantum_similarity') as mock_similarity:
                    
                    mock_convert.return_value = BridgeResult(
                        success=True, text="test",
                        metadata={'total_processing_time_ms': 80.0},
                        circuit=Mock(depth=lambda: 10)
                    )
                    mock_batch.return_value = [mock_convert.return_value] * 4
                    mock_similarity.return_value = SimilarityResult(
                        classical_cosine=0.8, quantum_fidelity=0.7,
                        computation_time_ms=90.0  # Above PRD target
                    )
                    
                    results = bridge.benchmark_bridge_performance()
        
        summary = results['summary']
        assert 'overall_avg_similarity_ms' in summary
        assert 'overall_success_rate' in summary
        assert 'prd_compliance' in summary
        assert 'benchmark_timestamp' in summary
        
        # Check PRD compliance structure
        prd_compliance = summary['prd_compliance']
        assert 'similarity_target_met' in prd_compliance
        assert 'overall_success_high' in prd_compliance
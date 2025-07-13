"""
Unit tests for embedding processing functionality.

Tests the EmbeddingProcessor class and related functionality for
Task 03: SentenceTransformer Integration and Embedding Processing.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock

from quantum_rerank.core.embeddings import EmbeddingProcessor, EmbeddingConfig
from quantum_rerank.core.embedding_validators import EmbeddingValidator


class TestEmbeddingConfig:
    """Test EmbeddingConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EmbeddingConfig()
        
        assert config.model_name == 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
        assert config.embedding_dim == 768
        assert config.max_sequence_length == 512
        assert config.batch_size == 32
        assert config.device == 'auto'
        assert config.normalize_embeddings is True
        assert len(config.fallback_models) == 2
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = EmbeddingConfig(
            model_name='custom-model',
            embedding_dim=512,
            batch_size=16,
            device='cpu'
        )
        
        assert config.model_name == 'custom-model'
        assert config.embedding_dim == 512
        assert config.batch_size == 16
        assert config.device == 'cpu'


class TestEmbeddingProcessor:
    """Test EmbeddingProcessor class."""
    
    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock SentenceTransformer for testing."""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(2, 768)
        return mock_model
    
    @pytest.fixture
    def processor_with_mock(self, mock_sentence_transformer):
        """EmbeddingProcessor with mocked SentenceTransformer."""
        with patch('quantum_rerank.core.embeddings.SentenceTransformer', return_value=mock_sentence_transformer):
            processor = EmbeddingProcessor()
            processor.model = mock_sentence_transformer
            return processor
    
    def test_init_auto_device(self):
        """Test device auto-detection."""
        with patch('quantum_rerank.core.embeddings.SentenceTransformer') as mock_st:
            with patch('torch.cuda.is_available', return_value=False):
                processor = EmbeddingProcessor()
                assert processor.device == 'cpu'
            
            with patch('torch.cuda.is_available', return_value=True):
                processor = EmbeddingProcessor()
                assert processor.device == 'cuda'
    
    def test_init_custom_device(self):
        """Test custom device specification."""
        config = EmbeddingConfig(device='cpu')
        with patch('quantum_rerank.core.embeddings.SentenceTransformer'):
            processor = EmbeddingProcessor(config)
            assert processor.device == 'cpu'
    
    @patch('quantum_rerank.core.embeddings.SentenceTransformer')
    def test_model_loading_fallback(self, mock_st):
        """Test model loading with fallback."""
        # First model fails, second succeeds
        mock_st.side_effect = [Exception("Model not found"), Mock()]
        
        processor = EmbeddingProcessor()
        
        # Should have tried multiple models
        assert mock_st.call_count >= 2
    
    @patch('quantum_rerank.core.embeddings.SentenceTransformer')
    def test_model_loading_all_fail(self, mock_st):
        """Test behavior when all models fail to load."""
        mock_st.side_effect = Exception("No models available")
        
        with pytest.raises(RuntimeError, match="Failed to load any embedding model"):
            EmbeddingProcessor()
    
    def test_encode_texts_empty_input(self, processor_with_mock):
        """Test encoding empty text list."""
        result = processor_with_mock.encode_texts([])
        assert result.shape[0] == 0
    
    def test_encode_texts_success(self, processor_with_mock):
        """Test successful text encoding."""
        texts = ["test text 1", "test text 2"]
        expected_embeddings = np.random.rand(2, 768)
        processor_with_mock.model.encode.return_value = expected_embeddings
        
        result = processor_with_mock.encode_texts(texts)
        
        assert result.shape == (2, 768)
        processor_with_mock.model.encode.assert_called_once()
    
    def test_encode_texts_with_batch_size(self, processor_with_mock):
        """Test text encoding with custom batch size."""
        texts = ["text1", "text2", "text3"]
        processor_with_mock.model.encode.return_value = np.random.rand(3, 768)
        
        result = processor_with_mock.encode_texts(texts, batch_size=2)
        
        # Verify batch_size parameter was passed
        call_args = processor_with_mock.model.encode.call_args
        assert call_args[1]['batch_size'] == 2
    
    def test_encode_single_text(self, processor_with_mock):
        """Test single text encoding."""
        text = "single test text"
        expected_embedding = np.random.rand(1, 768)
        processor_with_mock.model.encode.return_value = expected_embedding
        
        result = processor_with_mock.encode_single_text(text)
        
        assert result.shape == (768,)
        processor_with_mock.model.encode.assert_called_once_with([text])
    
    def test_preprocess_for_quantum_truncation(self, processor_with_mock):
        """Test quantum preprocessing with truncation."""
        # Create embedding larger than quantum capacity
        large_embedding = np.random.rand(1, 32)  # 32 > 16 (2^4)
        
        processed, metadata = processor_with_mock.preprocess_for_quantum(large_embedding, n_qubits=4)
        
        assert processed.shape[1] == 16  # 2^4
        assert 'truncation' in metadata['processing_applied']
        assert np.allclose(np.linalg.norm(processed[0]), 1.0)
    
    def test_preprocess_for_quantum_padding(self, processor_with_mock):
        """Test quantum preprocessing with zero padding."""
        # Create embedding smaller than quantum capacity
        small_embedding = np.random.rand(1, 8)  # 8 < 16 (2^4)
        
        processed, metadata = processor_with_mock.preprocess_for_quantum(small_embedding, n_qubits=4)
        
        assert processed.shape[1] == 16  # 2^4
        assert 'zero_padding' in metadata['processing_applied']
        assert np.allclose(np.linalg.norm(processed[0]), 1.0)
    
    def test_preprocess_for_quantum_exact_size(self, processor_with_mock):
        """Test quantum preprocessing with exact size match."""
        # Create embedding exactly matching quantum capacity
        exact_embedding = np.random.rand(1, 16)  # Exactly 2^4
        
        processed, metadata = processor_with_mock.preprocess_for_quantum(exact_embedding, n_qubits=4)
        
        assert processed.shape[1] == 16
        assert 'normalization' in metadata['processing_applied']
        assert np.allclose(np.linalg.norm(processed[0]), 1.0)
    
    def test_preprocess_for_quantum_different_qubits(self, processor_with_mock):
        """Test quantum preprocessing with different qubit counts."""
        embedding = np.random.rand(1, 10)
        
        # Test with 2 qubits (capacity = 4)
        processed_2q, _ = processor_with_mock.preprocess_for_quantum(embedding, n_qubits=2)
        assert processed_2q.shape[1] == 4
        
        # Test with 3 qubits (capacity = 8)  
        processed_3q, _ = processor_with_mock.preprocess_for_quantum(embedding, n_qubits=3)
        assert processed_3q.shape[1] == 8
    
    def test_create_embedding_batches(self, processor_with_mock):
        """Test creation of embedding batches."""
        texts = ["text1", "text2", "text3", "text4", "text5"]
        processor_with_mock.model.encode.side_effect = [
            np.random.rand(2, 768),  # First batch
            np.random.rand(2, 768),  # Second batch  
            np.random.rand(1, 768)   # Third batch
        ]
        
        batches = processor_with_mock.create_embedding_batches(texts, batch_size=2)
        
        assert len(batches) == 3
        assert len(batches[0][0]) == 2  # First batch: 2 texts
        assert len(batches[1][0]) == 2  # Second batch: 2 texts
        assert len(batches[2][0]) == 1  # Third batch: 1 text
    
    def test_compute_classical_similarity(self, processor_with_mock):
        """Test classical cosine similarity computation."""
        # Create test embeddings
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([1.0, 0.0, 0.0])  # Identical
        emb3 = np.array([0.0, 1.0, 0.0])  # Orthogonal
        emb4 = np.array([-1.0, 0.0, 0.0])  # Opposite
        
        # Test identical embeddings
        sim_identical = processor_with_mock.compute_classical_similarity(emb1, emb2)
        assert np.isclose(sim_identical, 1.0)
        
        # Test orthogonal embeddings
        sim_orthogonal = processor_with_mock.compute_classical_similarity(emb1, emb3)
        assert np.isclose(sim_orthogonal, 0.5)  # Scaled to [0,1]
        
        # Test opposite embeddings
        sim_opposite = processor_with_mock.compute_classical_similarity(emb1, emb4)
        assert np.isclose(sim_opposite, 0.0)  # Scaled to [0,1]
    
    def test_compute_fidelity_similarity(self, processor_with_mock):
        """Test quantum fidelity similarity computation."""
        # Create test embeddings
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([1.0, 0.0, 0.0])  # Identical
        emb3 = np.array([0.0, 1.0, 0.0])  # Orthogonal
        
        # Test identical embeddings
        fid_identical = processor_with_mock.compute_fidelity_similarity(emb1, emb2)
        assert np.isclose(fid_identical, 1.0)
        
        # Test orthogonal embeddings
        fid_orthogonal = processor_with_mock.compute_fidelity_similarity(emb1, emb3)
        assert np.isclose(fid_orthogonal, 0.0)
    
    def test_benchmark_embedding_performance(self, processor_with_mock):
        """Test performance benchmarking."""
        # Mock model responses
        processor_with_mock.model.encode.side_effect = [
            np.random.rand(1, 768),  # Single text
            np.random.rand(4, 768)   # Batch texts
        ]
        
        results = processor_with_mock.benchmark_embedding_performance()
        
        # Check required metrics are present
        assert 'single_encoding_ms' in results
        assert 'batch_encoding_ms' in results
        assert 'batch_per_text_ms' in results
        assert 'quantum_preprocessing_ms' in results
        assert 'classical_similarity_ms' in results
        assert 'fidelity_similarity_ms' in results
        assert 'embedding_memory_mb' in results
        assert 'prd_compliance' in results
        
        # Check PRD compliance structure
        prd = results['prd_compliance']
        assert 'single_encoding_under_100ms' in prd
        assert 'similarity_under_100ms' in prd
        assert 'fidelity_under_100ms' in prd
        assert 'preprocessing_efficient' in prd
    
    def test_validate_embedding_quality(self, processor_with_mock):
        """Test embedding quality validation."""
        # Mock embeddings with known properties
        test_embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [0.2, 0.8, 0.1, 0.1]
        ])
        processor_with_mock.model.encode.return_value = test_embeddings
        
        results = processor_with_mock.validate_embedding_quality()
        
        # Check required fields
        assert 'embedding_dim' in results
        assert 'all_finite' in results
        assert 'normalized' in results
        assert 'embedding_range' in results
        assert 'quantum_compatible' in results
        assert 'cosine_similarity_stats' in results
        assert 'fidelity_similarity_stats' in results
        
        # Check that all values are finite
        assert results['all_finite'] is True
        
        # Check similarity stats structure
        for sim_type in ['cosine_similarity_stats', 'fidelity_similarity_stats']:
            stats = results[sim_type]
            assert 'mean' in stats
            assert 'min' in stats
            assert 'max' in stats
            assert 'std' in stats


class TestEmbeddingValidator:
    """Test EmbeddingValidator class."""
    
    @pytest.fixture
    def validator_with_mock(self):
        """EmbeddingValidator with mocked EmbeddingProcessor."""
        mock_processor = Mock(spec=EmbeddingProcessor)
        validator = EmbeddingValidator(mock_processor)
        return validator, mock_processor
    
    def test_init_default(self):
        """Test validator initialization with default processor."""
        with patch('quantum_rerank.core.embedding_validators.EmbeddingProcessor'):
            validator = EmbeddingValidator()
            assert validator.prd_targets is not None
            assert 'single_encoding_ms' in validator.prd_targets
    
    def test_validate_embedding_basic_properties_valid(self, validator_with_mock):
        """Test validation with valid embeddings."""
        validator, _ = validator_with_mock
        
        # Create valid normalized embeddings
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        result = validator.validate_embedding_basic_properties(embeddings)
        
        assert result.passed is True
        assert result.score > 0.0
        assert len(result.errors) == 0
        assert result.details['shape'] == (3, 3)
        assert result.details['finite_value_rate'] == 1.0
    
    def test_validate_embedding_basic_properties_invalid_shape(self, validator_with_mock):
        """Test validation with invalid embedding shape."""
        validator, _ = validator_with_mock
        
        # Create 1D array (invalid)
        embeddings = np.array([1.0, 2.0, 3.0])
        
        result = validator.validate_embedding_basic_properties(embeddings)
        
        assert result.passed is False
        assert result.score == 0.0
        assert len(result.errors) > 0
        assert "Expected 2D array" in result.errors[0]
    
    def test_validate_embedding_basic_properties_non_finite(self, validator_with_mock):
        """Test validation with non-finite values."""
        validator, _ = validator_with_mock
        
        # Create embeddings with NaN and inf
        embeddings = np.array([
            [1.0, 2.0, np.nan],
            [np.inf, 0.0, 1.0]
        ])
        
        result = validator.validate_embedding_basic_properties(embeddings)
        
        assert result.details['finite_value_rate'] < 1.0
        assert len(result.warnings) > 0
        assert "Non-finite values detected" in result.warnings[0]
    
    def test_validate_quantum_compatibility_success(self, validator_with_mock):
        """Test quantum compatibility validation success."""
        validator, mock_processor = validator_with_mock
        
        # Mock preprocessing to return valid quantum embeddings
        processed_embeddings = np.array([
            [0.5, 0.5, 0.5, 0.5],  # Normalized
            [0.7, 0.7, 0.0, 0.0]   # Normalized
        ])
        processed_embeddings = processed_embeddings / np.linalg.norm(processed_embeddings, axis=1, keepdims=True)
        
        mock_processor.preprocess_for_quantum.return_value = (
            processed_embeddings,
            {'target_amplitudes': 4, 'processing_applied': ['normalization']}
        )
        
        embeddings = np.random.rand(2, 8)
        result = validator.validate_quantum_compatibility(embeddings, n_qubits=2)
        
        assert result.passed is True
        assert result.score > 0.0
        assert result.details['quantum_normalized'] is True
    
    def test_validate_quantum_compatibility_failure(self, validator_with_mock):
        """Test quantum compatibility validation failure."""
        validator, mock_processor = validator_with_mock
        
        # Mock preprocessing to return improperly normalized embeddings
        processed_embeddings = np.array([
            [2.0, 0.0, 0.0, 0.0],  # Not normalized
            [0.0, 3.0, 0.0, 0.0]   # Not normalized
        ])
        
        mock_processor.preprocess_for_quantum.return_value = (
            processed_embeddings,
            {'target_amplitudes': 4, 'processing_applied': []}
        )
        
        embeddings = np.random.rand(2, 4)
        result = validator.validate_quantum_compatibility(embeddings, n_qubits=2)
        
        assert result.passed is False
        assert result.score == 0.0
        assert len(result.errors) > 0
        assert "not properly normalized" in result.errors[0]
    
    def test_validate_similarity_quality(self, validator_with_mock):
        """Test similarity quality validation."""
        validator, mock_processor = validator_with_mock
        
        # Mock embeddings for test texts
        test_embeddings = np.array([
            [1.0, 0.0, 0.0],  # quantum
            [0.9, 0.1, 0.0],  # quantum (similar)
            [0.0, 1.0, 0.0],  # ML
            [0.0, 0.0, 1.0],  # classical (different)
            [0.1, 0.1, 0.8]   # NLP
        ])
        mock_processor.encode_texts.return_value = test_embeddings
        
        # Mock similarity computations
        mock_processor.compute_classical_similarity.side_effect = [
            0.95,  # quantum-quantum (high)
            0.3,   # quantum-classical (low)
            # ... other combinations
        ]
        mock_processor.compute_fidelity_similarity.side_effect = [
            0.90,  # quantum-quantum (high)
            0.1,   # quantum-classical (low)
            # ... other combinations
        ]
        
        result = validator.validate_similarity_quality()
        
        assert result.passed is True
        assert 'cosine_similarity_matrix' in result.details
        assert 'fidelity_similarity_matrix' in result.details
        assert 'expected_relationships' in result.details
    
    def test_run_performance_benchmarks(self, validator_with_mock):
        """Test performance benchmarking."""
        validator, mock_processor = validator_with_mock
        
        # Mock various operations
        mock_processor.encode_single_text.return_value = np.random.rand(768)
        mock_processor.encode_texts.return_value = np.random.rand(4, 768)
        mock_processor.compute_classical_similarity.return_value = 0.8
        mock_processor.preprocess_for_quantum.return_value = (
            np.random.rand(4, 16), {}
        )
        
        benchmarks = validator.run_performance_benchmarks()
        
        assert len(benchmarks) >= 3  # At least single, batch, similarity tests
        
        for benchmark in benchmarks:
            assert hasattr(benchmark, 'test_name')
            assert hasattr(benchmark, 'duration_ms')
            assert hasattr(benchmark, 'success')
            assert hasattr(benchmark, 'target_met')
            assert hasattr(benchmark, 'target_value')
            assert hasattr(benchmark, 'actual_value')
    
    def test_generate_validation_report(self, validator_with_mock):
        """Test comprehensive validation report generation."""
        validator, mock_processor = validator_with_mock
        
        # Mock all required operations
        mock_processor.encode_texts.return_value = np.random.rand(4, 768)
        mock_processor.encode_single_text.return_value = np.random.rand(768)
        mock_processor.compute_classical_similarity.return_value = 0.8
        mock_processor.compute_fidelity_similarity.return_value = 0.75
        mock_processor.preprocess_for_quantum.return_value = (
            np.random.rand(4, 16), {'processing_applied': ['normalization']}
        )
        
        # Mock config
        mock_processor.config = Mock()
        mock_processor.config.model_name = 'test-model'
        mock_processor.config.embedding_dim = 768
        
        report = validator.generate_validation_report()
        
        assert 'timestamp' in report
        assert 'validator_config' in report
        assert 'validations' in report
        assert 'benchmarks' in report
        assert 'summary' in report
        
        # Check summary structure
        summary = report['summary']
        assert 'overall_validation_passed' in summary
        assert 'average_validation_score' in summary
        assert 'all_benchmarks_passed' in summary
        assert 'prd_compliance' in summary
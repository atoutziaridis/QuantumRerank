"""
Integration tests for embedding processing pipeline.

Tests the full integration of embedding processing components for
Task 03: SentenceTransformer Integration and Embedding Processing.
"""

import pytest
import numpy as np
import time
from typing import List

from quantum_rerank.core.embeddings import EmbeddingProcessor, EmbeddingConfig
from quantum_rerank.core.quantum_embedding_bridge import QuantumEmbeddingBridge
from quantum_rerank.core.embedding_validators import EmbeddingValidator
from quantum_rerank.core.quantum_circuits import BasicQuantumCircuits


class TestEmbeddingProcessorIntegration:
    """Integration tests for EmbeddingProcessor with real models."""
    
    @pytest.fixture
    def real_processor(self):
        """Real EmbeddingProcessor for integration testing."""
        # Use a smaller, faster model for testing
        config = EmbeddingConfig(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            embedding_dim=384,
            batch_size=4
        )
        return EmbeddingProcessor(config)
    
    def test_real_model_loading(self, real_processor):
        """Test loading real SentenceTransformer model."""
        assert real_processor.model is not None
        assert real_processor.config.embedding_dim == 384
        assert real_processor.device in ['cpu', 'cuda']
    
    def test_real_text_encoding(self, real_processor):
        """Test encoding real texts with actual model."""
        test_texts = [
            "Quantum computing leverages quantum mechanics",
            "Machine learning processes data automatically",
            "Natural language processing understands text"
        ]
        
        embeddings = real_processor.encode_texts(test_texts)
        
        assert embeddings.shape == (3, 384)
        assert np.all(np.isfinite(embeddings))
        assert not np.allclose(embeddings[0], embeddings[1])  # Should be different
    
    def test_real_single_text_encoding(self, real_processor):
        """Test single text encoding with real model."""
        text = "This is a test sentence for embedding."
        
        embedding = real_processor.encode_single_text(text)
        
        assert embedding.shape == (384,)
        assert np.all(np.isfinite(embedding))
        assert np.linalg.norm(embedding) > 0  # Non-zero embedding
    
    def test_real_quantum_preprocessing(self, real_processor):
        """Test quantum preprocessing with real embeddings."""
        texts = ["quantum", "classical"]
        embeddings = real_processor.encode_texts(texts)
        
        processed, metadata = real_processor.preprocess_for_quantum(embeddings, n_qubits=4)
        
        assert processed.shape == (2, 16)  # 2^4 = 16
        assert np.allclose(np.linalg.norm(processed, axis=1), 1.0)  # Normalized
        assert 'processing_applied' in metadata
        assert metadata['target_amplitudes'] == 16
    
    def test_real_similarity_computations(self, real_processor):
        """Test similarity computations with real embeddings."""
        texts = [
            "quantum computing and quantum algorithms",
            "quantum physics and quantum mechanics",  # Similar to above
            "classical machine learning methods"       # Different topic
        ]
        
        embeddings = real_processor.encode_texts(texts)
        
        # Classical cosine similarity
        sim_quantum = real_processor.compute_classical_similarity(embeddings[0], embeddings[1])
        sim_different = real_processor.compute_classical_similarity(embeddings[0], embeddings[2])
        
        # Quantum fidelity similarity
        fid_quantum = real_processor.compute_fidelity_similarity(embeddings[0], embeddings[1])
        fid_different = real_processor.compute_fidelity_similarity(embeddings[0], embeddings[2])
        
        # Related texts should be more similar
        assert sim_quantum > sim_different
        assert fid_quantum > fid_different
        
        # Values should be in valid ranges
        assert 0 <= sim_quantum <= 1
        assert 0 <= sim_different <= 1
        assert 0 <= fid_quantum <= 1
        assert 0 <= fid_different <= 1
    
    def test_real_batch_processing(self, real_processor):
        """Test batch processing with real model."""
        texts = [
            "Text one about quantum computing",
            "Text two about machine learning", 
            "Text three about information retrieval",
            "Text four about natural language processing",
            "Text five about artificial intelligence"
        ]
        
        batches = real_processor.create_embedding_batches(texts, batch_size=2)
        
        assert len(batches) == 3  # 5 texts with batch_size=2
        assert len(batches[0][0]) == 2  # First batch
        assert len(batches[1][0]) == 2  # Second batch  
        assert len(batches[2][0]) == 1  # Third batch
        
        # Check embeddings
        for batch_texts, batch_embeddings in batches:
            assert batch_embeddings.shape[0] == len(batch_texts)
            assert batch_embeddings.shape[1] == 384
    
    def test_real_performance_benchmark(self, real_processor):
        """Test performance benchmarking with real model."""
        results = real_processor.benchmark_embedding_performance()
        
        # Check all required metrics
        required_metrics = [
            'single_encoding_ms', 'batch_encoding_ms', 'batch_per_text_ms',
            'quantum_preprocessing_ms', 'classical_similarity_ms',
            'fidelity_similarity_ms', 'embedding_memory_mb', 'prd_compliance'
        ]
        
        for metric in required_metrics:
            assert metric in results
            if 'ms' in metric or 'mb' in metric:
                assert results[metric] >= 0  # Non-negative timing/memory
        
        # Check PRD compliance structure
        prd = results['prd_compliance']
        assert all(isinstance(v, bool) for v in prd.values())
    
    def test_real_embedding_quality_validation(self, real_processor):
        """Test embedding quality validation with real model."""
        results = real_processor.validate_embedding_quality()
        
        assert results['embedding_dim'] == 384
        assert results['all_finite'] is True
        assert results['normalized'] is True  # Model should normalize by default
        assert results['quantum_compatible'] is True
        
        # Check similarity statistics
        for sim_type in ['cosine_similarity_stats', 'fidelity_similarity_stats']:
            stats = results[sim_type]
            assert 0 <= stats['mean'] <= 1
            assert 0 <= stats['min'] <= 1
            assert 0 <= stats['max'] <= 1
            assert stats['std'] >= 0


class TestQuantumEmbeddingBridgeIntegration:
    """Integration tests for QuantumEmbeddingBridge with real components."""
    
    @pytest.fixture
    def real_bridge(self):
        """Real QuantumEmbeddingBridge for integration testing."""
        # Use smaller model and fewer qubits for faster testing
        embedding_config = EmbeddingConfig(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            embedding_dim=384
        )
        return QuantumEmbeddingBridge(n_qubits=3, embedding_config=embedding_config)
    
    def test_real_bridge_initialization(self, real_bridge):
        """Test real bridge initialization."""
        assert real_bridge.n_qubits == 3
        assert real_bridge.embedding_processor is not None
        assert real_bridge.quantum_circuits is not None
    
    def test_real_text_to_quantum_circuit_amplitude(self, real_bridge):
        """Test real text to quantum circuit conversion with amplitude encoding."""
        test_text = "Quantum computing uses superposition and entanglement"
        
        result = real_bridge.text_to_quantum_circuit(test_text, encoding_method='amplitude')
        
        assert result.success is True
        assert result.text == test_text
        assert result.circuit is not None
        assert result.statevector is not None
        assert result.metadata is not None
        
        # Check metadata
        metadata = result.metadata
        assert metadata['text_length'] == len(test_text)
        assert metadata['encoding_method'] == 'amplitude'
        assert metadata['quantum_circuit_depth'] <= 15  # PRD constraint
        assert metadata['prd_compliant'] is True
    
    def test_real_text_to_quantum_circuit_all_encodings(self, real_bridge):
        """Test all encoding methods with real bridge."""
        test_text = "Test sentence for quantum encoding"
        encoding_methods = ['amplitude', 'angle', 'dense_angle']
        
        for method in encoding_methods:
            result = real_bridge.text_to_quantum_circuit(test_text, encoding_method=method)
            
            assert result.success is True, f"Failed for method {method}"
            assert result.metadata['encoding_method'] == method
            assert result.circuit.num_qubits == 3
    
    def test_real_batch_texts_to_circuits(self, real_bridge):
        """Test real batch text to circuit conversion."""
        test_texts = [
            "Quantum algorithms for optimization",
            "Classical machine learning techniques",
            "Hybrid quantum-classical computing"
        ]
        
        results = real_bridge.batch_texts_to_circuits(test_texts, encoding_method='amplitude')
        
        assert len(results) == 3
        assert all(result.success for result in results)
        
        for i, result in enumerate(results):
            assert result.text == test_texts[i]
            assert result.circuit is not None
            assert result.statevector is not None
            assert result.metadata['batch_index'] == i
    
    def test_real_quantum_similarity_computation(self, real_bridge):
        """Test real quantum similarity computation."""
        text1 = "Quantum computing and quantum algorithms"
        text2 = "Quantum physics and quantum mechanics"  # Similar topic
        text3 = "Classical computing and traditional algorithms"  # Different topic
        
        # Similar texts
        sim_result_similar = real_bridge.compute_quantum_similarity(text1, text2)
        
        # Different texts
        sim_result_different = real_bridge.compute_quantum_similarity(text1, text3)
        
        # Check results structure
        for result in [sim_result_similar, sim_result_different]:
            assert not np.isnan(result.classical_cosine)
            assert not np.isnan(result.quantum_fidelity)
            assert result.quantum_amplitude_overlap is not None
            assert result.computation_time_ms > 0
            assert result.metadata is not None
        
        # Similar texts should have higher similarity
        assert sim_result_similar.classical_cosine > sim_result_different.classical_cosine
        assert sim_result_similar.quantum_fidelity > sim_result_different.quantum_fidelity
    
    def test_real_bridge_performance_benchmark(self, real_bridge):
        """Test real bridge performance benchmarking."""
        test_texts = [
            "Quantum computing research",
            "Machine learning applications",
            "Information retrieval systems"
        ]
        
        results = real_bridge.benchmark_bridge_performance(test_texts)
        
        # Check structure
        encoding_methods = ['amplitude', 'angle', 'dense_angle']
        for method in encoding_methods:
            assert method in results
            method_result = results[method]
            
            # Check metrics
            assert 'avg_single_conversion_ms' in method_result
            assert 'batch_conversion_ms' in method_result
            assert 'avg_similarity_computation_ms' in method_result
            assert 'success_rate' in method_result
            assert 'prd_targets' in method_result
            
            # Check values are reasonable
            assert method_result['success_rate'] > 0.5  # At least 50% success
            assert method_result['avg_single_conversion_ms'] > 0
        
        # Check summary
        summary = results['summary']
        assert 'overall_avg_similarity_ms' in summary
        assert 'overall_success_rate' in summary
        assert 'prd_compliance' in summary


class TestEmbeddingValidatorIntegration:
    """Integration tests for EmbeddingValidator with real components."""
    
    @pytest.fixture
    def real_validator(self):
        """Real EmbeddingValidator for integration testing."""
        embedding_config = EmbeddingConfig(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            embedding_dim=384
        )
        processor = EmbeddingProcessor(embedding_config)
        return EmbeddingValidator(processor)
    
    def test_real_validator_initialization(self, real_validator):
        """Test real validator initialization."""
        assert real_validator.embedding_processor is not None
        assert real_validator.prd_targets is not None
        assert len(real_validator.prd_targets) > 0
    
    def test_real_basic_properties_validation(self, real_validator):
        """Test basic properties validation with real embeddings."""
        test_texts = ["quantum computing", "machine learning", "information retrieval"]
        embeddings = real_validator.embedding_processor.encode_texts(test_texts)
        
        result = real_validator.validate_embedding_basic_properties(embeddings)
        
        assert result.passed is True
        assert result.score > 0.8  # Should be high quality
        assert len(result.errors) == 0
        assert result.details['shape'] == (3, 384)
        assert result.details['finite_value_rate'] == 1.0
    
    def test_real_quantum_compatibility_validation(self, real_validator):
        """Test quantum compatibility validation with real embeddings."""
        test_texts = ["test text 1", "test text 2"]
        embeddings = real_validator.embedding_processor.encode_texts(test_texts)
        
        result = real_validator.validate_quantum_compatibility(embeddings, n_qubits=3)
        
        assert result.passed is True
        assert result.score > 0.8
        assert result.details['quantum_normalized'] is True
        assert result.details['processed_shape'] == (2, 8)  # 2^3 = 8
    
    def test_real_similarity_quality_validation(self, real_validator):
        """Test similarity quality validation with real embeddings."""
        result = real_validator.validate_similarity_quality()
        
        assert result.passed is True
        assert result.score > 0.7  # Should pass quality threshold
        assert 'cosine_similarity_matrix' in result.details
        assert 'fidelity_similarity_matrix' in result.details
        assert 'expected_relationships' in result.details
        
        # Check similarity matrices are valid
        cosine_matrix = np.array(result.details['cosine_similarity_matrix'])
        fidelity_matrix = np.array(result.details['fidelity_similarity_matrix'])
        
        assert np.all(cosine_matrix >= 0) and np.all(cosine_matrix <= 1)
        assert np.all(fidelity_matrix >= 0) and np.all(fidelity_matrix <= 1)
        assert np.allclose(np.diag(cosine_matrix), 1.0)  # Self-similarity = 1
    
    def test_real_performance_benchmarks(self, real_validator):
        """Test performance benchmarks with real components."""
        benchmarks = real_validator.run_performance_benchmarks()
        
        assert len(benchmarks) >= 3  # Should have multiple benchmarks
        
        benchmark_names = [b.test_name for b in benchmarks]
        expected_names = ['single_text_encoding', 'batch_text_encoding', 'similarity_computation']
        
        for name in expected_names:
            assert name in benchmark_names
        
        # Check benchmark properties
        for benchmark in benchmarks:
            assert benchmark.duration_ms >= 0
            assert benchmark.success is True  # Should succeed with real components
            assert isinstance(benchmark.target_met, bool)
            assert benchmark.target_value > 0
    
    def test_real_validation_report_generation(self, real_validator):
        """Test comprehensive validation report with real components."""
        report = real_validator.generate_validation_report()
        
        # Check report structure
        assert 'timestamp' in report
        assert 'validator_config' in report
        assert 'validations' in report
        assert 'benchmarks' in report
        assert 'summary' in report
        
        # Check validations
        validations = report['validations']
        validation_types = ['basic_properties', 'quantum_compatibility', 'similarity_quality']
        
        for val_type in validation_types:
            assert val_type in validations
            validation = validations[val_type]
            assert hasattr(validation, 'passed')
            assert hasattr(validation, 'score')
        
        # Check benchmarks
        benchmarks = report['benchmarks']
        assert len(benchmarks) > 0
        
        # Check summary
        summary = report['summary']
        assert 'overall_validation_passed' in summary
        assert 'average_validation_score' in summary
        assert 'prd_compliance' in summary
        assert isinstance(summary['prd_compliance'], bool)


class TestFullPipelineIntegration:
    """End-to-end integration tests for the complete embedding pipeline."""
    
    @pytest.fixture
    def full_pipeline(self):
        """Complete pipeline setup for end-to-end testing."""
        embedding_config = EmbeddingConfig(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            embedding_dim=384,
            batch_size=2
        )
        
        processor = EmbeddingProcessor(embedding_config)
        bridge = QuantumEmbeddingBridge(n_qubits=3, embedding_config=embedding_config)
        validator = EmbeddingValidator(processor)
        
        return {
            'processor': processor,
            'bridge': bridge,
            'validator': validator
        }
    
    def test_end_to_end_text_processing(self, full_pipeline):
        """Test complete end-to-end text processing pipeline."""
        processor = full_pipeline['processor']
        bridge = full_pipeline['bridge']
        
        # Sample texts representing a real-world scenario
        query = "quantum machine learning algorithms"
        documents = [
            "Quantum algorithms for machine learning optimization",
            "Classical neural networks and deep learning",
            "Quantum computing applications in AI",
            "Traditional machine learning methods"
        ]
        
        # Step 1: Process query
        query_result = bridge.text_to_quantum_circuit(query)
        assert query_result.success is True
        
        # Step 2: Process documents
        doc_results = bridge.batch_texts_to_circuits(documents)
        assert len(doc_results) == 4
        assert all(result.success for result in doc_results)
        
        # Step 3: Compute similarities
        similarities = []
        for doc in documents:
            sim_result = bridge.compute_quantum_similarity(query, doc)
            similarities.append({
                'document': doc,
                'classical_cosine': sim_result.classical_cosine,
                'quantum_fidelity': sim_result.quantum_fidelity,
                'quantum_amplitude': sim_result.quantum_amplitude_overlap
            })
        
        # Step 4: Verify semantic relationships
        # Documents 0 and 2 should be most similar to query (quantum ML related)
        quantum_ml_similarities = [similarities[0]['classical_cosine'], similarities[2]['classical_cosine']]
        other_similarities = [similarities[1]['classical_cosine'], similarities[3]['classical_cosine']]
        
        assert max(quantum_ml_similarities) > max(other_similarities)
    
    def test_end_to_end_performance_validation(self, full_pipeline):
        """Test end-to-end performance meets PRD requirements."""
        validator = full_pipeline['validator']
        bridge = full_pipeline['bridge']
        
        # Generate comprehensive validation report
        validation_report = validator.generate_validation_report()
        
        # Run bridge benchmarks
        bridge_benchmarks = bridge.benchmark_bridge_performance()
        
        # Check overall compliance
        assert validation_report['summary']['prd_compliance'] is True
        
        # Check specific performance targets
        summary = bridge_benchmarks['summary']
        assert summary['prd_compliance']['similarity_target_met'] is True
        assert summary['overall_success_rate'] > 0.9
    
    def test_end_to_end_error_handling(self, full_pipeline):
        """Test error handling throughout the pipeline."""
        bridge = full_pipeline['bridge']
        
        # Test with problematic inputs
        problematic_inputs = [
            "",  # Empty string
            " ",  # Whitespace only
            "a" * 1000,  # Very long text
            "Special chars: @#$%^&*()",  # Special characters
        ]
        
        for text in problematic_inputs:
            try:
                result = bridge.text_to_quantum_circuit(text)
                # Should either succeed or fail gracefully
                assert hasattr(result, 'success')
                if not result.success:
                    assert result.error is not None
            except Exception as e:
                # Any exceptions should be handled gracefully
                pytest.fail(f"Unhandled exception for input '{text}': {e}")
    
    def test_end_to_end_memory_efficiency(self, full_pipeline):
        """Test memory efficiency of the pipeline."""
        import psutil
        import os
        
        processor = full_pipeline['processor']
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process a reasonably large batch
        large_texts = [f"Test document number {i} about various topics" for i in range(20)]
        
        embeddings = processor.encode_texts(large_texts)
        processed_embeddings, _ = processor.preprocess_for_quantum(embeddings)
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not exceed PRD memory target for batch processing
        # (This is a rough check - actual limits depend on system)
        assert memory_increase < 500  # MB - reasonable limit for test batch
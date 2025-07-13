"""
Unit tests for benchmarking framework.

Tests the core performance benchmarking functionality for
Task 08: Performance Benchmarking Framework.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path

from quantum_rerank.benchmarks.benchmark_framework import (
    PerformanceBenchmarker, BenchmarkConfig, BenchmarkResult
)


class TestBenchmarkConfig:
    """Test BenchmarkConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BenchmarkConfig()
        
        assert config.similarity_computation_target_ms == 100.0
        assert config.batch_processing_target_ms == 500.0
        assert config.memory_usage_target_gb == 2.0
        assert config.accuracy_improvement_target == 0.15
        assert config.num_trials == 10
        assert config.warmup_trials == 3
        assert config.statistical_confidence == 0.95
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = BenchmarkConfig(
            similarity_computation_target_ms=50.0,
            num_trials=5,
            warmup_trials=1
        )
        
        assert config.similarity_computation_target_ms == 50.0
        assert config.num_trials == 5
        assert config.warmup_trials == 1


class TestBenchmarkResult:
    """Test BenchmarkResult class."""
    
    def test_benchmark_result_creation(self):
        """Test BenchmarkResult creation."""
        result = BenchmarkResult(
            test_name="test_similarity",
            component="quantum_engine",
            metric_type="latency",
            duration_ms=75.5,
            target_ms=100.0,
            target_met=True,
            success=True
        )
        
        assert result.test_name == "test_similarity"
        assert result.component == "quantum_engine"
        assert result.metric_type == "latency"
        assert result.duration_ms == 75.5
        assert result.target_ms == 100.0
        assert result.target_met is True
        assert result.success is True
        assert isinstance(result.timestamp, datetime)


class TestPerformanceBenchmarker:
    """Test PerformanceBenchmarker class."""
    
    @pytest.fixture
    def mock_components(self):
        """Mock all benchmarker components."""
        with patch('quantum_rerank.benchmarks.benchmark_framework.EmbeddingProcessor') as mock_ep:
            with patch('quantum_rerank.benchmarks.benchmark_framework.QuantumSimilarityEngine') as mock_qse:
                with patch('quantum_rerank.benchmarks.benchmark_framework.QuantumEmbeddingBridge') as mock_qeb:
                    with patch('quantum_rerank.benchmarks.benchmark_framework.TwoStageRetriever') as mock_tsr:
                        # Configure mocks
                        mock_ep.return_value = Mock()
                        mock_qse.return_value = Mock()
                        mock_qeb.return_value = Mock()
                        mock_tsr.return_value = Mock()
                        
                        yield {
                            'embedding_processor': mock_ep.return_value,
                            'quantum_engine': mock_qse.return_value,
                            'quantum_bridge': mock_qeb.return_value,
                            'two_stage_retriever': mock_tsr.return_value
                        }
    
    @pytest.fixture
    def benchmarker(self, mock_components):
        """Create benchmarker with mocked components."""
        config = BenchmarkConfig(num_trials=3, warmup_trials=1)
        
        with patch('pathlib.Path.mkdir'):
            benchmarker = PerformanceBenchmarker(config)
            # Manually set mocked components
            for attr, mock_obj in mock_components.items():
                setattr(benchmarker, attr, mock_obj)
            return benchmarker
    
    def test_benchmarker_initialization(self, mock_components):
        """Test benchmarker initialization."""
        config = BenchmarkConfig(num_trials=5)
        
        with patch('pathlib.Path.mkdir'):
            benchmarker = PerformanceBenchmarker(config)
            
            assert benchmarker.config.num_trials == 5
            assert benchmarker.results == []
            assert isinstance(benchmarker.output_dir, Path)
    
    def test_benchmark_component_latency_success(self, benchmarker):
        """Test successful component latency benchmarking."""
        def mock_test_function(data):
            time.sleep(0.01)  # 10ms
            return "success"
        
        result = benchmarker.benchmark_component_latency(
            component_name="test_component",
            test_function=mock_test_function,
            test_data="test_data",
            target_ms=50.0
        )
        
        assert result.test_name == "test_component_latency"
        assert result.component == "test_component"
        assert result.metric_type == "latency"
        assert result.success is True
        assert result.target_met is True  # Should be under 50ms
        assert result.duration_ms > 0
        assert len(result.trials) == benchmarker.config.num_trials
    
    def test_benchmark_component_latency_failure(self, benchmarker):
        """Test component latency benchmarking with failures."""
        def failing_function(data):
            raise RuntimeError("Test failure")
        
        result = benchmarker.benchmark_component_latency(
            component_name="failing_component",
            test_function=failing_function,
            test_data="test_data",
            target_ms=100.0
        )
        
        assert result.test_name == "failing_component_latency"
        assert result.success is False
        assert len(result.trials) == 0
        assert result.duration_ms == float('inf')
    
    def test_benchmark_similarity_computation(self, benchmarker):
        """Test similarity computation benchmarking."""
        # Mock embedding processor methods
        test_embeddings = np.random.rand(3, 768)
        benchmarker.embedding_processor.encode_texts.return_value = test_embeddings
        benchmarker.embedding_processor.compute_classical_similarity.return_value = 0.8
        benchmarker.embedding_processor.compute_fidelity_similarity.return_value = 0.75
        benchmarker.quantum_engine.compute_similarity.return_value = 0.85
        
        results = benchmarker.benchmark_similarity_computation()
        
        assert len(results) == 3  # classical_cosine, quantum_fidelity, quantum_engine
        assert all(result.success for result in results)
        assert all("similarity" in result.test_name for result in results)
        assert all(result.target_ms == 100.0 for result in results)
    
    def test_benchmark_batch_processing(self, benchmarker):
        """Test batch processing benchmarking."""
        # Mock batch operations
        benchmarker.embedding_processor.encode_texts.return_value = np.random.rand(10, 768)
        benchmarker.quantum_bridge.batch_texts_to_circuits.return_value = [Mock() for _ in range(10)]
        
        results = benchmarker.benchmark_batch_processing()
        
        assert len(results) > 0
        batch_50_results = [r for r in results if r.batch_size == 50]
        assert len(batch_50_results) >= 2  # embedding and quantum
        
        for result in results:
            assert result.batch_size in benchmarker.config.test_batch_sizes
            assert "batch" in result.test_name
    
    def test_benchmark_memory_usage(self, benchmarker):
        """Test memory usage benchmarking."""
        # Mock memory-intensive operations
        benchmarker.embedding_processor.encode_texts.return_value = np.random.rand(100, 768)
        benchmarker.quantum_bridge.batch_texts_to_circuits.return_value = [Mock() for _ in range(100)]
        
        with patch('psutil.Process') as mock_process:
            # Mock memory info
            mock_memory_info = Mock()
            mock_memory_info.rss = 1024 * 1024 * 1024  # 1GB in bytes
            mock_process.return_value.memory_info.return_value = mock_memory_info
            
            results = benchmarker.benchmark_memory_usage()
        
        assert len(results) == len(benchmarker.config.test_document_counts)
        
        for result in results:
            assert result.metric_type == "memory"
            assert result.document_count in benchmarker.config.test_document_counts
            assert "memory_usage" in result.test_name
    
    def test_benchmark_end_to_end_pipeline(self, benchmarker):
        """Test end-to-end pipeline benchmarking."""
        # Mock pipeline result
        mock_ranked_docs = ["doc1", "doc2", "doc3"]
        benchmarker.two_stage_retriever.rerank.return_value = mock_ranked_docs
        
        results = benchmarker.benchmark_end_to_end_pipeline()
        
        assert len(results) == 1
        result = results[0]
        assert result.test_name == "end_to_end_reranking_latency"
        assert result.component == "end_to_end_reranking"
        assert result.target_ms == benchmarker.config.batch_processing_target_ms
    
    def test_run_comprehensive_benchmark(self, benchmarker):
        """Test comprehensive benchmark suite."""
        # Mock all required operations
        benchmarker.embedding_processor.encode_texts.return_value = np.random.rand(10, 768)
        benchmarker.embedding_processor.compute_classical_similarity.return_value = 0.8
        benchmarker.embedding_processor.compute_fidelity_similarity.return_value = 0.75
        benchmarker.quantum_engine.compute_similarity.return_value = 0.85
        benchmarker.quantum_bridge.batch_texts_to_circuits.return_value = [Mock() for _ in range(10)]
        benchmarker.two_stage_retriever.rerank.return_value = ["doc1", "doc2"]
        
        with patch('psutil.Process') as mock_process:
            mock_memory_info = Mock()
            mock_memory_info.rss = 512 * 1024 * 1024  # 512MB
            mock_process.return_value.memory_info.return_value = mock_memory_info
            
            benchmark_suite = benchmarker.run_comprehensive_benchmark()
        
        expected_categories = ['similarity_computation', 'batch_processing', 'memory_usage', 'end_to_end']
        assert all(category in benchmark_suite for category in expected_categories)
        
        # Check that all categories have results
        for category, results in benchmark_suite.items():
            assert len(results) > 0
            assert all(isinstance(result, BenchmarkResult) for result in results)
    
    def test_get_prd_compliance_summary_no_results(self, benchmarker):
        """Test PRD compliance summary with no results."""
        summary = benchmarker.get_prd_compliance_summary()
        
        assert "error" in summary
        assert summary["error"] == "No benchmark results available"
    
    def test_get_prd_compliance_summary_with_results(self, benchmarker):
        """Test PRD compliance summary with results."""
        # Add mock results
        benchmarker.results = [
            BenchmarkResult(
                test_name="similarity_test",
                component="quantum",
                metric_type="latency",
                duration_ms=75.0,
                target_ms=100.0,
                target_met=True,
                success=True
            ),
            BenchmarkResult(
                test_name="batch_test", 
                component="system",
                metric_type="latency",
                duration_ms=450.0,
                target_ms=500.0,
                target_met=True,
                success=True
            ),
            BenchmarkResult(
                test_name="memory_test",
                component="system",
                metric_type="memory",
                duration_ms=0.0,
                memory_mb=1500.0,  # 1.5GB
                document_count=100,
                success=True
            )
        ]
        
        summary = benchmarker.get_prd_compliance_summary()
        
        assert "overall_prd_compliance" in summary
        assert "similarity_computation_compliant" in summary
        assert "batch_processing_compliant" in summary
        assert "memory_usage_compliant" in summary
        assert summary["total_tests_run"] == 3
        assert summary["successful_tests"] == 3
    
    def test_benchmark_component_with_memory_tracking(self, benchmarker):
        """Test component benchmarking with memory tracking."""
        def memory_intensive_function(data):
            # Simulate memory allocation
            dummy_data = np.random.rand(1000, 1000)  # ~8MB
            time.sleep(0.01)
            return dummy_data.shape
        
        with patch('psutil.Process') as mock_process:
            # Mock increasing memory usage
            memory_values = [100, 120, 108]  # MB, simulating peak then drop
            mock_process.return_value.memory_info.side_effect = [
                Mock(rss=val * 1024 * 1024) for val in memory_values * 10
            ]
            
            result = benchmarker.benchmark_component_latency(
                component_name="memory_test",
                test_function=memory_intensive_function,
                test_data="test_data"
            )
        
        assert result.success is True
        assert result.memory_mb is not None


class TestBenchmarkIntegration:
    """Integration tests for benchmarking system."""
    
    def test_benchmark_with_real_operations(self):
        """Test benchmarking with actual (simple) operations."""
        config = BenchmarkConfig(num_trials=2, warmup_trials=1)
        
        def simple_similarity(embeddings):
            """Simple similarity computation for testing."""
            emb1, emb2 = embeddings
            return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        with patch('pathlib.Path.mkdir'):
            with patch('quantum_rerank.benchmarks.benchmark_framework.EmbeddingProcessor'):
                with patch('quantum_rerank.benchmarks.benchmark_framework.QuantumSimilarityEngine'):
                    with patch('quantum_rerank.benchmarks.benchmark_framework.QuantumEmbeddingBridge'):
                        with patch('quantum_rerank.benchmarks.benchmark_framework.TwoStageRetriever'):
                            benchmarker = PerformanceBenchmarker(config)
        
        # Test with simple function
        test_embeddings = (np.random.rand(100), np.random.rand(100))
        
        result = benchmarker.benchmark_component_latency(
            component_name="simple_similarity",
            test_function=simple_similarity,
            test_data=test_embeddings,
            target_ms=10.0
        )
        
        assert result.success is True
        assert result.duration_ms > 0
        assert len(result.trials) == config.num_trials
        assert result.mean == result.duration_ms
        assert result.std >= 0
    
    def test_benchmark_statistical_properties(self):
        """Test statistical properties of benchmark results."""
        config = BenchmarkConfig(num_trials=20, warmup_trials=2)
        
        def variable_latency_function(data):
            """Function with variable latency for statistical testing."""
            # Add some randomness to latency
            delay = np.random.normal(0.01, 0.002)  # 10ms ± 2ms
            time.sleep(max(0.001, delay))
            return "result"
        
        with patch('pathlib.Path.mkdir'):
            with patch('quantum_rerank.benchmarks.benchmark_framework.EmbeddingProcessor'):
                with patch('quantum_rerank.benchmarks.benchmark_framework.QuantumSimilarityEngine'):
                    with patch('quantum_rerank.benchmarks.benchmark_framework.QuantumEmbeddingBridge'):
                        with patch('quantum_rerank.benchmarks.benchmark_framework.TwoStageRetriever'):
                            benchmarker = PerformanceBenchmarker(config)
        
        result = benchmarker.benchmark_component_latency(
            component_name="variable_latency",
            test_function=variable_latency_function,
            test_data="test_data"
        )
        
        assert result.success is True
        assert len(result.trials) == config.num_trials
        assert result.std > 0  # Should have some variance
        assert result.min_val <= result.mean <= result.max_val
        assert result.median > 0
    
    def test_error_handling_in_benchmark(self):
        """Test error handling during benchmarking."""
        config = BenchmarkConfig(num_trials=3, warmup_trials=1)
        
        def intermittent_failure_function(data):
            """Function that fails intermittently."""
            if np.random.random() < 0.5:  # 50% failure rate
                raise RuntimeError("Random failure")
            time.sleep(0.01)
            return "success"
        
        with patch('pathlib.Path.mkdir'):
            with patch('quantum_rerank.benchmarks.benchmark_framework.EmbeddingProcessor'):
                with patch('quantum_rerank.benchmarks.benchmark_framework.QuantumSimilarityEngine'):
                    with patch('quantum_rerank.benchmarks.benchmark_framework.QuantumEmbeddingBridge'):
                        with patch('quantum_rerank.benchmarks.benchmark_framework.TwoStageRetriever'):
                            benchmarker = PerformanceBenchmarker(config)
        
        result = benchmarker.benchmark_component_latency(
            component_name="intermittent_failure",
            test_function=intermittent_failure_function,
            test_data="test_data"
        )
        
        # Should handle failures gracefully
        assert isinstance(result, BenchmarkResult)
        # May succeed or fail depending on random outcomes, but should not crash
"""
Integration tests for HybridQuantumClassicalPipeline.

Tests the complete hybrid pipeline integration including complexity assessment,
routing decisions, classical/quantum reranking, and performance monitoring.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock

from quantum_rerank.routing.hybrid_pipeline import (
    HybridQuantumClassicalPipeline,
    ClassicalReranker,
    QuantumReranker,
    HybridRerankingResult,
    RerankingResult
)
from quantum_rerank.routing.complexity_assessment_engine import ComplexityAssessmentEngine
from quantum_rerank.routing.routing_decision_engine import RoutingDecisionEngine, RoutingMethod
from quantum_rerank.config.routing_config import HybridPipelineConfig


class TestHybridQuantumClassicalPipeline:
    """Integration tests for the hybrid pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization with default configuration."""
        pipeline = HybridQuantumClassicalPipeline()
        
        assert pipeline.config is not None
        assert pipeline.complexity_engine is not None
        assert pipeline.routing_engine is not None
        assert pipeline.classical_reranker is not None
        assert pipeline.quantum_reranker is not None
        assert pipeline.pipeline_stats['total_queries'] == 0
    
    def test_pipeline_initialization_with_config(self):
        """Test pipeline initialization with custom configuration."""
        config = HybridPipelineConfig(
            default_top_k=5,
            max_total_latency_ms=300.0,
            enable_result_caching=True
        )
        
        pipeline = HybridQuantumClassicalPipeline(config)
        
        assert pipeline.config.default_top_k == 5
        assert pipeline.config.max_total_latency_ms == 300.0
        assert pipeline.config.enable_result_caching is True
        assert pipeline._result_cache is not None
    
    def test_simple_text_reranking_classical(self):
        """Test simple text reranking routed to classical."""
        pipeline = HybridQuantumClassicalPipeline()
        
        # Simple query that should route to classical
        query = {'text': 'headache treatment'}
        candidates = [
            {'text': 'take aspirin for headache relief'},
            {'text': 'rest in a quiet room'},
            {'text': 'drink plenty of water'}
        ]
        
        result = pipeline.rerank(query, candidates, top_k=3)
        
        assert isinstance(result, HybridRerankingResult)
        assert result.success is True
        assert result.routing_decision.method == RoutingMethod.CLASSICAL
        assert len(result.reranked_candidates) == 3
        assert result.processing_time_ms > 0
        assert result.memory_usage_mb > 0
    
    def test_complex_multimodal_reranking_quantum(self):
        """Test complex multimodal reranking routed to quantum."""
        pipeline = HybridQuantumClassicalPipeline()
        
        # Complex multimodal query that should route to quantum
        query = {
            'text': 'pt w/ CP & SOB, r/o MI',  # Noisy medical text
            'clinical_data': {
                'age': 65,
                'symptoms': ['chest pain', 'shortness of breath'],
                'bp': 'unknown',  # Missing data
                'diagnosis': 'rule out myocardial infarction'
            }
        }
        
        candidates = [
            {
                'text': 'cardiac catheterization for chest pain evaluation',
                'clinical_data': {'procedure': 'cardiac_cath', 'urgency': 'high'}
            },
            {
                'text': 'ECG shows ST elevation, possible STEMI',
                'clinical_data': {'test': 'ECG', 'finding': 'ST_elevation'}
            }
        ]
        
        result = pipeline.rerank(query, candidates, top_k=2)
        
        assert isinstance(result, HybridRerankingResult)
        assert result.success is True
        assert result.routing_decision.method in [RoutingMethod.QUANTUM, RoutingMethod.HYBRID]
        assert len(result.reranked_candidates) == 2
        assert result.quantum_advantage_score > 0.0
    
    def test_medium_complexity_hybrid_reranking(self):
        """Test medium complexity query routed to hybrid."""
        config = HybridPipelineConfig(
            hybrid_combination_method="weighted_average",
            hybrid_weight_classical=0.4,
            hybrid_weight_quantum=0.6
        )
        pipeline = HybridQuantumClassicalPipeline(config)
        
        # Medium complexity query
        query = {
            'text': 'patient with diabetes and hypertension',
            'clinical_data': {'conditions': ['diabetes', 'hypertension']}
        }
        
        candidates = [
            {'text': 'metformin for diabetes management'},
            {'text': 'ACE inhibitors for hypertension'},
            {'text': 'lifestyle modifications for both conditions'}
        ]
        
        # Mock routing to force hybrid
        with patch.object(pipeline.routing_engine, 'route_query') as mock_route:
            mock_route.return_value = Mock(
                method=RoutingMethod.HYBRID,
                confidence=0.5,
                complexity_score=0.5,
                complexity_factors={},
                estimated_latency_ms=300.0,
                estimated_memory_mb=1500.0,
                reasoning="hybrid routing test",
                decision_time_ms=5.0,
                quantum_advantage_score=0.4
            )
            
            result = pipeline.rerank(query, candidates, top_k=3)
            
            assert result.routing_decision.method == RoutingMethod.HYBRID
            assert len(result.reranked_candidates) == 3
            assert result.quantum_advantage_score > 0.0
    
    def test_result_caching(self):
        """Test result caching functionality."""
        config = HybridPipelineConfig(enable_result_caching=True)
        pipeline = HybridQuantumClassicalPipeline(config)
        
        query = {'text': 'simple test query'}
        candidates = [{'text': 'test candidate'}]
        
        # First call - should be cache miss
        result1 = pipeline.rerank(query, candidates, top_k=1)
        first_processing_time = result1.processing_time_ms
        
        # Second call - should be cache hit (faster)
        result2 = pipeline.rerank(query, candidates, top_k=1)
        second_processing_time = result2.processing_time_ms
        
        assert result1.success is True
        assert result2.success is True
        assert len(result1.reranked_candidates) == len(result2.reranked_candidates)
        # Cache hit should be faster (though timing can vary)
        assert second_processing_time <= first_processing_time + 10  # Allow some variance
    
    def test_performance_constraints_fallback(self):
        """Test fallback to classical when performance constraints are violated."""
        config = HybridPipelineConfig()
        config.routing_config.latency_threshold_ms = 50.0  # Very low threshold
        
        pipeline = HybridQuantumClassicalPipeline(config)
        
        # High complexity query that would normally go to quantum
        query = {
            'text': 'complex medical emergency case',
            'clinical_data': {'urgency': 'high', 'complexity': 'very_high'}
        }
        candidates = [{'text': 'emergency procedure'}]
        
        result = pipeline.rerank(query, candidates, top_k=1)
        
        # Should fallback to classical due to performance constraints
        assert result.routing_decision.method == RoutingMethod.CLASSICAL
        assert result.routing_decision.fallback_reason is not None
        assert 'latency_exceeded' in result.routing_decision.fallback_reason
    
    def test_batch_processing_performance(self):
        """Test batch processing performance meets targets."""
        pipeline = HybridQuantumClassicalPipeline()
        
        # Create test batch
        test_queries = [
            {'text': f'query {i}'},
            {'text': f'medical query {i}', 'clinical_data': {'test': i}}
        ]
        
        test_candidates = [
            [{'text': f'candidate {j} for query {i}'} for j in range(10)]
            for i in range(len(test_queries))
        ]
        
        # Benchmark the pipeline
        benchmark_results = pipeline.benchmark_pipeline(test_queries, test_candidates)
        
        assert benchmark_results['total_queries'] == len(test_queries)
        assert benchmark_results['success_rate'] > 0.8  # At least 80% success
        assert benchmark_results['avg_processing_time_ms'] > 0
        assert benchmark_results['max_processing_time_ms'] < 1000  # Under 1 second
    
    def test_pipeline_stats_tracking(self):
        """Test pipeline statistics tracking."""
        pipeline = HybridQuantumClassicalPipeline()
        
        # Process several queries
        queries = [
            {'text': 'simple query'},  # Should go to classical
            {'text': 'complex medical emergency', 'clinical_data': {'urgent': True}},  # Should go to quantum
            {'text': 'medium complexity query', 'clinical_data': {'moderate': True}}  # Could go to hybrid
        ]
        
        for query in queries:
            candidates = [{'text': 'test candidate'}]
            pipeline.rerank(query, candidates, top_k=1)
        
        stats = pipeline.get_pipeline_stats()
        
        assert stats['total_queries'] == 3
        assert stats['quantum_queries'] + stats['classical_queries'] + stats['hybrid_queries'] == 3
        assert stats['avg_processing_time_ms'] > 0
        assert stats['avg_memory_usage_mb'] > 0
        assert 'meets_latency_target' in stats
        assert 'meets_memory_target' in stats
    
    def test_error_handling(self):
        """Test error handling in pipeline."""
        pipeline = HybridQuantumClassicalPipeline()
        
        # Mock complexity engine to raise error
        with patch.object(pipeline.complexity_engine, 'assess_complexity', side_effect=Exception("Test error")):
            query = {'text': 'test query'}
            candidates = [{'text': 'test candidate'}]
            
            result = pipeline.rerank(query, candidates, top_k=1)
            
            assert result.success is False
            assert result.error_message == "Test error"
            assert len(result.reranked_candidates) == 0
            assert result.routing_decision.method == RoutingMethod.CLASSICAL  # Fallback
    
    def test_adaptive_learning_integration(self):
        """Test integration with adaptive learning system."""
        pipeline = HybridQuantumClassicalPipeline()
        
        # Process a query
        query = {'text': 'test query for adaptive learning'}
        candidates = [{'text': 'test candidate'}]
        
        result = pipeline.rerank(query, candidates, top_k=1)
        
        # The routing decision should be recorded for adaptive learning
        # This happens automatically in the pipeline
        assert result.success is True
        
        # Check that routing engine has recorded the decision
        routing_stats = pipeline.routing_engine.get_routing_stats()
        assert routing_stats['total_routings'] > 0
    
    def test_cache_size_limit(self):
        """Test cache size limit enforcement."""
        config = HybridPipelineConfig(enable_result_caching=True, cache_size=2)
        pipeline = HybridQuantumClassicalPipeline(config)
        
        # Fill cache beyond limit
        for i in range(5):
            query = {'text': f'query {i}'}
            candidates = [{'text': f'candidate {i}'}]
            pipeline.rerank(query, candidates, top_k=1)
        
        # Cache should not exceed size limit
        assert len(pipeline._result_cache) <= 2
    
    def test_clear_cache_functionality(self):
        """Test cache clearing functionality."""
        config = HybridPipelineConfig(enable_result_caching=True)
        pipeline = HybridQuantumClassicalPipeline(config)
        
        # Add something to cache
        query = {'text': 'test query'}
        candidates = [{'text': 'test candidate'}]
        pipeline.rerank(query, candidates, top_k=1)
        
        assert len(pipeline._result_cache) > 0
        
        # Clear cache
        pipeline.clear_cache()
        
        assert len(pipeline._result_cache) == 0
    
    def test_optimize_for_performance(self):
        """Test performance optimization."""
        pipeline = HybridQuantumClassicalPipeline()
        
        # Process some queries to generate stats
        query = {'text': 'test query'}
        candidates = [{'text': 'test candidate'}]
        pipeline.rerank(query, candidates, top_k=1)
        
        # Check stats exist
        assert pipeline.pipeline_stats['total_queries'] > 0
        
        # Optimize for performance
        pipeline.optimize_for_performance()
        
        # Stats should be reset
        assert pipeline.pipeline_stats['total_queries'] == 0
        assert pipeline.pipeline_stats['avg_processing_time_ms'] == 0.0
    
    def test_weighted_average_fusion(self):
        """Test weighted average fusion method."""
        config = HybridPipelineConfig(
            hybrid_combination_method="weighted_average",
            hybrid_weight_classical=0.3,
            hybrid_weight_quantum=0.7
        )
        pipeline = HybridQuantumClassicalPipeline(config)
        
        # Create mock results
        classical_result = Mock()
        classical_result.reranked_candidates = [
            ({'text': 'candidate 1'}, 0.8, {}),
            ({'text': 'candidate 2'}, 0.6, {})
        ]
        
        quantum_result = Mock()
        quantum_result.reranked_candidates = [
            ({'text': 'candidate 1'}, 0.5, {}),
            ({'text': 'candidate 2'}, 0.9, {})
        ]
        
        # Test fusion
        fused_results = pipeline._weighted_average_fusion(classical_result, quantum_result)
        
        assert len(fused_results) == 2
        
        # Check that weights are applied correctly
        for candidate, score, metadata in fused_results:
            assert 'fusion_method' in metadata
            assert metadata['fusion_method'] == 'weighted_average'
            assert 'classical_weight' in metadata
            assert 'quantum_weight' in metadata
            assert metadata['classical_weight'] == 0.3
            assert metadata['quantum_weight'] == 0.7
    
    def test_rank_fusion(self):
        """Test rank fusion method."""
        config = HybridPipelineConfig(
            hybrid_combination_method="rank_fusion",
            hybrid_weight_classical=0.4,
            hybrid_weight_quantum=0.6
        )
        pipeline = HybridQuantumClassicalPipeline(config)
        
        # Create mock results with different rankings
        classical_result = Mock()
        classical_result.reranked_candidates = [
            ({'text': 'candidate 1'}, 0.9, {}),  # Rank 0
            ({'text': 'candidate 2'}, 0.7, {})   # Rank 1
        ]
        
        quantum_result = Mock()
        quantum_result.reranked_candidates = [
            ({'text': 'candidate 2'}, 0.8, {}),  # Rank 0
            ({'text': 'candidate 1'}, 0.6, {})   # Rank 1
        ]
        
        # Test fusion
        fused_results = pipeline._rank_fusion(classical_result, quantum_result)
        
        assert len(fused_results) == 2
        
        # Check that ranks are used correctly
        for candidate, score, metadata in fused_results:
            assert 'fusion_method' in metadata
            assert metadata['fusion_method'] == 'rank_fusion'
            assert 'classical_rank' in metadata
            assert 'quantum_rank' in metadata
    
    def test_ensemble_fusion(self):
        """Test ensemble fusion method."""
        config = HybridPipelineConfig(hybrid_combination_method="ensemble")
        pipeline = HybridQuantumClassicalPipeline(config)
        
        # Create mock results
        classical_result = Mock()
        classical_result.reranked_candidates = [
            ({'text': 'candidate 1'}, 0.8, {})
        ]
        classical_result.confidence_score = 0.7
        
        quantum_result = Mock()
        quantum_result.reranked_candidates = [
            ({'text': 'candidate 1'}, 0.6, {})
        ]
        quantum_result.confidence_score = 0.8
        
        # Test fusion
        fused_results = pipeline._ensemble_fusion(classical_result, quantum_result)
        
        assert len(fused_results) == 1
        
        # Check that confidence is used in ensemble
        candidate, score, metadata = fused_results[0]
        assert 'fusion_method' in metadata
        assert metadata['fusion_method'] == 'ensemble'
        assert 'confidence_factor' in metadata
        assert 'original_score' in metadata


class TestClassicalReranker:
    """Integration tests for classical reranker."""
    
    def test_classical_text_reranking(self):
        """Test classical text reranking."""
        config = HybridPipelineConfig()
        reranker = ClassicalReranker(config)
        
        query = {'text': 'diabetes treatment'}
        candidates = [
            {'text': 'insulin therapy for diabetes'},
            {'text': 'metformin for type 2 diabetes'},
            {'text': 'diet and exercise for diabetes'}
        ]
        
        result = reranker.rerank(query, candidates, top_k=3)
        
        assert isinstance(result, RerankingResult)
        assert result.success is True
        assert result.method == RoutingMethod.CLASSICAL
        assert len(result.reranked_candidates) == 3
        assert result.processing_time_ms > 0
        assert result.confidence_score > 0.0
    
    def test_classical_multimodal_reranking(self):
        """Test classical multimodal reranking."""
        config = HybridPipelineConfig()
        reranker = ClassicalReranker(config)
        
        query = {
            'text': 'patient with chest pain',
            'clinical_data': {'symptoms': ['chest pain'], 'age': 45}
        }
        
        candidates = [
            {
                'text': 'ECG for chest pain evaluation',
                'clinical_data': {'procedure': 'ECG'}
            },
            {
                'text': 'cardiac enzymes test',
                'clinical_data': {'test': 'enzymes'}
            }
        ]
        
        result = reranker.rerank(query, candidates, top_k=2)
        
        assert result.success is True
        assert len(result.reranked_candidates) == 2
        
        # Check metadata
        for candidate, score, metadata in result.reranked_candidates:
            assert 'similarity_type' in metadata
            assert 'query_modalities' in metadata
            assert 'candidate_modalities' in metadata


class TestQuantumReranker:
    """Integration tests for quantum reranker."""
    
    def test_quantum_text_reranking(self):
        """Test quantum text reranking."""
        config = HybridPipelineConfig()
        reranker = QuantumReranker(config)
        
        query = {'text': 'complex medical case'}
        candidates = [
            {'text': 'differential diagnosis approach'},
            {'text': 'comprehensive evaluation protocol'}
        ]
        
        result = reranker.rerank(query, candidates, top_k=2)
        
        assert isinstance(result, RerankingResult)
        assert result.success is True
        assert result.method == RoutingMethod.QUANTUM
        assert len(result.reranked_candidates) == 2
        assert result.processing_time_ms > 0
        assert result.quantum_advantage_score >= 0.0
    
    def test_quantum_multimodal_reranking(self):
        """Test quantum multimodal reranking."""
        config = HybridPipelineConfig(quantum_enable_multimodal=True)
        reranker = QuantumReranker(config)
        
        query = {
            'text': 'emergency case with multiple symptoms',
            'clinical_data': {'urgency': 'high', 'complexity': 'high'}
        }
        
        candidates = [
            {
                'text': 'immediate intervention protocol',
                'clinical_data': {'priority': 'urgent'}
            }
        ]
        
        result = reranker.rerank(query, candidates, top_k=1)
        
        assert result.success is True
        assert len(result.reranked_candidates) == 1
        assert result.quantum_advantage_score >= 0.0


if __name__ == '__main__':
    pytest.main([__file__])
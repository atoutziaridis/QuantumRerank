"""
Comprehensive tests for QMMR-05 Evaluation Framework.

Tests the complete evaluation pipeline including dataset generation,
quantum advantage assessment, clinical validation, performance optimization,
and comprehensive reporting.
"""

import pytest
import tempfile
import shutil
import os
import json
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from datetime import datetime

from quantum_rerank.config.evaluation_config import (
    MultimodalMedicalEvaluationConfig, ComprehensiveEvaluationConfig,
    DatasetGenerationConfig, QuantumAdvantageConfig, ClinicalValidationConfig,
    PerformanceOptimizationConfig
)
from quantum_rerank.evaluation.multimodal_medical_dataset_generator import (
    MultimodalMedicalDatasetGenerator, MultimodalMedicalDataset,
    MultimodalMedicalQuery, MultimodalMedicalCandidate
)
from quantum_rerank.evaluation.quantum_advantage_assessor import (
    QuantumAdvantageAssessor, QuantumAdvantageReport, QuantumAdvantageMetrics
)
from quantum_rerank.evaluation.clinical_validation_framework import (
    ClinicalValidationFramework, ClinicalValidationReport,
    SafetyAssessment, PrivacyAssessment, ClinicalUtilityAssessment
)
from quantum_rerank.evaluation.performance_optimizer import (
    PerformanceOptimizer, OptimizedSystem, PerformanceMetrics
)
from quantum_rerank.evaluation.comprehensive_evaluation_pipeline import (
    ComprehensiveEvaluationPipeline, ComprehensiveEvaluationReport,
    EvaluationPhaseResult, FinalValidationResult
)


class TestMultimodalMedicalEvaluationConfig:
    """Test evaluation configuration classes."""
    
    def test_multimodal_medical_evaluation_config_defaults(self):
        """Test default configuration values."""
        config = MultimodalMedicalEvaluationConfig()
        
        assert config.min_multimodal_queries == 200
        assert config.min_documents_per_query == 100
        assert config.test_set_size == 1000
        assert config.max_similarity_latency_ms == 150.0
        assert config.max_batch_latency_ms == 1000.0
        assert config.max_memory_usage_gb == 4.0
        assert config.significance_level == 0.05
        assert config.min_quantum_advantage == 0.05
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        config = MultimodalMedicalEvaluationConfig(
            min_multimodal_queries=100,
            test_set_size=500
        )
        # Should not raise exception
        
        # Invalid configuration - test_set_size < min_multimodal_queries
        with pytest.raises(ValueError):
            MultimodalMedicalEvaluationConfig(
                min_multimodal_queries=100,
                test_set_size=50
            )
        
        # Invalid validation split
        with pytest.raises(ValueError):
            MultimodalMedicalEvaluationConfig(validation_split=1.5)
    
    def test_comprehensive_evaluation_config(self):
        """Test comprehensive evaluation configuration."""
        config = ComprehensiveEvaluationConfig()
        
        assert config.evaluation is not None
        assert config.dataset is not None
        assert config.quantum_advantage is not None
        assert config.clinical_validation is not None
        assert config.performance_optimization is not None
        
        # Test validation
        assert config.validate_configuration() is True
        
        # Test serialization
        config_dict = config.to_dict()
        assert 'evaluation' in config_dict
        assert 'dataset' in config_dict
        
        # Test deserialization
        restored_config = ComprehensiveEvaluationConfig.from_dict(config_dict)
        assert restored_config.evaluation.min_multimodal_queries == config.evaluation.min_multimodal_queries


class TestMultimodalMedicalDatasetGenerator:
    """Test multimodal medical dataset generation."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return MultimodalMedicalEvaluationConfig(
            min_multimodal_queries=10,  # Small for testing
            min_documents_per_query=5
        )
    
    @pytest.fixture
    def generator(self, config):
        """Dataset generator fixture."""
        return MultimodalMedicalDatasetGenerator(config)
    
    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert generator.config is not None
        assert generator.text_generator is not None
        assert generator.clinical_generator is not None
        assert generator.image_generator is not None
    
    def test_dataset_generation(self, generator):
        """Test complete dataset generation."""
        dataset = generator.generate_comprehensive_dataset()
        
        assert isinstance(dataset, MultimodalMedicalDataset)
        assert len(dataset.queries) > 0
        assert len(dataset.candidates) > 0
        assert len(dataset.relevance_judgments) > 0
        
        # Check dataset info
        info = dataset.get_info()
        assert 'total_queries' in info
        assert 'modality_distribution' in info
        assert 'specialty_distribution' in info
    
    def test_query_generation_by_type(self, generator):
        """Test query generation for different types."""
        for scenario_type in generator.config.medical_scenarios:
            queries = generator._generate_queries_by_type(scenario_type, 2)
            
            assert len(queries) == 2
            for query in queries:
                assert isinstance(query, MultimodalMedicalQuery)
                assert query.query_type == scenario_type
                assert query.complexity_level in generator.config.complexity_levels
    
    def test_multimodal_query_creation(self, generator):
        """Test multimodal query creation with different modalities."""
        query = MultimodalMedicalQuery(
            id="test_001",
            query_type="diagnostic_inquiry",
            complexity_level="moderate",
            specialty="radiology"
        )
        
        # Add modalities
        enhanced_query = generator._add_modalities_to_query(query, "imaging_interpretation")
        
        # Should have image for imaging interpretation
        assert enhanced_query.image is not None or enhanced_query.text is not None
    
    def test_noise_injection(self, generator):
        """Test noise injection for robustness testing."""
        # Create simple dataset
        dataset = MultimodalMedicalDataset()
        query = MultimodalMedicalQuery(
            id="test_001",
            query_type="diagnostic_inquiry",
            complexity_level="simple",
            specialty="radiology",
            text="Patient with chest pain"
        )
        dataset.add_query(query)
        
        # Apply noise
        noisy_dataset = generator._add_noise_variations(dataset)
        
        assert isinstance(noisy_dataset, MultimodalMedicalDataset)
        assert len(noisy_dataset.queries) == 1
    
    def test_medical_image_generation(self, generator):
        """Test medical image generation."""
        image_gen = generator.image_generator
        
        # Test different modalities
        for modality in ['XR', 'CT', 'MR', 'US']:
            image = image_gen.generate_medical_image(modality, 'chest')
            assert image is not None
            assert image.size == (224, 224)
    
    def test_clinical_data_generation(self, generator):
        """Test clinical data generation."""
        clinical_gen = generator.clinical_generator
        
        clinical_data = clinical_gen.generate_clinical_data('moderate', 'urgent')
        
        assert 'vital_signs' in clinical_data
        assert 'lab_values' in clinical_data
        assert isinstance(clinical_data['vital_signs'], dict)
        assert isinstance(clinical_data['lab_values'], dict)


class TestQuantumAdvantageAssessor:
    """Test quantum advantage assessment."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return MultimodalMedicalEvaluationConfig(
            min_multimodal_queries=10,
            min_documents_per_query=5
        )
    
    @pytest.fixture
    def assessor(self, config):
        """Quantum advantage assessor fixture."""
        return QuantumAdvantageAssessor(config)
    
    @pytest.fixture
    def test_dataset(self, config):
        """Test dataset fixture."""
        generator = MultimodalMedicalDatasetGenerator(config)
        return generator.generate_comprehensive_dataset()
    
    def test_assessor_initialization(self, assessor):
        """Test assessor initialization."""
        assert assessor.config is not None
        assert assessor.quantum_evaluator is not None
        assert assessor.classical_evaluator is not None
        assert assessor.statistical_tester is not None
    
    def test_quantum_advantage_assessment(self, assessor, test_dataset):
        """Test complete quantum advantage assessment."""
        report = assessor.assess_quantum_advantage(test_dataset)
        
        assert isinstance(report, QuantumAdvantageReport)
        assert len(report.complexity_results) > 0
        assert report.overall_advantage is not None
        
        # Check advantage metrics
        advantage = report.overall_advantage
        assert isinstance(advantage, QuantumAdvantageMetrics)
        assert 0 <= advantage.overall_advantage_score() <= 1
    
    def test_complexity_filtering(self, assessor, test_dataset):
        """Test dataset filtering by complexity."""
        for complexity_level in ['simple', 'moderate', 'complex', 'very_complex']:
            filtered_dataset = assessor._filter_by_complexity(test_dataset, complexity_level)
            
            assert isinstance(filtered_dataset, MultimodalMedicalDataset)
            # May be empty for some complexity levels
    
    def test_classical_baseline_evaluation(self, assessor, test_dataset):
        """Test classical baseline evaluation."""
        classical_evaluator = assessor.classical_evaluator
        
        for baseline_name in ['bm25', 'bert', 'clip', 'multimodal_transformer']:
            results = classical_evaluator.evaluate_baseline(baseline_name, test_dataset)
            
            assert isinstance(results, dict)
            assert 'ndcg_at_10' in results
            assert 'avg_latency_ms' in results
            assert 0 <= results['ndcg_at_10'] <= 1
    
    def test_statistical_significance_testing(self, assessor):
        """Test statistical significance testing."""
        statistical_tester = assessor.statistical_tester
        
        # Create test data
        quantum_scores = [0.8, 0.75, 0.82, 0.78, 0.81]
        classical_scores = [0.7, 0.72, 0.74, 0.71, 0.73]
        
        significance_results = statistical_tester.test_significance(
            quantum_scores, classical_scores, 'test_metric'
        )
        
        assert 'p_value' in significance_results
        assert 'effect_size' in significance_results
        assert 'confidence_interval_lower' in significance_results
        assert 'confidence_interval_upper' in significance_results
    
    def test_quantum_advantage_metrics_calculation(self):
        """Test quantum advantage metrics calculation."""
        metrics = QuantumAdvantageMetrics()
        metrics.accuracy_improvement = 0.1
        metrics.entanglement_utilization = 0.5
        metrics.uncertainty_quality = 0.3
        
        overall_score = metrics.overall_advantage_score()
        assert 0 <= overall_score <= 1
    
    def test_advantage_report_generation(self, assessor, test_dataset):
        """Test advantage report generation."""
        report = assessor.assess_quantum_advantage(test_dataset)
        summary = report.generate_summary()
        
        assert 'overall_advantage_score' in summary
        assert 'significant_improvements' in summary
        assert 'statistical_validation' in summary


class TestClinicalValidationFramework:
    """Test clinical validation framework."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return MultimodalMedicalEvaluationConfig()
    
    @pytest.fixture
    def validator(self, config):
        """Clinical validator fixture."""
        return ClinicalValidationFramework(config)
    
    @pytest.fixture
    def mock_system(self):
        """Mock quantum system."""
        system = Mock()
        system.name = "MockQuantumSystem"
        return system
    
    @pytest.fixture
    def test_dataset(self, config):
        """Test dataset fixture."""
        generator = MultimodalMedicalDatasetGenerator(config)
        return generator.generate_comprehensive_dataset()
    
    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator.config is not None
        assert validator.safety_assessor is not None
        assert validator.privacy_checker is not None
        assert validator.expert_panel is not None
    
    def test_clinical_validation(self, validator, mock_system, test_dataset):
        """Test complete clinical validation."""
        validation_report = validator.conduct_clinical_validation(mock_system, test_dataset)
        
        assert isinstance(validation_report, ClinicalValidationReport)
        assert validation_report.safety_assessment is not None
        assert validation_report.privacy_assessment is not None
        assert validation_report.utility_assessment is not None
        assert validation_report.regulatory_assessment is not None
    
    def test_safety_assessment(self, validator, mock_system, test_dataset):
        """Test safety assessment."""
        safety_assessment = validator.safety_assessor.assess_safety(mock_system, test_dataset)
        
        assert isinstance(safety_assessment, SafetyAssessment)
        assert 0 <= safety_assessment.safety_score <= 1
        assert safety_assessment.safety_level is not None
        assert isinstance(safety_assessment.adverse_events_detected, int)
        assert isinstance(safety_assessment.mitigation_recommendations, list)
    
    def test_privacy_assessment(self, validator, mock_system):
        """Test privacy assessment."""
        privacy_assessment = validator.privacy_checker.assess_privacy_compliance(mock_system)
        
        assert isinstance(privacy_assessment, PrivacyAssessment)
        assert 0 <= privacy_assessment.overall_compliance_score() <= 1
        assert privacy_assessment.hipaa_compliance is not None
        assert privacy_assessment.gdpr_compliance is not None
    
    def test_clinical_utility_assessment(self, validator, mock_system, test_dataset):
        """Test clinical utility assessment."""
        utility_assessment = validator._assess_clinical_utility(mock_system, test_dataset)
        
        assert isinstance(utility_assessment, ClinicalUtilityAssessment)
        assert 0 <= utility_assessment.overall_utility_score() <= 1
        assert 0 <= utility_assessment.diagnostic_accuracy <= 1
        assert 0 <= utility_assessment.workflow_integration_score <= 1
    
    def test_expert_panel_validation(self, validator, mock_system, test_dataset):
        """Test expert panel validation."""
        expert_validation = validator.expert_panel.validate_system(mock_system, test_dataset)
        
        assert 'panel_approval' in expert_validation
        assert 'consensus_score' in expert_validation
        assert 'expert_feedback' in expert_validation
        assert isinstance(expert_validation['expert_feedback'], list)
    
    def test_validation_report_summary(self, validator, mock_system, test_dataset):
        """Test validation report summary generation."""
        validation_report = validator.conduct_clinical_validation(mock_system, test_dataset)
        summary = validation_report.generate_summary()
        
        assert 'safety' in summary
        assert 'privacy' in summary
        assert 'utility' in summary
        assert 'regulatory' in summary
        assert 'overall' in summary


class TestPerformanceOptimizer:
    """Test performance optimization."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return MultimodalMedicalEvaluationConfig()
    
    @pytest.fixture
    def optimizer(self, config):
        """Performance optimizer fixture."""
        return PerformanceOptimizer(config)
    
    @pytest.fixture
    def mock_system(self):
        """Mock quantum system."""
        system = Mock()
        system.name = "MockQuantumSystem"
        return system
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.config is not None
        assert optimizer.optimization_targets is not None
        assert len(optimizer.optimizers) > 0
        assert optimizer.performance_measurer is not None
    
    def test_system_optimization(self, optimizer, mock_system):
        """Test complete system optimization."""
        optimized_result = optimizer.optimize_system(mock_system)
        
        assert isinstance(optimized_result, OptimizedSystem)
        assert optimized_result.system is not None
        assert optimized_result.optimization_report is not None
        
        # Check optimization report
        report = optimized_result.optimization_report
        assert report.baseline_performance is not None
        assert report.final_performance is not None
        assert len(report.optimization_steps) > 0
    
    def test_performance_measurement(self, optimizer, mock_system):
        """Test performance measurement."""
        measurer = optimizer.performance_measurer
        metrics = measurer.measure_performance(mock_system)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.avg_latency_ms >= 0
        assert metrics.avg_memory_usage_mb >= 0
        assert metrics.queries_per_second >= 0
        assert 0 <= metrics.cache_hit_rate <= 1
    
    def test_individual_optimizers(self, optimizer, mock_system):
        """Test individual optimization strategies."""
        for optimizer_name, opt in optimizer.optimizers.items():
            try:
                optimized_system = opt.optimize(mock_system)
                assert optimized_system is not None
            except Exception as e:
                pytest.fail(f"Optimizer {optimizer_name} failed: {e}")
    
    def test_target_validation(self, optimizer):
        """Test performance target validation."""
        # Create test metrics
        metrics = PerformanceMetrics()
        metrics.avg_latency_ms = 90  # Below target
        metrics.avg_memory_usage_mb = 1500  # Below target
        metrics.queries_per_second = 120  # Above target
        
        validation = optimizer._validate_performance_targets(metrics)
        
        assert 'latency_target_met' in validation
        assert 'memory_target_met' in validation
        assert 'throughput_target_met' in validation
        assert validation['latency_target_met'] is True
        assert validation['memory_target_met'] is True
        assert validation['throughput_target_met'] is True
    
    def test_optimization_improvement_calculation(self, optimizer):
        """Test optimization improvement calculation."""
        before = PerformanceMetrics()
        before.avg_latency_ms = 150
        before.avg_memory_usage_mb = 2000
        before.queries_per_second = 80
        
        after = PerformanceMetrics()
        after.avg_latency_ms = 120  # 20% improvement
        after.avg_memory_usage_mb = 1600  # 20% improvement
        after.queries_per_second = 100  # 25% improvement
        
        improvements = optimizer._calculate_improvement(before, after)
        
        assert 'latency_improvement_percent' in improvements
        assert 'memory_improvement_percent' in improvements
        assert 'throughput_improvement_percent' in improvements
        assert improvements['latency_improvement_percent'] == 20.0
        assert improvements['memory_improvement_percent'] == 20.0
        assert improvements['throughput_improvement_percent'] == 25.0


class TestComprehensiveEvaluationPipeline:
    """Test comprehensive evaluation pipeline."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        config = ComprehensiveEvaluationConfig()
        config.evaluation.min_multimodal_queries = 5  # Very small for testing
        config.evaluation.min_documents_per_query = 3
        return config
    
    @pytest.fixture
    def pipeline(self, config):
        """Evaluation pipeline fixture."""
        return ComprehensiveEvaluationPipeline(config)
    
    @pytest.fixture
    def mock_system(self):
        """Mock quantum system."""
        system = Mock()
        system.name = "MockQuantumMultimodalMedicalReranker"
        system.version = "1.0.0"
        return system
    
    @pytest.fixture
    def temp_dir(self):
        """Temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.config is not None
        assert pipeline.dataset_generator is not None
        assert pipeline.quantum_advantage_assessor is not None
        assert pipeline.clinical_validator is not None
        assert pipeline.performance_optimizer is not None
        assert pipeline.report_generator is not None
    
    def test_comprehensive_evaluation(self, pipeline, mock_system):
        """Test complete evaluation pipeline."""
        evaluation_report = pipeline.run_comprehensive_evaluation(mock_system)
        
        assert isinstance(evaluation_report, ComprehensiveEvaluationReport)
        assert evaluation_report.evaluation_id is not None
        assert evaluation_report.evaluation_timestamp is not None
        assert len(evaluation_report.phase_results) > 0
        assert evaluation_report.overall_evaluation_score >= 0
        assert evaluation_report.system_readiness_level in ["not_ready", "pilot_ready", "production_ready"]
    
    def test_phase_execution(self, pipeline):
        """Test individual phase execution."""
        # Test dataset generation phase
        dataset_result = pipeline._phase_dataset_generation()
        
        assert 'dataset' in dataset_result
        assert 'dataset_info' in dataset_result
        assert isinstance(dataset_result['dataset'], MultimodalMedicalDataset)
    
    def test_evaluation_report_generation(self, pipeline, mock_system):
        """Test evaluation report generation."""
        evaluation_report = pipeline.run_comprehensive_evaluation(mock_system)
        
        # Test summary generation
        summary = evaluation_report.generate_summary()
        
        assert 'evaluation_id' in summary
        assert 'overall_score' in summary
        assert 'readiness_level' in summary
        assert 'quantum_advantage' in summary
        assert 'clinical_validation' in summary
        assert 'performance' in summary
    
    def test_report_saving(self, pipeline, mock_system, temp_dir):
        """Test evaluation report saving."""
        evaluation_report = pipeline.run_comprehensive_evaluation(mock_system)
        
        # Save reports
        file_paths = pipeline.save_evaluation_report(evaluation_report, temp_dir)
        
        assert 'json' in file_paths
        assert 'markdown' in file_paths
        assert 'html' in file_paths
        
        # Check files exist
        for format_name, file_path in file_paths.items():
            assert os.path.exists(file_path)
            assert os.path.getsize(file_path) > 0
    
    def test_final_validation(self, pipeline, mock_system):
        """Test final validation phase."""
        # Create minimal dataset for testing
        dataset = MultimodalMedicalDataset()
        
        validation_result = pipeline._phase_final_validation(mock_system, dataset)
        
        assert 'validation' in validation_result
        validation = validation_result['validation']
        
        assert isinstance(validation, FinalValidationResult)
        assert 0 <= validation.readiness_score <= 1
        assert validation.deployment_recommendation is not None
    
    def test_success_criteria_determination(self, pipeline):
        """Test success criteria determination."""
        # Create mock evaluation report
        evaluation_report = ComprehensiveEvaluationReport(
            evaluation_id="test_001",
            evaluation_timestamp=datetime.now(),
            config={}
        )
        
        # Add mock quantum advantage report
        quantum_advantage = QuantumAdvantageMetrics()
        quantum_advantage.accuracy_improvement = 0.1  # 10% improvement
        quantum_advantage.p_value = 0.03  # Significant
        
        mock_quantum_report = Mock()
        mock_quantum_report.overall_advantage = quantum_advantage
        evaluation_report.quantum_advantage_report = mock_quantum_report
        
        criteria = pipeline._determine_success_criteria(evaluation_report)
        
        assert 'quantum_advantage_demonstrated' in criteria
        assert 'statistically_significant' in criteria
        assert criteria['quantum_advantage_demonstrated'] is True
        assert criteria['statistically_significant'] is True
    
    def test_recommendation_generation(self, pipeline):
        """Test recommendation generation."""
        # Create mock evaluation report
        evaluation_report = ComprehensiveEvaluationReport(
            evaluation_id="test_001",
            evaluation_timestamp=datetime.now(),
            config={}
        )
        evaluation_report.system_readiness_level = "pilot_ready"
        
        recommendations = pipeline._generate_comprehensive_recommendations(evaluation_report)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("pilot deployment" in rec.lower() for rec in recommendations)
    
    def test_evaluation_phase_error_handling(self, pipeline):
        """Test error handling in evaluation phases."""
        evaluation_report = ComprehensiveEvaluationReport(
            evaluation_id="test_001",
            evaluation_timestamp=datetime.now(),
            config={}
        )
        
        # Test phase that raises exception
        def failing_phase():
            raise ValueError("Test error")
        
        phase_result = pipeline._execute_phase(
            "Test Failing Phase",
            failing_phase,
            evaluation_report
        )
        
        assert phase_result.success is False
        assert phase_result.error_message == "Test error"
        assert len(evaluation_report.phase_results) == 1
    
    def test_report_format_generation(self, pipeline, mock_system):
        """Test different report format generation."""
        evaluation_report = pipeline.run_comprehensive_evaluation(mock_system)
        
        # Generate reports
        reports = pipeline.report_generator.generate_report(evaluation_report)
        
        assert 'json' in reports
        assert 'markdown' in reports
        assert 'html' in reports
        
        # Test JSON report
        json_report = json.loads(reports['json'])
        assert 'evaluation_metadata' in json_report
        assert 'summary' in json_report
        
        # Test markdown report
        markdown_report = reports['markdown']
        assert '# Quantum Multimodal Medical Reranker Evaluation Report' in markdown_report
        assert '## Executive Summary' in markdown_report
        
        # Test HTML report
        html_report = reports['html']
        assert '<html>' in html_report
        assert '<title>QMMR Evaluation Report</title>' in html_report


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""
    
    @pytest.fixture
    def full_config(self):
        """Full configuration for integration testing."""
        return ComprehensiveEvaluationConfig()
    
    @pytest.fixture
    def minimal_config(self):
        """Minimal configuration for quick testing."""
        config = ComprehensiveEvaluationConfig()
        config.evaluation.min_multimodal_queries = 3
        config.evaluation.min_documents_per_query = 2
        return config
    
    def test_end_to_end_evaluation_flow(self, minimal_config):
        """Test complete end-to-end evaluation flow."""
        # Create mock system
        class MockQuantumSystem:
            def __init__(self):
                self.name = "TestQuantumSystem"
                self.capabilities = ["multimodal", "quantum"]
        
        system = MockQuantumSystem()
        
        # Run evaluation
        pipeline = ComprehensiveEvaluationPipeline(minimal_config)
        evaluation_report = pipeline.run_comprehensive_evaluation(system)
        
        # Verify complete evaluation
        assert evaluation_report.dataset_info is not None
        assert evaluation_report.overall_evaluation_score >= 0
        assert len(evaluation_report.phase_results) >= 5  # At least 5 phases
        assert evaluation_report.system_readiness_level in ["not_ready", "pilot_ready", "production_ready"]
    
    def test_evaluation_with_partial_failures(self, minimal_config):
        """Test evaluation resilience with partial component failures."""
        # Create system that will cause some failures
        system = Mock()
        system.name = "FailureProneSystem"
        
        # Mock failures in quantum advantage assessment
        with patch('quantum_rerank.evaluation.quantum_advantage_assessor.QuantumAdvantageAssessor.assess_quantum_advantage') as mock_qa:
            mock_qa.side_effect = Exception("Quantum assessment failed")
            
            pipeline = ComprehensiveEvaluationPipeline(minimal_config)
            evaluation_report = pipeline.run_comprehensive_evaluation(system)
            
            # Should still complete with partial results
            assert evaluation_report is not None
            assert any(not phase.success for phase in evaluation_report.phase_results)
    
    def test_different_readiness_levels(self, minimal_config):
        """Test evaluation with different system readiness levels."""
        system = Mock()
        system.name = "TestSystem"
        
        pipeline = ComprehensiveEvaluationPipeline(minimal_config)
        
        # Mock different performance levels
        with patch.object(pipeline, '_phase_final_validation') as mock_validation:
            # Test production ready system
            mock_validation.return_value = {
                'validation': FinalValidationResult(
                    performance_validation={'performance_score': 0.95},
                    accuracy_validation={'accuracy_score': 0.95},
                    clinical_utility_validation={'utility_score': 0.95},
                    production_readiness={'readiness_score': 0.95},
                    overall_validation_passed=True,
                    readiness_score=0.95,
                    deployment_recommendation="Production ready"
                )
            }
            
            evaluation_report = pipeline.run_comprehensive_evaluation(system)
            assert evaluation_report.system_readiness_level == "production_ready"
    
    def test_configuration_impact_on_evaluation(self):
        """Test how different configurations impact evaluation."""
        # Test with strict configuration
        strict_config = ComprehensiveEvaluationConfig()
        strict_config.evaluation.min_quantum_advantage = 0.1  # Higher threshold
        strict_config.clinical_validation.diagnostic_accuracy_threshold = 0.95
        
        # Test with lenient configuration
        lenient_config = ComprehensiveEvaluationConfig()
        lenient_config.evaluation.min_quantum_advantage = 0.02  # Lower threshold
        lenient_config.clinical_validation.diagnostic_accuracy_threshold = 0.85
        
        system = Mock()
        system.name = "TestSystem"
        
        # Both should run without error
        strict_pipeline = ComprehensiveEvaluationPipeline(strict_config)
        lenient_pipeline = ComprehensiveEvaluationPipeline(lenient_config)
        
        # Configurations should affect success criteria
        assert strict_config.evaluation.min_quantum_advantage > lenient_config.evaluation.min_quantum_advantage
        assert strict_config.clinical_validation.diagnostic_accuracy_threshold > lenient_config.clinical_validation.diagnostic_accuracy_threshold
    
    def test_large_dataset_handling(self):
        """Test evaluation with larger dataset."""
        # Create configuration with more data
        large_config = ComprehensiveEvaluationConfig()
        large_config.evaluation.min_multimodal_queries = 50
        large_config.evaluation.min_documents_per_query = 20
        
        system = Mock()
        system.name = "TestSystemLarge"
        
        pipeline = ComprehensiveEvaluationPipeline(large_config)
        
        # Should handle larger dataset without memory issues
        with patch.object(pipeline.dataset_generator, 'generate_comprehensive_dataset') as mock_gen:
            # Mock dataset to avoid actually generating large dataset
            mock_dataset = MultimodalMedicalDataset()
            mock_dataset.metadata = {'total_queries': 50, 'total_candidates': 1000}
            mock_gen.return_value = mock_dataset
            
            evaluation_report = pipeline.run_comprehensive_evaluation(system)
            assert evaluation_report is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
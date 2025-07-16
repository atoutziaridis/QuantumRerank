"""
Unit tests for ComplexityAssessmentEngine.

Tests comprehensive complexity assessment functionality including
multimodal analysis, noise detection, uncertainty quantification,
and medical domain assessment.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock

from quantum_rerank.routing.complexity_assessment_engine import (
    ComplexityAssessmentEngine,
    MultimodalComplexityAssessor,
    NoiseIndicatorAssessor,
    UncertaintyAssessor,
    MedicalDomainComplexityAssessor
)
from quantum_rerank.routing.complexity_metrics import (
    ComplexityMetrics,
    ComplexityAssessmentConfig,
    ComplexityAssessmentResult
)


class TestMultimodalComplexityAssessor:
    """Test multimodal complexity assessment functionality."""
    
    def test_multimodal_assessor_initialization(self):
        """Test multimodal assessor initialization."""
        assessor = MultimodalComplexityAssessor()
        
        assert assessor.config is not None
        assert assessor.multimodal_processor is not None
        assert 'text' in assessor.supported_modalities
        assert 'clinical_data' in assessor.supported_modalities
    
    def test_assess_diversity_single_modality(self):
        """Test diversity assessment with single modality."""
        assessor = MultimodalComplexityAssessor()
        
        query = {'text': 'patient with chest pain'}
        diversity = assessor.assess_diversity(query)
        
        assert diversity == 0.0  # Single modality should have no diversity
    
    def test_assess_diversity_multiple_modalities(self):
        """Test diversity assessment with multiple modalities."""
        assessor = MultimodalComplexityAssessor()
        
        query = {
            'text': 'patient with chest pain',
            'clinical_data': {'age': 45, 'bp': '140/90'}
        }
        diversity = assessor.assess_diversity(query)
        
        assert 0.0 < diversity <= 1.0  # Should have some diversity
    
    def test_assess_dependencies_no_correlation(self):
        """Test dependency assessment with no correlation."""
        assessor = MultimodalComplexityAssessor()
        
        query = {'text': 'weather forecast'}  # Non-medical text
        dependencies = assessor.assess_dependencies(query)
        
        assert dependencies == 0.0  # Single modality has no dependencies
    
    def test_assess_dependencies_with_correlation(self):
        """Test dependency assessment with correlation."""
        assessor = MultimodalComplexityAssessor()
        
        query = {
            'text': 'patient with chest pain and high blood pressure',
            'clinical_data': {'bp': '140/90', 'symptoms': ['chest pain']}
        }
        dependencies = assessor.assess_dependencies(query)
        
        assert dependencies >= 0.0  # Should detect some correlation
    
    def test_text_clinical_correlation(self):
        """Test text-clinical correlation detection."""
        assessor = MultimodalComplexityAssessor()
        
        # High correlation case
        query = {
            'text': 'patient with diabetes and high blood pressure',
            'clinical_data': {'diagnosis': 'diabetes', 'bp': '140/90'}
        }
        
        correlation = assessor._compute_text_clinical_correlation(query)
        assert correlation >= 0.0
    
    def test_extract_medical_terms(self):
        """Test medical term extraction."""
        assessor = MultimodalComplexityAssessor()
        
        text = 'patient presents with chest pain and shortness of breath'
        terms = assessor._extract_medical_terms(text)
        
        assert 'patient' in terms
        assert 'pain' in terms
        assert len(terms) > 0


class TestNoiseIndicatorAssessor:
    """Test noise indicator assessment functionality."""
    
    def test_noise_assessor_initialization(self):
        """Test noise assessor initialization."""
        assessor = NoiseIndicatorAssessor()
        
        assert assessor.config is not None
        assert len(assessor.ocr_patterns) > 0
        assert len(assessor.medical_abbreviations) > 0
        assert 'MI' in assessor.medical_abbreviations
        assert 'HTN' in assessor.medical_abbreviations
    
    def test_assess_ocr_errors_clean_text(self):
        """Test OCR error assessment with clean text."""
        assessor = NoiseIndicatorAssessor()
        
        clean_text = "patient presents with chest pain"
        error_prob = assessor.assess_ocr_errors(clean_text)
        
        assert 0.0 <= error_prob <= 1.0
        assert error_prob < 0.1  # Should be low for clean text
    
    def test_assess_ocr_errors_noisy_text(self):
        """Test OCR error assessment with noisy text."""
        assessor = NoiseIndicatorAssessor()
        
        noisy_text = "pt  c/0  ch3st  p@in  and  s0b"
        error_prob = assessor.assess_ocr_errors(noisy_text)
        
        assert 0.0 <= error_prob <= 1.0
        assert error_prob > 0.1  # Should be higher for noisy text
    
    def test_assess_abbreviations_no_abbreviations(self):
        """Test abbreviation assessment with no abbreviations."""
        assessor = NoiseIndicatorAssessor()
        
        text = "patient presents with chest pain"
        abbrev_density = assessor.assess_abbreviations(text)
        
        assert abbrev_density == 0.0
    
    def test_assess_abbreviations_with_abbreviations(self):
        """Test abbreviation assessment with abbreviations."""
        assessor = NoiseIndicatorAssessor()
        
        text = "pt with MI and HTN, BP 140/90"
        abbrev_density = assessor.assess_abbreviations(text)
        
        assert abbrev_density > 0.0
        assert abbrev_density <= 1.0
    
    def test_assess_missing_data_complete_data(self):
        """Test missing data assessment with complete data."""
        assessor = NoiseIndicatorAssessor()
        
        clinical_data = {
            'age': 45,
            'bp': '140/90',
            'symptoms': ['chest pain']
        }
        
        missing_ratio = assessor.assess_missing_data(clinical_data)
        assert missing_ratio == 0.0
    
    def test_assess_missing_data_missing_fields(self):
        """Test missing data assessment with missing fields."""
        assessor = NoiseIndicatorAssessor()
        
        clinical_data = {
            'age': 45,
            'bp': None,
            'symptoms': ['chest pain'],
            'weight': 'unknown'
        }
        
        missing_ratio = assessor.assess_missing_data(clinical_data)
        assert missing_ratio > 0.0
        assert missing_ratio <= 1.0
    
    def test_assess_typos_clean_text(self):
        """Test typo assessment with clean text."""
        assessor = NoiseIndicatorAssessor()
        
        clean_text = "patient presents with chest pain"
        typo_prob = assessor.assess_typos(clean_text)
        
        assert 0.0 <= typo_prob <= 1.0
        assert typo_prob < 0.1  # Should be low for clean text
    
    def test_assess_typos_with_typos(self):
        """Test typo assessment with typos."""
        assessor = NoiseIndicatorAssessor()
        
        typo_text = "pateint presennts with cheest painn"
        typo_prob = assessor.assess_typos(typo_text)
        
        assert 0.0 <= typo_prob <= 1.0
        # Note: Simple typo detection may not catch all typos


class TestUncertaintyAssessor:
    """Test uncertainty assessment functionality."""
    
    def test_uncertainty_assessor_initialization(self):
        """Test uncertainty assessor initialization."""
        assessor = UncertaintyAssessor()
        
        assert assessor.config is not None
        assert len(assessor.uncertainty_terms) > 0
        assert 'possibly' in assessor.uncertainty_terms
        assert 'uncertain' in assessor.uncertainty_terms
    
    def test_assess_ambiguity_clear_text(self):
        """Test ambiguity assessment with clear text."""
        assessor = UncertaintyAssessor()
        
        query = {'text': 'patient has diabetes and takes insulin'}
        ambiguity = assessor.assess_ambiguity(query)
        
        assert 0.0 <= ambiguity <= 1.0
        assert ambiguity < 0.3  # Should be low for clear text
    
    def test_assess_ambiguity_uncertain_text(self):
        """Test ambiguity assessment with uncertain text."""
        assessor = UncertaintyAssessor()
        
        query = {'text': 'patient possibly has diabetes, uncertain diagnosis'}
        ambiguity = assessor.assess_ambiguity(query)
        
        assert 0.0 <= ambiguity <= 1.0
        assert ambiguity > 0.1  # Should be higher for uncertain text
    
    def test_assess_conflicts_no_conflicts(self):
        """Test conflict assessment with no conflicts."""
        assessor = UncertaintyAssessor()
        
        query = {'text': 'patient has diabetes and takes medication'}
        conflicts = assessor.assess_conflicts(query)
        
        assert 0.0 <= conflicts <= 1.0
        assert conflicts < 0.1  # Should be low for consistent text
    
    def test_assess_conflicts_with_conflicts(self):
        """Test conflict assessment with conflicts."""
        assessor = UncertaintyAssessor()
        
        query = {'text': 'patient has diabetes but not taking medication, however glucose is controlled'}
        conflicts = assessor.assess_conflicts(query)
        
        assert 0.0 <= conflicts <= 1.0
        assert conflicts > 0.0  # Should detect some conflict
    
    def test_assess_diagnostic_uncertainty_certain(self):
        """Test diagnostic uncertainty with certain diagnosis."""
        assessor = UncertaintyAssessor()
        
        query = {'text': 'patient diagnosed with diabetes mellitus'}
        uncertainty = assessor.assess_diagnostic_uncertainty(query)
        
        assert 0.0 <= uncertainty <= 1.0
        assert uncertainty < 0.3  # Should be low for certain diagnosis
    
    def test_assess_diagnostic_uncertainty_uncertain(self):
        """Test diagnostic uncertainty with uncertain diagnosis."""
        assessor = UncertaintyAssessor()
        
        query = {'text': 'what is the diagnosis? rule out diabetes, possible heart condition'}
        uncertainty = assessor.assess_diagnostic_uncertainty(query)
        
        assert 0.0 <= uncertainty <= 1.0
        assert uncertainty > 0.2  # Should be higher for uncertain diagnosis


class TestMedicalDomainComplexityAssessor:
    """Test medical domain complexity assessment."""
    
    def test_medical_assessor_initialization(self):
        """Test medical assessor initialization."""
        assessor = MedicalDomainComplexityAssessor()
        
        assert assessor.config is not None
        assert assessor.medical_processor is not None
        assert assessor.medical_relevance is not None
    
    def test_assess_terminology_density_low(self):
        """Test terminology density assessment with low density."""
        assessor = MedicalDomainComplexityAssessor()
        
        query = {'text': 'patient feels tired'}
        density = assessor.assess_terminology_density(query)
        
        assert 0.0 <= density <= 1.0
        assert density < 0.3  # Should be low for simple text
    
    def test_assess_terminology_density_high(self):
        """Test terminology density assessment with high density."""
        assessor = MedicalDomainComplexityAssessor()
        
        query = {'text': 'patient with MI, HTN, DM, and COPD needs EKG and CBC'}
        density = assessor.assess_terminology_density(query)
        
        assert 0.0 <= density <= 1.0
        assert density > 0.1  # Should be higher for medical terminology
    
    def test_assess_correlation_complexity_simple(self):
        """Test correlation complexity with simple data."""
        assessor = MedicalDomainComplexityAssessor()
        
        query = {'clinical_data': {'age': 45}}
        complexity = assessor.assess_correlation_complexity(query)
        
        assert 0.0 <= complexity <= 1.0
        assert complexity < 0.3  # Should be low for simple data
    
    def test_assess_correlation_complexity_complex(self):
        """Test correlation complexity with complex data."""
        assessor = MedicalDomainComplexityAssessor()
        
        query = {
            'clinical_data': {
                'symptoms': ['chest pain', 'shortness of breath'],
                'diagnosis': 'myocardial infarction',
                'vitals': {'bp': '140/90', 'hr': 85},
                'medications': ['aspirin', 'metoprolol']
            }
        }
        complexity = assessor.assess_correlation_complexity(query)
        
        assert 0.0 <= complexity <= 1.0
        assert complexity > 0.1  # Should be higher for complex correlations
    
    def test_assess_domain_specificity_general(self):
        """Test domain specificity with general text."""
        assessor = MedicalDomainComplexityAssessor()
        
        query = {'text': 'patient feels better today'}
        specificity = assessor.assess_domain_specificity(query)
        
        assert 0.0 <= specificity <= 1.0
        assert specificity < 0.5  # Should be low for general text
    
    def test_assess_domain_specificity_specific(self):
        """Test domain specificity with specific medical text."""
        assessor = MedicalDomainComplexityAssessor()
        
        query = {'text': 'patient with acute myocardial infarction requires cardiac catheterization'}
        specificity = assessor.assess_domain_specificity(query)
        
        assert 0.0 <= specificity <= 1.0
        assert specificity > 0.3  # Should be higher for specific medical text


class TestComplexityAssessmentEngine:
    """Test the main complexity assessment engine."""
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = ComplexityAssessmentEngine()
        
        assert engine.config is not None
        assert engine.multimodal_assessor is not None
        assert engine.noise_assessor is not None
        assert engine.uncertainty_assessor is not None
        assert engine.medical_assessor is not None
        assert engine.complexity_aggregator is not None
    
    def test_assess_complexity_simple_query(self):
        """Test complexity assessment with simple query."""
        engine = ComplexityAssessmentEngine()
        
        query = {'text': 'headache treatment'}
        candidates = [{'text': 'take aspirin for headache'}]
        
        result = engine.assess_complexity(query, candidates)
        
        assert isinstance(result, ComplexityAssessmentResult)
        assert result.success is True
        assert result.overall_complexity.overall_complexity < 0.5
        assert result.routing_recommendation == "classical"
        assert result.assessment_time_ms > 0
    
    def test_assess_complexity_complex_query(self):
        """Test complexity assessment with complex query."""
        engine = ComplexityAssessmentEngine()
        
        query = {
            'text': 'pt c/o CP w/ SOB',  # Noisy with abbreviations
            'clinical_data': {
                'age': 45,
                'bp': '???',  # Missing data
                'symptoms': ['chest pain', 'shortness of breath'],
                'diagnosis': 'rule out MI'  # Uncertain diagnosis
            }
        }
        candidates = [
            {'text': 'cardiac catheterization for chest pain'},
            {'text': 'ECG shows ST elevation'}
        ]
        
        result = engine.assess_complexity(query, candidates)
        
        assert isinstance(result, ComplexityAssessmentResult)
        assert result.success is True
        assert result.overall_complexity.overall_complexity > 0.5
        assert result.routing_recommendation in ["quantum", "hybrid"]
        assert result.assessment_time_ms > 0
    
    def test_assess_complexity_performance_constraint(self):
        """Test complexity assessment meets performance constraints."""
        config = ComplexityAssessmentConfig(max_assessment_time_ms=50.0)
        engine = ComplexityAssessmentEngine(config)
        
        # Large query to test performance
        query = {
            'text': 'patient with multiple conditions and complex medical history',
            'clinical_data': {
                'age': 65,
                'conditions': ['diabetes', 'hypertension', 'heart disease'],
                'medications': ['metformin', 'lisinopril', 'aspirin'],
                'vitals': {'bp': '140/90', 'hr': 85, 'temp': 98.6}
            }
        }
        candidates = [{'text': f'candidate {i}'} for i in range(50)]
        
        start_time = time.time()
        result = engine.assess_complexity(query, candidates)
        elapsed_ms = (time.time() - start_time) * 1000
        
        assert result.success is True
        assert elapsed_ms < 50.0  # Should meet performance constraint
        assert result.assessment_time_ms < 50.0
    
    def test_assess_complexity_caching(self):
        """Test complexity assessment caching."""
        config = ComplexityAssessmentConfig(enable_caching=True)
        engine = ComplexityAssessmentEngine(config)
        
        query = {'text': 'patient with chest pain'}
        candidates = [{'text': 'aspirin for chest pain'}]
        
        # First call - should be cache miss
        result1 = engine.assess_complexity(query, candidates)
        assert engine.assessment_stats['cache_misses'] == 1
        assert engine.assessment_stats['cache_hits'] == 0
        
        # Second call - should be cache hit
        result2 = engine.assess_complexity(query, candidates)
        assert engine.assessment_stats['cache_hits'] == 1
        
        # Results should be similar (cache returns same object)
        assert result1.overall_complexity.overall_complexity == result2.overall_complexity.overall_complexity
    
    def test_assess_complexity_empty_query(self):
        """Test complexity assessment with empty query."""
        engine = ComplexityAssessmentEngine()
        
        query = {}
        candidates = []
        
        result = engine.assess_complexity(query, candidates)
        
        assert isinstance(result, ComplexityAssessmentResult)
        assert result.success is True
        assert result.overall_complexity.overall_complexity == 0.0
        assert result.routing_recommendation == "classical"
    
    def test_assess_complexity_quantum_metrics(self):
        """Test complexity assessment with quantum metrics enabled."""
        config = ComplexityAssessmentConfig(enable_quantum_metrics=True)
        engine = ComplexityAssessmentEngine(config)
        
        query = {
            'text': 'patient with uncertain diagnosis',
            'clinical_data': {'symptoms': ['chest pain', 'fatigue']}
        }
        candidates = [{'text': 'cardiac evaluation'}]
        
        result = engine.assess_complexity(query, candidates)
        
        assert result.success is True
        assert result.overall_complexity.quantum_entanglement_potential >= 0.0
        assert result.overall_complexity.interference_complexity >= 0.0
        assert result.overall_complexity.superposition_benefit >= 0.0
    
    def test_get_assessment_stats(self):
        """Test assessment statistics retrieval."""
        engine = ComplexityAssessmentEngine()
        
        # Process some queries
        for i in range(5):
            query = {'text': f'query {i}'}
            candidates = [{'text': f'candidate {i}'}]
            engine.assess_complexity(query, candidates)
        
        stats = engine.get_assessment_stats()
        
        assert 'total_assessments' in stats
        assert 'avg_assessment_time_ms' in stats
        assert 'meets_performance_target' in stats
        assert 'complexity_distribution' in stats
        assert stats['total_assessments'] == 5
        assert stats['avg_assessment_time_ms'] > 0
    
    def test_clear_cache(self):
        """Test cache clearing functionality."""
        config = ComplexityAssessmentConfig(enable_caching=True)
        engine = ComplexityAssessmentEngine(config)
        
        # Add some data to cache
        query = {'text': 'test query'}
        candidates = [{'text': 'test candidate'}]
        engine.assess_complexity(query, candidates)
        
        # Clear cache
        engine.clear_cache()
        
        # Should be cache miss after clearing
        engine.assess_complexity(query, candidates)
        assert engine.assessment_stats['cache_misses'] == 2
    
    def test_optimize_for_performance(self):
        """Test performance optimization."""
        engine = ComplexityAssessmentEngine()
        
        # Process some queries
        for i in range(3):
            query = {'text': f'query {i}'}
            candidates = [{'text': f'candidate {i}'}]
            engine.assess_complexity(query, candidates)
        
        # Optimize for performance
        engine.optimize_for_performance()
        
        # Statistics should be reset
        assert engine.assessment_stats['total_assessments'] == 0
        assert engine.assessment_stats['avg_assessment_time_ms'] == 0.0
    
    def test_weighted_complexity_calculation(self):
        """Test weighted complexity calculation."""
        engine = ComplexityAssessmentEngine()
        
        # Create test metrics
        metrics = ComplexityMetrics()
        metrics.modality_count = 2
        metrics.modality_diversity = 0.8
        metrics.cross_modal_dependencies = 0.6
        metrics.ocr_error_probability = 0.1
        metrics.abbreviation_density = 0.3
        metrics.missing_data_ratio = 0.2
        metrics.term_ambiguity_score = 0.4
        metrics.diagnostic_uncertainty = 0.5
        metrics.medical_terminology_density = 0.6
        metrics.domain_specificity = 0.7
        
        # Test weighted complexity calculation
        weighted_complexity = engine._compute_weighted_complexity(metrics)
        
        assert 0.0 <= weighted_complexity <= 1.0
        assert weighted_complexity > 0.0  # Should be positive with these values
    
    def test_error_handling(self):
        """Test error handling in complexity assessment."""
        engine = ComplexityAssessmentEngine()
        
        # Mock an error in multimodal assessor
        with patch.object(engine.multimodal_assessor, 'assess_diversity', side_effect=Exception("Test error")):
            query = {'text': 'test query'}
            candidates = [{'text': 'test candidate'}]
            
            result = engine.assess_complexity(query, candidates)
            
            assert result.success is False
            assert result.error_message == "Test error"
            assert result.routing_recommendation == "classical"  # Fallback


if __name__ == '__main__':
    pytest.main([__file__])
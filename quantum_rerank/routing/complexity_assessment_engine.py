"""
Complexity Assessment Engine for Quantum-Classical Routing.

This module implements comprehensive complexity assessment for multimodal medical queries
and candidates to determine optimal routing between classical and quantum rerankers.

Based on:
- QMMR-02 task specification
- Quantum research insights (contextual search, geometric similarity, graph walks)
- Medical domain complexity characteristics
- Performance constraints (<50ms assessment time)
"""

import re
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import math

from .complexity_metrics import (
    ComplexityMetrics, 
    ComplexityAssessmentResult, 
    ComplexityAssessmentConfig,
    ComplexityAggregator
)
from ..core.medical_domain_processor import MedicalDomainProcessor
from ..core.multimodal_embedding_processor import MultimodalEmbeddingProcessor
from ..evaluation.medical_relevance import MedicalRelevanceJudgments
from ..config.multimodal_config import MultimodalMedicalConfig

logger = logging.getLogger(__name__)


class MultimodalComplexityAssessor:
    """Assesses complexity related to multimodal data integration."""
    
    def __init__(self, config: ComplexityAssessmentConfig = None):
        self.config = config or ComplexityAssessmentConfig()
        self.multimodal_processor = MultimodalEmbeddingProcessor()
        
        # Supported modalities
        self.supported_modalities = ['text', 'clinical_data', 'image', 'audio', 'structured_data']
        
        logger.debug("MultimodalComplexityAssessor initialized")
    
    def assess_diversity(self, query: Dict[str, Any]) -> float:
        """Assess diversity of modalities in query."""
        present_modalities = [k for k in query.keys() if k in self.supported_modalities and query[k]]
        
        if len(present_modalities) <= 1:
            return 0.0
        
        # Compute entropy of modality presence
        modality_weights = self._compute_modality_weights(query, present_modalities)
        
        # Shannon entropy normalized to [0, 1]
        entropy = -sum(w * math.log2(w) for w in modality_weights if w > 0)
        max_entropy = math.log2(len(self.supported_modalities))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def assess_dependencies(self, query: Dict[str, Any]) -> float:
        """Assess cross-modal dependencies in query."""
        present_modalities = [k for k in query.keys() if k in self.supported_modalities and query[k]]
        
        if len(present_modalities) <= 1:
            return 0.0
        
        # Analyze semantic correlations between modalities
        correlations = []
        
        # Text-Clinical correlation
        if 'text' in present_modalities and 'clinical_data' in present_modalities:
            text_clinical_corr = self._compute_text_clinical_correlation(query)
            correlations.append(text_clinical_corr)
        
        # Text-Image correlation (if available)
        if 'text' in present_modalities and 'image' in present_modalities:
            text_image_corr = self._compute_text_image_correlation(query)
            correlations.append(text_image_corr)
        
        # Clinical-Image correlation (if available)  
        if 'clinical_data' in present_modalities and 'image' in present_modalities:
            clinical_image_corr = self._compute_clinical_image_correlation(query)
            correlations.append(clinical_image_corr)
        
        # Average correlation strength
        return np.mean(correlations) if correlations else 0.0
    
    def _compute_modality_weights(self, query: Dict[str, Any], modalities: List[str]) -> List[float]:
        """Compute relative weights of modalities based on information content."""
        weights = []
        
        for modality in modalities:
            if modality == 'text':
                # Text weight based on length and complexity
                text_content = query.get('text', '')
                weight = min(len(text_content.split()) / 100.0, 1.0)  # Normalize by 100 words
            elif modality == 'clinical_data':
                # Clinical data weight based on number of fields
                clinical_data = query.get('clinical_data', {})
                if isinstance(clinical_data, dict):
                    weight = min(len(clinical_data) / 10.0, 1.0)  # Normalize by 10 fields
                else:
                    weight = 0.5
            elif modality == 'image':
                # Image weight (constant for now, could be based on image complexity)
                weight = 0.8
            else:
                weight = 0.5  # Default weight
            
            weights.append(weight)
        
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        return [w / total_weight for w in weights] if total_weight > 0 else [1.0 / len(weights)] * len(weights)
    
    def _compute_text_clinical_correlation(self, query: Dict[str, Any]) -> float:
        """Compute correlation between text and clinical data."""
        text = query.get('text', '')
        clinical_data = query.get('clinical_data', {})
        
        if not text or not clinical_data:
            return 0.0
        
        # Extract medical terms from text
        medical_terms = self._extract_medical_terms(text)
        
        # Extract clinical terms from clinical data
        clinical_terms = self._extract_clinical_terms(clinical_data)
        
        # Compute overlap
        if not medical_terms or not clinical_terms:
            return 0.0
        
        overlap = len(set(medical_terms) & set(clinical_terms))
        total_terms = len(set(medical_terms) | set(clinical_terms))
        
        return overlap / total_terms if total_terms > 0 else 0.0
    
    def _compute_text_image_correlation(self, query: Dict[str, Any]) -> float:
        """Compute correlation between text and image data."""
        # Placeholder for image-text correlation
        # In practice, this would use vision-language models
        return 0.5  # Moderate correlation assumed
    
    def _compute_clinical_image_correlation(self, query: Dict[str, Any]) -> float:
        """Compute correlation between clinical data and image."""
        # Placeholder for clinical-image correlation
        # In practice, this would analyze medical images with clinical context
        return 0.6  # Higher correlation assumed for medical images
    
    def _extract_medical_terms(self, text: str) -> List[str]:
        """Extract medical terms from text."""
        # Simple medical term extraction (could be enhanced with NER)
        medical_keywords = [
            'pain', 'symptom', 'diagnosis', 'treatment', 'patient', 'medical', 'clinical',
            'disease', 'condition', 'therapy', 'medication', 'procedure', 'examination'
        ]
        
        text_lower = text.lower()
        found_terms = [term for term in medical_keywords if term in text_lower]
        
        return found_terms
    
    def _extract_clinical_terms(self, clinical_data: Dict[str, Any]) -> List[str]:
        """Extract clinical terms from clinical data."""
        terms = []
        
        if isinstance(clinical_data, dict):
            for key, value in clinical_data.items():
                terms.append(key.lower())
                if isinstance(value, str):
                    terms.extend(value.lower().split())
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            terms.extend(item.lower().split())
        
        return terms


class NoiseIndicatorAssessor:
    """Assesses noise indicators in queries and documents."""
    
    def __init__(self, config: ComplexityAssessmentConfig = None):
        self.config = config or ComplexityAssessmentConfig()
        
        # OCR error patterns
        self.ocr_patterns = [
            r'\b[a-z]{1,2}\b',  # Very short words (often OCR errors)
            r'[0-9][a-z]|[a-z][0-9]',  # Mixed alphanumeric
            r'[^a-zA-Z0-9\s.,!?;:(){}[\]"\'`~@#$%^&*+=<>/-]',  # Unusual characters
            r'\s{2,}',  # Multiple spaces
            r'[A-Z]{3,}',  # Long sequences of capitals
        ]
        
        # Medical abbreviations (from medical_domain_processor)
        self.medical_abbreviations = {
            'MI', 'CAD', 'CHF', 'HTN', 'BP', 'HR', 'ECG', 'EKG', 'COPD', 'SOB',
            'DOE', 'URI', 'PNA', 'TB', 'RR', 'O2', 'DM', 'T1DM', 'T2DM', 'BG',
            'BS', 'HbA1c', 'CVA', 'TIA', 'ICP', 'LOC', 'GCS', 'CNS', 'PNS',
            'GI', 'UTI', 'BUN', 'Cr', 'CBC', 'WBC', 'RBC', 'PLT', 'PT', 'PTT'
        }
        
        logger.debug("NoiseIndicatorAssessor initialized")
    
    def assess_ocr_errors(self, text: str) -> float:
        """Assess probability of OCR errors in text."""
        if not text:
            return 0.0
        
        total_score = 0.0
        text_length = len(text)
        
        for pattern in self.ocr_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            pattern_score = len(matches) / text_length if text_length > 0 else 0.0
            total_score += pattern_score
        
        # Normalize to [0, 1] range
        return min(total_score, 1.0)
    
    def assess_abbreviations(self, text: str) -> float:
        """Assess density of medical abbreviations in text."""
        if not text:
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0
        
        abbreviation_count = sum(1 for word in words if word.strip('.,!?;:()') in self.medical_abbreviations)
        
        return abbreviation_count / len(words)
    
    def assess_missing_data(self, clinical_data: Dict[str, Any]) -> float:
        """Assess missing data ratio in clinical data."""
        if not clinical_data or not isinstance(clinical_data, dict):
            return 1.0  # Complete missing data
        
        total_fields = 0
        missing_fields = 0
        
        for key, value in clinical_data.items():
            total_fields += 1
            
            if value is None:
                missing_fields += 1
            elif isinstance(value, str) and (not value.strip() or value.lower() in ['unknown', 'n/a', 'null', '???']):
                missing_fields += 1
            elif isinstance(value, (list, dict)) and not value:
                missing_fields += 1
        
        return missing_fields / total_fields if total_fields > 0 else 0.0
    
    def assess_typos(self, text: str) -> float:
        """Assess probability of typos in text."""
        if not text:
            return 0.0
        
        # Simple typo detection heuristics
        words = text.split()
        if not words:
            return 0.0
        
        typo_indicators = 0
        
        for word in words:
            clean_word = word.strip('.,!?;:()').lower()
            
            # Check for common typo patterns
            if len(clean_word) > 3:
                # Repeated characters (like 'goood' instead of 'good')
                if re.search(r'(.)\1{2,}', clean_word):
                    typo_indicators += 1
                
                # Missing vowels in long words
                if len(clean_word) > 6 and not re.search(r'[aeiou]', clean_word):
                    typo_indicators += 1
                
                # Unusual character combinations
                if re.search(r'[qwxz]{2,}', clean_word):
                    typo_indicators += 1
        
        return typo_indicators / len(words)


class UncertaintyAssessor:
    """Assesses uncertainty markers in queries and documents."""
    
    def __init__(self, config: ComplexityAssessmentConfig = None):
        self.config = config or ComplexityAssessmentConfig()
        
        # Uncertainty indicators
        self.uncertainty_terms = [
            'possibly', 'maybe', 'uncertain', 'unclear', 'ambiguous', 'doubtful',
            'suspected', 'potential', 'likely', 'probable', 'appears', 'seems',
            'suggests', 'indicates', 'consistent with', 'rule out', 'differential'
        ]
        
        # Conflicting indicators
        self.conflict_patterns = [
            r'but\s+not',
            r'however\s+.*\s+not',
            r'although\s+.*\s+but',
            r'despite\s+.*\s+still',
            r'contradicts?',
            r'conflicts?\s+with',
            r'inconsistent\s+with'
        ]
        
        logger.debug("UncertaintyAssessor initialized")
    
    def assess_ambiguity(self, query: Dict[str, Any]) -> float:
        """Assess term ambiguity in query."""
        ambiguity_score = 0.0
        
        # Text ambiguity
        if 'text' in query:
            text_ambiguity = self._assess_text_ambiguity(query['text'])
            ambiguity_score += text_ambiguity * 0.6
        
        # Clinical data ambiguity
        if 'clinical_data' in query:
            clinical_ambiguity = self._assess_clinical_ambiguity(query['clinical_data'])
            ambiguity_score += clinical_ambiguity * 0.4
        
        return min(ambiguity_score, 1.0)
    
    def assess_conflicts(self, query: Dict[str, Any]) -> float:
        """Assess conflicting information in query."""
        conflict_score = 0.0
        
        # Text conflicts
        if 'text' in query:
            text_conflicts = self._assess_text_conflicts(query['text'])
            conflict_score += text_conflicts * 0.7
        
        # Clinical data conflicts
        if 'clinical_data' in query:
            clinical_conflicts = self._assess_clinical_conflicts(query['clinical_data'])
            conflict_score += clinical_conflicts * 0.3
        
        return min(conflict_score, 1.0)
    
    def assess_diagnostic_uncertainty(self, query: Dict[str, Any]) -> float:
        """Assess diagnostic uncertainty in query."""
        uncertainty_score = 0.0
        
        # Look for diagnostic uncertainty markers
        if 'text' in query:
            text = query['text'].lower()
            
            # Count uncertainty terms
            uncertainty_count = sum(1 for term in self.uncertainty_terms if term in text)
            uncertainty_score += uncertainty_count * 0.1
            
            # Look for differential diagnosis indicators
            if 'differential' in text or 'rule out' in text:
                uncertainty_score += 0.3
            
            # Look for question marks or uncertain phrasing
            if '?' in text or 'what is' in text or 'could be' in text:
                uncertainty_score += 0.2
        
        return min(uncertainty_score, 1.0)
    
    def _assess_text_ambiguity(self, text: str) -> float:
        """Assess ambiguity in text."""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        ambiguity_count = sum(1 for term in self.uncertainty_terms if term in text_lower)
        
        # Normalize by text length
        words = text.split()
        return ambiguity_count / len(words) if words else 0.0
    
    def _assess_clinical_ambiguity(self, clinical_data: Dict[str, Any]) -> float:
        """Assess ambiguity in clinical data."""
        if not clinical_data or not isinstance(clinical_data, dict):
            return 0.0
        
        ambiguity_score = 0.0
        total_fields = 0
        
        for key, value in clinical_data.items():
            total_fields += 1
            
            if isinstance(value, str):
                value_lower = value.lower()
                if any(term in value_lower for term in self.uncertainty_terms):
                    ambiguity_score += 1
                
                # Check for vague values
                if value_lower in ['unknown', 'variable', 'varies', 'unclear', 'tbd']:
                    ambiguity_score += 1
        
        return ambiguity_score / total_fields if total_fields > 0 else 0.0
    
    def _assess_text_conflicts(self, text: str) -> float:
        """Assess conflicting information in text."""
        if not text:
            return 0.0
        
        conflict_count = sum(1 for pattern in self.conflict_patterns if re.search(pattern, text, re.IGNORECASE))
        
        # Normalize by text length
        sentences = text.split('.')
        return conflict_count / len(sentences) if sentences else 0.0
    
    def _assess_clinical_conflicts(self, clinical_data: Dict[str, Any]) -> float:
        """Assess conflicting information in clinical data."""
        if not clinical_data or not isinstance(clinical_data, dict):
            return 0.0
        
        # Simple conflict detection (could be enhanced)
        conflicts = 0
        
        # Check for contradictory values
        if 'diagnosis' in clinical_data and 'symptoms' in clinical_data:
            # Placeholder for more sophisticated conflict detection
            pass
        
        return min(conflicts * 0.1, 1.0)


class MedicalDomainComplexityAssessor:
    """Assesses medical domain-specific complexity."""
    
    def __init__(self, config: ComplexityAssessmentConfig = None):
        self.config = config or ComplexityAssessmentConfig()
        self.medical_processor = MedicalDomainProcessor()
        self.medical_relevance = MedicalRelevanceJudgments()
        
        logger.debug("MedicalDomainComplexityAssessor initialized")
    
    def assess_terminology_density(self, query: Dict[str, Any]) -> float:
        """Assess density of medical terminology."""
        terminology_score = 0.0
        
        # Process text component
        if 'text' in query:
            text_result = self.medical_processor.process_medical_query({'text': query['text']})
            
            # Count expanded abbreviations
            abbreviations = text_result.get('expanded_abbreviations', {})
            terminology_score += len(abbreviations) * 0.1
            
            # Count extracted entities
            entities = text_result.get('extracted_entities', [])
            medical_entities = [e for e in entities if e.get('type') in ['DISEASE', 'MEDICATION', 'PROCEDURE']]
            terminology_score += len(medical_entities) * 0.05
        
        # Process clinical data
        if 'clinical_data' in query:
            clinical_result = self.medical_processor.process_medical_query({'clinical_data': query['clinical_data']})
            
            # Count clinical entities
            clinical_entities = clinical_result.get('clinical_entities', [])
            terminology_score += len(clinical_entities) * 0.05
        
        return min(terminology_score, 1.0)
    
    def assess_correlation_complexity(self, query: Dict[str, Any]) -> float:
        """Assess complexity of clinical correlations."""
        correlation_score = 0.0
        
        # Analyze clinical data correlations
        if 'clinical_data' in query:
            clinical_data = query['clinical_data']
            
            if isinstance(clinical_data, dict):
                # Count interrelated fields
                related_fields = 0
                
                # Check for common medical relationships
                if 'symptoms' in clinical_data and 'diagnosis' in clinical_data:
                    related_fields += 1
                
                if 'vitals' in clinical_data and 'medications' in clinical_data:
                    related_fields += 1
                
                if 'lab_results' in clinical_data and 'diagnosis' in clinical_data:
                    related_fields += 1
                
                correlation_score += related_fields * 0.2
        
        # Analyze text-clinical correlations
        if 'text' in query and 'clinical_data' in query:
            # Use multimodal processor to assess correlation
            correlation_score += 0.3  # Placeholder
        
        return min(correlation_score, 1.0)
    
    def assess_domain_specificity(self, query: Dict[str, Any]) -> float:
        """Assess domain specificity of query."""
        specificity_score = 0.0
        
        # Process with medical relevance system
        if 'text' in query:
            domain, confidence = self.medical_relevance.classify_medical_domain(query['text'])
            
            # Higher confidence indicates higher specificity
            specificity_score += confidence * 0.7
            
            # Specific domains get higher scores
            if domain in ['cardiology', 'diabetes', 'neurology', 'oncology']:
                specificity_score += 0.2
        
        return min(specificity_score, 1.0)


class ComplexityAssessmentEngine:
    """
    Comprehensive complexity assessment engine for medical queries and candidates.
    
    Analyzes multimodal medical data to determine routing between classical and quantum rerankers.
    """
    
    def __init__(self, config: ComplexityAssessmentConfig = None):
        self.config = config or ComplexityAssessmentConfig()
        
        # Initialize assessment modules
        self.multimodal_assessor = MultimodalComplexityAssessor(config)
        self.noise_assessor = NoiseIndicatorAssessor(config)
        self.uncertainty_assessor = UncertaintyAssessor(config)
        self.medical_assessor = MedicalDomainComplexityAssessor(config)
        
        # Performance monitoring
        self.assessment_stats = {
            'total_assessments': 0,
            'avg_assessment_time_ms': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Caching system
        self._complexity_cache = {} if config.enable_caching else None
        
        # Complexity aggregator for trend analysis
        self.complexity_aggregator = ComplexityAggregator()
        
        logger.info("ComplexityAssessmentEngine initialized")
    
    def assess_complexity(self, query: Dict[str, Any], candidates: List[Dict[str, Any]]) -> ComplexityAssessmentResult:
        """
        Assess complexity for query and candidates with <50ms constraint.
        
        Args:
            query: Multimodal query data
            candidates: List of candidate documents
            
        Returns:
            ComplexityAssessmentResult with routing recommendation
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(query, candidates)
            if self._complexity_cache and cache_key in self._complexity_cache:
                self.assessment_stats['cache_hits'] += 1
                return self._complexity_cache[cache_key]
            
            self.assessment_stats['cache_misses'] += 1
            
            # Assess query complexity
            query_complexity = self._assess_query_complexity(query)
            
            # Assess candidates complexity (sample if too many)
            max_candidates = 20  # Limit for performance
            candidate_sample = candidates[:max_candidates] if len(candidates) > max_candidates else candidates
            
            candidate_complexities = [
                self._assess_candidate_complexity(candidate) 
                for candidate in candidate_sample
            ]
            
            # Compute overall complexity metrics
            overall_complexity = self._compute_overall_complexity(query_complexity, candidate_complexities)
            
            # Generate routing recommendation
            routing_recommendation, routing_confidence = self._generate_routing_recommendation(overall_complexity)
            
            # Performance tracking
            elapsed = (time.time() - start_time) * 1000
            self._update_assessment_stats(elapsed)
            
            # Create result
            result = ComplexityAssessmentResult(
                query_complexity=query_complexity,
                candidate_complexities=candidate_complexities,
                overall_complexity=overall_complexity,
                routing_recommendation=routing_recommendation,
                routing_confidence=routing_confidence,
                assessment_time_ms=elapsed,
                success=True,
                quantum_advantage_score=self._compute_quantum_advantage_score(overall_complexity)
            )
            
            # Cache result
            if self._complexity_cache:
                self._cache_result(cache_key, result)
            
            # Update aggregator
            self.complexity_aggregator.add_metrics(overall_complexity)
            
            return result
            
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            logger.error(f"Complexity assessment failed: {e}")
            
            # Return fallback result
            return ComplexityAssessmentResult(
                query_complexity=ComplexityMetrics(),
                candidate_complexities=[],
                overall_complexity=ComplexityMetrics(),
                routing_recommendation="classical",  # Safe fallback
                routing_confidence=0.5,
                assessment_time_ms=elapsed,
                success=False,
                error_message=str(e)
            )
    
    def _assess_query_complexity(self, query: Dict[str, Any]) -> ComplexityMetrics:
        """Assess complexity of individual query."""
        metrics = ComplexityMetrics()
        
        # Multimodal complexity
        modalities = [k for k in query.keys() if k in ['text', 'clinical_data', 'image'] and query[k]]
        metrics.modality_count = len(modalities)
        metrics.modality_diversity = self.multimodal_assessor.assess_diversity(query)
        metrics.cross_modal_dependencies = self.multimodal_assessor.assess_dependencies(query)
        
        # Noise indicators
        if 'text' in query:
            metrics.ocr_error_probability = self.noise_assessor.assess_ocr_errors(query['text'])
            metrics.abbreviation_density = self.noise_assessor.assess_abbreviations(query['text'])
        
        if 'clinical_data' in query:
            metrics.missing_data_ratio = self.noise_assessor.assess_missing_data(query['clinical_data'])
        
        # Uncertainty markers
        metrics.term_ambiguity_score = self.uncertainty_assessor.assess_ambiguity(query)
        metrics.conflicting_information = self.uncertainty_assessor.assess_conflicts(query)
        metrics.diagnostic_uncertainty = self.uncertainty_assessor.assess_diagnostic_uncertainty(query)
        
        # Medical domain complexity
        metrics.medical_terminology_density = self.medical_assessor.assess_terminology_density(query)
        metrics.clinical_correlation_complexity = self.medical_assessor.assess_correlation_complexity(query)
        metrics.domain_specificity = self.medical_assessor.assess_domain_specificity(query)
        
        # Quantum-inspired metrics
        if self.config.enable_quantum_metrics:
            metrics.quantum_entanglement_potential = self._compute_entanglement_potential(query)
            metrics.interference_complexity = self._compute_interference_complexity(query)
            metrics.superposition_benefit = self._compute_superposition_benefit(query)
        
        # Compute overall complexity
        metrics.overall_complexity = self._compute_weighted_complexity(metrics)
        
        return metrics
    
    def _assess_candidate_complexity(self, candidate: Dict[str, Any]) -> ComplexityMetrics:
        """Assess complexity of individual candidate."""
        # Similar to query complexity assessment but simplified
        return self._assess_query_complexity(candidate)
    
    def _compute_overall_complexity(self, query_complexity: ComplexityMetrics, 
                                   candidate_complexities: List[ComplexityMetrics]) -> ComplexityMetrics:
        """Compute overall complexity from query and candidates."""
        overall = ComplexityMetrics()
        
        # Start with query complexity
        overall.modality_count = query_complexity.modality_count
        overall.modality_diversity = query_complexity.modality_diversity
        overall.cross_modal_dependencies = query_complexity.cross_modal_dependencies
        overall.ocr_error_probability = query_complexity.ocr_error_probability
        overall.abbreviation_density = query_complexity.abbreviation_density
        overall.missing_data_ratio = query_complexity.missing_data_ratio
        overall.term_ambiguity_score = query_complexity.term_ambiguity_score
        overall.conflicting_information = query_complexity.conflicting_information
        overall.diagnostic_uncertainty = query_complexity.diagnostic_uncertainty
        overall.medical_terminology_density = query_complexity.medical_terminology_density
        overall.clinical_correlation_complexity = query_complexity.clinical_correlation_complexity
        overall.domain_specificity = query_complexity.domain_specificity
        overall.quantum_entanglement_potential = query_complexity.quantum_entanglement_potential
        overall.interference_complexity = query_complexity.interference_complexity
        overall.superposition_benefit = query_complexity.superposition_benefit
        
        # Incorporate candidate complexities
        if candidate_complexities:
            avg_candidate_complexity = np.mean([c.overall_complexity for c in candidate_complexities])
            
            # Weight query complexity higher than candidate complexity
            overall.overall_complexity = (
                0.7 * query_complexity.overall_complexity + 
                0.3 * avg_candidate_complexity
            )
        else:
            overall.overall_complexity = query_complexity.overall_complexity
        
        return overall
    
    def _compute_weighted_complexity(self, metrics: ComplexityMetrics) -> float:
        """Compute weighted overall complexity score."""
        # Multimodal component
        multimodal_score = (
            metrics.modality_count / 3.0 * 0.4 +  # Max 3 modalities
            metrics.modality_diversity * 0.3 +
            metrics.cross_modal_dependencies * 0.3
        )
        
        # Noise component
        noise_score = (
            metrics.ocr_error_probability * 0.3 +
            metrics.abbreviation_density * 0.3 +
            metrics.missing_data_ratio * 0.4
        )
        
        # Uncertainty component
        uncertainty_score = (
            metrics.term_ambiguity_score * 0.4 +
            metrics.conflicting_information * 0.3 +
            metrics.diagnostic_uncertainty * 0.3
        )
        
        # Medical domain component
        medical_score = (
            metrics.medical_terminology_density * 0.3 +
            metrics.clinical_correlation_complexity * 0.4 +
            metrics.domain_specificity * 0.3
        )
        
        # Quantum component (if enabled)
        quantum_score = 0.0
        if self.config.enable_quantum_metrics:
            quantum_score = (
                metrics.quantum_entanglement_potential * 0.4 +
                metrics.interference_complexity * 0.3 +
                metrics.superposition_benefit * 0.3
            )
        
        # Weighted combination
        if self.config.enable_quantum_metrics:
            overall_score = (
                self.config.multimodal_weight * multimodal_score +
                self.config.noise_weight * noise_score +
                self.config.uncertainty_weight * uncertainty_score +
                self.config.medical_domain_weight * medical_score +
                0.1 * quantum_score  # Small quantum component
            )
        else:
            overall_score = (
                self.config.multimodal_weight * multimodal_score +
                self.config.noise_weight * noise_score +
                self.config.uncertainty_weight * uncertainty_score +
                self.config.medical_domain_weight * medical_score
            )
        
        return min(overall_score, 1.0)
    
    def _compute_entanglement_potential(self, query: Dict[str, Any]) -> float:
        """Compute potential for quantum entanglement benefits."""
        # Based on cross-modal dependencies
        if len([k for k in query.keys() if k in ['text', 'clinical_data', 'image'] and query[k]]) > 1:
            return self.multimodal_assessor.assess_dependencies(query)
        return 0.0
    
    def _compute_interference_complexity(self, query: Dict[str, Any]) -> float:
        """Compute complexity suitable for quantum interference."""
        # Based on uncertainty and ambiguity
        return self.uncertainty_assessor.assess_ambiguity(query)
    
    def _compute_superposition_benefit(self, query: Dict[str, Any]) -> float:
        """Compute potential superposition benefits."""
        # Based on multiple possible states/interpretations
        return self.uncertainty_assessor.assess_diagnostic_uncertainty(query)
    
    def _generate_routing_recommendation(self, complexity: ComplexityMetrics) -> Tuple[str, float]:
        """Generate routing recommendation based on complexity."""
        overall_score = complexity.overall_complexity
        
        # Routing thresholds
        if overall_score >= 0.7:
            return "quantum", min(overall_score, 1.0)
        elif overall_score <= 0.3:
            return "classical", 1.0 - overall_score
        else:
            return "hybrid", 0.5
    
    def _compute_quantum_advantage_score(self, complexity: ComplexityMetrics) -> float:
        """Compute estimated quantum advantage score."""
        # Quantum advantages in complex, noisy, uncertain scenarios
        quantum_factors = [
            complexity.cross_modal_dependencies,
            complexity.quantum_entanglement_potential,
            complexity.interference_complexity,
            complexity.superposition_benefit,
            complexity.conflicting_information,
            complexity.diagnostic_uncertainty
        ]
        
        return np.mean(quantum_factors)
    
    def _get_cache_key(self, query: Dict[str, Any], candidates: List[Dict[str, Any]]) -> str:
        """Generate cache key for complexity assessment."""
        import hashlib
        
        # Create deterministic hash of query and candidates
        query_str = str(sorted(query.items()))
        candidates_str = str([sorted(c.items()) for c in candidates[:5]])  # Limit for performance
        
        content = f"{query_str}|||{candidates_str}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _cache_result(self, cache_key: str, result: ComplexityAssessmentResult):
        """Cache complexity assessment result."""
        if len(self._complexity_cache) >= self.config.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._complexity_cache))
            del self._complexity_cache[oldest_key]
        
        self._complexity_cache[cache_key] = result
    
    def _update_assessment_stats(self, elapsed_ms: float):
        """Update assessment performance statistics."""
        self.assessment_stats['total_assessments'] += 1
        
        # Update rolling average
        n = self.assessment_stats['total_assessments']
        current_avg = self.assessment_stats['avg_assessment_time_ms']
        
        self.assessment_stats['avg_assessment_time_ms'] = (
            (current_avg * (n - 1) + elapsed_ms) / n
        )
    
    def get_assessment_stats(self) -> Dict[str, Any]:
        """Get comprehensive assessment statistics."""
        stats = self.assessment_stats.copy()
        
        # Add cache statistics
        if self._complexity_cache:
            total_requests = stats['cache_hits'] + stats['cache_misses']
            stats['cache_hit_rate'] = stats['cache_hits'] / total_requests if total_requests > 0 else 0
            stats['cache_size'] = len(self._complexity_cache)
        
        # Add performance analysis
        stats['meets_performance_target'] = stats['avg_assessment_time_ms'] < self.config.max_assessment_time_ms
        
        # Add complexity distribution
        stats['complexity_distribution'] = self.complexity_aggregator.get_complexity_distribution()
        
        return stats
    
    def clear_cache(self):
        """Clear complexity assessment cache."""
        if self._complexity_cache:
            self._complexity_cache.clear()
            logger.info("Complexity assessment cache cleared")
    
    def optimize_for_performance(self):
        """Optimize engine for performance."""
        # Clear cache to free memory
        self.clear_cache()
        
        # Reset statistics
        self.assessment_stats = {
            'total_assessments': 0,
            'avg_assessment_time_ms': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info("ComplexityAssessmentEngine optimized for performance")
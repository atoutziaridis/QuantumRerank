"""
Multimodal Quantum Similarity Engine for Medical Data.

This module extends the QuantumSimilarityEngine to handle multimodal medical data
with quantum entanglement-based fusion as specified in QMMR-03 task.

Based on:
- QMMR-03 task requirements
- Quantum entanglement for cross-modal relationships
- Performance constraints (<100ms similarity, <500ms batch)
- Medical domain optimization
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Any, Tuple

from .quantum_similarity_engine import QuantumSimilarityEngine, SimilarityEngineConfig
from .multimodal_swap_test import MultimodalSwapTest
from .multimodal_embedding_processor import MultimodalEmbeddingProcessor
from .medical_domain_processor import MedicalDomainProcessor
from ..routing.complexity_assessment_engine import ComplexityAssessmentEngine
from ..config.multimodal_config import MultimodalMedicalConfig

logger = logging.getLogger(__name__)


class MultimodalQuantumSimilarityEngine(QuantumSimilarityEngine):
    """
    Quantum similarity engine for multimodal medical data with entanglement-based fusion.
    
    Extends the base QuantumSimilarityEngine to handle complex multimodal medical scenarios
    where quantum advantages are most pronounced.
    """
    
    def __init__(self, config: SimilarityEngineConfig = None):
        """
        Initialize multimodal quantum similarity engine.
        
        Args:
            config: Similarity engine configuration
        """
        # Initialize base similarity engine
        super().__init__(config)
        
        # Replace components with multimodal versions
        self.embedding_processor = MultimodalEmbeddingProcessor()
        self.swap_test = MultimodalSwapTest(self.config)
        
        # Add multimodal-specific components
        self.complexity_engine = ComplexityAssessmentEngine()
        self.medical_processor = MedicalDomainProcessor()
        
        # Performance monitoring for multimodal operations
        self.multimodal_stats = {
            'total_computations': 0,
            'avg_computation_time_ms': 0.0,
            'modality_usage_stats': {},
            'quantum_advantage_cases': 0,
            'cross_modal_correlations': [],
            'uncertainty_metrics': []
        }
        
        logger.info("MultimodalQuantumSimilarityEngine initialized with quantum entanglement fusion")
    
    def compute_multimodal_similarity(self, 
                                    query: Dict[str, Any], 
                                    candidate: Dict[str, Any]) -> Tuple[float, Dict]:
        """
        Compute quantum similarity for multimodal medical data.
        
        Args:
            query: Multimodal query data
            candidate: Multimodal candidate data
            
        Returns:
            Tuple of (similarity_score, metadata)
        """
        start_time = time.time()
        
        try:
            # Assess complexity to determine processing approach
            complexity_result = self.complexity_engine.assess_complexity(query, [candidate])
            
            # Process medical domain aspects
            processed_query = self.medical_processor.process_medical_query(query)
            processed_candidate = self.medical_processor.process_medical_query(candidate)
            
            # Extract multimodal embeddings
            query_result = self.embedding_processor.encode_multimodal(processed_query)
            candidate_result = self.embedding_processor.encode_multimodal(processed_candidate)
            
            # Prepare modality dictionaries for quantum processing
            query_modalities = self._prepare_modality_dict(query_result)
            candidate_modalities = self._prepare_modality_dict(candidate_result)
            
            # Quantum multimodal fidelity computation
            fidelity, fidelity_metadata = self.swap_test.compute_multimodal_fidelity(
                query_modalities, candidate_modalities
            )
            
            # Assess quantum advantage
            quantum_advantage_indicators = self._assess_quantum_advantage(fidelity, fidelity_metadata)
            
            # Add comprehensive metadata
            total_time = (time.time() - start_time) * 1000
            metadata = {
                'total_computation_time_ms': total_time,
                'complexity_score': complexity_result.overall_complexity.overall_complexity,
                'medical_domain': processed_query.get('domain', 'unknown'),
                'quantum_advantage_indicators': quantum_advantage_indicators,
                'multimodal_processing': True,
                'modalities_used': query_result.modalities_used + candidate_result.modalities_used,
                'fidelity_metadata': fidelity_metadata,
                'query_processing_time_ms': query_result.processing_time_ms,
                'candidate_processing_time_ms': candidate_result.processing_time_ms,
                'success': True
            }
            
            # Update statistics
            self._update_multimodal_stats(metadata, quantum_advantage_indicators)
            
            # Check performance constraint
            if total_time > 100:  # PRD constraint
                logger.warning(f"Multimodal similarity computation exceeded 100ms: {total_time:.2f}ms")
            
            return fidelity, metadata
            
        except Exception as e:
            logger.error(f"Multimodal similarity computation failed: {e}")
            total_time = (time.time() - start_time) * 1000
            
            return 0.0, {
                'total_computation_time_ms': total_time,
                'success': False,
                'error': str(e),
                'multimodal_processing': True
            }
    
    def batch_compute_multimodal_similarity(self, 
                                          query: Dict[str, Any], 
                                          candidates: List[Dict[str, Any]]) -> List[Tuple[float, Dict]]:
        """
        Batch compute multimodal similarities with <500ms constraint.
        
        Args:
            query: Multimodal query data
            candidates: List of multimodal candidate data
            
        Returns:
            List of (similarity_score, metadata) tuples
        """
        start_time = time.time()
        
        # Assess complexity for batch routing
        complexity_result = self.complexity_engine.assess_complexity(query, candidates)
        
        # Process query once
        processed_query = self.medical_processor.process_medical_query(query)
        query_result = self.embedding_processor.encode_multimodal(processed_query)
        query_modalities = self._prepare_modality_dict(query_result)
        
        # Batch process candidates
        results = []
        candidate_modalities_list = []
        
        # Prepare all candidate modalities
        for candidate in candidates:
            processed_candidate = self.medical_processor.process_medical_query(candidate)
            candidate_result = self.embedding_processor.encode_multimodal(processed_candidate)
            candidate_modalities = self._prepare_modality_dict(candidate_result)
            candidate_modalities_list.append(candidate_modalities)
        
        # Batch compute quantum fidelities
        batch_results = self.swap_test.batch_compute_multimodal_fidelity(
            query_modalities, candidate_modalities_list
        )
        
        # Process results and add metadata
        for i, (fidelity, fidelity_metadata) in enumerate(batch_results):
            # Assess quantum advantage for this pair
            quantum_advantage = self._assess_quantum_advantage(fidelity, fidelity_metadata)
            
            metadata = {
                'batch_index': i,
                'batch_processing': True,
                'complexity_score': complexity_result.candidate_complexities[i].overall_complexity if i < len(complexity_result.candidate_complexities) else 0.0,
                'quantum_advantage_indicators': quantum_advantage,
                'fidelity_metadata': fidelity_metadata,
                'success': True
            }
            
            results.append((fidelity, metadata))
        
        # Verify batch processing constraint
        elapsed = (time.time() - start_time) * 1000
        if elapsed > 500:  # PRD constraint
            logger.warning(f"Batch processing exceeded 500ms: {elapsed:.2f}ms for {len(candidates)} candidates")
        
        # Add batch timing to all results
        for i, (fidelity, metadata) in enumerate(results):
            metadata['batch_total_time_ms'] = elapsed
            metadata['batch_per_item_ms'] = elapsed / len(candidates)
            results[i] = (fidelity, metadata)
        
        return results
    
    def _prepare_modality_dict(self, embedding_result) -> Dict[str, np.ndarray]:
        """
        Prepare modality dictionary from embedding result.
        
        Args:
            embedding_result: MultimodalEmbeddingResult
            
        Returns:
            Dictionary mapping modality names to embeddings
        """
        modalities = {}
        
        if embedding_result.text_embedding is not None:
            modalities['text'] = embedding_result.text_embedding
        
        if embedding_result.clinical_embedding is not None:
            modalities['clinical'] = embedding_result.clinical_embedding
        
        # Add image embedding if available (future extension)
        # if embedding_result.image_embedding is not None:
        #     modalities['image'] = embedding_result.image_embedding
        
        return modalities
    
    def _assess_quantum_advantage(self, fidelity: float, metadata: Dict) -> Dict[str, float]:
        """
        Assess indicators of quantum advantage in multimodal processing.
        
        Args:
            fidelity: Computed quantum fidelity
            metadata: Fidelity computation metadata
            
        Returns:
            Dictionary of quantum advantage indicators
        """
        # High entanglement suggests quantum advantage
        entanglement_score = metadata.get('entanglement_measure', 0.0)
        
        # Cross-modal correlations captured by quantum entanglement
        cross_modal_fidelities = metadata.get('cross_modal_fidelities', {})
        if cross_modal_fidelities:
            cross_modal_strength = np.mean(list(cross_modal_fidelities.values()))
        else:
            cross_modal_strength = 0.0
        
        # Uncertainty quantification quality
        uncertainty_metrics = metadata.get('uncertainty_quantification', {})
        quantum_uncertainty = uncertainty_metrics.get('quantum_uncertainty', 1.0)
        uncertainty_quality = 1 - quantum_uncertainty
        
        # Multimodal coherence (how well modalities align)
        modalities_used = metadata.get('modalities_used', [])
        multimodal_coherence = len(modalities_used) / 3.0 if modalities_used else 0.0  # Max 3 modalities
        
        # Circuit efficiency (how much quantum computation was used)
        circuit_depth = metadata.get('quantum_circuit_depth', 0)
        circuit_efficiency = min(circuit_depth / 15.0, 1.0)  # Normalize by max depth
        
        # Statistical significance of quantum measurements
        confidence_intervals = uncertainty_metrics.get('confidence_intervals', {})
        if confidence_intervals:
            # Use width of 95% confidence interval as quality measure
            ci_95 = confidence_intervals.get('ci_95', {'width': 1.0})
            statistical_quality = max(0, 1 - ci_95['width'])
        else:
            statistical_quality = 0.0
        
        # Overall quantum advantage score
        quantum_advantage_factors = {
            'entanglement_advantage': entanglement_score,
            'cross_modal_advantage': cross_modal_strength,
            'uncertainty_advantage': uncertainty_quality,
            'multimodal_coherence': multimodal_coherence,
            'circuit_efficiency': circuit_efficiency,
            'statistical_quality': statistical_quality
        }
        
        # Weighted combination of factors
        weights = {
            'entanglement_advantage': 0.3,
            'cross_modal_advantage': 0.25,
            'uncertainty_advantage': 0.2,
            'multimodal_coherence': 0.1,
            'circuit_efficiency': 0.1,
            'statistical_quality': 0.05
        }
        
        overall_advantage = sum(
            weights[factor] * value
            for factor, value in quantum_advantage_factors.items()
        )
        
        quantum_advantage_factors['overall_quantum_advantage'] = overall_advantage
        
        return quantum_advantage_factors
    
    def _update_multimodal_stats(self, metadata: Dict, quantum_advantage: Dict):
        """
        Update multimodal processing statistics.
        
        Args:
            metadata: Processing metadata
            quantum_advantage: Quantum advantage indicators
        """
        self.multimodal_stats['total_computations'] += 1
        
        # Update average computation time
        current_time = metadata.get('total_computation_time_ms', 0)
        total_computations = self.multimodal_stats['total_computations']
        current_avg = self.multimodal_stats['avg_computation_time_ms']
        
        self.multimodal_stats['avg_computation_time_ms'] = (
            (current_avg * (total_computations - 1) + current_time) / total_computations
        )
        
        # Update modality usage statistics
        modalities_used = metadata.get('modalities_used', [])
        for modality in modalities_used:
            if modality not in self.multimodal_stats['modality_usage_stats']:
                self.multimodal_stats['modality_usage_stats'][modality] = 0
            self.multimodal_stats['modality_usage_stats'][modality] += 1
        
        # Update quantum advantage cases
        overall_advantage = quantum_advantage.get('overall_quantum_advantage', 0)
        if overall_advantage > 0.6:  # Threshold for significant quantum advantage
            self.multimodal_stats['quantum_advantage_cases'] += 1
        
        # Track cross-modal correlations
        cross_modal_strength = quantum_advantage.get('cross_modal_advantage', 0)
        self.multimodal_stats['cross_modal_correlations'].append(cross_modal_strength)
        
        # Keep only recent correlations (last 100)
        if len(self.multimodal_stats['cross_modal_correlations']) > 100:
            self.multimodal_stats['cross_modal_correlations'] = self.multimodal_stats['cross_modal_correlations'][-100:]
        
        # Track uncertainty metrics
        uncertainty_quality = quantum_advantage.get('uncertainty_advantage', 0)
        self.multimodal_stats['uncertainty_metrics'].append(uncertainty_quality)
        
        # Keep only recent uncertainty metrics (last 100)
        if len(self.multimodal_stats['uncertainty_metrics']) > 100:
            self.multimodal_stats['uncertainty_metrics'] = self.multimodal_stats['uncertainty_metrics'][-100:]
    
    def get_multimodal_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive multimodal performance report.
        
        Returns:
            Performance metrics and quantum advantage analysis
        """
        base_report = self.get_performance_report()
        
        # Add multimodal-specific metrics
        multimodal_report = {
            'base_performance': base_report,
            'multimodal_stats': self.multimodal_stats.copy(),
            'quantum_advantage_rate': 0.0,
            'avg_cross_modal_correlation': 0.0,
            'avg_uncertainty_quality': 0.0,
            'modality_distribution': {},
            'performance_analysis': {}
        }
        
        # Calculate derived metrics
        if self.multimodal_stats['total_computations'] > 0:
            multimodal_report['quantum_advantage_rate'] = (
                self.multimodal_stats['quantum_advantage_cases'] / 
                self.multimodal_stats['total_computations']
            )
        
        if self.multimodal_stats['cross_modal_correlations']:
            multimodal_report['avg_cross_modal_correlation'] = np.mean(
                self.multimodal_stats['cross_modal_correlations']
            )
        
        if self.multimodal_stats['uncertainty_metrics']:
            multimodal_report['avg_uncertainty_quality'] = np.mean(
                self.multimodal_stats['uncertainty_metrics']
            )
        
        # Modality distribution
        total_modality_uses = sum(self.multimodal_stats['modality_usage_stats'].values())
        if total_modality_uses > 0:
            for modality, count in self.multimodal_stats['modality_usage_stats'].items():
                multimodal_report['modality_distribution'][modality] = count / total_modality_uses
        
        # Performance analysis
        multimodal_report['performance_analysis'] = {
            'meets_latency_target': self.multimodal_stats['avg_computation_time_ms'] < 100,
            'quantum_advantage_effective': multimodal_report['quantum_advantage_rate'] > 0.3,
            'cross_modal_fusion_quality': multimodal_report['avg_cross_modal_correlation'] > 0.5,
            'uncertainty_quantification_quality': multimodal_report['avg_uncertainty_quality'] > 0.5
        }
        
        return multimodal_report
    
    def optimize_for_multimodal_performance(self):
        """Optimize engine specifically for multimodal performance."""
        # Clear caches
        if hasattr(self.embedding_processor, 'clear_cache'):
            self.embedding_processor.clear_cache()
        
        # Optimize memory usage
        if hasattr(self.embedding_processor, 'optimize_memory'):
            self.embedding_processor.optimize_memory()
        
        # Reset multimodal statistics
        self.multimodal_stats = {
            'total_computations': 0,
            'avg_computation_time_ms': 0.0,
            'modality_usage_stats': {},
            'quantum_advantage_cases': 0,
            'cross_modal_correlations': [],
            'uncertainty_metrics': []
        }
        
        logger.info("MultimodalQuantumSimilarityEngine optimized for performance")
    
    def validate_quantum_advantages(self, test_queries: List[Dict[str, Any]], 
                                  test_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate quantum advantages on test data.
        
        Args:
            test_queries: List of test query data
            test_candidates: List of test candidate data
            
        Returns:
            Validation results and quantum advantage analysis
        """
        validation_results = {
            'total_tests': len(test_queries),
            'quantum_advantages': [],
            'cross_modal_effectiveness': [],
            'uncertainty_improvements': [],
            'performance_metrics': []
        }
        
        for i, (query, candidate) in enumerate(zip(test_queries, test_candidates)):
            # Compute quantum similarity
            similarity, metadata = self.compute_multimodal_similarity(query, candidate)
            
            # Extract quantum advantage indicators
            quantum_advantage = metadata.get('quantum_advantage_indicators', {})
            validation_results['quantum_advantages'].append(quantum_advantage.get('overall_quantum_advantage', 0))
            validation_results['cross_modal_effectiveness'].append(quantum_advantage.get('cross_modal_advantage', 0))
            validation_results['uncertainty_improvements'].append(quantum_advantage.get('uncertainty_advantage', 0))
            validation_results['performance_metrics'].append(metadata.get('total_computation_time_ms', 0))
        
        # Calculate summary statistics
        validation_results['summary'] = {
            'avg_quantum_advantage': np.mean(validation_results['quantum_advantages']),
            'avg_cross_modal_effectiveness': np.mean(validation_results['cross_modal_effectiveness']),
            'avg_uncertainty_improvement': np.mean(validation_results['uncertainty_improvements']),
            'avg_computation_time_ms': np.mean(validation_results['performance_metrics']),
            'quantum_advantage_cases': sum(1 for qa in validation_results['quantum_advantages'] if qa > 0.6),
            'performance_target_met': all(pt < 100 for pt in validation_results['performance_metrics'])
        }
        
        return validation_results
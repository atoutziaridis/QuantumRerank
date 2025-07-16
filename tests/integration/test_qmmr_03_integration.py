"""
Integration tests for QMMR-03 Quantum Multimodal Similarity Engine.

Tests the complete integration of multimodal quantum similarity computation
including circuits, SWAP test, uncertainty quantification, and performance.
"""

import pytest
import numpy as np
import time
from typing import Dict, List, Any

from quantum_rerank.core.multimodal_quantum_similarity_engine import MultimodalQuantumSimilarityEngine
from quantum_rerank.core.multimodal_quantum_circuits import MultimodalQuantumCircuits
from quantum_rerank.core.multimodal_swap_test import MultimodalSwapTest
from quantum_rerank.core.quantum_entanglement_metrics import QuantumEntanglementAnalyzer
from quantum_rerank.core.uncertainty_quantification import MultimodalUncertaintyQuantifier
from quantum_rerank.config.settings import SimilarityEngineConfig


class TestQMMR03Integration:
    """Integration test suite for QMMR-03 multimodal quantum similarity engine."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up integration test environment."""
        # Configuration for integration testing
        self.config = SimilarityEngineConfig(
            n_qubits=3,
            shots=512,  # Reduced for faster testing
            max_circuit_depth=15
        )
        
        # Initialize components
        self.engine = MultimodalQuantumSimilarityEngine(self.config)
        self.circuits = MultimodalQuantumCircuits(self.config)
        self.swap_test = MultimodalSwapTest(self.config)
        self.entanglement_analyzer = QuantumEntanglementAnalyzer()
        self.uncertainty_quantifier = MultimodalUncertaintyQuantifier()
        
        # Sample medical data for testing
        self.medical_queries = [
            {
                'text': 'Patient presents with acute chest pain, shortness of breath, and elevated troponin levels',
                'clinical_data': {
                    'age': 65,
                    'gender': 'M',
                    'vitals': {'BP': '140/90', 'HR': 95, 'RR': 18, 'T': 98.6},
                    'symptoms': ['chest pain', 'dyspnea', 'diaphoresis'],
                    'lab_results': {'troponin': 0.8, 'CK-MB': 12, 'BNP': 300},
                    'medications': ['aspirin', 'metoprolol', 'lisinopril'],
                    'history': ['hypertension', 'diabetes', 'smoking']
                }
            },
            {
                'text': 'Elderly female with progressive dyspnea, orthopnea, and lower extremity edema',
                'clinical_data': {
                    'age': 78,
                    'gender': 'F',
                    'vitals': {'BP': '110/70', 'HR': 88, 'RR': 22, 'T': 98.2},
                    'symptoms': ['dyspnea', 'orthopnea', 'edema', 'fatigue'],
                    'lab_results': {'BNP': 850, 'creatinine': 1.4, 'albumin': 3.2},
                    'medications': ['furosemide', 'carvedilol', 'lisinopril'],
                    'history': ['CHF', 'atrial fibrillation', 'CKD']
                }
            },
            {
                'text': 'Young adult with sudden onset severe headache, photophobia, and neck stiffness',
                'clinical_data': {
                    'age': 28,
                    'gender': 'F',
                    'vitals': {'BP': '130/85', 'HR': 110, 'RR': 20, 'T': 101.2},
                    'symptoms': ['headache', 'photophobia', 'neck stiffness', 'nausea'],
                    'lab_results': {'WBC': 14000, 'glucose': 95, 'lactate': 2.1},
                    'medications': ['acetaminophen'],
                    'history': ['migraine', 'oral contraceptives']
                }
            }
        ]
        
        self.medical_candidates = [
            {
                'text': 'Myocardial infarction with ST elevation, requires immediate catheterization',
                'clinical_data': {
                    'age': 62,
                    'gender': 'M',
                    'vitals': {'BP': '135/85', 'HR': 98, 'RR': 16, 'T': 98.8},
                    'symptoms': ['chest pain', 'diaphoresis', 'nausea'],
                    'lab_results': {'troponin': 1.2, 'CK-MB': 18, 'cholesterol': 240},
                    'medications': ['aspirin', 'clopidogrel', 'atorvastatin'],
                    'diagnosis': 'STEMI'
                }
            },
            {
                'text': 'Congestive heart failure exacerbation with fluid overload',
                'clinical_data': {
                    'age': 75,
                    'gender': 'F',
                    'vitals': {'BP': '105/65', 'HR': 92, 'RR': 24, 'T': 98.1},
                    'symptoms': ['dyspnea', 'edema', 'fatigue'],
                    'lab_results': {'BNP': 920, 'creatinine': 1.6, 'sodium': 132},
                    'medications': ['furosemide', 'metoprolol', 'digoxin'],
                    'diagnosis': 'CHF exacerbation'
                }
            },
            {
                'text': 'Bacterial meningitis with altered mental status and fever',
                'clinical_data': {
                    'age': 25,
                    'gender': 'M',
                    'vitals': {'BP': '125/80', 'HR': 115, 'RR': 18, 'T': 102.1},
                    'symptoms': ['headache', 'fever', 'confusion', 'neck stiffness'],
                    'lab_results': {'WBC': 18000, 'CSF_WBC': 1200, 'glucose': 40},
                    'medications': ['ceftriaxone', 'vancomycin', 'dexamethasone'],
                    'diagnosis': 'bacterial meningitis'
                }
            }
        ]
    
    def test_complete_multimodal_pipeline(self):
        """Test complete multimodal quantum similarity pipeline."""
        query = self.medical_queries[0]
        candidate = self.medical_candidates[0]
        
        # Execute complete pipeline
        start_time = time.time()
        fidelity, metadata = self.engine.compute_multimodal_similarity(query, candidate)
        total_time = (time.time() - start_time) * 1000
        
        # Verify basic results
        assert 0 <= fidelity <= 1
        assert metadata['success'] is True
        assert metadata['multimodal_processing'] is True
        
        # Verify performance constraint
        assert total_time < 100  # PRD requirement: <100ms
        assert metadata['total_computation_time_ms'] < 100
        
        # Verify metadata structure
        required_keys = [
            'complexity_score',
            'quantum_advantage_indicators',
            'fidelity_metadata',
            'modalities_used'
        ]
        
        for key in required_keys:
            assert key in metadata, f"Missing required metadata key: {key}"
        
        # Verify quantum advantage assessment
        qa_indicators = metadata['quantum_advantage_indicators']
        assert 'overall_quantum_advantage' in qa_indicators
        assert 0 <= qa_indicators['overall_quantum_advantage'] <= 1
    
    def test_quantum_circuit_constraints(self):
        """Test that quantum circuits meet PRD constraints."""
        # Test with different modality combinations
        text_emb = np.random.rand(256)
        clinical_emb = np.random.rand(256)
        image_emb = np.random.rand(256)
        
        # Two-modality circuit
        circuit_2mod = self.circuits.create_multimodal_state_preparation_circuit(
            text_emb, clinical_emb
        )
        
        # Three-modality circuit
        circuit_3mod = self.circuits.create_multimodal_state_preparation_circuit(
            text_emb, clinical_emb, image_emb
        )
        
        # Verify PRD constraints
        assert circuit_2mod.depth() <= 15, f"Circuit depth {circuit_2mod.depth()} exceeds limit"
        assert circuit_2mod.num_qubits <= 4, f"Circuit uses {circuit_2mod.num_qubits} qubits, limit is 4"
        
        assert circuit_3mod.depth() <= 15, f"Circuit depth {circuit_3mod.depth()} exceeds limit"
        assert circuit_3mod.num_qubits <= 4, f"Circuit uses {circuit_3mod.num_qubits} qubits, limit is 4"
        
        # Verify circuits have entanglement
        validation_2mod = self.circuits.validate_circuit_constraints(circuit_2mod)
        validation_3mod = self.circuits.validate_circuit_constraints(circuit_3mod)
        
        assert validation_2mod['has_entanglement'], "Two-modality circuit lacks entanglement"
        assert validation_3mod['has_entanglement'], "Three-modality circuit lacks entanglement"
    
    def test_entanglement_analysis(self):
        """Test quantum entanglement analysis functionality."""
        # Create sample circuit with known entanglement
        text_emb = np.random.rand(128)
        clinical_emb = np.random.rand(128)
        
        circuit = self.circuits.create_multimodal_state_preparation_circuit(
            text_emb, clinical_emb
        )
        
        # Analyze entanglement
        entanglement_metrics = self.entanglement_analyzer.analyze_circuit_entanglement(circuit)
        
        # Verify entanglement metrics structure
        assert hasattr(entanglement_metrics, 'von_neumann_entropy')
        assert hasattr(entanglement_metrics, 'text_clinical_entanglement')
        assert hasattr(entanglement_metrics, 'entangling_gate_count')
        
        # Verify bounds
        assert 0 <= entanglement_metrics.von_neumann_entropy <= 1
        assert 0 <= entanglement_metrics.text_clinical_entanglement <= 1
        assert entanglement_metrics.entangling_gate_count >= 0
        
        # Verify overall entanglement score
        overall_score = entanglement_metrics.get_overall_entanglement_score()
        assert 0 <= overall_score <= 1
    
    def test_uncertainty_quantification_integration(self):
        """Test uncertainty quantification integration."""
        # Generate sample quantum measurement data
        measurement_data = {
            'fidelity': 0.82,
            'shots': 1024,
            'counts': {'0': 850, '1': 174},
            'circuit_depth': 12,
            'entanglement_measure': 0.7
        }
        
        cross_modal_fidelities = {
            'text_clinical': 0.85,
            'text_image': 0.78,
            'clinical_image': 0.80
        }
        
        # Quantify uncertainty
        uncertainty_results = self.uncertainty_quantifier.quantify_multimodal_uncertainty(
            measurement_data['fidelity'],
            cross_modal_fidelities,
            measurement_data
        )
        
        # Verify structure
        assert 'overall' in uncertainty_results
        for modality_pair in cross_modal_fidelities.keys():
            assert modality_pair in uncertainty_results
        
        # Verify uncertainty quality
        overall_metrics = uncertainty_results['overall']
        quality_scores = self.uncertainty_quantifier.assess_uncertainty_quality(overall_metrics)
        
        assert 'overall_quality' in quality_scores
        assert 0 <= quality_scores['overall_quality'] <= 1
    
    def test_batch_processing_performance(self):
        """Test batch processing meets <500ms constraint."""
        query = self.medical_queries[0]
        candidates = self.medical_candidates[:2]  # Small batch for testing
        
        # Execute batch processing
        start_time = time.time()
        results = self.engine.batch_compute_multimodal_similarity(query, candidates)
        total_time = (time.time() - start_time) * 1000
        
        # Verify performance constraint
        assert total_time < 500, f"Batch processing took {total_time:.2f}ms, exceeds 500ms limit"
        
        # Verify results structure
        assert len(results) == len(candidates)
        
        for i, (fidelity, metadata) in enumerate(results):
            assert 0 <= fidelity <= 1
            assert metadata['batch_index'] == i
            assert metadata['batch_processing'] is True
            assert 'batch_total_time_ms' in metadata
    
    def test_multimodal_similarity_accuracy(self):
        """Test multimodal similarity accuracy for medical domain."""
        # Test similar cases (should have high similarity)
        cardiac_query = self.medical_queries[0]  # Chest pain case
        cardiac_candidate = self.medical_candidates[0]  # MI case
        
        cardiac_fidelity, _ = self.engine.compute_multimodal_similarity(
            cardiac_query, cardiac_candidate
        )
        
        # Test dissimilar cases (should have lower similarity)
        cardiac_query = self.medical_queries[0]  # Chest pain case
        neuro_candidate = self.medical_candidates[2]  # Meningitis case
        
        dissimilar_fidelity, _ = self.engine.compute_multimodal_similarity(
            cardiac_query, neuro_candidate
        )
        
        # Similar cases should have higher fidelity than dissimilar ones
        assert cardiac_fidelity > dissimilar_fidelity, \
            f"Similar cases fidelity ({cardiac_fidelity:.3f}) not higher than dissimilar ({dissimilar_fidelity:.3f})"
        
        # Both should be within valid range
        assert 0 <= cardiac_fidelity <= 1
        assert 0 <= dissimilar_fidelity <= 1
    
    def test_quantum_advantage_detection(self):
        """Test quantum advantage detection in complex cases."""
        # Use a complex query with multiple modalities
        complex_query = self.medical_queries[0]
        complex_candidate = self.medical_candidates[0]
        
        fidelity, metadata = self.engine.compute_multimodal_similarity(
            complex_query, complex_candidate
        )
        
        # Extract quantum advantage indicators
        qa_indicators = metadata['quantum_advantage_indicators']
        
        # Verify quantum advantage components
        required_qa_keys = [
            'entanglement_advantage',
            'cross_modal_advantage',
            'uncertainty_advantage',
            'overall_quantum_advantage'
        ]
        
        for key in required_qa_keys:
            assert key in qa_indicators, f"Missing quantum advantage indicator: {key}"
            assert 0 <= qa_indicators[key] <= 1, f"Invalid range for {key}: {qa_indicators[key]}"
        
        # For complex multimodal cases, should show some quantum advantage
        assert qa_indicators['overall_quantum_advantage'] > 0.1, \
            "Expected some quantum advantage for complex multimodal case"
    
    def test_cross_modal_fidelity_analysis(self):
        """Test cross-modal fidelity computation and analysis."""
        query = self.medical_queries[0]
        candidate = self.medical_candidates[0]
        
        fidelity, metadata = self.engine.compute_multimodal_similarity(query, candidate)
        
        # Verify cross-modal fidelity metadata
        fidelity_metadata = metadata['fidelity_metadata']
        
        if 'cross_modal_fidelities' in fidelity_metadata:
            cross_modal_fidelities = fidelity_metadata['cross_modal_fidelities']
            
            # Verify structure and ranges
            for modality_pair, cf_fidelity in cross_modal_fidelities.items():
                assert 0 <= cf_fidelity <= 1, \
                    f"Cross-modal fidelity {modality_pair}: {cf_fidelity} out of range"
            
            # Text-clinical correlation should exist for medical data
            text_clinical_keys = [k for k in cross_modal_fidelities.keys() 
                                if 'text' in k and 'clinical' in k]
            assert len(text_clinical_keys) > 0, "Missing text-clinical cross-modal fidelity"
    
    def test_performance_monitoring_and_optimization(self):
        """Test performance monitoring and optimization features."""
        # Run several computations to build statistics
        query = self.medical_queries[0]
        
        for i, candidate in enumerate(self.medical_candidates):
            self.engine.compute_multimodal_similarity(query, candidate)
        
        # Get performance report
        report = self.engine.get_multimodal_performance_report()
        
        # Verify report structure
        assert 'multimodal_stats' in report
        assert 'quantum_advantage_rate' in report
        assert 'performance_analysis' in report
        
        # Verify statistics tracking
        stats = report['multimodal_stats']
        assert stats['total_computations'] == len(self.medical_candidates)
        assert stats['avg_computation_time_ms'] > 0
        
        # Test optimization
        self.engine.optimize_for_multimodal_performance()
        
        # Stats should be reset after optimization
        optimized_stats = self.engine.multimodal_stats
        assert optimized_stats['total_computations'] == 0
    
    def test_error_handling_and_robustness(self):
        """Test error handling and robustness of the system."""
        # Test with invalid input data
        invalid_query = {'invalid_key': 'invalid_value'}
        valid_candidate = self.medical_candidates[0]
        
        fidelity, metadata = self.engine.compute_multimodal_similarity(
            invalid_query, valid_candidate
        )
        
        # Should handle gracefully
        assert metadata is not None
        assert 'success' in metadata
        # May succeed with empty embeddings or fail gracefully
        
        # Test with empty data
        empty_query = {}
        empty_candidate = {}
        
        fidelity, metadata = self.engine.compute_multimodal_similarity(
            empty_query, empty_candidate
        )
        
        # Should handle gracefully
        assert isinstance(fidelity, (int, float))
        assert isinstance(metadata, dict)
    
    def test_memory_and_resource_usage(self):
        """Test memory usage and resource management."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple computations
        query = self.medical_queries[0]
        
        for _ in range(10):  # Reduced for testing
            for candidate in self.medical_candidates:
                fidelity, metadata = self.engine.compute_multimodal_similarity(
                    query, candidate
                )
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not exceed 2GB increase (PRD constraint)
        assert memory_increase < 2048, \
            f"Memory increase {memory_increase:.1f}MB exceeds 2GB limit"
        
        # Test memory optimization
        self.engine.optimize_for_multimodal_performance()
        
        # Memory should not continue growing unbounded
        optimized_memory = process.memory_info().rss / 1024 / 1024  # MB
        assert optimized_memory <= final_memory + 100, \
            "Memory optimization did not prevent memory growth"
    
    def test_end_to_end_medical_scenarios(self):
        """Test end-to-end scenarios with realistic medical data."""
        scenarios = [
            {
                'name': 'Cardiac Emergency',
                'query': self.medical_queries[0],
                'relevant_candidate': self.medical_candidates[0],
                'irrelevant_candidate': self.medical_candidates[2],
                'expected_relevance_difference': 0.1
            },
            {
                'name': 'Heart Failure',
                'query': self.medical_queries[1],
                'relevant_candidate': self.medical_candidates[1],
                'irrelevant_candidate': self.medical_candidates[2],
                'expected_relevance_difference': 0.1
            },
            {
                'name': 'Neurological Emergency',
                'query': self.medical_queries[2],
                'relevant_candidate': self.medical_candidates[2],
                'irrelevant_candidate': self.medical_candidates[0],
                'expected_relevance_difference': 0.1
            }
        ]
        
        for scenario in scenarios:
            # Compute similarity with relevant candidate
            relevant_fidelity, relevant_metadata = self.engine.compute_multimodal_similarity(
                scenario['query'], scenario['relevant_candidate']
            )
            
            # Compute similarity with irrelevant candidate
            irrelevant_fidelity, irrelevant_metadata = self.engine.compute_multimodal_similarity(
                scenario['query'], scenario['irrelevant_candidate']
            )
            
            # Verify discriminative power
            relevance_difference = relevant_fidelity - irrelevant_fidelity
            assert relevance_difference >= scenario['expected_relevance_difference'], \
                f"Scenario '{scenario['name']}': Insufficient discriminative power " \
                f"(difference: {relevance_difference:.3f}, expected: {scenario['expected_relevance_difference']})"
            
            # Verify both computations were successful
            assert relevant_metadata['success'] is True
            assert irrelevant_metadata['success'] is True
            
            # Verify performance constraints
            assert relevant_metadata['total_computation_time_ms'] < 100
            assert irrelevant_metadata['total_computation_time_ms'] < 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
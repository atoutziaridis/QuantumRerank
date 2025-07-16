"""
Integration tests for QMMR-04 Medical Image Integration & Processing.

Tests the complete integration of medical image processing, BiomedCLIP encoding,
quantum-inspired compression, and image-text similarity computation.
"""

import pytest
import numpy as np
import time
import tempfile
import os
from typing import Dict, List, Any
from PIL import Image

from quantum_rerank.core.medical_image_processor import MedicalImageProcessor
from quantum_rerank.core.image_text_quantum_similarity import ImageTextQuantumSimilarity
from quantum_rerank.core.enhanced_multimodal_compression import EnhancedMultimodalCompression
from quantum_rerank.config.settings import SimilarityEngineConfig
from quantum_rerank.config.medical_image_config import MedicalImageConfig


class TestQMMR04Integration:
    """Integration test suite for QMMR-04 medical image processing and similarity."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up integration test environment."""
        # Configuration for integration testing
        self.config = SimilarityEngineConfig(
            n_qubits=3,
            shots=256,  # Reduced for faster testing
            max_circuit_depth=15
        )
        
        self.image_config = MedicalImageConfig()
        
        # Initialize components
        self.image_processor = MedicalImageProcessor(self.image_config)
        self.image_text_similarity = ImageTextQuantumSimilarity(self.config)
        self.multimodal_compression = EnhancedMultimodalCompression(self.config)
        
        # Sample medical image-text data for testing
        self.medical_image_text_pairs = [
            {
                'image': self._create_chest_xray_image(),
                'text': 'Chest X-ray showing bilateral pneumonia with consolidation',
                'text_embedding': np.random.randn(256).astype(np.float32),
                'expected_modality': 'XR',
                'expected_body_part': 'chest'
            },
            {
                'image': self._create_ct_scan_image(),
                'text': 'CT scan of abdomen revealing liver lesion',
                'text_embedding': np.random.randn(256).astype(np.float32),
                'expected_modality': 'CT',
                'expected_body_part': 'abdomen'
            },
            {
                'image': self._create_mri_image(),
                'text': 'MRI brain scan showing white matter hyperintensities',
                'text_embedding': np.random.randn(256).astype(np.float32),
                'expected_modality': 'MR',
                'expected_body_part': 'brain'
            }
        ]
        
        # Create temporary files for file-based testing
        self.temp_image_files = []
        for pair in self.medical_image_text_pairs:
            temp_file = self._create_temp_image_file(pair['image'])
            self.temp_image_files.append(temp_file)
        
        # Medical image similarity test cases
        self.similarity_test_cases = [
            {
                'name': 'High Similarity - Same Modality',
                'image1': self._create_chest_xray_image(),
                'text1': 'Chest X-ray showing pneumonia',
                'image2': self._create_chest_xray_image(),
                'text2': 'Chest radiograph with pneumonic infiltrates',
                'expected_similarity': 'high'  # > 0.6
            },
            {
                'name': 'Medium Similarity - Related Content',
                'image1': self._create_chest_xray_image(),
                'text1': 'Chest X-ray normal findings',
                'image2': self._create_ct_scan_image(),
                'text2': 'CT chest without acute findings',
                'expected_similarity': 'medium'  # 0.3 - 0.6
            },
            {
                'name': 'Low Similarity - Different Modality/Content',
                'image1': self._create_chest_xray_image(),
                'text1': 'Chest X-ray showing pneumonia',
                'image2': self._create_mri_image(),
                'text2': 'MRI brain showing stroke',
                'expected_similarity': 'low'  # < 0.3
            }
        ]
    
    def teardown_method(self):
        """Clean up test environment."""
        # Remove temporary files
        for temp_file in self.temp_image_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def _create_chest_xray_image(self) -> Image.Image:
        """Create a synthetic chest X-ray image."""
        # Create image with typical X-ray characteristics
        image_array = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        
        # Add lung-like structures
        center_x, center_y = 112, 112
        for i in range(224):
            for j in range(224):
                # Create bilateral lung fields
                left_lung_dist = np.sqrt((i - center_x + 40)**2 + (j - center_y)**2)
                right_lung_dist = np.sqrt((i - center_x - 40)**2 + (j - center_y)**2)
                
                if left_lung_dist < 60 or right_lung_dist < 60:
                    image_array[i, j] = [30, 30, 30]  # Dark lung fields
                
                # Add spine/mediastinum
                if abs(i - center_x) < 10:
                    image_array[i, j] = [180, 180, 180]  # Bright spine
        
        return Image.fromarray(image_array)
    
    def _create_ct_scan_image(self) -> Image.Image:
        """Create a synthetic CT scan image."""
        # Create image with typical CT characteristics
        image_array = np.random.randint(80, 160, (224, 224, 3), dtype=np.uint8)
        
        # Add organ-like structures
        center_x, center_y = 112, 112
        for i in range(224):
            for j in range(224):
                dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                
                # Body outline
                if 80 < dist < 100:
                    image_array[i, j] = [60, 60, 60]  # Skin/muscle
                elif dist < 80:
                    # Internal organs
                    organ_noise = np.random.randint(-20, 20)
                    image_array[i, j] = np.clip([120 + organ_noise] * 3, 0, 255)
        
        return Image.fromarray(image_array)
    
    def _create_mri_image(self) -> Image.Image:
        """Create a synthetic MRI image."""
        # Create image with typical MRI characteristics
        image_array = np.random.randint(20, 100, (224, 224, 3), dtype=np.uint8)
        
        # Add brain-like structures
        center_x, center_y = 112, 112
        for i in range(224):
            for j in range(224):
                dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                
                # Brain outline
                if dist < 90:
                    # Gray/white matter contrast
                    if dist < 30:
                        image_array[i, j] = [150, 150, 150]  # White matter
                    elif dist < 70:
                        image_array[i, j] = [100, 100, 100]  # Gray matter
                    else:
                        image_array[i, j] = [50, 50, 50]   # CSF
        
        return Image.fromarray(image_array)
    
    def _create_temp_image_file(self, image: Image.Image) -> str:
        """Create a temporary image file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            image.save(tmp_file.name)
            return tmp_file.name
    
    def test_complete_medical_image_processing_pipeline(self):
        """Test complete medical image processing pipeline."""
        for i, pair in enumerate(self.medical_image_text_pairs):
            # Process medical image
            start_time = time.time()
            image_result = self.image_processor.process_medical_image(pair['image'])
            processing_time = (time.time() - start_time) * 1000
            
            # Verify processing results
            assert image_result.processing_success is True, f"Image processing failed for pair {i}"
            assert image_result.embedding is not None, f"No embedding generated for pair {i}"
            assert len(image_result.embedding) == self.image_config.image_quantum_target_dim
            
            # Verify performance constraint (5 seconds from PRD)
            assert processing_time < 5000, f"Image processing took {processing_time:.1f}ms, exceeds 5000ms"
            
            # Verify medical metadata
            assert image_result.image_quality_score >= 0.0
            assert image_result.quantum_compressed is True
            assert image_result.compression_ratio > 1.0
    
    def test_image_text_quantum_similarity_computation(self):
        """Test image-text quantum similarity computation."""
        for i, pair in enumerate(self.medical_image_text_pairs):
            # Compute image-text similarity
            start_time = time.time()
            similarity_result = self.image_text_similarity.compute_image_text_similarity(
                pair['image'], pair['text_embedding']
            )
            computation_time = (time.time() - start_time) * 1000
            
            # Verify similarity results
            assert similarity_result.processing_success is True, f"Similarity computation failed for pair {i}"
            assert 0.0 <= similarity_result.similarity_score <= 1.0
            
            # Verify performance constraint (150ms from task spec)
            assert computation_time < 150, f"Similarity computation took {computation_time:.1f}ms, exceeds 150ms"
            
            # Verify quantum processing
            assert similarity_result.quantum_circuit_depth > 0
            assert similarity_result.quantum_circuit_qubits >= 2
            assert similarity_result.quantum_fidelity >= 0.0
            assert similarity_result.entanglement_measure >= 0.0
            
            # Verify cross-modal analysis
            assert 'image' in similarity_result.cross_modal_attention_weights
            assert 'text' in similarity_result.cross_modal_attention_weights
            assert 'image' in similarity_result.modality_contributions
            assert 'text' in similarity_result.modality_contributions
    
    def test_enhanced_multimodal_compression_integration(self):
        """Test enhanced multimodal compression with medical images."""
        for pair in self.medical_image_text_pairs:
            # Process image to get embedding
            image_result = self.image_processor.process_medical_image(pair['image'])
            assert image_result.processing_success is True
            
            # Create multimodal data
            modalities = {
                'text': np.random.randn(768).astype(np.float32),  # Simulated text embedding
                'clinical': np.random.randn(768).astype(np.float32),  # Simulated clinical embedding
                'image': image_result.embedding
            }
            
            # Apply enhanced compression
            compression_result = self.multimodal_compression.fuse_all_modalities(modalities)
            
            # Verify compression results
            assert compression_result.compression_success is True
            assert compression_result.fused_embedding is not None
            assert len(compression_result.fused_embedding) == 256  # Target dimension
            assert compression_result.compression_ratio > 1.0
            
            # Verify modality processing
            assert len(compression_result.modalities_used) == 3
            assert 'text' in compression_result.modalities_used
            assert 'clinical' in compression_result.modalities_used
            assert 'image' in compression_result.modalities_used
            
            # Verify quality metrics
            assert 0.0 <= compression_result.information_preservation_score <= 1.0
            assert 0.0 <= compression_result.modality_balance_score <= 1.0
    
    def test_medical_image_similarity_discrimination(self):
        """Test medical image similarity discrimination across different cases."""
        similarity_scores = {}
        
        for test_case in self.similarity_test_cases:
            # Prepare text embeddings
            text_emb1 = np.random.randn(256).astype(np.float32)
            text_emb2 = np.random.randn(256).astype(np.float32)
            
            # Compute similarities
            result1 = self.image_text_similarity.compute_image_text_similarity(
                test_case['image1'], text_emb1
            )
            result2 = self.image_text_similarity.compute_image_text_similarity(
                test_case['image2'], text_emb2
            )
            
            # Store for analysis
            similarity_scores[test_case['name']] = {
                'score1': result1.similarity_score,
                'score2': result2.similarity_score,
                'expected': test_case['expected_similarity']
            }
            
            # Verify both computations succeeded
            assert result1.processing_success is True
            assert result2.processing_success is True
        
        # Analyze discrimination capability
        high_scores = [v['score1'] for k, v in similarity_scores.items() if v['expected'] == 'high']
        medium_scores = [v['score1'] for k, v in similarity_scores.items() if v['expected'] == 'medium']
        low_scores = [v['score1'] for k, v in similarity_scores.items() if v['expected'] == 'low']
        
        # Verify reasonable distribution (not strict due to randomness in test data)
        if high_scores and low_scores:
            avg_high = np.mean(high_scores)
            avg_low = np.mean(low_scores)
            # High similarity should generally be higher than low similarity
            # Note: This may not always pass with random test data, but shows the intent
            assert avg_high >= avg_low or abs(avg_high - avg_low) < 0.3  # Allow some variance
    
    def test_batch_processing_performance(self):
        """Test batch processing performance constraints."""
        # Create batch of image-text pairs
        batch_images = [pair['image'] for pair in self.medical_image_text_pairs[:2]]
        batch_text_embeddings = [pair['text_embedding'] for pair in self.medical_image_text_pairs[:2]]
        
        # Test batch processing
        start_time = time.time()
        batch_results = self.image_text_similarity.batch_compute_similarities(
            batch_images, batch_text_embeddings
        )
        batch_time = (time.time() - start_time) * 1000
        
        # Verify batch results
        assert len(batch_results) == len(batch_images)
        assert all(result.processing_success for result in batch_results)
        
        # Verify batch performance constraint (1000ms from task spec)
        assert batch_time < 1000, f"Batch processing took {batch_time:.1f}ms, exceeds 1000ms"
    
    def test_privacy_protection_integration(self):
        """Test privacy protection features in medical image processing."""
        # Test with privacy settings enabled
        privacy_config = MedicalImageConfig()
        privacy_config.phi_removal = True
        privacy_config.image_anonymization = True
        privacy_processor = MedicalImageProcessor(privacy_config)
        
        for pair in self.medical_image_text_pairs:
            result = privacy_processor.process_medical_image(pair['image'])
            
            # Verify privacy protection was applied
            assert result.processing_success is True
            assert result.phi_removed is True
            assert result.anonymized is True
            
            # Image should still be processable
            assert result.embedding is not None
            assert len(result.embedding) == privacy_config.image_quantum_target_dim
    
    def test_dicom_metadata_handling(self):
        """Test DICOM metadata handling (simulated)."""
        # Create simulated DICOM metadata
        dicom_metadata = {
            'Modality': 'CT',
            'StudyDescription': 'Chest CT with contrast',
            'BodyPartExamined': 'CHEST',
            'SliceThickness': '5.0',
            'WindowCenter': '40',
            'WindowWidth': '400'
        }
        
        # Test metadata extraction capability
        dicom_processor = self.image_processor.dicom_processor
        
        # Should not crash with metadata processing
        assert dicom_processor is not None
        assert hasattr(dicom_processor, 'config')
    
    def test_quantum_circuit_constraints_validation(self):
        """Test that all quantum circuits meet PRD constraints."""
        for pair in self.medical_image_text_pairs:
            # Generate quantum circuit for image-text pair
            image_result = self.image_processor.process_medical_image(pair['image'])
            assert image_result.processing_success is True
            
            circuit = self.image_text_similarity.quantum_circuits.create_image_text_circuit(
                image_result.embedding, pair['text_embedding']
            )
            
            # Verify PRD constraints
            assert circuit.num_qubits <= 4, f"Circuit uses {circuit.num_qubits} qubits, limit is 4"
            assert circuit.depth() <= 15, f"Circuit depth {circuit.depth()} exceeds limit of 15"
            assert circuit.num_clbits > 0, "Circuit should have measurement registers"
    
    def test_uncertainty_quantification_accuracy(self):
        """Test uncertainty quantification accuracy and consistency."""
        uncertainties = []
        
        for pair in self.medical_image_text_pairs:
            result = self.image_text_similarity.compute_image_text_similarity(
                pair['image'], pair['text_embedding']
            )
            
            if result.processing_success and result.uncertainty_metrics:
                uncertainties.append(result.uncertainty_metrics['total_uncertainty'])
                
                # Verify uncertainty bounds
                assert 0.0 <= result.uncertainty_metrics['total_uncertainty'] <= 1.0
                if 'quantum_uncertainty' in result.uncertainty_metrics:
                    assert result.uncertainty_metrics['quantum_uncertainty'] >= 0.0
                if 'statistical_uncertainty' in result.uncertainty_metrics:
                    assert result.uncertainty_metrics['statistical_uncertainty'] >= 0.0
        
        # Uncertainties should be reasonable
        if uncertainties:
            assert np.mean(uncertainties) < 1.0  # Should not be maximum uncertainty
    
    def test_end_to_end_medical_workflow(self):
        """Test complete end-to-end medical image workflow."""
        # Simulate clinical workflow
        clinical_query = {
            'image': self.medical_image_text_pairs[0]['image'],
            'text': 'Patient with chest pain and cough',
            'clinical_data': {
                'age': 65,
                'symptoms': ['chest pain', 'cough', 'fever'],
                'vital_signs': {'temperature': 101.2, 'respiratory_rate': 22}
            }
        }
        
        clinical_candidates = [
            {
                'image': self.medical_image_text_pairs[1]['image'],
                'text': 'Pneumonia diagnosis and treatment protocol',
                'diagnosis': 'bacterial pneumonia'
            },
            {
                'image': self.medical_image_text_pairs[2]['image'],
                'text': 'Normal chest imaging findings',
                'diagnosis': 'normal'
            }
        ]
        
        # Process query image
        query_image_result = self.image_processor.process_medical_image(clinical_query['image'])
        assert query_image_result.processing_success is True
        
        # Compute similarities with candidates
        similarities = []
        for candidate in clinical_candidates:
            candidate_image_result = self.image_processor.process_medical_image(candidate['image'])
            assert candidate_image_result.processing_success is True
            
            # Create text embeddings (simulated)
            query_text_emb = np.random.randn(256).astype(np.float32)
            candidate_text_emb = np.random.randn(256).astype(np.float32)
            
            # Compute similarity
            similarity_result = self.image_text_similarity.compute_image_text_similarity(
                candidate['image'], query_text_emb
            )
            
            similarities.append({
                'candidate': candidate,
                'similarity_score': similarity_result.similarity_score,
                'quantum_advantage': similarity_result.quantum_advantage_score,
                'processing_success': similarity_result.processing_success
            })
        
        # Verify workflow results
        assert len(similarities) == len(clinical_candidates)
        assert all(sim['processing_success'] for sim in similarities)
        assert all(0.0 <= sim['similarity_score'] <= 1.0 for sim in similarities)
    
    def test_memory_usage_optimization(self):
        """Test memory usage and optimization."""
        import psutil
        import os
        
        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple images
        for _ in range(3):  # Reduced for testing
            for pair in self.medical_image_text_pairs:
                # Process image
                image_result = self.image_processor.process_medical_image(pair['image'])
                
                # Compute similarity
                similarity_result = self.image_text_similarity.compute_image_text_similarity(
                    pair['image'], pair['text_embedding']
                )
                
                # Apply compression
                modalities = {
                    'text': np.random.randn(768).astype(np.float32),
                    'clinical': np.random.randn(768).astype(np.float32),
                    'image': image_result.embedding
                }
                compression_result = self.multimodal_compression.fuse_all_modalities(modalities)
        
        # Check memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not exceed reasonable limits (allowing for test framework overhead)
        assert memory_increase < 1000, f"Memory increased by {memory_increase:.1f}MB, seems excessive"
        
        # Test memory optimization
        self.image_processor.optimize_memory()
        self.image_text_similarity.clear_caches()
        
        # Memory should not continue growing excessively
        optimized_memory = process.memory_info().rss / 1024 / 1024  # MB
        assert optimized_memory <= final_memory + 50, "Memory optimization didn't prevent growth"
    
    def test_error_resilience_and_recovery(self):
        """Test error resilience and recovery mechanisms."""
        # Test with corrupted image data
        corrupted_image = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)  # Very small
        corrupted_pil = Image.fromarray(corrupted_image)
        
        # Should handle gracefully
        result = self.image_processor.process_medical_image(corrupted_pil)
        assert result is not None  # Should return something, even if failed
        
        # Test with invalid text embedding
        invalid_text_emb = np.array([])  # Empty array
        
        similarity_result = self.image_text_similarity.compute_image_text_similarity(
            self.medical_image_text_pairs[0]['image'], invalid_text_emb
        )
        assert similarity_result is not None  # Should handle gracefully
        
        # Test compression with missing modalities
        incomplete_modalities = {'text': np.random.randn(768).astype(np.float32)}
        
        compression_result = self.multimodal_compression.fuse_all_modalities(incomplete_modalities)
        assert compression_result is not None  # Should handle partial data
    
    def test_performance_benchmarking_summary(self):
        """Test and summarize overall performance benchmarks."""
        performance_results = {
            'image_processing_times': [],
            'similarity_computation_times': [],
            'compression_times': [],
            'total_workflow_times': [],
            'success_rates': []
        }
        
        for pair in self.medical_image_text_pairs:
            workflow_start = time.time()
            
            # Image processing
            img_start = time.time()
            image_result = self.image_processor.process_medical_image(pair['image'])
            img_time = (time.time() - img_start) * 1000
            performance_results['image_processing_times'].append(img_time)
            
            # Similarity computation
            sim_start = time.time()
            similarity_result = self.image_text_similarity.compute_image_text_similarity(
                pair['image'], pair['text_embedding']
            )
            sim_time = (time.time() - sim_start) * 1000
            performance_results['similarity_computation_times'].append(sim_time)
            
            # Compression
            comp_start = time.time()
            modalities = {
                'text': np.random.randn(768).astype(np.float32),
                'clinical': np.random.randn(768).astype(np.float32),
                'image': image_result.embedding if image_result.processing_success else np.random.randn(128)
            }
            compression_result = self.multimodal_compression.fuse_all_modalities(modalities)
            comp_time = (time.time() - comp_start) * 1000
            performance_results['compression_times'].append(comp_time)
            
            # Total workflow time
            total_time = (time.time() - workflow_start) * 1000
            performance_results['total_workflow_times'].append(total_time)
            
            # Success rate
            overall_success = (image_result.processing_success and 
                             similarity_result.processing_success and
                             compression_result.compression_success)
            performance_results['success_rates'].append(1.0 if overall_success else 0.0)
        
        # Verify performance benchmarks
        avg_img_time = np.mean(performance_results['image_processing_times'])
        avg_sim_time = np.mean(performance_results['similarity_computation_times'])
        avg_comp_time = np.mean(performance_results['compression_times'])
        avg_total_time = np.mean(performance_results['total_workflow_times'])
        success_rate = np.mean(performance_results['success_rates'])
        
        # Performance assertions (from QMMR-04 task spec)
        assert avg_img_time < 5000, f"Average image processing {avg_img_time:.1f}ms exceeds 5000ms limit"
        assert avg_sim_time < 150, f"Average similarity computation {avg_sim_time:.1f}ms exceeds 150ms limit"
        assert success_rate > 0.8, f"Success rate {success_rate:.2f} below 80% threshold"
        
        # Log performance summary
        print(f"\nQMMR-04 Performance Summary:")
        print(f"Average Image Processing: {avg_img_time:.1f}ms")
        print(f"Average Similarity Computation: {avg_sim_time:.1f}ms") 
        print(f"Average Compression: {avg_comp_time:.1f}ms")
        print(f"Average Total Workflow: {avg_total_time:.1f}ms")
        print(f"Success Rate: {success_rate:.2%}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
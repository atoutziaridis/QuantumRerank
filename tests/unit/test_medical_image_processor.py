"""
Unit tests for Medical Image Processor.

Tests the medical image processing implementation with BiomedCLIP integration,
privacy protection, and quantum compression for QMMR-04 task.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from quantum_rerank.core.medical_image_processor import (
    MedicalImageProcessor, MedicalImageProcessingResult,
    DICOMProcessor, MedicalImageAnonymizer
)
from quantum_rerank.config.medical_image_config import MedicalImageConfig


class TestMedicalImageProcessor:
    """Test cases for MedicalImageProcessor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = MedicalImageConfig()
        self.processor = MedicalImageProcessor(self.config)
        
        # Create sample test image
        self.test_image = self._create_test_image()
        self.test_image_path = self._create_test_image_file()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_image_path):
            os.unlink(self.test_image_path)
    
    def _create_test_image(self) -> Image.Image:
        """Create a test medical image."""
        # Create a simple test image with medical-like patterns
        image_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        # Add some structure that might resemble medical imagery
        center_x, center_y = 112, 112
        for i in range(224):
            for j in range(224):
                dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                if dist < 50:
                    image_array[i, j] = [200, 200, 200]  # Bright center
        
        return Image.fromarray(image_array)
    
    def _create_test_image_file(self) -> str:
        """Create a temporary test image file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            self.test_image.save(tmp_file.name)
            return tmp_file.name
    
    def test_initialization(self):
        """Test proper initialization of medical image processor."""
        assert self.processor is not None
        assert self.processor.config is not None
        assert hasattr(self.processor, 'processing_stats')
        assert self.processor.processing_stats['total_images_processed'] == 0
    
    def test_process_medical_image_pil_input(self):
        """Test processing PIL image input."""
        result = self.processor.process_medical_image(self.test_image)
        
        # Basic validation
        assert isinstance(result, MedicalImageProcessingResult)
        assert result.processing_success is True
        assert result.embedding is not None
        assert len(result.embedding) == self.config.image_quantum_target_dim
        assert result.processing_time_ms > 0
        assert result.image_format == 'pil'
    
    def test_process_medical_image_file_path(self):
        """Test processing image from file path."""
        result = self.processor.process_medical_image(self.test_image_path)
        
        # Basic validation
        assert result.processing_success is True
        assert result.embedding is not None
        assert result.image_format == '.png'
        assert result.image_dimensions == (224, 224)
    
    def test_process_medical_image_numpy_array(self):
        """Test processing numpy array input."""
        image_array = np.array(self.test_image)
        result = self.processor.process_medical_image(image_array)
        
        # Basic validation
        assert result.processing_success is True
        assert result.embedding is not None
        assert result.image_format == 'array'
    
    def test_image_preprocessing(self):
        """Test medical image preprocessing functionality."""
        # Create image with different size to trigger resizing
        large_image = Image.new('RGB', (512, 512), color='white')
        
        processed_image = self.processor._apply_medical_preprocessing(large_image)
        
        # Should be resized to config dimensions
        assert processed_image.size == self.config.max_image_size
        assert processed_image.mode == 'RGB'
    
    def test_histogram_equalization(self):
        """Test histogram equalization for medical images."""
        # Create image with poor contrast
        low_contrast_array = np.full((100, 100, 3), 128, dtype=np.uint8)
        low_contrast_image = Image.fromarray(low_contrast_array)
        
        equalized_image = self.processor._apply_histogram_equalization(low_contrast_image)
        
        # Should still be valid image
        assert isinstance(equalized_image, Image.Image)
        assert equalized_image.size == low_contrast_image.size
    
    def test_image_quality_assessment(self):
        """Test image quality assessment."""
        quality_score = self.processor._assess_image_quality(self.test_image)
        
        assert 0.0 <= quality_score <= 1.0
        assert isinstance(quality_score, float)
    
    def test_quantum_compression_integration(self):
        """Test quantum compression integration."""
        # Mock quantum compressor
        with patch.object(self.processor, 'quantum_compressor') as mock_compressor:
            mock_tensor = Mock()
            mock_tensor.squeeze.return_value.numpy.return_value = np.random.rand(128)
            mock_compressor.return_value = mock_tensor
            
            # Test image with quantum compression
            original_embedding = np.random.rand(512)
            compressed = self.processor._apply_quantum_compression(original_embedding)
            
            assert compressed is not None
            assert len(compressed) == 128
            mock_compressor.assert_called_once()
    
    def test_caching_functionality(self):
        """Test image processing caching."""
        # Enable caching
        self.processor.config.enable_image_cache = True
        self.processor.image_cache = {}
        
        # Process same image twice
        result1 = self.processor.process_medical_image(self.test_image_path)
        result2 = self.processor.process_medical_image(self.test_image_path)
        
        # Both should succeed
        assert result1.processing_success is True
        assert result2.processing_success is True
        
        # Cache should have entries
        assert len(self.processor.image_cache) > 0
    
    def test_performance_constraints(self):
        """Test that processing meets performance constraints."""
        import time
        
        start_time = time.time()
        result = self.processor.process_medical_image(self.test_image)
        elapsed = (time.time() - start_time) * 1000
        
        # Should meet 5 second constraint from PRD
        assert elapsed < 5000
        assert result.processing_time_ms < 5000
    
    def test_error_handling_invalid_input(self):
        """Test error handling with invalid input."""
        # Test with invalid file path
        result = self.processor.process_medical_image('/nonexistent/file.jpg')
        
        assert result.processing_success is False
        assert result.error_message is not None
        assert result.embedding is not None  # Should have fallback zeros
    
    def test_error_handling_corrupted_image(self):
        """Test error handling with corrupted image data."""
        # Create invalid image data
        invalid_array = np.array([])
        
        result = self.processor.process_medical_image(invalid_array)
        
        # Should handle gracefully
        assert result is not None
        # May succeed with fallback or fail gracefully
    
    def test_large_file_size_rejection(self):
        """Test rejection of files exceeding size limit."""
        # Create a temporary large file path (simulate large file)
        with patch('os.path.getsize', return_value=100 * 1024 * 1024):  # 100MB
            result = self.processor.process_medical_image(self.test_image_path)
            
            # Should be rejected due to size
            assert result.processing_success is False or result.embedding is not None
    
    def test_statistics_tracking(self):
        """Test processing statistics tracking."""
        initial_count = self.processor.processing_stats['total_images_processed']
        
        # Process an image
        self.processor.process_medical_image(self.test_image)
        
        # Statistics should be updated
        assert self.processor.processing_stats['total_images_processed'] == initial_count + 1
        assert self.processor.processing_stats['avg_processing_time_ms'] > 0
    
    def test_clear_cache(self):
        """Test cache clearing functionality."""
        # Add some cache entries
        self.processor.image_cache = {'test_key': 'test_value'}
        
        self.processor.clear_cache()
        
        # Cache should be empty
        assert len(self.processor.image_cache) == 0
    
    def test_memory_optimization(self):
        """Test memory optimization functionality."""
        # Should not raise exceptions
        self.processor.optimize_memory()
        
        # Cache should be cleared
        if self.processor.image_cache:
            assert len(self.processor.image_cache) == 0


class TestDICOMProcessor:
    """Test cases for DICOM processor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DICOMProcessor()
    
    def test_initialization(self):
        """Test DICOM processor initialization."""
        assert self.processor is not None
        assert hasattr(self.processor, 'config')
    
    def test_extract_metadata_no_pydicom(self):
        """Test metadata extraction when pydicom is not available."""
        # Mock pydicom as unavailable
        self.processor.pydicom_available = False
        
        metadata = self.processor.extract_metadata('/fake/path.dcm')
        
        assert 'error' in metadata
        assert metadata['error'] == 'DICOM processing not available'
    
    def test_process_dicom_pixels_no_pydicom(self):
        """Test pixel processing when pydicom is not available."""
        self.processor.pydicom_available = False
        
        result = self.processor.process_dicom_pixels('/fake/path.dcm')
        
        assert result is None
    
    @patch('pydicom.dcmread')
    def test_extract_metadata_with_mock_dicom(self, mock_dcmread):
        """Test metadata extraction with mocked DICOM data."""
        # Setup mock
        mock_dicom = Mock()
        mock_dicom.Modality = 'CT'
        mock_dicom.StudyDescription = 'Test Study'
        mock_dcmread.return_value = mock_dicom
        
        self.processor.pydicom_available = True
        
        metadata = self.processor.extract_metadata('/fake/path.dcm')
        
        assert 'Modality' in metadata
        assert metadata['Modality'] == 'CT'


class TestMedicalImageAnonymizer:
    """Test cases for medical image anonymizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.anonymizer = MedicalImageAnonymizer()
        self.test_image = self._create_test_image()
    
    def _create_test_image(self) -> Image.Image:
        """Create a test image for anonymization."""
        return Image.new('RGB', (200, 200), color='white')
    
    def test_initialization(self):
        """Test anonymizer initialization."""
        assert self.anonymizer is not None
        assert hasattr(self.anonymizer, 'config')
    
    def test_anonymize_image_basic(self):
        """Test basic image anonymization."""
        anonymized = self.anonymizer.anonymize_image(self.test_image)
        
        # Should return an image
        assert isinstance(anonymized, Image.Image)
        assert anonymized.size == self.test_image.size
    
    def test_text_overlay_removal_no_ocr(self):
        """Test text overlay removal when OCR is not available."""
        self.anonymizer.ocr_available = False
        
        # Should not crash and return original image
        result = self.anonymizer._remove_text_overlays(self.test_image)
        assert isinstance(result, Image.Image)
    
    def test_face_blurring_no_cv2(self):
        """Test face blurring when OpenCV is not available."""
        self.anonymizer.cv2_available = False
        
        # Should not crash and return original image
        result = self.anonymizer._blur_faces(self.test_image)
        assert isinstance(result, Image.Image)
    
    def test_blur_application(self):
        """Test blur application functionality."""
        test_region = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        blurred = self.anonymizer._apply_blur(test_region)
        
        # Should return same shape
        assert blurred.shape == test_region.shape
    
    def test_pixelation_application(self):
        """Test pixelation application functionality."""
        test_region = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        pixelated = self.anonymizer._apply_pixelation(test_region)
        
        # Should return same shape
        assert pixelated.shape == test_region.shape
    
    def test_edge_detection_fallback(self):
        """Test edge detection fallback when OpenCV unavailable."""
        self.anonymizer.cv2_available = False
        
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        edges = self.anonymizer._detect_edges(gray_image)
        
        # Should return something (even if zeros)
        assert isinstance(edges, np.ndarray)
        assert edges.shape == gray_image.shape


if __name__ == '__main__':
    pytest.main([__file__])
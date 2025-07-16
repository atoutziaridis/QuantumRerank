"""
Medical Image Processor with BiomedCLIP Integration.

This module implements comprehensive medical image processing with BiomedCLIP
for medical image encoding, privacy protection, and quantum compression
as specified in QMMR-04 task.

Based on:
- QMMR-04 task requirements
- BiomedCLIP research for medical image understanding
- Privacy protection for medical data (HIPAA compliance)
- Quantum-inspired compression techniques
"""

import numpy as np
import torch
import time
import logging
import os
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from PIL import Image, ImageEnhance, ImageFilter
import io

from ..config.medical_image_config import (
    MedicalImageConfig, DICOMProcessingConfig, ImagePrivacyConfig
)
from .quantum_compression import QuantumCompressionHead, QuantumCompressionConfig

logger = logging.getLogger(__name__)


@dataclass
class MedicalImageProcessingResult:
    """Result of medical image processing."""
    
    # Core embeddings
    embedding: Optional[np.ndarray] = None
    original_embedding: Optional[np.ndarray] = None
    
    # Processing metadata
    processing_time_ms: float = 0.0
    image_format: str = 'unknown'
    image_dimensions: Tuple[int, int] = (0, 0)
    
    # Medical metadata
    modality: str = 'unknown'
    body_part: str = 'unknown'
    study_description: str = ''
    
    # Privacy and security
    phi_removed: bool = False
    anonymized: bool = False
    
    # Quantum processing
    quantum_compressed: bool = False
    compression_ratio: float = 1.0
    
    # Quality metrics
    image_quality_score: float = 0.0
    processing_success: bool = False
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class DICOMProcessor:
    """Processor for DICOM medical images with privacy protection."""
    
    def __init__(self, config: DICOMProcessingConfig = None):
        self.config = config or DICOMProcessingConfig()
        
        # Try to import DICOM libraries
        self.pydicom_available = False
        self.nibabel_available = False
        
        try:
            import pydicom
            self.pydicom = pydicom
            self.pydicom_available = True
            logger.info("DICOM processing enabled with pydicom")
        except ImportError:
            logger.warning("pydicom not available, DICOM support limited")
        
        try:
            import nibabel
            self.nibabel = nibabel
            self.nibabel_available = True
            logger.info("NIfTI processing enabled with nibabel")
        except ImportError:
            logger.warning("nibabel not available, NIfTI support limited")
    
    def extract_metadata(self, dicom_path: str) -> Dict[str, Any]:
        """Extract metadata from DICOM file with privacy protection."""
        if not self.pydicom_available:
            return {'error': 'DICOM processing not available'}
        
        try:
            # Read DICOM file
            dicom_data = self.pydicom.dcmread(dicom_path)
            
            # Extract critical metadata
            metadata = {}
            
            for tag in self.config.critical_tags:
                try:
                    value = getattr(dicom_data, tag, 'unknown')
                    metadata[tag] = str(value) if value else 'unknown'
                except Exception:
                    metadata[tag] = 'unknown'
            
            # Privacy protection: remove patient identifiers
            if self.config.anonymize_dicom_tags:
                for identifier in self.config.remove_patient_identifiers:
                    metadata.pop(identifier, None)
            
            return metadata
            
        except Exception as e:
            logger.error(f"DICOM metadata extraction failed: {e}")
            return {'error': str(e)}
    
    def process_dicom_pixels(self, dicom_path: str) -> Optional[np.ndarray]:
        """Process DICOM pixel data with medical-specific preprocessing."""
        if not self.pydicom_available:
            return None
        
        try:
            # Read DICOM file
            dicom_data = self.pydicom.dcmread(dicom_path)
            
            # Extract pixel array
            pixel_array = dicom_data.pixel_array
            
            # Apply DICOM-specific processing
            if self.config.normalize_pixel_data:
                pixel_array = self._normalize_dicom_pixels(pixel_array, dicom_data)
            
            if self.config.apply_window_center and hasattr(dicom_data, 'WindowCenter'):
                pixel_array = self._apply_windowing(pixel_array, dicom_data)
            
            return pixel_array
            
        except Exception as e:
            logger.error(f"DICOM pixel processing failed: {e}")
            return None
    
    def _normalize_dicom_pixels(self, pixel_array: np.ndarray, dicom_data) -> np.ndarray:
        """Normalize DICOM pixel values to standard range."""
        # Apply rescale slope and intercept if available
        if hasattr(dicom_data, 'RescaleSlope') and hasattr(dicom_data, 'RescaleIntercept'):
            slope = float(dicom_data.RescaleSlope)
            intercept = float(dicom_data.RescaleIntercept)
            pixel_array = pixel_array * slope + intercept
        
        # Normalize to 0-255 range for standard image processing
        pixel_min = np.min(pixel_array)
        pixel_max = np.max(pixel_array)
        
        if pixel_max > pixel_min:
            pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
        else:
            pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
        
        return pixel_array
    
    def _apply_windowing(self, pixel_array: np.ndarray, dicom_data) -> np.ndarray:
        """Apply DICOM windowing for optimal visualization."""
        try:
            window_center = float(dicom_data.WindowCenter)
            window_width = float(dicom_data.WindowWidth)
            
            # Apply windowing
            window_min = window_center - window_width / 2
            window_max = window_center + window_width / 2
            
            # Clip values to window range
            windowed = np.clip(pixel_array, window_min, window_max)
            
            # Normalize to 0-255
            windowed = ((windowed - window_min) / (window_max - window_min) * 255).astype(np.uint8)
            
            return windowed
            
        except Exception as e:
            logger.warning(f"DICOM windowing failed: {e}")
            return pixel_array


class MedicalImageAnonymizer:
    """Anonymizer for medical images to remove PHI and protect privacy."""
    
    def __init__(self, config: ImagePrivacyConfig = None):
        self.config = config or ImagePrivacyConfig()
        
        # Initialize text detection if available
        self.ocr_available = False
        try:
            import pytesseract
            self.tesseract = pytesseract
            self.ocr_available = True
            logger.info("OCR enabled for text detection")
        except ImportError:
            logger.warning("pytesseract not available, text detection limited")
        
        # Initialize face detection if available
        self.cv2_available = False
        try:
            import cv2
            self.cv2 = cv2
            self.cv2_available = True
            logger.info("OpenCV enabled for face detection")
        except ImportError:
            logger.warning("OpenCV not available, face detection disabled")
    
    def anonymize_image(self, image: Image.Image) -> Image.Image:
        """Anonymize medical image by removing PHI."""
        try:
            anonymized_image = image.copy()
            
            # Remove text overlays
            if self.config.remove_text_overlays:
                anonymized_image = self._remove_text_overlays(anonymized_image)
            
            # Remove burned-in annotations
            if self.config.remove_burned_in_annotations:
                anonymized_image = self._remove_annotations(anonymized_image)
            
            # Blur patient faces (if applicable)
            if self.config.blur_patient_faces:
                anonymized_image = self._blur_faces(anonymized_image)
            
            return anonymized_image
            
        except Exception as e:
            logger.error(f"Image anonymization failed: {e}")
            return image  # Return original if anonymization fails
    
    def _remove_text_overlays(self, image: Image.Image) -> Image.Image:
        """Remove text overlays from medical images."""
        if not self.ocr_available:
            return image
        
        try:
            # Convert to array for processing
            image_array = np.array(image)
            
            # Detect text using OCR
            text_data = self.tesseract.image_to_data(
                image_array, 
                output_type=self.tesseract.Output.DICT
            )
            
            # Remove detected text regions
            for i, confidence in enumerate(text_data['conf']):
                if int(confidence) > self.config.text_detection_threshold * 100:
                    x = text_data['left'][i]
                    y = text_data['top'][i]
                    w = text_data['width'][i]
                    h = text_data['height'][i]
                    
                    # Apply anonymization method
                    if self.config.text_replacement_method == 'blur':
                        region = image_array[y:y+h, x:x+w]
                        blurred_region = self._apply_blur(region)
                        image_array[y:y+h, x:x+w] = blurred_region
                    elif self.config.text_replacement_method == 'black_box':
                        image_array[y:y+h, x:x+w] = 0
                    elif self.config.text_replacement_method == 'white_box':
                        image_array[y:y+h, x:x+w] = 255
            
            return Image.fromarray(image_array)
            
        except Exception as e:
            logger.warning(f"Text overlay removal failed: {e}")
            return image
    
    def _remove_annotations(self, image: Image.Image) -> Image.Image:
        """Remove burned-in annotations from medical images."""
        # Simple approach: detect high-contrast text-like regions
        try:
            image_array = np.array(image)
            
            # Convert to grayscale for processing
            if len(image_array.shape) == 3:
                gray = np.mean(image_array, axis=2).astype(np.uint8)
            else:
                gray = image_array
            
            # Detect high-contrast regions (potential annotations)
            edges = self._detect_edges(gray)
            
            # Find contours of potential annotations
            contours = self._find_annotation_contours(edges)
            
            # Remove detected annotation regions
            for contour in contours:
                x, y, w, h = self._get_bounding_box(contour)
                
                # Apply anonymization
                if self.config.text_replacement_method == 'blur':
                    region = image_array[y:y+h, x:x+w]
                    blurred_region = self._apply_blur(region)
                    image_array[y:y+h, x:x+w] = blurred_region
                else:
                    image_array[y:y+h, x:x+w] = 0
            
            return Image.fromarray(image_array)
            
        except Exception as e:
            logger.warning(f"Annotation removal failed: {e}")
            return image
    
    def _blur_faces(self, image: Image.Image) -> Image.Image:
        """Blur patient faces in medical images."""
        if not self.cv2_available:
            return image
        
        try:
            image_array = np.array(image)
            
            # Convert to grayscale for face detection
            if len(image_array.shape) == 3:
                gray = self.cv2.cvtColor(image_array, self.cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
            
            # Load face cascade classifier
            face_cascade = self.cv2.CascadeClassifier(
                self.cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            # Blur detected faces
            for (x, y, w, h) in faces:
                face_region = image_array[y:y+h, x:x+w]
                
                if self.config.face_anonymization_method == 'blur':
                    blurred_face = self._apply_blur(face_region, kernel_size=21)
                    image_array[y:y+h, x:x+w] = blurred_face
                elif self.config.face_anonymization_method == 'pixelate':
                    pixelated_face = self._apply_pixelation(face_region, pixel_size=10)
                    image_array[y:y+h, x:x+w] = pixelated_face
                elif self.config.face_anonymization_method == 'black_box':
                    image_array[y:y+h, x:x+w] = 0
            
            return Image.fromarray(image_array)
            
        except Exception as e:
            logger.warning(f"Face blurring failed: {e}")
            return image
    
    def _apply_blur(self, region: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """Apply Gaussian blur to image region."""
        if not self.cv2_available:
            # Fallback: simple averaging
            return np.full_like(region, np.mean(region))
        
        return self.cv2.GaussianBlur(region, (kernel_size, kernel_size), 0)
    
    def _apply_pixelation(self, region: np.ndarray, pixel_size: int = 10) -> np.ndarray:
        """Apply pixelation effect to image region."""
        h, w = region.shape[:2]
        
        # Downsample then upsample to create pixelation effect
        small_region = region[::pixel_size, ::pixel_size]
        
        # Repeat pixels to match original size
        pixelated = np.repeat(np.repeat(small_region, pixel_size, axis=0), pixel_size, axis=1)
        
        # Crop to original size
        return pixelated[:h, :w]
    
    def _detect_edges(self, gray_image: np.ndarray) -> np.ndarray:
        """Detect edges in grayscale image."""
        if not self.cv2_available:
            # Simple edge detection fallback
            return np.zeros_like(gray_image)
        
        return self.cv2.Canny(gray_image, 50, 150)
    
    def _find_annotation_contours(self, edges: np.ndarray) -> List:
        """Find contours of potential annotations."""
        if not self.cv2_available:
            return []
        
        contours, _ = self.cv2.findContours(edges, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and aspect ratio (typical for text)
        filtered_contours = []
        for contour in contours:
            area = self.cv2.contourArea(contour)
            if 100 < area < 10000:  # Reasonable size for annotations
                x, y, w, h = self.cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.2 < aspect_ratio < 10:  # Reasonable aspect ratio for text
                    filtered_contours.append(contour)
        
        return filtered_contours
    
    def _get_bounding_box(self, contour) -> Tuple[int, int, int, int]:
        """Get bounding box of contour."""
        if not self.cv2_available:
            return (0, 0, 0, 0)
        
        return self.cv2.boundingRect(contour)


class MedicalImageProcessor:
    """
    Medical image processing with BiomedCLIP integration and quantum compression.
    
    Implements comprehensive medical image processing including privacy protection,
    DICOM support, and quantum-inspired compression for multimodal similarity.
    """
    
    def __init__(self, config: MedicalImageConfig = None):
        """
        Initialize medical image processor.
        
        Args:
            config: Medical image configuration
        """
        self.config = config or MedicalImageConfig()
        
        # Initialize BiomedCLIP components
        self.image_encoder = None
        self.image_processor = None
        self._load_biomedclip_encoder()
        
        # Initialize quantum compression
        self.quantum_compressor = None
        if self.config.quantum_compression_enabled:
            self._initialize_quantum_compression()
        
        # Initialize supporting processors
        self.dicom_processor = DICOMProcessor()
        self.image_anonymizer = MedicalImageAnonymizer()
        
        # Performance monitoring
        self.processing_stats = {
            'total_images_processed': 0,
            'avg_processing_time_ms': 0.0,
            'image_format_distribution': {},
            'error_rate': 0.0,
            'privacy_operations_count': 0,
            'quantum_compression_rate': 0.0
        }
        
        # Image cache for performance
        self.image_cache = {} if self.config.enable_image_cache else None
        
        logger.info(f"MedicalImageProcessor initialized with {self.config.image_encoder}")
    
    def _load_biomedclip_encoder(self):
        """Load BiomedCLIP image encoder with fallback."""
        try:
            from transformers import AutoModel, AutoProcessor
            
            # Try to load BiomedCLIP
            try:
                self.image_encoder = AutoModel.from_pretrained(self.config.image_encoder)
                self.image_processor = AutoProcessor.from_pretrained(self.config.image_encoder)
                logger.info(f"Successfully loaded BiomedCLIP: {self.config.image_encoder}")
            except Exception as e:
                logger.warning(f"Failed to load BiomedCLIP: {e}")
                logger.info(f"Falling back to: {self.config.fallback_encoder}")
                
                # Fallback to standard CLIP
                self.image_encoder = AutoModel.from_pretrained(self.config.fallback_encoder)
                self.image_processor = AutoProcessor.from_pretrained(self.config.fallback_encoder)
            
            # Set to evaluation mode
            self.image_encoder.eval()
            
        except Exception as e:
            logger.error(f"Failed to load image encoder: {e}")
            logger.info("Using basic CNN fallback")
            self._load_fallback_encoder()
    
    def _load_fallback_encoder(self):
        """Load basic CNN encoder as fallback."""
        try:
            import torchvision.models as models
            
            # Use ResNet as basic fallback
            self.image_encoder = models.resnet50(pretrained=True)
            self.image_encoder.eval()
            
            # Remove final classification layer
            self.image_encoder.fc = torch.nn.Identity()
            
            logger.info("Loaded ResNet50 as fallback encoder")
            
        except Exception as e:
            logger.error(f"Failed to load fallback encoder: {e}")
            self.image_encoder = None
            self.image_processor = None
    
    def _initialize_quantum_compression(self):
        """Initialize quantum compression for medical images."""
        try:
            compression_config = QuantumCompressionConfig(
                input_dim=self.config.image_embedding_dim,
                output_dim=self.config.image_quantum_target_dim,
                compression_stages=3,
                enable_bloch_parameterization=True
            )
            
            self.quantum_compressor = QuantumCompressionHead(compression_config)
            logger.info(f"Quantum compression initialized: "
                       f"{compression_config.input_dim} -> {compression_config.output_dim}")
            
        except Exception as e:
            logger.error(f"Failed to initialize quantum compression: {e}")
            self.quantum_compressor = None
    
    def process_medical_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> MedicalImageProcessingResult:
        """
        Process medical image with privacy protection and quantum compression.
        
        Args:
            image_input: Image file path, numpy array, or PIL Image
            
        Returns:
            MedicalImageProcessingResult with embeddings and metadata
        """
        start_time = time.time()
        result = MedicalImageProcessingResult()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(image_input)
            if self.image_cache and cache_key in self.image_cache:
                cached_result = self.image_cache[cache_key]
                cached_result.processing_time_ms = (time.time() - start_time) * 1000
                return cached_result
            
            # Load and validate image
            image, metadata = self._load_and_validate_image(image_input)
            if image is None:
                result.error_message = "Failed to load image"
                return result
            
            # Update result with basic metadata
            result.image_format = metadata.get('format', 'unknown')
            result.image_dimensions = image.size
            result.modality = metadata.get('modality', 'unknown')
            
            # Privacy protection
            if self.config.phi_removal or self.config.image_anonymization:
                image = self.image_anonymizer.anonymize_image(image)
                result.phi_removed = True
                result.anonymized = True
                self.processing_stats['privacy_operations_count'] += 1
            
            # Medical preprocessing
            if self.config.medical_image_preprocessing:
                image = self._apply_medical_preprocessing(image)
            
            # Generate embeddings
            if self.image_encoder is not None:
                original_embedding = self._encode_with_model(image)
                result.original_embedding = original_embedding
                
                # Quantum compression
                if self.quantum_compressor and original_embedding is not None:
                    compressed_embedding = self._apply_quantum_compression(original_embedding)
                    result.embedding = compressed_embedding
                    result.quantum_compressed = True
                    result.compression_ratio = len(original_embedding) / len(compressed_embedding)
                else:
                    result.embedding = original_embedding
                    result.quantum_compressed = False
                    result.compression_ratio = 1.0
            else:
                result.error_message = "No image encoder available"
                return result
            
            # Extract additional metadata
            result.study_description = metadata.get('StudyDescription', '')
            result.body_part = metadata.get('BodyPartExamined', 'unknown')
            
            # Quality assessment
            result.image_quality_score = self._assess_image_quality(image)
            
            # Performance tracking
            elapsed = (time.time() - start_time) * 1000
            result.processing_time_ms = elapsed
            result.processing_success = True
            
            # Update statistics
            self._update_processing_stats(elapsed, success=True, image_format=result.image_format)
            
            # Cache result
            if self.image_cache:
                self._cache_result(cache_key, result)
            
            # Check performance constraint
            if elapsed > 5000:  # 5 second constraint from PRD
                result.warnings.append(f"Processing time {elapsed:.0f}ms exceeds 5000ms constraint")
            
            return result
            
        except Exception as e:
            # Error handling
            elapsed = (time.time() - start_time) * 1000
            result.processing_time_ms = elapsed
            result.error_message = str(e)
            result.processing_success = False
            
            self._update_processing_stats(elapsed, success=False)
            
            logger.error(f"Medical image processing failed: {e}")
            
            # Return zero embedding if fallback is enabled
            if self.config.fallback_to_zeros:
                result.embedding = np.zeros(self.config.image_quantum_target_dim)
                result.processing_success = False
            
            return result
    
    def _load_and_validate_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """Load and validate image from various input types."""
        metadata = {}
        
        try:
            if isinstance(image_input, str):
                # File path
                if not os.path.exists(image_input):
                    logger.error(f"Image file not found: {image_input}")
                    return None, metadata
                
                # Check file size
                file_size_mb = os.path.getsize(image_input) / (1024 * 1024)
                if file_size_mb > self.config.max_image_file_size_mb:
                    logger.error(f"Image file too large: {file_size_mb:.1f}MB > {self.config.max_image_file_size_mb}MB")
                    return None, metadata
                
                # Extract file metadata
                file_ext = os.path.splitext(image_input)[1].lower()
                metadata['format'] = file_ext
                
                # Handle DICOM files
                if file_ext in ['.dicom', '.dcm']:
                    dicom_metadata = self.dicom_processor.extract_metadata(image_input)
                    metadata.update(dicom_metadata)
                    
                    # Process DICOM pixels
                    pixel_array = self.dicom_processor.process_dicom_pixels(image_input)
                    if pixel_array is not None:
                        # Convert to PIL Image
                        if len(pixel_array.shape) == 2:
                            image = Image.fromarray(pixel_array, mode='L')
                        else:
                            image = Image.fromarray(pixel_array)
                    else:
                        return None, metadata
                else:
                    # Standard image formats
                    image = Image.open(image_input)
                
            elif isinstance(image_input, np.ndarray):
                # NumPy array
                if len(image_input.shape) == 2:
                    image = Image.fromarray(image_input, mode='L')
                elif len(image_input.shape) == 3:
                    image = Image.fromarray(image_input)
                else:
                    logger.error(f"Invalid image array shape: {image_input.shape}")
                    return None, metadata
                
                metadata['format'] = 'array'
                
            elif isinstance(image_input, Image.Image):
                # PIL Image
                image = image_input.copy()
                metadata['format'] = 'pil'
                
            else:
                logger.error(f"Unsupported image input type: {type(image_input)}")
                return None, metadata
            
            # Validate image
            if image.size[0] == 0 or image.size[1] == 0:
                logger.error("Invalid image dimensions")
                return None, metadata
            
            return image, metadata
            
        except Exception as e:
            logger.error(f"Image loading failed: {e}")
            return None, metadata
    
    def _apply_medical_preprocessing(self, image: Image.Image) -> Image.Image:
        """Apply medical-specific image preprocessing."""
        try:
            # Resize to standard dimensions
            if image.size != self.config.max_image_size:
                image = image.resize(self.config.max_image_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                if image.mode == 'L':
                    # Convert grayscale to RGB
                    image = image.convert('RGB')
                elif image.mode == 'RGBA':
                    # Convert RGBA to RGB with white background
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1])
                    image = background
                else:
                    image = image.convert('RGB')
            
            # Medical image enhancements
            if self.config.histogram_equalization:
                image = self._apply_histogram_equalization(image)
            
            if self.config.contrast_enhancement:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.2)  # Slight contrast enhancement
            
            if self.config.noise_reduction:
                image = image.filter(ImageFilter.MedianFilter(size=3))
            
            return image
            
        except Exception as e:
            logger.warning(f"Medical preprocessing failed: {e}")
            return image
    
    def _apply_histogram_equalization(self, image: Image.Image) -> Image.Image:
        """Apply histogram equalization for medical images."""
        try:
            # Convert to numpy for processing
            image_array = np.array(image)
            
            if len(image_array.shape) == 3:
                # RGB image - apply to each channel
                for i in range(3):
                    channel = image_array[:, :, i]
                    hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
                    cdf = hist.cumsum()
                    cdf_normalized = cdf * 255 / cdf[-1]
                    image_array[:, :, i] = np.interp(channel.flatten(), bins[:-1], cdf_normalized).reshape(channel.shape)
            else:
                # Grayscale image
                hist, bins = np.histogram(image_array.flatten(), 256, [0, 256])
                cdf = hist.cumsum()
                cdf_normalized = cdf * 255 / cdf[-1]
                image_array = np.interp(image_array.flatten(), bins[:-1], cdf_normalized).reshape(image_array.shape)
            
            return Image.fromarray(image_array.astype(np.uint8))
            
        except Exception as e:
            logger.warning(f"Histogram equalization failed: {e}")
            return image
    
    def _encode_with_model(self, image: Image.Image) -> Optional[np.ndarray]:
        """Encode image using the loaded model."""
        try:
            if self.image_processor is not None:
                # BiomedCLIP or CLIP processing
                inputs = self.image_processor(images=image, return_tensors="pt")
                
                with torch.no_grad():
                    if hasattr(self.image_encoder, 'get_image_features'):
                        # CLIP-style model
                        outputs = self.image_encoder.get_image_features(**inputs)
                    else:
                        # Generic model
                        outputs = self.image_encoder(**inputs)
                        if hasattr(outputs, 'pooler_output'):
                            outputs = outputs.pooler_output
                        elif hasattr(outputs, 'last_hidden_state'):
                            outputs = outputs.last_hidden_state.mean(dim=1)
                        else:
                            outputs = outputs[0] if isinstance(outputs, tuple) else outputs
                
                # Convert to numpy
                embedding = outputs.squeeze().cpu().numpy()
                
            else:
                # Fallback: use basic CNN
                # Convert PIL to tensor
                import torchvision.transforms as transforms
                
                transform = transforms.Compose([
                    transforms.Resize(self.config.max_image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                image_tensor = transform(image).unsqueeze(0)
                
                with torch.no_grad():
                    embedding = self.image_encoder(image_tensor).squeeze().cpu().numpy()
            
            # Normalize embedding
            if len(embedding.shape) > 0 and np.linalg.norm(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Image encoding failed: {e}")
            return None
    
    def _apply_quantum_compression(self, embedding: np.ndarray) -> Optional[np.ndarray]:
        """Apply quantum compression to image embedding."""
        try:
            if self.quantum_compressor is None:
                return embedding
            
            # Convert to tensor
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
            
            # Apply compression
            with torch.no_grad():
                compressed_tensor = self.quantum_compressor(embedding_tensor.unsqueeze(0))
                compressed_embedding = compressed_tensor.squeeze(0).numpy()
            
            # Normalize compressed embedding
            if np.linalg.norm(compressed_embedding) > 0:
                compressed_embedding = compressed_embedding / np.linalg.norm(compressed_embedding)
            
            self.processing_stats['quantum_compression_rate'] += 1
            
            return compressed_embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Quantum compression failed: {e}")
            return embedding
    
    def _assess_image_quality(self, image: Image.Image) -> float:
        """Assess medical image quality."""
        try:
            # Convert to numpy for analysis
            image_array = np.array(image)
            
            # Calculate quality metrics
            # 1. Contrast measure
            if len(image_array.shape) == 3:
                gray = np.mean(image_array, axis=2)
            else:
                gray = image_array
            
            contrast = np.std(gray) / np.mean(gray) if np.mean(gray) > 0 else 0
            
            # 2. Sharpness measure (Laplacian variance)
            try:
                if hasattr(self, 'cv2') and self.cv2 is not None:
                    laplacian_var = self.cv2.Laplacian(gray.astype(np.uint8), self.cv2.CV_64F).var()
                    sharpness = min(laplacian_var / 1000, 1.0)  # Normalize
                else:
                    sharpness = 0.5  # Default
            except:
                sharpness = 0.5
            
            # 3. Brightness adequacy
            mean_brightness = np.mean(gray) / 255
            brightness_score = 1 - abs(mean_brightness - 0.5) * 2  # Optimal around 0.5
            
            # Combine metrics
            quality_score = (contrast * 0.4 + sharpness * 0.4 + brightness_score * 0.2)
            
            return float(np.clip(quality_score, 0, 1))
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return 0.5  # Default quality score
    
    def _get_cache_key(self, image_input: Union[str, np.ndarray, Image.Image]) -> str:
        """Generate cache key for image input."""
        try:
            if isinstance(image_input, str):
                # Use file path and modification time
                mtime = os.path.getmtime(image_input)
                return hashlib.md5(f"{image_input}_{mtime}".encode()).hexdigest()
            else:
                # Use image content hash
                if isinstance(image_input, Image.Image):
                    image_bytes = io.BytesIO()
                    image_input.save(image_bytes, format='PNG')
                    content = image_bytes.getvalue()
                else:
                    content = image_input.tobytes()
                
                return hashlib.md5(content).hexdigest()
        except:
            return "no_cache"
    
    def _cache_result(self, cache_key: str, result: MedicalImageProcessingResult):
        """Cache processing result with size management."""
        if len(self.image_cache) >= self.config.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.image_cache))
            del self.image_cache[oldest_key]
        
        self.image_cache[cache_key] = result
    
    def _update_processing_stats(self, elapsed_ms: float, success: bool, image_format: str = 'unknown'):
        """Update processing statistics."""
        self.processing_stats['total_images_processed'] += 1
        
        if success:
            # Update average processing time
            n = self.processing_stats['total_images_processed']
            current_avg = self.processing_stats['avg_processing_time_ms']
            self.processing_stats['avg_processing_time_ms'] = (
                (current_avg * (n - 1) + elapsed_ms) / n
            )
            
            # Update format distribution
            if image_format not in self.processing_stats['image_format_distribution']:
                self.processing_stats['image_format_distribution'][image_format] = 0
            self.processing_stats['image_format_distribution'][image_format] += 1
        
        # Update error rate
        total_processed = self.processing_stats['total_images_processed']
        if not success:
            current_errors = total_processed * self.processing_stats['error_rate']
            self.processing_stats['error_rate'] = (current_errors + 1) / total_processed
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        stats = self.processing_stats.copy()
        
        # Add cache statistics
        if self.image_cache:
            stats['cache_size'] = len(self.image_cache)
            stats['cache_enabled'] = True
        else:
            stats['cache_enabled'] = False
        
        # Add configuration info
        stats['quantum_compression_enabled'] = self.quantum_compressor is not None
        stats['biomedclip_enabled'] = self.image_processor is not None
        stats['privacy_protection_enabled'] = self.config.phi_removal or self.config.image_anonymization
        
        return stats
    
    def clear_cache(self):
        """Clear image processing cache."""
        if self.image_cache:
            self.image_cache.clear()
            logger.info("Image processing cache cleared")
    
    def optimize_memory(self):
        """Optimize memory usage."""
        if self.image_cache:
            self.image_cache.clear()
        
        # Run garbage collection
        import gc
        gc.collect()
        
        logger.info("Memory optimization completed for image processor")
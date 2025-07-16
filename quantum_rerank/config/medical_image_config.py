"""
Medical Image Configuration for Quantum Multimodal System.

Configuration for medical image processing with BiomedCLIP integration,
privacy protection, and quantum compression as specified in QMMR-04.

Based on:
- QMMR-04 task requirements
- BiomedCLIP integration specifications
- Medical privacy and security requirements
- Quantum compression constraints
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class MedicalImageConfig:
    """Configuration for medical image processing in quantum multimodal system."""
    
    # Image encoder configuration
    image_encoder: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    fallback_encoder: str = "openai/clip-vit-base-patch16"  # Fallback if BiomedCLIP unavailable
    image_embedding_dim: int = 512
    max_image_size: Tuple[int, int] = (224, 224)
    
    # Supported medical image formats
    supported_formats: List[str] = field(default_factory=lambda: [
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
        '.dicom', '.dcm', '.nii', '.nifti'
    ])
    
    # Processing constraints
    max_image_file_size_mb: int = 50
    image_processing_timeout_seconds: int = 10
    batch_processing_size: int = 20
    
    # Quantum integration
    image_quantum_compression_ratio: float = 4.0  # 512D -> 128D
    image_quantum_target_dim: int = 128
    quantum_compression_enabled: bool = True
    
    # Medical-specific preprocessing
    medical_image_preprocessing: bool = True
    histogram_equalization: bool = True
    noise_reduction: bool = True
    contrast_enhancement: bool = True
    
    # DICOM processing
    dicom_metadata_extraction: bool = True
    dicom_pixel_data_normalization: bool = True
    dicom_window_leveling: bool = True
    
    # Medical image augmentation (conservative for clinical accuracy)
    medical_image_augmentation: bool = False
    augmentation_probability: float = 0.1
    rotation_range: float = 5.0  # degrees
    brightness_range: float = 0.1
    
    # Privacy and security
    phi_removal: bool = True
    image_anonymization: bool = True
    secure_processing: bool = True
    metadata_scrubbing: bool = True
    
    # Performance optimization
    enable_image_cache: bool = True
    cache_size: int = 1000
    enable_gpu_processing: bool = True
    memory_optimization: bool = True
    
    # Error handling
    retry_failed_processing: bool = True
    max_retry_attempts: int = 3
    fallback_to_zeros: bool = True
    
    # Logging and monitoring
    enable_processing_stats: bool = True
    log_processing_errors: bool = True
    performance_monitoring: bool = True


@dataclass
class DICOMProcessingConfig:
    """Configuration for DICOM medical image processing."""
    
    # DICOM tag extraction
    extract_patient_info: bool = False  # Privacy: disabled by default
    extract_study_info: bool = True
    extract_series_info: bool = True
    extract_image_info: bool = True
    
    # DICOM modality support
    supported_modalities: List[str] = field(default_factory=lambda: [
        'CT', 'MR', 'XR', 'US', 'MG', 'DX', 'CR', 'DR',
        'PT', 'NM', 'RF', 'SC', 'OT'
    ])
    
    # DICOM processing
    normalize_pixel_data: bool = True
    apply_window_center: bool = True
    apply_window_width: bool = True
    convert_to_hounsfield: bool = True
    
    # Privacy protection
    anonymize_dicom_tags: bool = True
    remove_private_tags: bool = True
    remove_patient_identifiers: List[str] = field(default_factory=lambda: [
        'PatientName', 'PatientID', 'PatientBirthDate',
        'PatientSex', 'PatientAge', 'PatientWeight',
        'PatientAddress', 'PatientTelephoneNumbers'
    ])
    
    # Metadata extraction priorities
    critical_tags: List[str] = field(default_factory=lambda: [
        'Modality', 'StudyDescription', 'SeriesDescription',
        'BodyPartExamined', 'ViewPosition', 'ImageType',
        'SliceThickness', 'PixelSpacing', 'WindowCenter', 'WindowWidth'
    ])


@dataclass
class ImagePrivacyConfig:
    """Configuration for medical image privacy protection."""
    
    # PHI removal
    remove_text_overlays: bool = True
    remove_burned_in_annotations: bool = True
    detect_patient_identifiers: bool = True
    
    # Image anonymization
    blur_patient_faces: bool = True
    remove_metadata_tags: bool = True
    scramble_unique_identifiers: bool = True
    
    # Detection algorithms
    text_detection_threshold: float = 0.8
    face_detection_threshold: float = 0.7
    identifier_detection_models: List[str] = field(default_factory=lambda: [
        'tesseract',  # OCR for text detection
        'opencv'      # Basic face detection
    ])
    
    # Anonymization methods
    text_replacement_method: str = 'blur'  # 'blur', 'black_box', 'white_box'
    face_anonymization_method: str = 'blur'  # 'blur', 'pixelate', 'black_box'
    
    # Security measures
    secure_temp_storage: bool = True
    immediate_cleanup: bool = True
    audit_processing_steps: bool = True


@dataclass
class CrossModalAttentionConfig:
    """Configuration for cross-modal attention between images and text."""
    
    # Attention mechanism
    attention_type: str = 'multiplicative'  # 'additive', 'multiplicative', 'scaled_dot_product'
    attention_heads: int = 8
    attention_dropout: float = 0.1
    
    # Dimension mapping
    image_attention_dim: int = 64
    text_attention_dim: int = 64
    combined_attention_dim: int = 128
    
    # Cross-modal fusion
    fusion_method: str = 'concatenate'  # 'concatenate', 'element_wise', 'weighted_sum'
    learned_fusion_weights: bool = True
    
    # Training parameters
    attention_learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clipping: float = 1.0


@dataclass
class ImageTextSimilarityConfig:
    """Configuration for image-text quantum similarity computation."""
    
    # Quantum circuit parameters
    n_qubits: int = 4
    max_circuit_depth: int = 15
    quantum_shots: int = 1024
    
    # Similarity computation
    similarity_metric: str = 'quantum_fidelity'  # 'quantum_fidelity', 'cosine', 'euclidean'
    enable_cross_modal_entanglement: bool = True
    entanglement_strength: float = 0.8
    
    # Performance optimization
    enable_circuit_optimization: bool = True
    cache_quantum_states: bool = True
    batch_quantum_processing: bool = True
    
    # Uncertainty quantification
    enable_uncertainty_quantification: bool = True
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    bootstrap_samples: int = 1000


def get_default_medical_image_config() -> MedicalImageConfig:
    """Get default medical image configuration."""
    return MedicalImageConfig()


def get_optimized_config_for_performance() -> MedicalImageConfig:
    """Get configuration optimized for performance."""
    config = MedicalImageConfig()
    
    # Performance optimizations
    config.image_quantum_target_dim = 64  # Smaller for faster processing
    config.quantum_compression_ratio = 8.0  # Higher compression
    config.batch_processing_size = 50  # Larger batches
    config.enable_gpu_processing = True
    config.memory_optimization = True
    
    # Reduced preprocessing for speed
    config.medical_image_preprocessing = True
    config.histogram_equalization = False  # Skip for speed
    config.noise_reduction = False  # Skip for speed
    
    # Larger cache for better performance
    config.cache_size = 2000
    
    logger.info("Created performance-optimized medical image configuration")
    return config


def get_privacy_focused_config() -> MedicalImageConfig:
    """Get configuration focused on maximum privacy protection."""
    config = MedicalImageConfig()
    
    # Maximum privacy settings
    config.phi_removal = True
    config.image_anonymization = True
    config.secure_processing = True
    config.metadata_scrubbing = True
    
    # Disable caching for privacy
    config.enable_image_cache = False
    
    # Enhanced DICOM privacy
    dicom_config = DICOMProcessingConfig()
    dicom_config.extract_patient_info = False
    dicom_config.anonymize_dicom_tags = True
    dicom_config.remove_private_tags = True
    
    # Enhanced image privacy
    privacy_config = ImagePrivacyConfig()
    privacy_config.remove_text_overlays = True
    privacy_config.remove_burned_in_annotations = True
    privacy_config.blur_patient_faces = True
    privacy_config.audit_processing_steps = True
    
    logger.info("Created privacy-focused medical image configuration")
    return config


def validate_medical_image_config(config: MedicalImageConfig) -> Dict[str, bool]:
    """
    Validate medical image configuration for consistency and compliance.
    
    Args:
        config: Medical image configuration to validate
        
    Returns:
        Dictionary of validation results
    """
    validation_results = {}
    
    # Check image dimensions
    validation_results['image_size_valid'] = (
        config.max_image_size[0] > 0 and config.max_image_size[1] > 0
    )
    
    # Check quantum compression ratio
    validation_results['compression_ratio_valid'] = (
        1.0 <= config.image_quantum_compression_ratio <= 10.0
    )
    
    # Check target dimensions
    validation_results['target_dim_valid'] = (
        16 <= config.image_quantum_target_dim <= 512
    )
    
    # Check file size limits
    validation_results['file_size_limit_valid'] = (
        1 <= config.max_image_file_size_mb <= 100
    )
    
    # Check timeout settings
    validation_results['timeout_valid'] = (
        1 <= config.image_processing_timeout_seconds <= 60
    )
    
    # Check batch size
    validation_results['batch_size_valid'] = (
        1 <= config.batch_processing_size <= 100
    )
    
    # Check privacy settings consistency
    validation_results['privacy_settings_consistent'] = (
        config.secure_processing if config.phi_removal else True
    )
    
    # Overall validation
    validation_results['overall_valid'] = all(validation_results.values())
    
    return validation_results
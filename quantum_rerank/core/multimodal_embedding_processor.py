"""
Multimodal Embedding Processor for QuantumRerank.

This module extends the existing EmbeddingProcessor to handle multimodal medical data
(text + clinical data) while maintaining all PRD performance constraints.

Based on:
- QMMR-01 task requirements
- Research insights from QPMeL (32x parameter efficiency)
- Quantum-inspired embeddings projection techniques
- PRD performance constraints (<100ms, <2GB memory)
"""

import numpy as np
import torch
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import gc
import json

from .embeddings import EmbeddingProcessor, EmbeddingConfig
from ..config.multimodal_config import MultimodalMedicalConfig, ClinicalDataConfig
from .quantum_compression import QuantumCompressionHead, QuantumCompressionConfig

logger = logging.getLogger(__name__)


@dataclass
class MultimodalEmbeddingResult:
    """Result of multimodal embedding processing."""
    text_embedding: Optional[np.ndarray] = None
    clinical_embedding: Optional[np.ndarray] = None
    fused_embedding: Optional[np.ndarray] = None
    processing_time_ms: float = 0.0
    modalities_used: List[str] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.modalities_used is None:
            self.modalities_used = []


class ClinicalDataProcessor:
    """
    Processes clinical data for embedding generation.
    
    Handles structured clinical data including demographics, vitals, lab results,
    medications, procedures, diagnoses, symptoms, and allergies.
    """
    
    def __init__(self, config: ClinicalDataConfig = None):
        self.config = config or ClinicalDataConfig()
        
        # Initialize clinical encoder
        self.clinical_encoder = None
        self._load_clinical_encoder()
        
        # Medical entity extraction (optional)
        self.entity_extractor = None
        if self.config.extract_medical_entities:
            self._load_entity_extractor()
        
        # Performance monitoring
        self.processing_stats = {
            'total_processed': 0,
            'avg_processing_time_ms': 0.0,
            'error_count': 0
        }
        
        logger.info(f"ClinicalDataProcessor initialized with {self.config.encoder_model}")
    
    def _load_clinical_encoder(self):
        """Load clinical BERT encoder with fallback."""
        try:
            from transformers import AutoTokenizer, AutoModel
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.encoder_model)
            self.clinical_encoder = AutoModel.from_pretrained(self.config.encoder_model)
            
            # Set to evaluation mode
            self.clinical_encoder.eval()
            
            logger.info(f"Successfully loaded clinical encoder: {self.config.encoder_model}")
            
        except Exception as e:
            logger.error(f"Failed to load clinical encoder: {e}")
            logger.info("Using fallback sentence transformer for clinical data")
            
            # Fallback to sentence transformer
            from sentence_transformers import SentenceTransformer
            self.clinical_encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            self.tokenizer = None
    
    def _load_entity_extractor(self):
        """Load medical entity extractor (optional)."""
        try:
            import spacy
            self.entity_extractor = spacy.load(self.config.entity_model)
            logger.info(f"Loaded medical entity extractor: {self.config.entity_model}")
        except Exception as e:
            logger.warning(f"Failed to load entity extractor: {e}")
            self.entity_extractor = None
    
    def process_clinical_data(self, clinical_data: Union[Dict, str]) -> Dict[str, Any]:
        """
        Process clinical data into structured format.
        
        Args:
            clinical_data: Clinical data as dict or string
            
        Returns:
            Processed clinical data with extracted features
        """
        start_time = time.time()
        
        try:
            # Convert to string if dict
            if isinstance(clinical_data, dict):
                clinical_text = self._dict_to_clinical_text(clinical_data)
            else:
                clinical_text = str(clinical_data)
            
            # Extract medical entities if enabled
            entities = []
            if self.entity_extractor and clinical_text:
                entities = self._extract_medical_entities(clinical_text)
            
            # Expand abbreviations if enabled
            if self.config.expand_abbreviations:
                clinical_text = self._expand_clinical_abbreviations(clinical_text)
            
            # Normalize clinical text
            if self.config.normalize_values:
                clinical_text = self._normalize_clinical_text(clinical_text)
            
            # Update processing stats
            elapsed = (time.time() - start_time) * 1000
            self._update_processing_stats(elapsed, success=True)
            
            return {
                'processed_text': clinical_text,
                'entities': entities,
                'processing_time_ms': elapsed,
                'success': True
            }
            
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            self._update_processing_stats(elapsed, success=False)
            
            logger.error(f"Clinical data processing failed: {e}")
            return {
                'processed_text': "",
                'entities': [],
                'processing_time_ms': elapsed,
                'success': False,
                'error': str(e)
            }
    
    def encode_clinical_data(self, clinical_data: Union[Dict, str]) -> np.ndarray:
        """
        Encode clinical data to embedding vector.
        
        Args:
            clinical_data: Clinical data to encode
            
        Returns:
            Clinical embedding vector
        """
        # Process clinical data
        processed = self.process_clinical_data(clinical_data)
        
        if not processed['success']:
            # Return zero embedding on failure
            return np.zeros(768)  # Default BERT dimension
        
        clinical_text = processed['processed_text']
        
        if not clinical_text.strip():
            return np.zeros(768)
        
        try:
            # Generate embedding
            if self.tokenizer:  # Using transformers
                inputs = self.tokenizer(
                    clinical_text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.config.max_sequence_length
                )
                
                with torch.no_grad():
                    outputs = self.clinical_encoder(**inputs)
                    # Use [CLS] token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            else:  # Using sentence transformer
                embedding = self.clinical_encoder.encode(clinical_text)
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Clinical embedding generation failed: {e}")
            return np.zeros(768)
    
    def _dict_to_clinical_text(self, clinical_dict: Dict) -> str:
        """Convert clinical data dict to text representation."""
        text_parts = []
        
        # Process different clinical data types
        for data_type in self.config.supported_data_types:
            if data_type in clinical_dict:
                value = clinical_dict[data_type]
                if isinstance(value, dict):
                    # Flatten nested dict
                    for k, v in value.items():
                        text_parts.append(f"{data_type} {k}: {v}")
                elif isinstance(value, list):
                    # Join list items
                    text_parts.append(f"{data_type}: {', '.join(map(str, value))}")
                else:
                    text_parts.append(f"{data_type}: {value}")
        
        return ". ".join(text_parts)
    
    def _extract_medical_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract medical entities from text."""
        if not self.entity_extractor:
            return []
        
        try:
            doc = self.entity_extractor(text)
            entities = []
            
            for ent in doc.ents:
                if ent.label_ in self.config.entity_types:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': getattr(ent, 'score', 1.0)
                    })
            
            return entities
            
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []
    
    def _expand_clinical_abbreviations(self, text: str) -> str:
        """Expand clinical abbreviations in text."""
        # Common clinical abbreviations
        abbreviations = {
            'BP': 'blood pressure',
            'HR': 'heart rate',
            'RR': 'respiratory rate',
            'T': 'temperature',
            'BMI': 'body mass index',
            'CBC': 'complete blood count',
            'BUN': 'blood urea nitrogen',
            'Cr': 'creatinine',
            'Hgb': 'hemoglobin',
            'Hct': 'hematocrit',
            'WBC': 'white blood cell count',
            'RBC': 'red blood cell count',
            'PLT': 'platelet count',
            **self.config.custom_abbreviations
        }
        
        expanded_text = text
        for abbrev, expansion in abbreviations.items():
            expanded_text = expanded_text.replace(abbrev, f"{abbrev} ({expansion})")
        
        return expanded_text
    
    def _normalize_clinical_text(self, text: str) -> str:
        """Normalize clinical text for consistency."""
        # Basic normalization
        normalized = text.lower()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        # Standardize units
        unit_standardization = {
            'mg/dl': 'mg/dL',
            'mmol/l': 'mmol/L',
            'beats/min': 'bpm',
            'breaths/min': 'breaths per minute'
        }
        
        for old_unit, new_unit in unit_standardization.items():
            normalized = normalized.replace(old_unit, new_unit)
        
        return normalized
    
    def _update_processing_stats(self, elapsed_ms: float, success: bool):
        """Update processing statistics."""
        self.processing_stats['total_processed'] += 1
        
        if success:
            # Update average processing time
            n = self.processing_stats['total_processed']
            current_avg = self.processing_stats['avg_processing_time_ms']
            self.processing_stats['avg_processing_time_ms'] = (
                (current_avg * (n - 1) + elapsed_ms) / n
            )
        else:
            self.processing_stats['error_count'] += 1


class MultimodalEmbeddingProcessor(EmbeddingProcessor):
    """
    Extends EmbeddingProcessor to handle multimodal medical data
    while maintaining PRD performance constraints.
    
    Combines text and clinical data processing with quantum compression
    for efficient multimodal similarity computation.
    """
    
    def __init__(self, config: MultimodalMedicalConfig = None):
        """
        Initialize multimodal embedding processor.
        
        Args:
            config: Multimodal configuration
        """
        # Initialize base embedding processor
        super().__init__()
        
        # Initialize multimodal config
        self.multimodal_config = config or MultimodalMedicalConfig()
        
        # Initialize clinical data processor
        clinical_config = ClinicalDataConfig()
        self.clinical_processor = ClinicalDataProcessor(clinical_config)
        
        # Initialize quantum compression for multimodal fusion
        self.quantum_compressor = None
        self._initialize_quantum_compression()
        
        # Embedding cache for performance
        self.embedding_cache = {} if self.multimodal_config.enable_embedding_cache else None
        
        # Performance monitoring
        self.multimodal_stats = {
            'total_multimodal_processed': 0,
            'avg_multimodal_time_ms': 0.0,
            'text_only_count': 0,
            'clinical_only_count': 0,
            'full_multimodal_count': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info(f"MultimodalEmbeddingProcessor initialized: "
                   f"text_dim={self.multimodal_config.text_dim}, "
                   f"clinical_dim={self.multimodal_config.clinical_dim}, "
                   f"target_quantum_dim={self.multimodal_config.target_quantum_dim}")
    
    def _initialize_quantum_compression(self):
        """Initialize quantum compression for multimodal fusion."""
        try:
            # Create quantum compression config
            compression_config = QuantumCompressionConfig(
                input_dim=self.multimodal_config.text_dim + self.multimodal_config.clinical_dim,
                output_dim=self.multimodal_config.target_quantum_dim,
                compression_stages=3,
                enable_bloch_parameterization=True
            )
            
            self.quantum_compressor = QuantumCompressionHead(compression_config)
            logger.info(f"Quantum compression initialized: "
                       f"{compression_config.input_dim} -> {compression_config.output_dim}")
            
        except Exception as e:
            logger.error(f"Failed to initialize quantum compression: {e}")
            self.quantum_compressor = None
    
    def encode_multimodal(self, data: Dict[str, Any]) -> MultimodalEmbeddingResult:
        """
        Encode multimodal medical data maintaining <100ms constraint.
        
        Args:
            data: Dictionary containing 'text' and/or 'clinical_data'
            
        Returns:
            MultimodalEmbeddingResult with embeddings and metadata
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(data)
        if self.embedding_cache and cache_key in self.embedding_cache:
            self.multimodal_stats['cache_hits'] += 1
            cached_result = self.embedding_cache[cache_key]
            cached_result.processing_time_ms = (time.time() - start_time) * 1000
            return cached_result
        
        self.multimodal_stats['cache_misses'] += 1
        
        try:
            result = MultimodalEmbeddingResult()
            
            # Process text embedding
            if 'text' in data and data['text']:
                result.text_embedding = self.encode_single_text(data['text'])
                result.modalities_used.append('text')
            
            # Process clinical data embedding
            if 'clinical_data' in data and data['clinical_data']:
                result.clinical_embedding = self.clinical_processor.encode_clinical_data(
                    data['clinical_data']
                )
                result.modalities_used.append('clinical')
            
            # Handle missing modalities
            if not result.modalities_used:
                result.error_message = "No valid modalities found in input data"
                return result
            
            # Create fused embedding
            result.fused_embedding = self._create_fused_embedding(
                result.text_embedding, result.clinical_embedding
            )
            
            # Calculate processing time
            elapsed = (time.time() - start_time) * 1000
            result.processing_time_ms = elapsed
            
            # Check performance constraint
            if elapsed > self.multimodal_config.max_latency_ms:
                logger.warning(f"Multimodal encoding exceeded latency: {elapsed:.2f}ms")
            
            # Update statistics
            self._update_multimodal_stats(result)
            
            # Cache result
            if self.embedding_cache:
                self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            logger.error(f"Multimodal encoding failed: {e}")
            
            return MultimodalEmbeddingResult(
                processing_time_ms=elapsed,
                error_message=str(e)
            )
    
    def _create_fused_embedding(self, 
                               text_embedding: Optional[np.ndarray],
                               clinical_embedding: Optional[np.ndarray]) -> np.ndarray:
        """
        Create fused embedding from text and clinical embeddings.
        
        Args:
            text_embedding: Text embedding vector
            clinical_embedding: Clinical embedding vector
            
        Returns:
            Fused embedding vector
        """
        # Handle missing modalities
        if text_embedding is None and clinical_embedding is None:
            return np.zeros(self.multimodal_config.target_quantum_dim)
        
        if text_embedding is None:
            text_embedding = np.zeros(self.multimodal_config.text_dim)
        
        if clinical_embedding is None:
            clinical_embedding = np.zeros(self.multimodal_config.clinical_dim)
        
        # Concatenate embeddings
        combined_embedding = np.concatenate([text_embedding, clinical_embedding])
        
        # Apply quantum compression if available
        if self.quantum_compressor:
            try:
                combined_tensor = torch.tensor(combined_embedding, dtype=torch.float32)
                compressed_tensor = self.quantum_compressor(combined_tensor.unsqueeze(0))
                fused_embedding = compressed_tensor.squeeze(0).detach().numpy()
            except Exception as e:
                logger.error(f"Quantum compression failed: {e}")
                # Fallback to simple dimensionality reduction
                fused_embedding = self._fallback_fusion(combined_embedding)
        else:
            fused_embedding = self._fallback_fusion(combined_embedding)
        
        # Normalize fused embedding
        fused_embedding = fused_embedding / np.linalg.norm(fused_embedding)
        
        return fused_embedding
    
    def _fallback_fusion(self, combined_embedding: np.ndarray) -> np.ndarray:
        """Fallback fusion method when quantum compression fails."""
        # Simple PCA-like reduction
        target_dim = self.multimodal_config.target_quantum_dim
        input_dim = len(combined_embedding)
        
        if input_dim <= target_dim:
            # Pad with zeros if needed
            padding = target_dim - input_dim
            return np.pad(combined_embedding, (0, padding), mode='constant')
        else:
            # Simple truncation (could be improved with learned projection)
            return combined_embedding[:target_dim]
    
    def encode_multimodal_batch(self, 
                               data_batch: List[Dict[str, Any]]) -> List[MultimodalEmbeddingResult]:
        """
        Efficiently encode batch of multimodal data.
        
        Args:
            data_batch: List of multimodal data dictionaries
            
        Returns:
            List of multimodal embedding results
        """
        start_time = time.time()
        
        # Separate text and clinical data for batch processing
        text_data = []
        clinical_data = []
        
        for data in data_batch:
            text_data.append(data.get('text', ''))
            clinical_data.append(data.get('clinical_data', ''))
        
        # Batch encode text data
        text_embeddings = []
        valid_texts = [text for text in text_data if text]
        if valid_texts:
            text_embeddings_batch = self.encode_texts(valid_texts)
            text_idx = 0
            
            for text in text_data:
                if text:
                    text_embeddings.append(text_embeddings_batch[text_idx])
                    text_idx += 1
                else:
                    text_embeddings.append(None)
        else:
            text_embeddings = [None] * len(data_batch)
        
        # Process clinical data (sequential for now)
        clinical_embeddings = []
        for clinical in clinical_data:
            if clinical:
                clinical_embeddings.append(
                    self.clinical_processor.encode_clinical_data(clinical)
                )
            else:
                clinical_embeddings.append(None)
        
        # Create results
        results = []
        for i, (text_emb, clinical_emb) in enumerate(zip(text_embeddings, clinical_embeddings)):
            result = MultimodalEmbeddingResult()
            result.text_embedding = text_emb
            result.clinical_embedding = clinical_emb
            
            # Determine modalities used
            if text_emb is not None:
                result.modalities_used.append('text')
            if clinical_emb is not None:
                result.modalities_used.append('clinical')
            
            # Create fused embedding
            result.fused_embedding = self._create_fused_embedding(text_emb, clinical_emb)
            
            results.append(result)
        
        # Add batch timing
        batch_time = (time.time() - start_time) * 1000
        for result in results:
            result.processing_time_ms = batch_time / len(results)
        
        return results
    
    def _get_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate cache key for multimodal data."""
        import hashlib
        
        # Create deterministic string representation
        key_data = {
            'text': data.get('text', ''),
            'clinical_data': str(data.get('clinical_data', ''))
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _cache_result(self, cache_key: str, result: MultimodalEmbeddingResult):
        """Cache embedding result with size management."""
        if len(self.embedding_cache) >= self.multimodal_config.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest_key]
        
        self.embedding_cache[cache_key] = result
    
    def _update_multimodal_stats(self, result: MultimodalEmbeddingResult):
        """Update multimodal processing statistics."""
        self.multimodal_stats['total_multimodal_processed'] += 1
        
        # Update average processing time
        n = self.multimodal_stats['total_multimodal_processed']
        current_avg = self.multimodal_stats['avg_multimodal_time_ms']
        self.multimodal_stats['avg_multimodal_time_ms'] = (
            (current_avg * (n - 1) + result.processing_time_ms) / n
        )
        
        # Update modality counts
        if len(result.modalities_used) == 1:
            if 'text' in result.modalities_used:
                self.multimodal_stats['text_only_count'] += 1
            else:
                self.multimodal_stats['clinical_only_count'] += 1
        elif len(result.modalities_used) == 2:
            self.multimodal_stats['full_multimodal_count'] += 1
    
    def get_multimodal_stats(self) -> Dict[str, Any]:
        """Get comprehensive multimodal processing statistics."""
        stats = self.multimodal_stats.copy()
        
        # Add cache statistics
        if self.embedding_cache:
            total_requests = stats['cache_hits'] + stats['cache_misses']
            stats['cache_hit_rate'] = stats['cache_hits'] / total_requests if total_requests > 0 else 0
            stats['cache_size'] = len(self.embedding_cache)
        
        # Add clinical processor stats
        stats['clinical_processor_stats'] = self.clinical_processor.processing_stats
        
        # Add quantum compression stats
        if self.quantum_compressor:
            stats['compression_ratio'] = self.quantum_compressor.get_compression_ratio()
            stats['compression_params'] = self.quantum_compressor.get_parameter_count()
        
        return stats
    
    def clear_cache(self):
        """Clear embedding cache."""
        if self.embedding_cache:
            self.embedding_cache.clear()
            logger.info("Multimodal embedding cache cleared")
    
    def optimize_memory(self):
        """Optimize memory usage by clearing cache and running garbage collection."""
        if self.embedding_cache:
            self.embedding_cache.clear()
        
        # Run garbage collection
        gc.collect()
        
        logger.info("Memory optimization completed")
    
    def validate_performance(self) -> Dict[str, bool]:
        """Validate performance against PRD constraints."""
        stats = self.get_multimodal_stats()
        
        validation_results = {
            'latency_under_100ms': stats['avg_multimodal_time_ms'] < 100.0,
            'quantum_compression_working': self.quantum_compressor is not None,
            'clinical_processing_working': stats['clinical_processor_stats']['error_count'] == 0,
            'cache_hit_rate_good': stats.get('cache_hit_rate', 0) > 0.1 if self.embedding_cache else True
        }
        
        return validation_results
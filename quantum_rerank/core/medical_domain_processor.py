"""
Medical Domain Processor for QuantumRerank.

This module provides medical domain-specific processing capabilities for multimodal data,
including abbreviation expansion, entity extraction, and clinical context enhancement.

Based on:
- QMMR-01 task requirements
- Medical domain knowledge from existing medical_relevance.py
- Clinical data processing best practices
- Performance constraints from PRD
"""

import re
import logging
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass
import time
import json

from ..evaluation.medical_relevance import MedicalRelevanceJudgments
from ..config.multimodal_config import MedicalDomainConfig

logger = logging.getLogger(__name__)


@dataclass
class MedicalProcessingResult:
    """Result of medical domain processing."""
    processed_text: str
    expanded_abbreviations: Dict[str, str]
    extracted_entities: List[Dict[str, Any]]
    medical_domain: str
    domain_confidence: float
    processing_time_ms: float
    success: bool
    error_message: Optional[str] = None


class MedicalAbbreviationExpander:
    """
    Expands medical abbreviations in text for better semantic understanding.
    
    Uses comprehensive medical abbreviation dictionary and context-aware expansion.
    """
    
    def __init__(self, custom_abbreviations: Optional[Dict[str, str]] = None):
        """
        Initialize abbreviation expander.
        
        Args:
            custom_abbreviations: Additional abbreviations to include
        """
        self.custom_abbreviations = custom_abbreviations or {}
        
        # Comprehensive medical abbreviation dictionary
        self.medical_abbreviations = {
            # Cardiovascular
            'MI': 'myocardial infarction',
            'CAD': 'coronary artery disease',
            'CHF': 'congestive heart failure',
            'HTN': 'hypertension',
            'BP': 'blood pressure',
            'HR': 'heart rate',
            'ECG': 'electrocardiogram',
            'EKG': 'electrocardiogram',
            'A-fib': 'atrial fibrillation',
            'AF': 'atrial fibrillation',
            'VT': 'ventricular tachycardia',
            'VF': 'ventricular fibrillation',
            'PVD': 'peripheral vascular disease',
            'DVT': 'deep vein thrombosis',
            'PE': 'pulmonary embolism',
            
            # Respiratory
            'COPD': 'chronic obstructive pulmonary disease',
            'SOB': 'shortness of breath',
            'DOE': 'dyspnea on exertion',
            'URI': 'upper respiratory infection',
            'PNA': 'pneumonia',
            'TB': 'tuberculosis',
            'RR': 'respiratory rate',
            'O2': 'oxygen',
            'ABG': 'arterial blood gas',
            'CPAP': 'continuous positive airway pressure',
            'BiPAP': 'bilevel positive airway pressure',
            
            # Endocrine/Diabetes
            'DM': 'diabetes mellitus',
            'T1DM': 'type 1 diabetes mellitus',
            'T2DM': 'type 2 diabetes mellitus',
            'BG': 'blood glucose',
            'BS': 'blood sugar',
            'HbA1c': 'hemoglobin A1c',
            'FBS': 'fasting blood sugar',
            'GTT': 'glucose tolerance test',
            'DKA': 'diabetic ketoacidosis',
            'HHNS': 'hyperosmolar hyperglycemic nonketotic syndrome',
            
            # Neurological
            'CVA': 'cerebrovascular accident',
            'TIA': 'transient ischemic attack',
            'ICP': 'intracranial pressure',
            'LOC': 'loss of consciousness',
            'GCS': 'Glasgow Coma Scale',
            'CNS': 'central nervous system',
            'PNS': 'peripheral nervous system',
            'EEG': 'electroencephalogram',
            'LP': 'lumbar puncture',
            'MS': 'multiple sclerosis',
            
            # Gastrointestinal
            'GI': 'gastrointestinal',
            'N/V': 'nausea and vomiting',
            'Abd': 'abdominal',
            'BM': 'bowel movement',
            'NGT': 'nasogastric tube',
            'PEG': 'percutaneous endoscopic gastrostomy',
            'IBD': 'inflammatory bowel disease',
            'IBS': 'irritable bowel syndrome',
            'GERD': 'gastroesophageal reflux disease',
            
            # Renal/Urological
            'UTI': 'urinary tract infection',
            'BUN': 'blood urea nitrogen',
            'Cr': 'creatinine',
            'eGFR': 'estimated glomerular filtration rate',
            'ARF': 'acute renal failure',
            'CRF': 'chronic renal failure',
            'ESRD': 'end-stage renal disease',
            'HD': 'hemodialysis',
            'PD': 'peritoneal dialysis',
            
            # Hematology/Oncology
            'CBC': 'complete blood count',
            'H&H': 'hemoglobin and hematocrit',
            'Hgb': 'hemoglobin',
            'Hct': 'hematocrit',
            'WBC': 'white blood cell count',
            'RBC': 'red blood cell count',
            'PLT': 'platelet count',
            'PT': 'prothrombin time',
            'PTT': 'partial thromboplastin time',
            'INR': 'international normalized ratio',
            'Ca': 'cancer',
            'Mets': 'metastases',
            'Chemo': 'chemotherapy',
            'XRT': 'radiation therapy',
            
            # General medical
            'Pt': 'patient',
            'Px': 'patient',
            'Hx': 'history',
            'PMH': 'past medical history',
            'PSH': 'past surgical history',
            'SH': 'social history',
            'FH': 'family history',
            'ROS': 'review of systems',
            'PE': 'physical examination',
            'VS': 'vital signs',
            'T': 'temperature',
            'Wt': 'weight',
            'Ht': 'height',
            'BMI': 'body mass index',
            'c/o': 'complains of',
            'w/': 'with',
            'w/o': 'without',
            's/p': 'status post',
            'r/o': 'rule out',
            'neg': 'negative',
            'pos': 'positive',
            'WNL': 'within normal limits',
            'NAD': 'no acute distress',
            'NKDA': 'no known drug allergies',
            'NKA': 'no known allergies',
            
            # Procedures
            'EGD': 'esophagogastroduodenoscopy',
            'Cath': 'catheterization',
            'PCI': 'percutaneous coronary intervention',
            'CABG': 'coronary artery bypass graft',
            'TURP': 'transurethral resection of the prostate',
            'C&S': 'culture and sensitivity',
            'I&D': 'incision and drainage',
            'D&C': 'dilation and curettage',
            'TAH': 'total abdominal hysterectomy',
            'BSO': 'bilateral salpingo-oophorectomy'
        }
        
        # Merge with custom abbreviations
        self.all_abbreviations = {**self.medical_abbreviations, **self.custom_abbreviations}
        
        # Compile regex patterns for efficient matching
        self.abbreviation_patterns = self._compile_abbreviation_patterns()
        
        logger.info(f"MedicalAbbreviationExpander initialized with {len(self.all_abbreviations)} abbreviations")
    
    def _compile_abbreviation_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for abbreviation matching."""
        patterns = {}
        
        for abbrev in self.all_abbreviations.keys():
            # Create pattern that matches whole word boundaries
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            patterns[abbrev] = re.compile(pattern, re.IGNORECASE)
        
        return patterns
    
    def expand(self, text: str) -> str:
        """
        Expand medical abbreviations in text.
        
        Args:
            text: Input text with abbreviations
            
        Returns:
            Text with expanded abbreviations
        """
        if not text:
            return text
        
        expanded_text = text
        
        # Apply abbreviation expansion
        for abbrev, expansion in self.all_abbreviations.items():
            if abbrev in self.abbreviation_patterns:
                pattern = self.abbreviation_patterns[abbrev]
                replacement = f"{abbrev} ({expansion})"
                expanded_text = pattern.sub(replacement, expanded_text)
        
        return expanded_text
    
    def get_expanded_abbreviations(self, text: str) -> Dict[str, str]:
        """
        Get dictionary of abbreviations found and expanded in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping abbreviations to their expansions
        """
        found_abbreviations = {}
        
        for abbrev, expansion in self.all_abbreviations.items():
            if abbrev in self.abbreviation_patterns:
                pattern = self.abbreviation_patterns[abbrev]
                if pattern.search(text):
                    found_abbreviations[abbrev] = expansion
        
        return found_abbreviations


class ClinicalEntityExtractor:
    """
    Extracts clinical entities from medical text using rule-based and pattern-based approaches.
    """
    
    def __init__(self, entity_types: Optional[List[str]] = None):
        """
        Initialize clinical entity extractor.
        
        Args:
            entity_types: Types of entities to extract
        """
        self.entity_types = entity_types or [
            'DISEASE', 'SYMPTOM', 'MEDICATION', 'PROCEDURE', 'ANATOMY', 'LAB_VALUE'
        ]
        
        # Entity patterns for different types
        self.entity_patterns = {
            'DISEASE': [
                r'\b(?:diabetes|hypertension|pneumonia|cancer|stroke|heart failure|COPD|asthma)\b',
                r'\b\w+itis\b',  # Inflammatory conditions
                r'\b\w+osis\b',  # Pathological conditions
                r'\b\w+emia\b',  # Blood conditions
                r'\b\w+pathy\b'  # Disease conditions
            ],
            'SYMPTOM': [
                r'\b(?:pain|fever|nausea|vomiting|shortness of breath|fatigue|weakness)\b',
                r'\b(?:headache|dizziness|chest pain|abdominal pain|back pain)\b',
                r'\b(?:cough|wheezing|dyspnea|palpitations|syncope)\b'
            ],
            'MEDICATION': [
                r'\b\w+cillin\b',  # Antibiotics
                r'\b\w+pril\b',   # ACE inhibitors
                r'\b\w+sartan\b', # ARBs
                r'\b\w+statin\b', # Statins
                r'\b(?:aspirin|warfarin|heparin|insulin|metformin|prednisone)\b'
            ],
            'PROCEDURE': [
                r'\b(?:surgery|biopsy|catheterization|endoscopy|angiography)\b',
                r'\b\w+ectomy\b',  # Surgical removals
                r'\b\w+plasty\b',  # Surgical repairs
                r'\b\w+scopy\b'    # Viewing procedures
            ],
            'ANATOMY': [
                r'\b(?:heart|lung|liver|kidney|brain|stomach|intestine|pancreas)\b',
                r'\b(?:artery|vein|valve|muscle|bone|joint|nerve)\b',
                r'\b(?:left|right|anterior|posterior|superior|inferior)\s+\w+\b'
            ],
            'LAB_VALUE': [
                r'\b(?:glucose|cholesterol|hemoglobin|creatinine|sodium|potassium)\b',
                r'\b\d+\s*(?:mg/dL|mmol/L|g/dL|mEq/L|%)\b',
                r'\b(?:WBC|RBC|PLT|Hgb|Hct|BUN|Cr)\s*:?\s*\d+\b'
            ]
        }
        
        # Compile patterns
        self.compiled_patterns = {}
        for entity_type, patterns in self.entity_patterns.items():
            self.compiled_patterns[entity_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
        
        logger.info(f"ClinicalEntityExtractor initialized for {len(self.entity_types)} entity types")
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract clinical entities from text.
        
        Args:
            text: Input medical text
            
        Returns:
            List of extracted entities with type, text, and position
        """
        if not text:
            return []
        
        entities = []
        
        for entity_type in self.entity_types:
            if entity_type in self.compiled_patterns:
                for pattern in self.compiled_patterns[entity_type]:
                    for match in pattern.finditer(text):
                        entity = {
                            'text': match.group(),
                            'type': entity_type,
                            'start': match.start(),
                            'end': match.end(),
                            'confidence': 0.8  # Rule-based confidence
                        }
                        entities.append(entity)
        
        # Remove duplicates and sort by position
        entities = self._remove_duplicate_entities(entities)
        entities.sort(key=lambda x: x['start'])
        
        return entities
    
    def _remove_duplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities based on text and position overlap."""
        if not entities:
            return entities
        
        unique_entities = []
        
        for entity in entities:
            is_duplicate = False
            
            for existing in unique_entities:
                # Check for text overlap
                if (entity['start'] < existing['end'] and 
                    entity['end'] > existing['start']):
                    # Keep the one with higher confidence
                    if entity['confidence'] > existing['confidence']:
                        unique_entities.remove(existing)
                        break
                    else:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_entities.append(entity)
        
        return unique_entities


class MedicalDomainProcessor:
    """
    Comprehensive medical domain processor that enhances multimodal medical queries
    with domain-specific knowledge and processing.
    """
    
    def __init__(self, config: MedicalDomainConfig = None):
        """
        Initialize medical domain processor.
        
        Args:
            config: Medical domain configuration
        """
        self.config = config or MedicalDomainConfig()
        
        # Initialize components
        self.abbreviation_expander = MedicalAbbreviationExpander(
            self.config.custom_abbreviations
        )
        
        self.entity_extractor = ClinicalEntityExtractor()
        
        # Initialize medical relevance system for domain classification
        self.medical_relevance = MedicalRelevanceJudgments()
        
        # Performance monitoring
        self.processing_stats = {
            'total_processed': 0,
            'avg_processing_time_ms': 0.0,
            'success_count': 0,
            'error_count': 0
        }
        
        logger.info("MedicalDomainProcessor initialized")
    
    def process_medical_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process multimodal medical query with domain-specific enhancements.
        
        Args:
            query: Query containing text and/or clinical_data
            
        Returns:
            Enhanced query with medical domain processing
        """
        start_time = time.time()
        
        try:
            processed_query = query.copy()
            
            # Process text component
            if 'text' in query and query['text']:
                text_result = self._process_text_component(query['text'])
                processed_query.update(text_result)
            
            # Process clinical data component
            if 'clinical_data' in query and query['clinical_data']:
                clinical_result = self._process_clinical_component(query['clinical_data'])
                processed_query.update(clinical_result)
            
            # Add processing metadata
            elapsed = (time.time() - start_time) * 1000
            processed_query['processing_metadata'] = {
                'processing_time_ms': elapsed,
                'success': True,
                'processor': 'MedicalDomainProcessor'
            }
            
            # Update statistics
            self._update_processing_stats(elapsed, success=True)
            
            return processed_query
            
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            logger.error(f"Medical query processing failed: {e}")
            
            # Update statistics
            self._update_processing_stats(elapsed, success=False)
            
            # Return original query with error metadata
            error_query = query.copy()
            error_query['processing_metadata'] = {
                'processing_time_ms': elapsed,
                'success': False,
                'error': str(e),
                'processor': 'MedicalDomainProcessor'
            }
            
            return error_query
    
    def _process_text_component(self, text: str) -> Dict[str, Any]:
        """Process text component of medical query."""
        result = {}
        
        # Expand abbreviations
        if self.config.abbreviation_expansion:
            expanded_text = self.abbreviation_expander.expand(text)
            expanded_abbreviations = self.abbreviation_expander.get_expanded_abbreviations(text)
            
            result['text'] = expanded_text
            result['expanded_abbreviations'] = expanded_abbreviations
        else:
            result['text'] = text
            result['expanded_abbreviations'] = {}
        
        # Extract entities
        if self.config.entity_extraction:
            entities = self.entity_extractor.extract(text)
            result['extracted_entities'] = entities
        else:
            result['extracted_entities'] = []
        
        # Classify medical domain
        if self.config.domain_classification:
            domain, confidence = self.medical_relevance.classify_medical_domain(text)
            result['medical_domain'] = domain
            result['domain_confidence'] = confidence
        else:
            result['medical_domain'] = 'general'
            result['domain_confidence'] = 0.5
        
        return result
    
    def _process_clinical_component(self, clinical_data: Any) -> Dict[str, Any]:
        """Process clinical data component of medical query."""
        result = {}
        
        # Convert clinical data to text representation
        if isinstance(clinical_data, dict):
            clinical_text = self._clinical_dict_to_text(clinical_data)
        else:
            clinical_text = str(clinical_data)
        
        # Extract clinical entities
        if self.config.entity_extraction:
            clinical_entities = self.entity_extractor.extract(clinical_text)
            result['clinical_entities'] = clinical_entities
        else:
            result['clinical_entities'] = []
        
        # Classify clinical domain
        if self.config.domain_classification:
            clinical_domain, confidence = self.medical_relevance.classify_medical_domain(clinical_text)
            result['clinical_domain'] = clinical_domain
            result['clinical_domain_confidence'] = confidence
        else:
            result['clinical_domain'] = 'general'
            result['clinical_domain_confidence'] = 0.5
        
        # Store processed clinical text
        result['clinical_text'] = clinical_text
        
        return result
    
    def _clinical_dict_to_text(self, clinical_dict: Dict) -> str:
        """Convert clinical data dictionary to text representation."""
        text_parts = []
        
        # Process different clinical data types
        supported_types = [
            'demographics', 'vitals', 'lab_results', 'medications', 
            'procedures', 'diagnoses', 'symptoms', 'allergies',
            'chief_complaint', 'history', 'physical_exam'
        ]
        
        for data_type in supported_types:
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
    
    def _update_processing_stats(self, elapsed_ms: float, success: bool):
        """Update processing statistics."""
        self.processing_stats['total_processed'] += 1
        
        if success:
            self.processing_stats['success_count'] += 1
            
            # Update average processing time
            n = self.processing_stats['success_count']
            current_avg = self.processing_stats['avg_processing_time_ms']
            self.processing_stats['avg_processing_time_ms'] = (
                (current_avg * (n - 1) + elapsed_ms) / n
            )
        else:
            self.processing_stats['error_count'] += 1
    
    def process_medical_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process medical document with domain-specific enhancements.
        
        Args:
            document: Document containing text and metadata
            
        Returns:
            Enhanced document with medical processing
        """
        # Similar processing as query but adapted for documents
        return self.process_medical_query(document)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        stats = self.processing_stats.copy()
        
        # Calculate success rate
        total = stats['total_processed']
        if total > 0:
            stats['success_rate'] = stats['success_count'] / total
            stats['error_rate'] = stats['error_count'] / total
        else:
            stats['success_rate'] = 0.0
            stats['error_rate'] = 0.0
        
        # Add component stats
        stats['abbreviation_count'] = len(self.abbreviation_expander.all_abbreviations)
        stats['entity_types'] = len(self.entity_extractor.entity_types)
        
        return stats
    
    def validate_medical_processing(self, test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate medical processing on test queries.
        
        Args:
            test_queries: List of test queries
            
        Returns:
            Validation results
        """
        validation_results = {
            'total_queries': len(test_queries),
            'successful_processing': 0,
            'failed_processing': 0,
            'avg_processing_time_ms': 0.0,
            'domain_classification_coverage': 0.0
        }
        
        processing_times = []
        domain_classifications = []
        
        for query in test_queries:
            processed = self.process_medical_query(query)
            
            if processed.get('processing_metadata', {}).get('success', False):
                validation_results['successful_processing'] += 1
                processing_times.append(processed['processing_metadata']['processing_time_ms'])
                
                if 'medical_domain' in processed:
                    domain_classifications.append(processed['medical_domain'])
        
        validation_results['failed_processing'] = (
            validation_results['total_queries'] - validation_results['successful_processing']
        )
        
        if processing_times:
            validation_results['avg_processing_time_ms'] = sum(processing_times) / len(processing_times)
        
        if domain_classifications:
            # Calculate coverage of domain classifications
            unique_domains = set(domain_classifications)
            total_supported = len(self.config.supported_domains)
            validation_results['domain_classification_coverage'] = len(unique_domains) / total_supported
        
        return validation_results
    
    def optimize_for_performance(self):
        """Optimize processor for performance."""
        # Pre-compile frequently used patterns
        self.abbreviation_expander._compile_abbreviation_patterns()
        
        logger.info("MedicalDomainProcessor optimized for performance")
    
    def get_supported_domains(self) -> List[str]:
        """Get list of supported medical domains."""
        return self.config.supported_domains
    
    def get_abbreviation_count(self) -> int:
        """Get number of supported abbreviations."""
        return len(self.abbreviation_expander.all_abbreviations)
    
    def get_entity_types(self) -> List[str]:
        """Get list of supported entity types."""
        return self.entity_extractor.entity_types
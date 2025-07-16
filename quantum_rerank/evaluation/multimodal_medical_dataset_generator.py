"""
Multimodal Medical Dataset Generator for QMMR-05 evaluation.

Generates comprehensive synthetic medical datasets with text, clinical data, and image
modalities for evaluating quantum multimodal medical reranker performance.
"""

import logging
import random
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from PIL import Image
import numpy as np
import json

from quantum_rerank.config.evaluation_config import (
    MultimodalMedicalEvaluationConfig, DatasetGenerationConfig
)

logger = logging.getLogger(__name__)


@dataclass
class MultimodalMedicalQuery:
    """Represents a multimodal medical query with various data modalities."""
    
    id: str
    query_type: str  # diagnostic_inquiry, treatment_recommendation, etc.
    complexity_level: str  # simple, moderate, complex, very_complex
    specialty: str  # radiology, cardiology, etc.
    
    # Modalities
    text: Optional[str] = None
    clinical_data: Optional[Dict[str, Any]] = None
    image: Optional[Image.Image] = None
    image_metadata: Optional[Dict[str, Any]] = None
    
    # Ground truth and metadata
    ground_truth_diagnosis: Optional[str] = None
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    medical_urgency: str = "routine"  # routine, urgent, emergency
    expected_difficulty: float = 0.5  # 0-1 scale
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary representation."""
        return {
            'id': self.id,
            'query_type': self.query_type,
            'complexity_level': self.complexity_level,
            'specialty': self.specialty,
            'text': self.text,
            'clinical_data': self.clinical_data,
            'image_metadata': self.image_metadata,
            'ground_truth_diagnosis': self.ground_truth_diagnosis,
            'relevance_scores': self.relevance_scores,
            'medical_urgency': self.medical_urgency,
            'expected_difficulty': self.expected_difficulty
        }


@dataclass
class MultimodalMedicalCandidate:
    """Represents a candidate document/response for medical queries."""
    
    id: str
    content_type: str  # guideline, case_study, research_paper, protocol
    specialty: str
    
    # Content modalities
    text: Optional[str] = None
    clinical_data: Optional[Dict[str, Any]] = None
    image: Optional[Image.Image] = None
    image_metadata: Optional[Dict[str, Any]] = None
    
    # Relevance metadata
    diagnosis: Optional[str] = None
    treatment_recommendations: List[str] = field(default_factory=list)
    evidence_level: str = "moderate"  # low, moderate, high
    clinical_applicability: float = 0.5  # 0-1 scale
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert candidate to dictionary representation."""
        return {
            'id': self.id,
            'content_type': self.content_type,
            'specialty': self.specialty,
            'text': self.text,
            'clinical_data': self.clinical_data,
            'image_metadata': self.image_metadata,
            'diagnosis': self.diagnosis,
            'treatment_recommendations': self.treatment_recommendations,
            'evidence_level': self.evidence_level,
            'clinical_applicability': self.clinical_applicability
        }


@dataclass
class MultimodalMedicalDataset:
    """Complete dataset with queries, candidates, and relevance judgments."""
    
    queries: List[MultimodalMedicalQuery] = field(default_factory=list)
    candidates: Dict[str, List[MultimodalMedicalCandidate]] = field(default_factory=dict)
    relevance_judgments: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_query(self, query: MultimodalMedicalQuery):
        """Add a query to the dataset."""
        self.queries.append(query)
        self.candidates[query.id] = []
        self.relevance_judgments[query.id] = {}
    
    def add_candidate(self, query_id: str, candidate: MultimodalMedicalCandidate):
        """Add a candidate for a specific query."""
        if query_id not in self.candidates:
            self.candidates[query_id] = []
        self.candidates[query_id].append(candidate)
    
    def add_relevance_judgment(self, query_id: str, candidate_id: str, relevance: float):
        """Add a relevance judgment for a query-candidate pair."""
        if query_id not in self.relevance_judgments:
            self.relevance_judgments[query_id] = {}
        self.relevance_judgments[query_id][candidate_id] = relevance
    
    def get_candidates(self, query_id: str) -> List[MultimodalMedicalCandidate]:
        """Get candidates for a specific query."""
        return self.candidates.get(query_id, [])
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information and statistics."""
        modality_stats = {
            'text_only': 0,
            'image_only': 0,
            'clinical_only': 0,
            'text_image': 0,
            'text_clinical': 0,
            'image_clinical': 0,
            'all_modalities': 0
        }
        
        specialty_stats = {}
        complexity_stats = {}
        
        for query in self.queries:
            # Count modalities
            has_text = query.text is not None
            has_image = query.image is not None
            has_clinical = query.clinical_data is not None
            
            if has_text and has_image and has_clinical:
                modality_stats['all_modalities'] += 1
            elif has_text and has_image:
                modality_stats['text_image'] += 1
            elif has_text and has_clinical:
                modality_stats['text_clinical'] += 1
            elif has_image and has_clinical:
                modality_stats['image_clinical'] += 1
            elif has_text:
                modality_stats['text_only'] += 1
            elif has_image:
                modality_stats['image_only'] += 1
            elif has_clinical:
                modality_stats['clinical_only'] += 1
            
            # Count specialties
            specialty_stats[query.specialty] = specialty_stats.get(query.specialty, 0) + 1
            
            # Count complexity levels
            complexity_stats[query.complexity_level] = complexity_stats.get(query.complexity_level, 0) + 1
        
        return {
            'total_queries': len(self.queries),
            'total_candidates': sum(len(candidates) for candidates in self.candidates.values()),
            'total_relevance_judgments': sum(len(judgments) for judgments in self.relevance_judgments.values()),
            'modality_distribution': modality_stats,
            'specialty_distribution': specialty_stats,
            'complexity_distribution': complexity_stats,
            'metadata': self.metadata
        }


class MedicalTextGenerator:
    """Generates synthetic medical text for various scenarios."""
    
    def __init__(self):
        # Medical terminology and templates
        self.symptoms = [
            'chest pain', 'shortness of breath', 'fatigue', 'dizziness', 'headache',
            'nausea', 'vomiting', 'fever', 'cough', 'abdominal pain', 'back pain',
            'joint pain', 'muscle weakness', 'confusion', 'palpitations'
        ]
        
        self.conditions = [
            'pneumonia', 'myocardial infarction', 'stroke', 'pulmonary embolism',
            'heart failure', 'COPD', 'diabetes', 'hypertension', 'sepsis',
            'acute appendicitis', 'kidney stones', 'gallbladder disease'
        ]
        
        self.imaging_findings = {
            'XR': ['consolidation', 'pleural effusion', 'cardiomegaly', 'pneumothorax'],
            'CT': ['hypodensity', 'enhancement', 'mass effect', 'hemorrhage'],
            'MR': ['hyperintensity', 'hypointensity', 'diffusion restriction', 'enhancement'],
            'US': ['echogenicity', 'fluid collection', 'vascularity', 'motion']
        }
        
        self.query_templates = {
            'diagnostic_inquiry': [
                "Patient presents with {symptoms}. What is the most likely diagnosis?",
                "{age}-year-old {gender} with {symptoms} for {duration}. Differential diagnosis?",
                "Chief complaint: {symptoms}. Clinical presentation suggests?"
            ],
            'treatment_recommendation': [
                "Best treatment approach for {condition} in {age}-year-old patient?",
                "Management guidelines for {condition} with {comorbidities}?",
                "Evidence-based treatment for {condition} - what does literature show?"
            ],
            'imaging_interpretation': [
                "{modality} shows {findings}. Clinical significance?",
                "How to interpret {findings} on {modality} in context of {symptoms}?",
                "{modality} findings: {findings}. Next steps?"
            ]
        }
    
    def generate_query_text(self, query_type: str, specialty: str, complexity_level: str) -> str:
        """Generate medical query text based on type and complexity."""
        templates = self.query_templates.get(query_type, ["Generic medical query about {condition}"])
        template = random.choice(templates)
        
        # Fill template with medical terms
        params = {
            'symptoms': ', '.join(random.sample(self.symptoms, random.randint(1, 3))),
            'condition': random.choice(self.conditions),
            'age': random.randint(18, 90),
            'gender': random.choice(['male', 'female']),
            'duration': random.choice(['2 days', '1 week', '3 hours', '1 month']),
            'modality': random.choice(['chest X-ray', 'CT scan', 'MRI', 'ultrasound']),
            'findings': random.choice(self.imaging_findings.get('XR', ['abnormal findings'])),
            'comorbidities': random.choice(['diabetes', 'hypertension', 'none'])
        }
        
        # Adjust complexity
        if complexity_level == 'very_complex':
            # Add more medical details
            additional_context = f" Patient has history of {random.choice(self.conditions)}. Current medications include {random.choice(['beta-blockers', 'ACE inhibitors', 'statins'])}."
            return template.format(**params) + additional_context
        elif complexity_level == 'simple':
            # Use simpler language
            simple_params = {k: v for k, v in params.items() if k in ['symptoms', 'condition', 'age']}
            return template.format(**simple_params)
        
        return template.format(**params)
    
    def generate_candidate_text(self, content_type: str, specialty: str, diagnosis: Optional[str] = None) -> str:
        """Generate candidate document text."""
        if content_type == 'guideline':
            return f"Clinical practice guideline for {diagnosis or random.choice(self.conditions)}. Recommended approach includes assessment of {random.choice(self.symptoms)} and consideration of {random.choice(['medication', 'surgical intervention', 'lifestyle modification'])}."
        elif content_type == 'case_study':
            return f"Case report: {random.randint(20, 80)}-year-old patient presented with {random.choice(self.symptoms)}. Workup revealed {diagnosis or random.choice(self.conditions)}. Treatment outcome was {random.choice(['favorable', 'complicated', 'pending'])}."
        elif content_type == 'research_paper':
            return f"Research study on {diagnosis or random.choice(self.conditions)} involving {random.randint(50, 500)} patients. Primary endpoint showed {random.choice(['significant improvement', 'no significant difference', 'mixed results'])} with p-value < 0.05."
        else:
            return f"Medical content about {diagnosis or random.choice(self.conditions)} in {specialty} practice."


class ClinicalDataGenerator:
    """Generates synthetic clinical data structures."""
    
    def __init__(self):
        self.vital_sign_ranges = {
            'temperature': (96.0, 104.0),
            'heart_rate': (50, 120),
            'blood_pressure_systolic': (90, 180),
            'blood_pressure_diastolic': (60, 110),
            'respiratory_rate': (12, 30),
            'oxygen_saturation': (85, 100)
        }
        
        self.lab_values = {
            'white_blood_count': (4000, 12000),
            'hemoglobin': (10.0, 18.0),
            'platelets': (150000, 400000),
            'glucose': (70, 200),
            'creatinine': (0.5, 2.0),
            'bun': (7, 25)
        }
    
    def generate_clinical_data(self, complexity_level: str, medical_urgency: str) -> Dict[str, Any]:
        """Generate clinical data based on complexity and urgency."""
        data = {
            'vital_signs': {},
            'lab_values': {},
            'medications': [],
            'allergies': [],
            'medical_history': []
        }
        
        # Generate vital signs
        for vital, (min_val, max_val) in self.vital_sign_ranges.items():
            if medical_urgency == 'emergency':
                # Generate more extreme values for emergency cases
                if random.random() < 0.3:  # 30% chance of abnormal
                    if random.random() < 0.5:
                        value = random.uniform(min_val, min_val + (max_val - min_val) * 0.2)
                    else:
                        value = random.uniform(min_val + (max_val - min_val) * 0.8, max_val)
                else:
                    value = random.uniform(min_val + (max_val - min_val) * 0.3, max_val - (max_val - min_val) * 0.3)
            else:
                value = random.uniform(min_val, max_val)
            
            data['vital_signs'][vital] = round(value, 1)
        
        # Generate lab values based on complexity
        num_labs = 3 if complexity_level == 'simple' else 6 if complexity_level == 'moderate' else 10
        lab_names = random.sample(list(self.lab_values.keys()), min(num_labs, len(self.lab_values)))
        
        for lab in lab_names:
            min_val, max_val = self.lab_values[lab]
            data['lab_values'][lab] = round(random.uniform(min_val, max_val), 2)
        
        # Add medications and history based on complexity
        if complexity_level in ['complex', 'very_complex']:
            data['medications'] = random.sample([
                'lisinopril', 'metformin', 'atorvastatin', 'amlodipine', 'omeprazole'
            ], random.randint(1, 3))
            
            data['medical_history'] = random.sample([
                'hypertension', 'diabetes', 'hyperlipidemia', 'COPD', 'heart disease'
            ], random.randint(0, 2))
        
        return data


class MedicalImageGenerator:
    """Generates synthetic medical images for testing."""
    
    def __init__(self):
        self.image_size = (224, 224)
        self.modality_characteristics = {
            'XR': {'base_intensity': (50, 200), 'contrast': 'high'},
            'CT': {'base_intensity': (80, 160), 'contrast': 'medium'},
            'MR': {'base_intensity': (20, 100), 'contrast': 'high'},
            'US': {'base_intensity': (30, 150), 'contrast': 'low'},
            'MG': {'base_intensity': (40, 180), 'contrast': 'high'}
        }
    
    def generate_medical_image(self, modality: str, body_part: str, pathology: Optional[str] = None) -> Image.Image:
        """Generate synthetic medical image based on modality and anatomy."""
        width, height = self.image_size
        characteristics = self.modality_characteristics.get(modality, {'base_intensity': (50, 200), 'contrast': 'medium'})
        
        min_intensity, max_intensity = characteristics['base_intensity']
        
        # Create base image
        image_array = np.random.randint(min_intensity, max_intensity, (height, width, 3), dtype=np.uint8)
        
        # Add modality-specific patterns
        if modality == 'XR':
            image_array = self._add_xray_patterns(image_array, body_part, pathology)
        elif modality == 'CT':
            image_array = self._add_ct_patterns(image_array, body_part, pathology)
        elif modality == 'MR':
            image_array = self._add_mri_patterns(image_array, body_part, pathology)
        elif modality == 'US':
            image_array = self._add_ultrasound_patterns(image_array, body_part, pathology)
        
        return Image.fromarray(image_array)
    
    def _add_xray_patterns(self, image_array: np.ndarray, body_part: str, pathology: Optional[str]) -> np.ndarray:
        """Add X-ray specific patterns."""
        height, width = image_array.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        if body_part == 'chest':
            # Add lung fields
            for i in range(height):
                for j in range(width):
                    left_lung_dist = np.sqrt((i - center_y)**2 + (j - center_x + 40)**2)
                    right_lung_dist = np.sqrt((i - center_y)**2 + (j - center_x - 40)**2)
                    
                    if left_lung_dist < 60 or right_lung_dist < 60:
                        image_array[i, j] = [30, 30, 30]  # Dark lung fields
                    
                    # Add spine
                    if abs(j - center_x) < 10 and abs(i - center_y) < 80:
                        image_array[i, j] = [180, 180, 180]  # Bright spine
            
            # Add pathology if specified
            if pathology == 'pneumonia':
                # Add consolidation pattern
                patch_x, patch_y = random.randint(50, width-50), random.randint(50, height-50)
                for i in range(max(0, patch_y-20), min(height, patch_y+20)):
                    for j in range(max(0, patch_x-20), min(width, patch_x+20)):
                        if np.sqrt((i-patch_y)**2 + (j-patch_x)**2) < 15:
                            image_array[i, j] = [120, 120, 120]  # Consolidation
        
        return image_array
    
    def _add_ct_patterns(self, image_array: np.ndarray, body_part: str, pathology: Optional[str]) -> np.ndarray:
        """Add CT scan specific patterns."""
        height, width = image_array.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Add circular body outline
        for i in range(height):
            for j in range(width):
                dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                
                if 80 < dist < 100:
                    image_array[i, j] = [60, 60, 60]  # Skin/muscle
                elif dist < 80:
                    # Internal structures with variations
                    organ_noise = np.random.randint(-20, 20)
                    image_array[i, j] = np.clip([120 + organ_noise] * 3, 0, 255)
        
        return image_array
    
    def _add_mri_patterns(self, image_array: np.ndarray, body_part: str, pathology: Optional[str]) -> np.ndarray:
        """Add MRI specific patterns."""
        height, width = image_array.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        if body_part == 'brain':
            # Add brain outline and structures
            for i in range(height):
                for j in range(width):
                    dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                    
                    if dist < 90:
                        if dist < 30:
                            image_array[i, j] = [150, 150, 150]  # White matter
                        elif dist < 70:
                            image_array[i, j] = [100, 100, 100]  # Gray matter
                        else:
                            image_array[i, j] = [50, 50, 50]     # CSF
        
        return image_array
    
    def _add_ultrasound_patterns(self, image_array: np.ndarray, body_part: str, pathology: Optional[str]) -> np.ndarray:
        """Add ultrasound specific patterns."""
        # Add characteristic ultrasound noise and shadows
        height, width = image_array.shape[:2]
        
        # Add acoustic shadows and enhancement
        for i in range(0, height, 10):
            for j in range(0, width, 15):
                if random.random() < 0.3:  # Random shadows
                    for di in range(10):
                        for dj in range(15):
                            if i+di < height and j+dj < width:
                                image_array[i+di, j+dj] = [20, 20, 20]  # Shadow
        
        return image_array


class MultimodalMedicalDatasetGenerator:
    """
    Main dataset generator for comprehensive multimodal medical evaluation.
    
    Generates synthetic medical datasets with various modalities, complexity levels,
    and medical scenarios for evaluating quantum multimodal medical reranker.
    """
    
    def __init__(self, config: MultimodalMedicalEvaluationConfig, dataset_config: Optional[DatasetGenerationConfig] = None):
        self.config = config
        self.dataset_config = dataset_config or DatasetGenerationConfig()
        
        # Initialize generators
        self.text_generator = MedicalTextGenerator()
        self.clinical_generator = ClinicalDataGenerator()
        self.image_generator = MedicalImageGenerator()
        
        # Complexity assessment (placeholder for actual complexity engine)
        self.complexity_assessor = self._create_complexity_assessor()
        
        # Ground truth annotator
        self.relevance_annotator = self._create_relevance_annotator()
        
        logger.info("Initialized MultimodalMedicalDatasetGenerator")
    
    def _create_complexity_assessor(self):
        """Create complexity assessment component."""
        # Placeholder - would integrate with actual complexity assessment engine
        class MockComplexityAssessor:
            def assess_complexity(self, query_dict: Dict, candidates: List) -> float:
                # Simple heuristic based on content length and medical terms
                text_length = len(query_dict.get('text', ''))
                has_clinical = bool(query_dict.get('clinical_data'))
                has_image = bool(query_dict.get('image_metadata'))
                
                complexity = 0.1  # Base complexity
                complexity += min(text_length / 1000, 0.3)  # Text complexity
                complexity += 0.2 if has_clinical else 0
                complexity += 0.3 if has_image else 0
                complexity += len(candidates) / 100 * 0.1  # Candidate pool complexity
                
                return min(complexity, 1.0)
        
        return MockComplexityAssessor()
    
    def _create_relevance_annotator(self):
        """Create relevance annotation component."""
        class MockRelevanceAnnotator:
            def annotate_relevance(self, query: MultimodalMedicalQuery, candidates: List[MultimodalMedicalCandidate]) -> Dict[str, float]:
                relevance_scores = {}
                
                for candidate in candidates:
                    # Simple relevance scoring based on text similarity and medical matching
                    score = 0.5  # Base relevance
                    
                    # Boost for specialty match
                    if candidate.specialty == query.specialty:
                        score += 0.2
                    
                    # Boost for diagnosis match
                    if (candidate.diagnosis and query.ground_truth_diagnosis and 
                        candidate.diagnosis.lower() in query.ground_truth_diagnosis.lower()):
                        score += 0.3
                    
                    # Add some random variation
                    score += random.uniform(-0.1, 0.1)
                    
                    relevance_scores[candidate.id] = max(0.0, min(1.0, score))
                
                return relevance_scores
        
        return MockRelevanceAnnotator()
    
    def generate_comprehensive_dataset(self) -> MultimodalMedicalDataset:
        """
        Generate comprehensive dataset covering various medical scenarios.
        """
        logger.info("Starting comprehensive dataset generation...")
        start_time = time.time()
        
        dataset = MultimodalMedicalDataset()
        dataset.metadata = {
            'generation_time': time.time(),
            'config': self.config.__dict__,
            'dataset_config': self.dataset_config.__dict__
        }
        
        # Generate queries by type
        queries_per_type = self.config.min_multimodal_queries // len(self.config.medical_scenarios)
        
        for scenario_type in self.config.medical_scenarios:
            logger.info(f"Generating queries for scenario: {scenario_type}")
            queries = self._generate_queries_by_type(scenario_type, queries_per_type)
            
            for query in queries:
                dataset.add_query(query)
                
                # Generate candidates for each query
                candidates = self._generate_candidates_for_query(query)
                for candidate in candidates:
                    dataset.add_candidate(query.id, candidate)
                
                # Generate relevance judgments
                relevance_scores = self.relevance_annotator.annotate_relevance(query, candidates)
                for candidate_id, relevance in relevance_scores.items():
                    dataset.add_relevance_judgment(query.id, candidate_id, relevance)
        
        # Add noise variations for robustness testing
        logger.info("Adding noise variations for robustness testing...")
        noisy_dataset = self._add_noise_variations(dataset)
        
        generation_time = time.time() - start_time
        logger.info(f"Dataset generation completed in {generation_time:.2f} seconds")
        
        noisy_dataset.metadata['generation_duration_seconds'] = generation_time
        noisy_dataset.metadata['final_stats'] = noisy_dataset.get_info()
        
        return noisy_dataset
    
    def _generate_queries_by_type(self, scenario_type: str, num_queries: int) -> List[MultimodalMedicalQuery]:
        """Generate queries for specific medical scenario type."""
        queries = []
        
        for i in range(num_queries):
            # Select specialty and complexity
            specialty = np.random.choice(
                list(self.dataset_config.specialty_distribution.keys()),
                p=list(self.dataset_config.specialty_distribution.values())
            )
            
            complexity_level = random.choice(self.config.complexity_levels)
            
            # Create base query
            query_id = f"{scenario_type}_{specialty}_{i:03d}"
            query = MultimodalMedicalQuery(
                id=query_id,
                query_type=scenario_type,
                complexity_level=complexity_level,
                specialty=specialty
            )
            
            # Add modalities based on scenario type and configuration
            query = self._add_modalities_to_query(query, scenario_type)
            
            # Add ground truth and metadata
            query.ground_truth_diagnosis = random.choice(self.text_generator.conditions)
            query.medical_urgency = random.choice(['routine', 'urgent', 'emergency'])
            query.expected_difficulty = self.complexity_assessor.assess_complexity(query.to_dict(), [])
            
            queries.append(query)
        
        return queries
    
    def _add_modalities_to_query(self, query: MultimodalMedicalQuery, scenario_type: str) -> MultimodalMedicalQuery:
        """Add appropriate modalities to query based on scenario type."""
        
        # Determine modality combination
        modality_probs = [
            self.dataset_config.text_only_ratio,
            self.dataset_config.image_only_ratio,
            self.dataset_config.text_image_ratio,
            self.dataset_config.text_clinical_ratio,
            self.dataset_config.all_modalities_ratio
        ]
        
        modality_choice = np.random.choice(5, p=modality_probs)
        
        # Add text modality
        if modality_choice in [0, 2, 3, 4]:  # text_only, text_image, text_clinical, all_modalities
            query.text = self.text_generator.generate_query_text(
                scenario_type, query.specialty, query.complexity_level
            )
        
        # Add image modality
        if modality_choice in [1, 2, 4] or scenario_type == 'imaging_interpretation':  # image_only, text_image, all_modalities
            # Select image modality
            image_modality = np.random.choice(
                list(self.dataset_config.image_modality_distribution.keys()),
                p=list(self.dataset_config.image_modality_distribution.values())
            )
            
            # Generate image
            body_parts = ['chest', 'abdomen', 'brain', 'pelvis', 'extremity']
            body_part = random.choice(body_parts)
            
            pathology = random.choice([None, 'pneumonia', 'mass', 'fracture'])
            
            query.image = self.image_generator.generate_medical_image(
                image_modality, body_part, pathology
            )
            
            query.image_metadata = {
                'modality': image_modality,
                'body_part': body_part,
                'pathology': pathology,
                'view': random.choice(['AP', 'lateral', 'oblique']) if image_modality == 'XR' else 'axial'
            }
        
        # Add clinical data modality
        if modality_choice in [3, 4]:  # text_clinical, all_modalities
            query.clinical_data = self.clinical_generator.generate_clinical_data(
                query.complexity_level, query.medical_urgency
            )
        
        return query
    
    def _generate_candidates_for_query(self, query: MultimodalMedicalQuery) -> List[MultimodalMedicalCandidate]:
        """Generate candidate documents for a given query."""
        candidates = []
        num_candidates = self.config.min_documents_per_query
        
        content_types = ['guideline', 'case_study', 'research_paper', 'protocol']
        
        for i in range(num_candidates):
            candidate_id = f"{query.id}_candidate_{i:03d}"
            content_type = random.choice(content_types)
            
            # Vary specialty - some match, some don't
            if random.random() < 0.6:  # 60% specialty match
                specialty = query.specialty
            else:
                specialty = random.choice(list(self.dataset_config.specialty_distribution.keys()))
            
            candidate = MultimodalMedicalCandidate(
                id=candidate_id,
                content_type=content_type,
                specialty=specialty
            )
            
            # Add relevant diagnosis for some candidates
            if random.random() < 0.4:  # 40% have matching diagnosis
                candidate.diagnosis = query.ground_truth_diagnosis
            else:
                candidate.diagnosis = random.choice(self.text_generator.conditions)
            
            # Generate text content
            candidate.text = self.text_generator.generate_candidate_text(
                content_type, specialty, candidate.diagnosis
            )
            
            # Add clinical data for complex candidates
            if random.random() < 0.3:  # 30% have clinical data
                candidate.clinical_data = self.clinical_generator.generate_clinical_data(
                    'moderate', 'routine'
                )
            
            # Add images for some candidates
            if random.random() < 0.2:  # 20% have images
                modality = np.random.choice(
                    list(self.dataset_config.image_modality_distribution.keys()),
                    p=list(self.dataset_config.image_modality_distribution.values())
                )
                
                candidate.image = self.image_generator.generate_medical_image(
                    modality, 'chest', candidate.diagnosis if random.random() < 0.5 else None
                )
                
                candidate.image_metadata = {
                    'modality': modality,
                    'body_part': 'chest',
                    'view': 'AP'
                }
            
            # Set evidence level and applicability
            candidate.evidence_level = random.choice(['low', 'moderate', 'high'])
            candidate.clinical_applicability = random.uniform(0.3, 1.0)
            
            candidates.append(candidate)
        
        return candidates
    
    def _add_noise_variations(self, dataset: MultimodalMedicalDataset) -> MultimodalMedicalDataset:
        """Add noise variations to test robustness."""
        logger.info("Adding noise variations for robustness testing...")
        
        # Note: This creates variations in-place for the demo
        # In practice, you might want to create separate noisy versions
        
        for query in dataset.queries:
            # Add OCR errors to text
            if query.text and random.random() < self.config.ocr_error_rate:
                query.text = self._add_ocr_errors(query.text)
            
            # Add missing data to clinical records
            if query.clinical_data and random.random() < self.config.missing_data_rate:
                query.clinical_data = self._add_missing_clinical_data(query.clinical_data)
            
            # Add image artifacts (simulated by metadata)
            if query.image_metadata and random.random() < self.config.image_artifact_probability:
                query.image_metadata['artifacts'] = ['motion', 'noise', 'poor_contrast']
        
        # Add noise to candidates as well
        for candidates in dataset.candidates.values():
            for candidate in candidates:
                if candidate.text and random.random() < self.config.text_noise_level:
                    candidate.text = self._add_text_noise(candidate.text)
        
        return dataset
    
    def _add_ocr_errors(self, text: str) -> str:
        """Simulate OCR errors in text."""
        # Common OCR substitutions
        substitutions = {
            'o': '0', '0': 'o', 'i': '1', '1': 'i', 's': '5', '5': 's',
            'rn': 'm', 'm': 'rn', 'cl': 'd', 'vv': 'w'
        }
        
        words = text.split()
        for i, word in enumerate(words):
            if random.random() < 0.1:  # 10% chance per word
                for old, new in substitutions.items():
                    if old in word.lower():
                        words[i] = word.replace(old, new, 1)
                        break
        
        return ' '.join(words)
    
    def _add_missing_clinical_data(self, clinical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate missing clinical data."""
        modified_data = clinical_data.copy()
        
        # Randomly remove some lab values
        if 'lab_values' in modified_data and modified_data['lab_values']:
            labs_to_remove = random.sample(
                list(modified_data['lab_values'].keys()),
                min(2, len(modified_data['lab_values']))
            )
            for lab in labs_to_remove:
                del modified_data['lab_values'][lab]
        
        return modified_data
    
    def _add_text_noise(self, text: str) -> str:
        """Add subtle noise to text."""
        words = text.split()
        for i, word in enumerate(words):
            if random.random() < 0.05 and len(word) > 3:  # 5% chance for longer words
                # Random character substitution
                pos = random.randint(1, len(word) - 2)
                char_list = list(word)
                char_list[pos] = random.choice('abcdefghijklmnopqrstuvwxyz')
                words[i] = ''.join(char_list)
        
        return ' '.join(words)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test dataset generation
    config = MultimodalMedicalEvaluationConfig(
        min_multimodal_queries=10,  # Small for testing
        min_documents_per_query=20
    )
    
    generator = MultimodalMedicalDatasetGenerator(config)
    dataset = generator.generate_comprehensive_dataset()
    
    print("Dataset Information:")
    info = dataset.get_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
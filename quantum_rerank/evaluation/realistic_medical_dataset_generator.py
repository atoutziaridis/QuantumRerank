"""
Realistic Medical Dataset Generator for Unbiased QMMR-05 Evaluation.

Generates complex, realistic medical datasets using actual medical terminologies,
clinical guidelines, and representative document structures to ensure
evaluation validity and eliminate bias.
"""

import logging
import random
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from PIL import Image
import requests
import re

from quantum_rerank.config.evaluation_config import MultimodalMedicalEvaluationConfig
from quantum_rerank.evaluation.multimodal_medical_dataset_generator import (
    MultimodalMedicalDataset, MultimodalMedicalQuery, MultimodalMedicalCandidate
)

logger = logging.getLogger(__name__)


@dataclass
class RealMedicalDocument:
    """Represents a real medical document with complex structure."""
    
    doc_id: str
    doc_type: str  # clinical_guideline, case_report, research_paper, protocol
    title: str
    abstract: str
    full_text: str
    
    # Medical metadata
    specialty: str
    conditions: List[str]
    procedures: List[str]
    medications: List[str]
    
    # Document structure
    sections: Dict[str, str]  # section_name -> content
    references: List[str]
    
    # Quality indicators
    evidence_level: str  # A, B, C, D
    publication_year: int
    journal_impact_factor: Optional[float] = None
    
    # Content complexity metrics
    word_count: int = 0
    unique_medical_terms: int = 0
    complexity_score: float = 0.0


class MedicalTerminologyDatabase:
    """Comprehensive medical terminology database for realistic content generation."""
    
    def __init__(self):
        # Initialize comprehensive medical terminology
        self.conditions = self._load_medical_conditions()
        self.symptoms = self._load_symptoms()
        self.procedures = self._load_procedures()
        self.medications = self._load_medications()
        self.anatomical_terms = self._load_anatomical_terms()
        self.lab_tests = self._load_lab_tests()
        self.imaging_findings = self._load_imaging_findings()
        
        # Medical specialties with their focus areas
        self.specialties = {
            'cardiology': {
                'conditions': ['myocardial infarction', 'heart failure', 'arrhythmia', 'coronary artery disease'],
                'procedures': ['cardiac catheterization', 'angioplasty', 'echocardiography', 'stress test'],
                'medications': ['metoprolol', 'lisinopril', 'atorvastatin', 'aspirin'],
                'imaging': ['echocardiogram', 'cardiac MRI', 'coronary angiography']
            },
            'pulmonology': {
                'conditions': ['pneumonia', 'COPD', 'asthma', 'pulmonary embolism'],
                'procedures': ['bronchoscopy', 'pulmonary function test', 'thoracentesis'],
                'medications': ['albuterol', 'prednisone', 'azithromycin', 'heparin'],
                'imaging': ['chest X-ray', 'CT chest', 'pulmonary angiogram']
            },
            'neurology': {
                'conditions': ['stroke', 'epilepsy', 'multiple sclerosis', 'migraine'],
                'procedures': ['lumbar puncture', 'EEG', 'nerve conduction study'],
                'medications': ['levetiracetam', 'gabapentin', 'sumatriptan', 'baclofen'],
                'imaging': ['brain MRI', 'CT head', 'cerebral angiography']
            },
            'emergency_medicine': {
                'conditions': ['trauma', 'sepsis', 'cardiac arrest', 'respiratory failure'],
                'procedures': ['intubation', 'central line', 'chest tube', 'cardioversion'],
                'medications': ['epinephrine', 'norepinephrine', 'midazolam', 'fentanyl'],
                'imaging': ['trauma pan-scan', 'bedside ultrasound', 'portable chest X-ray']
            },
            'radiology': {
                'conditions': ['mass lesion', 'fracture', 'hemorrhage', 'infection'],
                'procedures': ['CT scan', 'MRI', 'ultrasound', 'biopsy'],
                'medications': ['contrast agent', 'gadolinium', 'barium'],
                'imaging': ['radiography', 'fluoroscopy', 'nuclear medicine']
            }
        }
        
        # Clinical document templates for realistic structure
        self.document_templates = self._load_document_templates()
        
        logger.info("Initialized comprehensive medical terminology database")
    
    def _load_medical_conditions(self) -> List[str]:
        """Load comprehensive list of medical conditions."""
        return [
            # Cardiovascular
            'myocardial infarction', 'congestive heart failure', 'atrial fibrillation', 
            'hypertension', 'coronary artery disease', 'valvular heart disease',
            'cardiomyopathy', 'pericarditis', 'aortic stenosis', 'mitral regurgitation',
            
            # Respiratory
            'pneumonia', 'chronic obstructive pulmonary disease', 'asthma',
            'pulmonary embolism', 'pleural effusion', 'pneumothorax',
            'acute respiratory distress syndrome', 'lung cancer', 'tuberculosis',
            
            # Neurological
            'stroke', 'transient ischemic attack', 'epilepsy', 'multiple sclerosis',
            'Parkinson disease', 'Alzheimer disease', 'migraine', 'seizure disorder',
            'subarachnoid hemorrhage', 'intracerebral hemorrhage',
            
            # Infectious
            'sepsis', 'urinary tract infection', 'cellulitis', 'meningitis',
            'endocarditis', 'osteomyelitis', 'pneumocystis pneumonia',
            
            # Gastrointestinal
            'acute appendicitis', 'cholecystitis', 'pancreatitis', 'inflammatory bowel disease',
            'gastroesophageal reflux disease', 'peptic ulcer disease', 'hepatitis',
            
            # Endocrine
            'diabetes mellitus', 'diabetic ketoacidosis', 'hyperthyroidism', 'hypothyroidism',
            'adrenal insufficiency', 'hyperparathyroidism',
            
            # Renal
            'acute kidney injury', 'chronic kidney disease', 'nephrolithiasis',
            'glomerulonephritis', 'polycystic kidney disease',
            
            # Hematologic
            'anemia', 'thrombocytopenia', 'deep vein thrombosis', 'leukemia',
            'lymphoma', 'multiple myeloma', 'sickle cell disease'
        ]
    
    def _load_symptoms(self) -> List[str]:
        """Load comprehensive symptom list."""
        return [
            'chest pain', 'shortness of breath', 'dyspnea on exertion', 'orthopnea',
            'paroxysmal nocturnal dyspnea', 'palpitations', 'syncope', 'presyncope',
            'fatigue', 'weakness', 'malaise', 'fever', 'chills', 'night sweats',
            'weight loss', 'weight gain', 'anorexia', 'nausea', 'vomiting',
            'diarrhea', 'constipation', 'abdominal pain', 'headache', 'dizziness',
            'confusion', 'altered mental status', 'seizure', 'focal neurological deficit',
            'hemiparesis', 'aphasia', 'dysarthria', 'visual disturbance',
            'cough', 'hemoptysis', 'wheezing', 'stridor', 'hoarseness',
            'dysphagia', 'odynophagia', 'heartburn', 'melena', 'hematochezia',
            'dysuria', 'frequency', 'urgency', 'hematuria', 'oliguria', 'anuria',
            'joint pain', 'muscle pain', 'back pain', 'neck pain', 'stiffness',
            'swelling', 'edema', 'lymphadenopathy', 'rash', 'pruritus',
            'diaphoresis', 'pallor', 'cyanosis', 'jaundice'
        ]
    
    def _load_procedures(self) -> List[str]:
        """Load comprehensive procedure list."""
        return [
            # Diagnostic procedures
            'electrocardiography', 'echocardiography', 'stress testing',
            'cardiac catheterization', 'coronary angiography', 'electrophysiology study',
            'pulmonary function testing', 'bronchoscopy', 'thoracentesis',
            'electroencephalography', 'lumbar puncture', 'nerve conduction study',
            'electromyography', 'upper endoscopy', 'colonoscopy', 'sigmoidoscopy',
            
            # Therapeutic procedures
            'percutaneous coronary intervention', 'coronary artery bypass grafting',
            'valve replacement', 'pacemaker implantation', 'defibrillator implantation',
            'ablation therapy', 'thrombolytic therapy', 'mechanical ventilation',
            'hemodialysis', 'peritoneal dialysis', 'plasmapheresis',
            
            # Surgical procedures
            'appendectomy', 'cholecystectomy', 'laparoscopy', 'thoracotomy',
            'craniotomy', 'carotid endarterectomy', 'nephrectomy', 'transplantation',
            
            # Emergency procedures
            'endotracheal intubation', 'central venous catheterization',
            'arterial line placement', 'chest tube insertion', 'cardioversion',
            'defibrillation', 'cardiopulmonary resuscitation'
        ]
    
    def _load_medications(self) -> List[str]:
        """Load comprehensive medication list with generic and brand names."""
        return [
            # Cardiovascular medications
            'aspirin', 'clopidogrel', 'warfarin', 'heparin', 'enoxaparin',
            'metoprolol', 'carvedilol', 'lisinopril', 'losartan', 'amlodipine',
            'hydrochlorothiazide', 'furosemide', 'spironolactone',
            'atorvastatin', 'simvastatin', 'rosuvastatin',
            
            # Respiratory medications
            'albuterol', 'ipratropium', 'fluticasone', 'budesonide',
            'montelukast', 'theophylline', 'acetylcysteine',
            
            # Neurological medications
            'levetiracetam', 'phenytoin', 'carbamazepine', 'valproic acid',
            'gabapentin', 'pregabalin', 'baclofen', 'tizanidine',
            'sumatriptan', 'topiramate', 'amitriptyline',
            
            # Infectious disease medications
            'amoxicillin', 'azithromycin', 'ciprofloxacin', 'vancomycin',
            'ceftriaxone', 'piperacillin-tazobactam', 'meropenem',
            'fluconazole', 'acyclovir', 'oseltamivir',
            
            # Endocrine medications
            'insulin', 'metformin', 'glipizide', 'levothyroxine',
            'prednisone', 'hydrocortisone', 'dexamethasone',
            
            # Pain and sedation
            'morphine', 'fentanyl', 'oxycodone', 'tramadol',
            'acetaminophen', 'ibuprofen', 'ketorolac',
            'midazolam', 'lorazepam', 'propofol'
        ]
    
    def _load_anatomical_terms(self) -> List[str]:
        """Load anatomical terms for precise medical descriptions."""
        return [
            'left anterior descending artery', 'right coronary artery', 'circumflex artery',
            'left main coronary artery', 'aortic root', 'mitral valve', 'tricuspid valve',
            'left ventricle', 'right ventricle', 'left atrium', 'right atrium',
            'ascending aorta', 'aortic arch', 'descending aorta',
            
            'right upper lobe', 'right middle lobe', 'right lower lobe',
            'left upper lobe', 'left lower lobe', 'lingula',
            'trachea', 'main bronchi', 'segmental bronchi', 'pleural space',
            
            'frontal lobe', 'parietal lobe', 'temporal lobe', 'occipital lobe',
            'cerebellum', 'brainstem', 'basal ganglia', 'thalamus',
            'corpus callosum', 'ventricular system', 'subarachnoid space',
            
            'hepatic segments', 'portal vein', 'hepatic artery', 'bile ducts',
            'gallbladder', 'pancreatic head', 'pancreatic tail', 'spleen',
            
            'renal cortex', 'renal medulla', 'renal pelvis', 'ureter',
            'bladder', 'urethra', 'prostate', 'seminal vesicles'
        ]
    
    def _load_lab_tests(self) -> List[str]:
        """Load comprehensive laboratory test list."""
        return [
            # Basic metabolic panel
            'sodium', 'potassium', 'chloride', 'carbon dioxide', 'blood urea nitrogen',
            'creatinine', 'glucose', 'calcium', 'magnesium', 'phosphorus',
            
            # Complete blood count
            'white blood cell count', 'red blood cell count', 'hemoglobin', 'hematocrit',
            'mean corpuscular volume', 'platelet count', 'neutrophils', 'lymphocytes',
            'monocytes', 'eosinophils', 'basophils',
            
            # Liver function tests
            'alanine aminotransferase', 'aspartate aminotransferase', 'alkaline phosphatase',
            'total bilirubin', 'direct bilirubin', 'albumin', 'total protein',
            
            # Cardiac markers
            'troponin I', 'troponin T', 'creatine kinase-MB', 'B-type natriuretic peptide',
            'N-terminal pro-B-type natriuretic peptide',
            
            # Coagulation studies
            'prothrombin time', 'international normalized ratio', 'activated partial thromboplastin time',
            'fibrinogen', 'D-dimer',
            
            # Inflammatory markers
            'erythrocyte sedimentation rate', 'C-reactive protein', 'procalcitonin',
            
            # Endocrine tests
            'thyroid stimulating hormone', 'free thyroxine', 'hemoglobin A1c',
            'cortisol', 'adrenocorticotropic hormone',
            
            # Arterial blood gas
            'pH', 'partial pressure of carbon dioxide', 'partial pressure of oxygen',
            'bicarbonate', 'base excess', 'oxygen saturation'
        ]
    
    def _load_imaging_findings(self) -> Dict[str, List[str]]:
        """Load imaging findings by modality."""
        return {
            'chest_xray': [
                'bilateral infiltrates', 'unilateral infiltrate', 'pleural effusion',
                'pneumothorax', 'cardiomegaly', 'pulmonary edema', 'consolidation',
                'atelectasis', 'hyperinflation', 'nodular opacities', 'mass lesion',
                'hilar lymphadenopathy', 'costophrenic angle blunting'
            ],
            'ct_chest': [
                'ground glass opacities', 'honeycombing', 'tree-in-bud pattern',
                'mosaic attenuation', 'pulmonary embolism', 'mediastinal lymphadenopathy',
                'pleural thickening', 'emphysematous changes', 'bronchiectasis'
            ],
            'echocardiography': [
                'left ventricular systolic dysfunction', 'diastolic dysfunction',
                'mitral regurgitation', 'aortic stenosis', 'tricuspid regurgitation',
                'pericardial effusion', 'wall motion abnormalities', 'ventricular hypertrophy'
            ],
            'brain_mri': [
                'acute infarction', 'chronic infarction', 'hemorrhage', 'mass effect',
                'midline shift', 'hydrocephalus', 'white matter hyperintensities',
                'diffusion restriction', 'enhancement', 'edema'
            ],
            'abdominal_ct': [
                'hepatomegaly', 'splenomegaly', 'ascites', 'lymphadenopathy',
                'bowel obstruction', 'free air', 'free fluid', 'abscess',
                'biliary dilatation', 'pancreatic ductal dilatation'
            ]
        }
    
    def _load_document_templates(self) -> Dict[str, Dict[str, str]]:
        """Load document structure templates for different document types."""
        return {
            'clinical_guideline': {
                'abstract': 'Clinical practice guideline for {condition} management in {setting}. This evidence-based guideline provides recommendations for diagnosis, treatment, and monitoring of patients with {condition}.',
                'introduction': 'Background and scope of the guideline for {condition} management.',
                'methodology': 'Systematic review of literature and expert consensus process.',
                'recommendations': 'Evidence-based recommendations for clinical practice.',
                'implementation': 'Strategies for implementing guideline recommendations.',
                'monitoring': 'Quality indicators and monitoring recommendations.'
            },
            'case_report': {
                'abstract': 'Case report of {age}-year-old {gender} presenting with {chief_complaint}.',
                'case_presentation': 'Detailed clinical presentation and history.',
                'diagnostic_workup': 'Comprehensive diagnostic evaluation including {tests}.',
                'treatment': 'Treatment approach and clinical course.',
                'outcome': 'Patient outcome and follow-up.',
                'discussion': 'Clinical significance and literature review.'
            },
            'research_paper': {
                'abstract': 'Study investigating {research_question} in patients with {condition}.',
                'introduction': 'Background, rationale, and study objectives.',
                'methods': 'Study design, participants, and statistical analysis.',
                'results': 'Primary and secondary outcome results.',
                'discussion': 'Interpretation of findings and clinical implications.',
                'conclusion': 'Summary of key findings and recommendations.'
            },
            'protocol': {
                'abstract': 'Clinical protocol for management of {condition} in {setting}.',
                'scope': 'Applicability and target population.',
                'definitions': 'Key terms and clinical criteria.',
                'procedure': 'Step-by-step clinical procedures.',
                'monitoring': 'Patient monitoring and safety considerations.',
                'quality_assurance': 'Quality metrics and continuous improvement.'
            }
        }


class RealMedicalContentGenerator:
    """Generates realistic medical content using evidence-based templates and terminology."""
    
    def __init__(self, terminology_db: MedicalTerminologyDatabase):
        self.terminology_db = terminology_db
        self.content_complexity_targets = {
            'simple': {'word_count': (500, 1500), 'medical_terms': (20, 50)},
            'moderate': {'word_count': (1500, 3000), 'medical_terms': (50, 100)},
            'complex': {'word_count': (3000, 6000), 'medical_terms': (100, 200)},
            'very_complex': {'word_count': (6000, 12000), 'medical_terms': (200, 400)}
        }
    
    def generate_clinical_guideline(
        self, 
        specialty: str, 
        condition: str, 
        complexity_level: str
    ) -> RealMedicalDocument:
        """Generate realistic clinical practice guideline."""
        
        doc_id = f"guideline_{specialty}_{condition}_{random.randint(1000, 9999)}"
        
        # Get specialty-specific content
        specialty_data = self.terminology_db.specialties.get(specialty, {})
        
        # Generate structured content
        title = f"Clinical Practice Guideline for {condition.title()} Management in {specialty.title()}"
        
        abstract = self._generate_guideline_abstract(condition, specialty, specialty_data)
        
        sections = self._generate_guideline_sections(
            condition, specialty, specialty_data, complexity_level
        )
        
        full_text = self._combine_sections(sections)
        
        # Extract medical entities
        conditions = self._extract_conditions(full_text)
        procedures = self._extract_procedures(full_text)
        medications = self._extract_medications(full_text)
        
        # Calculate complexity metrics
        word_count = len(full_text.split())
        unique_medical_terms = len(set(conditions + procedures + medications))
        complexity_score = self._calculate_complexity_score(word_count, unique_medical_terms, sections)
        
        return RealMedicalDocument(
            doc_id=doc_id,
            doc_type='clinical_guideline',
            title=title,
            abstract=abstract,
            full_text=full_text,
            specialty=specialty,
            conditions=conditions,
            procedures=procedures,
            medications=medications,
            sections=sections,
            references=self._generate_references(15, 25),
            evidence_level='A',
            publication_year=random.randint(2020, 2024),
            journal_impact_factor=random.uniform(5.0, 15.0),
            word_count=word_count,
            unique_medical_terms=unique_medical_terms,
            complexity_score=complexity_score
        )
    
    def generate_case_report(
        self, 
        specialty: str, 
        primary_condition: str, 
        complexity_level: str
    ) -> RealMedicalDocument:
        """Generate realistic clinical case report."""
        
        doc_id = f"case_{specialty}_{primary_condition}_{random.randint(1000, 9999)}"
        
        # Patient demographics
        age = random.randint(18, 90)
        gender = random.choice(['male', 'female'])
        
        specialty_data = self.terminology_db.specialties.get(specialty, {})
        
        title = f"Case Report: {primary_condition.title()} in {age}-Year-Old {gender.title()}"
        
        abstract = self._generate_case_abstract(age, gender, primary_condition, specialty)
        
        sections = self._generate_case_sections(
            age, gender, primary_condition, specialty, specialty_data, complexity_level
        )
        
        full_text = self._combine_sections(sections)
        
        # Extract medical entities
        conditions = self._extract_conditions(full_text)
        procedures = self._extract_procedures(full_text)
        medications = self._extract_medications(full_text)
        
        # Calculate complexity metrics
        word_count = len(full_text.split())
        unique_medical_terms = len(set(conditions + procedures + medications))
        complexity_score = self._calculate_complexity_score(word_count, unique_medical_terms, sections)
        
        return RealMedicalDocument(
            doc_id=doc_id,
            doc_type='case_report',
            title=title,
            abstract=abstract,
            full_text=full_text,
            specialty=specialty,
            conditions=conditions,
            procedures=procedures,
            medications=medications,
            sections=sections,
            references=self._generate_references(8, 15),
            evidence_level='C',
            publication_year=random.randint(2021, 2024),
            journal_impact_factor=random.uniform(2.0, 8.0),
            word_count=word_count,
            unique_medical_terms=unique_medical_terms,
            complexity_score=complexity_score
        )
    
    def generate_research_paper(
        self, 
        specialty: str, 
        research_focus: str, 
        complexity_level: str
    ) -> RealMedicalDocument:
        """Generate realistic research paper."""
        
        doc_id = f"research_{specialty}_{research_focus}_{random.randint(1000, 9999)}"
        
        specialty_data = self.terminology_db.specialties.get(specialty, {})
        
        title = f"{research_focus.title()} in {specialty.title()}: A Randomized Controlled Trial"
        
        abstract = self._generate_research_abstract(research_focus, specialty)
        
        sections = self._generate_research_sections(
            research_focus, specialty, specialty_data, complexity_level
        )
        
        full_text = self._combine_sections(sections)
        
        # Extract medical entities
        conditions = self._extract_conditions(full_text)
        procedures = self._extract_procedures(full_text)
        medications = self._extract_medications(full_text)
        
        # Calculate complexity metrics
        word_count = len(full_text.split())
        unique_medical_terms = len(set(conditions + procedures + medications))
        complexity_score = self._calculate_complexity_score(word_count, unique_medical_terms, sections)
        
        return RealMedicalDocument(
            doc_id=doc_id,
            doc_type='research_paper',
            title=title,
            abstract=abstract,
            full_text=full_text,
            specialty=specialty,
            conditions=conditions,
            procedures=procedures,
            medications=medications,
            sections=sections,
            references=self._generate_references(30, 60),
            evidence_level='A',
            publication_year=random.randint(2019, 2024),
            journal_impact_factor=random.uniform(8.0, 25.0),
            word_count=word_count,
            unique_medical_terms=unique_medical_terms,
            complexity_score=complexity_score
        )
    
    def _generate_guideline_abstract(self, condition: str, specialty: str, specialty_data: Dict) -> str:
        """Generate realistic guideline abstract."""
        template = self.terminology_db.document_templates['clinical_guideline']['abstract']
        
        abstract = template.format(
            condition=condition,
            setting=f"{specialty} practice"
        )
        
        # Add evidence-based details
        procedures = random.sample(specialty_data.get('procedures', []), min(2, len(specialty_data.get('procedures', []))))
        medications = random.sample(specialty_data.get('medications', []), min(2, len(specialty_data.get('medications', []))))
        
        abstract += f" Key recommendations include early {procedures[0] if procedures else 'diagnostic evaluation'}"
        if medications:
            abstract += f" and evidence-based pharmacotherapy with {medications[0]}"
        abstract += f". Implementation of these guidelines is expected to improve patient outcomes and standardize {specialty} practice."
        
        return abstract
    
    def _generate_guideline_sections(
        self, 
        condition: str, 
        specialty: str, 
        specialty_data: Dict, 
        complexity_level: str
    ) -> Dict[str, str]:
        """Generate comprehensive guideline sections."""
        
        target_params = self.content_complexity_targets[complexity_level]
        
        sections = {}
        
        # Introduction
        sections['introduction'] = self._generate_detailed_introduction(
            condition, specialty, specialty_data, target_params['word_count'][0] // 6
        )
        
        # Methodology
        sections['methodology'] = self._generate_methodology_section(target_params['word_count'][0] // 8)
        
        # Diagnostic recommendations
        sections['diagnosis'] = self._generate_diagnostic_recommendations(
            condition, specialty_data, target_params['word_count'][0] // 4
        )
        
        # Treatment recommendations
        sections['treatment'] = self._generate_treatment_recommendations(
            condition, specialty_data, target_params['word_count'][0] // 3
        )
        
        # Monitoring
        sections['monitoring'] = self._generate_monitoring_recommendations(
            condition, specialty_data, target_params['word_count'][0] // 6
        )
        
        # Special populations
        if complexity_level in ['complex', 'very_complex']:
            sections['special_populations'] = self._generate_special_populations_section(
                condition, target_params['word_count'][0] // 8
            )
        
        return sections
    
    def _generate_detailed_introduction(
        self, 
        condition: str, 
        specialty: str, 
        specialty_data: Dict, 
        target_words: int
    ) -> str:
        """Generate detailed introduction section."""
        
        content = f"Background: {condition.title()} represents a significant clinical challenge in {specialty} practice. "
        
        # Epidemiology
        content += f"The prevalence of {condition} has been increasing, with current estimates suggesting "
        content += f"an incidence rate of {random.randint(5, 50)} per 100,000 population annually. "
        
        # Clinical significance
        content += f"This condition is associated with substantial morbidity and mortality if not appropriately managed. "
        
        # Current practice gaps
        content += f"Despite advances in {specialty} care, significant variations in clinical practice persist, "
        content += f"highlighting the need for evidence-based standardized approaches. "
        
        # Guideline scope
        specialty_conditions = specialty_data.get('conditions', [])
        if specialty_conditions:
            related_conditions = random.sample(specialty_conditions, min(2, len(specialty_conditions)))
            content += f"This guideline addresses {condition} management while considering related conditions "
            content += f"including {', '.join(related_conditions)}. "
        
        # Guideline development
        content += f"The development of this guideline involved systematic literature review, "
        content += f"expert consensus, and consideration of patient values and preferences. "
        
        # Target audience
        content += f"The intended audience includes {specialty} physicians, "
        content += f"advanced practice providers, and multidisciplinary care teams involved in {condition} management."
        
        return self._expand_content_to_target(content, target_words, specialty_data)
    
    def _generate_diagnostic_recommendations(
        self, 
        condition: str, 
        specialty_data: Dict, 
        target_words: int
    ) -> str:
        """Generate comprehensive diagnostic recommendations."""
        
        content = f"Diagnostic Approach for {condition.title()}\n\n"
        
        # Clinical presentation
        symptoms = random.sample(self.terminology_db.symptoms, min(5, len(self.terminology_db.symptoms)))
        content += f"Clinical Presentation: Patients with {condition} typically present with {symptoms[0]}, "
        content += f"often accompanied by {symptoms[1]} and {symptoms[2]}. "
        content += f"Additional symptoms may include {symptoms[3]} and {symptoms[4]}. "
        
        # Physical examination
        content += f"Physical examination should focus on comprehensive assessment including vital signs, "
        content += f"cardiovascular examination, pulmonary assessment, and neurological evaluation as indicated. "
        
        # Laboratory studies
        lab_tests = random.sample(self.terminology_db.lab_tests, min(6, len(self.terminology_db.lab_tests)))
        content += f"Recommended laboratory studies include {lab_tests[0]}, {lab_tests[1]}, "
        content += f"and {lab_tests[2]}. Additional testing with {lab_tests[3]} and {lab_tests[4]} "
        content += f"should be considered based on clinical presentation. "
        
        # Imaging studies
        if specialty_data.get('imaging'):
            imaging_studies = random.sample(specialty_data['imaging'], min(2, len(specialty_data['imaging'])))
            content += f"Imaging studies should include {imaging_studies[0]} as initial evaluation. "
            if len(imaging_studies) > 1:
                content += f"Advanced imaging with {imaging_studies[1]} may be indicated in complex cases. "
        
        # Diagnostic procedures
        procedures = specialty_data.get('procedures', [])
        if procedures:
            diagnostic_procedures = random.sample(procedures, min(2, len(procedures)))
            content += f"Specialized diagnostic procedures including {diagnostic_procedures[0]} "
            content += f"should be considered when initial evaluation is inconclusive. "
        
        # Differential diagnosis
        related_conditions = random.sample(self.terminology_db.conditions, min(4, len(self.terminology_db.conditions)))
        content += f"Differential diagnosis should include {related_conditions[0]}, {related_conditions[1]}, "
        content += f"{related_conditions[2]}, and {related_conditions[3]}. "
        
        return self._expand_content_to_target(content, target_words, specialty_data)
    
    def _generate_treatment_recommendations(
        self, 
        condition: str, 
        specialty_data: Dict, 
        target_words: int
    ) -> str:
        """Generate comprehensive treatment recommendations."""
        
        content = f"Treatment Recommendations for {condition.title()}\n\n"
        
        # General principles
        content += f"Treatment of {condition} should follow a multidisciplinary approach with "
        content += f"individualized care plans based on patient characteristics, disease severity, "
        content += f"and comorbidity profile. "
        
        # Pharmacological treatment
        medications = specialty_data.get('medications', [])
        if medications:
            primary_meds = random.sample(medications, min(3, len(medications)))
            content += f"First-line pharmacological therapy includes {primary_meds[0]} "
            content += f"with initial dosing adjusted for renal function and patient tolerance. "
            
            if len(primary_meds) > 1:
                content += f"Combination therapy with {primary_meds[1]} may be considered "
                content += f"for patients with inadequate response to monotherapy. "
            
            if len(primary_meds) > 2:
                content += f"Alternative agents including {primary_meds[2]} should be reserved "
                content += f"for patients with contraindications to first-line therapy. "
        
        # Non-pharmacological interventions
        content += f"Non-pharmacological interventions play a crucial role in comprehensive management. "
        content += f"Patient education, lifestyle modifications, and adherence counseling are essential components. "
        
        # Procedural interventions
        procedures = specialty_data.get('procedures', [])
        if procedures:
            therapeutic_procedures = random.sample(procedures, min(2, len(procedures)))
            content += f"Procedural interventions including {therapeutic_procedures[0]} "
            content += f"should be considered for patients meeting specific clinical criteria. "
        
        # Monitoring and follow-up
        content += f"Regular monitoring is essential with clinical assessment every 3-6 months "
        content += f"and laboratory monitoring as clinically indicated. "
        
        # Treatment goals
        content += f"Treatment goals include symptom resolution, prevention of complications, "
        content += f"improvement in quality of life, and optimization of functional status. "
        
        return self._expand_content_to_target(content, target_words, specialty_data)
    
    def _expand_content_to_target(self, content: str, target_words: int, specialty_data: Dict) -> str:
        """Expand content to reach target word count with realistic medical details."""
        
        current_words = len(content.split())
        
        if current_words >= target_words:
            return content
        
        # Add additional clinical details
        additional_content = []
        
        # Add contraindications and cautions
        additional_content.append(
            "Contraindications include known hypersensitivity, severe renal impairment, "
            "and concurrent use of interacting medications. "
        )
        
        # Add drug interactions
        if specialty_data.get('medications'):
            med = random.choice(specialty_data['medications'])
            additional_content.append(
                f"Important drug interactions with {med} include cytochrome P450 inhibitors "
                "and medications affecting renal clearance. "
            )
        
        # Add adverse effects
        additional_content.append(
            "Common adverse effects include gastrointestinal upset, dizziness, and fatigue. "
            "Serious adverse reactions are rare but may include hepatotoxicity and cardiovascular events. "
        )
        
        # Add patient counseling points
        additional_content.append(
            "Patient counseling should emphasize medication adherence, recognition of adverse effects, "
            "and importance of regular follow-up appointments. "
        )
        
        # Add quality measures
        additional_content.append(
            "Quality measures include time to diagnosis, appropriate medication selection, "
            "adherence to monitoring guidelines, and patient-reported outcome measures. "
        )
        
        # Combine and trim to target
        expanded_content = content + " ".join(additional_content)
        words = expanded_content.split()
        
        if len(words) > target_words:
            words = words[:target_words]
            expanded_content = " ".join(words)
        
        return expanded_content
    
    def _combine_sections(self, sections: Dict[str, str]) -> str:
        """Combine sections into full document text."""
        full_text = ""
        for section_name, content in sections.items():
            full_text += f"\n\n{section_name.upper().replace('_', ' ')}\n\n{content}"
        return full_text.strip()
    
    def _extract_conditions(self, text: str) -> List[str]:
        """Extract medical conditions from text."""
        conditions = []
        text_lower = text.lower()
        
        for condition in self.terminology_db.conditions:
            if condition.lower() in text_lower:
                conditions.append(condition)
        
        return list(set(conditions))
    
    def _extract_procedures(self, text: str) -> List[str]:
        """Extract medical procedures from text."""
        procedures = []
        text_lower = text.lower()
        
        for procedure in self.terminology_db.procedures:
            if procedure.lower() in text_lower:
                procedures.append(procedure)
        
        return list(set(procedures))
    
    def _extract_medications(self, text: str) -> List[str]:
        """Extract medications from text."""
        medications = []
        text_lower = text.lower()
        
        for medication in self.terminology_db.medications:
            if medication.lower() in text_lower:
                medications.append(medication)
        
        return list(set(medications))
    
    def _calculate_complexity_score(self, word_count: int, unique_terms: int, sections: Dict) -> float:
        """Calculate document complexity score."""
        # Base score from content metrics
        complexity = 0.0
        
        # Word count contribution (normalized)
        complexity += min(word_count / 10000, 1.0) * 0.3
        
        # Medical terminology density
        terminology_density = unique_terms / max(word_count / 100, 1)  # Terms per 100 words
        complexity += min(terminology_density / 10, 1.0) * 0.4
        
        # Section complexity
        section_complexity = len(sections) / 10
        complexity += min(section_complexity, 1.0) * 0.3
        
        return min(complexity, 1.0)
    
    def _generate_references(self, min_refs: int, max_refs: int) -> List[str]:
        """Generate realistic medical references."""
        num_refs = random.randint(min_refs, max_refs)
        references = []
        
        journals = [
            "New England Journal of Medicine", "The Lancet", "JAMA", "Nature Medicine",
            "Circulation", "American Journal of Cardiology", "Chest", "Neurology",
            "Gastroenterology", "Journal of Clinical Oncology", "Critical Care Medicine"
        ]
        
        for i in range(num_refs):
            journal = random.choice(journals)
            year = random.randint(2015, 2024)
            volume = random.randint(300, 400)
            pages = f"{random.randint(1000, 2000)}-{random.randint(2001, 3000)}"
            
            ref = f"Author et al. Clinical study title. {journal}. {year};{volume}:{pages}."
            references.append(ref)
        
        return references
    
    # Additional helper methods for case reports and research papers...
    def _generate_case_abstract(self, age: int, gender: str, condition: str, specialty: str) -> str:
        """Generate case report abstract."""
        return (f"We report a case of {age}-year-old {gender} presenting with {condition} "
                f"in the {specialty} setting. The case demonstrates unique clinical features "
                f"and highlights important diagnostic and therapeutic considerations.")
    
    def _generate_case_sections(self, age: int, gender: str, condition: str, specialty: str, 
                              specialty_data: Dict, complexity_level: str) -> Dict[str, str]:
        """Generate case report sections."""
        sections = {}
        
        # Case presentation
        symptoms = random.sample(self.terminology_db.symptoms, 3)
        sections['case_presentation'] = (
            f"A {age}-year-old {gender} presented to the {specialty} service with "
            f"chief complaint of {symptoms[0]}. Associated symptoms included {symptoms[1]} "
            f"and {symptoms[2]}. Past medical history was significant for hypertension."
        )
        
        # Diagnostic workup
        procedures = specialty_data.get('procedures', [])[:2]
        sections['diagnostic_workup'] = (
            f"Initial evaluation included comprehensive history and physical examination. "
            f"Diagnostic studies performed included {procedures[0] if procedures else 'laboratory studies'} "
            f"and appropriate imaging studies."
        )
        
        return sections
    
    def _generate_research_abstract(self, research_focus: str, specialty: str) -> str:
        """Generate research paper abstract."""
        return (f"Background: {research_focus} remains an important clinical challenge in {specialty}. "
                f"Methods: Randomized controlled trial involving patients with relevant conditions. "
                f"Results: Primary endpoint was met with statistical significance. "
                f"Conclusions: Findings support evidence-based approach to {research_focus} management.")
    
    def _generate_research_sections(self, research_focus: str, specialty: str, 
                                  specialty_data: Dict, complexity_level: str) -> Dict[str, str]:
        """Generate research paper sections."""
        sections = {}
        
        sections['introduction'] = (
            f"Background and rationale for studying {research_focus} in {specialty} practice. "
            f"Previous studies have shown mixed results regarding optimal management approaches."
        )
        
        sections['methods'] = (
            f"Randomized controlled trial design with appropriate inclusion and exclusion criteria. "
            f"Primary endpoint was clinically relevant outcome measure. Statistical analysis "
            f"performed using appropriate methods with significance set at p<0.05."
        )
        
        sections['results'] = (
            f"Total of {random.randint(200, 500)} patients enrolled. Primary endpoint achieved "
            f"statistical significance (p={random.uniform(0.001, 0.049):.3f}). "
            f"Secondary endpoints also demonstrated favorable outcomes."
        )
        
        return sections
    
    def _generate_methodology_section(self, target_words: int) -> str:
        """Generate methodology section for guidelines."""
        content = (
            "Systematic literature search was conducted using MEDLINE, Embase, and Cochrane databases. "
            "Search terms included relevant medical subject headings and free text terms. "
            "Studies were selected based on predefined inclusion and exclusion criteria. "
            "Evidence quality was assessed using GRADE methodology. "
            "Expert panel reviewed evidence and developed recommendations through consensus process."
        )
        return content
    
    def _generate_monitoring_recommendations(self, condition: str, specialty_data: Dict, target_words: int) -> str:
        """Generate monitoring recommendations."""
        lab_tests = random.sample(self.terminology_db.lab_tests, 3)
        content = (
            f"Regular monitoring is essential for patients with {condition}. "
            f"Laboratory monitoring should include {lab_tests[0]}, {lab_tests[1]}, and {lab_tests[2]} "
            f"at baseline and during follow-up. Clinical assessment should occur at regular intervals "
            f"with focus on symptom resolution and functional improvement."
        )
        return content
    
    def _generate_special_populations_section(self, condition: str, target_words: int) -> str:
        """Generate special populations section."""
        content = (
            f"Special considerations are required for specific patient populations with {condition}. "
            f"Pediatric patients require age-appropriate dosing and monitoring. "
            f"Elderly patients may require dose adjustments due to altered pharmacokinetics. "
            f"Pregnant and lactating women require careful risk-benefit assessment."
        )
        return content


class RealisticMedicalDatasetGenerator:
    """
    Enhanced dataset generator using realistic medical content and unbiased evaluation methods.
    
    Generates complex, realistic medical datasets using actual medical terminologies,
    clinical guidelines, and representative document structures to ensure evaluation validity.
    """
    
    def __init__(self, config: MultimodalMedicalEvaluationConfig):
        self.config = config
        
        # Initialize medical terminology database
        self.terminology_db = MedicalTerminologyDatabase()
        
        # Initialize content generator
        self.content_generator = RealMedicalContentGenerator(self.terminology_db)
        
        # Document type distribution for realistic dataset
        self.document_type_distribution = {
            'clinical_guideline': 0.25,  # 25% guidelines
            'case_report': 0.35,         # 35% case reports
            'research_paper': 0.30,      # 30% research papers
            'protocol': 0.10             # 10% protocols
        }
        
        # Complexity distribution reflecting real-world scenarios
        self.complexity_distribution = {
            'simple': 0.15,      # 15% simple cases
            'moderate': 0.35,    # 35% moderate cases
            'complex': 0.35,     # 35% complex cases
            'very_complex': 0.15 # 15% very complex cases
        }
        
        logger.info("Initialized RealisticMedicalDatasetGenerator with comprehensive medical knowledge base")
    
    def generate_unbiased_dataset(self) -> MultimodalMedicalDataset:
        """
        Generate unbiased, realistic medical dataset with complex documents.
        """
        logger.info("Generating unbiased realistic medical dataset...")
        
        dataset = MultimodalMedicalDataset()
        dataset.metadata = {
            'generation_type': 'realistic_unbiased',
            'terminology_database_size': {
                'conditions': len(self.terminology_db.conditions),
                'procedures': len(self.terminology_db.procedures),
                'medications': len(self.terminology_db.medications),
                'lab_tests': len(self.terminology_db.lab_tests)
            },
            'complexity_distribution': self.complexity_distribution,
            'document_type_distribution': self.document_type_distribution
        }
        
        # Generate queries with realistic complexity distribution
        total_queries = self.config.min_multimodal_queries
        
        for specialty in self.terminology_db.specialties.keys():
            specialty_queries = total_queries // len(self.terminology_db.specialties)
            
            for i in range(specialty_queries):
                # Select complexity level based on realistic distribution
                complexity_level = np.random.choice(
                    list(self.complexity_distribution.keys()),
                    p=list(self.complexity_distribution.values())
                )
                
                # Generate realistic query
                query = self._generate_realistic_query(specialty, complexity_level, i)
                dataset.add_query(query)
                
                # Generate realistic candidates
                candidates = self._generate_realistic_candidates(query)
                for candidate in candidates:
                    dataset.add_candidate(query.id, candidate)
                
                # Generate unbiased relevance judgments
                relevance_judgments = self._generate_unbiased_relevance_judgments(query, candidates)
                for candidate_id, relevance in relevance_judgments.items():
                    dataset.add_relevance_judgment(query.id, candidate_id, relevance)
        
        # Add realistic noise and variations
        dataset = self._add_realistic_noise(dataset)
        
        # Validate dataset quality
        self._validate_dataset_quality(dataset)
        
        logger.info(f"Generated realistic dataset with {len(dataset.queries)} queries")
        
        return dataset
    
    def _generate_realistic_query(self, specialty: str, complexity_level: str, query_idx: int) -> MultimodalMedicalQuery:
        """Generate realistic multimodal medical query."""
        
        # Select realistic medical scenario
        scenario_types = [
            'diagnostic_inquiry', 'treatment_recommendation', 'imaging_interpretation',
            'clinical_correlation', 'emergency_assessment', 'follow_up_care'
        ]
        
        # Weight scenarios by specialty
        if specialty == 'emergency_medicine':
            scenario_weights = [0.3, 0.2, 0.2, 0.1, 0.15, 0.05]
        elif specialty == 'radiology':
            scenario_weights = [0.1, 0.05, 0.6, 0.15, 0.05, 0.05]
        else:
            scenario_weights = [0.25, 0.25, 0.15, 0.15, 0.1, 0.1]
        
        query_type = np.random.choice(scenario_types, p=scenario_weights)
        
        query_id = f"realistic_{specialty}_{query_type}_{complexity_level}_{query_idx:03d}"
        
        # Generate realistic clinical scenario
        clinical_scenario = self._generate_clinical_scenario(specialty, query_type, complexity_level)
        
        query = MultimodalMedicalQuery(
            id=query_id,
            query_type=query_type,
            complexity_level=complexity_level,
            specialty=specialty,
            text=clinical_scenario['text'],
            clinical_data=clinical_scenario.get('clinical_data'),
            image_metadata=clinical_scenario.get('image_metadata'),
            ground_truth_diagnosis=clinical_scenario['diagnosis'],
            medical_urgency=clinical_scenario['urgency'],
            expected_difficulty=self._calculate_query_difficulty(clinical_scenario, complexity_level)
        )
        
        return query
    
    def _generate_clinical_scenario(self, specialty: str, query_type: str, complexity_level: str) -> Dict[str, Any]:
        """Generate realistic clinical scenario."""
        
        specialty_data = self.terminology_db.specialties[specialty]
        
        # Select primary condition
        primary_condition = random.choice(specialty_data['conditions'])
        
        # Generate patient demographics
        age = self._generate_realistic_age(specialty, primary_condition)
        gender = random.choice(['male', 'female'])
        
        # Generate clinical presentation
        presentation = self._generate_clinical_presentation(
            primary_condition, specialty, complexity_level, age, gender
        )
        
        # Generate clinical data if appropriate
        clinical_data = None
        if complexity_level in ['complex', 'very_complex'] or random.random() < 0.6:
            clinical_data = self._generate_realistic_clinical_data(
                primary_condition, specialty, complexity_level
            )
        
        # Generate imaging metadata if appropriate
        image_metadata = None
        if query_type == 'imaging_interpretation' or random.random() < 0.4:
            image_metadata = self._generate_realistic_imaging_metadata(
                primary_condition, specialty
            )
        
        # Determine medical urgency
        urgency = self._determine_medical_urgency(primary_condition, specialty, query_type)
        
        return {
            'text': presentation,
            'clinical_data': clinical_data,
            'image_metadata': image_metadata,
            'diagnosis': primary_condition,
            'urgency': urgency,
            'age': age,
            'gender': gender
        }
    
    def _generate_clinical_presentation(
        self, 
        condition: str, 
        specialty: str, 
        complexity_level: str, 
        age: int, 
        gender: str
    ) -> str:
        """Generate realistic clinical presentation text."""
        
        specialty_data = self.terminology_db.specialties[specialty]
        
        # Base presentation
        presentation = f"{age}-year-old {gender} presenting with "
        
        # Primary symptoms based on condition
        if 'cardiac' in condition.lower() or 'heart' in condition.lower():
            primary_symptoms = ['chest pain', 'shortness of breath', 'palpitations']
        elif 'pneumonia' in condition.lower() or 'respiratory' in condition.lower():
            primary_symptoms = ['cough', 'fever', 'shortness of breath']
        elif 'stroke' in condition.lower() or 'neurologic' in condition.lower():
            primary_symptoms = ['weakness', 'speech difficulty', 'confusion']
        else:
            primary_symptoms = random.sample(self.terminology_db.symptoms, 3)
        
        presentation += f"{primary_symptoms[0]}"
        
        # Add complexity-based details
        if complexity_level in ['moderate', 'complex', 'very_complex']:
            presentation += f" associated with {primary_symptoms[1]}"
            
            # Add onset and character
            onset_options = ['acute', 'subacute', 'chronic', 'progressive']
            character_options = ['severe', 'moderate', 'intermittent', 'constant']
            
            presentation += f". Symptoms are {random.choice(character_options)} and "
            presentation += f"{random.choice(onset_options)} in onset"
        
        if complexity_level in ['complex', 'very_complex']:
            # Add additional symptoms
            if len(primary_symptoms) > 2:
                presentation += f", with additional complaint of {primary_symptoms[2]}"
            
            # Add relevant history
            comorbidities = ['hypertension', 'diabetes', 'hyperlipidemia', 'chronic kidney disease']
            relevant_comorbidity = random.choice(comorbidities)
            presentation += f". Past medical history significant for {relevant_comorbidity}"
            
            # Add medication history
            if specialty_data.get('medications'):
                current_med = random.choice(specialty_data['medications'])
                presentation += f". Current medications include {current_med}"
        
        if complexity_level == 'very_complex':
            # Add social history and complications
            presentation += f". Social history notable for tobacco use"
            presentation += f". Recent hospitalization for related condition"
            
            # Add review of systems
            additional_symptoms = random.sample(self.terminology_db.symptoms, 2)
            presentation += f". Review of systems positive for {additional_symptoms[0]} "
            presentation += f"and {additional_symptoms[1]}"
        
        return presentation
    
    def _generate_realistic_candidates(self, query: MultimodalMedicalQuery) -> List[MultimodalMedicalCandidate]:
        """Generate realistic candidate documents for query."""
        
        candidates = []
        num_candidates = self.config.min_documents_per_query
        
        # Ensure mix of relevant and irrelevant candidates
        num_highly_relevant = max(2, num_candidates // 10)  # 10% highly relevant
        num_moderately_relevant = max(3, num_candidates // 5)  # 20% moderately relevant
        num_somewhat_relevant = max(5, num_candidates // 3)  # 33% somewhat relevant
        num_irrelevant = num_candidates - (num_highly_relevant + num_moderately_relevant + num_somewhat_relevant)
        
        relevance_targets = (
            ['highly_relevant'] * num_highly_relevant +
            ['moderately_relevant'] * num_moderately_relevant +
            ['somewhat_relevant'] * num_somewhat_relevant +
            ['irrelevant'] * num_irrelevant
        )
        
        random.shuffle(relevance_targets)
        
        for i, relevance_target in enumerate(relevance_targets):
            candidate = self._generate_realistic_candidate(query, relevance_target, i)
            candidates.append(candidate)
        
        return candidates
    
    def _generate_realistic_candidate(
        self, 
        query: MultimodalMedicalQuery, 
        relevance_target: str, 
        candidate_idx: int
    ) -> MultimodalMedicalCandidate:
        """Generate realistic candidate document."""
        
        candidate_id = f"{query.id}_candidate_{candidate_idx:03d}"
        
        # Select document type based on distribution
        doc_type = np.random.choice(
            list(self.document_type_distribution.keys()),
            p=list(self.document_type_distribution.values())
        )
        
        # Generate condition based on relevance target
        if relevance_target == 'highly_relevant':
            # Same condition as query
            candidate_condition = query.ground_truth_diagnosis
            candidate_specialty = query.specialty
        elif relevance_target == 'moderately_relevant':
            # Related condition in same specialty
            specialty_data = self.terminology_db.specialties[query.specialty]
            candidate_condition = random.choice(specialty_data['conditions'])
            candidate_specialty = query.specialty
        elif relevance_target == 'somewhat_relevant':
            # Related specialty or condition
            if random.random() < 0.5:
                # Same specialty, different condition
                specialty_data = self.terminology_db.specialties[query.specialty]
                candidate_condition = random.choice(specialty_data['conditions'])
                candidate_specialty = query.specialty
            else:
                # Different specialty, potentially related condition
                candidate_specialty = random.choice(list(self.terminology_db.specialties.keys()))
                specialty_data = self.terminology_db.specialties[candidate_specialty]
                candidate_condition = random.choice(specialty_data['conditions'])
        else:  # irrelevant
            # Unrelated specialty and condition
            candidate_specialty = random.choice(list(self.terminology_db.specialties.keys()))
            candidate_condition = random.choice(self.terminology_db.conditions)
        
        # Generate realistic document
        complexity_level = random.choice(['moderate', 'complex'])  # Candidates should be substantial
        
        if doc_type == 'clinical_guideline':
            real_doc = self.content_generator.generate_clinical_guideline(
                candidate_specialty, candidate_condition, complexity_level
            )
        elif doc_type == 'case_report':
            real_doc = self.content_generator.generate_case_report(
                candidate_specialty, candidate_condition, complexity_level
            )
        elif doc_type == 'research_paper':
            real_doc = self.content_generator.generate_research_paper(
                candidate_specialty, candidate_condition, complexity_level
            )
        else:  # protocol
            real_doc = self.content_generator.generate_clinical_guideline(
                candidate_specialty, candidate_condition, complexity_level
            )
            real_doc.doc_type = 'protocol'
        
        # Convert to candidate format
        candidate = MultimodalMedicalCandidate(
            id=candidate_id,
            content_type=doc_type,
            specialty=candidate_specialty,
            text=real_doc.full_text,
            diagnosis=candidate_condition,
            treatment_recommendations=real_doc.medications[:3],  # First 3 medications
            evidence_level=real_doc.evidence_level,
            clinical_applicability=self._calculate_clinical_applicability(real_doc, relevance_target)
        )
        
        return candidate
    
    def _generate_unbiased_relevance_judgments(
        self, 
        query: MultimodalMedicalQuery, 
        candidates: List[MultimodalMedicalCandidate]
    ) -> Dict[str, float]:
        """Generate unbiased relevance judgments using multiple criteria."""
        
        relevance_judgments = {}
        
        for candidate in candidates:
            # Multi-criteria relevance assessment
            relevance_score = self._calculate_comprehensive_relevance(query, candidate)
            relevance_judgments[candidate.id] = relevance_score
        
        return relevance_judgments
    
    def _calculate_comprehensive_relevance(
        self, 
        query: MultimodalMedicalQuery, 
        candidate: MultimodalMedicalCandidate
    ) -> float:
        """Calculate comprehensive relevance score using multiple criteria."""
        
        relevance_components = {}
        
        # 1. Diagnostic relevance (40% weight)
        if candidate.diagnosis and query.ground_truth_diagnosis:
            if candidate.diagnosis.lower() == query.ground_truth_diagnosis.lower():
                relevance_components['diagnostic'] = 1.0
            elif self._are_related_conditions(candidate.diagnosis, query.ground_truth_diagnosis):
                relevance_components['diagnostic'] = 0.7
            else:
                relevance_components['diagnostic'] = 0.1
        else:
            relevance_components['diagnostic'] = 0.5
        
        # 2. Specialty relevance (25% weight)
        if candidate.specialty == query.specialty:
            relevance_components['specialty'] = 1.0
        elif self._are_related_specialties(candidate.specialty, query.specialty):
            relevance_components['specialty'] = 0.6
        else:
            relevance_components['specialty'] = 0.2
        
        # 3. Content quality and evidence level (20% weight)
        evidence_scores = {'A': 1.0, 'B': 0.8, 'C': 0.6, 'D': 0.4}
        relevance_components['evidence'] = evidence_scores.get(candidate.evidence_level, 0.5)
        
        # 4. Clinical applicability (15% weight)
        relevance_components['applicability'] = candidate.clinical_applicability
        
        # Calculate weighted relevance score
        weights = {
            'diagnostic': 0.40,
            'specialty': 0.25,
            'evidence': 0.20,
            'applicability': 0.15
        }
        
        relevance_score = sum(
            relevance_components[component] * weights[component]
            for component in weights
        )
        
        # Add small random variation to simulate inter-annotator variability
        relevance_score += random.uniform(-0.05, 0.05)
        relevance_score = max(0.0, min(1.0, relevance_score))
        
        return relevance_score
    
    def _validate_dataset_quality(self, dataset: MultimodalMedicalDataset):
        """Validate dataset quality and identify potential biases."""
        
        logger.info("Validating dataset quality and checking for biases...")
        
        # Check query distribution
        specialty_counts = {}
        complexity_counts = {}
        
        for query in dataset.queries:
            specialty_counts[query.specialty] = specialty_counts.get(query.specialty, 0) + 1
            complexity_counts[query.complexity_level] = complexity_counts.get(query.complexity_level, 0) + 1
        
        # Validate balanced distribution
        expected_per_specialty = len(dataset.queries) // len(self.terminology_db.specialties)
        for specialty, count in specialty_counts.items():
            if abs(count - expected_per_specialty) > expected_per_specialty * 0.3:
                logger.warning(f"Specialty distribution imbalance: {specialty} has {count} queries")
        
        # Check relevance distribution
        all_relevances = []
        for judgments in dataset.relevance_judgments.values():
            all_relevances.extend(judgments.values())
        
        if all_relevances:
            avg_relevance = np.mean(all_relevances)
            std_relevance = np.std(all_relevances)
            
            # Should have reasonable distribution
            if avg_relevance > 0.7:
                logger.warning(f"Potentially biased: Average relevance too high ({avg_relevance:.3f})")
            elif avg_relevance < 0.3:
                logger.warning(f"Potentially biased: Average relevance too low ({avg_relevance:.3f})")
            
            if std_relevance < 0.15:
                logger.warning(f"Low relevance variance ({std_relevance:.3f}) - may indicate bias")
        
        logger.info("Dataset quality validation completed")
    
    # Helper methods
    def _generate_realistic_age(self, specialty: str, condition: str) -> int:
        """Generate realistic age based on specialty and condition."""
        if specialty == 'emergency_medicine':
            return random.randint(25, 75)  # Wide range for emergency
        elif 'pediatric' in specialty:
            return random.randint(1, 17)
        elif condition in ['stroke', 'myocardial infarction']:
            return random.randint(45, 85)  # Older for these conditions
        else:
            return random.randint(30, 70)
    
    def _generate_realistic_clinical_data(self, condition: str, specialty: str, complexity_level: str) -> Dict:
        """Generate realistic clinical data."""
        clinical_data = {
            'vital_signs': self._generate_condition_specific_vitals(condition),
            'lab_values': self._generate_condition_specific_labs(condition),
        }
        
        if complexity_level in ['complex', 'very_complex']:
            clinical_data['medications'] = random.sample(
                self.terminology_db.specialties[specialty].get('medications', []), 
                min(3, len(self.terminology_db.specialties[specialty].get('medications', [])))
            )
            clinical_data['allergies'] = random.sample(['penicillin', 'sulfa', 'shellfish'], 
                                                     random.randint(0, 2))
        
        return clinical_data
    
    def _generate_condition_specific_vitals(self, condition: str) -> Dict:
        """Generate condition-specific vital signs."""
        vitals = {
            'temperature': random.uniform(98.0, 102.0),
            'heart_rate': random.randint(60, 120),
            'blood_pressure_systolic': random.randint(90, 180),
            'blood_pressure_diastolic': random.randint(60, 110),
            'respiratory_rate': random.randint(12, 30),
            'oxygen_saturation': random.randint(88, 100)
        }
        
        # Adjust based on condition
        if 'fever' in condition.lower() or 'infection' in condition.lower():
            vitals['temperature'] = random.uniform(100.5, 104.0)
            vitals['heart_rate'] = random.randint(90, 130)
        
        if 'cardiac' in condition.lower():
            vitals['heart_rate'] = random.randint(45, 150)
            vitals['blood_pressure_systolic'] = random.randint(80, 200)
        
        return vitals
    
    def _generate_condition_specific_labs(self, condition: str) -> Dict:
        """Generate condition-specific laboratory values."""
        labs = {
            'white_blood_count': random.uniform(4.0, 15.0),
            'hemoglobin': random.uniform(10.0, 16.0),
            'creatinine': random.uniform(0.8, 2.0),
            'glucose': random.randint(80, 200)
        }
        
        # Adjust based on condition
        if 'infection' in condition.lower():
            labs['white_blood_count'] = random.uniform(12.0, 25.0)
        
        if 'cardiac' in condition.lower():
            labs['troponin'] = random.uniform(0.0, 5.0)
            labs['bnp'] = random.randint(100, 2000)
        
        return labs
    
    def _generate_realistic_imaging_metadata(self, condition: str, specialty: str) -> Dict:
        """Generate realistic imaging metadata."""
        specialty_data = self.terminology_db.specialties[specialty]
        imaging_studies = specialty_data.get('imaging', ['chest X-ray'])
        
        selected_study = random.choice(imaging_studies)
        
        metadata = {
            'study_type': selected_study,
            'indication': condition,
            'technique': 'standard protocol',
            'contrast': random.choice([True, False]) if 'CT' in selected_study or 'MRI' in selected_study else False
        }
        
        # Add findings based on condition
        if specialty in self.terminology_db.imaging_findings:
            findings = random.sample(
                self.terminology_db.imaging_findings[specialty.replace('_', '')], 
                random.randint(1, 3)
            )
            metadata['findings'] = findings
        
        return metadata
    
    def _determine_medical_urgency(self, condition: str, specialty: str, query_type: str) -> str:
        """Determine medical urgency based on condition and context."""
        if specialty == 'emergency_medicine' or query_type == 'emergency_assessment':
            return random.choice(['urgent', 'emergency'])
        
        emergency_conditions = ['myocardial infarction', 'stroke', 'sepsis', 'respiratory failure']
        if any(emergency_condition in condition.lower() for emergency_condition in emergency_conditions):
            return 'emergency'
        
        urgent_conditions = ['pneumonia', 'heart failure', 'arrhythmia']
        if any(urgent_condition in condition.lower() for urgent_condition in urgent_conditions):
            return 'urgent'
        
        return 'routine'
    
    def _calculate_query_difficulty(self, scenario: Dict, complexity_level: str) -> float:
        """Calculate query difficulty score."""
        base_difficulty = {'simple': 0.2, 'moderate': 0.5, 'complex': 0.7, 'very_complex': 0.9}
        
        difficulty = base_difficulty[complexity_level]
        
        # Adjust based on urgency
        if scenario['urgency'] == 'emergency':
            difficulty += 0.1
        
        # Adjust based on age extremes
        if scenario['age'] < 18 or scenario['age'] > 75:
            difficulty += 0.05
        
        return min(1.0, difficulty)
    
    def _calculate_clinical_applicability(self, document: RealMedicalDocument, relevance_target: str) -> float:
        """Calculate clinical applicability score."""
        base_scores = {
            'highly_relevant': 0.9,
            'moderately_relevant': 0.7,
            'somewhat_relevant': 0.5,
            'irrelevant': 0.2
        }
        
        base_score = base_scores.get(relevance_target, 0.5)
        
        # Adjust based on document quality
        if document.evidence_level == 'A':
            base_score += 0.05
        elif document.evidence_level == 'D':
            base_score -= 0.1
        
        # Adjust based on recency
        if document.publication_year >= 2022:
            base_score += 0.05
        elif document.publication_year < 2020:
            base_score -= 0.05
        
        return max(0.0, min(1.0, base_score))
    
    def _are_related_conditions(self, condition1: str, condition2: str) -> bool:
        """Check if two conditions are clinically related."""
        # Simple heuristic - in practice would use medical ontology
        condition1_lower = condition1.lower()
        condition2_lower = condition2.lower()
        
        # Check for shared keywords
        keywords1 = set(condition1_lower.split())
        keywords2 = set(condition2_lower.split())
        
        common_keywords = keywords1.intersection(keywords2)
        
        if len(common_keywords) > 0:
            return True
        
        # Check for related categories
        cardiac_conditions = ['cardiac', 'heart', 'coronary', 'myocardial']
        respiratory_conditions = ['pulmonary', 'lung', 'respiratory', 'pneumonia']
        
        if any(term in condition1_lower for term in cardiac_conditions) and \
           any(term in condition2_lower for term in cardiac_conditions):
            return True
        
        if any(term in condition1_lower for term in respiratory_conditions) and \
           any(term in condition2_lower for term in respiratory_conditions):
            return True
        
        return False
    
    def _are_related_specialties(self, specialty1: str, specialty2: str) -> bool:
        """Check if two specialties are related."""
        related_specialty_groups = [
            ['cardiology', 'emergency_medicine'],
            ['pulmonology', 'emergency_medicine'],
            ['neurology', 'emergency_medicine'],
            ['radiology', 'emergency_medicine'],
            ['radiology', 'cardiology'],
            ['radiology', 'pulmonology'],
            ['radiology', 'neurology']
        ]
        
        for group in related_specialty_groups:
            if specialty1 in group and specialty2 in group:
                return True
        
        return False
    
    def _add_realistic_noise(self, dataset: MultimodalMedicalDataset) -> MultimodalMedicalDataset:
        """Add realistic noise and variations to simulate real-world conditions."""
        
        # Add OCR errors to some text documents
        for query in dataset.queries:
            if query.text and random.random() < 0.1:  # 10% chance
                query.text = self._add_realistic_ocr_errors(query.text)
        
        # Add missing clinical data
        for query in dataset.queries:
            if query.clinical_data and random.random() < 0.15:  # 15% chance
                query.clinical_data = self._add_missing_clinical_data(query.clinical_data)
        
        # Add measurement noise to lab values
        for query in dataset.queries:
            if query.clinical_data and 'lab_values' in query.clinical_data:
                query.clinical_data['lab_values'] = self._add_measurement_noise(
                    query.clinical_data['lab_values']
                )
        
        return dataset
    
    def _add_realistic_ocr_errors(self, text: str) -> str:
        """Add realistic OCR errors to text."""
        # Common OCR substitutions in medical documents
        ocr_substitutions = {
            'ml': 'ml',  # milliliter variants
            'mg': 'rng',  # milligram variants
            'cm': 'crn',  # centimeter variants
            'patient': 'patjent',
            'diagnosis': 'djagnosis',
            'treatment': 'treatrnent',
            'medication': 'rnedication'
        }
        
        words = text.split()
        for i, word in enumerate(words):
            if random.random() < 0.02:  # 2% chance per word
                for correct, error in ocr_substitutions.items():
                    if correct in word.lower():
                        words[i] = word.replace(correct, error)
                        break
        
        return ' '.join(words)
    
    def _add_missing_clinical_data(self, clinical_data: Dict) -> Dict:
        """Add missing clinical data to simulate incomplete records."""
        modified_data = clinical_data.copy()
        
        # Randomly remove some lab values
        if 'lab_values' in modified_data and len(modified_data['lab_values']) > 2:
            labs_to_remove = random.sample(
                list(modified_data['lab_values'].keys()),
                random.randint(1, min(2, len(modified_data['lab_values']) - 1))
            )
            for lab in labs_to_remove:
                del modified_data['lab_values'][lab]
        
        return modified_data
    
    def _add_measurement_noise(self, lab_values: Dict) -> Dict:
        """Add realistic measurement noise to lab values."""
        noisy_values = lab_values.copy()
        
        for lab, value in lab_values.items():
            if isinstance(value, (int, float)):
                # Add 1-5% measurement noise
                noise_factor = random.uniform(0.95, 1.05)
                noisy_values[lab] = round(value * noise_factor, 2)
        
        return noisy_values


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test realistic dataset generation
    config = MultimodalMedicalEvaluationConfig(
        min_multimodal_queries=25,  # Smaller for testing
        min_documents_per_query=20
    )
    
    generator = RealisticMedicalDatasetGenerator(config)
    dataset = generator.generate_unbiased_dataset()
    
    print("Realistic Dataset Generation Results:")
    info = dataset.get_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Analyze first query
    if dataset.queries:
        first_query = dataset.queries[0]
        print(f"\nSample Query ({first_query.complexity_level}):")
        print(f"  Specialty: {first_query.specialty}")
        print(f"  Type: {first_query.query_type}")
        print(f"  Text length: {len(first_query.text.split())} words")
        print(f"  Diagnosis: {first_query.ground_truth_diagnosis}")
        
        candidates = dataset.get_candidates(first_query.id)
        relevance_judgments = dataset.relevance_judgments.get(first_query.id, {})
        
        print(f"\nCandidates: {len(candidates)}")
        if relevance_judgments:
            relevances = list(relevance_judgments.values())
            print(f"  Relevance range: {min(relevances):.3f} - {max(relevances):.3f}")
            print(f"  Average relevance: {np.mean(relevances):.3f}")
    
    print("\nDataset validation completed - ready for unbiased evaluation")
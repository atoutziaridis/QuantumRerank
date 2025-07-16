"""
Medical Domain Relevance Judgment System for Quantum Reranking Evaluation.

This module creates relevance judgments for medical queries and documents,
using domain knowledge, keyword matching, and semantic similarity.

Based on:
- Medical domain ontologies and classification
- MeSH (Medical Subject Headings) terminology
- Clinical query classification patterns
"""

import re
import logging
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .ir_metrics import RelevanceJudgment
from ..core.embeddings import EmbeddingProcessor

logger = logging.getLogger(__name__)


@dataclass
class MedicalQuery:
    """Medical query with domain classification."""
    query_id: str
    query_text: str
    medical_domain: str
    query_type: str  # diagnostic, treatment, symptom, general
    key_terms: List[str]
    complexity_level: str  # simple, moderate, complex


@dataclass 
class MedicalDocument:
    """Medical document with extracted features."""
    doc_id: str
    title: str
    abstract: str
    full_text: str
    medical_domain: str
    key_terms: List[str]
    sections: Dict[str, str]


class MedicalRelevanceJudgments:
    """
    Medical domain relevance judgment system.
    
    Creates automated relevance judgments based on:
    1. Domain matching (cardiology, diabetes, etc.)
    2. Medical keyword matching
    3. Semantic similarity
    4. Query-document type matching
    """
    
    def __init__(self, embedding_processor: Optional[EmbeddingProcessor] = None):
        """Initialize medical relevance system."""
        self.embedding_processor = embedding_processor or EmbeddingProcessor()
        
        # Medical domain keywords (expanded from PMC parser)
        self.domain_keywords = {
            'cardiology': {
                'primary': [
                    'cardiac', 'heart', 'myocardial', 'coronary', 'cardiovascular',
                    'arrhythmia', 'infarction', 'hypertension', 'atherosclerosis',
                    'angina', 'stenosis', 'valve', 'aortic', 'mitral'
                ],
                'secondary': [
                    'blood pressure', 'chest pain', 'ECG', 'EKG', 'echocardiogram',
                    'cardiac catheterization', 'angioplasty', 'bypass', 'stent'
                ]
            },
            'diabetes': {
                'primary': [
                    'diabetes', 'insulin', 'glucose', 'glycemic', 'diabetic',
                    'hyperglycemia', 'hypoglycemia', 'endocrine', 'metabolic',
                    'pancreatic', 'HbA1c', 'blood sugar'
                ],
                'secondary': [
                    'metformin', 'glucagon', 'ketoacidosis', 'neuropathy',
                    'retinopathy', 'nephropathy', 'type 1', 'type 2'
                ]
            },
            'respiratory': {
                'primary': [
                    'lung', 'pulmonary', 'respiratory', 'pneumonia', 'asthma',
                    'COPD', 'bronchial', 'ventilation', 'breathing', 'airway'
                ],
                'secondary': [
                    'spirometry', 'oxygen', 'inhaler', 'bronchodilator',
                    'emphysema', 'bronchitis', 'tuberculosis', 'pleural'
                ]
            },
            'neurology': {
                'primary': [
                    'brain', 'neural', 'neurological', 'stroke', 'seizure',
                    'alzheimer', 'parkinson', 'epilepsy', 'cognitive', 'dementia'
                ],
                'secondary': [
                    'EEG', 'MRI', 'CT scan', 'neurotransmitter', 'dopamine',
                    'serotonin', 'migraine', 'headache', 'memory loss'
                ]
            },
            'oncology': {
                'primary': [
                    'cancer', 'tumor', 'malignant', 'oncology', 'chemotherapy',
                    'radiation', 'metastasis', 'carcinoma', 'lymphoma', 'leukemia'
                ],
                'secondary': [
                    'biopsy', 'staging', 'grade', 'prognosis', 'survival',
                    'remission', 'relapse', 'immunotherapy', 'targeted therapy'
                ]
            }
        }
        
        # Query type patterns
        self.query_type_patterns = {
            'diagnostic': [
                r'diagnos\w+', r'test\w+', r'screen\w+', r'detect\w+', 
                r'identif\w+', r'what is', r'how to diagnose'
            ],
            'treatment': [
                r'treat\w+', r'therap\w+', r'manag\w+', r'cure\w+',
                r'medicin\w+', r'drug\w+', r'how to treat'
            ],
            'symptom': [
                r'symptom\w+', r'sign\w+', r'present\w+', r'manifest\w+',
                r'what are the symptoms', r'how does.*present'
            ],
            'prognosis': [
                r'prognos\w+', r'outcome\w+', r'survival\w+', r'recover\w+',
                r'what is the prognosis', r'long term'
            ]
        }
        
        # Medical abbreviations
        self.medical_abbreviations = {
            'MI': 'myocardial infarction',
            'HTN': 'hypertension', 
            'DM': 'diabetes mellitus',
            'COPD': 'chronic obstructive pulmonary disease',
            'CHF': 'congestive heart failure',
            'CAD': 'coronary artery disease',
            'CVA': 'cerebrovascular accident',
            'URI': 'upper respiratory infection',
            'UTI': 'urinary tract infection',
            'DVT': 'deep vein thrombosis'
        }
        
        logger.info("Medical relevance judgment system initialized")
    
    def expand_medical_abbreviations(self, text: str) -> str:
        """Expand medical abbreviations in text."""
        expanded_text = text
        for abbrev, expansion in self.medical_abbreviations.items():
            # Replace abbreviation with both abbreviation and expansion
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            replacement = f"{abbrev} ({expansion})"
            expanded_text = re.sub(pattern, replacement, expanded_text, flags=re.IGNORECASE)
        
        return expanded_text
    
    def classify_medical_domain(self, text: str) -> Tuple[str, float]:
        """
        Classify text into medical domain with confidence score.
        
        Args:
            text: Text to classify
            
        Returns:
            Tuple of (domain, confidence_score)
        """
        # Expand abbreviations first
        expanded_text = self.expand_medical_abbreviations(text.lower())
        
        domain_scores = {}
        
        for domain, keyword_sets in self.domain_keywords.items():
            score = 0
            
            # Primary keywords (higher weight)
            for keyword in keyword_sets['primary']:
                if keyword in expanded_text:
                    score += 2
            
            # Secondary keywords (lower weight)
            for keyword in keyword_sets['secondary']:
                if keyword in expanded_text:
                    score += 1
            
            domain_scores[domain] = score
        
        if not domain_scores or max(domain_scores.values()) == 0:
            return 'general', 0.0
        
        best_domain = max(domain_scores, key=domain_scores.get)
        max_score = domain_scores[best_domain]
        
        # Normalize confidence score
        total_keywords = len(self.domain_keywords[best_domain]['primary']) * 2 + \
                        len(self.domain_keywords[best_domain]['secondary'])
        confidence = min(1.0, max_score / total_keywords)
        
        return best_domain, confidence
    
    def classify_query_type(self, query_text: str) -> str:
        """Classify query type (diagnostic, treatment, etc.)."""
        query_lower = query_text.lower()
        
        for query_type, patterns in self.query_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return query_type
        
        return 'general'
    
    def extract_key_terms(self, text: str, min_length: int = 3) -> List[str]:
        """Extract key medical terms from text."""
        # Expand abbreviations
        expanded_text = self.expand_medical_abbreviations(text)
        
        # Extract medical terms using patterns
        medical_patterns = [
            r'\b[A-Z]{2,}\b',  # Abbreviations
            r'\b\w*ology\b',   # -ology terms
            r'\b\w*itis\b',    # -itis terms  
            r'\b\w*osis\b',    # -osis terms
            r'\b\w*pathy\b',   # -pathy terms
            r'\b\w*therapy\b', # -therapy terms
        ]
        
        key_terms = set()
        
        # Extract pattern-based terms
        for pattern in medical_patterns:
            matches = re.findall(pattern, expanded_text, re.IGNORECASE)
            key_terms.update(term.lower() for term in matches if len(term) >= min_length)
        
        # Extract domain-specific terms
        for domain_keywords in self.domain_keywords.values():
            for keyword_set in domain_keywords.values():
                for keyword in keyword_set:
                    if keyword in expanded_text.lower():
                        key_terms.add(keyword)
        
        return list(key_terms)
    
    def assess_query_complexity(self, query_text: str) -> str:
        """Assess query complexity level."""
        # Count complexity indicators
        complexity_indicators = {
            'multiple_domains': 0,
            'multiple_conditions': 0, 
            'complex_terms': 0,
            'long_query': 0
        }
        
        # Check for multiple domains
        domain_matches = 0
        for domain_keywords in self.domain_keywords.values():
            for keyword_set in domain_keywords.values():
                if any(keyword in query_text.lower() for keyword in keyword_set):
                    domain_matches += 1
                    break
        
        if domain_matches > 1:
            complexity_indicators['multiple_domains'] = 1
        
        # Check for multiple conditions (using "and", "with", "plus")
        conjunction_words = ['and', 'with', 'plus', 'combined with', 'along with']
        if any(word in query_text.lower() for word in conjunction_words):
            complexity_indicators['multiple_conditions'] = 1
        
        # Check for complex medical terms (long terms, Latin terms)
        words = query_text.split()
        complex_words = [word for word in words if len(word) > 10]
        if len(complex_words) > 2:
            complexity_indicators['complex_terms'] = 1
        
        # Check query length
        if len(words) > 15:
            complexity_indicators['long_query'] = 1
        
        # Determine complexity level
        complexity_score = sum(complexity_indicators.values())
        
        if complexity_score >= 3:
            return 'complex'
        elif complexity_score >= 1:
            return 'moderate'
        else:
            return 'simple'
    
    def create_medical_query(self, query_id: str, query_text: str) -> MedicalQuery:
        """Create medical query with classification."""
        domain, _ = self.classify_medical_domain(query_text)
        query_type = self.classify_query_type(query_text)
        key_terms = self.extract_key_terms(query_text)
        complexity = self.assess_query_complexity(query_text)
        
        return MedicalQuery(
            query_id=query_id,
            query_text=query_text,
            medical_domain=domain,
            query_type=query_type,
            key_terms=key_terms,
            complexity_level=complexity
        )
    
    def create_medical_document(self, doc_id: str, title: str, 
                              abstract: str, full_text: str,
                              sections: Optional[Dict[str, str]] = None) -> MedicalDocument:
        """Create medical document with classification."""
        # Combine text for classification
        combined_text = f"{title} {abstract}"
        
        domain, _ = self.classify_medical_domain(combined_text)
        key_terms = self.extract_key_terms(combined_text)
        
        return MedicalDocument(
            doc_id=doc_id,
            title=title,
            abstract=abstract,
            full_text=full_text,
            medical_domain=domain,
            key_terms=key_terms,
            sections=sections or {}
        )
    
    def calculate_domain_relevance(self, query: MedicalQuery, 
                                 document: MedicalDocument) -> float:
        """Calculate relevance based on medical domain matching."""
        if query.medical_domain == document.medical_domain:
            return 1.0
        elif query.medical_domain == 'general' or document.medical_domain == 'general':
            return 0.5  # Partial relevance for general content
        else:
            return 0.0
    
    def calculate_keyword_relevance(self, query: MedicalQuery,
                                  document: MedicalDocument) -> float:
        """Calculate relevance based on keyword overlap."""
        if not query.key_terms or not document.key_terms:
            return 0.0
        
        # Calculate Jaccard similarity
        query_terms = set(query.key_terms)
        doc_terms = set(document.key_terms)
        
        intersection = len(query_terms.intersection(doc_terms))
        union = len(query_terms.union(doc_terms))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_semantic_relevance(self, query: MedicalQuery,
                                   document: MedicalDocument) -> float:
        """Calculate relevance based on semantic similarity."""
        try:
            # Use title and abstract for semantic comparison
            doc_text = f"{document.title} {document.abstract}"
            
            # Generate embeddings
            query_embedding = self.embedding_processor.encode_single_text(query.query_text)
            doc_embedding = self.embedding_processor.encode_single_text(doc_text)
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception as e:
            logger.warning(f"Error calculating semantic relevance: {e}")
            return 0.0
    
    def calculate_relevance_score(self, query: MedicalQuery,
                                document: MedicalDocument,
                                weights: Optional[Dict[str, float]] = None) -> Tuple[int, float]:
        """
        Calculate overall relevance score.
        
        Args:
            query: Medical query
            document: Medical document
            weights: Weights for different relevance factors
            
        Returns:
            Tuple of (relevance_level, confidence)
        """
        if weights is None:
            weights = {'domain': 0.4, 'keyword': 0.3, 'semantic': 0.3}
        
        # Calculate individual relevance scores
        domain_relevance = self.calculate_domain_relevance(query, document)
        keyword_relevance = self.calculate_keyword_relevance(query, document)
        semantic_relevance = self.calculate_semantic_relevance(query, document)
        
        # Weighted average
        overall_score = (
            weights['domain'] * domain_relevance +
            weights['keyword'] * keyword_relevance +
            weights['semantic'] * semantic_relevance
        )
        
        # Convert to relevance levels
        if overall_score >= 0.7:
            relevance_level = 2  # Highly relevant
        elif overall_score >= 0.4:
            relevance_level = 1  # Relevant
        else:
            relevance_level = 0  # Not relevant
        
        # Confidence based on agreement between methods
        component_scores = [domain_relevance, keyword_relevance, semantic_relevance]
        confidence = 1.0 - np.std(component_scores)  # Higher std = lower confidence
        confidence = max(0.0, min(1.0, confidence))
        
        return relevance_level, confidence
    
    def create_relevance_judgments(self, queries: List[MedicalQuery],
                                 documents: List[MedicalDocument]) -> List[RelevanceJudgment]:
        """
        Create relevance judgments for all query-document pairs.
        
        Args:
            queries: List of medical queries
            documents: List of medical documents
            
        Returns:
            List of relevance judgments
        """
        judgments = []
        
        logger.info(f"Creating relevance judgments for {len(queries)} queries "
                   f"and {len(documents)} documents")
        
        for i, query in enumerate(queries):
            logger.debug(f"Processing query {i+1}/{len(queries)}: {query.query_text[:50]}...")
            
            for document in documents:
                relevance_level, confidence = self.calculate_relevance_score(query, document)
                
                judgment = RelevanceJudgment(
                    query_id=query.query_id,
                    doc_id=document.doc_id,
                    relevance=relevance_level,
                    confidence=confidence
                )
                
                judgments.append(judgment)
        
        # Statistics
        relevance_counts = {0: 0, 1: 0, 2: 0}
        for judgment in judgments:
            relevance_counts[judgment.relevance] += 1
        
        total = len(judgments)
        logger.info(f"Created {total} relevance judgments:")
        logger.info(f"  Not relevant (0): {relevance_counts[0]} ({relevance_counts[0]/total*100:.1f}%)")
        logger.info(f"  Relevant (1): {relevance_counts[1]} ({relevance_counts[1]/total*100:.1f}%)")
        logger.info(f"  Highly relevant (2): {relevance_counts[2]} ({relevance_counts[2]/total*100:.1f}%)")
        
        return judgments


def create_medical_test_queries() -> List[MedicalQuery]:
    """Create a set of test medical queries for evaluation."""
    test_query_texts = [
        # Cardiology queries
        ("How to diagnose myocardial infarction?", "cardiology_diagnostic_001"),
        ("Treatment options for coronary artery disease", "cardiology_treatment_001"),
        ("Symptoms of heart failure", "cardiology_symptom_001"),
        
        # Diabetes queries
        ("Management of type 2 diabetes", "diabetes_treatment_001"),
        ("Blood glucose monitoring techniques", "diabetes_diagnostic_001"),
        ("Diabetic complications and prevention", "diabetes_prognosis_001"),
        
        # Respiratory queries
        ("COPD exacerbation treatment", "respiratory_treatment_001"),
        ("Asthma diagnosis in children", "respiratory_diagnostic_001"),
        ("Pneumonia symptoms and signs", "respiratory_symptom_001"),
        
        # Complex multi-domain queries
        ("Diabetes with cardiovascular complications management", "complex_multi_001"),
        ("Heart failure and COPD comorbidity treatment", "complex_multi_002"),
        
        # General medical queries
        ("Infection control in hospitals", "general_001"),
        ("Patient safety protocols", "general_002")
    ]
    
    relevance_system = MedicalRelevanceJudgments()
    queries = []
    
    for query_text, query_id in test_query_texts:
        medical_query = relevance_system.create_medical_query(query_id, query_text)
        queries.append(medical_query)
    
    return queries
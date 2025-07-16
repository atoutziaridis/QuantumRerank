"""
Medical Corpus Quantum Kernel Trainer.

This module implements comprehensive training of quantum kernels specifically
on PMC medical corpus data, optimizing for medical document ranking and 
similarity computation with domain-specific adaptations.

Based on QRF-05 requirements for training quantum kernels on medical corpus.
"""

import logging
import time
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
import random

from .medical_data_preparation import TrainingPair, MedicalTrainingConfig
from .quantum_kernel_trainer import QuantumKernelTrainer, KTAOptimizationConfig
from ..evaluation.medical_relevance import MedicalDocument, MedicalQuery, MedicalRelevanceJudgments
from ..core.embeddings import EmbeddingProcessor
from ..core.quantum_kernel_engine import QuantumKernelEngine
from ..config.settings import QuantumConfig

logger = logging.getLogger(__name__)

# Import PMC parser - ensure it's available globally for pickle
import sys
import os
sys.path.append('.')
sys.path.append(os.getcwd())

# Import at module level for pickle compatibility
try:
    from pmc_xml_parser import PMCArticle, PMCXMLParser
    logger.info("PMC XML parser imported successfully")
except ImportError as e:
    logger.warning(f"PMC XML parser not available: {e}")
    PMCArticle = None
    PMCXMLParser = None


@dataclass
class MedicalCorpusConfig:
    """Configuration for medical corpus training."""
    pmc_data_path: str = "parsed_pmc_articles.pkl"
    target_training_pairs: int = 5000
    domain_balance_strategy: str = "weighted"  # weighted, equal, proportional
    min_pairs_per_domain: int = 200
    cross_domain_pairs_ratio: float = 0.3
    hierarchical_pairs_ratio: float = 0.2
    validation_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 42


@dataclass
class MedicalCorpusTrainingResult:
    """Results from medical corpus training."""
    corpus_analysis: Dict[str, Any]
    training_pairs_generated: int
    domain_distribution: Dict[str, int]
    kta_training_result: Any  # QuantumKernelTrainingResult
    domain_specific_results: Dict[str, Dict[str, float]]
    cross_domain_performance: Dict[str, float]
    training_time_seconds: float
    performance_improvements: Dict[str, float]


class MedicalCorpusAnalyzer:
    """
    Analyzes PMC medical corpus and creates balanced training datasets.
    
    Handles the skewed domain distribution (76% general, 10% oncology, 9% neurology,
    4% diabetes, 1% respiratory) and creates balanced training pairs.
    """
    
    def __init__(self, config: Optional[MedicalCorpusConfig] = None):
        """Initialize medical corpus analyzer."""
        self.config = config or MedicalCorpusConfig()
        
        # Initialize components
        self.embedding_processor = EmbeddingProcessor()
        self.medical_relevance = MedicalRelevanceJudgments(self.embedding_processor)
        
        # Corpus data
        self.pmc_articles: List[PMCArticle] = []
        self.medical_documents: List[MedicalDocument] = []
        self.corpus_statistics: Dict[str, Any] = {}
        
        # Set random seed
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        
        logger.info("Medical corpus analyzer initialized")
    
    def load_pmc_corpus(self) -> List[PMCArticle]:
        """Load PMC corpus from parsed data."""
        logger.info(f"Loading PMC corpus from {self.config.pmc_data_path}")
        
        try:
            with open(self.config.pmc_data_path, 'rb') as f:
                self.pmc_articles = pickle.load(f)
            
            logger.info(f"Loaded {len(self.pmc_articles)} PMC articles")
            
            # Analyze corpus
            self.corpus_statistics = self._analyze_corpus_distribution()
            self._log_corpus_analysis()
            
            return self.pmc_articles
            
        except Exception as e:
            logger.error(f"Failed to load PMC corpus: {e}")
            raise
    
    def _analyze_corpus_distribution(self) -> Dict[str, Any]:
        """Analyze domain distribution and corpus characteristics."""
        domain_counts = Counter(article.medical_domain for article in self.pmc_articles)
        
        # Calculate domain statistics
        total_articles = len(self.pmc_articles)
        domain_percentages = {
            domain: count / total_articles * 100 
            for domain, count in domain_counts.items()
        }
        
        # Analyze article characteristics
        avg_title_length = np.mean([len(article.title.split()) for article in self.pmc_articles])
        avg_abstract_length = np.mean([len(article.abstract.split()) for article in self.pmc_articles])
        avg_fulltext_length = np.mean([len(article.full_text.split()) for article in self.pmc_articles])
        
        # Keywords analysis
        all_keywords = []
        for article in self.pmc_articles:
            all_keywords.extend(article.keywords)
        keyword_counts = Counter(all_keywords)
        
        return {
            'total_articles': total_articles,
            'domain_counts': dict(domain_counts),
            'domain_percentages': domain_percentages,
            'avg_title_length': avg_title_length,
            'avg_abstract_length': avg_abstract_length,
            'avg_fulltext_length': avg_fulltext_length,
            'most_common_keywords': dict(keyword_counts.most_common(20)),
            'unique_keywords': len(keyword_counts)
        }
    
    def _log_corpus_analysis(self):
        """Log corpus analysis results."""
        stats = self.corpus_statistics
        
        logger.info("PMC Corpus Analysis:")
        logger.info(f"  Total articles: {stats['total_articles']}")
        logger.info(f"  Domain distribution:")
        for domain, count in sorted(stats['domain_counts'].items(), key=lambda x: x[1], reverse=True):
            percentage = stats['domain_percentages'][domain]
            logger.info(f"    {domain}: {count} articles ({percentage:.1f}%)")
        
        logger.info(f"  Average lengths:")
        logger.info(f"    Title: {stats['avg_title_length']:.1f} words")
        logger.info(f"    Abstract: {stats['avg_abstract_length']:.1f} words")
        logger.info(f"    Full text: {stats['avg_fulltext_length']:.1f} words")
        
        logger.info(f"  Keywords: {stats['unique_keywords']} unique terms")
    
    def convert_to_medical_documents(self) -> List[MedicalDocument]:
        """Convert PMC articles to medical documents."""
        logger.info("Converting PMC articles to medical documents")
        
        self.medical_documents = []
        
        for article in self.pmc_articles:
            # Create medical document
            doc = MedicalDocument(
                doc_id=article.pmc_id,
                title=article.title,
                abstract=article.abstract,
                full_text=article.full_text,
                medical_domain=article.medical_domain,
                key_terms=article.keywords[:10],  # Use top 10 keywords
                sections=article.sections
            )
            self.medical_documents.append(doc)
        
        logger.info(f"Converted {len(self.medical_documents)} medical documents")
        return self.medical_documents
    
    def create_balanced_training_pairs(self) -> List[TrainingPair]:
        """Create balanced training pairs from medical corpus."""
        logger.info("Creating balanced training pairs from medical corpus")
        
        if not self.medical_documents:
            self.convert_to_medical_documents()
        
        # Generate different types of training pairs
        intra_domain_pairs = self._create_intra_domain_pairs()
        cross_domain_pairs = self._create_cross_domain_pairs()
        hierarchical_pairs = self._create_hierarchical_pairs()
        
        # Combine all pairs
        all_pairs = intra_domain_pairs + cross_domain_pairs + hierarchical_pairs
        
        # Balance and sample to target
        balanced_pairs = self._balance_training_pairs(all_pairs)
        
        # Generate embeddings
        final_pairs = self._generate_embeddings_for_pairs(balanced_pairs)
        
        logger.info(f"Created {len(final_pairs)} balanced training pairs")
        return final_pairs
    
    def _create_intra_domain_pairs(self) -> List[TrainingPair]:
        """Create training pairs within the same medical domain."""
        logger.info("Creating intra-domain training pairs")
        
        # Group documents by domain
        domain_docs = defaultdict(list)
        for doc in self.medical_documents:
            domain_docs[doc.medical_domain].append(doc)
        
        pairs = []
        
        for domain, docs in domain_docs.items():
            if len(docs) < 2:
                continue
            
            # Generate queries from domain documents
            domain_queries = self._generate_domain_queries(domain, docs)
            
            # Create positive pairs (same domain)
            for query in domain_queries:
                relevant_docs = self._find_relevant_docs_in_domain(query, docs)
                
                for doc in relevant_docs[:3]:  # Top 3 relevant docs per query
                    pair = TrainingPair(
                        query_id=query.query_id,
                        doc_id=doc.doc_id,
                        query_text=query.query_text,
                        doc_text=f"{doc.title} {doc.abstract}",
                        query_embedding=np.array([]),  # Will be filled later
                        doc_embedding=np.array([]),    # Will be filled later
                        relevance_label=2,  # Highly relevant (same domain)
                        medical_domain=domain,
                        pair_type='intra_domain_positive'
                    )
                    pairs.append(pair)
            
            # Create negative pairs within domain (different specific topics)
            for query in domain_queries:
                irrelevant_docs = self._find_irrelevant_docs_in_domain(query, docs)
                
                for doc in irrelevant_docs[:2]:  # Top 2 irrelevant docs per query
                    pair = TrainingPair(
                        query_id=query.query_id,
                        doc_id=doc.doc_id,
                        query_text=query.query_text,
                        doc_text=f"{doc.title} {doc.abstract}",
                        query_embedding=np.array([]),
                        doc_embedding=np.array([]),
                        relevance_label=0,  # Not relevant
                        medical_domain=domain,
                        pair_type='intra_domain_negative'
                    )
                    pairs.append(pair)
        
        logger.info(f"Created {len(pairs)} intra-domain pairs")
        return pairs
    
    def _create_cross_domain_pairs(self) -> List[TrainingPair]:
        """Create training pairs across different medical domains."""
        logger.info("Creating cross-domain training pairs")
        
        pairs = []
        domains = list(set(doc.medical_domain for doc in self.medical_documents))
        
        for i, domain1 in enumerate(domains):
            domain1_docs = [doc for doc in self.medical_documents if doc.medical_domain == domain1]
            domain1_queries = self._generate_domain_queries(domain1, domain1_docs[:5])
            
            for j, domain2 in enumerate(domains[i+1:], i+1):
                domain2_docs = [doc for doc in self.medical_documents if doc.medical_domain == domain2]
                
                # Create cross-domain pairs (typically negative)
                for query in domain1_queries[:3]:  # Limit for efficiency
                    for doc in domain2_docs[:2]:
                        # Check if there's some cross-domain relevance
                        relevance = self._assess_cross_domain_relevance(query, doc)
                        
                        pair = TrainingPair(
                            query_id=query.query_id,
                            doc_id=doc.doc_id,
                            query_text=query.query_text,
                            doc_text=f"{doc.title} {doc.abstract}",
                            query_embedding=np.array([]),
                            doc_embedding=np.array([]),
                            relevance_label=relevance,
                            medical_domain=f"{domain1}_to_{domain2}",
                            pair_type='cross_domain'
                        )
                        pairs.append(pair)
        
        logger.info(f"Created {len(pairs)} cross-domain pairs")
        return pairs
    
    def _create_hierarchical_pairs(self) -> List[TrainingPair]:
        """Create hierarchical pairs (general medical vs specific domains)."""
        logger.info("Creating hierarchical training pairs")
        
        pairs = []
        
        # Get general medical documents
        general_docs = [doc for doc in self.medical_documents if doc.medical_domain == 'general']
        specific_docs = [doc for doc in self.medical_documents if doc.medical_domain != 'general']
        
        # Create queries from specific domains
        specific_domains = list(set(doc.medical_domain for doc in specific_docs))
        
        for domain in specific_domains:
            domain_docs = [doc for doc in specific_docs if doc.medical_domain == domain]
            domain_queries = self._generate_domain_queries(domain, domain_docs[:3])
            
            for query in domain_queries[:2]:  # Limit for efficiency
                # Find relevant general documents
                for general_doc in general_docs[:3]:
                    # Check if general document is relevant to specific query
                    relevance = self._assess_hierarchical_relevance(query, general_doc)
                    
                    pair = TrainingPair(
                        query_id=query.query_id,
                        doc_id=general_doc.doc_id,
                        query_text=query.query_text,
                        doc_text=f"{general_doc.title} {general_doc.abstract}",
                        query_embedding=np.array([]),
                        doc_embedding=np.array([]),
                        relevance_label=relevance,
                        medical_domain=f"{domain}_to_general",
                        pair_type='hierarchical'
                    )
                    pairs.append(pair)
        
        logger.info(f"Created {len(pairs)} hierarchical pairs")
        return pairs
    
    def _generate_domain_queries(self, domain: str, docs: List[MedicalDocument], 
                                num_queries: int = 5) -> List[MedicalQuery]:
        """Generate medical queries for a specific domain."""
        queries = []
        
        # Domain-specific query templates
        domain_templates = {
            'oncology': [
                "Treatment options for {term}",
                "Diagnosis of {term}",
                "Prognosis and outcomes for {term}",
                "Risk factors for {term}",
                "Management of {term}"
            ],
            'neurology': [
                "Neurological symptoms of {term}",
                "Brain imaging for {term}",
                "Treatment protocols for {term}",
                "Cognitive effects of {term}",
                "Rehabilitation for {term}"
            ],
            'diabetes': [
                "Blood glucose management in {term}",
                "Complications of {term}",
                "Medication for {term}",
                "Dietary management of {term}",
                "Monitoring strategies for {term}"
            ],
            'respiratory': [
                "Breathing difficulties in {term}",
                "Pulmonary function in {term}",
                "Treatment of {term}",
                "Prevention of {term}",
                "Emergency management of {term}"
            ],
            'general': [
                "Clinical guidelines for {term}",
                "Evidence-based treatment of {term}",
                "Patient management for {term}",
                "Best practices for {term}",
                "Research updates on {term}"
            ]
        }
        
        templates = domain_templates.get(domain, domain_templates['general'])
        
        # Extract key terms from documents
        key_terms = set()
        for doc in docs:
            key_terms.update(doc.key_terms[:3])  # Top 3 terms per doc
        
        # Generate queries
        for i, term in enumerate(list(key_terms)[:num_queries]):
            template = templates[i % len(templates)]
            query_text = template.format(term=term)
            
            query = self.medical_relevance.create_medical_query(
                query_id=f"{domain}_query_{i}",
                query_text=query_text
            )
            queries.append(query)
        
        return queries
    
    def _find_relevant_docs_in_domain(self, query: MedicalQuery, 
                                    docs: List[MedicalDocument]) -> List[MedicalDocument]:
        """Find relevant documents within the same domain."""
        relevant_docs = []
        
        for doc in docs:
            # Calculate relevance based on keyword overlap and semantic similarity
            keyword_overlap = len(set(query.key_terms) & set(doc.key_terms))
            
            # Simple text similarity
            query_words = set(query.query_text.lower().split())
            doc_words = set((doc.title + " " + doc.abstract).lower().split())
            text_overlap = len(query_words & doc_words)
            
            # Combined relevance score
            relevance_score = keyword_overlap * 2 + text_overlap
            
            if relevance_score > 2:  # Threshold for relevance
                relevant_docs.append(doc)
        
        # Sort by relevance and return top docs
        return relevant_docs[:5]
    
    def _find_irrelevant_docs_in_domain(self, query: MedicalQuery,
                                      docs: List[MedicalDocument]) -> List[MedicalDocument]:
        """Find irrelevant documents within the same domain."""
        irrelevant_docs = []
        
        for doc in docs:
            # Calculate relevance
            keyword_overlap = len(set(query.key_terms) & set(doc.key_terms))
            query_words = set(query.query_text.lower().split())
            doc_words = set((doc.title + " " + doc.abstract).lower().split())
            text_overlap = len(query_words & doc_words)
            
            relevance_score = keyword_overlap * 2 + text_overlap
            
            if relevance_score <= 1:  # Low relevance threshold
                irrelevant_docs.append(doc)
        
        return irrelevant_docs[:5]
    
    def _assess_cross_domain_relevance(self, query: MedicalQuery, 
                                     doc: MedicalDocument) -> int:
        """Assess relevance between query and document from different domains."""
        # Check for general medical terms that might be cross-domain relevant
        general_medical_terms = {
            'patient', 'treatment', 'diagnosis', 'therapy', 'medication',
            'clinical', 'medical', 'disease', 'condition', 'symptoms'
        }
        
        query_words = set(query.query_text.lower().split())
        doc_words = set((doc.title + " " + doc.abstract).lower().split())
        
        # Check for cross-domain medical relevance
        general_overlap = len((query_words | doc_words) & general_medical_terms)
        
        if general_overlap > 3:
            return 1  # Moderate relevance
        else:
            return 0  # Not relevant
    
    def _assess_hierarchical_relevance(self, query: MedicalQuery,
                                     general_doc: MedicalDocument) -> int:
        """Assess relevance between specific query and general medical document."""
        # General medical documents might be relevant to specific queries
        query_words = set(query.query_text.lower().split())
        doc_words = set((general_doc.title + " " + general_doc.abstract).lower().split())
        
        # Look for medical terminology overlap
        medical_terms = {
            'patient', 'treatment', 'diagnosis', 'therapy', 'medication',
            'clinical', 'medical', 'disease', 'condition', 'symptoms',
            'health', 'care', 'outcome', 'management', 'intervention'
        }
        
        query_medical = len(query_words & medical_terms)
        doc_medical = len(doc_words & medical_terms)
        overlap = len(query_words & doc_words)
        
        if query_medical > 2 and doc_medical > 3 and overlap > 1:
            return 1  # Moderate relevance
        else:
            return 0  # Not relevant
    
    def _balance_training_pairs(self, all_pairs: List[TrainingPair]) -> List[TrainingPair]:
        """Balance training pairs according to strategy."""
        logger.info(f"Balancing {len(all_pairs)} training pairs")
        
        # Group by domain and type
        domain_pairs = defaultdict(list)
        type_pairs = defaultdict(list)
        
        for pair in all_pairs:
            domain_pairs[pair.medical_domain].append(pair)
            type_pairs[pair.pair_type].append(pair)
        
        balanced_pairs = []
        
        if self.config.domain_balance_strategy == "weighted":
            # Weight by inverse frequency to balance skewed distribution
            domain_weights = self._calculate_domain_weights()
            
            for domain, pairs in domain_pairs.items():
                weight = domain_weights.get(domain, 1.0)
                target_pairs = int(self.config.target_training_pairs * weight)
                sampled_pairs = random.sample(pairs, min(target_pairs, len(pairs)))
                balanced_pairs.extend(sampled_pairs)
        
        elif self.config.domain_balance_strategy == "equal":
            # Equal representation per domain
            pairs_per_domain = self.config.target_training_pairs // len(domain_pairs)
            
            for domain, pairs in domain_pairs.items():
                sampled_pairs = random.sample(pairs, min(pairs_per_domain, len(pairs)))
                balanced_pairs.extend(sampled_pairs)
        
        else:  # proportional
            # Keep original proportions
            target_ratio = self.config.target_training_pairs / len(all_pairs)
            
            for domain, pairs in domain_pairs.items():
                target_pairs = int(len(pairs) * target_ratio)
                sampled_pairs = random.sample(pairs, min(target_pairs, len(pairs)))
                balanced_pairs.extend(sampled_pairs)
        
        # Shuffle and limit to target
        random.shuffle(balanced_pairs)
        final_pairs = balanced_pairs[:self.config.target_training_pairs]
        
        logger.info(f"Balanced to {len(final_pairs)} training pairs")
        return final_pairs
    
    def _calculate_domain_weights(self) -> Dict[str, float]:
        """Calculate domain weights for balancing."""
        domain_counts = self.corpus_statistics['domain_counts']
        total_articles = self.corpus_statistics['total_articles']
        
        # Inverse frequency weighting
        domain_weights = {}
        for domain, count in domain_counts.items():
            frequency = count / total_articles
            weight = 1.0 / frequency  # Inverse frequency
            domain_weights[domain] = weight
        
        # Normalize weights
        total_weight = sum(domain_weights.values())
        normalized_weights = {
            domain: weight / total_weight 
            for domain, weight in domain_weights.items()
        }
        
        return normalized_weights
    
    def _generate_embeddings_for_pairs(self, pairs: List[TrainingPair]) -> List[TrainingPair]:
        """Generate embeddings for training pairs."""
        logger.info(f"Generating embeddings for {len(pairs)} training pairs")
        
        # Extract unique texts
        query_texts = list(set(pair.query_text for pair in pairs))
        doc_texts = list(set(pair.doc_text for pair in pairs))
        
        # Generate embeddings in batches
        logger.info("Generating query embeddings...")
        query_embeddings = self.embedding_processor.encode_texts(query_texts)
        query_embedding_map = {text: emb for text, emb in zip(query_texts, query_embeddings)}
        
        logger.info("Generating document embeddings...")
        doc_embeddings = self.embedding_processor.encode_texts(doc_texts)
        doc_embedding_map = {text: emb for text, emb in zip(doc_texts, doc_embeddings)}
        
        # Assign embeddings to pairs
        for pair in pairs:
            pair.query_embedding = query_embedding_map[pair.query_text]
            pair.doc_embedding = doc_embedding_map[pair.doc_text]
        
        return pairs


class MedicalCorpusQuantumTrainer:
    """
    Trainer for quantum kernels on medical corpus data.
    
    Integrates corpus analysis with quantum kernel training to optimize
    quantum similarity computation for medical document ranking.
    """
    
    def __init__(self, corpus_config: Optional[MedicalCorpusConfig] = None,
                 quantum_config: Optional[QuantumConfig] = None,
                 kta_config: Optional[KTAOptimizationConfig] = None):
        """Initialize medical corpus quantum trainer."""
        self.corpus_config = corpus_config or MedicalCorpusConfig()
        self.quantum_config = quantum_config or QuantumConfig()
        self.kta_config = kta_config or KTAOptimizationConfig()
        
        # Initialize components
        self.corpus_analyzer = MedicalCorpusAnalyzer(self.corpus_config)
        self.quantum_trainer = QuantumKernelTrainer(self.quantum_config, self.kta_config)
        
        # Training data
        self.training_pairs: List[TrainingPair] = []
        self.validation_pairs: List[TrainingPair] = []
        self.test_pairs: List[TrainingPair] = []
        
        logger.info("Medical corpus quantum trainer initialized")
    
    def train_on_medical_corpus(self) -> MedicalCorpusTrainingResult:
        """
        Execute complete training on medical corpus.
        
        Returns:
            Comprehensive training results
        """
        logger.info("Starting medical corpus quantum kernel training")
        start_time = time.time()
        
        # Stage 1: Load and analyze corpus
        logger.info("Stage 1: Loading and analyzing PMC medical corpus")
        pmc_articles = self.corpus_analyzer.load_pmc_corpus()
        corpus_analysis = self.corpus_analyzer.corpus_statistics
        
        # Stage 2: Create balanced training pairs
        logger.info("Stage 2: Creating balanced training pairs")
        all_pairs = self.corpus_analyzer.create_balanced_training_pairs()
        
        # Split into train/validation/test
        self._split_training_data(all_pairs)
        
        # Stage 3: Execute quantum kernel training
        logger.info("Stage 3: Training quantum kernels with KTA optimization")
        kta_result = self.quantum_trainer.train_on_medical_corpus(
            self.training_pairs, self.validation_pairs
        )
        
        # Stage 4: Domain-specific validation
        logger.info("Stage 4: Validating across medical domains")
        domain_results = self._validate_across_domains()
        
        # Stage 5: Cross-domain performance analysis
        logger.info("Stage 5: Analyzing cross-domain performance")
        cross_domain_results = self._analyze_cross_domain_performance()
        
        # Stage 6: Calculate performance improvements
        logger.info("Stage 6: Measuring performance improvements")
        performance_improvements = self._measure_performance_improvements()
        
        training_time = time.time() - start_time
        
        # Compile results
        result = MedicalCorpusTrainingResult(
            corpus_analysis=corpus_analysis,
            training_pairs_generated=len(all_pairs),
            domain_distribution=self._get_training_domain_distribution(),
            kta_training_result=kta_result,
            domain_specific_results=domain_results,
            cross_domain_performance=cross_domain_results,
            training_time_seconds=training_time,
            performance_improvements=performance_improvements
        )
        
        logger.info(f"Medical corpus training completed in {training_time:.2f}s")
        logger.info(f"KTA score: {kta_result.best_kta_score:.4f}")
        logger.info(f"Performance improvements: {performance_improvements}")
        
        return result
    
    def _split_training_data(self, all_pairs: List[TrainingPair]):
        """Split training data into train/validation/test sets."""
        # Shuffle pairs
        random.shuffle(all_pairs)
        
        # Calculate split indices
        total = len(all_pairs)
        test_size = int(total * self.corpus_config.test_split)
        val_size = int(total * self.corpus_config.validation_split)
        train_size = total - test_size - val_size
        
        self.test_pairs = all_pairs[:test_size]
        self.validation_pairs = all_pairs[test_size:test_size + val_size]
        self.training_pairs = all_pairs[test_size + val_size:]
        
        logger.info(f"Data split: {len(self.training_pairs)} train, "
                   f"{len(self.validation_pairs)} val, {len(self.test_pairs)} test")
    
    def _validate_across_domains(self) -> Dict[str, Dict[str, float]]:
        """Validate quantum kernel performance across medical domains."""
        domain_results = {}
        
        # Group test pairs by domain
        domain_test_pairs = defaultdict(list)
        for pair in self.test_pairs:
            domain = pair.medical_domain.split('_')[0]  # Get base domain
            domain_test_pairs[domain].append(pair)
        
        # Evaluate each domain
        for domain, pairs in domain_test_pairs.items():
            if len(pairs) >= 10:  # Minimum pairs for evaluation
                domain_result = self.quantum_trainer.evaluate_on_medical_ranking(pairs)
                domain_results[domain] = domain_result
        
        return domain_results
    
    def _analyze_cross_domain_performance(self) -> Dict[str, float]:
        """Analyze cross-domain transfer performance."""
        cross_domain_pairs = [pair for pair in self.test_pairs 
                            if 'to' in pair.medical_domain]
        
        if not cross_domain_pairs:
            return {}
        
        cross_domain_result = self.quantum_trainer.evaluate_on_medical_ranking(cross_domain_pairs)
        
        return {
            'cross_domain_correlation': cross_domain_result.get('ranking_correlation', 0),
            'cross_domain_accuracy': cross_domain_result.get('classification_accuracy', 0),
            'cross_domain_pairs_tested': len(cross_domain_pairs)
        }
    
    def _measure_performance_improvements(self) -> Dict[str, float]:
        """Measure performance improvements vs baseline."""
        # This would compare against classical baselines
        # For now, return KTA-based improvements
        
        kta_score = self.quantum_trainer.kta_optimizer.best_score
        baseline_kta = 0.1  # Typical random parameter KTA score
        
        kta_improvement = ((kta_score - baseline_kta) / baseline_kta * 100) if baseline_kta > 0 else 0
        
        return {
            'kta_improvement_percent': kta_improvement,
            'final_kta_score': kta_score,
            'baseline_kta_score': baseline_kta,
            'target_achieved': kta_score > 0.6  # QRF-05 target
        }
    
    def _get_training_domain_distribution(self) -> Dict[str, int]:
        """Get domain distribution of training pairs."""
        domain_counts = Counter()
        
        for pair in self.training_pairs:
            domain = pair.medical_domain.split('_')[0]  # Get base domain
            domain_counts[domain] += 1
        
        return dict(domain_counts)
    
    def save_training_results(self, result: MedicalCorpusTrainingResult, 
                            output_dir: str = "medical_corpus_training"):
        """Save comprehensive training results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save training pairs
        with open(output_path / "training_pairs.pkl", 'wb') as f:
            pickle.dump(self.training_pairs, f)
        
        with open(output_path / "validation_pairs.pkl", 'wb') as f:
            pickle.dump(self.validation_pairs, f)
        
        with open(output_path / "test_pairs.pkl", 'wb') as f:
            pickle.dump(self.test_pairs, f)
        
        # Save results
        with open(output_path / "training_results.pkl", 'wb') as f:
            pickle.dump(result, f)
        
        # Save trained quantum kernel
        if hasattr(self.quantum_trainer, 'quantum_kernel_engine'):
            kernel_path = output_path / "trained_quantum_kernel.pkl"
            self.quantum_trainer.save_trained_model(str(kernel_path), result.kta_training_result)
        
        logger.info(f"Training results saved to {output_path}")


def train_quantum_kernels_on_medical_corpus(
    corpus_config: Optional[MedicalCorpusConfig] = None,
    quantum_config: Optional[QuantumConfig] = None,
    kta_config: Optional[KTAOptimizationConfig] = None
) -> MedicalCorpusTrainingResult:
    """
    Convenience function to train quantum kernels on medical corpus.
    
    Args:
        corpus_config: Medical corpus configuration
        quantum_config: Quantum configuration
        kta_config: KTA optimization configuration
        
    Returns:
        Training results
    """
    trainer = MedicalCorpusQuantumTrainer(corpus_config, quantum_config, kta_config)
    return trainer.train_on_medical_corpus()
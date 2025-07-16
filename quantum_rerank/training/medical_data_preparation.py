"""
Medical Training Data Preparation for Quantum Parameter Optimization.

This module creates structured training datasets from medical corpus data for
optimizing quantum circuit parameters and parameter predictors on medical
domain-specific query-document pairs.

Based on QRF-04 requirements for medical domain quantum parameter training.
"""

import logging
import random
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
from collections import defaultdict, Counter

from ..evaluation.medical_relevance import (
    MedicalQuery, MedicalDocument, MedicalRelevanceJudgments,
    create_medical_test_queries
)
from ..core.embeddings import EmbeddingProcessor

logger = logging.getLogger(__name__)


@dataclass
class TrainingPair:
    """Query-document training pair with relevance label."""
    query_id: str
    doc_id: str
    query_text: str
    doc_text: str
    query_embedding: np.ndarray
    doc_embedding: np.ndarray
    relevance_label: int  # 0=irrelevant, 1=relevant, 2=highly_relevant
    medical_domain: str
    pair_type: str  # 'positive', 'negative', 'hard_negative'
    

@dataclass
class MedicalTrainingConfig:
    """Configuration for medical training data preparation."""
    target_pairs: int = 10000
    positive_ratio: float = 0.3
    negative_ratio: float = 0.5
    hard_negative_ratio: float = 0.2
    min_domain_coverage: int = 100  # Minimum pairs per domain
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    balance_domains: bool = True
    include_cross_domain: bool = True
    random_seed: int = 42


class MedicalTrainingDataset:
    """
    Medical training dataset for quantum parameter optimization.
    
    Creates structured query-document pairs from PMC medical corpus with
    proper relevance labels and balanced domain coverage.
    """
    
    def __init__(self, config: Optional[MedicalTrainingConfig] = None):
        """Initialize medical training dataset."""
        self.config = config or MedicalTrainingConfig()
        self.embedding_processor = EmbeddingProcessor()
        self.medical_relevance = MedicalRelevanceJudgments(self.embedding_processor)
        
        # Training data storage
        self.training_pairs: List[TrainingPair] = []
        self.domain_distribution: Dict[str, int] = Counter()
        self.pair_type_distribution: Dict[str, int] = Counter()
        
        # Set random seed for reproducibility
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        
        logger.info(f"Medical training dataset initialized with config: {self.config}")
    
    def create_training_pairs(self, documents: List[MedicalDocument],
                            queries: Optional[List[MedicalQuery]] = None) -> List[TrainingPair]:
        """
        Create balanced training pairs from medical documents and queries.
        
        Args:
            documents: Medical documents from corpus
            queries: Medical queries (generated if not provided)
            
        Returns:
            List of training pairs with embeddings and labels
        """
        logger.info(f"Creating training pairs from {len(documents)} documents")
        
        # Generate queries if not provided
        if queries is None:
            queries = self._generate_medical_queries(documents)
        
        logger.info(f"Using {len(queries)} queries for training pair generation")
        
        # Create different types of training pairs
        positive_pairs = self._create_positive_pairs(queries, documents)
        negative_pairs = self._create_negative_pairs(queries, documents)
        hard_negative_pairs = self._create_hard_negative_pairs(queries, documents)
        
        # Combine and balance pairs
        all_pairs = positive_pairs + negative_pairs + hard_negative_pairs
        balanced_pairs = self._balance_training_pairs(all_pairs)
        
        # Generate embeddings for all pairs
        final_pairs = self._generate_embeddings(balanced_pairs)
        
        self.training_pairs = final_pairs
        self._update_statistics()
        
        logger.info(f"Created {len(final_pairs)} training pairs")
        logger.info(f"Domain distribution: {dict(self.domain_distribution)}")
        logger.info(f"Pair type distribution: {dict(self.pair_type_distribution)}")
        
        return final_pairs
    
    def _generate_medical_queries(self, documents: List[MedicalDocument]) -> List[MedicalQuery]:
        """Generate medical queries from document content and standard test queries."""
        queries = []
        
        # Add standard medical test queries
        standard_queries = create_medical_test_queries()
        queries.extend(standard_queries)
        
        # Generate domain-specific queries from documents
        domain_queries = self._generate_domain_specific_queries(documents)
        queries.extend(domain_queries)
        
        # Generate complex multi-domain queries
        multi_domain_queries = self._generate_multi_domain_queries(documents)
        queries.extend(multi_domain_queries)
        
        logger.info(f"Generated {len(queries)} queries: "
                   f"{len(standard_queries)} standard, "
                   f"{len(domain_queries)} domain-specific, "
                   f"{len(multi_domain_queries)} multi-domain")
        
        return queries
    
    def _generate_domain_specific_queries(self, documents: List[MedicalDocument],
                                        queries_per_domain: int = 10) -> List[MedicalQuery]:
        """Generate domain-specific queries from document content."""
        domain_docs = defaultdict(list)
        
        # Group documents by domain
        for doc in documents:
            domain_docs[doc.medical_domain].append(doc)
        
        queries = []
        
        # Generate queries for each domain
        for domain, docs in domain_docs.items():
            domain_queries = self._extract_queries_from_domain_docs(domain, docs, queries_per_domain)
            queries.extend(domain_queries)
        
        return queries
    
    def _extract_queries_from_domain_docs(self, domain: str, docs: List[MedicalDocument],
                                        target_count: int) -> List[MedicalQuery]:
        """Extract queries from documents in a specific domain."""
        queries = []
        
        # Common query templates for medical domains
        query_templates = {
            'cardiology': [
                "How to diagnose {condition}?",
                "Treatment options for {condition}",
                "Risk factors for {condition}",
                "Prevention of {condition}",
                "Symptoms of {condition}"
            ],
            'diabetes': [
                "Management of {condition}",
                "Blood glucose monitoring in {condition}",
                "Complications of {condition}",
                "Diet recommendations for {condition}",
                "Medication for {condition}"
            ],
            'respiratory': [
                "Diagnosis of {condition}",
                "Treatment protocols for {condition}",
                "Emergency management of {condition}",
                "Prevention strategies for {condition}",
                "Monitoring {condition}"
            ],
            'neurology': [
                "Neurological assessment of {condition}",
                "Brain imaging for {condition}",
                "Treatment approaches for {condition}",
                "Prognosis of {condition}",
                "Rehabilitation for {condition}"
            ],
            'general': [
                "Clinical guidelines for {condition}",
                "Evidence-based treatment of {condition}",
                "Patient management for {condition}",
                "Research updates on {condition}",
                "Best practices for {condition}"
            ]
        }
        
        templates = query_templates.get(domain, query_templates['general'])
        
        # Extract key terms and conditions from documents
        conditions = set()
        for doc in docs[:20]:  # Use first 20 docs to avoid too many queries
            conditions.update(doc.key_terms[:3])  # Use top 3 key terms
        
        # Generate queries using templates and conditions
        for i, condition in enumerate(list(conditions)[:target_count]):
            template = templates[i % len(templates)]
            query_text = template.format(condition=condition)
            
            query = self.medical_relevance.create_medical_query(
                query_id=f"{domain}_generated_{i}",
                query_text=query_text
            )
            queries.append(query)
        
        return queries[:target_count]
    
    def _generate_multi_domain_queries(self, documents: List[MedicalDocument],
                                     count: int = 20) -> List[MedicalQuery]:
        """Generate complex multi-domain medical queries."""
        queries = []
        
        # Multi-domain query templates
        multi_templates = [
            "Management of {condition1} in patients with {condition2}",
            "Treatment of {condition1} and {condition2} comorbidity",
            "{condition1} with {condition2} complications",
            "Diagnosis of {condition1} versus {condition2}",
            "Risk factors for {condition1} in {condition2} patients",
            "Drug interactions between {condition1} and {condition2} medications",
            "Surgical considerations for {condition1} with {condition2}",
            "Monitoring {condition1} patients with {condition2}"
        ]
        
        # Extract conditions from different domains
        domain_conditions = defaultdict(list)
        for doc in documents:
            if doc.key_terms:
                domain_conditions[doc.medical_domain].extend(doc.key_terms[:2])
        
        # Generate multi-domain queries
        domains = list(domain_conditions.keys())
        for i in range(count):
            if len(domains) >= 2:
                domain1, domain2 = random.sample(domains, 2)
                
                if domain_conditions[domain1] and domain_conditions[domain2]:
                    condition1 = random.choice(domain_conditions[domain1])
                    condition2 = random.choice(domain_conditions[domain2])
                    template = random.choice(multi_templates)
                    
                    query_text = template.format(condition1=condition1, condition2=condition2)
                    
                    query = self.medical_relevance.create_medical_query(
                        query_id=f"multi_domain_{i}",
                        query_text=query_text
                    )
                    queries.append(query)
        
        return queries
    
    def _create_positive_pairs(self, queries: List[MedicalQuery],
                             documents: List[MedicalDocument]) -> List[TrainingPair]:
        """Create positive training pairs (relevant query-document matches)."""
        positive_pairs = []
        target_positive = int(self.config.target_pairs * self.config.positive_ratio)
        
        logger.info(f"Creating {target_positive} positive training pairs")
        
        for query in queries:
            # Find relevant documents for this query
            relevant_docs = self._find_relevant_documents(query, documents, min_relevance=1)
            
            for doc in relevant_docs[:5]:  # Limit per query to avoid bias
                pair = TrainingPair(
                    query_id=query.query_id,
                    doc_id=doc.doc_id,
                    query_text=query.query_text,
                    doc_text=f"{doc.title} {doc.abstract}",
                    query_embedding=np.array([]),  # Will be filled later
                    doc_embedding=np.array([]),    # Will be filled later
                    relevance_label=2,  # Highly relevant
                    medical_domain=query.medical_domain,
                    pair_type='positive'
                )
                positive_pairs.append(pair)
                
                if len(positive_pairs) >= target_positive:
                    break
            
            if len(positive_pairs) >= target_positive:
                break
        
        return positive_pairs[:target_positive]
    
    def _create_negative_pairs(self, queries: List[MedicalQuery],
                             documents: List[MedicalDocument]) -> List[TrainingPair]:
        """Create negative training pairs (irrelevant query-document matches)."""
        negative_pairs = []
        target_negative = int(self.config.target_pairs * self.config.negative_ratio)
        
        logger.info(f"Creating {target_negative} negative training pairs")
        
        for query in queries:
            # Find irrelevant documents (different domain, low relevance)
            irrelevant_docs = self._find_irrelevant_documents(query, documents)
            
            for doc in random.sample(irrelevant_docs, min(3, len(irrelevant_docs))):
                pair = TrainingPair(
                    query_id=query.query_id,
                    doc_id=doc.doc_id,
                    query_text=query.query_text,
                    doc_text=f"{doc.title} {doc.abstract}",
                    query_embedding=np.array([]),
                    doc_embedding=np.array([]),
                    relevance_label=0,  # Irrelevant
                    medical_domain=query.medical_domain,
                    pair_type='negative'
                )
                negative_pairs.append(pair)
                
                if len(negative_pairs) >= target_negative:
                    break
            
            if len(negative_pairs) >= target_negative:
                break
        
        return negative_pairs[:target_negative]
    
    def _create_hard_negative_pairs(self, queries: List[MedicalQuery],
                                  documents: List[MedicalDocument]) -> List[TrainingPair]:
        """Create hard negative pairs (same domain but not specifically relevant)."""
        hard_negative_pairs = []
        target_hard_negative = int(self.config.target_pairs * self.config.hard_negative_ratio)
        
        logger.info(f"Creating {target_hard_negative} hard negative training pairs")
        
        for query in queries:
            # Find documents in same domain but not specifically relevant
            same_domain_docs = [doc for doc in documents 
                              if doc.medical_domain == query.medical_domain]
            
            # Calculate relevance and select moderately relevant docs as hard negatives
            for doc in same_domain_docs:
                relevance_score, confidence = self.medical_relevance.calculate_relevance_score(query, doc)
                
                # Hard negatives: same domain, moderate relevance (not 0, not 2)
                if relevance_score == 1 and confidence < 0.7:
                    pair = TrainingPair(
                        query_id=query.query_id,
                        doc_id=doc.doc_id,
                        query_text=query.query_text,
                        doc_text=f"{doc.title} {doc.abstract}",
                        query_embedding=np.array([]),
                        doc_embedding=np.array([]),
                        relevance_label=1,  # Partially relevant
                        medical_domain=query.medical_domain,
                        pair_type='hard_negative'
                    )
                    hard_negative_pairs.append(pair)
                    
                    if len(hard_negative_pairs) >= target_hard_negative:
                        break
            
            if len(hard_negative_pairs) >= target_hard_negative:
                break
        
        return hard_negative_pairs[:target_hard_negative]
    
    def _find_relevant_documents(self, query: MedicalQuery, documents: List[MedicalDocument],
                               min_relevance: int = 1) -> List[MedicalDocument]:
        """Find documents relevant to the query."""
        relevant_docs = []
        
        for doc in documents:
            relevance_score, confidence = self.medical_relevance.calculate_relevance_score(query, doc)
            
            if relevance_score >= min_relevance and confidence > 0.5:
                relevant_docs.append(doc)
        
        # Sort by relevance (domain match first, then keyword overlap)
        def relevance_key(doc):
            domain_match = 2 if doc.medical_domain == query.medical_domain else 0
            keyword_overlap = len(set(query.key_terms) & set(doc.key_terms))
            return domain_match + keyword_overlap
        
        relevant_docs.sort(key=relevance_key, reverse=True)
        return relevant_docs
    
    def _find_irrelevant_documents(self, query: MedicalQuery,
                                 documents: List[MedicalDocument]) -> List[MedicalDocument]:
        """Find documents irrelevant to the query."""
        irrelevant_docs = []
        
        for doc in documents:
            # Different domain and no keyword overlap = irrelevant
            if (doc.medical_domain != query.medical_domain and 
                doc.medical_domain != 'general' and
                query.medical_domain != 'general'):
                
                keyword_overlap = len(set(query.key_terms) & set(doc.key_terms))
                if keyword_overlap == 0:
                    irrelevant_docs.append(doc)
        
        return irrelevant_docs
    
    def _balance_training_pairs(self, pairs: List[TrainingPair]) -> List[TrainingPair]:
        """Balance training pairs across domains and types."""
        if not self.config.balance_domains:
            return pairs[:self.config.target_pairs]
        
        # Group pairs by domain and type
        domain_pairs = defaultdict(list)
        type_pairs = defaultdict(list)
        
        for pair in pairs:
            domain_pairs[pair.medical_domain].append(pair)
            type_pairs[pair.pair_type].append(pair)
        
        # Ensure minimum coverage per domain
        balanced_pairs = []
        domains = list(domain_pairs.keys())
        
        # Calculate pairs per domain
        pairs_per_domain = max(
            self.config.min_domain_coverage,
            self.config.target_pairs // len(domains)
        )
        
        for domain in domains:
            domain_pairs_list = domain_pairs[domain]
            
            # Balance types within domain
            positive_count = int(pairs_per_domain * self.config.positive_ratio)
            negative_count = int(pairs_per_domain * self.config.negative_ratio)
            hard_negative_count = pairs_per_domain - positive_count - negative_count
            
            domain_positive = [p for p in domain_pairs_list if p.pair_type == 'positive']
            domain_negative = [p for p in domain_pairs_list if p.pair_type == 'negative']
            domain_hard_negative = [p for p in domain_pairs_list if p.pair_type == 'hard_negative']
            
            # Sample balanced pairs from each type
            balanced_pairs.extend(random.sample(domain_positive, min(positive_count, len(domain_positive))))
            balanced_pairs.extend(random.sample(domain_negative, min(negative_count, len(domain_negative))))
            balanced_pairs.extend(random.sample(domain_hard_negative, min(hard_negative_count, len(domain_hard_negative))))
        
        # Shuffle and return target number
        random.shuffle(balanced_pairs)
        return balanced_pairs[:self.config.target_pairs]
    
    def _generate_embeddings(self, pairs: List[TrainingPair]) -> List[TrainingPair]:
        """Generate embeddings for all training pairs."""
        logger.info(f"Generating embeddings for {len(pairs)} training pairs")
        
        # Extract unique texts for batch processing
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
    
    def _update_statistics(self):
        """Update dataset statistics."""
        self.domain_distribution = Counter(pair.medical_domain for pair in self.training_pairs)
        self.pair_type_distribution = Counter(pair.pair_type for pair in self.training_pairs)
    
    def split_dataset(self, pairs: Optional[List[TrainingPair]] = None) -> Tuple[List[TrainingPair], List[TrainingPair], List[TrainingPair]]:
        """Split dataset into train/validation/test sets."""
        if pairs is None:
            pairs = self.training_pairs
        
        # Shuffle pairs
        shuffled_pairs = pairs.copy()
        random.shuffle(shuffled_pairs)
        
        # Calculate split indices
        total = len(shuffled_pairs)
        train_end = int(total * self.config.train_split)
        val_end = train_end + int(total * self.config.val_split)
        
        train_pairs = shuffled_pairs[:train_end]
        val_pairs = shuffled_pairs[train_end:val_end]
        test_pairs = shuffled_pairs[val_end:]
        
        logger.info(f"Dataset split: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test")
        
        return train_pairs, val_pairs, test_pairs
    
    def save_dataset(self, filepath: str, pairs: Optional[List[TrainingPair]] = None):
        """Save training dataset to disk."""
        if pairs is None:
            pairs = self.training_pairs
        
        dataset_data = {
            'pairs': pairs,
            'config': self.config,
            'domain_distribution': dict(self.domain_distribution),
            'pair_type_distribution': dict(self.pair_type_distribution)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(dataset_data, f)
        
        logger.info(f"Saved {len(pairs)} training pairs to {filepath}")
    
    def load_dataset(self, filepath: str) -> List[TrainingPair]:
        """Load training dataset from disk."""
        with open(filepath, 'rb') as f:
            dataset_data = pickle.load(f)
        
        self.training_pairs = dataset_data['pairs']
        self.config = dataset_data['config']
        self.domain_distribution = Counter(dataset_data['domain_distribution'])
        self.pair_type_distribution = Counter(dataset_data['pair_type_distribution'])
        
        logger.info(f"Loaded {len(self.training_pairs)} training pairs from {filepath}")
        return self.training_pairs


class MedicalDataPreparationPipeline:
    """
    Complete pipeline for preparing medical training data for quantum optimization.
    """
    
    def __init__(self, config: Optional[MedicalTrainingConfig] = None):
        """Initialize medical data preparation pipeline."""
        self.config = config or MedicalTrainingConfig()
        self.dataset = MedicalTrainingDataset(self.config)
        
        logger.info("Medical data preparation pipeline initialized")
    
    def run(self, documents: List[MedicalDocument],
            queries: Optional[List[MedicalQuery]] = None,
            output_dir: str = "medical_training_data") -> Dict[str, Any]:
        """
        Run complete medical data preparation pipeline.
        
        Args:
            documents: Medical documents from corpus
            queries: Medical queries (optional)
            output_dir: Output directory for results
            
        Returns:
            Dictionary with pipeline results and file paths
        """
        logger.info("Starting medical data preparation pipeline")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Create training pairs
        logger.info("Step 1: Creating training pairs")
        training_pairs = self.dataset.create_training_pairs(documents, queries)
        
        # Step 2: Split dataset
        logger.info("Step 2: Splitting dataset")
        train_pairs, val_pairs, test_pairs = self.dataset.split_dataset(training_pairs)
        
        # Step 3: Save datasets
        logger.info("Step 3: Saving datasets")
        train_path = output_path / "medical_train_pairs.pkl"
        val_path = output_path / "medical_val_pairs.pkl"
        test_path = output_path / "medical_test_pairs.pkl"
        full_path = output_path / "medical_full_dataset.pkl"
        
        self.dataset.save_dataset(str(train_path), train_pairs)
        self.dataset.save_dataset(str(val_path), val_pairs)
        self.dataset.save_dataset(str(test_path), test_pairs)
        self.dataset.save_dataset(str(full_path), training_pairs)
        
        # Step 4: Generate statistics
        stats = self._generate_statistics(training_pairs, train_pairs, val_pairs, test_pairs)
        
        # Save statistics
        stats_path = output_path / "dataset_statistics.pkl"
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)
        
        results = {
            'total_pairs': len(training_pairs),
            'train_pairs': len(train_pairs),
            'val_pairs': len(val_pairs),
            'test_pairs': len(test_pairs),
            'domain_distribution': dict(self.dataset.domain_distribution),
            'pair_type_distribution': dict(self.dataset.pair_type_distribution),
            'statistics': stats,
            'file_paths': {
                'train': str(train_path),
                'val': str(val_path),
                'test': str(test_path),
                'full': str(full_path),
                'stats': str(stats_path)
            }
        }
        
        logger.info("Medical data preparation pipeline completed successfully")
        logger.info(f"Results: {results}")
        
        return results
    
    def _generate_statistics(self, full_pairs: List[TrainingPair],
                           train_pairs: List[TrainingPair],
                           val_pairs: List[TrainingPair],
                           test_pairs: List[TrainingPair]) -> Dict[str, Any]:
        """Generate comprehensive dataset statistics."""
        
        def analyze_pairs(pairs: List[TrainingPair], name: str) -> Dict[str, Any]:
            if not pairs:
                return {}
            
            domains = Counter(pair.medical_domain for pair in pairs)
            types = Counter(pair.pair_type for pair in pairs)
            labels = Counter(pair.relevance_label for pair in pairs)
            
            # Embedding statistics
            query_embeddings = np.array([pair.query_embedding for pair in pairs])
            doc_embeddings = np.array([pair.doc_embedding for pair in pairs])
            
            return {
                'count': len(pairs),
                'domain_distribution': dict(domains),
                'type_distribution': dict(types),
                'label_distribution': dict(labels),
                'embedding_stats': {
                    'query_embedding_shape': query_embeddings.shape,
                    'doc_embedding_shape': doc_embeddings.shape,
                    'query_embedding_mean': np.mean(query_embeddings, axis=0)[:5].tolist(),  # First 5 dims
                    'doc_embedding_mean': np.mean(doc_embeddings, axis=0)[:5].tolist()      # First 5 dims
                }
            }
        
        stats = {
            'full_dataset': analyze_pairs(full_pairs, 'full'),
            'train_dataset': analyze_pairs(train_pairs, 'train'),
            'val_dataset': analyze_pairs(val_pairs, 'val'),
            'test_dataset': analyze_pairs(test_pairs, 'test'),
            'config': self.config
        }
        
        return stats


def create_medical_training_pairs(documents: List[MedicalDocument],
                                queries: Optional[List[MedicalQuery]] = None,
                                config: Optional[MedicalTrainingConfig] = None) -> List[TrainingPair]:
    """
    Convenience function to create medical training pairs.
    
    Args:
        documents: Medical documents
        queries: Medical queries (optional)
        config: Training configuration (optional)
        
    Returns:
        List of training pairs
    """
    dataset = MedicalTrainingDataset(config)
    return dataset.create_training_pairs(documents, queries)
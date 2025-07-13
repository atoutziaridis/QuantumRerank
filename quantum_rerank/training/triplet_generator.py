"""
Triplet Dataset Generation for QPMeL Training.

Generates (anchor, positive, negative) triplets from IR datasets like NFCorpus,
MS MARCO, or any dataset with relevance judgments.
"""

import random
import logging
from typing import List, Tuple, Dict, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class TripletGeneratorConfig:
    """Configuration for triplet generation."""
    # Sampling parameters
    negatives_per_positive: int = 5  # How many negatives per positive
    max_triplets_per_query: int = 100  # Limit triplets per query
    min_positive_relevance: int = 1  # Minimum relevance score for positives
    
    # Quality control
    min_text_length: int = 10  # Minimum characters in text
    max_text_length: int = 1000  # Maximum characters (for efficiency)
    
    # Balancing
    balance_queries: bool = True  # Ensure similar number of triplets per query
    shuffle_triplets: bool = True  # Shuffle final triplet list

class TripletGenerator:
    """
    Generates training triplets from IR datasets with relevance judgments.
    
    Supports various input formats:
    - IR datasets (queries, documents, qrels)
    - Custom triplet data
    - Sentence similarity datasets
    """
    
    def __init__(self, config: TripletGeneratorConfig = None):
        self.config = config or TripletGeneratorConfig()
        self.triplets = []
        
    def from_ir_dataset(self, 
                       queries: Dict[str, str],
                       documents: Dict[str, str], 
                       qrels: Dict[str, Dict[str, int]]) -> List[Tuple[str, str, str]]:
        """
        Generate triplets from IR dataset with relevance judgments.
        
        Args:
            queries: Dict mapping query_id -> query_text
            documents: Dict mapping doc_id -> doc_text
            qrels: Dict mapping query_id -> {doc_id -> relevance_score}
            
        Returns:
            List of (anchor=query, positive=relevant_doc, negative=irrelevant_doc) triplets
        """
        logger.info(f"Generating triplets from IR dataset: "
                   f"{len(queries)} queries, {len(documents)} documents")
        
        triplets = []
        
        for query_id, query_text in queries.items():
            if query_id not in qrels:
                continue
                
            # Filter and validate query text
            if not self._is_valid_text(query_text):
                continue
            
            # Get relevant and irrelevant documents for this query
            query_qrels = qrels[query_id]
            
            relevant_docs = []
            irrelevant_docs = []
            
            for doc_id, relevance in query_qrels.items():
                if doc_id not in documents:
                    continue
                    
                doc_text = documents[doc_id]
                if not self._is_valid_text(doc_text):
                    continue
                
                if relevance >= self.config.min_positive_relevance:
                    relevant_docs.append(doc_text)
                else:
                    irrelevant_docs.append(doc_text)
            
            # Add random irrelevant documents from the corpus
            all_doc_texts = [doc for doc in documents.values() if self._is_valid_text(doc)]
            additional_negatives = random.sample(
                all_doc_texts, 
                min(len(all_doc_texts), len(relevant_docs) * self.config.negatives_per_positive)
            )
            irrelevant_docs.extend(additional_negatives)
            
            # Generate triplets for this query
            query_triplets = self._generate_query_triplets(
                query_text, relevant_docs, irrelevant_docs
            )
            
            triplets.extend(query_triplets)
            
            if len(triplets) % 1000 == 0:
                logger.info(f"Generated {len(triplets)} triplets so far...")
        
        if self.config.shuffle_triplets:
            random.shuffle(triplets)
        
        logger.info(f"Generated {len(triplets)} triplets total")
        return triplets
    
    def from_nfcorpus(self, data_dir: str) -> List[Tuple[str, str, str]]:
        """
        Generate triplets from NFCorpus dataset.
        
        Args:
            data_dir: Path to NFCorpus data directory
            
        Returns:
            List of triplets
        """
        try:
            import ir_datasets
            dataset = ir_datasets.load("nfcorpus/train")
            
            # Load data
            queries = {q.query_id: q.text for q in dataset.queries_iter()}
            documents = {d.doc_id: f"{d.title} {d.text}" for d in dataset.docs_iter()}
            
            # Build qrels dictionary
            qrels = defaultdict(dict)
            for qrel in dataset.qrels_iter():
                qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
            
            return self.from_ir_dataset(queries, documents, dict(qrels))
            
        except ImportError:
            logger.error("ir_datasets not installed. Install with: pip install ir-datasets")
            return []
        except Exception as e:
            logger.error(f"Failed to load NFCorpus: {e}")
            return []
    
    def from_msmarco_dev(self, data_dir: str) -> List[Tuple[str, str, str]]:
        """
        Generate triplets from MS MARCO dev dataset.
        
        Args:
            data_dir: Path to MS MARCO data directory
            
        Returns:
            List of triplets
        """
        try:
            import ir_datasets
            dataset = ir_datasets.load("msmarco-passage/dev/small")
            
            # Load data (limited for memory efficiency)
            queries = {}
            for i, q in enumerate(dataset.queries_iter()):
                if i >= 1000:  # Limit to 1000 queries for efficiency
                    break
                queries[q.query_id] = q.text
            
            documents = {}
            for i, d in enumerate(dataset.docs_iter()):
                if i >= 10000:  # Limit to 10K documents
                    break
                documents[d.doc_id] = d.text
            
            # Build qrels
            qrels = defaultdict(dict)
            for qrel in dataset.qrels_iter():
                if qrel.query_id in queries:
                    qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
            
            return self.from_ir_dataset(queries, documents, dict(qrels))
            
        except ImportError:
            logger.error("ir_datasets not installed. Install with: pip install ir-datasets")
            return []
        except Exception as e:
            logger.error(f"Failed to load MS MARCO: {e}")
            return []
    
    def from_sentence_transformers(self, dataset_name: str = "all-nli") -> List[Tuple[str, str, str]]:
        """
        Generate triplets from SentenceTransformers datasets.
        
        Args:
            dataset_name: Name of the SentenceTransformers dataset
            
        Returns:
            List of triplets
        """
        try:
            from sentence_transformers import InputExample
            from sentence_transformers.datasets import NoDuplicatesDataLoader
            import datasets
            
            # Load dataset
            if dataset_name == "all-nli":
                # Load AllNLI dataset for triplet training
                dataset = datasets.load_dataset("sentence-transformers/all-nli", "triplet")
                
                triplets = []
                for example in dataset["train"]:
                    anchor = example["anchor"]
                    positive = example["positive"] 
                    negative = example["negative"]
                    
                    if (self._is_valid_text(anchor) and 
                        self._is_valid_text(positive) and 
                        self._is_valid_text(negative)):
                        triplets.append((anchor, positive, negative))
                
                if self.config.shuffle_triplets:
                    random.shuffle(triplets)
                
                # Limit size for efficiency
                max_triplets = 10000
                if len(triplets) > max_triplets:
                    triplets = triplets[:max_triplets]
                
                logger.info(f"Generated {len(triplets)} triplets from {dataset_name}")
                return triplets
                
        except ImportError:
            logger.error("sentence-transformers or datasets not installed")
            return []
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformers dataset: {e}")
            return []
        
        return []
    
    def from_custom_data(self, 
                        queries: List[str],
                        relevant_docs: List[List[str]],
                        irrelevant_docs: List[str]) -> List[Tuple[str, str, str]]:
        """
        Generate triplets from custom data.
        
        Args:
            queries: List of query texts
            relevant_docs: List of lists, where relevant_docs[i] contains docs relevant to queries[i]
            irrelevant_docs: List of irrelevant document texts
            
        Returns:
            List of triplets
        """
        if len(queries) != len(relevant_docs):
            raise ValueError("queries and relevant_docs must have same length")
        
        triplets = []
        
        for i, (query, query_relevant_docs) in enumerate(zip(queries, relevant_docs)):
            if not self._is_valid_text(query):
                continue
            
            # Filter valid relevant docs
            valid_relevant = [doc for doc in query_relevant_docs if self._is_valid_text(doc)]
            if not valid_relevant:
                continue
            
            # Generate triplets for this query
            query_triplets = self._generate_query_triplets(
                query, valid_relevant, irrelevant_docs
            )
            triplets.extend(query_triplets)
        
        if self.config.shuffle_triplets:
            random.shuffle(triplets)
        
        logger.info(f"Generated {len(triplets)} triplets from custom data")
        return triplets
    
    def _generate_query_triplets(self, 
                                query: str,
                                relevant_docs: List[str],
                                irrelevant_docs: List[str]) -> List[Tuple[str, str, str]]:
        """Generate triplets for a single query."""
        triplets = []
        
        if not relevant_docs or not irrelevant_docs:
            return triplets
        
        # Generate up to max_triplets_per_query
        max_triplets = min(
            self.config.max_triplets_per_query,
            len(relevant_docs) * self.config.negatives_per_positive
        )
        
        for _ in range(max_triplets):
            # Sample positive and negative
            positive = random.choice(relevant_docs)
            negative = random.choice(irrelevant_docs)
            
            # Ensure negative is different from positive
            if positive != negative:
                triplets.append((query, positive, negative))
        
        return triplets
    
    def _is_valid_text(self, text: str) -> bool:
        """Check if text meets quality requirements."""
        if not text or not isinstance(text, str):
            return False
        
        text = text.strip()
        
        if len(text) < self.config.min_text_length:
            return False
        
        if len(text) > self.config.max_text_length:
            return False
        
        return True
    
    def save_triplets(self, triplets: List[Tuple[str, str, str]], path: str):
        """Save triplets to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        triplet_dicts = [
            {"anchor": anchor, "positive": positive, "negative": negative}
            for anchor, positive, negative in triplets
        ]
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(triplet_dicts, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(triplets)} triplets to {path}")
    
    def load_triplets(self, path: str) -> List[Tuple[str, str, str]]:
        """Load triplets from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            triplet_dicts = json.load(f)
        
        triplets = [
            (t["anchor"], t["positive"], t["negative"])
            for t in triplet_dicts
        ]
        
        logger.info(f"Loaded {len(triplets)} triplets from {path}")
        return triplets

def create_synthetic_triplets(num_triplets: int = 1000) -> List[Tuple[str, str, str]]:
    """
    Create synthetic triplets for quick testing.
    
    Args:
        num_triplets: Number of triplets to generate
        
    Returns:
        List of synthetic triplets
    """
    domains = [
        ("quantum computing", [
            "quantum algorithms and their applications",
            "quantum circuits and gate operations", 
            "quantum error correction methods",
            "quantum machine learning techniques"
        ], [
            "classical computer programming",
            "traditional database systems",
            "web development frameworks",
            "mobile app development"
        ]),
        ("machine learning", [
            "neural networks and deep learning",
            "supervised learning algorithms",
            "unsupervised clustering methods",
            "reinforcement learning strategies"
        ], [
            "mechanical engineering design",
            "chemical reaction analysis",
            "historical literature review",
            "culinary recipe development"
        ]),
        ("information retrieval", [
            "search engine optimization techniques",
            "document ranking algorithms", 
            "text similarity measurements",
            "query expansion methods"
        ], [
            "musical composition theory",
            "geological survey data",
            "astronomical observations",
            "botanical classification"
        ])
    ]
    
    triplets = []
    
    for _ in range(num_triplets):
        # Choose random domain
        domain_name, positives, negatives = random.choice(domains)
        
        # Create query
        query = f"How does {domain_name} work?"
        
        # Sample positive and negative
        positive = random.choice(positives)
        negative = random.choice(negatives)
        
        triplets.append((query, positive, negative))
    
    return triplets

# Convenience functions for common datasets
def load_nfcorpus_triplets(data_dir: str = "./data") -> List[Tuple[str, str, str]]:
    """Load triplets from NFCorpus dataset."""
    generator = TripletGenerator()
    return generator.from_nfcorpus(data_dir)

def load_msmarco_triplets(data_dir: str = "./data") -> List[Tuple[str, str, str]]:
    """Load triplets from MS MARCO dataset.""" 
    generator = TripletGenerator()
    return generator.from_msmarco_dev(data_dir)

def load_sentence_transformers_triplets() -> List[Tuple[str, str, str]]:
    """Load triplets from SentenceTransformers AllNLI dataset."""
    generator = TripletGenerator()
    return generator.from_sentence_transformers("all-nli")
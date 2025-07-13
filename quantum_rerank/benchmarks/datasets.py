"""
Standard Benchmark Datasets for QuantumRerank Performance Testing.

Provides standardized datasets for consistent benchmarking across
quantum and classical similarity computation methods.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkQuery:
    """Single benchmark query with ground truth relevance."""
    query_id: str
    query_text: str
    relevant_docs: List[str]  # Document IDs
    relevance_scores: Dict[str, float]  # doc_id -> relevance score


@dataclass
class BenchmarkDocument:
    """Single benchmark document."""
    doc_id: str
    text: str
    metadata: Dict[str, Any]


@dataclass
class BenchmarkDataset:
    """Complete benchmark dataset with queries and documents."""
    name: str
    description: str
    queries: List[BenchmarkQuery]
    documents: List[BenchmarkDocument]
    metadata: Dict[str, Any]
    
    def get_document_by_id(self, doc_id: str) -> Optional[BenchmarkDocument]:
        """Get document by ID."""
        for doc in self.documents:
            if doc.doc_id == doc_id:
                return doc
        return None
    
    def get_query_by_id(self, query_id: str) -> Optional[BenchmarkQuery]:
        """Get query by ID."""
        for query in self.queries:
            if query.query_id == query_id:
                return query
        return None


class DatasetGenerator(ABC):
    """Abstract base class for dataset generators."""
    
    @abstractmethod
    def generate_dataset(self, size: str = "small") -> BenchmarkDataset:
        """Generate a benchmark dataset."""
        pass


class SyntheticSimilarityDataset(DatasetGenerator):
    """
    Generator for synthetic similarity datasets.
    
    Creates datasets with known similarity relationships for validation.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize synthetic dataset generator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def generate_dataset(self, size: str = "small") -> BenchmarkDataset:
        """
        Generate synthetic similarity dataset.
        
        Args:
            size: Dataset size ('small', 'medium', 'large')
            
        Returns:
            BenchmarkDataset with synthetic queries and documents
        """
        size_configs = {
            "small": {"queries": 10, "docs": 50},
            "medium": {"queries": 50, "docs": 200}, 
            "large": {"queries": 100, "docs": 500}
        }
        
        config = size_configs.get(size, size_configs["small"])
        
        # Topic clusters for generating related content
        topics = [
            "quantum computing",
            "machine learning", 
            "information retrieval",
            "natural language processing",
            "classical algorithms",
            "data structures",
            "artificial intelligence",
            "computer vision",
            "robotics",
            "software engineering"
        ]
        
        # Generate documents
        documents = []
        for i in range(config["docs"]):
            topic = np.random.choice(topics)
            doc_text = self._generate_document_text(topic, i)
            
            doc = BenchmarkDocument(
                doc_id=f"doc_{i:04d}",
                text=doc_text,
                metadata={"topic": topic, "synthetic": True}
            )
            documents.append(doc)
        
        # Generate queries with known relevant documents
        queries = []
        for i in range(config["queries"]):
            query_topic = np.random.choice(topics)
            query_text = self._generate_query_text(query_topic, i)
            
            # Find relevant documents (same topic + some random)
            relevant_doc_ids = []
            relevance_scores = {}
            
            # High relevance: same topic documents
            same_topic_docs = [d for d in documents if d.metadata["topic"] == query_topic]
            num_relevant = min(5, len(same_topic_docs))
            selected_relevant = np.random.choice(same_topic_docs, num_relevant, replace=False)
            
            for doc in selected_relevant:
                relevant_doc_ids.append(doc.doc_id)
                relevance_scores[doc.doc_id] = np.random.uniform(0.7, 1.0)
            
            # Medium relevance: related topics
            related_topics = self._get_related_topics(query_topic)
            for topic in related_topics[:2]:  # Max 2 related topics
                topic_docs = [d for d in documents if d.metadata["topic"] == topic]
                if topic_docs:
                    doc = np.random.choice(topic_docs)
                    if doc.doc_id not in relevant_doc_ids:
                        relevant_doc_ids.append(doc.doc_id)
                        relevance_scores[doc.doc_id] = np.random.uniform(0.3, 0.6)
            
            query = BenchmarkQuery(
                query_id=f"query_{i:04d}",
                query_text=query_text,
                relevant_docs=relevant_doc_ids,
                relevance_scores=relevance_scores
            )
            queries.append(query)
        
        return BenchmarkDataset(
            name=f"synthetic_similarity_{size}",
            description=f"Synthetic similarity dataset ({size} size) with {config['queries']} queries and {config['docs']} documents",
            queries=queries,
            documents=documents,
            metadata={
                "generator": "SyntheticSimilarityDataset",
                "random_seed": self.random_seed,
                "size": size,
                "topics": topics
            }
        )
    
    def _generate_document_text(self, topic: str, doc_id: int) -> str:
        """Generate synthetic document text for a topic."""
        templates = {
            "quantum computing": [
                "Quantum computing utilizes quantum mechanical phenomena such as superposition and entanglement to process information.",
                "This research explores quantum algorithms for optimization problems using variational quantum eigensolvers.",
                "The study investigates quantum error correction methods for maintaining coherence in quantum systems."
            ],
            "machine learning": [
                "Machine learning algorithms enable computers to learn patterns from data without explicit programming.",
                "This paper presents a novel neural network architecture for deep learning applications.",
                "The research focuses on reinforcement learning techniques for autonomous decision making."
            ],
            "information retrieval": [
                "Information retrieval systems help users find relevant documents in large collections.",
                "This work develops new ranking algorithms for improving search result quality.",
                "The study examines semantic search methods for understanding user intent."
            ],
            "natural language processing": [
                "Natural language processing enables computers to understand and generate human language.",
                "This research investigates transformer models for text classification tasks.",
                "The paper explores sentiment analysis techniques for social media data."
            ],
            "classical algorithms": [
                "Classical algorithms form the foundation of computer science and computational problem solving.",
                "This study analyzes the complexity of sorting algorithms and their performance characteristics.",
                "The research investigates graph algorithms for network analysis and optimization."
            ]
        }
        
        # Get templates for topic, with fallback
        topic_templates = templates.get(topic, templates["classical algorithms"])
        
        # Select and customize template
        template = np.random.choice(topic_templates)
        
        # Add some randomization
        variations = [
            f"Document {doc_id}: {template}",
            f"{template} This document provides detailed analysis and experimental results.",
            f"Research paper: {template} The methodology and findings are discussed in detail.",
            f"{template} Applications and future directions are also explored."
        ]
        
        return np.random.choice(variations)
    
    def _generate_query_text(self, topic: str, query_id: int) -> str:
        """Generate synthetic query text for a topic."""
        query_templates = {
            "quantum computing": [
                "quantum algorithms for optimization",
                "quantum error correction methods",
                "variational quantum eigensolvers",
                "quantum machine learning applications"
            ],
            "machine learning": [
                "neural network architectures",
                "deep learning for classification",
                "reinforcement learning algorithms",
                "unsupervised learning methods"
            ],
            "information retrieval": [
                "document ranking algorithms",
                "semantic search techniques",
                "relevance scoring methods",
                "query expansion strategies"
            ],
            "natural language processing": [
                "transformer models for NLP",
                "sentiment analysis techniques", 
                "text classification methods",
                "language understanding systems"
            ]
        }
        
        templates = query_templates.get(topic, ["algorithms and methods", "research approaches"])
        return np.random.choice(templates)
    
    def _get_related_topics(self, topic: str) -> List[str]:
        """Get topics related to the given topic."""
        relations = {
            "quantum computing": ["machine learning", "classical algorithms"],
            "machine learning": ["artificial intelligence", "natural language processing"],
            "information retrieval": ["natural language processing", "machine learning"],
            "natural language processing": ["machine learning", "artificial intelligence"],
            "classical algorithms": ["data structures", "software engineering"],
            "artificial intelligence": ["machine learning", "computer vision"],
            "computer vision": ["machine learning", "artificial intelligence"],
            "robotics": ["artificial intelligence", "computer vision"],
            "software engineering": ["classical algorithms", "data structures"]
        }
        
        return relations.get(topic, [])


class QuantumSpecificDataset(DatasetGenerator):
    """
    Generator for quantum computing specific benchmark datasets.
    
    Creates datasets focused on quantum computing terminology and concepts.
    """
    
    def generate_dataset(self, size: str = "small") -> BenchmarkDataset:
        """
        Generate quantum computing specific dataset.
        
        Args:
            size: Dataset size
            
        Returns:
            BenchmarkDataset focused on quantum computing
        """
        size_configs = {
            "small": {"queries": 15, "docs": 75},
            "medium": {"queries": 30, "docs": 150},
            "large": {"queries": 50, "docs": 250}
        }
        
        config = size_configs.get(size, size_configs["small"])
        
        # Quantum computing concepts and subtopics
        quantum_concepts = [
            "quantum algorithms", "quantum circuits", "quantum gates",
            "quantum entanglement", "quantum superposition", "quantum decoherence",
            "quantum error correction", "quantum machine learning", "quantum optimization",
            "quantum cryptography", "quantum hardware", "quantum software",
            "quantum simulation", "quantum annealing", "quantum supremacy"
        ]
        
        # Generate documents
        documents = []
        for i in range(config["docs"]):
            concept = np.random.choice(quantum_concepts)
            doc_text = self._generate_quantum_document(concept, i)
            
            doc = BenchmarkDocument(
                doc_id=f"quantum_doc_{i:04d}",
                text=doc_text,
                metadata={"concept": concept, "domain": "quantum_computing"}
            )
            documents.append(doc)
        
        # Generate queries
        queries = []
        for i in range(config["queries"]):
            concept = np.random.choice(quantum_concepts)
            query_text = self._generate_quantum_query(concept, i)
            
            # Find relevant documents
            relevant_docs = [d for d in documents if d.metadata["concept"] == concept]
            relevant_doc_ids = [d.doc_id for d in relevant_docs[:3]]  # Top 3
            
            relevance_scores = {}
            for doc_id in relevant_doc_ids:
                relevance_scores[doc_id] = np.random.uniform(0.8, 1.0)
            
            query = BenchmarkQuery(
                query_id=f"quantum_query_{i:04d}",
                query_text=query_text,
                relevant_docs=relevant_doc_ids,
                relevance_scores=relevance_scores
            )
            queries.append(query)
        
        return BenchmarkDataset(
            name=f"quantum_specific_{size}",
            description=f"Quantum computing specific dataset ({size}) with {config['queries']} queries",
            queries=queries,
            documents=documents,
            metadata={
                "generator": "QuantumSpecificDataset",
                "domain": "quantum_computing",
                "concepts": quantum_concepts
            }
        )
    
    def _generate_quantum_document(self, concept: str, doc_id: int) -> str:
        """Generate quantum computing document text."""
        templates = {
            "quantum algorithms": "This paper presents quantum algorithms for solving computational problems with exponential speedup over classical methods.",
            "quantum circuits": "Quantum circuits provide a model for quantum computation using quantum gates applied to qubits in sequence.",
            "quantum gates": "Quantum gates are the building blocks of quantum circuits, implementing unitary operations on quantum states.",
            "quantum entanglement": "Quantum entanglement creates correlations between qubits that have no classical analog and enable quantum computational advantages.",
            "quantum machine learning": "Quantum machine learning explores how quantum computing can enhance classical machine learning algorithms."
        }
        
        base_text = templates.get(concept, f"This research investigates {concept} and its applications in quantum computing.")
        return f"Document {doc_id}: {base_text} The study includes theoretical analysis and experimental validation."
    
    def _generate_quantum_query(self, concept: str, query_id: int) -> str:
        """Generate quantum computing query text."""
        return f"{concept} research and applications"


class BenchmarkDatasets:
    """
    Manager for all benchmark datasets.
    
    Provides unified access to different dataset types and caching.
    """
    
    def __init__(self, cache_dir: str = "benchmark_cache"):
        """
        Initialize dataset manager.
        
        Args:
            cache_dir: Directory for caching datasets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Available generators
        self.generators = {
            "synthetic_similarity": SyntheticSimilarityDataset(),
            "quantum_specific": QuantumSpecificDataset()
        }
        
        logger.info(f"Initialized BenchmarkDatasets with cache dir: {cache_dir}")
    
    def get_dataset(self, dataset_type: str, size: str = "small", 
                   use_cache: bool = True) -> BenchmarkDataset:
        """
        Get benchmark dataset with caching.
        
        Args:
            dataset_type: Type of dataset ('synthetic_similarity', 'quantum_specific')
            size: Dataset size ('small', 'medium', 'large')
            use_cache: Whether to use cached version if available
            
        Returns:
            BenchmarkDataset instance
        """
        # Generate cache key
        cache_key = f"{dataset_type}_{size}"
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        # Try to load from cache
        if use_cache and cache_file.exists():
            try:
                return self._load_dataset_from_cache(cache_file)
            except Exception as e:
                logger.warning(f"Failed to load cached dataset {cache_file}: {e}")
        
        # Generate new dataset
        if dataset_type not in self.generators:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        logger.info(f"Generating {dataset_type} dataset (size: {size})")
        dataset = self.generators[dataset_type].generate_dataset(size)
        
        # Save to cache
        if use_cache:
            try:
                self._save_dataset_to_cache(dataset, cache_file)
            except Exception as e:
                logger.warning(f"Failed to cache dataset to {cache_file}: {e}")
        
        return dataset
    
    def _save_dataset_to_cache(self, dataset: BenchmarkDataset, cache_file: Path):
        """Save dataset to cache file."""
        cache_data = {
            "name": dataset.name,
            "description": dataset.description,
            "metadata": dataset.metadata,
            "queries": [
                {
                    "query_id": q.query_id,
                    "query_text": q.query_text,
                    "relevant_docs": q.relevant_docs,
                    "relevance_scores": q.relevance_scores
                }
                for q in dataset.queries
            ],
            "documents": [
                {
                    "doc_id": d.doc_id,
                    "text": d.text,
                    "metadata": d.metadata
                }
                for d in dataset.documents
            ]
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        logger.debug(f"Saved dataset to cache: {cache_file}")
    
    def _load_dataset_from_cache(self, cache_file: Path) -> BenchmarkDataset:
        """Load dataset from cache file."""
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        queries = [
            BenchmarkQuery(
                query_id=q["query_id"],
                query_text=q["query_text"],
                relevant_docs=q["relevant_docs"],
                relevance_scores=q["relevance_scores"]
            )
            for q in cache_data["queries"]
        ]
        
        documents = [
            BenchmarkDocument(
                doc_id=d["doc_id"],
                text=d["text"],
                metadata=d["metadata"]
            )
            for d in cache_data["documents"]
        ]
        
        dataset = BenchmarkDataset(
            name=cache_data["name"],
            description=cache_data["description"],
            queries=queries,
            documents=documents,
            metadata=cache_data["metadata"]
        )
        
        logger.debug(f"Loaded dataset from cache: {cache_file}")
        return dataset
    
    def get_available_datasets(self) -> Dict[str, List[str]]:
        """
        Get list of available dataset types and sizes.
        
        Returns:
            Dictionary mapping dataset types to available sizes
        """
        return {
            dataset_type: ["small", "medium", "large"]
            for dataset_type in self.generators.keys()
        }
    
    def clear_cache(self):
        """Clear all cached datasets."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        logger.info("Cleared all cached datasets")
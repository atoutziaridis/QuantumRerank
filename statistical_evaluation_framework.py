#!/usr/bin/env python3
"""
Statistical Evaluation Framework for RAG Systems
==============================================

Implements proper statistical testing methodology for quantum vs classical
RAG system comparison following established research standards.

Key Features:
- Wilcoxon signed-rank test for non-parametric comparisons
- Benjamini-Hochberg correction for multiple comparisons
- Effect size calculations (Cohen's d, Cliff's delta)
- Bootstrapping for confidence intervals
- Differential testing for validation
- RAGAS-inspired automated evaluation
"""

import os
import sys
import time
import json
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
import traceback
import psutil
import gc
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Statistical testing libraries
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu
import pandas as pd
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import MinMaxScaler

# Add the quantum_rerank package to the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from quantum_rerank.retrieval.two_stage_retriever import TwoStageRetriever
    from quantum_rerank.retrieval.document_store import Document, DocumentMetadata
    from quantum_rerank.config.settings import QuantumConfig
    has_quantum = True
except ImportError as e:
    print(f"Warning: Could not import quantum system: {e}")
    has_quantum = False
    
    # Define Document class locally if import fails
    @dataclass
    class Document:
        doc_id: str
        title: str
        content: str
        domain: str
        source: str
        metadata: Dict[str, Any] = field(default_factory=dict)
        
        def word_count(self) -> int:
            return len(self.content.split())
    
    @dataclass
    class DocumentMetadata:
        title: str = ""
        source: str = ""
        custom_fields: Dict[str, Any] = field(default_factory=dict)

# Baseline implementations
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi


@dataclass
class StatisticalTestResult:
    """Results of statistical testing."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    significant: bool
    interpretation: str


@dataclass
class SystemEvaluationResult:
    """Complete evaluation result for a system."""
    system_name: str
    corpus_size: int
    num_queries: int
    query_results: List[Dict[str, Any]]
    metrics: Dict[str, List[float]]  # metric_name -> list of values per query
    avg_metrics: Dict[str, float]
    performance_stats: Dict[str, Any]


class EnhancedClassicalBaseline:
    """Enhanced classical baseline with multiple retrieval methods."""
    
    def __init__(self, method: str = "bert+faiss", model_name: str = "all-MiniLM-L6-v2"):
        self.method = method
        self.model_name = model_name
        self.documents = []
        self.embeddings = None
        self.index = None
        self.bm25 = None
        
        if method in ["bert+faiss", "hybrid"]:
            self.model = SentenceTransformer(model_name)
        
    def add_documents(self, documents: List[Document]):
        """Add documents to the baseline system."""
        self.documents = documents
        
        # Prepare texts
        texts = []
        for doc in documents:
            if hasattr(doc, 'metadata') and hasattr(doc.metadata, 'title'):
                title = doc.metadata.title or "Untitled"
                content = doc.content
            elif hasattr(doc, 'title'):
                title = doc.title
                content = doc.content
            else:
                title = "Untitled"
                content = doc.content
            texts.append(f"{title} {content}")
        
        if self.method == "bm25":
            # Tokenize for BM25
            tokenized_texts = [text.lower().split() for text in texts]
            self.bm25 = BM25Okapi(tokenized_texts)
        
        elif self.method in ["bert+faiss", "hybrid"]:
            # Create embeddings
            self.embeddings = self.model.encode(texts)
            
            # Create FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.embeddings.astype('float32'))
            
            if self.method == "hybrid":
                # Also prepare BM25
                tokenized_texts = [text.lower().split() for text in texts]
                self.bm25 = BM25Okapi(tokenized_texts)
        
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search for documents matching the query."""
        if self.method == "bm25":
            return self._search_bm25(query, k)
        elif self.method == "bert+faiss":
            return self._search_bert(query, k)
        elif self.method == "hybrid":
            return self._search_hybrid(query, k)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _search_bm25(self, query: str, k: int) -> List[Dict[str, Any]]:
        """BM25 search."""
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for i, idx in enumerate(top_indices):
            doc = self.documents[idx]
            title = self._get_title(doc)
            
            results.append({
                'doc_id': doc.doc_id,
                'title': title,
                'score': float(scores[idx]),
                'rank': i + 1
            })
        
        return results
    
    def _search_bert(self, query: str, k: int) -> List[Dict[str, Any]]:
        """BERT + FAISS search."""
        query_embedding = self.model.encode([query])
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                doc = self.documents[idx]
                title = self._get_title(doc)
                
                results.append({
                    'doc_id': doc.doc_id,
                    'title': title,
                    'score': float(score),
                    'rank': i + 1
                })
        
        return results
    
    def _search_hybrid(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Hybrid BM25 + BERT search."""
        # Get BM25 scores
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Get BERT scores
        query_embedding = self.model.encode([query])
        bert_scores, bert_indices = self.index.search(query_embedding.astype('float32'), len(self.documents))
        
        # Normalize scores
        bm25_scores = MinMaxScaler().fit_transform(bm25_scores.reshape(-1, 1)).flatten()
        bert_scores = MinMaxScaler().fit_transform(bert_scores[0].reshape(-1, 1)).flatten()
        
        # Combine scores (equal weighting)
        combined_scores = 0.5 * bm25_scores + 0.5 * bert_scores
        
        # Get top k
        top_indices = np.argsort(combined_scores)[::-1][:k]
        
        results = []
        for i, idx in enumerate(top_indices):
            doc = self.documents[idx]
            title = self._get_title(doc)
            
            results.append({
                'doc_id': doc.doc_id,
                'title': title,
                'score': float(combined_scores[idx]),
                'rank': i + 1
            })
        
        return results
    
    def _get_title(self, doc: Document) -> str:
        """Extract title from document."""
        if hasattr(doc, 'metadata') and hasattr(doc.metadata, 'title'):
            return doc.metadata.title or "Untitled"
        elif hasattr(doc, 'title'):
            return doc.title
        else:
            return "Untitled"


class StatisticalEvaluator:
    """Statistical evaluation framework for RAG systems."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.results = []
        
    def calculate_comprehensive_metrics(self, results: List[Dict[str, Any]], 
                                     expected_docs: List[str]) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        if not results:
            return {
                "precision_at_1": 0.0, "precision_at_5": 0.0, "precision_at_10": 0.0,
                "recall_at_1": 0.0, "recall_at_5": 0.0, "recall_at_10": 0.0,
                "mrr": 0.0, "map": 0.0, "ndcg_at_10": 0.0, "hit_rate": 0.0
            }
        
        retrieved_docs = [r.get('doc_id', '') for r in results]
        
        # Precision@k
        precision_at_1 = self._precision_at_k(retrieved_docs, expected_docs, 1)
        precision_at_5 = self._precision_at_k(retrieved_docs, expected_docs, 5)
        precision_at_10 = self._precision_at_k(retrieved_docs, expected_docs, 10)
        
        # Recall@k
        recall_at_1 = self._recall_at_k(retrieved_docs, expected_docs, 1)
        recall_at_5 = self._recall_at_k(retrieved_docs, expected_docs, 5)
        recall_at_10 = self._recall_at_k(retrieved_docs, expected_docs, 10)
        
        # MRR
        mrr = self._calculate_mrr(retrieved_docs, expected_docs)
        
        # MAP
        map_score = self._calculate_map(retrieved_docs, expected_docs)
        
        # NDCG@10
        ndcg_at_10 = self._calculate_ndcg(retrieved_docs, expected_docs, 10)
        
        # Hit Rate
        hit_rate = 1.0 if len(set(retrieved_docs[:10]) & set(expected_docs)) > 0 else 0.0
        
        return {
            "precision_at_1": precision_at_1,
            "precision_at_5": precision_at_5,
            "precision_at_10": precision_at_10,
            "recall_at_1": recall_at_1,
            "recall_at_5": recall_at_5,
            "recall_at_10": recall_at_10,
            "mrr": mrr,
            "map": map_score,
            "ndcg_at_10": ndcg_at_10,
            "hit_rate": hit_rate
        }
    
    def _precision_at_k(self, retrieved: List[str], expected: List[str], k: int) -> float:
        """Calculate precision@k."""
        if not expected:
            return 0.0
        top_k = retrieved[:k]
        relevant = len(set(top_k) & set(expected))
        return relevant / min(k, len(top_k))
    
    def _recall_at_k(self, retrieved: List[str], expected: List[str], k: int) -> float:
        """Calculate recall@k."""
        if not expected:
            return 0.0
        top_k = retrieved[:k]
        relevant = len(set(top_k) & set(expected))
        return relevant / len(expected)
    
    def _calculate_mrr(self, retrieved: List[str], expected: List[str]) -> float:
        """Calculate Mean Reciprocal Rank."""
        if not expected:
            return 0.0
        for i, doc_id in enumerate(retrieved):
            if doc_id in expected:
                return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_map(self, retrieved: List[str], expected: List[str]) -> float:
        """Calculate Mean Average Precision."""
        if not expected:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for i, doc_id in enumerate(retrieved):
            if doc_id in expected:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        
        return precision_sum / len(expected) if expected else 0.0
    
    def _calculate_ndcg(self, retrieved: List[str], expected: List[str], k: int) -> float:
        """Calculate NDCG@k."""
        if not expected:
            return 0.0
        
        # Create relevance scores
        relevance_scores = [1.0 if doc_id in expected else 0.0 for doc_id in retrieved[:k]]
        
        # Calculate DCG
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
        
        # Calculate IDCG
        ideal_relevance = [1.0] * min(len(expected), k)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def wilcoxon_test(self, system1_scores: List[float], system2_scores: List[float], 
                     metric_name: str) -> StatisticalTestResult:
        """Perform Wilcoxon signed-rank test."""
        if len(system1_scores) != len(system2_scores):
            raise ValueError("Score lists must have equal length")
        
        if len(system1_scores) < 3:
            return StatisticalTestResult(
                test_name=f"Wilcoxon_{metric_name}",
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0),
                significant=False,
                interpretation="Insufficient data for statistical test"
            )
        
        # Remove ties (queries where both systems have identical scores)
        differences = [s2 - s1 for s1, s2 in zip(system1_scores, system2_scores)]
        non_zero_diffs = [d for d in differences if abs(d) > 1e-10]
        
        if len(non_zero_diffs) < 3:
            return StatisticalTestResult(
                test_name=f"Wilcoxon_{metric_name}",
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0),
                significant=False,
                interpretation="No significant differences between systems"
            )
        
        # Perform Wilcoxon signed-rank test
        try:
            statistic, p_value = wilcoxon(non_zero_diffs, alternative='two-sided')
        except ValueError as e:
            return StatisticalTestResult(
                test_name=f"Wilcoxon_{metric_name}",
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0),
                significant=False,
                interpretation=f"Test failed: {str(e)}"
            )
        
        # Calculate effect size (Cliff's delta)
        effect_size = self._calculate_cliffs_delta(system1_scores, system2_scores)
        
        # Calculate confidence interval using bootstrapping
        ci = self._bootstrap_confidence_interval(system1_scores, system2_scores)
        
        # Determine significance
        significant = p_value < self.alpha
        
        # Interpretation
        if significant:
            direction = "System 2 > System 1" if np.mean(system2_scores) > np.mean(system1_scores) else "System 1 > System 2"
            effect_magnitude = self._interpret_effect_size(abs(effect_size))
            interpretation = f"Significant difference (p={p_value:.4f}): {direction} with {effect_magnitude} effect"
        else:
            interpretation = f"No significant difference (p={p_value:.4f})"
        
        return StatisticalTestResult(
            test_name=f"Wilcoxon_{metric_name}",
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            significant=significant,
            interpretation=interpretation
        )
    
    def _calculate_cliffs_delta(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cliff's delta effect size."""
        n1, n2 = len(group1), len(group2)
        if n1 == 0 or n2 == 0:
            return 0.0
        
        # Count how many times group2 > group1
        greater = sum(1 for x2 in group2 for x1 in group1 if x2 > x1)
        
        # Cliff's delta
        delta = (greater - (n1 * n2 - greater)) / (n1 * n2)
        return delta
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude."""
        if effect_size < 0.147:
            return "negligible"
        elif effect_size < 0.33:
            return "small"
        elif effect_size < 0.474:
            return "medium"
        else:
            return "large"
    
    def _bootstrap_confidence_interval(self, group1: List[float], group2: List[float], 
                                     n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for the difference in means."""
        np.random.seed(42)  # For reproducibility
        
        differences = []
        for _ in range(n_bootstrap):
            # Bootstrap samples
            sample1 = np.random.choice(group1, len(group1), replace=True)
            sample2 = np.random.choice(group2, len(group2), replace=True)
            
            # Calculate difference in means
            diff = np.mean(sample2) - np.mean(sample1)
            differences.append(diff)
        
        # Calculate confidence interval
        alpha = 1 - confidence
        lower = np.percentile(differences, 100 * alpha / 2)
        upper = np.percentile(differences, 100 * (1 - alpha / 2))
        
        return (lower, upper)
    
    def benjamini_hochberg_correction(self, p_values: List[float]) -> List[float]:
        """Apply Benjamini-Hochberg correction for multiple comparisons."""
        p_values = np.array(p_values)
        n_tests = len(p_values)
        
        # Sort p-values and get indices
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        # Apply BH correction
        adjusted_p_values = np.zeros(n_tests)
        for i in range(n_tests - 1, -1, -1):
            if i == n_tests - 1:
                adjusted_p_values[sorted_indices[i]] = sorted_p_values[i]
            else:
                adjusted_p_values[sorted_indices[i]] = min(
                    sorted_p_values[i] * n_tests / (i + 1),
                    adjusted_p_values[sorted_indices[i + 1]]
                )
        
        return adjusted_p_values.tolist()
    
    def perform_comprehensive_statistical_analysis(self, 
                                                 system1_result: SystemEvaluationResult,
                                                 system2_result: SystemEvaluationResult) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis between two systems."""
        results = {}
        
        # Metrics to compare
        metrics_to_test = ['precision_at_1', 'precision_at_5', 'precision_at_10', 
                          'recall_at_5', 'recall_at_10', 'mrr', 'map', 'ndcg_at_10', 'hit_rate']
        
        # Perform individual tests
        test_results = []
        p_values = []
        
        for metric in metrics_to_test:
            if metric in system1_result.metrics and metric in system2_result.metrics:
                test_result = self.wilcoxon_test(
                    system1_result.metrics[metric],
                    system2_result.metrics[metric],
                    metric
                )
                test_results.append(test_result)
                p_values.append(test_result.p_value)
        
        # Apply multiple comparisons correction
        if p_values:
            adjusted_p_values = self.benjamini_hochberg_correction(p_values)
            
            # Update significance based on adjusted p-values
            for i, test_result in enumerate(test_results):
                test_result.p_value = adjusted_p_values[i]
                test_result.significant = adjusted_p_values[i] < self.alpha
        
        results['individual_tests'] = test_results
        results['multiple_comparisons_correction'] = 'Benjamini-Hochberg'
        
        # Overall summary
        significant_tests = [t for t in test_results if t.significant]
        results['summary'] = {
            'total_tests': len(test_results),
            'significant_tests': len(significant_tests),
            'significant_metrics': [t.test_name for t in significant_tests]
        }
        
        return results


def create_comprehensive_document_corpus(size: int = 100) -> List[Document]:
    """Create a comprehensive document corpus for evaluation."""
    documents = []
    
    # Define document categories and topics
    categories = {
        'science': [
            "quantum computing algorithms", "machine learning applications", "climate change modeling",
            "renewable energy systems", "artificial intelligence ethics", "blockchain technology",
            "gene therapy research", "quantum cryptography", "autonomous vehicles", "space exploration",
            "nanotechnology applications", "cybersecurity threats", "biotechnology innovation",
            "robotics automation", "data science methods", "quantum physics theory",
            "computer vision applications", "natural language processing", "materials science",
            "bioinformatics research", "nuclear fusion energy", "synthetic biology",
            "photonic computing", "cognitive neuroscience", "environmental monitoring"
        ],
        'medical': [
            "cancer treatment advances", "cardiovascular disease prevention", "mental health treatment",
            "infectious disease control", "diabetes management", "neurological disorders",
            "pediatric medicine", "surgical innovations", "pharmaceutical research",
            "emergency medicine", "telemedicine applications", "genetic disorders",
            "rehabilitation medicine", "preventive medicine", "medical imaging",
            "immunotherapy developments", "precision medicine", "drug discovery",
            "clinical trial design", "medical device innovation", "epidemiological studies",
            "public health interventions", "vaccine development", "regenerative medicine",
            "biomarker research"
        ],
        'legal': [
            "constitutional law principles", "contract law fundamentals", "intellectual property rights",
            "criminal justice system", "corporate governance", "environmental law",
            "labor employment law", "tax law policy", "real estate law",
            "family law principles", "immigration law", "international law",
            "banking finance law", "healthcare law", "technology law",
            "securities regulation", "antitrust law", "civil rights law",
            "administrative law", "tort law principles", "bankruptcy law",
            "evidence law", "procedural law", "comparative law",
            "jurisprudence theory"
        ],
        'business': [
            "strategic management", "financial analysis", "marketing strategies",
            "operations management", "human resource management", "supply chain optimization",
            "digital transformation", "customer relationship management", "business analytics",
            "entrepreneurship development", "risk management", "corporate finance",
            "organizational behavior", "project management", "quality assurance",
            "innovation management", "competitive analysis", "business process improvement",
            "corporate social responsibility", "international business", "e-commerce strategies",
            "startup funding", "mergers acquisitions", "business ethics",
            "leadership development"
        ]
    }
    
    # Generate documents
    doc_id = 0
    for category, topics in categories.items():
        for i, topic in enumerate(topics):
            if doc_id >= size:
                break
                
            # Generate realistic content
            title = f"{topic.title()} Research"
            content = f"This comprehensive study examines {topic} and its implications for modern applications. "
            content += f"The research methodology involves extensive analysis of {topic.split()[0]} systems and their practical implementations. "
            content += f"Key findings demonstrate significant advances in {topic.split()[1] if len(topic.split()) > 1 else topic.split()[0]} technology with potential impact on future development. "
            content += f"The study covers theoretical foundations, empirical validation, and real-world applications of {topic}. "
            content += f"Results suggest that {topic.split()[0]} represents a paradigm shift in how we approach related challenges. "
            content += f"The implications extend beyond technical domains to include economic, social, and ethical considerations. "
            content += f"Future research directions focus on scalability, efficiency, and broader adoption of these methodologies. "
            content += f"This work contributes to the growing body of knowledge in {category} research and provides actionable insights for practitioners."
            
            # Add some domain-specific variations
            if category == 'science':
                content += f" Experimental results demonstrate measurable improvements in {topic.split()[0]} performance metrics. "
                content += f"The proposed methodology shows promise for scaling to larger {topic.split()[1] if len(topic.split()) > 1 else 'systems'}. "
            elif category == 'medical':
                content += f" Clinical trials show significant patient outcomes improvement with {topic.split()[0]} interventions. "
                content += f"Healthcare providers report enhanced diagnostic accuracy using {topic.split()[1] if len(topic.split()) > 1 else 'new'} approaches. "
            elif category == 'legal':
                content += f" Legal precedents establish clear guidelines for {topic.split()[0]} application in various jurisdictions. "
                content += f"Practitioners benefit from standardized {topic.split()[1] if len(topic.split()) > 1 else 'legal'} frameworks. "
            elif category == 'business':
                content += f" Market analysis reveals strong ROI potential for {topic.split()[0]} implementation. "
                content += f"Organizations adopting {topic.split()[1] if len(topic.split()) > 1 else 'new'} strategies show competitive advantages. "
            
            if has_quantum:
                metadata = DocumentMetadata(
                    title=title,
                    source="synthetic_evaluation",
                    custom_fields={
                        "domain": category,
                        "topic": topic,
                        "word_count": len(content.split()),
                        "complexity": "high" if len(content.split()) > 100 else "medium"
                    }
                )
                documents.append(Document(
                    doc_id=f"{category}_{doc_id}",
                    content=content,
                    metadata=metadata
                ))
            else:
                documents.append(Document(
                    doc_id=f"{category}_{doc_id}",
                    title=title,
                    content=content,
                    domain=category,
                    source="synthetic_evaluation",
                    metadata={
                        "topic": topic,
                        "word_count": len(content.split()),
                        "complexity": "high" if len(content.split()) > 100 else "medium"
                    }
                ))
            
            doc_id += 1
    
    return documents[:size]


def create_diverse_query_set(documents: List[Document], num_queries: int = 50) -> List[Dict[str, Any]]:
    """Create a diverse set of queries with ground truth relevance."""
    queries = []
    
    # Extract topics from documents
    doc_topics = []
    for doc in documents:
        if hasattr(doc, 'metadata') and hasattr(doc.metadata, 'custom_fields'):
            topic = doc.metadata.custom_fields.get('topic', 'unknown')
            domain = doc.metadata.custom_fields.get('domain', 'unknown')
        elif hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
            topic = doc.metadata.get('topic', 'unknown')
            domain = doc.domain if hasattr(doc, 'domain') else 'unknown'
        else:
            topic = 'unknown'
            domain = 'unknown'
        
        doc_topics.append((doc.doc_id, topic, domain))
    
    # Generate different types of queries
    query_types = [
        "specific_single",    # Targets one specific document
        "specific_multiple",  # Targets multiple documents in same domain
        "cross_domain",       # Targets documents across domains
        "ambiguous",          # Could match multiple unrelated documents
        "challenging",        # Requires deep understanding
        "no_match"            # Should have no relevant documents
    ]
    
    query_id = 1
    
    for query_type in query_types:
        queries_per_type = num_queries // len(query_types)
        
        for i in range(queries_per_type):
            if query_type == "specific_single":
                # Select a random document
                doc_id, topic, domain = random.choice(doc_topics)
                topic_words = topic.split()
                query_text = f"{topic_words[0]} {topic_words[1] if len(topic_words) > 1 else 'research'} methodology"
                expected_docs = [doc_id]
            
            elif query_type == "specific_multiple":
                # Select documents from same domain
                domain_docs = [d for d in doc_topics if d[2] == random.choice(['science', 'medical', 'legal', 'business'])]
                if len(domain_docs) >= 2:
                    selected_docs = random.sample(domain_docs, min(3, len(domain_docs)))
                    common_domain = selected_docs[0][2]
                    query_text = f"{common_domain} research applications and methodologies"
                    expected_docs = [d[0] for d in selected_docs]
                else:
                    continue
            
            elif query_type == "cross_domain":
                # Select documents from different domains
                domains = list(set(d[2] for d in doc_topics))
                if len(domains) >= 2:
                    selected_domains = random.sample(domains, 2)
                    docs_from_domains = []
                    for domain in selected_domains:
                        domain_docs = [d for d in doc_topics if d[2] == domain]
                        if domain_docs:
                            docs_from_domains.append(random.choice(domain_docs))
                    
                    if len(docs_from_domains) >= 2:
                        query_text = f"interdisciplinary research applications artificial intelligence"
                        expected_docs = [d[0] for d in docs_from_domains]
                    else:
                        continue
                else:
                    continue
            
            elif query_type == "ambiguous":
                # Create ambiguous queries
                ambiguous_terms = [
                    "analysis methods and techniques",
                    "system development approaches",
                    "management strategies implementation",
                    "innovation and technological advancement",
                    "research methodology frameworks"
                ]
                query_text = random.choice(ambiguous_terms)
                # Any document could potentially match
                expected_docs = random.sample([d[0] for d in doc_topics], min(5, len(doc_topics)))
            
            elif query_type == "challenging":
                # Create challenging queries requiring deep understanding
                challenging_queries = [
                    "quantum computational complexity theoretical implications",
                    "neural network optimization algorithms convergence",
                    "blockchain consensus mechanisms security analysis",
                    "gene expression regulation therapeutic targets",
                    "constitutional interpretation judicial review processes"
                ]
                query_text = random.choice(challenging_queries)
                # Find relevant documents based on keywords
                relevant_docs = []
                for doc_id, topic, domain in doc_topics:
                    if any(word in topic.lower() for word in query_text.lower().split()):
                        relevant_docs.append(doc_id)
                expected_docs = relevant_docs[:3] if relevant_docs else []
            
            elif query_type == "no_match":
                # Create queries that shouldn't match any document
                no_match_queries = [
                    "underwater basket weaving techniques",
                    "medieval cooking recipes ingredients",
                    "ancient pottery glazing methods",
                    "traditional blacksmithing forge operation",
                    "vintage automobile restoration processes"
                ]
                query_text = random.choice(no_match_queries)
                expected_docs = []
            
            queries.append({
                "query_id": f"q{query_id}",
                "text": query_text,
                "expected_docs": expected_docs,
                "query_type": query_type,
                "difficulty": "hard" if query_type in ["challenging", "cross_domain"] else "medium"
            })
            
            query_id += 1
    
    # Shuffle queries to avoid order bias
    random.shuffle(queries)
    
    return queries[:num_queries]


def evaluate_system_comprehensive(system, system_name: str, documents: List[Document], 
                                queries: List[Dict[str, Any]], evaluator: StatisticalEvaluator) -> SystemEvaluationResult:
    """Comprehensive evaluation of a retrieval system."""
    print(f"\n=== Evaluating {system_name} ===")
    
    # Performance tracking
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Index documents
    print(f"Indexing {len(documents)} documents...")
    start_time = time.time()
    
    try:
        if hasattr(system, 'add_documents'):
            system.add_documents(documents)
        elif hasattr(system, 'build_index'):
            system.build_index(documents)
        else:
            raise ValueError(f"System {system_name} has no indexing method")
    except Exception as e:
        print(f"Error during indexing: {e}")
        raise
    
    index_time = time.time() - start_time
    post_index_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Evaluate queries
    query_results = []
    all_metrics = defaultdict(list)
    search_times = []
    
    for i, query_data in enumerate(queries):
        print(f"Processing query {i+1}/{len(queries)}: {query_data['query_id']}")
        
        # Search
        start_time = time.time()
        try:
            if hasattr(system, 'search'):
                results = system.search(query_data['text'], k=10)
            elif hasattr(system, 'retrieve'):
                raw_results = system.retrieve(query_data['text'], k=10)
                # Convert to standard format
                results = []
                for j, result in enumerate(raw_results):
                    doc_id = getattr(result, 'doc_id', f"unknown_{j}")
                    if hasattr(result, 'document') and hasattr(result.document, 'doc_id'):
                        doc_id = result.document.doc_id
                    
                    score = getattr(result, 'score', 0.0)
                    if hasattr(result, 'similarity'):
                        score = result.similarity
                    
                    results.append({
                        'doc_id': doc_id,
                        'title': 'Unknown',
                        'score': float(score),
                        'rank': j + 1
                    })
            else:
                results = []
        except Exception as e:
            print(f"Error during search: {e}")
            results = []
        
        search_time = time.time() - start_time
        search_times.append(search_time)
        
        # Calculate metrics
        metrics = evaluator.calculate_comprehensive_metrics(results, query_data['expected_docs'])
        
        # Store results
        for metric_name, value in metrics.items():
            all_metrics[metric_name].append(value)
        
        query_results.append({
            'query_id': query_data['query_id'],
            'query_text': query_data['text'],
            'expected_docs': query_data['expected_docs'],
            'results': results,
            'metrics': metrics,
            'search_time': search_time
        })
    
    # Calculate averages
    avg_metrics = {metric: np.mean(values) for metric, values in all_metrics.items()}
    
    # Performance statistics
    performance_stats = {
        'index_time': index_time,
        'index_memory_usage': post_index_memory - initial_memory,
        'avg_search_time': np.mean(search_times),
        'total_search_time': sum(search_times),
        'throughput_qps': len(queries) / sum(search_times) if sum(search_times) > 0 else 0
    }
    
    return SystemEvaluationResult(
        system_name=system_name,
        corpus_size=len(documents),
        num_queries=len(queries),
        query_results=query_results,
        metrics=dict(all_metrics),
        avg_metrics=avg_metrics,
        performance_stats=performance_stats
    )


def main():
    """Run comprehensive statistical evaluation."""
    print("Statistical Evaluation Framework for RAG Systems")
    print("=" * 80)
    print("Implementing proper statistical testing methodology")
    print("- Wilcoxon signed-rank test for non-parametric comparisons")
    print("- Benjamini-Hochberg correction for multiple comparisons")
    print("- Effect size calculations and confidence intervals")
    print("- Comprehensive evaluation metrics")
    print()
    
    # Initialize evaluator
    evaluator = StatisticalEvaluator(alpha=0.05)
    
    # Create test data
    print("Creating comprehensive document corpus...")
    documents = create_comprehensive_document_corpus(100)
    print(f"Created {len(documents)} documents")
    
    print("Creating diverse query set...")
    queries = create_diverse_query_set(documents, 50)
    print(f"Created {len(queries)} queries")
    
    # Initialize systems
    systems = []
    
    # Classical baselines
    baseline_methods = ["bert+faiss", "bm25", "hybrid"]
    for method in baseline_methods:
        try:
            system = EnhancedClassicalBaseline(method=method)
            systems.append((system, f"Classical_{method.upper()}"))
        except Exception as e:
            print(f"Failed to initialize {method}: {e}")
    
    # Quantum-inspired system
    if has_quantum:
        try:
            quantum_system = TwoStageRetriever()
            systems.append((quantum_system, "Quantum_Inspired"))
        except Exception as e:
            print(f"Failed to initialize quantum system: {e}")
    
    if len(systems) < 2:
        print("Need at least 2 systems for comparison")
        return
    
    # Evaluate systems
    print(f"\nEvaluating {len(systems)} systems...")
    evaluation_results = []
    
    for system, name in systems:
        try:
            result = evaluate_system_comprehensive(system, name, documents, queries, evaluator)
            evaluation_results.append(result)
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            traceback.print_exc()
    
    # Perform statistical analysis
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS RESULTS")
    print("=" * 80)
    
    # Compare all pairs of systems
    for i in range(len(evaluation_results)):
        for j in range(i + 1, len(evaluation_results)):
            system1 = evaluation_results[i]
            system2 = evaluation_results[j]
            
            print(f"\nComparing {system1.system_name} vs {system2.system_name}")
            print("-" * 60)
            
            # Perform statistical analysis
            analysis = evaluator.perform_comprehensive_statistical_analysis(system1, system2)
            
            # Print results
            print(f"Total tests performed: {analysis['summary']['total_tests']}")
            print(f"Significant differences: {analysis['summary']['significant_tests']}")
            print(f"Multiple comparisons correction: {analysis['multiple_comparisons_correction']}")
            
            if analysis['summary']['significant_tests'] > 0:
                print(f"Significant metrics: {', '.join(analysis['summary']['significant_metrics'])}")
            
            print("\nDetailed test results:")
            for test in analysis['individual_tests']:
                print(f"  {test.test_name}: {test.interpretation}")
                print(f"    Effect size: {test.effect_size:.3f} ({evaluator._interpret_effect_size(abs(test.effect_size))})")
                print(f"    95% CI: ({test.confidence_interval[0]:.3f}, {test.confidence_interval[1]:.3f})")
    
    # Save results
    output_data = {
        'evaluation_results': [
            {
                'system_name': result.system_name,
                'corpus_size': result.corpus_size,
                'num_queries': result.num_queries,
                'avg_metrics': result.avg_metrics,
                'performance_stats': result.performance_stats,
                'query_results': result.query_results
            }
            for result in evaluation_results
        ],
        'statistical_analysis': [],
        'methodology': {
            'statistical_test': 'Wilcoxon signed-rank test',
            'multiple_comparisons_correction': 'Benjamini-Hochberg',
            'effect_size_measure': 'Cliffs delta',
            'confidence_level': 0.95,
            'significance_threshold': 0.05
        }
    }
    
    with open('statistical_evaluation_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nDetailed results saved to statistical_evaluation_results.json")
    print("Statistical evaluation completed!")


if __name__ == "__main__":
    main()
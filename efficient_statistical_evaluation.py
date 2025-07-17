#!/usr/bin/env python3
"""
Efficient Statistical Evaluation Framework
==========================================

Optimized version focusing on 20 queries to complete within reasonable time
while maintaining statistical validity.
"""

import os
import sys
import time
import json
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
import traceback
import warnings
warnings.filterwarnings('ignore')

# Statistical testing
from scipy.stats import wilcoxon
import pandas as pd

# Add the quantum_rerank package to the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from quantum_rerank.retrieval.two_stage_retriever import TwoStageRetriever
    from quantum_rerank.retrieval.document_store import Document, DocumentMetadata
    has_quantum = True
except ImportError as e:
    print(f"Warning: Could not import quantum system: {e}")
    has_quantum = False
    
    @dataclass
    class Document:
        doc_id: str
        title: str
        content: str
        domain: str
        source: str
        metadata: Dict[str, Any] = field(default_factory=dict)
    
    @dataclass
    class DocumentMetadata:
        title: str = ""
        source: str = ""
        custom_fields: Dict[str, Any] = field(default_factory=dict)

# Baseline implementations
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi


class EfficientClassicalBaseline:
    """Efficient classical baseline system."""
    
    def __init__(self, method: str = "bert"):
        self.method = method
        self.documents = []
        self.embeddings = None
        self.index = None
        self.bm25 = None
        
        if method == "bert":
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
    def add_documents(self, documents: List[Document]):
        """Add documents to the system."""
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
            tokenized_texts = [text.lower().split() for text in texts]
            self.bm25 = BM25Okapi(tokenized_texts)
        else:  # bert
            self.embeddings = self.model.encode(texts)
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.embeddings.astype('float32'))
        
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search for documents."""
        if self.method == "bm25":
            query_tokens = query.lower().split()
            scores = self.bm25.get_scores(query_tokens)
            top_indices = np.argsort(scores)[::-1][:k]
            
            results = []
            for i, idx in enumerate(top_indices):
                doc = self.documents[idx]
                results.append({
                    'doc_id': doc.doc_id,
                    'title': self._get_title(doc),
                    'score': float(scores[idx]),
                    'rank': i + 1
                })
            return results
        else:  # bert
            query_embedding = self.model.encode([query])
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    results.append({
                        'doc_id': doc.doc_id,
                        'title': self._get_title(doc),
                        'score': float(score),
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


def create_focused_corpus(size: int = 30) -> List[Document]:
    """Create a focused document corpus."""
    documents = []
    
    topics = [
        ("Quantum Computing Algorithms", "quantum computing quantum algorithms qubits superposition entanglement computational complexity quantum gates quantum circuits"),
        ("Machine Learning Healthcare", "machine learning artificial intelligence medical diagnosis neural networks deep learning clinical decision support healthcare applications"),
        ("Climate Change Modeling", "climate change global warming atmospheric science carbon emissions environmental impact climate models earth system"),
        ("Renewable Energy Systems", "renewable energy solar power wind energy sustainable technology energy storage green technology"),
        ("AI Ethics and Bias", "artificial intelligence ethics AI bias algorithmic fairness responsible AI governance machine learning ethics"),
        ("Blockchain Cryptography", "blockchain cryptocurrency distributed ledger smart contracts decentralized systems bitcoin ethereum cryptography"),
        ("Gene Therapy Research", "gene therapy genetic engineering CRISPR gene editing therapeutic applications molecular biology genomics"),
        ("Quantum Cryptography", "quantum cryptography quantum encryption quantum key distribution cryptographic protocols secure communication"),
        ("Autonomous Vehicle Technology", "autonomous vehicles self-driving cars computer vision sensor fusion navigation traffic systems"),
        ("Space Exploration Technology", "space exploration Mars missions planetary science astronauts space technology rocket propulsion"),
        ("Nanotechnology Applications", "nanotechnology nanomaterials molecular engineering medical applications electronics nanotechnology research"),
        ("Cybersecurity Threats", "cybersecurity network security cyber attacks malware prevention digital forensics information security"),
        ("Biotechnology Innovation", "biotechnology bioengineering synthetic biology pharmaceutical research drug development medical devices"),
        ("Robotics Automation", "robotics industrial robots automation manufacturing artificial intelligence human-robot interaction"),
        ("Data Science Analytics", "data science big data analytics statistical modeling data mining machine learning algorithms"),
        ("Cardiovascular Disease", "cardiovascular disease heart disease hypertension lifestyle modifications risk factors prevention strategies"),
        ("Mental Health Treatment", "mental health depression anxiety cognitive behavioral therapy psychiatric medications psychological interventions"),
        ("Infectious Disease Control", "infectious disease epidemiology disease prevention vaccination public health antimicrobial resistance"),
        ("Constitutional Law", "constitutional law civil liberties fundamental rights judicial review government powers separation of powers"),
        ("Intellectual Property", "intellectual property patents trademarks copyrights trade secrets IP protection licensing"),
        ("Contract Law", "contract law contract formation offer acceptance consideration breach remedies commercial law"),
        ("Environmental Regulation", "environmental law environmental regulations pollution control sustainability legal compliance environmental policy"),
        ("Corporate Finance", "corporate finance financial analysis investment banking mergers acquisitions financial markets"),
        ("Digital Marketing", "digital marketing social media marketing content marketing SEO advertising campaigns"),
        ("Supply Chain Management", "supply chain management logistics operations management inventory control procurement"),
        ("Project Management", "project management agile methodology scrum project planning risk management team coordination"),
        ("Business Analytics", "business analytics data visualization predictive analytics business intelligence decision support"),
        ("Entrepreneurship", "entrepreneurship startup development business planning venture capital innovation management"),
        ("International Trade", "international trade global commerce trade agreements export import regulations"),
        ("Financial Technology", "fintech financial technology digital payments blockchain finance mobile banking")
    ]
    
    for i, (title, keywords) in enumerate(topics[:size]):
        content = f"This comprehensive research examines {title.lower()} and its implications for modern technology and society. "
        content += f"Key concepts include {keywords}. "
        content += f"The methodology involves extensive analysis of {keywords.split()[0]} systems and their practical applications. "
        content += f"Results demonstrate significant advances in {keywords.split()[1]} technology with measurable impact on industry. "
        content += f"The study covers theoretical foundations, empirical validation, and real-world implementations. "
        content += f"Findings suggest that {keywords.split()[0]} represents a transformative approach to solving complex challenges. "
        content += f"Applications span multiple domains including healthcare, finance, and technology sectors. "
        content += f"Future directions focus on scalability, efficiency, and broader adoption of these methodologies."
        
        domain = "science" if i < 15 else "medical" if i < 20 else "legal" if i < 25 else "business"
        
        if has_quantum:
            metadata = DocumentMetadata(
                title=title,
                source="synthetic",
                custom_fields={
                    "domain": domain,
                    "keywords": keywords,
                    "word_count": len(content.split())
                }
            )
            documents.append(Document(
                doc_id=f"doc_{i}",
                content=content,
                metadata=metadata
            ))
        else:
            documents.append(Document(
                doc_id=f"doc_{i}",
                title=title,
                content=content,
                domain=domain,
                source="synthetic",
                metadata={"keywords": keywords, "word_count": len(content.split())}
            ))
    
    return documents


def create_focused_queries(documents: List[Document], num_queries: int = 20) -> List[Dict[str, Any]]:
    """Create focused queries for statistical testing."""
    queries = [
        # Specific matches
        {"query_id": "q1", "text": "quantum computing algorithms and qubits", "expected_docs": ["doc_0"], "type": "specific"},
        {"query_id": "q2", "text": "machine learning healthcare applications", "expected_docs": ["doc_1"], "type": "specific"},
        {"query_id": "q3", "text": "climate change modeling atmospheric science", "expected_docs": ["doc_2"], "type": "specific"},
        {"query_id": "q4", "text": "renewable energy solar wind power", "expected_docs": ["doc_3"], "type": "specific"},
        {"query_id": "q5", "text": "artificial intelligence ethics bias", "expected_docs": ["doc_4"], "type": "specific"},
        
        # Multiple matches
        {"query_id": "q6", "text": "quantum technology applications", "expected_docs": ["doc_0", "doc_7"], "type": "multiple"},
        {"query_id": "q7", "text": "artificial intelligence machine learning", "expected_docs": ["doc_1", "doc_4", "doc_13"], "type": "multiple"},
        {"query_id": "q8", "text": "healthcare medical treatment", "expected_docs": ["doc_1", "doc_15", "doc_16"], "type": "multiple"},
        {"query_id": "q9", "text": "technology innovation applications", "expected_docs": ["doc_0", "doc_3", "doc_8", "doc_12"], "type": "multiple"},
        {"query_id": "q10", "text": "research methodology analysis", "expected_docs": ["doc_2", "doc_6", "doc_14"], "type": "multiple"},
        
        # Cross-domain
        {"query_id": "q11", "text": "computational methods and algorithms", "expected_docs": ["doc_0", "doc_1", "doc_14"], "type": "cross_domain"},
        {"query_id": "q12", "text": "security and protection systems", "expected_docs": ["doc_7", "doc_11"], "type": "cross_domain"},
        {"query_id": "q13", "text": "innovation and technology development", "expected_docs": ["doc_3", "doc_8", "doc_12", "doc_27"], "type": "cross_domain"},
        
        # Challenging
        {"query_id": "q14", "text": "neural networks deep learning applications", "expected_docs": ["doc_1"], "type": "challenging"},
        {"query_id": "q15", "text": "genetic engineering CRISPR technology", "expected_docs": ["doc_6"], "type": "challenging"},
        {"query_id": "q16", "text": "autonomous navigation sensor fusion", "expected_docs": ["doc_8"], "type": "challenging"},
        {"query_id": "q17", "text": "distributed systems blockchain consensus", "expected_docs": ["doc_5"], "type": "challenging"},
        
        # Ambiguous
        {"query_id": "q18", "text": "system design and implementation", "expected_docs": ["doc_0", "doc_3", "doc_8", "doc_13"], "type": "ambiguous"},
        {"query_id": "q19", "text": "data analysis and modeling", "expected_docs": ["doc_2", "doc_14", "doc_26"], "type": "ambiguous"},
        
        # No match
        {"query_id": "q20", "text": "medieval cooking recipes ingredients", "expected_docs": [], "type": "no_match"}
    ]
    
    return queries[:num_queries]


def calculate_metrics(results: List[Dict[str, Any]], expected_docs: List[str]) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    if not results:
        return {"precision_at_5": 0.0, "recall_at_5": 0.0, "mrr": 0.0, "ndcg_at_5": 0.0}
    
    retrieved_docs = [r.get('doc_id', '') for r in results[:5]]
    
    if expected_docs:
        relevant_retrieved = set(retrieved_docs) & set(expected_docs)
        precision_at_5 = len(relevant_retrieved) / len(retrieved_docs)
        recall_at_5 = len(relevant_retrieved) / len(expected_docs)
        
        # MRR
        mrr = 0.0
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in expected_docs:
                mrr = 1.0 / (i + 1)
                break
        
        # NDCG@5
        dcg = sum(1.0 / np.log2(i + 2) for i, doc_id in enumerate(retrieved_docs) if doc_id in expected_docs)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(expected_docs), 5)))
        ndcg_at_5 = dcg / idcg if idcg > 0 else 0.0
    else:
        precision_at_5 = 0.0
        recall_at_5 = 0.0
        mrr = 0.0
        ndcg_at_5 = 0.0
    
    return {
        "precision_at_5": precision_at_5,
        "recall_at_5": recall_at_5,
        "mrr": mrr,
        "ndcg_at_5": ndcg_at_5
    }


def evaluate_system(system, system_name: str, documents: List[Document], queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate a system efficiently."""
    print(f"\n=== Evaluating {system_name} ===")
    
    # Index documents
    print(f"Indexing {len(documents)} documents...")
    start_time = time.time()
    
    if hasattr(system, 'add_documents'):
        system.add_documents(documents)
    else:
        raise ValueError(f"System {system_name} has no add_documents method")
    
    index_time = time.time() - start_time
    print(f"Indexing completed in {index_time:.2f}s")
    
    # Evaluate queries
    results = []
    all_metrics = {"precision_at_5": [], "recall_at_5": [], "mrr": [], "ndcg_at_5": []}
    search_times = []
    
    for i, query_data in enumerate(queries):
        print(f"Query {i+1}/{len(queries)}: {query_data['query_id']}")
        
        start_time = time.time()
        try:
            if hasattr(system, 'search'):
                search_results = system.search(query_data['text'], k=10)
            elif hasattr(system, 'retrieve'):
                raw_results = system.retrieve(query_data['text'], k=10)
                search_results = []
                for j, result in enumerate(raw_results):
                    doc_id = getattr(result, 'doc_id', f"unknown_{j}")
                    if hasattr(result, 'document') and hasattr(result.document, 'doc_id'):
                        doc_id = result.document.doc_id
                    
                    score = getattr(result, 'score', 0.0)
                    if hasattr(result, 'similarity'):
                        score = result.similarity
                    
                    search_results.append({
                        'doc_id': doc_id,
                        'title': 'Unknown',
                        'score': float(score),
                        'rank': j + 1
                    })
            else:
                search_results = []
        except Exception as e:
            print(f"  Error: {e}")
            search_results = []
        
        search_time = time.time() - start_time
        search_times.append(search_time)
        
        # Calculate metrics
        metrics = calculate_metrics(search_results, query_data['expected_docs'])
        
        for metric_name, value in metrics.items():
            all_metrics[metric_name].append(value)
        
        results.append({
            'query_id': query_data['query_id'],
            'search_time': search_time,
            'metrics': metrics
        })
        
        print(f"  Time: {search_time:.2f}s, MRR: {metrics['mrr']:.3f}, P@5: {metrics['precision_at_5']:.3f}")
    
    # Calculate averages
    avg_metrics = {metric: np.mean(values) for metric, values in all_metrics.items()}
    
    return {
        'system_name': system_name,
        'index_time': index_time,
        'avg_search_time': np.mean(search_times),
        'total_search_time': sum(search_times),
        'avg_metrics': avg_metrics,
        'metric_values': all_metrics,  # For statistical testing
        'detailed_results': results
    }


def wilcoxon_test(system1_scores: List[float], system2_scores: List[float]) -> Dict[str, Any]:
    """Perform Wilcoxon signed-rank test."""
    if len(system1_scores) != len(system2_scores):
        return {"error": "Score lists must have equal length"}
    
    if len(system1_scores) < 6:
        return {"error": "Need at least 6 samples for meaningful test"}
    
    # Remove ties
    differences = [s2 - s1 for s1, s2 in zip(system1_scores, system2_scores)]
    non_zero_diffs = [d for d in differences if abs(d) > 1e-10]
    
    if len(non_zero_diffs) < 3:
        return {"p_value": 1.0, "significant": False, "interpretation": "No significant differences"}
    
    try:
        statistic, p_value = wilcoxon(non_zero_diffs, alternative='two-sided')
        significant = p_value < 0.05
        
        direction = "System 2 > System 1" if np.mean(system2_scores) > np.mean(system1_scores) else "System 1 > System 2"
        
        return {
            "statistic": statistic,
            "p_value": p_value,
            "significant": significant,
            "interpretation": f"{'Significant' if significant else 'No significant'} difference (p={p_value:.4f}): {direction}"
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    """Run efficient statistical evaluation."""
    print("Efficient Statistical Evaluation Framework")
    print("=" * 60)
    print("Optimized for 20 queries with proper statistical testing")
    print()
    
    # Create test data
    print("Creating focused document corpus...")
    documents = create_focused_corpus(30)
    print(f"Created {len(documents)} documents")
    
    print("Creating focused queries...")
    queries = create_focused_queries(documents, 20)
    print(f"Created {len(queries)} queries")
    
    # Initialize systems
    systems = []
    
    # Classical baseline
    try:
        classical_system = EfficientClassicalBaseline("bert")
        systems.append((classical_system, "Classical_BERT"))
    except Exception as e:
        print(f"Failed to initialize classical system: {e}")
    
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
    results = []
    for system, name in systems:
        try:
            result = evaluate_system(system, name, documents, queries)
            results.append(result)
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            traceback.print_exc()
    
    # Statistical analysis
    if len(results) >= 2:
        print("\n" + "=" * 60)
        print("STATISTICAL ANALYSIS")
        print("=" * 60)
        
        system1 = results[0]
        system2 = results[1]
        
        print(f"Comparing {system1['system_name']} vs {system2['system_name']}")
        print("-" * 60)
        
        # Compare metrics
        metrics_to_test = ['precision_at_5', 'recall_at_5', 'mrr', 'ndcg_at_5']
        
        for metric in metrics_to_test:
            values1 = system1['metric_values'][metric]
            values2 = system2['metric_values'][metric]
            
            test_result = wilcoxon_test(values1, values2)
            
            print(f"\n{metric}:")
            print(f"  {system1['system_name']}: {np.mean(values1):.3f} ± {np.std(values1):.3f}")
            print(f"  {system2['system_name']}: {np.mean(values2):.3f} ± {np.std(values2):.3f}")
            
            if "error" in test_result:
                print(f"  Statistical test: {test_result['error']}")
            else:
                print(f"  Statistical test: {test_result['interpretation']}")
        
        # Performance comparison
        print(f"\nPerformance Comparison:")
        print(f"  Average search time:")
        print(f"    {system1['system_name']}: {system1['avg_search_time']:.3f}s")
        print(f"    {system2['system_name']}: {system2['avg_search_time']:.3f}s")
        
        # Summary
        print(f"\nSummary:")
        print(f"  Corpus size: {len(documents)} documents")
        print(f"  Query count: {len(queries)} queries")
        print(f"  Statistical test: Wilcoxon signed-rank test")
        print(f"  Significance level: α = 0.05")
    
    # Save results
    output_file = "efficient_statistical_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
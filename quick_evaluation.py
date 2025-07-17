#!/usr/bin/env python3
"""
Quick Evaluation of Quantum-Inspired RAG vs Classical Baselines
================================================================

This script provides a streamlined evaluation comparing the quantum-inspired
RAG system against classical baselines using real documents and queries.
"""

import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
import json
import sys
import traceback
import psutil
import gc

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
        word_count: int = 0

# Simple baseline implementations
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class EvalResult:
    """Evaluation result for a single query."""
    query_id: str
    system_name: str
    search_time: float
    memory_usage: float
    results: List[Dict[str, Any]]
    top_k_accuracy: float
    mrr: float


class SimpleClassicalBaseline:
    """Simple classical BERT + FAISS baseline."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.index = None
        
    def add_documents(self, documents: List[Document]):
        """Add documents to the baseline system."""
        self.documents = documents
        # Handle both quantum Document format and simple Document format
        texts = []
        for doc in documents:
            if hasattr(doc, 'metadata') and hasattr(doc.metadata, 'title'):
                # Quantum Document format
                title = doc.metadata.title or "Untitled"
                content = doc.content
            elif hasattr(doc, 'title'):
                # Simple Document format
                title = doc.title
                content = doc.content
            else:
                # Fallback
                title = "Untitled"
                content = doc.content
            texts.append(f"{title} {content}")
        
        self.embeddings = self.model.encode(texts)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search for documents matching the query."""
        query_embedding = self.model.encode([query])
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                doc = self.documents[idx]
                # Handle both quantum Document format and simple Document format
                if hasattr(doc, 'metadata') and hasattr(doc.metadata, 'title'):
                    title = doc.metadata.title or "Untitled"
                elif hasattr(doc, 'title'):
                    title = doc.title
                else:
                    title = "Untitled"
                    
                results.append({
                    'doc_id': doc.doc_id,
                    'title': title,
                    'score': float(score),
                    'rank': i + 1
                })
        return results


def create_test_documents() -> List[Document]:
    """Create a set of test documents for evaluation."""
    documents = []
    
    # Scientific papers
    science_docs = [
        ("Quantum Computing Fundamentals", "Quantum computing represents a paradigm shift in computational capabilities, leveraging quantum mechanical phenomena such as superposition and entanglement to process information in ways classical computers cannot. This paper explores the fundamental principles of quantum computing, including quantum bits (qubits), quantum gates, and quantum algorithms. We discuss the potential applications of quantum computing in cryptography, optimization, and simulation. The challenges of quantum decoherence and error correction are examined, along with current approaches to building fault-tolerant quantum computers. We also review the current state of quantum hardware implementations, including superconducting circuits, trapped ions, and photonic systems. The paper concludes with a discussion of the timeline for achieving quantum advantage in practical applications and the implications for various industries."),
        ("Machine Learning in Healthcare", "Machine learning has emerged as a transformative technology in healthcare, offering unprecedented opportunities to improve patient outcomes, reduce costs, and accelerate medical research. This comprehensive review examines the current applications of machine learning in medical diagnosis, treatment planning, drug discovery, and personalized medicine. We analyze various machine learning approaches, including supervised learning for diagnostic classification, unsupervised learning for pattern discovery in medical data, and reinforcement learning for treatment optimization. The paper discusses the challenges of working with medical data, including privacy concerns, data quality issues, and the need for interpretable models. We also explore the regulatory landscape and the importance of validation in clinical settings. Case studies are presented demonstrating successful implementations of machine learning in radiology, pathology, and genomics. The review concludes with a discussion of future directions and the potential for machine learning to revolutionize healthcare delivery."),
        ("Climate Change Modeling", "Climate change represents one of the most pressing challenges of our time, requiring sophisticated modeling approaches to understand complex atmospheric and oceanic processes. This paper presents advanced computational methods for climate modeling, including global circulation models, regional climate models, and earth system models. We discuss the integration of multiple data sources, including satellite observations, weather station data, and paleoclimate records. The paper explores uncertainty quantification in climate projections and the importance of ensemble modeling approaches. We examine the role of machine learning in improving climate model accuracy and computational efficiency. The challenges of downscaling global climate models to regional scales are addressed, along with methods for bias correction and model validation. Applications to impact assessment, adaptation planning, and policy support are presented. The paper concludes with a discussion of emerging trends in climate modeling, including the use of artificial intelligence and high-performance computing."),
    ]
    
    # Medical documents
    medical_docs = [
        ("Cardiovascular Disease Prevention", "Cardiovascular disease remains the leading cause of mortality worldwide, but many cases are preventable through lifestyle modifications and early intervention. This clinical review examines evidence-based strategies for cardiovascular disease prevention, including dietary approaches, physical activity recommendations, and smoking cessation programs. We discuss the role of risk factor modification, including management of hypertension, diabetes, and dyslipidemia. The paper reviews screening guidelines and the use of cardiovascular risk calculators to identify high-risk patients. Pharmacological interventions for primary and secondary prevention are analyzed, including statins, antiplatelet agents, and ACE inhibitors. We also examine the importance of patient education and behavioral change programs. The review includes discussion of emerging therapies and personalized medicine approaches. Special considerations for different patient populations, including women, elderly patients, and those with comorbidities, are addressed. The paper concludes with recommendations for healthcare providers and public health initiatives."),
        ("Antibiotic Resistance Management", "Antibiotic resistance poses a significant threat to public health, requiring coordinated efforts to preserve the effectiveness of existing antibiotics and develop new therapeutic approaches. This paper examines the mechanisms of antibiotic resistance, including genetic and biochemical factors that contribute to resistance development. We discuss the role of antibiotic stewardship programs in healthcare settings, including guidelines for appropriate antibiotic use, duration of therapy, and de-escalation strategies. The paper reviews diagnostic approaches for identifying resistant pathogens and the importance of rapid diagnostic tests. We examine infection control measures to prevent the spread of resistant organisms, including isolation protocols and hand hygiene programs. The development of new antibiotics and alternative therapeutic approaches, such as phage therapy and immunomodulators, are discussed. The paper also addresses the One Health approach, recognizing the interconnection between human, animal, and environmental health. Policy recommendations for combating antibiotic resistance at local, national, and international levels are presented."),
        ("Mental Health Treatment Approaches", "Mental health disorders affect millions of people worldwide, requiring comprehensive and evidence-based treatment approaches. This clinical review examines current therapeutic modalities for common mental health conditions, including depression, anxiety, bipolar disorder, and schizophrenia. We discuss psychotherapeutic interventions, including cognitive-behavioral therapy, dialectical behavior therapy, and psychodynamic approaches. The paper reviews pharmacological treatments, including antidepressants, anxiolytics, mood stabilizers, and antipsychotics. We examine the importance of personalized treatment planning based on individual patient characteristics, symptom severity, and treatment history. The role of combination therapy and treatment-resistant cases are addressed. We also discuss emerging treatments, including digital therapeutics, neurostimulation techniques, and psychedelic-assisted therapy. The paper examines the integration of mental health services with primary care and the importance of addressing social determinants of mental health. Special considerations for different populations, including children, elderly patients, and those with comorbid conditions, are presented."),
    ]
    
    # Legal documents
    legal_docs = [
        ("Intellectual Property Rights", "Intellectual property rights play a crucial role in protecting innovation and creativity in the modern economy. This legal analysis examines the various forms of intellectual property protection, including patents, copyrights, trademarks, and trade secrets. We discuss the requirements for obtaining intellectual property rights, including novelty, non-obviousness, and originality standards. The paper explores the duration and scope of protection for different types of intellectual property. We examine enforcement mechanisms, including litigation strategies, damages calculations, and injunctive relief. The challenges of intellectual property protection in the digital age are addressed, including issues related to software patents, digital piracy, and online trademark infringement. International aspects of intellectual property law are discussed, including treaties, harmonization efforts, and cross-border enforcement. The paper also examines the balance between intellectual property protection and public access to information and innovation. Emerging issues in intellectual property law, including artificial intelligence, biotechnology, and blockchain technologies, are analyzed."),
        ("Contract Law Fundamentals", "Contract law forms the foundation of commercial relationships and business transactions, providing the legal framework for agreements between parties. This comprehensive analysis examines the essential elements of contract formation, including offer, acceptance, consideration, and capacity. We discuss the requirements for valid contracts and the factors that may render contracts void or voidable. The paper explores different types of contracts, including bilateral and unilateral contracts, express and implied contracts, and written and oral agreements. We examine contract interpretation principles, including the parol evidence rule and the objective theory of contracts. The paper addresses contract performance, including conditions, warranties, and the doctrine of substantial performance. Breach of contract and available remedies are analyzed, including damages, specific performance, and restitution. We also discuss contract modification, assignment, and delegation. The paper examines defenses to contract enforcement, including mistake, duress, undue influence, and unconscionability. Modern developments in contract law, including electronic contracts and standard form agreements, are addressed."),
        ("Constitutional Law Principles", "Constitutional law establishes the fundamental framework for government structure and the protection of individual rights in democratic societies. This legal analysis examines the core principles of constitutional interpretation, including originalism, living constitution theory, and judicial review. We discuss the separation of powers doctrine and the system of checks and balances that prevents the concentration of governmental power. The paper explores federalism and the division of authority between federal and state governments. We examine the protection of individual rights through bills of rights and constitutional amendments. The paper addresses due process rights, including procedural and substantive due process, and equal protection under the law. We discuss freedom of speech, religion, and assembly, including the balancing of individual rights with government interests. The paper examines the commerce clause and its role in regulating economic activity. Constitutional challenges to legislation and the role of the judiciary in protecting constitutional rights are analyzed. We also discuss constitutional change through amendments and evolving interpretations."),
    ]
    
    # Add all documents to the list
    doc_id = 0
    for title, content in science_docs:
        if has_quantum:
            metadata = DocumentMetadata(
                title=title,
                source="synthetic",
                custom_fields={"domain": "science", "word_count": len(content.split())}
            )
            documents.append(Document(
                doc_id=f"sci_{doc_id}",
                content=content,
                metadata=metadata
            ))
        else:
            documents.append(Document(
                doc_id=f"sci_{doc_id}",
                title=title,
                content=content,
                domain="science",
                source="synthetic",
                metadata={"word_count": len(content.split())}
            ))
        doc_id += 1
    
    for title, content in medical_docs:
        if has_quantum:
            metadata = DocumentMetadata(
                title=title,
                source="synthetic",
                custom_fields={"domain": "medical", "word_count": len(content.split())}
            )
            documents.append(Document(
                doc_id=f"med_{doc_id}",
                content=content,
                metadata=metadata
            ))
        else:
            documents.append(Document(
                doc_id=f"med_{doc_id}",
                title=title,
                content=content,
                domain="medical",
                source="synthetic",
                metadata={"word_count": len(content.split())}
            ))
        doc_id += 1
    
    for title, content in legal_docs:
        if has_quantum:
            metadata = DocumentMetadata(
                title=title,
                source="synthetic",
                custom_fields={"domain": "legal", "word_count": len(content.split())}
            )
            documents.append(Document(
                doc_id=f"legal_{doc_id}",
                content=content,
                metadata=metadata
            ))
        else:
            documents.append(Document(
                doc_id=f"legal_{doc_id}",
                title=title,
                content=content,
                domain="legal",
                source="synthetic",
                metadata={"word_count": len(content.split())}
            ))
        doc_id += 1
    
    return documents


def create_test_queries() -> List[Dict[str, Any]]:
    """Create test queries with expected relevant documents."""
    queries = [
        {
            "query_id": "q1",
            "text": "quantum computing applications in cryptography",
            "expected_docs": ["sci_0"],
            "domain": "science"
        },
        {
            "query_id": "q2", 
            "text": "machine learning diagnosis medical imaging",
            "expected_docs": ["sci_1"],
            "domain": "science"
        },
        {
            "query_id": "q3",
            "text": "climate modeling earth system models",
            "expected_docs": ["sci_2"],
            "domain": "science"
        },
        {
            "query_id": "q4",
            "text": "heart disease prevention lifestyle modifications",
            "expected_docs": ["med_3"],
            "domain": "medical"
        },
        {
            "query_id": "q5",
            "text": "antibiotic resistance stewardship programs",
            "expected_docs": ["med_4"],
            "domain": "medical"
        },
        {
            "query_id": "q6",
            "text": "mental health cognitive behavioral therapy",
            "expected_docs": ["med_5"],
            "domain": "medical"
        },
        {
            "query_id": "q7",
            "text": "patent intellectual property novelty requirements",
            "expected_docs": ["legal_6"],
            "domain": "legal"
        },
        {
            "query_id": "q8",
            "text": "contract formation offer acceptance consideration",
            "expected_docs": ["legal_7"],
            "domain": "legal"
        },
        {
            "query_id": "q9",
            "text": "constitutional law separation of powers",
            "expected_docs": ["legal_8"],
            "domain": "legal"
        },
        {
            "query_id": "q10",
            "text": "artificial intelligence healthcare applications",
            "expected_docs": ["sci_1", "med_5"],  # Multiple relevant docs
            "domain": "cross-domain"
        }
    ]
    
    return queries


def calculate_metrics(results: List[Dict[str, Any]], expected_docs: List[str]) -> Dict[str, float]:
    """Calculate evaluation metrics for search results."""
    if not results or not expected_docs:
        return {"precision_at_5": 0.0, "recall_at_5": 0.0, "mrr": 0.0, "ndcg_at_5": 0.0}
    
    # Get top 5 results
    top_5_results = results[:5]
    retrieved_docs = [result.get('doc_id', '') for result in top_5_results]
    
    # Calculate precision@5
    relevant_retrieved = set(retrieved_docs) & set(expected_docs)
    precision_at_5 = len(relevant_retrieved) / len(top_5_results)
    
    # Calculate recall@5
    recall_at_5 = len(relevant_retrieved) / len(expected_docs)
    
    # Calculate MRR
    mrr = 0.0
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in expected_docs:
            mrr = 1.0 / (i + 1)
            break
    
    # Simple NDCG@5 calculation
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in expected_docs:
            dcg += 1.0 / np.log2(i + 2)
    
    # Ideal DCG (if all expected docs were at the top)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(expected_docs), 5)))
    ndcg_at_5 = dcg / idcg if idcg > 0 else 0.0
    
    return {
        "precision_at_5": precision_at_5,
        "recall_at_5": recall_at_5,
        "mrr": mrr,
        "ndcg_at_5": ndcg_at_5
    }


def evaluate_system(system, system_name: str, documents: List[Document], queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate a retrieval system."""
    print(f"\n=== Evaluating {system_name} ===")
    
    # Get initial memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Index documents
    print("Indexing documents...")
    start_time = time.time()
    if hasattr(system, 'add_documents'):
        system.add_documents(documents)
    elif hasattr(system, 'build_index'):
        system.build_index(documents)
    index_time = time.time() - start_time
    
    # Get memory after indexing
    post_index_memory = process.memory_info().rss / 1024 / 1024  # MB
    index_memory_usage = post_index_memory - initial_memory
    
    results = []
    total_search_time = 0
    all_metrics = []
    memory_measurements = []
    
    for query_data in queries:
        query_id = query_data["query_id"]
        query_text = query_data["text"]
        expected_docs = query_data["expected_docs"]
        
        print(f"Processing query {query_id}: {query_text[:50]}...")
        
        # Search with memory monitoring
        gc.collect()  # Force garbage collection before search
        pre_search_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        try:
            if hasattr(system, 'search'):
                search_results = system.search(query_text, k=10)
            elif hasattr(system, 'retrieve'):
                search_results = system.retrieve(query_text, k=10)
                # Convert to expected format
                converted_results = []
                for i, result in enumerate(search_results):
                    # Handle the result object properly
                    if hasattr(result, 'doc_id'):
                        doc_id = result.doc_id
                    elif hasattr(result, 'document') and hasattr(result.document, 'doc_id'):
                        doc_id = result.document.doc_id
                    else:
                        doc_id = f"unknown_{i}"
                    
                    # Get title
                    title = "Unknown"
                    if hasattr(result, 'title'):
                        title = result.title
                    elif hasattr(result, 'document') and hasattr(result.document, 'metadata') and hasattr(result.document.metadata, 'title'):
                        title = result.document.metadata.title or "Unknown"
                    elif hasattr(result, 'metadata') and hasattr(result.metadata, 'title'):
                        title = result.metadata.title or "Unknown"
                    
                    # Get score
                    score = 0.0
                    if hasattr(result, 'score'):
                        score = float(result.score)
                    elif hasattr(result, 'similarity'):
                        score = float(result.similarity)
                    
                    converted_results.append({
                        'doc_id': doc_id,
                        'title': title,
                        'score': score,
                        'rank': i + 1
                    })
                search_results = converted_results
            else:
                search_results = []
        except Exception as e:
            print(f"Error during search: {e}")
            search_results = []
        
        search_time = time.time() - start_time
        total_search_time += search_time
        
        # Get memory after search
        post_search_memory = process.memory_info().rss / 1024 / 1024  # MB
        search_memory_usage = post_search_memory - pre_search_memory
        memory_measurements.append(search_memory_usage)
        
        # Calculate metrics
        metrics = calculate_metrics(search_results, expected_docs)
        all_metrics.append(metrics)
        
        # Store result
        result = {
            "query_id": query_id,
            "query_text": query_text,
            "search_time": search_time,
            "search_memory_usage": search_memory_usage,
            "num_results": len(search_results),
            "top_result": search_results[0] if search_results else None,
            "all_results": search_results,  # Store all results for detailed analysis
            "metrics": metrics
        }
        results.append(result)
        
        print(f"  Search time: {search_time:.3f}s")
        print(f"  Memory usage: {search_memory_usage:.2f}MB")
        print(f"  Results: {len(search_results)}")
        print(f"  Precision@5: {metrics['precision_at_5']:.3f}")
        print(f"  MRR: {metrics['mrr']:.3f}")
    
    # Calculate average metrics
    avg_metrics = {}
    for metric_name in all_metrics[0].keys():
        avg_metrics[metric_name] = np.mean([m[metric_name] for m in all_metrics])
    
    # Calculate memory statistics
    peak_memory = post_index_memory
    avg_search_memory = np.mean(memory_measurements) if memory_measurements else 0
    
    summary = {
        "system_name": system_name,
        "index_time": index_time,
        "index_memory_usage": index_memory_usage,
        "peak_memory_usage": peak_memory,
        "avg_search_memory": avg_search_memory,
        "total_search_time": total_search_time,
        "avg_search_time": total_search_time / len(queries),
        "throughput_qps": len(queries) / total_search_time if total_search_time > 0 else 0,
        "num_queries": len(queries),
        "avg_metrics": avg_metrics,
        "detailed_results": results
    }
    
    return summary


def main():
    """Run the evaluation."""
    print("Quick Evaluation of Quantum-Inspired RAG vs Classical Baselines")
    print("=" * 70)
    
    # Create test data
    print("Creating test documents and queries...")
    documents = create_test_documents()
    queries = create_test_queries()
    
    print(f"Created {len(documents)} documents and {len(queries)} queries")
    
    # Initialize systems
    systems = []
    
    # Classical baseline
    print("\nInitializing classical baseline...")
    try:
        classical_system = SimpleClassicalBaseline()
        systems.append((classical_system, "Classical BERT+FAISS"))
    except Exception as e:
        print(f"Failed to initialize classical system: {e}")
    
    # Quantum-inspired system
    if has_quantum:
        print("Initializing quantum-inspired system...")
        try:
            quantum_system = TwoStageRetriever()
            systems.append((quantum_system, "Quantum-Inspired RAG"))
        except Exception as e:
            print(f"Failed to initialize quantum system: {e}")
    else:
        print("Quantum system not available")
    
    if not systems:
        print("No systems available for evaluation")
        return
    
    # Run evaluations
    evaluation_results = []
    
    for system, system_name in systems:
        try:
            result = evaluate_system(system, system_name, documents, queries)
            evaluation_results.append(result)
        except Exception as e:
            print(f"Error evaluating {system_name}: {e}")
            traceback.print_exc()
    
    # Print comprehensive comparison
    print("\n" + "=" * 90)
    print("COMPREHENSIVE EVALUATION RESULTS: QUANTUM-INSPIRED RAG vs CLASSICAL BASELINE")
    print("=" * 90)
    
    if len(evaluation_results) >= 2:
        classical_result = evaluation_results[0]
        quantum_result = evaluation_results[1]
        
        print(f"\n{'RETRIEVAL QUALITY METRICS':<40}")
        print("-" * 90)
        print(f"{'Metric':<20} {'Classical':<15} {'Quantum':<15} {'Difference':<15} {'Winner':<15}")
        print("-" * 90)
        
        # Compare metrics
        metrics_to_compare = ['precision_at_5', 'recall_at_5', 'mrr', 'ndcg_at_5']
        quality_winners = {'quantum': 0, 'classical': 0, 'tie': 0}
        
        for metric in metrics_to_compare:
            classical_val = classical_result['avg_metrics'][metric]
            quantum_val = quantum_result['avg_metrics'][metric]
            diff = quantum_val - classical_val
            diff_str = f"{diff:+.3f}"
            
            if abs(diff) < 0.001:
                winner = "TIE"
                quality_winners['tie'] += 1
            elif diff > 0:
                winner = "QUANTUM"
                quality_winners['quantum'] += 1
            else:
                winner = "CLASSICAL"
                quality_winners['classical'] += 1
            
            print(f"{metric:<20} {classical_val:<15.3f} {quantum_val:<15.3f} {diff_str:<15} {winner:<15}")
        
        # Compare performance metrics
        print(f"\n{'PERFORMANCE METRICS':<40}")
        print("-" * 90)
        print(f"{'Metric':<20} {'Classical':<15} {'Quantum':<15} {'Difference':<15} {'Winner':<15}")
        print("-" * 90)
        
        performance_winners = {'quantum': 0, 'classical': 0, 'tie': 0}
        
        # Search time comparison
        classical_time = classical_result['avg_search_time']
        quantum_time = quantum_result['avg_search_time']
        time_diff = quantum_time - classical_time
        time_diff_str = f"{time_diff:+.3f}s"
        time_winner = "CLASSICAL" if time_diff > 0 else "QUANTUM"
        performance_winners[time_winner.lower()] += 1
        
        print(f"{'Avg Search Time':<20} {classical_time:<15.3f} {quantum_time:<15.3f} {time_diff_str:<15} {time_winner:<15}")
        
        # Throughput comparison
        classical_qps = classical_result['throughput_qps']
        quantum_qps = quantum_result['throughput_qps']
        qps_diff = quantum_qps - classical_qps
        qps_diff_str = f"{qps_diff:+.2f}"
        qps_winner = "QUANTUM" if qps_diff > 0 else "CLASSICAL"
        performance_winners[qps_winner.lower()] += 1
        
        print(f"{'Throughput (QPS)':<20} {classical_qps:<15.2f} {quantum_qps:<15.2f} {qps_diff_str:<15} {qps_winner:<15}")
        
        # Memory comparison
        classical_memory = classical_result['peak_memory_usage']
        quantum_memory = quantum_result['peak_memory_usage']
        memory_diff = quantum_memory - classical_memory
        memory_diff_str = f"{memory_diff:+.2f}MB"
        memory_winner = "CLASSICAL" if memory_diff > 0 else "QUANTUM"
        performance_winners[memory_winner.lower()] += 1
        
        print(f"{'Peak Memory (MB)':<20} {classical_memory:<15.2f} {quantum_memory:<15.2f} {memory_diff_str:<15} {memory_winner:<15}")
        
        # Query-by-query analysis
        print(f"\n{'QUERY-BY-QUERY ANALYSIS':<40}")
        print("-" * 90)
        print(f"{'Query ID':<10} {'Classical MRR':<15} {'Quantum MRR':<15} {'Difference':<15} {'Winner':<15}")
        print("-" * 90)
        
        query_winners = {'quantum': 0, 'classical': 0, 'tie': 0}
        
        for i, query_data in enumerate(queries):
            query_id = query_data['query_id']
            classical_mrr = classical_result['detailed_results'][i]['metrics']['mrr']
            quantum_mrr = quantum_result['detailed_results'][i]['metrics']['mrr']
            mrr_diff = quantum_mrr - classical_mrr
            mrr_diff_str = f"{mrr_diff:+.3f}"
            
            if abs(mrr_diff) < 0.001:
                winner = "TIE"
                query_winners['tie'] += 1
            elif mrr_diff > 0:
                winner = "QUANTUM"
                query_winners['quantum'] += 1
            else:
                winner = "CLASSICAL"
                query_winners['classical'] += 1
            
            print(f"{query_id:<10} {classical_mrr:<15.3f} {quantum_mrr:<15.3f} {mrr_diff_str:<15} {winner:<15}")
        
        # Final verdict
        print(f"\n{'FINAL VERDICT':<40}")
        print("=" * 90)
        
        print(f"Quality Metrics: Quantum wins {quality_winners['quantum']}, Classical wins {quality_winners['classical']}, Ties {quality_winners['tie']}")
        print(f"Performance Metrics: Quantum wins {performance_winners['quantum']}, Classical wins {performance_winners['classical']}, Ties {performance_winners['tie']}")
        print(f"Per-Query Analysis: Quantum wins {query_winners['quantum']}, Classical wins {query_winners['classical']}, Ties {query_winners['tie']}")
        
        # Overall conclusion
        total_quantum_wins = quality_winners['quantum'] + performance_winners['quantum'] + query_winners['quantum']
        total_classical_wins = quality_winners['classical'] + performance_winners['classical'] + query_winners['classical']
        
        print(f"\n{'OVERALL CONCLUSION':<40}")
        print("-" * 90)
        
        if total_quantum_wins > total_classical_wins:
            print("🏆 QUANTUM-INSPIRED RAG SYSTEM WINS OVERALL")
            print("   - Superior in more categories across quality, performance, and individual queries")
        elif total_classical_wins > total_quantum_wins:
            print("🏆 CLASSICAL BERT+FAISS SYSTEM WINS OVERALL")  
            print("   - Superior in more categories across quality, performance, and individual queries")
        else:
            print("🤝 BOTH SYSTEMS PERFORM SIMILARLY OVERALL")
            print("   - Each system has strengths in different areas")
        
        # Specific recommendations
        print(f"\n{'RECOMMENDATIONS':<40}")
        print("-" * 90)
        
        if quantum_memory < classical_memory:
            print("✅ Choose Quantum-Inspired RAG for memory-constrained environments")
        else:
            print("✅ Choose Classical BERT+FAISS for memory-constrained environments")
        
        if quantum_time < classical_time:
            print("✅ Choose Quantum-Inspired RAG for latency-sensitive applications")
        else:
            print("✅ Choose Classical BERT+FAISS for latency-sensitive applications")
        
        if quality_winners['quantum'] > quality_winners['classical']:
            print("✅ Choose Quantum-Inspired RAG for better retrieval quality")
        else:
            print("✅ Choose Classical BERT+FAISS for better retrieval quality")
    
    else:
        for result in evaluation_results:
            print(f"\n{result['system_name']} Results:")
            print(f"  Index time: {result['index_time']:.3f}s")
            print(f"  Index memory: {result['index_memory_usage']:.2f}MB")
            print(f"  Peak memory: {result['peak_memory_usage']:.2f}MB")
            print(f"  Avg search time: {result['avg_search_time']:.3f}s")
            print(f"  Throughput: {result['throughput_qps']:.2f} QPS")
            print(f"  Avg Precision@5: {result['avg_metrics']['precision_at_5']:.3f}")
            print(f"  Avg MRR: {result['avg_metrics']['mrr']:.3f}")
            print(f"  Avg NDCG@5: {result['avg_metrics']['ndcg_at_5']:.3f}")
    
    # Save results
    output_file = "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\nDetailed results saved to {output_file}")
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
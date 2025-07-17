#!/usr/bin/env python3
"""
Realistic Evaluation Framework: Quantum-Inspired RAG vs Classical Baselines

This addresses the fundamental flaws in the previous evaluation:
1. Larger, more diverse document corpus (50+ documents)
2. Challenging queries with multiple relevant documents
3. Realistic precision/recall scenarios
4. Proper blind evaluation without perfect matches
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
import psutil
import gc

# Add the quantum_rerank package to the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from quantum_rerank.retrieval.two_stage_retriever import TwoStageRetriever
    from quantum_rerank.retrieval.document_store import Document, DocumentMetadata
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

# Simple baseline implementations
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity


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


def create_large_diverse_corpus() -> List[Document]:
    """Create a large, diverse document corpus with 50+ documents."""
    documents = []
    
    # Scientific papers (20 documents)
    science_topics = [
        ("Quantum Computing Foundations", "quantum algorithms quantum gates qubits superposition entanglement computational complexity"),
        ("Machine Learning in Healthcare", "artificial intelligence medical diagnosis neural networks deep learning clinical decision support"),
        ("Climate Change Modeling", "global warming climate models atmospheric science carbon emissions environmental impact"),
        ("Renewable Energy Systems", "solar power wind energy renewable resources sustainable technology energy storage"),
        ("Artificial Intelligence Ethics", "AI ethics machine learning bias algorithmic fairness responsible AI governance"),
        ("Blockchain Technology", "cryptocurrency distributed ledger smart contracts decentralized systems bitcoin ethereum"),
        ("Gene Therapy Research", "genetic engineering CRISPR gene editing therapeutic applications molecular biology"),
        ("Quantum Cryptography", "quantum encryption quantum key distribution cryptographic protocols secure communication"),
        ("Autonomous Vehicles", "self-driving cars computer vision sensor fusion autonomous navigation traffic systems"),
        ("Space Exploration", "space missions Mars exploration planetary science astronauts space technology"),
        ("Nanotechnology Applications", "nanomaterials nanotechnology molecular engineering medical applications electronics"),
        ("Cybersecurity Threats", "network security cyber attacks malware prevention digital forensics information security"),
        ("Biotechnology Innovation", "bioengineering synthetic biology pharmaceutical research drug development medical devices"),
        ("Robotics and Automation", "industrial robots automation manufacturing artificial intelligence human-robot interaction"),
        ("Data Science Methods", "big data analytics statistical modeling data mining machine learning algorithms"),
        ("Quantum Physics Theory", "quantum mechanics wave-particle duality quantum field theory theoretical physics"),
        ("Computer Vision Applications", "image processing pattern recognition object detection deep learning neural networks"),
        ("Natural Language Processing", "text analysis language models computational linguistics speech recognition translation"),
        ("Materials Science", "advanced materials nanotechnology composite materials engineering properties applications"),
        ("Bioinformatics Research", "genomics protein analysis computational biology sequence analysis molecular modeling")
    ]
    
    # Medical documents (15 documents)
    medical_topics = [
        ("Cancer Treatment Advances", "oncology chemotherapy immunotherapy targeted therapy clinical trials cancer research"),
        ("Cardiovascular Disease Prevention", "heart disease hypertension lifestyle modifications risk factors prevention strategies"),
        ("Mental Health Treatment", "depression anxiety cognitive behavioral therapy psychiatric medications psychological interventions"),
        ("Infectious Disease Control", "epidemiology disease prevention vaccination public health antimicrobial resistance"),
        ("Diabetes Management", "blood glucose insulin diabetes complications lifestyle management medical treatment"),
        ("Neurological Disorders", "brain diseases Alzheimer's Parkinson's neurodegenerative disorders treatment approaches"),
        ("Pediatric Medicine", "child health pediatric care childhood diseases vaccination schedules developmental milestones"),
        ("Surgical Innovations", "minimally invasive surgery robotic surgery surgical techniques medical devices"),
        ("Pharmaceutical Research", "drug development clinical trials FDA approval pharmaceutical industry medication safety"),
        ("Emergency Medicine", "trauma care emergency procedures critical care life support emergency protocols"),
        ("Telemedicine Applications", "remote healthcare digital health telehealth virtual consultations medical technology"),
        ("Genetic Disorders", "inherited diseases genetic testing gene therapy personalized medicine genetic counseling"),
        ("Rehabilitation Medicine", "physical therapy occupational therapy recovery treatment disability management"),
        ("Preventive Medicine", "disease prevention health screening lifestyle medicine public health interventions"),
        ("Medical Imaging", "radiology MRI CT scan ultrasound diagnostic imaging medical technology")
    ]
    
    # Legal documents (15 documents)
    legal_topics = [
        ("Contract Law Principles", "contract formation offer acceptance consideration breach remedies commercial law"),
        ("Constitutional Rights", "civil liberties constitutional law fundamental rights judicial review government powers"),
        ("Criminal Justice System", "criminal law prosecution defense court procedures sentencing criminal procedure"),
        ("Intellectual Property Law", "patents trademarks copyrights trade secrets IP protection licensing"),
        ("Corporate Governance", "business law corporate structure fiduciary duties shareholder rights compliance"),
        ("Environmental Law", "environmental regulations pollution control sustainability legal compliance environmental policy"),
        ("Labor and Employment", "employment law workplace rights discrimination harassment labor relations"),
        ("Tax Law and Policy", "taxation tax code tax planning tax policy revenue generation"),
        ("Real Estate Law", "property rights real estate transactions zoning laws property law"),
        ("Family Law", "marriage divorce child custody family courts domestic relations"),
        ("Immigration Law", "immigration policy visa applications citizenship deportation asylum procedures"),
        ("International Law", "treaties international relations diplomatic law global governance"),
        ("Banking and Finance", "financial regulations banking law securities regulation financial institutions"),
        ("Healthcare Law", "medical malpractice healthcare regulations HIPAA compliance healthcare policy"),
        ("Technology Law", "cyber law data privacy technology regulation digital rights")
    ]
    
    # Generate documents with realistic content
    doc_id = 0
    
    for title, keywords in science_topics:
        content = f"This comprehensive study examines {title.lower()} and its implications for modern technology. "
        content += f"Key concepts include {keywords}. "
        content += f"The research methodology involves extensive analysis of {keywords.split()[0]} systems and their applications. "
        content += f"Results demonstrate significant advances in {keywords.split()[1]} technology with potential impact on future development. "
        content += f"The study covers theoretical foundations, practical implementations, and real-world applications of {keywords.split()[2]}. "
        content += f"Findings suggest that {keywords.split()[0]} represents a paradigm shift in how we approach computational problems. "
        content += f"The implications extend beyond technical domains to include economic, social, and ethical considerations. "
        content += f"Future research directions focus on scalability, efficiency, and broader adoption of these technologies."
        
        if has_quantum:
            metadata = DocumentMetadata(
                title=title,
                source="academic",
                custom_fields={"domain": "science", "keywords": keywords, "word_count": len(content.split())}
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
                source="academic",
                metadata={"keywords": keywords, "word_count": len(content.split())}
            ))
        doc_id += 1
    
    for title, keywords in medical_topics:
        content = f"This clinical review examines current approaches to {title.lower()} in modern healthcare settings. "
        content += f"Medical professionals focus on {keywords} to improve patient outcomes. "
        content += f"Evidence-based practices demonstrate effectiveness of {keywords.split()[0]} interventions in clinical settings. "
        content += f"Patient care protocols emphasize {keywords.split()[1]} management and comprehensive treatment approaches. "
        content += f"Healthcare providers utilize {keywords.split()[2]} techniques to enhance diagnostic accuracy and treatment efficacy. "
        content += f"The medical literature supports {keywords.split()[0]} as a primary intervention for related conditions. "
        content += f"Clinical guidelines recommend {keywords.split()[1]} monitoring and patient education programs. "
        content += f"Future developments in {keywords.split()[2]} promise improved outcomes for patients and healthcare systems."
        
        if has_quantum:
            metadata = DocumentMetadata(
                title=title,
                source="clinical",
                custom_fields={"domain": "medical", "keywords": keywords, "word_count": len(content.split())}
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
                source="clinical",
                metadata={"keywords": keywords, "word_count": len(content.split())}
            ))
        doc_id += 1
    
    for title, keywords in legal_topics:
        content = f"This legal analysis examines the principles and applications of {title.lower()} in contemporary jurisprudence. "
        content += f"Legal practitioners must understand {keywords} to effectively represent clients and navigate complex cases. "
        content += f"Court decisions establish precedents regarding {keywords.split()[0]} and its interpretation under current law. "
        content += f"Legislative frameworks address {keywords.split()[1]} through comprehensive statutory provisions. "
        content += f"Legal scholars debate the implications of {keywords.split()[2]} for future legal development. "
        content += f"Practitioners rely on {keywords.split()[0]} precedents to build compelling legal arguments. "
        content += f"The legal system balances {keywords.split()[1]} considerations with broader societal interests. "
        content += f"Future legal developments will likely address emerging challenges in {keywords.split()[2]} regulation."
        
        if has_quantum:
            metadata = DocumentMetadata(
                title=title,
                source="legal",
                custom_fields={"domain": "legal", "keywords": keywords, "word_count": len(content.split())}
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
                source="legal",
                metadata={"keywords": keywords, "word_count": len(content.split())}
            ))
        doc_id += 1
    
    return documents


def create_challenging_queries() -> List[Dict[str, Any]]:
    """Create challenging queries that test real-world IR scenarios."""
    queries = [
        # Ambiguous queries with multiple potential matches
        {
            "query_id": "q1",
            "text": "machine learning applications",
            "expected_docs": ["sci_1", "sci_4", "sci_16", "sci_17"],  # Multiple ML-related docs
            "domain": "ambiguous"
        },
        {
            "query_id": "q2",
            "text": "quantum systems and applications",
            "expected_docs": ["sci_0", "sci_7", "sci_15"],  # Multiple quantum docs
            "domain": "ambiguous"
        },
        {
            "query_id": "q3",
            "text": "healthcare technology and innovation",
            "expected_docs": ["sci_1", "med_20", "med_25", "med_30"],  # Cross-domain
            "domain": "cross-domain"
        },
        {
            "query_id": "q4",
            "text": "disease prevention and management",
            "expected_docs": ["med_21", "med_23", "med_29"],  # Multiple medical docs
            "domain": "medical"
        },
        {
            "query_id": "q5",
            "text": "legal regulations and compliance",
            "expected_docs": ["legal_40", "legal_41", "legal_46"],  # Multiple legal docs
            "domain": "legal"
        },
        # Specific technical queries
        {
            "query_id": "q6",
            "text": "artificial intelligence ethics and bias",
            "expected_docs": ["sci_4"],  # Single specific match
            "domain": "specific"
        },
        {
            "query_id": "q7",
            "text": "genetic engineering and CRISPR technology",
            "expected_docs": ["sci_6", "med_27"],  # Cross-domain match
            "domain": "specific"
        },
        {
            "query_id": "q8",
            "text": "cybersecurity threats and network protection",
            "expected_docs": ["sci_11"],  # Single specific match
            "domain": "specific"
        },
        # Challenging edge cases
        {
            "query_id": "q9",
            "text": "sustainable energy and environmental impact",
            "expected_docs": ["sci_3", "legal_41"],  # Sparse matches
            "domain": "challenging"
        },
        {
            "query_id": "q10",
            "text": "blockchain cryptocurrency financial systems",
            "expected_docs": ["sci_5", "legal_47"],  # Cross-domain sparse
            "domain": "challenging"
        },
        # No clear matches (should have low precision)
        {
            "query_id": "q11",
            "text": "underwater basket weaving techniques",
            "expected_docs": [],  # No relevant docs
            "domain": "no-match"
        },
        {
            "query_id": "q12",
            "text": "medieval cooking recipes and ingredients",
            "expected_docs": [],  # No relevant docs
            "domain": "no-match"
        }
    ]
    
    return queries


def calculate_realistic_metrics(results: List[Dict[str, Any]], expected_docs: List[str]) -> Dict[str, float]:
    """Calculate realistic evaluation metrics."""
    if not results:
        return {"precision_at_5": 0.0, "precision_at_10": 0.0, "recall_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0, "ndcg_at_10": 0.0}
    
    # Get top results
    top_5_results = results[:5]
    top_10_results = results[:10]
    
    retrieved_docs_5 = [result.get('doc_id', '') for result in top_5_results]
    retrieved_docs_10 = [result.get('doc_id', '') for result in top_10_results]
    
    # Calculate precision@5 and precision@10
    if expected_docs:
        relevant_retrieved_5 = set(retrieved_docs_5) & set(expected_docs)
        relevant_retrieved_10 = set(retrieved_docs_10) & set(expected_docs)
        
        precision_at_5 = len(relevant_retrieved_5) / len(top_5_results) if top_5_results else 0.0
        precision_at_10 = len(relevant_retrieved_10) / len(top_10_results) if top_10_results else 0.0
        
        # Calculate recall@5 and recall@10
        recall_at_5 = len(relevant_retrieved_5) / len(expected_docs)
        recall_at_10 = len(relevant_retrieved_10) / len(expected_docs)
        
        # Calculate MRR
        mrr = 0.0
        for i, doc_id in enumerate(retrieved_docs_10):
            if doc_id in expected_docs:
                mrr = 1.0 / (i + 1)
                break
        
        # Calculate NDCG@10
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_docs_10):
            if doc_id in expected_docs:
                dcg += 1.0 / np.log2(i + 2)
        
        # Ideal DCG (if all expected docs were at the top)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(expected_docs), 10)))
        ndcg_at_10 = dcg / idcg if idcg > 0 else 0.0
    else:
        # For no-match queries, precision should be 0 if any docs are returned
        precision_at_5 = 0.0
        precision_at_10 = 0.0
        recall_at_5 = 0.0  # Undefined for no expected docs, but we'll use 0
        recall_at_10 = 0.0
        mrr = 0.0
        ndcg_at_10 = 0.0
    
    return {
        "precision_at_5": precision_at_5,
        "precision_at_10": precision_at_10,
        "recall_at_5": recall_at_5,
        "recall_at_10": recall_at_10,
        "mrr": mrr,
        "ndcg_at_10": ndcg_at_10
    }


def evaluate_system_realistic(system, system_name: str, documents: List[Document], queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate a retrieval system with realistic methodology."""
    print(f"\n=== Evaluating {system_name} ===")
    
    # Get initial memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Index documents
    print(f"Indexing {len(documents)} documents...")
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
        
        print(f"Processing query {query_id}: {query_text}")
        
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
        metrics = calculate_realistic_metrics(search_results, expected_docs)
        all_metrics.append(metrics)
        
        # Store result
        result = {
            "query_id": query_id,
            "query_text": query_text,
            "expected_docs": expected_docs,
            "search_time": search_time,
            "search_memory_usage": search_memory_usage,
            "num_results": len(search_results),
            "top_result": search_results[0] if search_results else None,
            "all_results": search_results,
            "metrics": metrics
        }
        results.append(result)
        
        print(f"  Search time: {search_time:.3f}s")
        print(f"  Results: {len(search_results)}")
        print(f"  Precision@5: {metrics['precision_at_5']:.3f}")
        print(f"  Precision@10: {metrics['precision_at_10']:.3f}")
        print(f"  Recall@5: {metrics['recall_at_5']:.3f}")
        print(f"  MRR: {metrics['mrr']:.3f}")
        print(f"  NDCG@10: {metrics['ndcg_at_10']:.3f}")
    
    # Calculate average metrics
    avg_metrics = {}
    for metric_name in all_metrics[0].keys():
        avg_metrics[metric_name] = np.mean([m[metric_name] for m in all_metrics])
    
    # Calculate memory statistics
    peak_memory = post_index_memory
    avg_search_memory = np.mean(memory_measurements) if memory_measurements else 0
    
    summary = {
        "system_name": system_name,
        "corpus_size": len(documents),
        "num_queries": len(queries),
        "index_time": index_time,
        "index_memory_usage": index_memory_usage,
        "peak_memory_usage": peak_memory,
        "avg_search_memory": avg_search_memory,
        "total_search_time": total_search_time,
        "avg_search_time": total_search_time / len(queries),
        "throughput_qps": len(queries) / total_search_time if total_search_time > 0 else 0,
        "avg_metrics": avg_metrics,
        "detailed_results": results
    }
    
    return summary


def main():
    """Run the realistic evaluation."""
    print("Realistic Evaluation of Quantum-Inspired RAG vs Classical Baselines")
    print("=" * 80)
    print("Addressing previous evaluation flaws:")
    print("- Larger corpus (50+ documents)")
    print("- Challenging queries with multiple/sparse matches")
    print("- Realistic precision/recall scenarios")
    print("- Proper blind evaluation")
    print()
    
    # Create realistic test data
    print("Creating large, diverse document corpus...")
    documents = create_large_diverse_corpus()
    print(f"Created {len(documents)} documents")
    
    print("Creating challenging queries...")
    queries = create_challenging_queries()
    print(f"Created {len(queries)} queries")
    
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
            result = evaluate_system_realistic(system, system_name, documents, queries)
            evaluation_results.append(result)
        except Exception as e:
            print(f"Error evaluating {system_name}: {e}")
            traceback.print_exc()
    
    # Print comprehensive comparison
    print("\n" + "=" * 80)
    print("REALISTIC EVALUATION RESULTS")
    print("=" * 80)
    
    if len(evaluation_results) >= 2:
        classical_result = evaluation_results[0]
        quantum_result = evaluation_results[1]
        
        print(f"\nCorpus Size: {classical_result['corpus_size']} documents")
        print(f"Query Count: {classical_result['num_queries']} queries")
        print()
        
        print("RETRIEVAL QUALITY METRICS")
        print("-" * 80)
        print(f"{'Metric':<20} {'Classical':<15} {'Quantum':<15} {'Difference':<15} {'Winner':<15}")
        print("-" * 80)
        
        metrics_to_compare = ['precision_at_5', 'precision_at_10', 'recall_at_5', 'recall_at_10', 'mrr', 'ndcg_at_10']
        
        for metric in metrics_to_compare:
            classical_val = classical_result['avg_metrics'][metric]
            quantum_val = quantum_result['avg_metrics'][metric]
            diff = quantum_val - classical_val
            diff_str = f"{diff:+.3f}"
            
            if abs(diff) < 0.01:
                winner = "TIE"
            elif diff > 0:
                winner = "QUANTUM"
            else:
                winner = "CLASSICAL"
            
            print(f"{metric:<20} {classical_val:<15.3f} {quantum_val:<15.3f} {diff_str:<15} {winner:<15}")
        
        print("\nPERFORMANCE METRICS")
        print("-" * 80)
        print(f"{'Metric':<20} {'Classical':<15} {'Quantum':<15} {'Difference':<15} {'Winner':<15}")
        print("-" * 80)
        
        # Search time comparison
        classical_time = classical_result['avg_search_time']
        quantum_time = quantum_result['avg_search_time']
        time_diff = quantum_time - classical_time
        time_diff_str = f"{time_diff:+.3f}s"
        time_winner = "CLASSICAL" if time_diff > 0 else "QUANTUM"
        
        print(f"{'Avg Search Time':<20} {classical_time:<15.3f} {quantum_time:<15.3f} {time_diff_str:<15} {time_winner:<15}")
        
        # Throughput comparison
        classical_qps = classical_result['throughput_qps']
        quantum_qps = quantum_result['throughput_qps']
        qps_diff = quantum_qps - classical_qps
        qps_diff_str = f"{qps_diff:+.2f}"
        qps_winner = "QUANTUM" if qps_diff > 0 else "CLASSICAL"
        
        print(f"{'Throughput (QPS)':<20} {classical_qps:<15.2f} {quantum_qps:<15.2f} {qps_diff_str:<15} {qps_winner:<15}")
        
        # Memory comparison
        classical_memory = classical_result['peak_memory_usage']
        quantum_memory = quantum_result['peak_memory_usage']
        memory_diff = quantum_memory - classical_memory
        memory_diff_str = f"{memory_diff:+.2f}MB"
        memory_winner = "CLASSICAL" if memory_diff > 0 else "QUANTUM"
        
        print(f"{'Peak Memory (MB)':<20} {classical_memory:<15.2f} {quantum_memory:<15.2f} {memory_diff_str:<15} {memory_winner:<15}")
        
        print("\nSAMPLE QUERY RESULTS")
        print("-" * 80)
        
        # Show first 3 queries as examples
        for i in range(min(3, len(queries))):
            query = queries[i]
            classical_metrics = classical_result['detailed_results'][i]['metrics']
            quantum_metrics = quantum_result['detailed_results'][i]['metrics']
            
            print(f"\nQuery {query['query_id']}: {query['text']}")
            print(f"Expected docs: {query['expected_docs']}")
            print(f"Classical - P@5: {classical_metrics['precision_at_5']:.3f}, MRR: {classical_metrics['mrr']:.3f}")
            print(f"Quantum   - P@5: {quantum_metrics['precision_at_5']:.3f}, MRR: {quantum_metrics['mrr']:.3f}")
    
    else:
        for result in evaluation_results:
            print(f"\n{result['system_name']} Results:")
            print(f"  Corpus size: {result['corpus_size']} documents")
            print(f"  Index time: {result['index_time']:.3f}s")
            print(f"  Peak memory: {result['peak_memory_usage']:.2f}MB")
            print(f"  Avg search time: {result['avg_search_time']:.3f}s")
            print(f"  Throughput: {result['throughput_qps']:.2f} QPS")
            print(f"  Avg Precision@5: {result['avg_metrics']['precision_at_5']:.3f}")
            print(f"  Avg Precision@10: {result['avg_metrics']['precision_at_10']:.3f}")
            print(f"  Avg Recall@5: {result['avg_metrics']['recall_at_5']:.3f}")
            print(f"  Avg MRR: {result['avg_metrics']['mrr']:.3f}")
            print(f"  Avg NDCG@10: {result['avg_metrics']['ndcg_at_10']:.3f}")
    
    # Save results
    output_file = "realistic_evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\nDetailed results saved to {output_file}")
    print("\nRealistic evaluation completed successfully!")


if __name__ == "__main__":
    main()
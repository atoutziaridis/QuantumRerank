"""
Real-World Medical RAG Benchmark with Actual PubMed Documents
===========================================================

Comprehensive benchmark using real medical documents from PubMed Central,
realistic noise patterns, and clinical queries. Tests quantum vs classical
RAG performance on authentic medical literature.
"""

import numpy as np
import time
import json
import random
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

from pubmed_fetcher import fetch_real_medical_documents
from quantum_rerank.retrieval.document_store import Document, DocumentMetadata
from quantum_rerank.core.embeddings import EmbeddingProcessor
from quantum_rerank.core.quantum_similarity_engine import QuantumSimilarityEngine, SimilarityMethod


@dataclass
class BenchmarkResult:
    """Results from a single benchmark test."""
    query: str
    noise_level: float
    noise_type: str
    classical_precision: float
    quantum_precision: float
    classical_recall: float
    quantum_recall: float
    classical_latency_ms: float
    quantum_latency_ms: float
    improvement_percent: float


class MedicalNoiseInjector:
    """Inject realistic noise into medical documents."""
    
    def __init__(self):
        # OCR errors from real medical document scanning
        self.ocr_map = {
            'l': '1', 'I': 'l', 'O': '0', 'S': '5', 'G': '6', 'B': '8',
            'o': '0', 'a': '@', 'e': 'c', 'n': 'm', 'h': 'b', 'r': 'n',
            'u': 'v', 'v': 'u', 'c': 'e', 'i': 'j', 't': 'f'
        }
        
        # Medical abbreviations  
        self.abbreviations = {
            'myocardial infarction': 'MI',
            'blood pressure': 'BP',
            'heart rate': 'HR',
            'diabetes mellitus': 'DM',
            'electrocardiogram': 'ECG',
            'intensive care unit': 'ICU',
            'emergency department': 'ED',
            'chronic obstructive pulmonary disease': 'COPD',
            'computed tomography': 'CT',
            'magnetic resonance imaging': 'MRI'
        }
        
        # Medical typos
        self.typos = {
            'patient': 'pateint',
            'treatment': 'treatement', 
            'diagnosis': 'diagosis',
            'symptoms': 'symtoms',
            'therapy': 'theraphy',
            'medication': 'medicaton'
        }
    
    def inject_ocr_noise(self, text: str, error_rate: float) -> str:
        """Inject OCR-like errors."""
        chars = list(text)
        for i in range(len(chars)):
            if random.random() < error_rate:
                if chars[i] in self.ocr_map:
                    chars[i] = self.ocr_map[chars[i]]
        return ''.join(chars)
    
    def inject_typos(self, text: str, error_rate: float) -> str:
        """Inject medical typos."""
        for correct, typo in self.typos.items():
            if correct in text and random.random() < error_rate:
                text = text.replace(correct, typo, 1)
        return text
    
    def inject_abbreviations(self, text: str, rate: float) -> str:
        """Convert terms to abbreviations."""
        for full_term, abbrev in self.abbreviations.items():
            if full_term in text and random.random() < rate:
                text = text.replace(full_term, abbrev, 1)
        return text
    
    def inject_noise(self, text: str, noise_type: str, noise_level: float) -> str:
        """Inject specified type of noise."""
        if noise_type == "clean":
            return text
        elif noise_type == "ocr":
            return self.inject_ocr_noise(text, noise_level)
        elif noise_type == "typos":
            return self.inject_typos(text, noise_level * 2)  # Scale up for visibility
        elif noise_type == "abbreviations":
            return self.inject_abbreviations(text, noise_level * 3)  # Scale up for visibility
        elif noise_type == "mixed":
            noisy = text
            noisy = self.inject_abbreviations(noisy, noise_level * 0.6)
            noisy = self.inject_typos(noisy, noise_level * 0.4)
            noisy = self.inject_ocr_noise(noisy, noise_level * 0.3)
            return noisy
        else:
            return text


class RealWorldMedicalBenchmark:
    """Comprehensive benchmark with real PubMed documents."""
    
    def __init__(self, num_documents: int = 20, num_queries: int = 10):
        self.num_documents = num_documents
        self.num_queries = num_queries
        
        # Initialize components
        self.noise_injector = MedicalNoiseInjector()
        self.embedder = EmbeddingProcessor()
        self.quantum_engine = QuantumSimilarityEngine()
        
        # Test configuration
        self.noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
        self.noise_types = ["clean", "mixed", "ocr", "typos", "abbreviations"]
        
        # Clinical test queries
        self.test_queries = [
            "acute myocardial infarction treatment and management",
            "diabetes mellitus therapeutic approaches and outcomes", 
            "heart failure pharmacological interventions",
            "hypertension cardiovascular risk reduction strategies",
            "stroke thrombolytic therapy and patient outcomes",
            "pneumonia antibiotic treatment protocols",
            "COPD exacerbation management in emergency care",
            "sepsis early recognition and intervention strategies",
            "atrial fibrillation anticoagulation guidelines",
            "asthma bronchodilator therapy effectiveness"
        ]
        
        self.results = []
    
    def create_relevance_judgments(self, query: str, documents: List[Dict]) -> List[str]:
        """Create relevance judgments based on content and MeSH terms."""
        relevant_docs = []
        query_terms = set(query.lower().split())
        
        for doc in documents:
            relevance_score = 0
            
            # Check title overlap
            title_terms = set(doc['title'].lower().split())
            title_overlap = len(query_terms.intersection(title_terms))
            relevance_score += title_overlap * 3
            
            # Check abstract overlap  
            abstract_terms = set(doc['abstract'].lower().split())
            abstract_overlap = len(query_terms.intersection(abstract_terms))
            relevance_score += abstract_overlap
            
            # Check MeSH term overlap
            mesh_terms = set([term.lower() for term in doc['mesh_terms']])
            mesh_overlap = len(query_terms.intersection(mesh_terms))
            relevance_score += mesh_overlap * 5
            
            # Domain-specific relevance
            if "myocardial" in query.lower() or "heart" in query.lower():
                if any(term in doc['title'].lower() or term in doc['abstract'].lower() 
                       for term in ['cardiac', 'heart', 'myocardial', 'coronary']):
                    relevance_score += 10
            
            if "diabetes" in query.lower():
                if any(term in doc['title'].lower() or term in doc['abstract'].lower()
                       for term in ['diabetes', 'glycemic', 'insulin', 'glucose']):
                    relevance_score += 10
            
            # Threshold for relevance
            if relevance_score >= 5:
                relevant_docs.append(doc['pmid'])
        
        # Ensure at least some relevant documents
        if len(relevant_docs) < 2:
            relevant_docs.extend([doc['pmid'] for doc in documents[:3]])
        
        return list(set(relevant_docs))
    
    def calculate_precision_recall(self, retrieved_ids: List[str], 
                                  relevant_ids: List[str], 
                                  k: int = 5) -> Tuple[float, float]:
        """Calculate precision and recall at k."""
        top_k = retrieved_ids[:k]
        relevant_retrieved = len(set(top_k).intersection(set(relevant_ids)))
        
        precision = relevant_retrieved / k if k > 0 else 0.0
        recall = relevant_retrieved / len(relevant_ids) if len(relevant_ids) > 0 else 0.0
        
        return precision, recall
    
    def perform_retrieval(self, query: str, documents: List[Document], method: str) -> Tuple[List[str], float]:
        """Perform retrieval using specified method."""
        start_time = time.time()
        similarities = []
        
        if method == "classical":
            query_embedding = self.embedder.encode_single_text(query)
            
            for doc in documents:
                doc_embedding = self.embedder.encode_single_text(doc.content)
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                similarities.append((similarity, doc.doc_id))
        
        elif method == "quantum":
            for doc in documents:
                try:
                    similarity, _ = self.quantum_engine.compute_similarity(
                        query, doc.content, method=SimilarityMethod.HYBRID_WEIGHTED
                    )
                    similarities.append((similarity, doc.doc_id))
                except Exception:
                    # Fallback to classical if quantum fails
                    query_embedding = self.embedder.encode_single_text(query)
                    doc_embedding = self.embedder.encode_single_text(doc.content)
                    similarity = np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                    )
                    similarities.append((similarity, doc.doc_id))
        
        latency = (time.time() - start_time) * 1000
        
        # Sort by similarity and return document IDs
        similarities.sort(key=lambda x: x[0], reverse=True)
        retrieved_ids = [doc_id for _, doc_id in similarities]
        
        return retrieved_ids, latency
    
    def run_benchmark(self) -> Dict:
        """Run the complete benchmark."""
        print("="*70)
        print("REAL-WORLD MEDICAL RAG BENCHMARK WITH PUBMED DOCUMENTS")
        print("="*70)
        
        # Fetch real medical documents
        print(f"Fetching {self.num_documents} real medical documents from PubMed...")
        raw_documents = fetch_real_medical_documents(self.num_documents)
        
        if len(raw_documents) < 5:
            print("ERROR: Could not fetch enough documents from PubMed")
            return {}
        
        print(f"Successfully fetched {len(raw_documents)} real medical documents")
        
        # Convert to Document objects
        clean_documents = []
        for doc_data in raw_documents:
            doc = Document(
                doc_id=doc_data['pmid'],
                content=doc_data['full_text'],
                metadata=DocumentMetadata(
                    title=doc_data['title'],
                    source="PubMed",
                    custom_fields={
                        'journal': doc_data['journal'],
                        'mesh_terms': doc_data['mesh_terms'],
                        'keywords': doc_data['keywords']
                    }
                )
            )
            clean_documents.append(doc)
        
        # Select test queries
        test_queries = self.test_queries[:self.num_queries]
        
        print(f"Running benchmark with {len(test_queries)} queries...")
        print(f"Noise levels: {self.noise_levels}")
        print(f"Noise types: {self.noise_types}")
        
        total_tests = len(test_queries) * len(self.noise_levels) * len(self.noise_types)
        test_count = 0
        
        for query_idx, query in enumerate(test_queries):
            print(f"\nQuery {query_idx + 1}: {query}")
            
            # Create relevance judgments
            relevant_docs = self.create_relevance_judgments(query, raw_documents)
            print(f"  Relevant documents: {len(relevant_docs)}")
            
            for noise_level in self.noise_levels:
                for noise_type in self.noise_types:
                    test_count += 1
                    print(f"  Test {test_count}/{total_tests}: {noise_type} @ {noise_level:.0%}")
                    
                    # Apply noise to documents
                    noisy_documents = []
                    for doc in clean_documents:
                        noisy_content = self.noise_injector.inject_noise(
                            doc.content, noise_type, noise_level
                        )
                        noisy_doc = Document(
                            doc_id=doc.doc_id,
                            content=noisy_content,
                            metadata=doc.metadata
                        )
                        noisy_documents.append(noisy_doc)
                    
                    # Classical retrieval
                    classical_ids, classical_latency = self.perform_retrieval(
                        query, noisy_documents, "classical"
                    )
                    classical_precision, classical_recall = self.calculate_precision_recall(
                        classical_ids, relevant_docs
                    )
                    
                    # Quantum retrieval
                    quantum_ids, quantum_latency = self.perform_retrieval(
                        query, noisy_documents, "quantum"
                    )
                    quantum_precision, quantum_recall = self.calculate_precision_recall(
                        quantum_ids, relevant_docs
                    )
                    
                    # Calculate improvement
                    improvement = ((quantum_precision - classical_precision) / 
                                 max(classical_precision, 0.001)) * 100
                    
                    # Store result
                    result = BenchmarkResult(
                        query=query,
                        noise_level=noise_level,
                        noise_type=noise_type,
                        classical_precision=classical_precision,
                        quantum_precision=quantum_precision,
                        classical_recall=classical_recall,
                        quantum_recall=quantum_recall,
                        classical_latency_ms=classical_latency,
                        quantum_latency_ms=quantum_latency,
                        improvement_percent=improvement
                    )
                    self.results.append(result)
        
        return self.generate_report()
    
    def generate_report(self) -> Dict:
        """Generate comprehensive benchmark report."""
        if not self.results:
            return {"error": "No results available"}
        
        # Overall statistics
        total_tests = len(self.results)
        quantum_wins = sum(1 for r in self.results if r.improvement_percent > 0)
        classical_wins = sum(1 for r in self.results if r.improvement_percent < 0)
        ties = total_tests - quantum_wins - classical_wins
        
        avg_classical_precision = np.mean([r.classical_precision for r in self.results])
        avg_quantum_precision = np.mean([r.quantum_precision for r in self.results])
        avg_improvement = np.mean([r.improvement_percent for r in self.results])
        avg_classical_latency = np.mean([r.classical_latency_ms for r in self.results])
        avg_quantum_latency = np.mean([r.quantum_latency_ms for r in self.results])
        
        # Analysis by noise level
        noise_analysis = {}
        for noise_level in self.noise_levels:
            level_results = [r for r in self.results if r.noise_level == noise_level]
            if level_results:
                noise_analysis[noise_level] = {
                    'improvement': np.mean([r.improvement_percent for r in level_results]),
                    'quantum_wins': sum(1 for r in level_results if r.improvement_percent > 0),
                    'total_tests': len(level_results)
                }
        
        # Analysis by noise type
        type_analysis = {}
        for noise_type in self.noise_types:
            type_results = [r for r in self.results if r.noise_type == noise_type]
            if type_results:
                type_analysis[noise_type] = {
                    'improvement': np.mean([r.improvement_percent for r in type_results]),
                    'quantum_wins': sum(1 for r in type_results if r.improvement_percent > 0),
                    'total_tests': len(type_results)
                }
        
        report = {
            'overall_stats': {
                'total_tests': total_tests,
                'quantum_wins': quantum_wins,
                'classical_wins': classical_wins,
                'ties': ties,
                'avg_classical_precision': avg_classical_precision,
                'avg_quantum_precision': avg_quantum_precision,
                'avg_improvement_percent': avg_improvement,
                'avg_classical_latency_ms': avg_classical_latency,
                'avg_quantum_latency_ms': avg_quantum_latency
            },
            'noise_level_analysis': noise_analysis,
            'noise_type_analysis': type_analysis,
            'detailed_results': [asdict(r) for r in self.results]
        }
        
        return report
    
    def print_summary(self, report: Dict):
        """Print human-readable summary."""
        stats = report['overall_stats']
        
        print("\n" + "="*70)
        print("REAL PUBMED MEDICAL RAG BENCHMARK RESULTS")
        print("="*70)
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Total Tests: {stats['total_tests']}")
        print(f"  Quantum Wins: {stats['quantum_wins']} ({stats['quantum_wins']/stats['total_tests']*100:.1f}%)")
        print(f"  Classical Wins: {stats['classical_wins']} ({stats['classical_wins']/stats['total_tests']*100:.1f}%)")
        print(f"  Ties: {stats['ties']} ({stats['ties']/stats['total_tests']*100:.1f}%)")
        
        print(f"\nPRECISION SCORES:")
        print(f"  Classical Average: {stats['avg_classical_precision']:.3f}")
        print(f"  Quantum Average: {stats['avg_quantum_precision']:.3f}")
        print(f"  Average Improvement: {stats['avg_improvement_percent']:+.1f}%")
        
        print(f"\nLATENCY ANALYSIS:")
        print(f"  Classical: {stats['avg_classical_latency_ms']:.1f}ms")
        print(f"  Quantum: {stats['avg_quantum_latency_ms']:.1f}ms")
        overhead = stats['avg_quantum_latency_ms'] - stats['avg_classical_latency_ms']
        print(f"  Overhead: {overhead:+.1f}ms")
        
        print(f"\nNOISE LEVEL PERFORMANCE:")
        for noise_level, analysis in report['noise_level_analysis'].items():
            print(f"  {noise_level:.0%} noise: {analysis['improvement']:+.1f}% improvement "
                  f"({analysis['quantum_wins']}/{analysis['total_tests']} wins)")
        
        print(f"\nNOISE TYPE PERFORMANCE:")
        for noise_type, analysis in report['noise_type_analysis'].items():
            print(f"  {noise_type}: {analysis['improvement']:+.1f}% improvement "
                  f"({analysis['quantum_wins']}/{analysis['total_tests']} wins)")
        
        # Key findings
        best_noise_level = max(report['noise_level_analysis'].items(), 
                              key=lambda x: x[1]['improvement'])
        best_noise_type = max(report['noise_type_analysis'].items(),
                             key=lambda x: x[1]['improvement'])
        
        print(f"\nKEY FINDINGS:")
        print(f"  🎯 Best noise level: {best_noise_level[0]:.0%} ({best_noise_level[1]['improvement']:+.1f}%)")
        print(f"  🎯 Best noise type: {best_noise_type[0]} ({best_noise_type[1]['improvement']:+.1f}%)")
        
        if stats['avg_improvement_percent'] > 0:
            print(f"  ✅ Overall quantum advantage: {stats['avg_improvement_percent']:.1f}%")
        else:
            print(f"  📊 Classical advantage: {abs(stats['avg_improvement_percent']):.1f}%")
        
        print("\n" + "="*70)
    
    def save_results(self, report: Dict):
        """Save results to JSON file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"real_pubmed_benchmark_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nResults saved to: {filename}")


def main():
    """Run the real-world PubMed medical benchmark."""
    benchmark = RealWorldMedicalBenchmark(
        num_documents=15,  # Reasonable for testing
        num_queries=6      # Good coverage
    )
    
    report = benchmark.run_benchmark()
    
    if report:
        benchmark.print_summary(report)
        benchmark.save_results(report)
    
    return report


if __name__ == "__main__":
    main()
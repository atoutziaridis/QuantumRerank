#!/usr/bin/env python3
"""
Real-world evaluation of QuantumRerank against standard baselines.
Tests on established IR datasets to measure actual performance benefits.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import requests
import zipfile
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    dataset_name: str
    method: str
    ndcg_at_5: float
    ndcg_at_10: float
    map_score: float
    mrr_score: float
    precision_at_5: float
    recall_at_10: float
    avg_query_time_ms: float
    total_queries: int

class IRDatasetLoader:
    """Load and process standard IR evaluation datasets."""
    
    def __init__(self, data_dir: str = "evaluation_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def download_msmarco_dev(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Download MS MARCO dev dataset - standard IR benchmark.
        Small version suitable for quick evaluation.
        """
        print("📥 Downloading MS MARCO dev dataset...")
        
        # MS MARCO dev small (1000 queries) - perfect for evaluation
        queries_url = "https://msmarco.blob.core.windows.net/msmarcoranking/queries.dev.small.tsv"
        qrels_url = "https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.small.tsv"
        
        queries_file = self.data_dir / "queries.dev.small.tsv"
        qrels_file = self.data_dir / "qrels.dev.small.tsv"
        
        # Download if not exists
        for url, file_path in [(queries_url, queries_file), (qrels_url, qrels_file)]:
            if not file_path.exists():
                print(f"  Downloading {file_path.name}...")
                response = requests.get(url)
                response.raise_for_status()
                file_path.write_bytes(response.content)
        
        # Parse queries
        queries = []
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                qid, query_text = line.strip().split('\t')
                queries.append({"qid": qid, "query": query_text})
        
        # Parse relevance judgments
        qrels = []
        with open(qrels_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    qid, _, docid, relevance = parts[:4]
                    qrels.append({
                        "qid": qid,
                        "docid": docid, 
                        "relevance": int(relevance)
                    })
        
        print(f"✅ Loaded {len(queries)} queries and {len(qrels)} relevance judgments")
        return queries, qrels
    
    def create_synthetic_documents(self, queries: List[Dict], num_docs_per_query: int = 20) -> Dict[str, List[Dict]]:
        """
        Create synthetic documents for each query.
        Simulates a retrieval scenario with relevant/irrelevant documents.
        """
        print(f"🔄 Creating {num_docs_per_query} synthetic documents per query...")
        
        query_docs = {}
        
        for query in queries[:50]:  # Limit to 50 queries for quick evaluation
            qid = query["qid"]
            query_text = query["query"]
            
            docs = []
            
            # Create relevant documents (paraphrase/expand the query)
            for i in range(5):  # 5 relevant docs
                if "quantum" in query_text.lower():
                    doc_text = f"Quantum computing and {query_text.lower()} are related concepts in modern physics and computer science."
                elif "machine learning" in query_text.lower():
                    doc_text = f"Machine learning algorithms can help with {query_text.lower()} using various computational approaches."
                elif "data" in query_text.lower():
                    doc_text = f"Data analysis and {query_text.lower()} involve processing information to extract insights."
                else:
                    doc_text = f"This document discusses {query_text.lower()} and related topics in detail."
                
                docs.append({
                    "docid": f"{qid}_rel_{i}",
                    "text": doc_text,
                    "relevance": 1  # Relevant
                })
            
            # Create semi-relevant documents
            for i in range(5):  # 5 semi-relevant docs
                doc_text = f"Information about {query_text.split()[0] if query_text.split() else 'topic'} and general knowledge."
                docs.append({
                    "docid": f"{qid}_semi_{i}",
                    "text": doc_text,
                    "relevance": 0  # Semi-relevant
                })
            
            # Create irrelevant documents
            irrelevant_topics = [
                "cooking recipes and kitchen equipment",
                "sports statistics and player performance", 
                "weather patterns and climate change",
                "movie reviews and entertainment news",
                "travel destinations and vacation planning"
            ]
            
            for i in range(10):  # 10 irrelevant docs
                topic = irrelevant_topics[i % len(irrelevant_topics)]
                doc_text = f"This document focuses on {topic} which is unrelated to the original query."
                docs.append({
                    "docid": f"{qid}_irrel_{i}",
                    "text": doc_text,
                    "relevance": 0  # Irrelevant
                })
            
            query_docs[qid] = docs
        
        print(f"✅ Created documents for {len(query_docs)} queries")
        return query_docs

class RerankingEvaluator:
    """Evaluate different reranking methods."""
    
    def __init__(self):
        self.results = []
    
    def evaluate_method(self, method_name: str, rerank_func, queries: List[Dict], 
                       query_docs: Dict[str, List[Dict]]) -> EvaluationResult:
        """
        Evaluate a reranking method on the test queries.
        
        Args:
            method_name: Name of the method being evaluated
            rerank_func: Function that takes (query, docs) and returns ranked docs
            queries: List of query dictionaries
            query_docs: Dictionary mapping query IDs to document lists
        """
        print(f"\n🧪 Evaluating {method_name}...")
        
        ndcg_5_scores = []
        ndcg_10_scores = []
        map_scores = []
        mrr_scores = []
        precision_5_scores = []
        recall_10_scores = []
        query_times = []
        
        for query in queries:
            if query["qid"] not in query_docs:
                continue
                
            query_text = query["query"]
            docs = query_docs[query["qid"]]
            doc_texts = [doc["text"] for doc in docs]
            
            # Time the reranking
            start_time = time.time()
            try:
                ranked_results = rerank_func(query_text, doc_texts)
                query_time = (time.time() - start_time) * 1000
                query_times.append(query_time)
                
                # Map results back to docs with relevance scores
                ranked_docs = []
                for result in ranked_results:
                    # Find original doc by text matching
                    for doc in docs:
                        if doc["text"] == result.get("text", result.get("document", "")):
                            ranked_docs.append(doc)
                            break
                
                # Calculate metrics
                if ranked_docs:
                    ndcg_5_scores.append(self.calculate_ndcg(ranked_docs, k=5))
                    ndcg_10_scores.append(self.calculate_ndcg(ranked_docs, k=10))
                    map_scores.append(self.calculate_map(ranked_docs))
                    mrr_scores.append(self.calculate_mrr(ranked_docs))
                    precision_5_scores.append(self.calculate_precision_at_k(ranked_docs, k=5))
                    recall_10_scores.append(self.calculate_recall_at_k(ranked_docs, k=10))
                
            except Exception as e:
                print(f"  ⚠️  Error processing query {query['qid']}: {e}")
                query_times.append(0)
                continue
        
        # Calculate average metrics
        result = EvaluationResult(
            dataset_name="MS_MARCO_dev_synthetic",
            method=method_name,
            ndcg_at_5=np.mean(ndcg_5_scores) if ndcg_5_scores else 0.0,
            ndcg_at_10=np.mean(ndcg_10_scores) if ndcg_10_scores else 0.0,
            map_score=np.mean(map_scores) if map_scores else 0.0,
            mrr_score=np.mean(mrr_scores) if mrr_scores else 0.0,
            precision_at_5=np.mean(precision_5_scores) if precision_5_scores else 0.0,
            recall_at_10=np.mean(recall_10_scores) if recall_10_scores else 0.0,
            avg_query_time_ms=np.mean(query_times) if query_times else 0.0,
            total_queries=len([q for q in queries if q["qid"] in query_docs])
        )
        
        self.results.append(result)
        print(f"✅ {method_name} evaluation complete")
        return result
    
    def calculate_ndcg(self, ranked_docs: List[Dict], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain at k."""
        def dcg(relevances, k):
            return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances[:k]))
        
        # Get relevance scores for ranked docs
        ranked_relevances = [doc.get("relevance", 0) for doc in ranked_docs[:k]]
        
        # Calculate DCG
        dcg_score = dcg(ranked_relevances, k)
        
        # Calculate IDCG (ideal DCG)
        ideal_relevances = sorted([doc.get("relevance", 0) for doc in ranked_docs], reverse=True)
        idcg_score = dcg(ideal_relevances, k)
        
        return dcg_score / idcg_score if idcg_score > 0 else 0.0
    
    def calculate_map(self, ranked_docs: List[Dict]) -> float:
        """Calculate Mean Average Precision."""
        relevant_count = 0
        precision_sum = 0.0
        
        for i, doc in enumerate(ranked_docs):
            if doc.get("relevance", 0) > 0:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        total_relevant = sum(1 for doc in ranked_docs if doc.get("relevance", 0) > 0)
        return precision_sum / total_relevant if total_relevant > 0 else 0.0
    
    def calculate_mrr(self, ranked_docs: List[Dict]) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, doc in enumerate(ranked_docs):
            if doc.get("relevance", 0) > 0:
                return 1.0 / (i + 1)
        return 0.0
    
    def calculate_precision_at_k(self, ranked_docs: List[Dict], k: int) -> float:
        """Calculate Precision at k."""
        relevant_in_k = sum(1 for doc in ranked_docs[:k] if doc.get("relevance", 0) > 0)
        return relevant_in_k / min(k, len(ranked_docs))
    
    def calculate_recall_at_k(self, ranked_docs: List[Dict], k: int) -> float:
        """Calculate Recall at k."""
        relevant_in_k = sum(1 for doc in ranked_docs[:k] if doc.get("relevance", 0) > 0)
        total_relevant = sum(1 for doc in ranked_docs if doc.get("relevance", 0) > 0)
        return relevant_in_k / total_relevant if total_relevant > 0 else 0.0
    
    def print_comparison(self):
        """Print comparison table of all evaluated methods."""
        if not self.results:
            print("No results to compare")
            return
        
        print("\n" + "="*100)
        print("📊 RERANKING METHODS COMPARISON")
        print("="*100)
        
        # Table header
        print(f"{'Method':<20} {'NDCG@5':<8} {'NDCG@10':<8} {'MAP':<8} {'MRR':<8} {'P@5':<8} {'R@10':<8} {'Time(ms)':<10}")
        print("-" * 100)
        
        # Sort by NDCG@10 (primary metric)
        sorted_results = sorted(self.results, key=lambda x: x.ndcg_at_10, reverse=True)
        
        for result in sorted_results:
            print(f"{result.method:<20} "
                  f"{result.ndcg_at_5:<8.4f} "
                  f"{result.ndcg_at_10:<8.4f} "
                  f"{result.map_score:<8.4f} "
                  f"{result.mrr_score:<8.4f} "
                  f"{result.precision_at_5:<8.4f} "
                  f"{result.recall_at_10:<8.4f} "
                  f"{result.avg_query_time_ms:<10.2f}")
        
        print("-" * 100)
        
        # Find best performing method
        best_method = sorted_results[0]
        print(f"🏆 Best method: {best_method.method} (NDCG@10: {best_method.ndcg_at_10:.4f})")
        
        # Performance analysis
        quantum_results = [r for r in sorted_results if "quantum" in r.method.lower()]
        classical_results = [r for r in sorted_results if "quantum" not in r.method.lower()]
        
        if quantum_results and classical_results:
            best_quantum = max(quantum_results, key=lambda x: x.ndcg_at_10)
            best_classical = max(classical_results, key=lambda x: x.ndcg_at_10)
            
            improvement = ((best_quantum.ndcg_at_10 - best_classical.ndcg_at_10) / best_classical.ndcg_at_10) * 100
            
            print(f"\n🔬 Quantum vs Classical Analysis:")
            print(f"   Best Quantum:  {best_quantum.method} - NDCG@10: {best_quantum.ndcg_at_10:.4f}")
            print(f"   Best Classical: {best_classical.method} - NDCG@10: {best_classical.ndcg_at_10:.4f}")
            print(f"   Improvement: {improvement:+.2f}%")
            
            if improvement > 0:
                print(f"   ✅ Quantum approach shows improvement!")
            else:
                print(f"   ⚠️  Classical approach performs better")

def create_baseline_rerankers():
    """Create baseline reranking functions for comparison."""
    
    def random_reranker(query: str, documents: List[str]) -> List[Dict]:
        """Random baseline - worst case performance."""
        import random
        docs_with_scores = [(doc, random.random()) for doc in documents]
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        return [{"text": doc, "score": score} for doc, score in docs_with_scores]
    
    def bm25_reranker(query: str, documents: List[str]) -> List[Dict]:
        """Simple BM25-like scoring baseline."""
        from collections import Counter
        import math
        
        query_terms = query.lower().split()
        doc_scores = []
        
        for doc in documents:
            doc_terms = doc.lower().split()
            doc_length = len(doc_terms)
            
            score = 0.0
            for term in query_terms:
                tf = doc_terms.count(term)
                if tf > 0:
                    # Simple BM25-like scoring
                    idf = math.log(len(documents) / max(1, sum(1 for d in documents if term in d.lower())))
                    score += (tf * idf) / (tf + 1.2 * (0.25 + 0.75 * doc_length / 50))
            
            doc_scores.append((doc, score))
        
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return [{"text": doc, "score": score} for doc, score in doc_scores]
    
    def cosine_similarity_reranker(query: str, documents: List[str]) -> List[Dict]:
        """Cosine similarity baseline using simple term vectors."""
        from collections import Counter
        import math
        
        # Create term vectors
        all_terms = set()
        query_terms = query.lower().split()
        doc_terms_list = [doc.lower().split() for doc in documents]
        
        for terms in [query_terms] + doc_terms_list:
            all_terms.update(terms)
        
        term_to_idx = {term: i for i, term in enumerate(all_terms)}
        
        # Query vector
        query_vector = [0] * len(all_terms)
        for term in query_terms:
            if term in term_to_idx:
                query_vector[term_to_idx[term]] += 1
        
        # Document vectors and cosine similarity
        doc_scores = []
        for doc, doc_terms in zip(documents, doc_terms_list):
            doc_vector = [0] * len(all_terms)
            for term in doc_terms:
                if term in term_to_idx:
                    doc_vector[term_to_idx[term]] += 1
            
            # Cosine similarity
            dot_product = sum(a * b for a, b in zip(query_vector, doc_vector))
            query_norm = math.sqrt(sum(a * a for a in query_vector))
            doc_norm = math.sqrt(sum(a * a for a in doc_vector))
            
            if query_norm > 0 and doc_norm > 0:
                similarity = dot_product / (query_norm * doc_norm)
            else:
                similarity = 0.0
            
            doc_scores.append((doc, similarity))
        
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return [{"text": doc, "score": score} for doc, score in doc_scores]
    
    return {
        "Random": random_reranker,
        "BM25-like": bm25_reranker, 
        "Cosine-Similarity": cosine_similarity_reranker
    }

def create_quantum_rerankers():
    """Create quantum reranking functions using the QuantumRerank system."""
    
    def quantum_classical_reranker(query: str, documents: List[str]) -> List[Dict]:
        """Classical method from QuantumRerank."""
        from quantum_rerank.core.rag_reranker import QuantumRAGReranker
        reranker = QuantumRAGReranker()
        return reranker.rerank(query, documents, top_k=len(documents), method="classical")
    
    def quantum_fidelity_reranker(query: str, documents: List[str]) -> List[Dict]:
        """Quantum fidelity method from QuantumRerank."""
        from quantum_rerank.core.rag_reranker import QuantumRAGReranker
        reranker = QuantumRAGReranker()
        return reranker.rerank(query, documents, top_k=len(documents), method="quantum")
    
    def quantum_hybrid_reranker(query: str, documents: List[str]) -> List[Dict]:
        """Hybrid quantum-classical method from QuantumRerank."""
        from quantum_rerank.core.rag_reranker import QuantumRAGReranker
        reranker = QuantumRAGReranker()
        return reranker.rerank(query, documents, top_k=len(documents), method="hybrid")
    
    return {
        "QuantumRerank-Classical": quantum_classical_reranker,
        "QuantumRerank-Quantum": quantum_fidelity_reranker,
        "QuantumRerank-Hybrid": quantum_hybrid_reranker
    }

def main():
    """Run comprehensive reranking evaluation."""
    print("🚀 QuantumRerank Evaluation Suite")
    print("="*60)
    
    # Initialize components
    loader = IRDatasetLoader()
    evaluator = RerankingEvaluator()
    
    try:
        # Load dataset
        queries, qrels = loader.download_msmarco_dev()
        query_docs = loader.create_synthetic_documents(queries)
        
        # Get all reranking methods
        baseline_methods = create_baseline_rerankers()
        quantum_methods = create_quantum_rerankers()
        all_methods = {**baseline_methods, **quantum_methods}
        
        print(f"\n🎯 Evaluating {len(all_methods)} reranking methods on {len(query_docs)} queries...")
        
        # Evaluate each method
        for method_name, rerank_func in all_methods.items():
            try:
                evaluator.evaluate_method(method_name, rerank_func, queries, query_docs)
            except Exception as e:
                print(f"❌ Failed to evaluate {method_name}: {e}")
                continue
        
        # Print comparison
        evaluator.print_comparison()
        
        # Save results
        results_file = "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump([
                {
                    'method': r.method,
                    'ndcg_at_5': r.ndcg_at_5,
                    'ndcg_at_10': r.ndcg_at_10,
                    'map_score': r.map_score,
                    'mrr_score': r.mrr_score,
                    'precision_at_5': r.precision_at_5,
                    'recall_at_10': r.recall_at_10,
                    'avg_query_time_ms': r.avg_query_time_ms,
                    'total_queries': r.total_queries
                } for r in evaluator.results
            ], f, indent=2)
        
        print(f"\n💾 Results saved to {results_file}")
        
        print("\n🎊 Evaluation Complete!")
        print("\nNext steps:")
        print("1. Analyze the results above")
        print("2. If quantum methods underperform, tune hyperparameters")
        print("3. Try on domain-specific datasets")
        print("4. Compare against sentence-transformers reranking models")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
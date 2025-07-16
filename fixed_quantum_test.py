#!/usr/bin/env python3
"""
Fixed test of actual quantum vs classical performance with real implementations.
"""

import sys
import time
import numpy as np
from typing import List, Tuple, Dict
import math

# Add project root to path
sys.path.append('.')

from quantum_rerank.core.quantum_similarity_engine import QuantumSimilarityEngine, SimilarityEngineConfig, SimilarityMethod
from quantum_rerank.core.embeddings import EmbeddingProcessor


def create_medical_test_queries_and_docs():
    """Create realistic medical test data."""
    queries = [
        "treatment options for diabetes mellitus type 2",
        "emergency management of anaphylactic shock",
        "differential diagnosis of chest pain",
        "postoperative care after cardiac surgery",
        "anticoagulation therapy in atrial fibrillation",
        "management of acute stroke symptoms",
        "pneumonia vs pulmonary edema differentiation",
        "hypertension medication dosing guidelines",
        "chronic kidney disease progression monitoring",
        "medication interactions with warfarin"
    ]
    
    documents = [
        "Diabetes mellitus type 2 management includes metformin as first-line therapy, lifestyle modifications, and regular monitoring of HbA1c levels",
        "Anaphylactic shock requires immediate epinephrine administration (0.3-0.5 mg IM), IV fluid resuscitation, and continuous monitoring",
        "Acute coronary syndrome presents with chest pain, ST-segment changes on ECG, and elevated cardiac biomarkers including troponin",
        "Post-cardiac surgery care involves pain management, wound monitoring, anticoagulation, and gradual activity progression",
        "Atrial fibrillation anticoagulation depends on CHA2DS2-VASc score with warfarin or direct oral anticoagulants recommended",
        "Acute stroke management requires rapid assessment, CT imaging, and potential thrombolytic therapy within therapeutic window",
        "Pneumonia typically shows focal consolidation on chest X-ray while pulmonary edema demonstrates bilateral infiltrates",
        "Hypertension treatment follows stepped approach: ACE inhibitors, calcium channel blockers, and thiazide diuretics",
        "Chronic kidney disease monitoring includes GFR calculation, proteinuria assessment, and electrolyte management",
        "Warfarin interactions are common with antibiotics, NSAIDs, and many medications requiring INR monitoring",
        
        # Add some less relevant documents for contrast
        "General wellness requires balanced nutrition, regular exercise, and adequate sleep for optimal health outcomes",
        "Healthcare quality improvement focuses on patient safety, clinical outcomes, and cost-effective care delivery",
        "Medical education emphasizes evidence-based practice, clinical reasoning, and continuous professional development",
        "Hospital administration involves resource allocation, staff management, and regulatory compliance requirements",
        "Medical research methodology includes randomized controlled trials, systematic reviews, and meta-analyses"
    ]
    
    # Create relevance judgments (0=irrelevant, 1=somewhat relevant, 2=relevant, 3=highly relevant)
    relevance = {}
    for i, query in enumerate(queries):
        relevance[f"q_{i}"] = {}
        for j, doc in enumerate(documents):
            if j == i and j < 10:  # Perfect match for first 10 docs
                relevance[f"q_{i}"][f"d_{j}"] = 3
            elif j < 10 and abs(i - j) <= 2:  # Close medical topics
                relevance[f"q_{i}"][f"d_{j}"] = 2
            elif j < 10:  # Other medical content
                relevance[f"q_{i}"][f"d_{j}"] = 1
            else:  # General health/admin content
                relevance[f"q_{i}"][f"d_{j}"] = 0
    
    return queries, documents, relevance


def compute_ir_metrics(rankings: List[Tuple[str, float]], relevance_scores: Dict[str, int], k_values: List[int] = [5, 10]) -> Dict[str, float]:
    """Compute standard IR metrics."""
    metrics = {}
    
    # Extract ranked document IDs
    ranked_docs = [doc_id for doc_id, score in rankings]
    
    # Calculate NDCG@k
    for k in k_values:
        dcg = 0.0
        for i, doc_id in enumerate(ranked_docs[:k]):
            if doc_id in relevance_scores:
                rel = relevance_scores[doc_id]
                dcg += (2**rel - 1) / np.log2(i + 2)
        
        # Ideal DCG
        ideal_rels = sorted(relevance_scores.values(), reverse=True)
        idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_rels[:k]))
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        metrics[f'ndcg@{k}'] = ndcg
    
    # Calculate Precision@k
    for k in k_values:
        relevant_at_k = sum(1 for doc_id in ranked_docs[:k] 
                           if doc_id in relevance_scores and relevance_scores[doc_id] >= 2)
        metrics[f'precision@{k}'] = relevant_at_k / k
    
    # Calculate MRR
    for i, doc_id in enumerate(ranked_docs):
        if doc_id in relevance_scores and relevance_scores[doc_id] >= 2:
            metrics['mrr'] = 1.0 / (i + 1)
            break
    else:
        metrics['mrr'] = 0.0
    
    # Calculate MAP
    relevant_docs = [doc_id for doc_id in relevance_scores if relevance_scores[doc_id] >= 2]
    if relevant_docs:
        ap = 0.0
        relevant_found = 0
        for i, doc_id in enumerate(ranked_docs):
            if doc_id in relevance_scores and relevance_scores[doc_id] >= 2:
                relevant_found += 1
                ap += relevant_found / (i + 1)
        metrics['map'] = ap / len(relevant_docs)
    else:
        metrics['map'] = 0.0
    
    return metrics


class SimpleBM25:
    """Simple BM25 implementation for comparison."""
    
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_freqs = []
        self.idf = {}
        self.doc_lens = []
        self.avgdl = 0
        
    def fit(self, corpus):
        """Fit BM25 on corpus."""
        self.corpus = corpus
        self.doc_lens = [len(doc.split()) for doc in corpus]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens)
        
        # Calculate document frequencies
        df = {}
        for doc in corpus:
            terms = set(doc.lower().split())
            for term in terms:
                df[term] = df.get(term, 0) + 1
        
        # Calculate IDF
        N = len(corpus)
        self.idf = {term: math.log((N - df[term] + 0.5) / (df[term] + 0.5)) 
                   for term in df}
        
        # Calculate term frequencies for each document
        self.doc_freqs = []
        for doc in corpus:
            tf = {}
            terms = doc.lower().split()
            for term in terms:
                tf[term] = tf.get(term, 0) + 1
            self.doc_freqs.append(tf)
    
    def get_scores(self, query):
        """Get BM25 scores for query."""
        query_terms = query.lower().split()
        scores = []
        
        for i, doc_tf in enumerate(self.doc_freqs):
            score = 0
            doc_len = self.doc_lens[i]
            
            for term in query_terms:
                if term in doc_tf and term in self.idf:
                    tf = doc_tf[term]
                    idf = self.idf[term]
                    
                    # BM25 formula
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    score += idf * numerator / denominator
            
            scores.append(score)
        
        return scores


def test_quantum_vs_classical_fixed():
    """Test actual quantum vs classical performance with fixed implementations."""
    
    print("🔬 REAL QUANTUM vs CLASSICAL PERFORMANCE TEST (FIXED)")
    print("=" * 65)
    
    # Create test data
    queries, documents, relevance_judgments = create_medical_test_queries_and_docs()
    print(f"Test data: {len(queries)} queries, {len(documents)} documents")
    
    # Initialize quantum method
    print("\n🌀 Initializing Quantum Method...")
    quantum_config = SimilarityEngineConfig(
        n_qubits=4,
        n_layers=2,
        similarity_method=SimilarityMethod.HYBRID_WEIGHTED,
        enable_caching=True
    )
    quantum_engine = QuantumSimilarityEngine(quantum_config)
    
    # Initialize classical baselines
    print("⚡ Initializing Classical Baselines...")
    embedding_processor = EmbeddingProcessor()
    bm25 = SimpleBM25()
    bm25.fit(documents)
    
    # Test each method
    methods = {
        'Quantum (Hybrid)': 'quantum_hybrid',
        'Quantum (Pure)': 'quantum_pure', 
        'Classical (BERT)': 'classical_bert',
        'BM25': 'bm25'
    }
    
    results = {}
    
    for method_name, method_key in methods.items():
        print(f"\n📊 Testing {method_name}...")
        
        all_metrics = []
        total_time = 0.0
        
        for i, query in enumerate(queries[:5]):  # Test on first 5 queries for speed
            query_id = f"q_{i}"
            query_relevance = relevance_judgments[query_id]
            
            start_time = time.time()
            
            try:
                if method_key == 'quantum_hybrid':
                    # Test quantum hybrid method
                    similarities = quantum_engine.compute_similarities_batch(
                        query, documents, SimilarityMethod.HYBRID_WEIGHTED
                    )
                    rankings = [(f"d_{j}", sim_score) for j, (doc, sim_score, meta) in enumerate(similarities)]
                    
                elif method_key == 'quantum_pure':
                    # Test pure quantum method
                    similarities = quantum_engine.compute_similarities_batch(
                        query, documents, SimilarityMethod.QUANTUM_FIDELITY
                    )
                    rankings = [(f"d_{j}", sim_score) for j, (doc, sim_score, meta) in enumerate(similarities)]
                    
                elif method_key == 'classical_bert':
                    # Test classical BERT similarity
                    similarities = quantum_engine.compute_similarities_batch(
                        query, documents, SimilarityMethod.CLASSICAL_COSINE
                    )
                    rankings = [(f"d_{j}", sim_score) for j, (doc, sim_score, meta) in enumerate(similarities)]
                    
                elif method_key == 'bm25':
                    # Test BM25
                    scores = bm25.get_scores(query)
                    rankings = [(f"d_{j}", score) for j, score in enumerate(scores)]
                    rankings.sort(key=lambda x: x[1], reverse=True)
                
                query_time = time.time() - start_time
                total_time += query_time
                
                # Calculate metrics
                metrics = compute_ir_metrics(rankings, query_relevance)
                metrics['latency_ms'] = query_time * 1000
                all_metrics.append(metrics)
                
                print(f"  Query {i+1}: NDCG@10={metrics.get('ndcg@10', 0):.3f}, "
                      f"Latency={metrics['latency_ms']:.1f}ms")
                
            except Exception as e:
                print(f"  ❌ Failed on query {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Aggregate results
        if all_metrics:
            avg_metrics = {}
            for metric in ['ndcg@5', 'ndcg@10', 'precision@5', 'precision@10', 'mrr', 'map', 'latency_ms']:
                values = [m.get(metric, 0) for m in all_metrics if metric in m]
                avg_metrics[metric] = np.mean(values) if values else 0.0
            
            results[method_name] = avg_metrics
            
            print(f"  ✅ Average: NDCG@10={avg_metrics['ndcg@10']:.3f}, "
                  f"P@5={avg_metrics['precision@5']:.3f}, "
                  f"Latency={avg_metrics['latency_ms']:.1f}ms")
        else:
            results[method_name] = {'error': 'All queries failed'}
    
    # Print final comparison
    print("\n" + "=" * 65)
    print("📈 FINAL PERFORMANCE COMPARISON")
    print("=" * 65)
    
    print(f"{'Method':<20} {'NDCG@10':<10} {'P@5':<8} {'MRR':<8} {'MAP':<8} {'Latency(ms)':<12}")
    print("-" * 85)
    
    for method_name, metrics in results.items():
        if 'error' not in metrics:
            print(f"{method_name:<20} {metrics['ndcg@10']:<10.3f} {metrics['precision@5']:<8.3f} "
                  f"{metrics['mrr']:<8.3f} {metrics['map']:<8.3f} {metrics['latency_ms']:<12.1f}")
        else:
            print(f"{method_name:<20} {'FAILED':<10} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<12}")
    
    print("-" * 85)
    
    # Analysis
    print("\n🎯 ANALYSIS:")
    
    if len([r for r in results.values() if 'error' not in r]) >= 2:
        # Find best performing method by NDCG@10
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        best_method = max(valid_results.keys(), key=lambda k: valid_results[k]['ndcg@10'])
        best_score = valid_results[best_method]['ndcg@10']
        
        print(f"🏆 Best performance: {best_method} (NDCG@10={best_score:.3f})")
        
        # Compare quantum methods to classical
        quantum_methods = [k for k in valid_results.keys() if 'Quantum' in k]
        classical_methods = [k for k in valid_results.keys() if 'Quantum' not in k]
        
        if quantum_methods and classical_methods:
            best_quantum = max(quantum_methods, key=lambda k: valid_results[k]['ndcg@10'])
            best_classical = max(classical_methods, key=lambda k: valid_results[k]['ndcg@10'])
            
            quantum_score = valid_results[best_quantum]['ndcg@10']
            classical_score = valid_results[best_classical]['ndcg@10']
            quantum_latency = valid_results[best_quantum]['latency_ms']
            classical_latency = valid_results[best_classical]['latency_ms']
            
            print(f"🌀 Best quantum: {best_quantum} (NDCG@10={quantum_score:.3f}, {quantum_latency:.1f}ms)")
            print(f"⚡ Best classical: {best_classical} (NDCG@10={classical_score:.3f}, {classical_latency:.1f}ms)")
            
            if quantum_score > classical_score:
                advantage = quantum_score - classical_score
                latency_ratio = quantum_latency / classical_latency
                print(f"✅ Quantum advantage: +{advantage:.3f} NDCG@10 ({latency_ratio:.1f}x latency cost)")
                
                if advantage >= 0.05:  # 5% improvement threshold
                    print(f"🚀 SIGNIFICANT quantum advantage detected!")
                elif advantage >= 0.02:  # 2% improvement threshold  
                    print(f"📈 MODERATE quantum advantage detected")
                else:
                    print(f"⚖️  MARGINAL quantum advantage detected")
                    
            else:
                disadvantage = classical_score - quantum_score
                latency_ratio = quantum_latency / classical_latency
                print(f"❌ No quantum advantage: -{disadvantage:.3f} NDCG@10 ({latency_ratio:.1f}x latency cost)")
                
                if disadvantage <= 0.02:
                    print(f"⚖️  Results are essentially equivalent within margin of error")
                else:
                    print(f"📉 Classical methods significantly outperform quantum")
    
    # Additional insights
    print("\n💡 INSIGHTS:")
    
    if 'Quantum (Pure)' in results and 'Quantum (Hybrid)' in results:
        pure_results = results['Quantum (Pure)']
        hybrid_results = results['Quantum (Hybrid)']
        
        if 'error' not in pure_results and 'error' not in hybrid_results:
            pure_score = pure_results['ndcg@10']
            hybrid_score = hybrid_results['ndcg@10']
            pure_latency = pure_results['latency_ms']
            hybrid_latency = hybrid_results['latency_ms']
            
            print(f"🔄 Pure quantum vs Hybrid:")
            print(f"   Pure: NDCG@10={pure_score:.3f}, Latency={pure_latency:.1f}ms")
            print(f"   Hybrid: NDCG@10={hybrid_score:.3f}, Latency={hybrid_latency:.1f}ms")
            
            if abs(pure_score - hybrid_score) < 0.01:
                print(f"   ⚖️  Similar accuracy, pure quantum is {hybrid_latency/pure_latency:.1f}x faster")
            elif pure_score > hybrid_score:
                print(f"   🌀 Pure quantum performs better by {pure_score - hybrid_score:.3f}")
            else:
                print(f"   🔄 Hybrid performs better by {hybrid_score - pure_score:.3f}")
    
    return results


if __name__ == "__main__":
    results = test_quantum_vs_classical_fixed()
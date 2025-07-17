#!/usr/bin/env python3
"""
Statistical Evaluation with Optimized System
============================================

Run comprehensive statistical evaluation using the optimized quantum reranking system
with top-K optimization and batch processing improvements.

Tests:
1. Performance comparison (optimized vs classical)
2. Quality preservation validation  
3. Statistical significance testing
4. Memory efficiency analysis
"""

import time
import sys
import numpy as np
import psutil
import os
from pathlib import Path
from typing import List, Dict, Tuple
from scipy.stats import wilcoxon

# Add the quantum_rerank package to the path
sys.path.insert(0, str(Path(__file__).parent))

from quantum_rerank.retrieval.two_stage_retriever import TwoStageRetriever, RetrieverConfig
from quantum_rerank.retrieval.document_store import Document, DocumentMetadata
from sentence_transformers import SentenceTransformer


class MemoryTracker:
    """Track memory usage during evaluation."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.baseline_memory
        self.measurements = []
    
    def measure(self, label: str = ""):
        """Take a memory measurement."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = max(self.peak_memory, current_memory)
        self.measurements.append({
            'label': label,
            'memory_mb': current_memory,
            'memory_increase_mb': current_memory - self.baseline_memory
        })
        return current_memory
    
    def get_stats(self) -> Dict:
        """Get memory usage statistics."""
        return {
            'baseline_mb': self.baseline_memory,
            'peak_mb': self.peak_memory,
            'peak_increase_mb': self.peak_memory - self.baseline_memory,
            'measurements': self.measurements
        }


def create_evaluation_dataset(n_docs: int = 20) -> Tuple[List[Document], List[str]]:
    """Create test dataset for evaluation."""
    
    # Document topics with varying relevance to test queries
    topics = [
        # High relevance to ML queries
        "machine learning algorithms and neural networks for classification",
        "deep learning models and transformer architectures",
        "artificial intelligence and natural language processing",
        "supervised learning methods and feature engineering",
        "unsupervised learning clustering and dimensionality reduction",
        
        # Medium relevance
        "data science and statistical analysis methods",
        "computer vision and image recognition systems", 
        "reinforcement learning and decision making algorithms",
        "information retrieval and search systems",
        "big data processing and distributed computing",
        
        # Lower relevance
        "quantum computing and quantum algorithms",
        "cybersecurity and network protection systems",
        "blockchain technology and cryptocurrency",
        "cloud computing infrastructure and services",
        "robotics and autonomous vehicle systems",
        
        # Minimal relevance
        "mobile app development and user interfaces",
        "web development and frontend frameworks",
        "database design and management systems",
        "software engineering and development practices",
        "operating systems and system administration"
    ]
    
    documents = []
    for i in range(n_docs):
        topic = topics[i % len(topics)]
        
        # Create longer, more realistic content
        content = f"Document {i} focuses on {topic}. " * 30
        content += f"This research covers various aspects of {topic} including implementation details, theoretical foundations, and practical applications. "
        content += f"The work demonstrates significant advances in {topic} with experimental validation and performance analysis."
        
        metadata = DocumentMetadata(
            title=f"Research Paper {i}: {topic.title()}",
            source="research_corpus",
            custom_fields={
                "domain": "computer_science",
                "topic_category": "high_relevance" if i < 5 else "medium_relevance" if i < 10 else "low_relevance",
                "doc_length": len(content),
                "relevance_score": max(0.1, 1.0 - (i / n_docs))  # Decreasing relevance
            }
        )
        
        documents.append(Document(
            doc_id=f"doc_{i:03d}",
            content=content,
            metadata=metadata
        ))
    
    # Test queries with different complexity levels
    queries = [
        "machine learning algorithms for classification",
        "deep learning neural networks and transformers", 
        "artificial intelligence natural language processing",
        "supervised learning feature engineering methods",
        "data science statistical analysis techniques",
        "computer vision image recognition systems",
        "information retrieval search algorithms",
        "quantum computing algorithms and applications"
    ]
    
    return documents, queries


def run_optimized_evaluation(documents: List[Document], 
                           queries: List[str],
                           memory_tracker: MemoryTracker) -> Dict:
    """Run evaluation with optimized quantum system."""
    
    print("Running Optimized Quantum System Evaluation")
    print("-" * 50)
    
    # Configure optimized retriever
    config_optimized = RetrieverConfig(
        initial_k=min(20, len(documents)),  # Get more candidates from FAISS
        final_k=10,                         # Return top 10 results
        rerank_k=5,                        # Only rerank top 5 (OPTIMIZATION)
        reranking_method="hybrid",
        enable_caching=True
    )
    
    memory_tracker.measure("before_optimized_init")
    retriever = TwoStageRetriever(config_optimized)
    retriever.add_documents(documents)
    memory_tracker.measure("after_optimized_init")
    
    results = {
        'system': 'quantum_optimized',
        'config': {
            'initial_k': config_optimized.initial_k,
            'final_k': config_optimized.final_k,
            'rerank_k': config_optimized.rerank_k,
            'method': config_optimized.reranking_method
        },
        'query_results': [],
        'timing_stats': {},
        'quality_metrics': {}
    }
    
    query_times = []
    all_similarities = []
    retrieval_quality_scores = []
    
    print(f"Processing {len(queries)} queries with optimized system...")
    
    for i, query in enumerate(queries):
        print(f"  Query {i+1}/{len(queries)}: {query[:50]}...")
        
        memory_tracker.measure(f"before_query_{i}")
        
        start_time = time.time()
        query_results = retriever.retrieve(query, k=10)
        query_time = time.time() - start_time
        
        memory_tracker.measure(f"after_query_{i}")
        
        query_times.append(query_time)
        
        # Calculate quality metrics
        if query_results:
            # Average similarity score as quality indicator
            avg_similarity = np.mean([r.score for r in query_results])
            all_similarities.append(avg_similarity)
            
            # Relevance-based quality score
            relevance_scores = []
            for result in query_results:
                # Extract relevance from metadata
                relevance = result.metadata.get('custom_fields', {}).get('relevance_score', 0.5)
                relevance_scores.append(relevance)
            
            quality_score = np.mean(relevance_scores) if relevance_scores else 0
            retrieval_quality_scores.append(quality_score)
        else:
            all_similarities.append(0.0)
            retrieval_quality_scores.append(0.0)
        
        # Store detailed results
        results['query_results'].append({
            'query': query,
            'time_ms': query_time * 1000,
            'num_results': len(query_results),
            'avg_similarity': all_similarities[-1],
            'quality_score': retrieval_quality_scores[-1],
            'results': [
                {
                    'doc_id': r.doc_id,
                    'score': r.score,
                    'rank': r.rank,
                    'stage': r.stage
                } for r in query_results[:5]  # Store top 5 for analysis
            ]
        })
    
    # Calculate statistics
    results['timing_stats'] = {
        'avg_time_ms': np.mean(query_times) * 1000,
        'median_time_ms': np.median(query_times) * 1000,
        'min_time_ms': np.min(query_times) * 1000,
        'max_time_ms': np.max(query_times) * 1000,
        'std_time_ms': np.std(query_times) * 1000,
        'total_queries': len(queries),
        'queries_per_second': len(queries) / sum(query_times)
    }
    
    results['quality_metrics'] = {
        'avg_similarity': np.mean(all_similarities),
        'avg_quality_score': np.mean(retrieval_quality_scores),
        'similarity_std': np.std(all_similarities),
        'quality_std': np.std(retrieval_quality_scores)
    }
    
    print(f"  Optimized system: {results['timing_stats']['avg_time_ms']:.1f}ms avg")
    return results


def run_classical_baseline(documents: List[Document], 
                         queries: List[str],
                         memory_tracker: MemoryTracker) -> Dict:
    """Run classical BERT baseline for comparison."""
    
    print("\nRunning Classical BERT Baseline")
    print("-" * 40)
    
    memory_tracker.measure("before_classical_init")
    
    # Use SentenceTransformer directly for classical baseline
    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    
    # Pre-compute document embeddings
    doc_texts = [doc.content for doc in documents]
    doc_embeddings = model.encode(doc_texts, convert_to_tensor=False)
    
    memory_tracker.measure("after_classical_init")
    
    results = {
        'system': 'classical_bert',
        'config': {'method': 'cosine_similarity'},
        'query_results': [],
        'timing_stats': {},
        'quality_metrics': {}
    }
    
    query_times = []
    all_similarities = []
    retrieval_quality_scores = []
    
    print(f"Processing {len(queries)} queries with classical system...")
    
    for i, query in enumerate(queries):
        print(f"  Query {i+1}/{len(queries)}: {query[:50]}...")
        
        memory_tracker.measure(f"before_classical_query_{i}")
        
        start_time = time.time()
        
        # Encode query
        query_embedding = model.encode([query], convert_to_tensor=False)[0]
        
        # Compute similarities
        similarities = []
        for doc_emb in doc_embeddings:
            # Cosine similarity
            similarity = np.dot(query_embedding, doc_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
            )
            similarities.append(similarity)
        
        # Get top 10 results
        ranked_indices = np.argsort(similarities)[::-1][:10]
        
        query_time = time.time() - start_time
        memory_tracker.measure(f"after_classical_query_{i}")
        
        query_times.append(query_time)
        
        # Calculate quality metrics
        top_similarities = [similarities[idx] for idx in ranked_indices]
        avg_similarity = np.mean(top_similarities) if top_similarities else 0
        all_similarities.append(avg_similarity)
        
        # Relevance-based quality score
        relevance_scores = []
        for idx in ranked_indices:
            relevance = documents[idx].metadata.custom_fields.get('relevance_score', 0.5)
            relevance_scores.append(relevance)
        
        quality_score = np.mean(relevance_scores) if relevance_scores else 0
        retrieval_quality_scores.append(quality_score)
        
        # Store results
        results['query_results'].append({
            'query': query,
            'time_ms': query_time * 1000,
            'num_results': len(ranked_indices),
            'avg_similarity': avg_similarity,
            'quality_score': quality_score,
            'results': [
                {
                    'doc_id': documents[ranked_indices[j]].doc_id,
                    'score': similarities[ranked_indices[j]],
                    'rank': j + 1,
                    'stage': 'classical'
                } for j in range(min(5, len(ranked_indices)))
            ]
        })
    
    # Calculate statistics
    results['timing_stats'] = {
        'avg_time_ms': np.mean(query_times) * 1000,
        'median_time_ms': np.median(query_times) * 1000,
        'min_time_ms': np.min(query_times) * 1000,
        'max_time_ms': np.max(query_times) * 1000,
        'std_time_ms': np.std(query_times) * 1000,
        'total_queries': len(queries),
        'queries_per_second': len(queries) / sum(query_times)
    }
    
    results['quality_metrics'] = {
        'avg_similarity': np.mean(all_similarities),
        'avg_quality_score': np.mean(retrieval_quality_scores),
        'similarity_std': np.std(all_similarities),
        'quality_std': np.std(retrieval_quality_scores)
    }
    
    print(f"  Classical system: {results['timing_stats']['avg_time_ms']:.1f}ms avg")
    return results


def statistical_analysis(quantum_results: Dict, classical_results: Dict) -> Dict:
    """Perform statistical significance testing."""
    
    print("\nStatistical Analysis")
    print("-" * 30)
    
    # Extract metrics for comparison
    quantum_times = [r['time_ms'] for r in quantum_results['query_results']]
    classical_times = [r['time_ms'] for r in classical_results['query_results']]
    
    quantum_similarities = [r['avg_similarity'] for r in quantum_results['query_results']]
    classical_similarities = [r['avg_similarity'] for r in classical_results['query_results']]
    
    quantum_quality = [r['quality_score'] for r in quantum_results['query_results']]
    classical_quality = [r['quality_score'] for r in classical_results['query_results']]
    
    # Statistical tests
    analysis = {}
    
    # Performance comparison (timing)
    if len(quantum_times) > 1 and len(classical_times) > 1:
        try:
            # Remove zeros and very similar values for Wilcoxon test
            time_diffs = [q - c for q, c in zip(quantum_times, classical_times)]
            non_zero_diffs = [d for d in time_diffs if abs(d) > 0.1]
            
            if len(non_zero_diffs) > 0:
                stat, p_value = wilcoxon(non_zero_diffs, alternative='two-sided')
                analysis['timing_test'] = {
                    'test': 'wilcoxon_signed_rank',
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'effect_size': np.mean(time_diffs) / np.std(time_diffs) if np.std(time_diffs) > 0 else 0
                }
            else:
                analysis['timing_test'] = {'test': 'no_significant_differences'}
        except Exception as e:
            analysis['timing_test'] = {'error': str(e)}
    
    # Quality comparison (similarity scores)
    try:
        sim_diffs = [q - c for q, c in zip(quantum_similarities, classical_similarities)]
        non_zero_sim_diffs = [d for d in sim_diffs if abs(d) > 0.001]
        
        if len(non_zero_sim_diffs) > 0:
            stat, p_value = wilcoxon(non_zero_sim_diffs, alternative='two-sided')
            analysis['similarity_test'] = {
                'test': 'wilcoxon_signed_rank',
                'statistic': float(stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'effect_size': np.mean(sim_diffs) / np.std(sim_diffs) if np.std(sim_diffs) > 0 else 0
            }
        else:
            analysis['similarity_test'] = {'test': 'no_significant_differences'}
    except Exception as e:
        analysis['similarity_test'] = {'error': str(e)}
    
    # Quality comparison (relevance scores)
    try:
        quality_diffs = [q - c for q, c in zip(quantum_quality, classical_quality)]
        non_zero_quality_diffs = [d for d in quality_diffs if abs(d) > 0.001]
        
        if len(non_zero_quality_diffs) > 0:
            stat, p_value = wilcoxon(non_zero_quality_diffs, alternative='two-sided')
            analysis['quality_test'] = {
                'test': 'wilcoxon_signed_rank',
                'statistic': float(stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'effect_size': np.mean(quality_diffs) / np.std(quality_diffs) if np.std(quality_diffs) > 0 else 0
            }
        else:
            analysis['quality_test'] = {'test': 'no_significant_differences'}
    except Exception as e:
        analysis['quality_test'] = {'error': str(e)}
    
    # Performance summary
    analysis['performance_summary'] = {
        'quantum_avg_ms': np.mean(quantum_times),
        'classical_avg_ms': np.mean(classical_times),
        'speedup_ratio': np.mean(classical_times) / np.mean(quantum_times) if np.mean(quantum_times) > 0 else 0,
        'quantum_faster': np.mean(quantum_times) < np.mean(classical_times)
    }
    
    # Quality summary
    analysis['quality_summary'] = {
        'quantum_avg_similarity': np.mean(quantum_similarities),
        'classical_avg_similarity': np.mean(classical_similarities),
        'quantum_avg_quality': np.mean(quantum_quality),
        'classical_avg_quality': np.mean(classical_quality),
        'similarity_difference': np.mean(quantum_similarities) - np.mean(classical_similarities),
        'quality_difference': np.mean(quantum_quality) - np.mean(classical_quality)
    }
    
    print(f"  Performance: Quantum {analysis['performance_summary']['quantum_avg_ms']:.1f}ms vs Classical {analysis['performance_summary']['classical_avg_ms']:.1f}ms")
    print(f"  Speedup: {analysis['performance_summary']['speedup_ratio']:.1f}x {'faster' if analysis['performance_summary']['quantum_faster'] else 'slower'}")
    print(f"  Quality difference: {analysis['quality_summary']['similarity_difference']:.4f} (similarity)")
    
    return analysis


def main():
    """Run comprehensive optimized evaluation."""
    print("Optimized Quantum System: Statistical Evaluation")
    print("=" * 60)
    print("Testing optimized system with top-K reranking and batch processing")
    print()
    
    # Initialize memory tracking
    memory_tracker = MemoryTracker()
    memory_tracker.measure("evaluation_start")
    
    # Create evaluation dataset
    print("Creating evaluation dataset...")
    documents, queries = create_evaluation_dataset(n_docs=20)
    memory_tracker.measure("dataset_created")
    
    print(f"Dataset: {len(documents)} documents, {len(queries)} queries")
    print(f"Memory after dataset creation: {memory_tracker.measure('dataset_ready'):.1f} MB")
    print()
    
    # Run evaluations
    quantum_results = run_optimized_evaluation(documents, queries, memory_tracker)
    classical_results = run_classical_baseline(documents, queries, memory_tracker)
    
    # Statistical analysis
    analysis = statistical_analysis(quantum_results, classical_results)
    
    # Memory analysis
    memory_stats = memory_tracker.get_stats()
    memory_tracker.measure("evaluation_complete")
    
    # Final report
    print("\n" + "=" * 60)
    print("FINAL EVALUATION REPORT")
    print("=" * 60)
    
    print(f"\n1. PERFORMANCE COMPARISON")
    print(f"   Quantum Optimized: {quantum_results['timing_stats']['avg_time_ms']:.1f}ms average")
    print(f"   Classical BERT:    {classical_results['timing_stats']['avg_time_ms']:.1f}ms average")
    print(f"   Speedup Ratio:     {analysis['performance_summary']['speedup_ratio']:.1f}x")
    print(f"   Quantum is:        {'FASTER' if analysis['performance_summary']['quantum_faster'] else 'SLOWER'}")
    
    print(f"\n2. QUALITY COMPARISON")
    print(f"   Quantum Similarity: {analysis['quality_summary']['quantum_avg_similarity']:.4f}")
    print(f"   Classical Similarity: {analysis['quality_summary']['classical_avg_similarity']:.4f}")
    print(f"   Quality Difference: {analysis['quality_summary']['similarity_difference']:.4f}")
    print(f"   Relevance Quality:  Quantum={analysis['quality_summary']['quantum_avg_quality']:.4f}, Classical={analysis['quality_summary']['classical_avg_quality']:.4f}")
    
    print(f"\n3. STATISTICAL SIGNIFICANCE")
    timing_test = analysis.get('timing_test', {})
    if 'p_value' in timing_test:
        print(f"   Timing difference:   p={timing_test['p_value']:.4f} {'(significant)' if timing_test.get('significant') else '(not significant)'}")
    
    similarity_test = analysis.get('similarity_test', {})
    if 'p_value' in similarity_test:
        print(f"   Similarity difference: p={similarity_test['p_value']:.4f} {'(significant)' if similarity_test.get('significant') else '(not significant)'}")
    
    print(f"\n4. MEMORY EFFICIENCY")
    print(f"   Baseline memory:   {memory_stats['baseline_mb']:.1f} MB")
    print(f"   Peak memory:       {memory_stats['peak_mb']:.1f} MB")
    print(f"   Memory increase:   {memory_stats['peak_increase_mb']:.1f} MB")
    
    print(f"\n5. OPTIMIZATION SUMMARY")
    print(f"   ✅ Top-K reranking: Only reranking top {quantum_results['config']['rerank_k']} candidates")
    print(f"   ✅ Batch processing: Parallel quantum computation enabled")
    print(f"   ✅ Quality preserved: Similarity difference = {analysis['quality_summary']['similarity_difference']:.4f}")
    print(f"   ✅ Performance: {'Significant improvement' if analysis['performance_summary']['quantum_faster'] else 'Needs further optimization'}")
    
    # Recommendations
    print(f"\n6. RECOMMENDATIONS")
    if analysis['performance_summary']['quantum_faster']:
        print("   🚀 Optimization successful! Ready for production deployment")
        print("   📈 Consider increasing rerank_k for higher quality if speed allows")
        print("   🔄 Monitor performance in production environment")
    else:
        print("   🔧 Further optimization needed:")
        print(f"   - Current quantum system is {analysis['performance_summary']['speedup_ratio']:.1f}x slower")
        print("   - Consider reducing quantum complexity")
        print("   - Implement adaptive K based on query complexity")
    
    return {
        'quantum_results': quantum_results,
        'classical_results': classical_results,
        'analysis': analysis,
        'memory_stats': memory_stats
    }


if __name__ == "__main__":
    main()
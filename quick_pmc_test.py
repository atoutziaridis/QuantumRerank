"""
Quick PMC quantum benchmark with real articles.
Fast validation using subset of articles for immediate results.
"""

import numpy as np
import time
import pickle
from pathlib import Path
from typing import List
from quantum_rerank.core.embeddings import EmbeddingProcessor
from quantum_rerank.core.quantum_similarity_engine import QuantumSimilarityEngine, SimilarityMethod
from noise_injector import MedicalNoiseInjector
from pmc_xml_parser import PMCArticle


def load_pmc_articles():
    """Load parsed PMC articles."""
    pickle_file = Path("parsed_pmc_articles.pkl")
    
    if pickle_file.exists():
        with open(pickle_file, 'rb') as f:
            articles = pickle.load(f)
        return articles[:20]  # Use first 20 for speed
    else:
        print("No parsed articles found. Run pmc_xml_parser.py first.")
        return []


def run_quick_pmc_test():
    """Run quick test with real PMC articles."""
    print("="*60)
    print("QUICK PMC QUANTUM BENCHMARK")
    print("Real PMC Open Access XML Articles")
    print("="*60)
    
    # Load articles
    articles = load_pmc_articles()
    
    if len(articles) < 10:
        print("ERROR: Need at least 10 articles")
        return
    
    print(f"Testing with {len(articles)} real PMC articles")
    
    # Show domain distribution
    domains = {}
    for article in articles:
        domain = article.medical_domain
        domains[domain] = domains.get(domain, 0) + 1
    print(f"Domain distribution: {domains}")
    
    # Initialize components
    embedder = EmbeddingProcessor()
    quantum_engine = QuantumSimilarityEngine()
    noise_injector = MedicalNoiseInjector()
    
    # Test queries matching article domains
    queries = [
        "myocardial infarction treatment strategies",
        "diabetes management and outcomes", 
        "cancer treatment approaches",
        "neurological disorders and brain function"
    ]
    
    # Test with different noise levels
    noise_levels = [0.0, 0.15, 0.25]
    
    results = []
    
    for noise_level in noise_levels:
        print(f"\n--- Testing with {noise_level:.0%} noise ---")
        
        # Apply noise to articles
        noisy_articles = []
        for article in articles:
            # Combine title, abstract, and some full text
            content = f"{article.title}. {article.abstract}"
            if len(article.full_text) > 500:
                content += f" {article.full_text[:500]}"
            
            if noise_level > 0:
                if noise_level >= 0.2:
                    noisy_content = noise_injector.create_abbreviation_noisy_version(content)
                else:
                    noisy_content = noise_injector.create_ocr_noisy_version(content)
            else:
                noisy_content = content
            
            noisy_articles.append({
                'pmc_id': article.pmc_id,
                'title': article.title,
                'content': noisy_content,
                'domain': article.medical_domain
            })
        
        for query_idx, query in enumerate(queries):
            print(f"\nQuery {query_idx + 1}: {query}")
            
            # Create simple relevance judgments based on domain matching
            relevant_articles = []
            for article in noisy_articles:
                domain = article['domain']
                title_lower = article['title'].lower()
                
                is_relevant = False
                if "myocardial" in query.lower() and domain == "cardiology":
                    is_relevant = True
                elif "diabetes" in query.lower() and domain == "diabetes":
                    is_relevant = True
                elif "cancer" in query.lower() and domain == "oncology":
                    is_relevant = True
                elif "neurological" in query.lower() and domain == "neurology":
                    is_relevant = True
                elif any(term in title_lower for term in query.lower().split()):
                    is_relevant = True
                
                if is_relevant:
                    relevant_articles.append(article['pmc_id'])
            
            print(f"  Relevant articles: {len(relevant_articles)}")
            
            # Classical retrieval
            query_emb = embedder.encode_single_text(query)
            classical_scores = []
            
            start_time = time.time()
            for article in noisy_articles:
                doc_emb = embedder.encode_single_text(article['content'])
                score = np.dot(query_emb, doc_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
                )
                classical_scores.append((score, article['pmc_id']))
            classical_time = (time.time() - start_time) * 1000
            
            classical_scores.sort(reverse=True)
            classical_top3 = [pmc_id for _, pmc_id in classical_scores[:3]]
            
            # Quantum retrieval
            quantum_scores = []
            
            start_time = time.time()
            for article in noisy_articles:
                try:
                    score, _ = quantum_engine.compute_similarity(
                        query, article['content'], method=SimilarityMethod.HYBRID_WEIGHTED
                    )
                    quantum_scores.append((score, article['pmc_id']))
                except Exception as e:
                    print(f"    Quantum failed for {article['pmc_id']}: {e}")
                    # Fallback to classical
                    doc_emb = embedder.encode_single_text(article['content'])
                    score = np.dot(query_emb, doc_emb) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
                    )
                    quantum_scores.append((score, article['pmc_id']))
            quantum_time = (time.time() - start_time) * 1000
            
            quantum_scores.sort(reverse=True)
            quantum_top3 = [pmc_id for _, pmc_id in quantum_scores[:3]]
            
            # Calculate precision@3
            classical_precision = len(set(classical_top3) & set(relevant_articles)) / 3
            quantum_precision = len(set(quantum_top3) & set(relevant_articles)) / 3
            
            improvement = ((quantum_precision - classical_precision) / 
                          max(classical_precision, 0.001)) * 100
            
            print(f"    Classical P@3: {classical_precision:.3f} ({classical_time:.1f}ms)")
            print(f"    Quantum P@3:   {quantum_precision:.3f} ({quantum_time:.1f}ms)")
            print(f"    Improvement: {improvement:+.1f}%")
            
            results.append({
                'noise_level': noise_level,
                'query': query,
                'classical_precision': classical_precision,
                'quantum_precision': quantum_precision,
                'improvement': improvement,
                'classical_time': classical_time,
                'quantum_time': quantum_time
            })
    
    # Summary
    print("\n" + "="*60)
    print("QUICK PMC QUANTUM BENCHMARK RESULTS")
    print("="*60)
    
    for noise_level in noise_levels:
        level_results = [r for r in results if r['noise_level'] == noise_level]
        
        avg_classical = np.mean([r['classical_precision'] for r in level_results])
        avg_quantum = np.mean([r['quantum_precision'] for r in level_results])
        avg_improvement = np.mean([r['improvement'] for r in level_results])
        quantum_wins = sum(1 for r in level_results if r['improvement'] > 0)
        
        print(f"\nNoise Level {noise_level:.0%}:")
        print(f"  Classical Precision: {avg_classical:.3f}")
        print(f"  Quantum Precision:   {avg_quantum:.3f}")
        print(f"  Average Improvement: {avg_improvement:+.1f}%")
        print(f"  Quantum Wins: {quantum_wins}/{len(level_results)}")
    
    # Overall stats
    overall_improvement = np.mean([r['improvement'] for r in results])
    overall_quantum_wins = sum(1 for r in results if r['improvement'] > 0)
    
    print(f"\nOVERALL RESULTS:")
    print(f"  Average Improvement: {overall_improvement:+.1f}%")
    print(f"  Quantum Wins: {overall_quantum_wins}/{len(results)} ({overall_quantum_wins/len(results)*100:.1f}%)")
    
    # High noise analysis
    high_noise_results = [r for r in results if r['noise_level'] >= 0.20]
    if high_noise_results:
        high_noise_improvement = np.mean([r['improvement'] for r in high_noise_results])
        print(f"  High Noise (≥20%) Improvement: {high_noise_improvement:+.1f}%")
    
    print(f"\nKEY FINDINGS:")
    print(f"  ✓ Tested with {len(articles)} real PMC full-text articles")
    print(f"  ✓ Realistic noise: OCR errors, medical abbreviations")
    print(f"  ✓ Clinical queries on authentic medical literature")
    
    if overall_improvement > 0:
        print(f"  🎯 Quantum shows {overall_improvement:.1f}% average advantage")
    
    if high_noise_results and high_noise_improvement > overall_improvement:
        print(f"  📈 Quantum advantage increases with noise: {high_noise_improvement:.1f}% at high noise")
    
    print("\n" + "="*60)
    
    return results


if __name__ == "__main__":
    run_quick_pmc_test()
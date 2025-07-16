"""
Proper Two-Stage Retrieval Test with Quantum Reranking
Testing quantum as a reranker, not just similarity replacement.
"""

import numpy as np
import time
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from quantum_rerank.core.embeddings import EmbeddingProcessor
from quantum_rerank.retrieval.two_stage_retriever import TwoStageRetriever, RetrieverConfig
from quantum_rerank.retrieval.document_store import Document, DocumentMetadata
from quantum_rerank.core.quantum_similarity_engine import SimilarityMethod, SimilarityEngineConfig
from noise_injector import MedicalNoiseInjector
from pmc_xml_parser import PMCArticle


def load_pmc_articles() -> List[PMCArticle]:
    """Load parsed PMC articles."""
    pickle_file = Path("parsed_pmc_articles.pkl")
    
    if pickle_file.exists():
        with open(pickle_file, 'rb') as f:
            articles = pickle.load(f)
        return articles[:50]  # Use more articles for better FAISS performance
    else:
        print("No parsed articles found. Run pmc_xml_parser.py first.")
        return []


def create_test_documents(articles: List[PMCArticle], noise_level: float = 0.0) -> List[Document]:
    """Create Document objects from PMC articles with optional noise."""
    noise_injector = MedicalNoiseInjector()
    documents = []
    
    for i, article in enumerate(articles):
        # Combine title, abstract, and first part of full text
        content = f"{article.title}. {article.abstract}"
        if len(article.full_text) > 1000:
            content += f" {article.full_text[:1000]}"
        
        # Apply noise if requested
        if noise_level > 0:
            if noise_level >= 0.2:
                content = noise_injector.create_abbreviation_noisy_version(content)
            else:
                content = noise_injector.create_ocr_noisy_version(content)
        
        # Create metadata
        metadata = DocumentMetadata(
            custom_fields={
                "pmc_id": article.pmc_id,
                "title": article.title,
                "medical_domain": article.medical_domain,
                "journal": article.journal,
                "authors": article.authors,
                "keywords": article.keywords
            }
        )
        
        # Create document
        doc = Document(
            doc_id=f"pmc_{i:04d}",
            content=content,
            metadata=metadata
        )
        documents.append(doc)
    
    return documents


def create_medical_queries() -> List[Dict]:
    """Create targeted medical queries with relevance criteria."""
    return [
        {
            "query": "heart failure treatment and management strategies",
            "relevant_domains": ["cardiology"],
            "relevant_keywords": ["heart", "cardiac", "failure", "treatment", "management", "cardiovascular"]
        },
        {
            "query": "diabetes mellitus complications and monitoring",
            "relevant_domains": ["diabetes"],
            "relevant_keywords": ["diabetes", "glucose", "insulin", "complications", "monitoring", "glycemic"]
        },
        {
            "query": "cancer treatment approaches and chemotherapy",
            "relevant_domains": ["oncology"],
            "relevant_keywords": ["cancer", "tumor", "chemotherapy", "treatment", "oncology", "malignant"]
        },
        {
            "query": "neurological disorders and brain function",
            "relevant_domains": ["neurology"],
            "relevant_keywords": ["brain", "neural", "neurological", "cognitive", "seizure", "stroke"]
        },
        {
            "query": "respiratory diseases and lung function",
            "relevant_domains": ["respiratory"],
            "relevant_keywords": ["lung", "respiratory", "breathing", "pulmonary", "pneumonia", "asthma"]
        }
    ]


def evaluate_relevance(documents: List[Document], query_info: Dict) -> List[str]:
    """Create relevance judgments for a query."""
    relevant_doc_ids = []
    
    for doc in documents:
        relevance_score = 0
        domain = doc.metadata.custom_fields.get("medical_domain", "")
        title = doc.metadata.custom_fields.get("title", "").lower()
        content = doc.content.lower()
        keywords = doc.metadata.custom_fields.get("keywords", [])
        
        # Domain match (high weight)
        if domain in query_info["relevant_domains"]:
            relevance_score += 10
        
        # Keyword matching in title (high weight)
        for keyword in query_info["relevant_keywords"]:
            if keyword.lower() in title:
                relevance_score += 5
        
        # Keyword matching in content (medium weight)
        for keyword in query_info["relevant_keywords"]:
            if keyword.lower() in content:
                relevance_score += 2
        
        # Keyword matching in metadata keywords (medium weight)
        for keyword in query_info["relevant_keywords"]:
            for doc_keyword in keywords:
                if keyword.lower() in doc_keyword.lower():
                    relevance_score += 3
        
        # Threshold for relevance
        if relevance_score >= 5:
            relevant_doc_ids.append(doc.doc_id)
    
    return relevant_doc_ids


def calculate_ranking_metrics(retrieved_ids: List[str], relevant_ids: List[str], k: int = 10) -> Dict:
    """Calculate ranking evaluation metrics."""
    if not retrieved_ids or not relevant_ids:
        return {"precision": 0.0, "recall": 0.0, "ndcg": 0.0, "mrr": 0.0}
    
    # Precision@K
    top_k = retrieved_ids[:k]
    precision = len(set(top_k) & set(relevant_ids)) / len(top_k)
    
    # Recall@K
    recall = len(set(top_k) & set(relevant_ids)) / len(relevant_ids)
    
    # NDCG@K
    relevance = [1 if doc_id in relevant_ids else 0 for doc_id in top_k]
    dcg = relevance[0] if relevance else 0
    for i in range(1, len(relevance)):
        dcg += relevance[i] / np.log2(i + 1)
    
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = ideal_relevance[0] if ideal_relevance else 0
    for i in range(1, len(ideal_relevance)):
        idcg += ideal_relevance[i] / np.log2(i + 1)
    
    ndcg = dcg / idcg if idcg > 0 else 0
    
    # MRR (Mean Reciprocal Rank)
    mrr = 0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            mrr = 1.0 / (i + 1)
            break
    
    return {
        "precision": precision,
        "recall": recall,
        "ndcg": ndcg,
        "mrr": mrr
    }


def run_proper_reranking_test():
    """Run proper two-stage retrieval test with quantum reranking."""
    print("="*70)
    print("PROPER TWO-STAGE QUANTUM RERANKING TEST")
    print("FAISS Initial Retrieval → Quantum Reranking")
    print("="*70)
    
    # Load articles
    articles = load_pmc_articles()
    if len(articles) < 20:
        print("ERROR: Need at least 20 articles")
        return
    
    print(f"Testing with {len(articles)} real PMC articles")
    
    # Show domain distribution
    domains = {}
    for article in articles:
        domain = article.medical_domain
        domains[domain] = domains.get(domain, 0) + 1
    print(f"Domain distribution: {domains}")
    
    # Test configurations
    noise_levels = [0.0, 0.15, 0.25]
    reranking_methods = ["classical", "quantum", "hybrid"]
    queries = create_medical_queries()
    
    all_results = []
    
    for noise_level in noise_levels:
        print(f"\n{'='*50}")
        print(f"TESTING WITH {noise_level:.0%} NOISE")
        print(f"{'='*50}")
        
        # Create documents with noise
        documents = create_test_documents(articles, noise_level)
        
        for method in reranking_methods:
            print(f"\n--- Testing {method.upper()} reranking ---")
            
            # Configure retriever for this method
            if method == "classical":
                similarity_method = SimilarityMethod.CLASSICAL_COSINE
            elif method == "quantum":
                similarity_method = SimilarityMethod.QUANTUM_FIDELITY
            else:  # hybrid
                similarity_method = SimilarityMethod.HYBRID_WEIGHTED
            
            config = RetrieverConfig(
                initial_k=30,  # Get 30 candidates from FAISS
                final_k=10,    # Rerank to top 10
                reranking_method=method,
                similarity_engine_config=SimilarityEngineConfig(
                    similarity_method=similarity_method,
                    hybrid_weights={"quantum": 0.6, "classical": 0.4}  # Favor quantum
                )
            )
            
            # Initialize retriever
            retriever = TwoStageRetriever(config)
            
            # Add documents to retriever
            print(f"  Adding {len(documents)} documents to FAISS index...")
            start_time = time.time()
            retriever.add_documents(documents)
            index_time = (time.time() - start_time) * 1000
            print(f"  Indexing completed in {index_time:.1f}ms")
            
            method_results = []
            
            for query_info in queries:
                query = query_info["query"]
                print(f"\n  Query: {query}")
                
                # Get relevance judgments
                relevant_ids = evaluate_relevance(documents, query_info)
                print(f"    Relevant documents: {len(relevant_ids)}")
                
                # Retrieve with current method
                start_time = time.time()
                try:
                    results = retriever.retrieve(query, k=10)
                    retrieval_time = (time.time() - start_time) * 1000
                    
                    # Extract document IDs
                    retrieved_ids = [result.doc_id for result in results]
                    
                    # Calculate metrics
                    metrics = calculate_ranking_metrics(retrieved_ids, relevant_ids, k=10)
                    
                    print(f"    {method.capitalize()} - P@10: {metrics['precision']:.3f}, "
                          f"NDCG@10: {metrics['ndcg']:.3f}, "
                          f"MRR: {metrics['mrr']:.3f} ({retrieval_time:.1f}ms)")
                    
                    method_results.append({
                        "query": query,
                        "method": method,
                        "noise_level": noise_level,
                        "metrics": metrics,
                        "retrieval_time": retrieval_time,
                        "relevant_count": len(relevant_ids),
                        "retrieved_ids": retrieved_ids[:5]  # Top 5 for analysis
                    })
                    
                except Exception as e:
                    print(f"    ERROR with {method}: {e}")
                    retrieval_time = 0
                    metrics = {"precision": 0, "recall": 0, "ndcg": 0, "mrr": 0}
                    
                    method_results.append({
                        "query": query,
                        "method": method,
                        "noise_level": noise_level,
                        "metrics": metrics,
                        "retrieval_time": retrieval_time,
                        "relevant_count": len(relevant_ids),
                        "error": str(e)
                    })
            
            all_results.extend(method_results)
    
    # Comprehensive analysis
    print("\n" + "="*70)
    print("COMPREHENSIVE RERANKING ANALYSIS")
    print("="*70)
    
    # Group results by method and noise level
    for noise_level in noise_levels:
        print(f"\n{'='*40}")
        print(f"NOISE LEVEL: {noise_level:.0%}")
        print(f"{'='*40}")
        
        for method in reranking_methods:
            method_results = [r for r in all_results 
                            if r["noise_level"] == noise_level and r["method"] == method]
            
            if not method_results:
                continue
            
            # Calculate averages
            avg_precision = np.mean([r["metrics"]["precision"] for r in method_results])
            avg_ndcg = np.mean([r["metrics"]["ndcg"] for r in method_results])
            avg_mrr = np.mean([r["metrics"]["mrr"] for r in method_results])
            avg_time = np.mean([r["retrieval_time"] for r in method_results])
            
            print(f"\n{method.upper()} Results:")
            print(f"  Average Precision@10: {avg_precision:.3f}")
            print(f"  Average NDCG@10:      {avg_ndcg:.3f}")
            print(f"  Average MRR:          {avg_mrr:.3f}")
            print(f"  Average Time:         {avg_time:.1f}ms")
    
    # Method comparison
    print(f"\n{'='*40}")
    print("METHOD COMPARISON")
    print(f"{'='*40}")
    
    for noise_level in noise_levels:
        print(f"\nAt {noise_level:.0%} noise:")
        
        methods_data = {}
        for method in reranking_methods:
            method_results = [r for r in all_results 
                            if r["noise_level"] == noise_level and r["method"] == method]
            if method_results:
                methods_data[method] = {
                    "precision": np.mean([r["metrics"]["precision"] for r in method_results]),
                    "ndcg": np.mean([r["metrics"]["ndcg"] for r in method_results]),
                    "mrr": np.mean([r["metrics"]["mrr"] for r in method_results])
                }
        
        # Compare quantum vs classical
        if "quantum" in methods_data and "classical" in methods_data:
            quantum_data = methods_data["quantum"]
            classical_data = methods_data["classical"]
            
            precision_improvement = ((quantum_data["precision"] - classical_data["precision"]) / 
                                   max(classical_data["precision"], 0.001)) * 100
            ndcg_improvement = ((quantum_data["ndcg"] - classical_data["ndcg"]) / 
                              max(classical_data["ndcg"], 0.001)) * 100
            mrr_improvement = ((quantum_data["mrr"] - classical_data["mrr"]) / 
                             max(classical_data["mrr"], 0.001)) * 100
            
            print(f"  Quantum vs Classical:")
            print(f"    Precision improvement: {precision_improvement:+.1f}%")
            print(f"    NDCG improvement:      {ndcg_improvement:+.1f}%")
            print(f"    MRR improvement:       {mrr_improvement:+.1f}%")
        
        # Compare hybrid vs classical
        if "hybrid" in methods_data and "classical" in methods_data:
            hybrid_data = methods_data["hybrid"]
            classical_data = methods_data["classical"]
            
            precision_improvement = ((hybrid_data["precision"] - classical_data["precision"]) / 
                                   max(classical_data["precision"], 0.001)) * 100
            
            print(f"  Hybrid vs Classical:")
            print(f"    Precision improvement: {precision_improvement:+.1f}%")
    
    print(f"\n{'='*70}")
    print("KEY FINDINGS:")
    print("✓ Proper two-stage retrieval: FAISS → Quantum reranking")
    print("✓ Real PMC medical articles with authentic medical content")
    print("✓ Realistic noise patterns affecting medical documents")
    print("✓ Comprehensive evaluation with standard IR metrics")
    
    # Identify best performing method
    overall_results = {}
    for method in reranking_methods:
        method_results = [r for r in all_results if r["method"] == method]
        if method_results:
            overall_results[method] = np.mean([r["metrics"]["ndcg"] for r in method_results])
    
    if overall_results:
        best_method = max(overall_results, key=overall_results.get)
        print(f"🎯 Best performing method overall: {best_method.upper()}")
        print(f"   Average NDCG@10: {overall_results[best_method]:.3f}")
    
    print("="*70)
    
    return all_results


if __name__ == "__main__":
    run_proper_reranking_test()
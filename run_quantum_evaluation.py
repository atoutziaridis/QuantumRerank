#!/usr/bin/env python3
"""
Simplified Quantum Reranker Evaluation

This test provides a focused evaluation of the quantum reranker in scenarios
where quantum computing should provide genuine advantages.
"""

import logging
import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Import the actual quantum reranker components
from quantum_rerank.core.quantum_similarity_engine import QuantumSimilarityEngine, SimilarityEngineConfig, SimilarityMethod
from quantum_rerank.core.embeddings import EmbeddingProcessor, EmbeddingConfig
from quantum_rerank.retrieval.two_stage_retriever import TwoStageRetriever, RetrieverConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_scenario():
    """Create a focused test scenario."""
    query = {
        "text": "Patient presenting with chest pain, shortness of breath, and fatigue. Recent travel history. What are the most likely diagnoses and recommended diagnostic approach?",
        "scenario": "ambiguous_symptoms_pe_vs_cardiac"
    }
    
    candidates = [
        {
            "text": "Pulmonary embolism diagnosis in patients with chest pain and dyspnea. Clinical presentation includes acute onset chest pain, shortness of breath, and fatigue. Risk factors include recent travel, immobilization, and hypercoagulable states. Diagnostic approach includes D-dimer, CT pulmonary angiogram, and Wells score calculation.",
            "relevance": 0.95,
            "type": "highly_relevant"
        },
        {
            "text": "Acute coronary syndrome evaluation in chest pain patients. Symptoms include crushing chest pain, dyspnea, diaphoresis, and fatigue. ECG findings, troponin levels, and risk stratification using TIMI or GRACE scores are essential for diagnosis and management.",
            "relevance": 0.90,
            "type": "highly_relevant"
        },
        {
            "text": "Anxiety disorders presenting with somatic symptoms. Panic attacks can mimic cardiac conditions with chest pain, palpitations, and shortness of breath. Important to rule out organic causes before psychiatric diagnosis.",
            "relevance": 0.60,
            "type": "moderately_relevant"
        },
        {
            "text": "Clinical decision making in ambiguous presentations. When multiple diagnoses are possible, systematic approach using Bayesian reasoning and clinical gestalt improves diagnostic accuracy.",
            "relevance": 0.40,
            "type": "subtly_relevant"
        },
        {
            "text": "Diabetes management and glycemic control. HbA1c targets, medication selection, and lifestyle interventions for type 2 diabetes mellitus.",
            "relevance": 0.10,
            "type": "irrelevant"
        }
    ]
    
    return query, candidates


def test_reranking_methods():
    """Test different reranking methods and compare performance."""
    logger.info("=== QUANTUM RERANKER EVALUATION ===")
    
    # Initialize embedding processor
    embedding_processor = EmbeddingProcessor(EmbeddingConfig())
    
    # Create test scenario
    query, candidates = create_test_scenario()
    
    logger.info(f"Testing query: {query['text'][:80]}...")
    logger.info(f"Number of candidates: {len(candidates)}")
    
    # Test different methods
    methods = ["classical", "quantum", "hybrid"]
    results = {}
    
    for method in methods:
        logger.info(f"\n--- Testing {method.upper()} method ---")
        
        # Initialize retriever for this method
        config = RetrieverConfig(
            initial_k=10,
            final_k=5,
            reranking_method=method,
            enable_caching=False
        )
        
        start_time = time.time()
        retriever = TwoStageRetriever(config=config, embedding_processor=embedding_processor)
        
        # Add documents
        texts = [c["text"] for c in candidates]
        metadatas = [{"relevance": c["relevance"], "type": c["type"]} for c in candidates]
        
        try:
            doc_ids = retriever.add_texts(texts, metadatas)
            
            # Perform retrieval
            retrieval_results = retriever.retrieve(query["text"], k=5)
            total_time = (time.time() - start_time) * 1000
            
            # Calculate NDCG@5
            retrieved_relevances = []
            for result in retrieval_results:
                # Find matching candidate
                for candidate in candidates:
                    if candidate["text"] == result.content:
                        retrieved_relevances.append(candidate["relevance"])
                        break
                else:
                    retrieved_relevances.append(0.0)
            
            # Pad to 5 if needed
            while len(retrieved_relevances) < 5:
                retrieved_relevances.append(0.0)
            
            # Calculate NDCG@5
            ideal_relevances = sorted([c["relevance"] for c in candidates], reverse=True)[:5]
            ndcg = calculate_ndcg(retrieved_relevances[:5], ideal_relevances)
            
            results[method] = {
                "ndcg": ndcg,
                "latency_ms": total_time,
                "retrieved_relevances": retrieved_relevances,
                "success": True
            }
            
            logger.info(f"  NDCG@5: {ndcg:.4f}")
            logger.info(f"  Latency: {total_time:.1f}ms")
            logger.info(f"  Retrieved relevances: {[f'{r:.2f}' for r in retrieved_relevances[:3]]}")
            
        except Exception as e:
            logger.error(f"  Error with {method} method: {e}")
            results[method] = {
                "ndcg": 0.0,
                "latency_ms": float('inf'),
                "retrieved_relevances": [0.0] * 5,
                "success": False,
                "error": str(e)
            }
    
    # Analysis
    logger.info("\n=== PERFORMANCE ANALYSIS ===")
    
    # Calculate improvements
    if results["classical"]["success"] and results["quantum"]["success"]:
        quantum_improvement = ((results["quantum"]["ndcg"] - results["classical"]["ndcg"]) / 
                             max(results["classical"]["ndcg"], 0.001)) * 100
        
        logger.info(f"Quantum vs Classical NDCG improvement: {quantum_improvement:.2f}%")
        
        if results["hybrid"]["success"]:
            hybrid_improvement = ((results["hybrid"]["ndcg"] - results["classical"]["ndcg"]) / 
                                max(results["classical"]["ndcg"], 0.001)) * 100
            logger.info(f"Hybrid vs Classical NDCG improvement: {hybrid_improvement:.2f}%")
    
    # Latency comparison
    logger.info("\nLatency comparison:")
    for method, result in results.items():
        if result["success"]:
            logger.info(f"  {method.capitalize()}: {result['latency_ms']:.1f}ms")
    
    # Final verdict
    logger.info("\n=== QUANTUM ADVANTAGE ASSESSMENT ===")
    
    if (results["quantum"]["success"] and results["classical"]["success"] and 
        results["quantum"]["ndcg"] > results["classical"]["ndcg"]):
        
        improvement = ((results["quantum"]["ndcg"] - results["classical"]["ndcg"]) / 
                      max(results["classical"]["ndcg"], 0.001)) * 100
        
        if improvement > 10:
            logger.info("✅ SIGNIFICANT QUANTUM ADVANTAGE DETECTED")
            logger.info(f"   Quantum reranker shows {improvement:.1f}% improvement in NDCG@5")
            logger.info("   Recommendation: Deploy quantum reranker for medical scenarios")
        elif improvement > 5:
            logger.info("⚡ MODEST QUANTUM ADVANTAGE DETECTED")
            logger.info(f"   Quantum reranker shows {improvement:.1f}% improvement in NDCG@5")
            logger.info("   Recommendation: Consider for specific high-value medical use cases")
        else:
            logger.info("📊 MINIMAL QUANTUM ADVANTAGE")
            logger.info(f"   Quantum reranker shows {improvement:.1f}% improvement in NDCG@5")
            logger.info("   Recommendation: Continue optimization before deployment")
    else:
        logger.info("❌ NO QUANTUM ADVANTAGE OBSERVED")
        logger.info("   Classical methods perform as well or better than quantum")
        logger.info("   Recommendation: Focus on algorithm improvements")
    
    return results


def calculate_ndcg(relevances, ideal_relevances):
    """Calculate NDCG@k."""
    def dcg(relevances):
        return sum([(2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevances)])
    
    dcg_score = dcg(relevances)
    idcg_score = dcg(ideal_relevances)
    
    return dcg_score / idcg_score if idcg_score > 0 else 0


if __name__ == "__main__":
    try:
        results = test_reranking_methods()
        logger.info("\n✅ Evaluation completed successfully")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
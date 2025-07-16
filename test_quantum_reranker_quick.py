#!/usr/bin/env python3
"""
Quick Real-World Quantum Reranker Test

Tests the actual quantum reranker on a smaller set of realistic queries
to get concrete performance results quickly.
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_quantum_advantage_queries():
    """Create specific queries where quantum should have advantage."""
    return [
        {
            "id": "quantum_001",
            "text": "Patient presenting with chest pain, shortness of breath, and fatigue. Recent travel history. What are the most likely diagnoses considering both cardiac and pulmonary causes?",
            "scenario": "ambiguous_symptoms",
            "complexity": "high"
        },
        {
            "id": "quantum_002", 
            "text": "45-year-old with muscle weakness and ptosis that worsens throughout the day. Consider both common and rare neuromuscular conditions.",
            "scenario": "rare_disease",
            "complexity": "very_high"
        },
        {
            "id": "quantum_003",
            "text": "Patient on warfarin, amiodarone, and aspirin considering adding fluconazole. Assess drug interactions and monitoring requirements.",
            "scenario": "drug_interactions",
            "complexity": "high"
        },
        {
            "id": "quantum_004",
            "text": "20-year-old athlete with chest pain after exercise versus 65-year-old diabetic with similar symptoms. Context-dependent risk assessment.",
            "scenario": "contextual_interpretation", 
            "complexity": "high"
        },
        {
            "id": "quantum_005",
            "text": "Acute dyspnea with clear lung sounds on auscultation. Differential diagnosis with probability estimates for multiple potential causes.",
            "scenario": "differential_diagnosis",
            "complexity": "very_high"
        }
    ]


def create_challenging_candidates(query_scenario: str) -> List[Dict[str, Any]]:
    """Create challenging candidate documents for ranking."""
    base_candidates = []
    
    if query_scenario == "ambiguous_symptoms":
        # High relevance - directly addresses chest pain with pulmonary cause
        base_candidates.append({
            "text": "Pulmonary embolism diagnosis in patients with chest pain and dyspnea. Clinical presentation includes acute onset chest pain, shortness of breath, and fatigue. Risk factors include recent travel, immobilization, and hypercoagulable states. Diagnostic approach includes D-dimer, CT pulmonary angiogram, and Wells score calculation.",
            "relevance": 0.95
        })
        
        # High relevance - directly addresses cardiac causes
        base_candidates.append({
            "text": "Acute coronary syndrome evaluation in chest pain patients. Symptoms include crushing chest pain, dyspnea, diaphoresis, and fatigue. ECG findings, troponin levels, and risk stratification using TIMI or GRACE scores are essential for diagnosis and management.",
            "relevance": 0.90
        })
        
        # Moderate relevance - related but different focus
        base_candidates.append({
            "text": "Anxiety disorders presenting with somatic symptoms. Panic attacks can mimic cardiac conditions with chest pain, palpitations, and shortness of breath. Important to rule out organic causes before psychiatric diagnosis.",
            "relevance": 0.60
        })
        
        # Subtle relevance - requires deep understanding
        base_candidates.append({
            "text": "Clinical decision making in ambiguous presentations. When multiple diagnoses are possible, systematic approach using Bayesian reasoning and clinical gestalt improves diagnostic accuracy. Pattern recognition versus analytical thinking in medical diagnosis.",
            "relevance": 0.40
        })
        
        # Low relevance
        base_candidates.append({
            "text": "Diabetes management and glycemic control. HbA1c targets, medication selection, and lifestyle interventions for type 2 diabetes mellitus. Monitoring for complications including retinopathy, nephropathy, and neuropathy.",
            "relevance": 0.10
        })
        
    elif query_scenario == "rare_disease":
        base_candidates.append({
            "text": "Myasthenia gravis diagnosis and management. Characterized by fluctuating muscle weakness and fatigability. Ptosis and diplopia are common early symptoms. Anticholinesterase test and acetylcholine receptor antibodies confirm diagnosis.",
            "relevance": 0.95
        })
        
        base_candidates.append({
            "text": "Differential diagnosis of muscle weakness. Includes myopathies, neuropathies, and neuromuscular junction disorders. History of fatigue pattern and physical examination findings guide diagnostic workup.",
            "relevance": 0.85
        })
        
        base_candidates.append({
            "text": "Stroke evaluation in patients with acute neurological deficits. Weakness, speech changes, and cranial nerve involvement. Time-sensitive evaluation with CT and MRI imaging.",
            "relevance": 0.50
        })
        
        base_candidates.append({
            "text": "Approach to rare diseases in clinical practice. When common diagnoses don't fit clinical picture, systematic evaluation for uncommon conditions. Importance of detailed history and physical examination.",
            "relevance": 0.35
        })
        
        base_candidates.append({
            "text": "Exercise physiology and athletic performance. Training adaptations, muscle fiber types, and energy systems. Nutrition and hydration strategies for optimal performance.",
            "relevance": 0.05
        })
        
    elif query_scenario == "drug_interactions":
        base_candidates.append({
            "text": "Warfarin drug interactions and monitoring. Fluconazole significantly increases warfarin effect through CYP2C9 inhibition. Requires dose reduction and increased INR monitoring. Amiodarone also potentiates anticoagulation.",
            "relevance": 0.95
        })
        
        base_candidates.append({
            "text": "Antifungal therapy considerations in anticoagulated patients. Azole antifungals inhibit cytochrome P450 enzymes affecting warfarin metabolism. Alternative antifungal agents may be preferred.",
            "relevance": 0.80
        })
        
        base_candidates.append({
            "text": "Pharmacokinetic drug interactions involving CYP enzymes. Understanding enzyme induction and inhibition helps predict drug interactions and guide dosing adjustments.",
            "relevance": 0.60
        })
        
        base_candidates.append({
            "text": "Clinical pharmacology principles in polypharmacy management. Multiple drug interactions become complex requiring systematic evaluation and monitoring protocols.",
            "relevance": 0.40
        })
        
        base_candidates.append({
            "text": "Hypertension management guidelines. ACE inhibitors, beta blockers, calcium channel blockers, and diuretics. Step-wise approach to blood pressure control.",
            "relevance": 0.10
        })
        
    elif query_scenario == "contextual_interpretation":
        base_candidates.append({
            "text": "Age-specific risk stratification in chest pain evaluation. Young athletes have different risk profile compared to older adults with cardiovascular risk factors. Context dramatically changes differential diagnosis and management approach.",
            "relevance": 0.90
        })
        
        base_candidates.append({
            "text": "Chest pain in young athletes: exercise-induced versus pathological causes. Hypertrophic cardiomyopathy, arrhythmogenic right ventricular cardiomyopathy, and coronary anomalies are important considerations.",
            "relevance": 0.85
        })
        
        base_candidates.append({
            "text": "Cardiovascular risk assessment in elderly diabetic patients. Multiple risk factors compound to create high-risk profile requiring aggressive evaluation and management of chest pain.",
            "relevance": 0.80
        })
        
        base_candidates.append({
            "text": "Clinical reasoning and diagnostic thinking. How context and patient characteristics influence medical decision making. Cognitive biases and heuristics in clinical practice.",
            "relevance": 0.45
        })
        
        base_candidates.append({
            "text": "Sports medicine and injury prevention. Common athletic injuries, rehabilitation protocols, and return-to-play decisions. Biomechanical analysis and performance optimization.",
            "relevance": 0.15
        })
        
    else:  # differential_diagnosis
        base_candidates.append({
            "text": "Dyspnea with clear lung sounds: systematic differential diagnosis. Consider pulmonary embolism, heart failure with preserved ejection fraction, anemia, metabolic acidosis, and anxiety disorders. Each requires specific diagnostic approach.",
            "relevance": 0.95
        })
        
        base_candidates.append({
            "text": "Acute dyspnea evaluation in emergency department. Rapid assessment protocol including vital signs, oxygen saturation, ECG, chest X-ray, and point-of-care ultrasound to narrow differential diagnosis.",
            "relevance": 0.85
        })
        
        base_candidates.append({
            "text": "Heart failure with preserved ejection fraction diagnosis. Often presents with dyspnea and clear lung sounds. Requires echocardiography and elevated BNP for diagnosis.",
            "relevance": 0.75
        })
        
        base_candidates.append({
            "text": "Probabilistic reasoning in medical diagnosis. Using likelihood ratios and Bayesian inference to estimate probability of different diagnoses based on clinical findings.",
            "relevance": 0.40
        })
        
        base_candidates.append({
            "text": "Infectious disease prevention and vaccination schedules. Adult immunization recommendations including influenza, pneumococcal, and COVID-19 vaccines.",
            "relevance": 0.05
        })
    
    # Add some generic medical content to make ranking challenging
    for i in range(10):
        base_candidates.append({
            "text": f"General medical knowledge topic {i+1}. Clinical guidelines and best practices for patient care. Evidence-based medicine principles and quality improvement initiatives.",
            "relevance": 0.15 + np.random.uniform(-0.05, 0.05)
        })
    
    return base_candidates


def test_quantum_reranker():
    """Test the quantum reranker on realistic medical queries."""
    logger.info("=" * 80)
    logger.info("QUANTUM RERANKER REAL-WORLD PERFORMANCE TEST")
    logger.info("=" * 80)
    
    # Initialize quantum reranker
    logger.info("Initializing quantum reranker...")
    embedding_processor = EmbeddingProcessor(EmbeddingConfig())
    
    # Results storage
    results = {
        "quantum": {"ndcg": [], "latency": []},
        "classical": {"ndcg": [], "latency": []},
        "hybrid": {"ndcg": [], "latency": []}
    }
    
    queries = create_quantum_advantage_queries()
    logger.info(f"Testing {len(queries)} quantum-advantage queries")
    
    for i, query in enumerate(queries):
        logger.info(f"\nQuery {i+1}/{len(queries)}: {query['scenario']}")
        logger.info(f"Text: {query['text'][:100]}...")
        
        # Create candidates
        candidates = create_challenging_candidates(query["scenario"])
        
        # Test each method
        for method in ["quantum", "classical", "hybrid"]:
            logger.info(f"  Testing {method} method...")
            
            # Initialize retriever for this method
            config = RetrieverConfig(
                initial_k=20,
                final_k=10,
                reranking_method=method,
                enable_caching=False
            )
            retriever = TwoStageRetriever(config=config, embedding_processor=embedding_processor)
            
            # Add documents
            texts = [c["text"] for c in candidates]
            metadatas = [{"relevance": c["relevance"]} for c in candidates]
            
            start_time = time.time()
            doc_ids = retriever.add_texts(texts, metadatas)
            
            # Perform retrieval
            retrieval_results = retriever.retrieve(query["text"], k=10)
            latency = (time.time() - start_time) * 1000
            
            # Calculate NDCG@10
            retrieved_relevances = []
            for result in retrieval_results:
                # Find matching candidate
                for candidate in candidates:
                    if candidate["text"] == result.content:
                        retrieved_relevances.append(candidate["relevance"])
                        break
                else:
                    retrieved_relevances.append(0.0)
            
            # Pad to 10 if needed
            while len(retrieved_relevances) < 10:
                retrieved_relevances.append(0.0)
            
            # Calculate NDCG
            ideal_relevances = sorted([c["relevance"] for c in candidates], reverse=True)[:10]
            ndcg = calculate_ndcg(retrieved_relevances[:10], ideal_relevances)
            
            results[method]["ndcg"].append(ndcg)
            results[method]["latency"].append(latency)
            
            logger.info(f"    NDCG@10: {ndcg:.4f}, Latency: {latency:.1f}ms")
    
    # Analyze results
    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)
    
    for method in ["quantum", "classical", "hybrid"]:
        ndcg_mean = np.mean(results[method]["ndcg"])
        ndcg_std = np.std(results[method]["ndcg"])
        latency_mean = np.mean(results[method]["latency"])
        
        logger.info(f"\n{method.upper()} METHOD:")
        logger.info(f"  NDCG@10: {ndcg_mean:.4f} ± {ndcg_std:.4f}")
        logger.info(f"  Latency: {latency_mean:.1f}ms")
    
    # Calculate quantum advantage
    quantum_ndcg = np.mean(results["quantum"]["ndcg"])
    classical_ndcg = np.mean(results["classical"]["ndcg"])
    improvement = ((quantum_ndcg - classical_ndcg) / classical_ndcg) * 100
    
    logger.info(f"\n" + "=" * 80)
    logger.info("QUANTUM ADVANTAGE ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Quantum vs Classical NDCG@10 improvement: {improvement:.2f}%")
    
    # Statistical significance
    from scipy import stats
    if len(results["quantum"]["ndcg"]) > 1:
        t_stat, p_value = stats.ttest_rel(results["quantum"]["ndcg"], results["classical"]["ndcg"])
        logger.info(f"Statistical significance: p = {p_value:.4f}")
        is_significant = p_value < 0.05
        logger.info(f"Statistically significant: {'Yes' if is_significant else 'No'}")
    
    # Final verdict
    logger.info(f"\n" + "=" * 80)
    logger.info("VERDICT")
    logger.info("=" * 80)
    
    if improvement > 5 and (len(results["quantum"]["ndcg"]) <= 1 or p_value < 0.05):
        logger.info("✅ QUANTUM RERANKER SHOWS SIGNIFICANT ADVANTAGE")
        logger.info(f"   Performance improvement: {improvement:.2f}%")
        logger.info("   Recommended for deployment on complex medical queries")
    elif improvement > 0:
        logger.info("⚡ QUANTUM RERANKER SHOWS MODEST IMPROVEMENT")
        logger.info(f"   Performance improvement: {improvement:.2f}%")
        logger.info("   May be beneficial for specific use cases")
    else:
        logger.info("❌ NO QUANTUM ADVANTAGE OBSERVED")
        logger.info(f"   Performance difference: {improvement:.2f}%")
        logger.info("   Classical methods sufficient for current use cases")
    
    return results


def calculate_ndcg(relevances, ideal_relevances):
    """Calculate NDCG@k."""
    def dcg(relevances):
        return sum([(2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevances)])
    
    dcg_score = dcg(relevances)
    idcg_score = dcg(ideal_relevances)
    
    return dcg_score / idcg_score if idcg_score > 0 else 0


if __name__ == "__main__":
    test_quantum_reranker()
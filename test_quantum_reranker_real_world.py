#!/usr/bin/env python3
"""
Real-World Quantum Reranker Evaluation

This test evaluates the actual quantum reranker performance in scenarios where
quantum computing provides genuine advantages:
1. Complex, ambiguous medical queries requiring contextual understanding
2. Multi-modal data with inherent uncertainty
3. Cases where classical approaches struggle with superposition of meanings
4. Scenarios requiring nuanced similarity beyond simple keyword matching
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
from quantum_rerank.retrieval.two_stage_retriever import TwoStageRetriever
from quantum_rerank.retrieval.faiss_store import QuantumFAISSStore
from quantum_rerank.retrieval.document_store import DocumentStore

# Import evaluation components
from quantum_rerank.evaluation.realistic_medical_dataset_generator import (
    RealisticMedicalDatasetGenerator, MedicalTerminologyDatabase
)
from quantum_rerank.evaluation.unbiased_evaluation_framework import (
    UnbiasedEvaluationFramework
)
from quantum_rerank.config.settings import QuantumConfig
from quantum_rerank.config.evaluation_config import MultimodalMedicalEvaluationConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuantumAdvantageScenarioGenerator:
    """Generates scenarios where quantum computing provides real advantages."""
    
    def __init__(self):
        self.terminology_db = MedicalTerminologyDatabase()
        
        # Define quantum-advantage scenarios
        self.quantum_advantage_scenarios = {
            "ambiguous_symptoms": {
                "description": "Symptoms that could indicate multiple conditions",
                "quantum_benefit": "Superposition allows considering multiple interpretations simultaneously"
            },
            "rare_disease_patterns": {
                "description": "Rare diseases with overlapping symptoms",
                "quantum_benefit": "Quantum interference helps identify subtle patterns"
            },
            "drug_interactions": {
                "description": "Complex multi-drug interactions",
                "quantum_benefit": "Entanglement captures complex dependencies"
            },
            "differential_diagnosis": {
                "description": "Cases requiring nuanced differential diagnosis",
                "quantum_benefit": "Quantum states represent uncertainty in diagnosis"
            },
            "contextual_interpretation": {
                "description": "Same symptoms, different contexts (age, history, etc.)",
                "quantum_benefit": "Quantum circuits adapt to contextual information"
            }
        }
    
    def generate_quantum_advantage_queries(self, num_queries: int = 20) -> List[Dict[str, Any]]:
        """Generate queries specifically designed for quantum advantage."""
        queries = []
        
        for i in range(num_queries):
            scenario_type = np.random.choice(list(self.quantum_advantage_scenarios.keys()))
            scenario = self.quantum_advantage_scenarios[scenario_type]
            
            if scenario_type == "ambiguous_symptoms":
                query = self._generate_ambiguous_symptom_query()
            elif scenario_type == "rare_disease_patterns":
                query = self._generate_rare_disease_query()
            elif scenario_type == "drug_interactions":
                query = self._generate_drug_interaction_query()
            elif scenario_type == "differential_diagnosis":
                query = self._generate_differential_diagnosis_query()
            else:  # contextual_interpretation
                query = self._generate_contextual_query()
            
            query["scenario_type"] = scenario_type
            query["quantum_benefit"] = scenario["quantum_benefit"]
            query["id"] = f"quantum_advantage_{scenario_type}_{i:03d}"
            
            queries.append(query)
        
        return queries
    
    def _generate_ambiguous_symptom_query(self) -> Dict[str, Any]:
        """Generate query with ambiguous symptoms."""
        # Symptoms that could indicate multiple conditions
        ambiguous_combinations = [
            {
                "symptoms": ["chest pain", "shortness of breath", "fatigue"],
                "possible_conditions": ["myocardial infarction", "pulmonary embolism", "anxiety disorder", "heart failure"],
                "context": "45-year-old with recent travel history"
            },
            {
                "symptoms": ["headache", "vision changes", "nausea"],
                "possible_conditions": ["migraine", "stroke", "brain tumor", "hypertensive crisis"],
                "context": "sudden onset in previously healthy patient"
            },
            {
                "symptoms": ["abdominal pain", "fever", "vomiting"],
                "possible_conditions": ["appendicitis", "gastroenteritis", "pancreatitis", "bowel obstruction"],
                "context": "progressive worsening over 24 hours"
            }
        ]
        
        combination = np.random.choice(ambiguous_combinations)
        
        query_text = f"Patient presenting with {', '.join(combination['symptoms'])}. "
        query_text += f"{combination['context']}. "
        query_text += "What are the most likely diagnoses and recommended diagnostic approach?"
        
        return {
            "text": query_text,
            "symptoms": combination["symptoms"],
            "possible_conditions": combination["possible_conditions"],
            "ambiguity_level": "high",
            "requires_context": True
        }
    
    def _generate_rare_disease_query(self) -> Dict[str, Any]:
        """Generate query about rare diseases with overlapping symptoms."""
        rare_disease_scenarios = [
            {
                "symptoms": ["muscle weakness", "ptosis", "dysphagia"],
                "rare_condition": "myasthenia gravis",
                "common_conditions": ["stroke", "muscular dystrophy"],
                "distinguishing_features": ["fluctuating symptoms", "fatigability"]
            },
            {
                "symptoms": ["joint pain", "skin rash", "kidney problems"],
                "rare_condition": "systemic lupus erythematosus",
                "common_conditions": ["rheumatoid arthritis", "psoriasis"],
                "distinguishing_features": ["malar rash", "photosensitivity"]
            }
        ]
        
        scenario = np.random.choice(rare_disease_scenarios)
        
        query_text = f"Patient with {', '.join(scenario['symptoms'])}. "
        query_text += f"Additional features include {', '.join(scenario['distinguishing_features'])}. "
        query_text += "Consider both common and rare diagnoses."
        
        return {
            "text": query_text,
            "symptoms": scenario["symptoms"],
            "target_condition": scenario["rare_condition"],
            "differential": scenario["common_conditions"],
            "pattern_complexity": "high"
        }
    
    def _generate_drug_interaction_query(self) -> Dict[str, Any]:
        """Generate complex drug interaction query."""
        interaction_scenarios = [
            {
                "current_meds": ["warfarin", "amiodarone", "aspirin"],
                "new_medication": "fluconazole",
                "interaction_type": "multiple CYP interactions",
                "risk_level": "high"
            },
            {
                "current_meds": ["metformin", "lisinopril", "atorvastatin"],
                "new_medication": "clarithromycin",
                "interaction_type": "QT prolongation risk",
                "risk_level": "moderate"
            }
        ]
        
        scenario = np.random.choice(interaction_scenarios)
        
        query_text = f"Patient on {', '.join(scenario['current_meds'])}. "
        query_text += f"Considering adding {scenario['new_medication']}. "
        query_text += "Assess drug interactions and recommend monitoring."
        
        return {
            "text": query_text,
            "medications": scenario["current_meds"] + [scenario["new_medication"]],
            "interaction_complexity": len(scenario["current_meds"]) + 1,
            "quantum_advantage": "captures non-linear interaction effects"
        }
    
    def _generate_differential_diagnosis_query(self) -> Dict[str, Any]:
        """Generate nuanced differential diagnosis query."""
        differential_scenarios = [
            {
                "presentation": "acute dyspnea with clear lungs on auscultation",
                "key_differentials": ["pulmonary embolism", "anxiety", "metabolic acidosis", "anemia"],
                "critical_factors": ["D-dimer", "ABG", "anxiety history"],
                "uncertainty_level": "high"
            },
            {
                "presentation": "chronic fatigue with normal basic labs",
                "key_differentials": ["depression", "thyroid disease", "chronic fatigue syndrome", "occult malignancy"],
                "critical_factors": ["TSH", "depression screening", "age-appropriate cancer screening"],
                "uncertainty_level": "very high"
            }
        ]
        
        scenario = np.random.choice(differential_scenarios)
        
        query_text = f"Patient with {scenario['presentation']}. "
        query_text += f"Key considerations include {', '.join(scenario['critical_factors'][:2])}. "
        query_text += "Provide differential diagnosis with probability estimates."
        
        return {
            "text": query_text,
            "presentation": scenario["presentation"],
            "differentials": scenario["key_differentials"],
            "uncertainty": scenario["uncertainty_level"],
            "requires_probabilistic_reasoning": True
        }
    
    def _generate_contextual_query(self) -> Dict[str, Any]:
        """Generate query where context dramatically changes interpretation."""
        contextual_scenarios = [
            {
                "symptom": "chest pain",
                "context1": "20-year-old athlete after exercise",
                "context2": "65-year-old diabetic with hypertension",
                "interpretation_change": "benign vs life-threatening"
            },
            {
                "symptom": "weight loss",
                "context1": "intentional dieting with exercise",
                "context2": "unintentional with night sweats",
                "interpretation_change": "healthy vs concerning"
            }
        ]
        
        scenario = np.random.choice(contextual_scenarios)
        context = np.random.choice([scenario["context1"], scenario["context2"]])
        
        query_text = f"{context} presenting with {scenario['symptom']}. "
        query_text += "Evaluate significance and recommend management."
        
        return {
            "text": query_text,
            "primary_symptom": scenario["symptom"],
            "context": context,
            "context_criticality": "essential",
            "interpretation_variance": scenario["interpretation_change"]
        }
    
    def generate_challenging_candidates(self, query: Dict[str, Any], num_candidates: int = 50) -> List[Dict[str, Any]]:
        """Generate candidates that are challenging to rank correctly."""
        candidates = []
        
        # Generate highly relevant candidates (20%)
        for i in range(num_candidates // 5):
            candidate = self._generate_relevant_candidate(query, relevance="high")
            candidate["id"] = f"{query['id']}_candidate_{i:03d}"
            candidate["true_relevance"] = 0.9 + np.random.uniform(-0.05, 0.05)
            candidates.append(candidate)
        
        # Generate moderately relevant candidates (30%)
        for i in range(num_candidates // 5, num_candidates // 2):
            candidate = self._generate_relevant_candidate(query, relevance="moderate")
            candidate["id"] = f"{query['id']}_candidate_{i:03d}"
            candidate["true_relevance"] = 0.6 + np.random.uniform(-0.1, 0.1)
            candidates.append(candidate)
        
        # Generate subtly relevant candidates (30%) - these are the hardest
        for i in range(num_candidates // 2, 4 * num_candidates // 5):
            candidate = self._generate_relevant_candidate(query, relevance="subtle")
            candidate["id"] = f"{query['id']}_candidate_{i:03d}"
            candidate["true_relevance"] = 0.4 + np.random.uniform(-0.1, 0.1)
            candidates.append(candidate)
        
        # Generate irrelevant candidates (20%)
        for i in range(4 * num_candidates // 5, num_candidates):
            candidate = self._generate_relevant_candidate(query, relevance="none")
            candidate["id"] = f"{query['id']}_candidate_{i:03d}"
            candidate["true_relevance"] = 0.1 + np.random.uniform(-0.05, 0.05)
            candidates.append(candidate)
        
        return candidates
    
    def _generate_relevant_candidate(self, query: Dict[str, Any], relevance: str) -> Dict[str, Any]:
        """Generate a candidate document with specified relevance level."""
        if relevance == "high":
            # Directly addresses the query
            if "possible_conditions" in query:
                condition = query["possible_conditions"][0]
                text = f"Clinical guidelines for {condition}. "
                text += f"Key symptoms include {', '.join(query.get('symptoms', []))}. "
                text += f"Diagnostic approach: {self._generate_diagnostic_approach(condition)}"
            else:
                text = f"Evidence-based approach to {query.get('presentation', 'clinical presentation')}. "
                text += self._generate_relevant_clinical_content(query)
        
        elif relevance == "moderate":
            # Related but not directly addressing
            if "possible_conditions" in query:
                condition = np.random.choice(query["possible_conditions"][1:])
                text = f"Review of {condition} management. "
                text += f"Common presentations and diagnostic considerations."
            else:
                text = f"General approach to similar clinical scenarios. "
                text += self._generate_somewhat_relevant_content(query)
        
        elif relevance == "subtle":
            # Requires quantum understanding to identify relevance
            text = self._generate_subtly_relevant_content(query)
        
        else:  # irrelevant
            # Different medical topic
            unrelated_topics = ["diabetes management", "hypertension guidelines", "vaccination schedules"]
            topic = np.random.choice(unrelated_topics)
            text = f"Guidelines for {topic}. " + self._generate_generic_medical_content()
        
        return {
            "text": text,
            "relevance_type": relevance,
            "length": len(text.split())
        }
    
    def _generate_diagnostic_approach(self, condition: str) -> str:
        """Generate diagnostic approach text."""
        approaches = {
            "myocardial infarction": "ECG, troponins, echocardiography",
            "pulmonary embolism": "D-dimer, CT angiography, V/Q scan",
            "stroke": "CT head, MRI brain, carotid ultrasound"
        }
        return approaches.get(condition, "comprehensive workup including labs and imaging")
    
    def _generate_relevant_clinical_content(self, query: Dict[str, Any]) -> str:
        """Generate relevant clinical content."""
        content = "Systematic evaluation includes: "
        if "symptoms" in query:
            content += f"Assessment of {', '.join(query['symptoms'])}. "
        content += "Risk stratification and evidence-based management. "
        content += "Consider patient-specific factors and comorbidities."
        return content
    
    def _generate_somewhat_relevant_content(self, query: Dict[str, Any]) -> str:
        """Generate somewhat relevant content."""
        content = "Clinical pearls for differential diagnosis. "
        content += "Important to consider both common and rare causes. "
        content += "Systematic approach improves diagnostic accuracy."
        return content
    
    def _generate_subtly_relevant_content(self, query: Dict[str, Any]) -> str:
        """Generate subtly relevant content that requires deep understanding."""
        # These are cases where keyword matching fails but semantic understanding succeeds
        subtle_contents = [
            "Pattern recognition in clinical medicine: when presentations don't fit typical categories",
            "The importance of clinical gestalt in ambiguous presentations",
            "Bayesian reasoning in medical diagnosis: updating probabilities with new information",
            "Systems thinking approach to complex medical cases"
        ]
        
        content = np.random.choice(subtle_contents) + ". "
        
        # Add subtle connections to query
        if "uncertainty" in query:
            content += "Managing diagnostic uncertainty requires probabilistic thinking. "
        if "context" in query:
            content += "Context shapes clinical interpretation significantly. "
        
        return content
    
    def _generate_generic_medical_content(self) -> str:
        """Generate generic medical content."""
        topics = [
            "Primary prevention strategies in clinical practice",
            "Quality improvement initiatives in healthcare",
            "Patient safety protocols and best practices",
            "Healthcare disparities and social determinants"
        ]
        return np.random.choice(topics) + ". Evidence-based recommendations and implementation strategies."


class RealWorldQuantumRerankerEvaluator:
    """Evaluates quantum reranker in real-world scenarios."""
    
    def __init__(self):
        # Initialize quantum configuration
        self.quantum_config = SimilarityEngineConfig(
            n_qubits=4,
            n_layers=2,
            similarity_method=SimilarityMethod.HYBRID_WEIGHTED,
            enable_caching=False,  # Disable for fair comparison
            performance_monitoring=True,
            adaptive_weighting=True
        )
        
        # Initialize components
        logger.info("Initializing quantum reranker components...")
        self.embedding_processor = EmbeddingProcessor(EmbeddingConfig())
        self.quantum_engine = QuantumSimilarityEngine(self.quantum_config)
        
        # Initialize retrieval system
        from quantum_rerank.retrieval.two_stage_retriever import RetrieverConfig
        retriever_config = RetrieverConfig(
            initial_k=50,
            final_k=10,
            reranking_method="hybrid",
            enable_caching=False  # Disable for fair comparison
        )
        self.retriever = TwoStageRetriever(
            config=retriever_config,
            embedding_processor=self.embedding_processor
        )
        # Access the internal stores
        self.faiss_store = self.retriever.faiss_store
        self.document_store = self.retriever.document_store
        
        # Initialize scenario generator
        self.scenario_generator = QuantumAdvantageScenarioGenerator()
        
        # Metrics to track
        self.metrics = {
            "quantum": {"ndcg@10": [], "map": [], "mrr": [], "latency": []},
            "classical": {"ndcg@10": [], "map": [], "mrr": [], "latency": []},
            "hybrid": {"ndcg@10": [], "map": [], "mrr": [], "latency": []}
        }
    
    def run_real_world_evaluation(self, num_queries: int = 10):
        """Run comprehensive real-world evaluation."""
        logger.info("=" * 80)
        logger.info("REAL-WORLD QUANTUM RERANKER EVALUATION")
        logger.info("=" * 80)
        
        # Generate quantum-advantage queries
        logger.info(f"\n1. Generating {num_queries} quantum-advantage queries...")
        queries = self.scenario_generator.generate_quantum_advantage_queries(num_queries)
        
        # Show query distribution
        scenario_counts = {}
        for query in queries:
            scenario_type = query["scenario_type"]
            scenario_counts[scenario_type] = scenario_counts.get(scenario_type, 0) + 1
        
        logger.info("   Query distribution:")
        for scenario, count in scenario_counts.items():
            logger.info(f"   - {scenario}: {count} queries")
        
        # Process each query
        logger.info("\n2. Processing queries through quantum and classical systems...")
        
        for i, query in enumerate(queries):
            logger.info(f"\n   Processing query {i+1}/{num_queries}: {query['scenario_type']}")
            
            # Generate challenging candidates
            candidates = self.scenario_generator.generate_challenging_candidates(query)
            
            # Add documents to retriever (handles both document store and FAISS)
            texts = [candidate["text"] for candidate in candidates]
            metadatas = [{"relevance": candidate["true_relevance"]} for candidate in candidates]
            
            # Add all documents at once
            doc_ids = self.retriever.add_texts(texts, metadatas)
            
            # Get query embedding
            query_embedding = self.embedding_processor.encode_texts([query["text"]])[0]
            
            # Evaluate different methods using the retriever
            self._evaluate_retrieval_method(query, candidates, "quantum")
            self._evaluate_retrieval_method(query, candidates, "classical") 
            self._evaluate_retrieval_method(query, candidates, "hybrid")
            
            # Clear for next query - reinitialize retriever
            self._reinitialize_retriever()
        
        # Analyze results
        self._analyze_results(queries)
    
    def _reinitialize_retriever(self):
        """Clear retriever data for next query without recreating objects."""
        # Use the clear method which properly resets everything
        self.retriever.clear()
    
    def _evaluate_retrieval_method(self, query: Dict[str, Any], 
                                  candidates: List[Dict[str, Any]], method: str):
        """Evaluate using the two-stage retriever with different methods."""
        start_time = time.time()
        
        # Set the reranking method
        self.retriever.config.reranking_method = method
        
        # Perform retrieval
        results = self.retriever.retrieve(query["text"], k=10)
        
        latency = (time.time() - start_time) * 1000  # ms
        
        # Extract relevance scores from results using metadata
        relevances = []
        for result in results:
            # Use metadata if available, otherwise use content matching
            if hasattr(result, 'metadata') and result.metadata and 'relevance' in result.metadata:
                relevances.append(result.metadata['relevance'])
            else:
                # Fallback to content matching (less efficient)
                found = False
                for candidate in candidates:
                    if candidate["text"] == result.content:
                        relevances.append(candidate["true_relevance"])
                        found = True
                        break
                if not found:
                    relevances.append(0.1)
        
        # Pad with zeros if we got fewer than 10 results
        while len(relevances) < 10:
            relevances.append(0.0)
        
        # Calculate metrics
        relevances = relevances[:10]  # Ensure exactly 10
        
        # NDCG@10
        ideal_relevances = sorted([c["true_relevance"] for c in candidates], reverse=True)[:10]
        ndcg = self._calculate_ndcg(relevances, ideal_relevances)
        
        # MRR (Mean Reciprocal Rank)
        mrr = self._calculate_mrr(relevances)
        
        # MAP (Mean Average Precision)
        map_score = self._calculate_map(relevances)
        
        # Store metrics
        self.metrics[method]["ndcg@10"].append(ndcg)
        self.metrics[method]["mrr"].append(mrr)
        self.metrics[method]["map"].append(map_score)
        self.metrics[method]["latency"].append(latency)
    
    def _evaluate_method(self, query: Dict[str, Any], query_embedding: np.ndarray, 
                        candidates: List[Dict[str, Any]], method: str):
        """Evaluate a specific method."""
        start_time = time.time()
        
        # Get embeddings for all candidates
        candidate_embeddings = []
        for candidate in candidates[:50]:  # Limit to 50 for efficiency
            embedding = self.embedding_processor.encode_texts([candidate["text"]])[0]
            candidate_embeddings.append(embedding)
        
        # Calculate similarities using different methods
        similarities = []
        for candidate_embedding in candidate_embeddings:
            if method == "quantum":
                # Use quantum fidelity similarity
                similarity = self.quantum_engine.compute_similarity(
                    query_embedding, candidate_embedding, 
                    method=SimilarityMethod.QUANTUM_FIDELITY
                )
            elif method == "classical":
                # Use classical cosine similarity
                similarity = self.quantum_engine.compute_similarity(
                    query_embedding, candidate_embedding, 
                    method=SimilarityMethod.CLASSICAL_COSINE
                )
            else:  # hybrid
                # Use hybrid weighted similarity
                similarity = self.quantum_engine.compute_similarity(
                    query_embedding, candidate_embedding, 
                    method=SimilarityMethod.HYBRID_WEIGHTED
                )
            similarities.append(similarity)
        
        # Rank by similarity
        ranked_indices = np.argsort(similarities)[::-1]  # Descending order
        reranked_results = [(f"candidate_{i}", similarities[i]) for i in ranked_indices]
        
        latency = (time.time() - start_time) * 1000  # ms
        
        # Calculate metrics
        top_10_indices = ranked_indices[:10]
        relevances = [candidates[i]["true_relevance"] for i in top_10_indices]
        
        # NDCG@10
        ideal_relevances = sorted([c["true_relevance"] for c in candidates], reverse=True)[:10]
        ndcg = self._calculate_ndcg(relevances, ideal_relevances)
        
        # MRR (Mean Reciprocal Rank)
        mrr = self._calculate_mrr(relevances)
        
        # MAP (Mean Average Precision)
        map_score = self._calculate_map(relevances)
        
        # Store metrics
        self.metrics[method]["ndcg@10"].append(ndcg)
        self.metrics[method]["mrr"].append(mrr)
        self.metrics[method]["map"].append(map_score)
        self.metrics[method]["latency"].append(latency)
    
    def _calculate_ndcg(self, relevances: List[float], ideal_relevances: List[float]) -> float:
        """Calculate NDCG."""
        def dcg(relevances):
            return sum([(2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevances)])
        
        dcg_score = dcg(relevances)
        idcg_score = dcg(ideal_relevances)
        
        return dcg_score / idcg_score if idcg_score > 0 else 0
    
    def _calculate_mrr(self, relevances: List[float]) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, rel in enumerate(relevances):
            if rel >= 0.7:  # Consider as relevant
                return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_map(self, relevances: List[float]) -> float:
        """Calculate Mean Average Precision."""
        relevant_count = 0
        precision_sum = 0
        
        for i, rel in enumerate(relevances):
            if rel >= 0.7:  # Consider as relevant
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        
        return precision_sum / max(1, relevant_count)
    
    def _analyze_results(self, queries: List[Dict[str, Any]]):
        """Analyze and display results."""
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 80)
        
        # Calculate average metrics
        for method in ["quantum", "classical", "hybrid"]:
            logger.info(f"\n{method.upper()} METHOD:")
            logger.info(f"  - NDCG@10: {np.mean(self.metrics[method]['ndcg@10']):.4f} (±{np.std(self.metrics[method]['ndcg@10']):.4f})")
            logger.info(f"  - MAP: {np.mean(self.metrics[method]['map']):.4f} (±{np.std(self.metrics[method]['map']):.4f})")
            logger.info(f"  - MRR: {np.mean(self.metrics[method]['mrr']):.4f} (±{np.std(self.metrics[method]['mrr']):.4f})")
            logger.info(f"  - Avg Latency: {np.mean(self.metrics[method]['latency']):.1f}ms (±{np.std(self.metrics[method]['latency']):.1f}ms)")
        
        # Calculate quantum advantage
        logger.info("\n" + "=" * 80)
        logger.info("QUANTUM ADVANTAGE ANALYSIS")
        logger.info("=" * 80)
        
        quantum_ndcg = np.mean(self.metrics["quantum"]["ndcg@10"])
        classical_ndcg = np.mean(self.metrics["classical"]["ndcg@10"])
        improvement = ((quantum_ndcg - classical_ndcg) / classical_ndcg) * 100
        
        logger.info(f"\nOverall Performance Improvement: {improvement:.2f}%")
        
        # Analyze by scenario type
        logger.info("\nPerformance by Scenario Type:")
        scenario_improvements = {}
        
        for scenario_type in self.scenario_generator.quantum_advantage_scenarios.keys():
            # Get indices for this scenario type
            indices = [i for i, q in enumerate(queries) if q["scenario_type"] == scenario_type]
            
            if indices:
                quantum_scores = [self.metrics["quantum"]["ndcg@10"][i] for i in indices]
                classical_scores = [self.metrics["classical"]["ndcg@10"][i] for i in indices]
                
                scenario_improvement = ((np.mean(quantum_scores) - np.mean(classical_scores)) / np.mean(classical_scores)) * 100
                scenario_improvements[scenario_type] = scenario_improvement
                
                logger.info(f"  - {scenario_type}: {scenario_improvement:.2f}% improvement")
                logger.info(f"    Quantum benefit: {self.scenario_generator.quantum_advantage_scenarios[scenario_type]['quantum_benefit']}")
        
        # Statistical significance test
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(
            self.metrics["quantum"]["ndcg@10"],
            self.metrics["classical"]["ndcg@10"]
        )
        
        logger.info(f"\nStatistical Significance:")
        logger.info(f"  - t-statistic: {t_stat:.4f}")
        logger.info(f"  - p-value: {p_value:.6f}")
        logger.info(f"  - Significant: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")
        
        # Final verdict
        logger.info("\n" + "=" * 80)
        logger.info("FINAL VERDICT")
        logger.info("=" * 80)
        
        if improvement > 5 and p_value < 0.05:
            logger.info("✅ QUANTUM RERANKER SHOWS SIGNIFICANT ADVANTAGE")
            logger.info(f"   - {improvement:.2f}% improvement in NDCG@10")
            logger.info(f"   - Statistically significant (p={p_value:.6f})")
            logger.info("   - Particularly effective for:")
            
            for scenario, imp in sorted(scenario_improvements.items(), key=lambda x: x[1], reverse=True)[:3]:
                if imp > 0:
                    logger.info(f"     • {scenario}: {imp:.2f}% improvement")
        
        elif improvement > 0:
            logger.info("⚡ QUANTUM RERANKER SHOWS MODEST ADVANTAGE")
            logger.info(f"   - {improvement:.2f}% improvement in NDCG@10")
            logger.info(f"   - Statistical significance: {'Yes' if p_value < 0.05 else 'No'}")
        
        else:
            logger.info("❌ NO SIGNIFICANT QUANTUM ADVANTAGE OBSERVED")
            logger.info(f"   - Performance difference: {improvement:.2f}%")
        
        # Latency analysis
        quantum_latency = np.mean(self.metrics["quantum"]["latency"])
        classical_latency = np.mean(self.metrics["classical"]["latency"])
        
        logger.info(f"\nLatency Trade-off:")
        logger.info(f"  - Quantum: {quantum_latency:.1f}ms")
        logger.info(f"  - Classical: {classical_latency:.1f}ms")
        logger.info(f"  - Overhead: {quantum_latency - classical_latency:.1f}ms ({((quantum_latency - classical_latency) / classical_latency * 100):.1f}%)")
        
        # Recommendation
        logger.info("\n" + "=" * 80)
        logger.info("RECOMMENDATION")
        logger.info("=" * 80)
        
        if improvement > 10 and p_value < 0.01 and quantum_latency < 200:
            logger.info("🎯 STRONG RECOMMENDATION: Deploy quantum reranker for production use")
            logger.info("   - Significant performance gains justify quantum approach")
            logger.info("   - Latency within acceptable bounds")
            logger.info("   - Best suited for complex, ambiguous medical queries")
        elif improvement > 5 and p_value < 0.05:
            logger.info("📊 CONDITIONAL RECOMMENDATION: Consider quantum reranker for specific use cases")
            logger.info("   - Moderate performance gains in certain scenarios")
            logger.info("   - Evaluate cost-benefit for your specific application")
        else:
            logger.info("⏸️  RECOMMENDATION: Continue development before deployment")
            logger.info("   - Current quantum advantage not sufficient for production")
            logger.info("   - Focus on algorithm optimization and circuit design")


def main():
    """Run the real-world quantum reranker evaluation."""
    logger.info("Starting Real-World Quantum Reranker Evaluation")
    logger.info(f"Timestamp: {datetime.now()}")
    
    try:
        evaluator = RealWorldQuantumRerankerEvaluator()
        evaluator.run_real_world_evaluation(num_queries=10)
        
        logger.info("\n✅ Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
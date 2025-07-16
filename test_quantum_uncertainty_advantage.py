#!/usr/bin/env python3
"""
Quantum Uncertainty Advantage Test

This test specifically evaluates quantum reranking in scenarios where the routing
system identified quantum advantages:
1. High-noise emergency scenarios
2. Diagnostic uncertainty scenarios

The goal is to test whether quantum methods (superposition, uncertainty representation)
actually perform better than classical methods in these degraded conditions.
"""

import logging
import sys
import os
import time
import numpy as np
import json
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from quantum_rerank.core.embeddings import EmbeddingProcessor, EmbeddingConfig
from quantum_rerank.retrieval.two_stage_retriever import TwoStageRetriever, RetrieverConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QuantumUncertaintyTester:
    """Tests quantum advantage specifically in high-uncertainty scenarios."""
    
    def __init__(self):
        self.embedding_processor = EmbeddingProcessor(EmbeddingConfig())
        
        # Results for uncertainty-focused scenarios
        self.results = {
            "quantum": {"ndcg": [], "confidence": [], "uncertainty_handling": [], "latency": []},
            "classical": {"ndcg": [], "confidence": [], "uncertainty_handling": [], "latency": []},
            "hybrid": {"ndcg": [], "confidence": [], "uncertainty_handling": [], "latency": []}
        }
    
    def run_uncertainty_advantage_test(self):
        """Run tests specifically on high-uncertainty, noisy scenarios."""
        logger.info("=" * 80)
        logger.info("QUANTUM UNCERTAINTY ADVANTAGE TEST")
        logger.info("=" * 80)
        logger.info("Testing quantum methods in their optimal conditions:")
        logger.info("1. High-noise emergency scenarios")
        logger.info("2. Diagnostic uncertainty scenarios")
        logger.info("3. Missing/conflicting data scenarios")
        
        # Create uncertainty-focused test scenarios
        uncertainty_scenarios = self._create_uncertainty_scenarios()
        
        logger.info(f"\nCreated {len(uncertainty_scenarios)} uncertainty scenarios")
        
        for i, scenario in enumerate(uncertainty_scenarios):
            logger.info(f"\n--- Scenario {i+1}/{len(uncertainty_scenarios)}: {scenario['id']} ---")
            logger.info(f"Type: {scenario['type']}")
            logger.info(f"Uncertainty level: {scenario['uncertainty_level']:.2f}")
            logger.info(f"Query: {scenario['query'][:80]}...")
            
            # Test each method on this uncertain scenario
            for method in ["classical", "quantum", "hybrid"]:
                logger.info(f"\nTesting {method.upper()} method on uncertain scenario...")
                metrics = self._evaluate_uncertain_scenario(scenario, method)
                
                # Store results
                self.results[method]["ndcg"].append(metrics["ndcg"])
                self.results[method]["confidence"].append(metrics["confidence"])
                self.results[method]["uncertainty_handling"].append(metrics["uncertainty_handling"])
                self.results[method]["latency"].append(metrics["latency"])
                
                logger.info(f"  NDCG@5: {metrics['ndcg']:.4f}")
                logger.info(f"  Confidence: {metrics['confidence']:.4f}")
                logger.info(f"  Uncertainty Handling: {metrics['uncertainty_handling']:.4f}")
                logger.info(f"  Latency: {metrics['latency']:.1f}ms")
        
        # Analyze uncertainty-specific results
        self._analyze_uncertainty_results()
    
    def _create_uncertainty_scenarios(self) -> List[Dict[str, Any]]:
        """Create scenarios with high uncertainty where quantum should excel."""
        
        scenarios = [
            # High-noise emergency scenarios (identified as quantum-optimal by router)
            {
                "id": "noisy_trauma_emergency",
                "type": "noisy_emergency",
                "uncertainty_level": 0.8,
                "query": "Trauma patient, altered consciousness, hypotension, multiple potential injuries, limited history due to patient condition",
                "documents": [
                    {
                        "content": "Trauma evaluation protocols: ATLS guidelines for systematic assessment of trauma patients with altered consciousness. Primary survey focuses on airway, breathing, circulation. Secondary survey identifies specific injuries. Mechanism of injury guides evaluation priorities.",
                        "true_relevance": 0.90,
                        "noise_impact": "High relevance but query noise makes matching difficult"
                    },
                    {
                        "content": "Emergency management of unconscious patients: Rapid assessment protocols for altered mental status. Consider hypoglycemia, drug overdose, head injury, shock states. Stabilize vital signs while investigating cause.",
                        "true_relevance": 0.85,
                        "noise_impact": "Very relevant but missing trauma context due to noise"
                    },
                    {
                        "content": "Hypotension in emergency settings: Differential diagnosis includes hemorrhage, cardiogenic shock, sepsis, neurogenic shock. Rapid fluid resuscitation and identification of underlying cause critical for patient outcomes.",
                        "true_relevance": 0.75,
                        "noise_impact": "Relevant but partial information due to uncertainty"
                    },
                    {
                        "content": "Alcohol intoxication assessment: Clinical presentation includes altered consciousness, hypotension, hypothermia. Important to exclude other causes of altered mental status. Supportive care and monitoring.",
                        "true_relevance": 0.35,
                        "noise_impact": "Could be relevant but uncertain without history"
                    },
                    {
                        "content": "Routine blood pressure monitoring in stable patients. Standard protocols for hypertension screening and management in outpatient settings.",
                        "true_relevance": 0.10,
                        "noise_impact": "Not relevant but noise could make it seem relevant"
                    }
                ]
            },
            {
                "id": "pediatric_fever_uncertainty",
                "type": "noisy_emergency", 
                "uncertainty_level": 0.9,
                "query": "Pediatric patient, high fever, irritability, possible neck stiffness, uncooperative for examination, parent anxiety affecting history",
                "documents": [
                    {
                        "content": "Pediatric meningitis: Emergency evaluation of fever and neck stiffness in children. Clinical presentation may be subtle in young children. Lumbar puncture indicated if meningeal signs present. Empirical antibiotics should not be delayed.",
                        "true_relevance": 0.95,
                        "noise_impact": "Highest relevance but uncertainty about neck stiffness affects matching"
                    },
                    {
                        "content": "Febrile seizures in children: Common cause of altered behavior in pediatric patients with fever. Usually benign but requires evaluation for underlying infection. Most children recover completely without sequelae.",
                        "true_relevance": 0.70,
                        "noise_impact": "Could be relevant but depends on uncertain clinical findings"
                    },
                    {
                        "content": "Pediatric examination techniques: Strategies for examining uncooperative children. Distraction techniques, parental involvement, and patience improve examination quality. Age-appropriate communication essential.",
                        "true_relevance": 0.50,
                        "noise_impact": "Somewhat relevant for examination challenges"
                    },
                    {
                        "content": "Parental anxiety in pediatric emergency settings: Impact of caregiver stress on history taking and patient cooperation. Communication strategies to reduce anxiety and improve information quality.",
                        "true_relevance": 0.30,
                        "noise_impact": "Relevant to context but not primary medical issue"
                    },
                    {
                        "content": "Routine immunization schedules for healthy children. Standard vaccination protocols and timing for preventive care visits.",
                        "true_relevance": 0.05,
                        "noise_impact": "Not relevant to acute presentation"
                    }
                ]
            },
            # Diagnostic uncertainty scenarios (also identified as quantum-optimal)
            {
                "id": "chest_pain_uncertainty",
                "type": "uncertain",
                "uncertainty_level": 0.85,
                "query": "Intermittent chest pain, stress-related symptoms, unclear family history, atypical presentation, multiple possible causes",
                "documents": [
                    {
                        "content": "Atypical chest pain evaluation: When presentation doesn't fit classic patterns for cardiac, pulmonary, or GI causes. Requires systematic approach to exclude serious conditions while avoiding overinvestigation. Stress testing may be helpful.",
                        "true_relevance": 0.88,
                        "noise_impact": "Perfect match but uncertainty in presentation affects confidence"
                    },
                    {
                        "content": "Cardiac risk stratification in uncertain presentations: Use of clinical decision rules when history and examination are equivocal. Consider stress testing, CT angiography, or observation protocols based on risk factors.",
                        "true_relevance": 0.80,
                        "noise_impact": "Very relevant but uncertainty about risk factors affects applicability"
                    },
                    {
                        "content": "Anxiety and panic disorders presenting as chest pain: Somatic symptoms can perfectly mimic cardiac conditions. Important to exclude organic causes first. CBT and medication can be effective treatments.",
                        "true_relevance": 0.65,
                        "noise_impact": "Could be highly relevant if stress-related, but uncertain"
                    },
                    {
                        "content": "Family history assessment in cardiovascular disease: Importance of detailed family history for risk stratification. Genetic counseling may be appropriate for certain high-risk families.",
                        "true_relevance": 0.40,
                        "noise_impact": "Would be relevant but family history is unclear"
                    },
                    {
                        "content": "Routine ECG interpretation in asymptomatic patients. Normal variants and age-related changes in electrocardiography.",
                        "true_relevance": 0.15,
                        "noise_impact": "Not directly relevant to symptomatic patient"
                    }
                ]
            },
            {
                "id": "cognitive_decline_uncertainty",
                "type": "uncertain",
                "uncertainty_level": 0.90,
                "query": "Elderly patient with cognitive changes, depression vs dementia vs delirium, overlapping symptoms, unclear progression timeline",
                "documents": [
                    {
                        "content": "Differential diagnosis of cognitive impairment: Distinguishing between depression, dementia, and delirium in elderly patients. Requires careful history, cognitive testing, and sometimes longitudinal observation. Each condition has distinct features but overlap is common.",
                        "true_relevance": 0.92,
                        "noise_impact": "Excellent match but overlapping symptoms create diagnostic uncertainty"
                    },
                    {
                        "content": "Delirium in hospitalized elderly: Acute confusional state often superimposed on underlying cognitive impairment. Multiple contributing factors including medications, infections, and metabolic disturbances. Reversible with appropriate treatment.",
                        "true_relevance": 0.78,
                        "noise_impact": "Very relevant if delirium present, but uncertain from presentation"
                    },
                    {
                        "content": "Depression in elderly patients: Can present as pseudodementia with cognitive complaints and functional decline. Usually more rapid onset than true dementia. Responds well to antidepressant therapy.",
                        "true_relevance": 0.75,
                        "noise_impact": "Highly relevant if depression primary, but uncertainty about mood symptoms"
                    },
                    {
                        "content": "Alzheimer's disease progression: Gradual onset cognitive decline affecting memory, executive function, and daily activities. Biomarkers and neuroimaging can support diagnosis. Cholinesterase inhibitors may provide modest benefit.",
                        "true_relevance": 0.70,
                        "noise_impact": "Relevant if dementia, but unclear progression timeline affects diagnosis"
                    },
                    {
                        "content": "Medication management in elderly patients: Polypharmacy considerations and drug interactions in geriatric populations. Regular medication review important for safety.",
                        "true_relevance": 0.25,
                        "noise_impact": "Potentially relevant but not addressing primary cognitive concerns"
                    }
                ]
            }
        ]
        
        return scenarios
    
    def _evaluate_uncertain_scenario(self, scenario: Dict[str, Any], method: str) -> Dict[str, float]:
        """Evaluate a method on an uncertain scenario."""
        
        # Create retriever with specific method
        config = RetrieverConfig(
            initial_k=len(scenario["documents"]),
            final_k=5,
            reranking_method=method,
            enable_caching=False
        )
        
        retriever = TwoStageRetriever(config=config, embedding_processor=self.embedding_processor)
        
        # Prepare documents with uncertainty indicators
        texts = [doc["content"] for doc in scenario["documents"]]
        metadatas = [
            {
                "true_relevance": doc["true_relevance"],
                "noise_impact": doc["noise_impact"],
                "uncertainty_level": scenario["uncertainty_level"]
            }
            for doc in scenario["documents"]
        ]
        
        start_time = time.time()
        
        # Add documents and retrieve
        doc_ids = retriever.add_texts(texts, metadatas)
        
        retrieval_start = time.time()
        results = retriever.retrieve(scenario["query"], k=5)
        latency = (time.time() - retrieval_start) * 1000
        
        # Calculate uncertainty-specific metrics
        retrieved_relevances = []
        uncertainty_scores = []
        
        for result in results:
            # Find true relevance
            for doc in scenario["documents"]:
                if doc["content"] == result.content:
                    retrieved_relevances.append(doc["true_relevance"])
                    # Uncertainty handling score: how well does the method rank uncertain but relevant docs?
                    uncertainty_penalty = 1.0 - scenario["uncertainty_level"]
                    uncertainty_scores.append(doc["true_relevance"] * (1.0 + uncertainty_penalty))
                    break
            else:
                retrieved_relevances.append(0.0)
                uncertainty_scores.append(0.0)
        
        # Pad to 5 results
        while len(retrieved_relevances) < 5:
            retrieved_relevances.append(0.0)
            uncertainty_scores.append(0.0)
        
        # Calculate metrics
        true_relevances = [doc["true_relevance"] for doc in scenario["documents"]]
        ideal_relevances = sorted(true_relevances, reverse=True)[:5]
        
        # NDCG@5
        ndcg = self._calculate_ndcg(retrieved_relevances, ideal_relevances)
        
        # Confidence score (higher is better for uncertain scenarios)
        confidence = self._calculate_confidence_in_uncertainty(retrieved_relevances, scenario["uncertainty_level"])
        
        # Uncertainty handling score (how well method deals with uncertain but relevant docs)
        uncertainty_handling = self._calculate_uncertainty_handling(uncertainty_scores, scenario["uncertainty_level"])
        
        return {
            "ndcg": ndcg,
            "confidence": confidence,
            "uncertainty_handling": uncertainty_handling,
            "latency": latency
        }
    
    def _calculate_ndcg(self, retrieved: List[float], ideal: List[float]) -> float:
        """Calculate NDCG@k."""
        def dcg(relevances):
            return sum([(2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevances)])
        
        dcg_score = dcg(retrieved)
        idcg_score = dcg(ideal)
        
        return dcg_score / idcg_score if idcg_score > 0 else 0
    
    def _calculate_confidence_in_uncertainty(self, relevances: List[float], uncertainty_level: float) -> float:
        """Calculate how confident the method is in uncertain conditions."""
        # In high uncertainty, we want methods that can still identify relevant docs
        # even when the signal is noisy
        
        if not relevances or max(relevances) == 0:
            return 0.0
        
        # Reward finding relevant docs despite uncertainty
        top_relevance = max(relevances)
        relevance_spread = max(relevances) - min(relevances)
        
        # Higher uncertainty makes good performance more valuable
        uncertainty_bonus = 1.0 + uncertainty_level
        
        confidence = (top_relevance * 0.6 + relevance_spread * 0.4) * uncertainty_bonus
        return min(confidence, 1.0)
    
    def _calculate_uncertainty_handling(self, uncertainty_scores: List[float], uncertainty_level: float) -> float:
        """Calculate how well the method handles uncertainty."""
        # This measures whether the method can rank uncertain but relevant docs appropriately
        
        if not uncertainty_scores:
            return 0.0
        
        # Weight by position (early positions matter more)
        weighted_score = sum(score * (1.0 / (i + 1)) for i, score in enumerate(uncertainty_scores))
        max_possible = sum(1.0 / (i + 1) for i in range(len(uncertainty_scores)))
        
        return weighted_score / max_possible if max_possible > 0 else 0.0
    
    def _analyze_uncertainty_results(self):
        """Analyze results specifically for uncertainty handling."""
        logger.info("\n" + "=" * 80)
        logger.info("UNCERTAINTY ADVANTAGE ANALYSIS")
        logger.info("=" * 80)
        
        # Calculate averages for uncertainty-specific metrics
        for method in ["classical", "quantum", "hybrid"]:
            ndcg_avg = np.mean(self.results[method]["ndcg"])
            confidence_avg = np.mean(self.results[method]["confidence"])
            uncertainty_avg = np.mean(self.results[method]["uncertainty_handling"])
            latency_avg = np.mean(self.results[method]["latency"])
            
            logger.info(f"\n{method.upper()} METHOD (Uncertainty-Optimized):")
            logger.info(f"  NDCG@5: {ndcg_avg:.4f}")
            logger.info(f"  Uncertainty Confidence: {confidence_avg:.4f}")
            logger.info(f"  Uncertainty Handling: {uncertainty_avg:.4f}")
            logger.info(f"  Latency: {latency_avg:.1f}ms")
        
        # Quantum advantage in uncertainty scenarios
        logger.info("\n" + "=" * 80)
        logger.info("QUANTUM UNCERTAINTY ADVANTAGE")
        logger.info("=" * 80)
        
        quantum_ndcg = np.mean(self.results["quantum"]["ndcg"])
        classical_ndcg = np.mean(self.results["classical"]["ndcg"])
        quantum_uncertainty = np.mean(self.results["quantum"]["uncertainty_handling"])
        classical_uncertainty = np.mean(self.results["classical"]["uncertainty_handling"])
        
        ndcg_improvement = ((quantum_ndcg - classical_ndcg) / classical_ndcg) * 100 if classical_ndcg > 0 else 0
        uncertainty_improvement = ((quantum_uncertainty - classical_uncertainty) / classical_uncertainty) * 100 if classical_uncertainty > 0 else 0
        
        logger.info(f"NDCG@5 Improvement: {ndcg_improvement:+.2f}%")
        logger.info(f"Uncertainty Handling Improvement: {uncertainty_improvement:+.2f}%")
        
        # Scenario-specific analysis
        logger.info(f"\nScenario-Specific Quantum Advantage:")
        scenario_types = ["noisy_emergency", "uncertain"]
        for i, stype in enumerate([s["type"] for s in self._create_uncertainty_scenarios()][:2]):  # Just first 2 types
            q_ndcg = self.results["quantum"]["ndcg"][i] if i < len(self.results["quantum"]["ndcg"]) else 0
            c_ndcg = self.results["classical"]["ndcg"][i] if i < len(self.results["classical"]["ndcg"]) else 0
            
            improvement = ((q_ndcg - c_ndcg) / c_ndcg) * 100 if c_ndcg > 0 else 0
            logger.info(f"  {stype}: {improvement:+.1f}%")
        
        # Final verdict for uncertainty scenarios
        logger.info("\n" + "=" * 80)
        logger.info("QUANTUM UNCERTAINTY VERDICT")
        logger.info("=" * 80)
        
        if uncertainty_improvement > 10:
            logger.info("✅ QUANTUM ADVANTAGE IN UNCERTAINTY SCENARIOS")
            logger.info(f"   Quantum shows {uncertainty_improvement:.1f}% better uncertainty handling")
            logger.info("   Recommendation: Use quantum for high-uncertainty medical cases")
        elif uncertainty_improvement > 0:
            logger.info("⚡ MODEST QUANTUM UNCERTAINTY ADVANTAGE")
            logger.info(f"   Quantum shows {uncertainty_improvement:.1f}% better uncertainty handling")
            logger.info("   Recommendation: Consider quantum for critical uncertain cases")
        else:
            logger.info("📊 NO QUANTUM UNCERTAINTY ADVANTAGE")
            logger.info("   Classical methods handle uncertainty as well or better")
            logger.info("   Recommendation: Focus on quantum algorithm optimization")
        
        # Save uncertainty-specific results
        uncertainty_results = {
            "timestamp": datetime.now().isoformat(),
            "test_focus": "uncertainty_and_noise_scenarios",
            "scenarios_tested": len(self.results["quantum"]["ndcg"]),
            "quantum_uncertainty_advantage": uncertainty_improvement,
            "quantum_ndcg_advantage": ndcg_improvement,
            "detailed_metrics": self.results
        }
        
        with open("quantum_uncertainty_results.json", "w") as f:
            json.dump(uncertainty_results, f, indent=2, default=float)
        
        logger.info(f"\nUncertainty-focused results saved to: quantum_uncertainty_results.json")


def main():
    """Run quantum uncertainty advantage testing."""
    logger.info("Starting Quantum Uncertainty Advantage Test")
    logger.info(f"Focus: Testing quantum methods in high-uncertainty, noisy scenarios")
    logger.info(f"Timestamp: {datetime.now()}")
    
    try:
        tester = QuantumUncertaintyTester()
        tester.run_uncertainty_advantage_test()
        
        logger.info("\n✅ Quantum uncertainty advantage test completed")
        
    except Exception as e:
        logger.error(f"Error during uncertainty testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
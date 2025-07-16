#!/usr/bin/env python3
"""
Real-World Medical Reranking Test

This test evaluates the quantum reranker in realistic medical scenarios where
quantum computing could provide genuine advantages. The test is designed to:

1. Use realistic medical documents and queries
2. Create scenarios where quantum superposition and entanglement matter
3. Provide meaningful performance metrics
4. Compare quantum, classical, and hybrid approaches fairly
"""

import logging
import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from quantum_rerank.core.embeddings import EmbeddingProcessor, EmbeddingConfig
from quantum_rerank.retrieval.two_stage_retriever import TwoStageRetriever, RetrieverConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MedicalScenarioGenerator:
    """Generates realistic medical scenarios for testing quantum advantage."""
    
    def __init__(self):
        self.medical_scenarios = self._load_medical_scenarios()
    
    def _load_medical_scenarios(self) -> List[Dict[str, Any]]:
        """Load realistic medical scenarios with known correct rankings."""
        return [
            {
                "id": "chest_pain_pe_vs_mi",
                "query": "35-year-old female with acute chest pain and dyspnea after long flight. D-dimer elevated. What is the most likely diagnosis and diagnostic approach?",
                "scenario_type": "differential_diagnosis",
                "documents": [
                    {
                        "content": "Pulmonary embolism in young adults: Clinical presentation typically includes acute dyspnea, pleuritic chest pain, and tachycardia. Risk factors include recent travel, oral contraceptives, and immobilization. D-dimer elevation supports diagnosis but lacks specificity. CT pulmonary angiogram (CTPA) is gold standard for diagnosis. Wells score helps risk stratification. Treatment involves anticoagulation with heparin followed by warfarin or DOACs.",
                        "relevance": 0.95,
                        "reasoning": "Directly addresses PE in context of travel and elevated D-dimer"
                    },
                    {
                        "content": "Acute coronary syndrome in women: Presentation may be atypical with dyspnea, fatigue, and jaw pain rather than classic crushing chest pain. ECG may show ST changes. Troponin elevation confirms myocardial damage. Risk factors include diabetes, hypertension, smoking, and family history. Treatment includes dual antiplatelet therapy, statins, and revascularization if indicated.",
                        "relevance": 0.70,
                        "reasoning": "Relevant for chest pain but less likely given age and travel history"
                    },
                    {
                        "content": "Anxiety disorders and panic attacks: Can present with chest pain, dyspnea, palpitations, and diaphoresis mimicking cardiac conditions. Important to rule out organic causes first. Young women are particularly susceptible. Treatment includes cognitive behavioral therapy and SSRIs.",
                        "relevance": 0.40,
                        "reasoning": "Possible but less likely given clinical context"
                    },
                    {
                        "content": "Pneumonia in adults: Presents with cough, fever, dyspnea, and chest pain. Chest X-ray shows infiltrates. Sputum culture and blood tests guide antibiotic selection. Hospitalization may be required based on severity scores.",
                        "relevance": 0.25,
                        "reasoning": "Less likely without fever and cough"
                    },
                    {
                        "content": "Gastroesophageal reflux disease: Can cause chest pain that mimics cardiac pain. Associated with heartburn, regurgitation, and dysphagia. Responds to proton pump inhibitors. Endoscopy may be needed for evaluation.",
                        "relevance": 0.15,
                        "reasoning": "Unlikely to cause acute dyspnea and D-dimer elevation"
                    }
                ]
            },
            {
                "id": "abdominal_pain_appendicitis",
                "query": "22-year-old male with progressive right lower quadrant pain, nausea, and low-grade fever. Pain started periumbilically and migrated. What is the most likely diagnosis?",
                "scenario_type": "progressive_symptoms",
                "documents": [
                    {
                        "content": "Acute appendicitis: Classic presentation includes periumbilical pain that migrates to right lower quadrant (McBurney's point), nausea, vomiting, and low-grade fever. Physical exam may show rebound tenderness, Rovsing's sign, and psoas sign. CT scan with contrast is diagnostic study of choice. Appendectomy is definitive treatment, preferably laparoscopic.",
                        "relevance": 0.98,
                        "reasoning": "Perfect match for classic appendicitis presentation"
                    },
                    {
                        "content": "Inflammatory bowel disease (Crohn's disease): Can present with right lower quadrant pain, especially with ileocecal involvement. Associated with diarrhea, weight loss, and extraintestinal manifestations. CT enterography and colonoscopy aid diagnosis. Treatment includes immunosuppressants and biologics.",
                        "relevance": 0.45,
                        "reasoning": "Can cause RLQ pain but less acute presentation"
                    },
                    {
                        "content": "Ovarian torsion: Acute onset severe pelvic pain, often with nausea and vomiting. More common in women of reproductive age. Ultrasound with Doppler shows decreased ovarian blood flow. Surgical detorsion and oophoropexy required emergently.",
                        "relevance": 0.05,
                        "reasoning": "Wrong gender, patient is male"
                    },
                    {
                        "content": "Gastroenteritis: Acute onset diarrhea, vomiting, and crampy abdominal pain. Usually self-limited. Supportive care with hydration. Antibiotics rarely needed unless bacterial cause suspected.",
                        "relevance": 0.30,
                        "reasoning": "Could cause abdominal pain but not typical migration pattern"
                    },
                    {
                        "content": "Kidney stones (nephrolithiasis): Severe colicky flank pain radiating to groin. May have hematuria. CT without contrast is diagnostic. Pain management and alpha-blockers may facilitate passage. Lithotripsy for larger stones.",
                        "relevance": 0.20,
                        "reasoning": "Different pain pattern and location"
                    }
                ]
            },
            {
                "id": "headache_differential",
                "query": "45-year-old woman with sudden severe headache, worst of her life, with neck stiffness and photophobia. What is the most concerning diagnosis?",
                "scenario_type": "emergency_differential",
                "documents": [
                    {
                        "content": "Subarachnoid hemorrhage: Presents with sudden onset 'thunderclap' headache, often described as 'worst headache of life.' Associated with neck stiffness, photophobia, and altered consciousness. Non-contrast CT head is first-line imaging, but lumbar puncture may be needed if CT negative. Aneurysmal SAH requires urgent neurosurgical intervention.",
                        "relevance": 0.95,
                        "reasoning": "Classic presentation for SAH - medical emergency"
                    },
                    {
                        "content": "Bacterial meningitis: Acute onset headache, fever, neck stiffness, and photophobia. Mental status changes common. Lumbar puncture shows elevated white cells, protein, and decreased glucose. Blood cultures important. Empirical antibiotic therapy should not be delayed for imaging.",
                        "relevance": 0.85,
                        "reasoning": "Very similar presentation, also life-threatening"
                    },
                    {
                        "content": "Migraine headache: Severe unilateral throbbing headache with photophobia, phonophobia, nausea. May have aura. Triggers include stress, foods, hormonal changes. Treatment includes triptans, NSAIDs, and preventive medications.",
                        "relevance": 0.30,
                        "reasoning": "Could cause severe headache but 'worst of life' and neck stiffness concerning"
                    },
                    {
                        "content": "Tension headache: Bilateral, pressing/tightening quality headache. Often associated with stress or muscle tension. No photophobia or nausea typically. Treatment with acetaminophen or NSAIDs.",
                        "relevance": 0.10,
                        "reasoning": "Doesn't fit severe presentation with meningeal signs"
                    },
                    {
                        "content": "Cluster headache: Severe unilateral orbital/temporal pain with ipsilateral autonomic features (lacrimation, nasal congestion). Occurs in clusters over weeks to months. Treatment includes high-flow oxygen and triptans.",
                        "relevance": 0.20,
                        "reasoning": "Severe but different pattern and no neck stiffness"
                    }
                ]
            },
            {
                "id": "dyspnea_heart_failure",
                "query": "68-year-old man with diabetes and hypertension presents with progressive dyspnea, orthopnea, and bilateral ankle swelling. What is the most likely diagnosis?",
                "scenario_type": "chronic_progressive",
                "documents": [
                    {
                        "content": "Heart failure with reduced ejection fraction: Progressive dyspnea, orthopnea, paroxysmal nocturnal dyspnea, and peripheral edema. Often follows myocardial infarction or chronic hypertension. Echocardiogram shows reduced EF <40%. BNP/NT-proBNP elevated. Treatment includes ACE inhibitors, beta-blockers, diuretics, and aldosterone antagonists.",
                        "relevance": 0.92,
                        "reasoning": "Perfect match for HFrEF with classic symptoms and risk factors"
                    },
                    {
                        "content": "Chronic obstructive pulmonary disease: Progressive dyspnea, chronic cough, sputum production. Associated with smoking history. Spirometry shows airflow obstruction. Treatment includes bronchodilators, inhaled corticosteroids, and pulmonary rehabilitation.",
                        "relevance": 0.50,
                        "reasoning": "Could cause dyspnea but doesn't explain ankle swelling"
                    },
                    {
                        "content": "Pulmonary embolism: Acute or subacute dyspnea, chest pain, tachycardia. Risk factors include immobilization, malignancy, hypercoagulable states. D-dimer screening, CTPA for diagnosis. Anticoagulation is mainstay of treatment.",
                        "relevance": 0.25,
                        "reasoning": "Could cause dyspnea but presentation is more progressive, not acute"
                    },
                    {
                        "content": "Chronic kidney disease: Progressive decline in kidney function. May cause fluid retention and hypertension. Associated with diabetes and hypertension. Creatinine and GFR indicate severity. May require dialysis or transplantation.",
                        "relevance": 0.40,
                        "reasoning": "Could contribute to fluid retention but primary issue is cardiac"
                    },
                    {
                        "content": "Sleep apnea: Loud snoring, witnessed apneas, daytime sleepiness. Associated with obesity and heart failure. Sleep study confirms diagnosis. Treatment with CPAP improves symptoms and cardiovascular outcomes.",
                        "relevance": 0.20,
                        "reasoning": "Could be comorbid condition but doesn't explain acute symptoms"
                    }
                ]
            },
            {
                "id": "cognitive_decline_dementia",
                "query": "72-year-old woman with 2-year history of progressive memory loss, difficulty with complex tasks, and personality changes. Family reports she gets lost in familiar places. What is the most likely diagnosis?",
                "scenario_type": "neurodegenerative",
                "documents": [
                    {
                        "content": "Alzheimer's disease: Progressive cognitive decline affecting memory, executive function, and visuospatial skills. Gradual onset with personality changes. MRI may show hippocampal atrophy. CSF biomarkers and PET amyloid imaging can support diagnosis. Treatment includes cholinesterase inhibitors and memantine.",
                        "relevance": 0.90,
                        "reasoning": "Classic presentation of AD with memory loss and visuospatial problems"
                    },
                    {
                        "content": "Vascular dementia: Stepwise cognitive decline often following strokes. Executive dysfunction prominent. MRI shows multiple infarcts or white matter disease. Risk factors include hypertension, diabetes, smoking. Prevention focuses on vascular risk reduction.",
                        "relevance": 0.60,
                        "reasoning": "Could cause cognitive decline but more stepwise progression typical"
                    },
                    {
                        "content": "Normal pressure hydrocephalus: Triad of cognitive impairment, gait disturbance, and urinary incontinence. MRI shows enlarged ventricles with normal CSF pressure. May improve with ventriculoperitoneal shunting.",
                        "relevance": 0.30,
                        "reasoning": "Could cause cognitive decline but missing gait and bladder symptoms"
                    },
                    {
                        "content": "Depression in elderly: Can present as pseudodementia with cognitive complaints, apathy, and functional decline. Usually more rapid onset. Responds to antidepressant therapy. Important to distinguish from true dementia.",
                        "relevance": 0.25,
                        "reasoning": "Could mimic dementia but 2-year progression suggests organic cause"
                    },
                    {
                        "content": "Medication-induced cognitive impairment: Anticholinergic medications, benzodiazepines, and others can cause cognitive decline in elderly. Usually reversible with medication discontinuation. Comprehensive medication review essential.",
                        "relevance": 0.15,
                        "reasoning": "Possible but doesn't explain visuospatial problems and personality changes"
                    }
                ]
            }
        ]
    
    def get_scenarios(self) -> List[Dict[str, Any]]:
        """Get all medical scenarios."""
        return self.medical_scenarios


class QuantumMedicalEvaluator:
    """Evaluates quantum reranking performance on medical scenarios."""
    
    def __init__(self):
        self.embedding_processor = EmbeddingProcessor(EmbeddingConfig())
        self.scenario_generator = MedicalScenarioGenerator()
        
        # Results storage
        self.results = {
            "quantum": {"ndcg": [], "map": [], "mrr": [], "latency": []},
            "classical": {"ndcg": [], "map": [], "mrr": [], "latency": []},
            "hybrid": {"ndcg": [], "map": [], "mrr": [], "latency": []}
        }
    
    def evaluate_all_scenarios(self):
        """Evaluate all medical scenarios with different reranking methods."""
        scenarios = self.scenario_generator.get_scenarios()
        
        logger.info("=" * 80)
        logger.info("REAL-WORLD MEDICAL QUANTUM RERANKING EVALUATION")
        logger.info("=" * 80)
        logger.info(f"Evaluating {len(scenarios)} medical scenarios")
        logger.info("Methods: Quantum, Classical, Hybrid")
        
        for i, scenario in enumerate(scenarios):
            logger.info(f"\n--- Scenario {i+1}/{len(scenarios)}: {scenario['id']} ---")
            logger.info(f"Type: {scenario['scenario_type']}")
            logger.info(f"Query: {scenario['query'][:100]}...")
            
            # Test each method
            for method in ["classical", "quantum", "hybrid"]:
                logger.info(f"\nTesting {method.upper()} method...")
                metrics = self._evaluate_scenario(scenario, method)
                
                # Store results
                self.results[method]["ndcg"].append(metrics["ndcg"])
                self.results[method]["map"].append(metrics["map"])
                self.results[method]["mrr"].append(metrics["mrr"])
                self.results[method]["latency"].append(metrics["latency"])
                
                logger.info(f"  NDCG@5: {metrics['ndcg']:.4f}")
                logger.info(f"  MAP: {metrics['map']:.4f}")
                logger.info(f"  MRR: {metrics['mrr']:.4f}")
                logger.info(f"  Latency: {metrics['latency']:.1f}ms")
        
        # Analyze results
        self._analyze_results()
    
    def _evaluate_scenario(self, scenario: Dict[str, Any], method: str) -> Dict[str, float]:
        """Evaluate a single scenario with a specific method."""
        # Create retriever
        config = RetrieverConfig(
            initial_k=len(scenario["documents"]),
            final_k=5,
            reranking_method=method,
            enable_caching=False  # Ensure fair comparison
        )
        
        retriever = TwoStageRetriever(config=config, embedding_processor=self.embedding_processor)
        
        # Add documents
        texts = [doc["content"] for doc in scenario["documents"]]
        metadatas = [{"relevance": doc["relevance"], "reasoning": doc["reasoning"]} 
                    for doc in scenario["documents"]]
        
        start_time = time.time()
        doc_ids = retriever.add_texts(texts, metadatas)
        
        # Perform retrieval
        retrieval_start = time.time()
        results = retriever.retrieve(scenario["query"], k=5)
        latency = (time.time() - retrieval_start) * 1000
        
        # Calculate metrics
        true_relevances = [doc["relevance"] for doc in scenario["documents"]]
        retrieved_relevances = []
        
        for result in results:
            # Find the relevance score by matching content
            for doc in scenario["documents"]:
                if doc["content"] == result.content:
                    retrieved_relevances.append(doc["relevance"])
                    break
            else:
                retrieved_relevances.append(0.0)  # Shouldn't happen
        
        # Pad with zeros if fewer than 5 results
        while len(retrieved_relevances) < 5:
            retrieved_relevances.append(0.0)
        
        # Calculate NDCG@5
        ideal_relevances = sorted(true_relevances, reverse=True)[:5]
        ndcg = self._calculate_ndcg(retrieved_relevances, ideal_relevances)
        
        # Calculate MAP
        map_score = self._calculate_map(retrieved_relevances)
        
        # Calculate MRR
        mrr = self._calculate_mrr(retrieved_relevances)
        
        return {
            "ndcg": ndcg,
            "map": map_score,
            "mrr": mrr,
            "latency": latency
        }
    
    def _calculate_ndcg(self, retrieved: List[float], ideal: List[float]) -> float:
        """Calculate NDCG@k."""
        def dcg(relevances):
            return sum([(2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevances)])
        
        dcg_score = dcg(retrieved)
        idcg_score = dcg(ideal)
        
        return dcg_score / idcg_score if idcg_score > 0 else 0
    
    def _calculate_map(self, relevances: List[float]) -> float:
        """Calculate Mean Average Precision."""
        relevant_count = 0
        precision_sum = 0
        
        for i, rel in enumerate(relevances):
            if rel >= 0.7:  # Consider highly relevant
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        
        return precision_sum / max(1, relevant_count)
    
    def _calculate_mrr(self, relevances: List[float]) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, rel in enumerate(relevances):
            if rel >= 0.7:  # Consider highly relevant
                return 1.0 / (i + 1)
        return 0.0
    
    def _analyze_results(self):
        """Analyze and display results."""
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 80)
        
        # Calculate averages
        for method in ["classical", "quantum", "hybrid"]:
            ndcg_avg = np.mean(self.results[method]["ndcg"])
            ndcg_std = np.std(self.results[method]["ndcg"])
            map_avg = np.mean(self.results[method]["map"])
            map_std = np.std(self.results[method]["map"])
            mrr_avg = np.mean(self.results[method]["mrr"])
            mrr_std = np.std(self.results[method]["mrr"])
            latency_avg = np.mean(self.results[method]["latency"])
            latency_std = np.std(self.results[method]["latency"])
            
            logger.info(f"\n{method.upper()} METHOD:")
            logger.info(f"  NDCG@5: {ndcg_avg:.4f} ± {ndcg_std:.4f}")
            logger.info(f"  MAP:     {map_avg:.4f} ± {map_std:.4f}")
            logger.info(f"  MRR:     {mrr_avg:.4f} ± {mrr_std:.4f}")
            logger.info(f"  Latency: {latency_avg:.1f} ± {latency_std:.1f} ms")
        
        # Compare methods
        logger.info("\n" + "=" * 80)
        logger.info("QUANTUM ADVANTAGE ANALYSIS")
        logger.info("=" * 80)
        
        quantum_ndcg = np.mean(self.results["quantum"]["ndcg"])
        classical_ndcg = np.mean(self.results["classical"]["ndcg"])
        hybrid_ndcg = np.mean(self.results["hybrid"]["ndcg"])
        
        quantum_improvement = ((quantum_ndcg - classical_ndcg) / classical_ndcg) * 100 if classical_ndcg > 0 else 0
        hybrid_improvement = ((hybrid_ndcg - classical_ndcg) / classical_ndcg) * 100 if classical_ndcg > 0 else 0
        
        logger.info(f"\nPerformance Improvements over Classical:")
        logger.info(f"  Quantum: {quantum_improvement:+.2f}%")
        logger.info(f"  Hybrid:  {hybrid_improvement:+.2f}%")
        
        # Statistical significance
        from scipy import stats
        if len(self.results["quantum"]["ndcg"]) > 1:
            t_stat, p_value = stats.ttest_rel(
                self.results["quantum"]["ndcg"],
                self.results["classical"]["ndcg"]
            )
            
            logger.info(f"\nStatistical Significance (Quantum vs Classical):")
            logger.info(f"  t-statistic: {t_stat:.4f}")
            logger.info(f"  p-value: {p_value:.6f}")
            logger.info(f"  Significant: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")
        
        # Scenario-specific analysis
        logger.info(f"\nScenario-Specific Results:")
        scenarios = self.scenario_generator.get_scenarios()
        
        for i, scenario in enumerate(scenarios):
            quantum_score = self.results["quantum"]["ndcg"][i]
            classical_score = self.results["classical"]["ndcg"][i]
            improvement = ((quantum_score - classical_score) / classical_score) * 100 if classical_score > 0 else 0
            
            logger.info(f"  {scenario['scenario_type']}: {improvement:+.1f}% (Q:{quantum_score:.3f} vs C:{classical_score:.3f})")
        
        # Final recommendation
        logger.info("\n" + "=" * 80)
        logger.info("RECOMMENDATION")
        logger.info("=" * 80)
        
        if quantum_improvement > 5 and (len(self.results["quantum"]["ndcg"]) <= 1 or p_value < 0.05):
            logger.info("✅ QUANTUM ADVANTAGE OBSERVED")
            logger.info(f"   Quantum reranking shows {quantum_improvement:.2f}% improvement")
            logger.info("   Recommended for medical document retrieval systems")
        elif quantum_improvement > 0:
            logger.info("⚡ MODEST QUANTUM IMPROVEMENT")
            logger.info(f"   Quantum reranking shows {quantum_improvement:.2f}% improvement")
            logger.info("   Consider for high-value medical applications")
        else:
            logger.info("📊 NO SIGNIFICANT QUANTUM ADVANTAGE")
            logger.info("   Classical methods perform as well or better")
            logger.info("   Continue algorithm development")
        
        # Save detailed results
        self._save_results()
    
    def _save_results(self):
        """Save detailed results to JSON file."""
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "scenarios": [s["id"] for s in self.scenario_generator.get_scenarios()],
            "metrics": self.results,
            "summary": {
                "quantum_avg_ndcg": np.mean(self.results["quantum"]["ndcg"]),
                "classical_avg_ndcg": np.mean(self.results["classical"]["ndcg"]),
                "hybrid_avg_ndcg": np.mean(self.results["hybrid"]["ndcg"]),
                "quantum_improvement_pct": ((np.mean(self.results["quantum"]["ndcg"]) - 
                                           np.mean(self.results["classical"]["ndcg"])) / 
                                          max(np.mean(self.results["classical"]["ndcg"]), 0.001)) * 100
            }
        }
        
        with open("medical_reranking_results.json", "w") as f:
            json.dump(results_data, f, indent=2)
        
        logger.info("\nDetailed results saved to: medical_reranking_results.json")


def main():
    """Run the real-world medical reranking evaluation."""
    logger.info("Starting Real-World Medical Quantum Reranking Evaluation")
    logger.info(f"Timestamp: {datetime.now()}")
    
    try:
        evaluator = QuantumMedicalEvaluator()
        evaluator.evaluate_all_scenarios()
        
        logger.info("\n✅ Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
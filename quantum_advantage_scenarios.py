#!/usr/bin/env python3
"""
Quantum Advantage Testing System

This system creates scenarios where quantum computing should provide genuine advantages:
1. Noisy, uncertain, ambiguous medical data
2. Multimodal data fusion with conflicts
3. High-uncertainty queries where classical confidence is low
4. Edge cases requiring probabilistic reasoning

The goal is to test quantum methods in their natural habitat, not against
classical methods in classical-friendly scenarios.
"""

import logging
import sys
import os
import time
import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import random

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from quantum_rerank.core.embeddings import EmbeddingProcessor, EmbeddingConfig
from quantum_rerank.retrieval.two_stage_retriever import TwoStageRetriever, RetrieverConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MultimodalData:
    """Represents multimodal medical data with potential conflicts."""
    text_symptoms: str
    lab_values: Dict[str, float]
    imaging_findings: str
    vitals: Dict[str, float]
    noise_level: float
    missing_data_pct: float
    conflicts: List[str]  # List of conflicting information


@dataclass
class QuantumScenario:
    """A scenario designed to test quantum advantages."""
    scenario_id: str
    query: str
    multimodal_data: MultimodalData
    scenario_type: str  # 'noisy', 'ambiguous', 'multimodal_conflict', 'uncertain'
    classical_expected_confidence: float  # How confident classical methods should be
    quantum_advantage_reason: str
    ground_truth_ranking: List[Tuple[str, float]]  # (doc_id, relevance)


class NoisyDataGenerator:
    """Generates noisy, uncertain medical scenarios."""
    
    def __init__(self):
        self.noise_types = [
            'measurement_error',
            'missing_values', 
            'conflicting_modalities',
            'temporal_uncertainty',
            'reporting_bias',
            'equipment_variance'
        ]
        
        self.medical_contexts = [
            'emergency_department',
            'icu_unstable',
            'remote_consultation',
            'pediatric_uncooperative',
            'elderly_polypharmacy',
            'psychiatric_comorbidity'
        ]
    
    def create_noisy_scenarios(self, num_scenarios: int = 10) -> List[QuantumScenario]:
        """Create scenarios with various types of noise and uncertainty."""
        scenarios = []
        
        # 1. High-noise emergency scenarios
        scenarios.extend(self._create_emergency_noise_scenarios(3))
        
        # 2. Multimodal conflict scenarios  
        scenarios.extend(self._create_multimodal_conflict_scenarios(3))
        
        # 3. Uncertainty and ambiguity scenarios
        scenarios.extend(self._create_uncertainty_scenarios(2))
        
        # 4. Missing data scenarios
        scenarios.extend(self._create_missing_data_scenarios(2))
        
        return scenarios[:num_scenarios]
    
    def _create_emergency_noise_scenarios(self, count: int) -> List[QuantumScenario]:
        """Create emergency scenarios with high noise and time pressure."""
        scenarios = []
        
        base_scenarios = [
            {
                "context": "Trauma patient, multiple injuries, limited history",
                "symptoms": "altered consciousness, hypotension, abdominal distension",
                "conflicts": ["patient unable to provide history", "vitals unstable", "multiple potential injury sites"],
                "quantum_advantage": "Superposition can represent multiple simultaneous injury possibilities"
            },
            {
                "context": "Pediatric fever, crying, limited cooperation",
                "symptoms": "high fever, irritability, possible neck stiffness",
                "conflicts": ["child uncooperative for exam", "fever masking other symptoms", "parent anxiety affecting history"],
                "quantum_advantage": "Quantum uncertainty naturally models diagnostic ambiguity with limited data"
            },
            {
                "context": "Psychiatric patient with medical emergency",
                "symptoms": "chest pain, agitation, possible drug ingestion",
                "conflicts": ["unreliable history", "psychiatric vs medical emergency", "drug interaction effects"],
                "quantum_advantage": "Quantum interference can separate signal from noise in conflicting presentations"
            }
        ]
        
        for i, scenario in enumerate(base_scenarios[:count]):
            # Add noise to lab values
            noisy_labs = self._add_lab_noise({
                "troponin": random.uniform(0.01, 0.5),
                "white_count": random.uniform(4.5, 15.0),
                "creatinine": random.uniform(0.8, 2.0),
                "lactate": random.uniform(1.0, 4.0)
            }, noise_level=0.3)
            
            # Add noise to vitals
            noisy_vitals = self._add_vital_noise({
                "heart_rate": random.uniform(60, 120),
                "blood_pressure_sys": random.uniform(90, 160),
                "respiratory_rate": random.uniform(12, 24),
                "temperature": random.uniform(36.0, 39.0)
            }, noise_level=0.2)
            
            multimodal_data = MultimodalData(
                text_symptoms=scenario["symptoms"],
                lab_values=noisy_labs,
                imaging_findings=f"Limited imaging due to {scenario['context']}. Possible abnormalities but poor quality.",
                vitals=noisy_vitals,
                noise_level=0.4,
                missing_data_pct=0.3,
                conflicts=scenario["conflicts"]
            )
            
            scenarios.append(QuantumScenario(
                scenario_id=f"emergency_noise_{i+1}",
                query=f"Emergency: {scenario['context']}. {scenario['symptoms']}. Rapid diagnosis needed with limited/noisy data.",
                multimodal_data=multimodal_data,
                scenario_type="noisy_emergency",
                classical_expected_confidence=0.3,  # Low confidence due to noise
                quantum_advantage_reason=scenario["quantum_advantage"],
                ground_truth_ranking=self._generate_noisy_documents(scenario["symptoms"])
            ))
        
        return scenarios
    
    def _create_multimodal_conflict_scenarios(self, count: int) -> List[QuantumScenario]:
        """Create scenarios where different modalities give conflicting information."""
        scenarios = []
        
        conflict_cases = [
            {
                "text": "Patient reports severe chest pain, appears comfortable",
                "labs": {"troponin": 0.01, "d_dimer": 150},  # Normal
                "imaging": "ECG shows ST elevation in leads II, III, aVF",  # Abnormal!
                "vitals": {"heart_rate": 70, "bp_sys": 120},  # Normal
                "conflict": "Text and vitals suggest low acuity, but ECG shows STEMI",
                "quantum_advantage": "Quantum entanglement can model complex text-ECG-lab correlations"
            },
            {
                "text": "Elderly patient with mild dyspnea and leg swelling",
                "labs": {"bnp": 1500, "creatinine": 2.1},  # Heart failure markers high
                "imaging": "Chest X-ray shows clear lungs, normal heart size",  # Conflicting!
                "vitals": {"respiratory_rate": 22, "o2_sat": 95},  # Borderline
                "conflict": "Labs suggest heart failure but imaging normal",
                "quantum_advantage": "Quantum superposition represents multiple disease states simultaneously"
            },
            {
                "text": "Young athlete with fatigue and palpitations",
                "labs": {"tsh": 2.1, "sodium": 140, "potassium": 4.0},  # Normal values
                "imaging": "Echocardiogram shows hypertrophic cardiomyopathy",  # Serious finding!
                "vitals": {"heart_rate": 45, "bp_sys": 110},  # Bradycardia
                "conflict": "Benign symptoms but serious structural heart disease",
                "quantum_advantage": "Quantum interference resolves seemingly contradictory clinical data"
            }
        ]
        
        for i, case in enumerate(conflict_cases[:count]):
            # Intentionally add conflicting noise
            noisy_labs = self._add_conflicting_noise(case["labs"])
            noisy_vitals = self._add_conflicting_noise(case["vitals"])
            
            multimodal_data = MultimodalData(
                text_symptoms=case["text"],
                lab_values=noisy_labs,
                imaging_findings=case["imaging"],
                vitals=noisy_vitals,
                noise_level=0.25,
                missing_data_pct=0.15,
                conflicts=[case["conflict"]]
            )
            
            scenarios.append(QuantumScenario(
                scenario_id=f"multimodal_conflict_{i+1}",
                query=f"Complex case with conflicting data: {case['text']}. Resolve discrepancies between modalities.",
                multimodal_data=multimodal_data,
                scenario_type="multimodal_conflict",
                classical_expected_confidence=0.4,  # Confused by conflicts
                quantum_advantage_reason=case["quantum_advantage"],
                ground_truth_ranking=self._generate_conflict_documents(case["text"], case["imaging"])
            ))
        
        return scenarios
    
    def _create_uncertainty_scenarios(self, count: int) -> List[QuantumScenario]:
        """Create scenarios with inherent diagnostic uncertainty."""
        uncertainty_cases = [
            {
                "symptoms": "Intermittent chest pain, stress-related, family history unclear",
                "uncertainty_factors": ["symptom timing variable", "stress vs organic", "incomplete family history"],
                "quantum_advantage": "Quantum probability amplitudes naturally represent diagnostic uncertainty"
            },
            {
                "symptoms": "Cognitive changes in elderly, depression vs dementia vs delirium",
                "uncertainty_factors": ["overlapping symptoms", "multiple possible causes", "progression unclear"],
                "quantum_advantage": "Quantum superposition models multiple overlapping cognitive diagnoses"
            }
        ]
        
        scenarios = []
        for i, case in enumerate(uncertainty_cases[:count]):
            # High uncertainty = high missing data + noise
            uncertain_labs = self._add_uncertainty_noise({
                "marker1": random.uniform(0.5, 1.5),
                "marker2": random.uniform(10, 50),
                "marker3": None  # Missing data
            })
            
            multimodal_data = MultimodalData(
                text_symptoms=case["symptoms"],
                lab_values=uncertain_labs,
                imaging_findings="Non-specific findings, requires clinical correlation",
                vitals={"heart_rate": random.uniform(60, 100)},
                noise_level=0.5,
                missing_data_pct=0.4,
                conflicts=case["uncertainty_factors"]
            )
            
            scenarios.append(QuantumScenario(
                scenario_id=f"uncertainty_{i+1}",
                query=f"Uncertain diagnosis: {case['symptoms']}. Multiple possibilities, limited definitive data.",
                multimodal_data=multimodal_data,
                scenario_type="uncertain",
                classical_expected_confidence=0.25,  # Very low confidence
                quantum_advantage_reason=case["quantum_advantage"],
                ground_truth_ranking=self._generate_uncertain_documents(case["symptoms"])
            ))
        
        return scenarios
    
    def _create_missing_data_scenarios(self, count: int) -> List[QuantumScenario]:
        """Create scenarios with significant missing data."""
        missing_cases = [
            {
                "available": "Chest pain, age 45, male",
                "missing": ["family history", "medications", "prior tests", "timing details"],
                "quantum_advantage": "Quantum inference can probabilistically fill missing information gaps"
            },
            {
                "available": "Altered mental status, found down",
                "missing": ["timeline", "medications", "medical history", "substance use"],
                "quantum_advantage": "Quantum exploration of multiple possible histories simultaneously"
            }
        ]
        
        scenarios = []
        for i, case in enumerate(missing_cases[:count]):
            # Extreme missing data
            sparse_data = MultimodalData(
                text_symptoms=case["available"],
                lab_values={"available_lab": random.uniform(0.5, 2.0)},  # Only one lab
                imaging_findings="Limited study, patient condition",
                vitals={"heart_rate": random.uniform(70, 110)},  # Only one vital
                noise_level=0.2,
                missing_data_pct=0.7,  # 70% missing!
                conflicts=[f"Missing: {', '.join(case['missing'])}"]
            )
            
            scenarios.append(QuantumScenario(
                scenario_id=f"missing_data_{i+1}",
                query=f"Limited information case: {case['available']}. Significant missing data: {', '.join(case['missing'])}.",
                multimodal_data=sparse_data,
                scenario_type="missing_data",
                classical_expected_confidence=0.2,  # Very low due to missing data
                quantum_advantage_reason=case["quantum_advantage"],
                ground_truth_ranking=self._generate_sparse_documents(case["available"])
            ))
        
        return scenarios
    
    def _add_lab_noise(self, labs: Dict[str, float], noise_level: float) -> Dict[str, float]:
        """Add realistic lab measurement noise."""
        noisy_labs = {}
        for name, value in labs.items():
            if random.random() < 0.9:  # 90% chance value exists
                noise = np.random.normal(0, noise_level * value)
                noisy_labs[name] = max(0, value + noise)  # Don't go negative
            # else: missing value
        return noisy_labs
    
    def _add_vital_noise(self, vitals: Dict[str, float], noise_level: float) -> Dict[str, float]:
        """Add realistic vital sign noise."""
        noisy_vitals = {}
        for name, value in vitals.items():
            if random.random() < 0.95:  # 95% chance vital exists
                noise = np.random.normal(0, noise_level * value)
                noisy_vitals[name] = max(0, value + noise)
        return noisy_vitals
    
    def _add_conflicting_noise(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Add noise that creates conflicts between modalities."""
        conflicting_data = {}
        for name, value in data.items():
            if isinstance(value, (int, float)):
                # Intentionally add bias in one direction
                bias = random.choice([-0.3, 0.3]) * float(value)
                noise = np.random.normal(bias, 0.1 * float(value))
                conflicting_data[name] = max(0, float(value) + noise)
            else:
                # For non-numeric values, just pass through
                conflicting_data[name] = float(hash(str(value)) % 100)  # Convert to numeric
        return conflicting_data
    
    def _add_uncertainty_noise(self, data: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        """Add high uncertainty and missing values."""
        uncertain_data = {}
        for name, value in data.items():
            if value is None or random.random() < 0.3:  # 30% chance missing
                uncertain_data[name] = None
            else:
                # High variance noise
                noise = np.random.normal(0, 0.5 * value)
                uncertain_data[name] = max(0, value + noise)
        return uncertain_data
    
    def _generate_noisy_documents(self, symptoms: str) -> List[Tuple[str, float]]:
        """Generate documents for noisy emergency scenarios."""
        # In noisy scenarios, even relevant docs have uncertain relevance
        return [
            ("emergency_protocol_chest_pain", 0.8 + random.uniform(-0.2, 0.1)),
            ("trauma_evaluation_guidelines", 0.7 + random.uniform(-0.3, 0.2)),
            ("pediatric_emergency_assessment", 0.6 + random.uniform(-0.2, 0.3)),
            ("psychiatric_medical_screening", 0.5 + random.uniform(-0.2, 0.4)),
            ("general_emergency_triage", 0.3 + random.uniform(-0.1, 0.3))
        ]
    
    def _generate_conflict_documents(self, text: str, imaging: str) -> List[Tuple[str, float]]:
        """Generate documents for conflicting multimodal scenarios."""
        return [
            ("multimodal_discrepancy_resolution", 0.9 + random.uniform(-0.1, 0.05)),
            ("imaging_clinical_correlation", 0.8 + random.uniform(-0.2, 0.1)),
            ("false_positive_imaging_findings", 0.7 + random.uniform(-0.3, 0.2)),
            ("atypical_presentations", 0.6 + random.uniform(-0.2, 0.3)),
            ("standard_diagnostic_guidelines", 0.4 + random.uniform(-0.2, 0.2))
        ]
    
    def _generate_uncertain_documents(self, symptoms: str) -> List[Tuple[str, float]]:
        """Generate documents for high uncertainty scenarios."""
        return [
            ("diagnostic_uncertainty_management", 0.85 + random.uniform(-0.15, 0.1)),
            ("probability_based_diagnosis", 0.75 + random.uniform(-0.25, 0.2)),
            ("differential_diagnosis_approach", 0.65 + random.uniform(-0.2, 0.25)),
            ("clinical_decision_making_uncertainty", 0.55 + random.uniform(-0.2, 0.3)),
            ("standard_workup_protocols", 0.35 + random.uniform(-0.15, 0.25))
        ]
    
    def _generate_sparse_documents(self, available: str) -> List[Tuple[str, float]]:
        """Generate documents for missing data scenarios."""
        return [
            ("limited_information_diagnosis", 0.8 + random.uniform(-0.2, 0.15)),
            ("clinical_reasoning_incomplete_data", 0.7 + random.uniform(-0.3, 0.2)),
            ("emergency_unknown_patient", 0.6 + random.uniform(-0.2, 0.3)),
            ("diagnostic_approach_missing_history", 0.5 + random.uniform(-0.2, 0.4)),
            ("general_medical_assessment", 0.3 + random.uniform(-0.1, 0.3))
        ]


class ClassicalConfidenceEstimator:
    """Estimates classical method confidence to identify when to use quantum."""
    
    def __init__(self, embedding_processor: EmbeddingProcessor):
        self.embedding_processor = embedding_processor
    
    def estimate_confidence(self, query: str, documents: List[str]) -> float:
        """Estimate how confident classical methods would be."""
        
        # Get embeddings
        query_emb = self.embedding_processor.encode_texts([query])[0]
        doc_embs = self.embedding_processor.encode_texts(documents)
        
        # Calculate similarities
        similarities = [np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)) 
                       for doc_emb in doc_embs]
        
        # Confidence metrics
        max_sim = max(similarities)
        sim_spread = max(similarities) - min(similarities)
        sim_std = np.std(similarities)
        
        # High confidence if: high max similarity, good spread, low std
        confidence = (max_sim * 0.4) + (sim_spread * 0.3) + (1.0 - min(sim_std * 2, 1.0)) * 0.3
        
        return min(confidence, 1.0)


class HybridQuantumRouter:
    """Routes queries to classical vs quantum based on scenario characteristics."""
    
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
        self.routing_stats = {
            "total_queries": 0,
            "routed_to_quantum": 0,
            "routed_to_classical": 0,
            "routing_accuracy": []
        }
    
    def should_use_quantum(self, scenario: QuantumScenario, classical_confidence: float) -> bool:
        """Decide whether to use quantum based on scenario characteristics."""
        
        self.routing_stats["total_queries"] += 1
        
        # Use quantum if:
        # 1. Classical confidence is low
        # 2. High noise/uncertainty
        # 3. Multimodal conflicts
        # 4. Missing data
        
        quantum_score = 0
        
        # Classical confidence factor
        if classical_confidence < self.confidence_threshold:
            quantum_score += 0.3
        
        # Noise and uncertainty factors
        quantum_score += scenario.multimodal_data.noise_level * 0.25
        quantum_score += scenario.multimodal_data.missing_data_pct * 0.25
        
        # Scenario type bonuses
        if scenario.scenario_type in ["multimodal_conflict", "uncertain"]:
            quantum_score += 0.3
        elif scenario.scenario_type in ["noisy_emergency", "missing_data"]:
            quantum_score += 0.2
        
        # Conflict count
        quantum_score += len(scenario.multimodal_data.conflicts) * 0.05
        
        use_quantum = quantum_score > 0.5
        
        if use_quantum:
            self.routing_stats["routed_to_quantum"] += 1
        else:
            self.routing_stats["routed_to_classical"] += 1
        
        return use_quantum
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        total = self.routing_stats["total_queries"]
        if total > 0:
            return {
                **self.routing_stats,
                "quantum_routing_rate": self.routing_stats["routed_to_quantum"] / total,
                "classical_routing_rate": self.routing_stats["routed_to_classical"] / total
            }
        return self.routing_stats


def main():
    """Run quantum advantage testing with noisy, uncertain scenarios."""
    logger.info("=" * 80)
    logger.info("QUANTUM ADVANTAGE TESTING: NOISY & UNCERTAIN SCENARIOS")
    logger.info("=" * 80)
    
    # Initialize components
    embedding_processor = EmbeddingProcessor(EmbeddingConfig())
    confidence_estimator = ClassicalConfidenceEstimator(embedding_processor)
    router = HybridQuantumRouter(confidence_threshold=0.6)
    data_generator = NoisyDataGenerator()
    
    # Generate challenging scenarios
    logger.info("Generating noisy, uncertain medical scenarios...")
    scenarios = data_generator.create_noisy_scenarios(8)
    
    logger.info(f"Created {len(scenarios)} quantum advantage scenarios:")
    for scenario in scenarios:
        logger.info(f"  - {scenario.scenario_id}: {scenario.scenario_type}")
        logger.info(f"    Noise: {scenario.multimodal_data.noise_level:.2f}, Missing: {scenario.multimodal_data.missing_data_pct:.2f}")
        logger.info(f"    Expected classical confidence: {scenario.classical_expected_confidence:.2f}")
    
    # Test routing decisions
    logger.info("\n" + "=" * 60)
    logger.info("HYBRID ROUTING ANALYSIS")
    logger.info("=" * 60)
    
    routing_decisions = []
    for scenario in scenarios:
        # Generate sample documents for this scenario
        doc_contents = [
            f"Medical document about {scenario.query[:50]}...",
            f"Clinical guidelines for {scenario.scenario_type} cases",
            f"Research on {scenario.multimodal_data.text_symptoms}",
            "General medical reference material",
            "Standard diagnostic protocols"
        ]
        
        # Estimate classical confidence
        classical_conf = confidence_estimator.estimate_confidence(scenario.query, doc_contents)
        
        # Router decision
        use_quantum = router.should_use_quantum(scenario, classical_conf)
        
        routing_decisions.append({
            "scenario_id": scenario.scenario_id,
            "scenario_type": scenario.scenario_type,
            "classical_confidence": classical_conf,
            "expected_confidence": scenario.classical_expected_confidence,
            "routed_to": "quantum" if use_quantum else "classical",
            "noise_level": scenario.multimodal_data.noise_level,
            "missing_data": scenario.multimodal_data.missing_data_pct,
            "conflicts": len(scenario.multimodal_data.conflicts)
        })
        
        logger.info(f"\n{scenario.scenario_id}:")
        logger.info(f"  Type: {scenario.scenario_type}")
        logger.info(f"  Classical confidence: {classical_conf:.3f} (expected: {scenario.classical_expected_confidence:.3f})")
        logger.info(f"  Noise: {scenario.multimodal_data.noise_level:.2f}, Missing: {scenario.multimodal_data.missing_data_pct:.2f}")
        logger.info(f"  Conflicts: {len(scenario.multimodal_data.conflicts)}")
        logger.info(f"  → Routed to: {'QUANTUM' if use_quantum else 'CLASSICAL'}")
        logger.info(f"  Quantum advantage: {scenario.quantum_advantage_reason}")
    
    # Routing statistics
    routing_stats = router.get_routing_stats()
    logger.info("\n" + "=" * 60)
    logger.info("ROUTING STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total queries: {routing_stats['total_queries']}")
    logger.info(f"Routed to quantum: {routing_stats['routed_to_quantum']} ({routing_stats['quantum_routing_rate']:.1%})")
    logger.info(f"Routed to classical: {routing_stats['routed_to_classical']} ({routing_stats['classical_routing_rate']:.1%})")
    
    # Analysis by scenario type
    logger.info("\n" + "=" * 60)
    logger.info("ROUTING BY SCENARIO TYPE")
    logger.info("=" * 60)
    
    scenario_routing = {}
    for decision in routing_decisions:
        stype = decision["scenario_type"]
        if stype not in scenario_routing:
            scenario_routing[stype] = {"quantum": 0, "classical": 0, "total": 0}
        
        scenario_routing[stype][decision["routed_to"]] += 1
        scenario_routing[stype]["total"] += 1
    
    for stype, stats in scenario_routing.items():
        quantum_rate = stats["quantum"] / stats["total"]
        logger.info(f"{stype}: {quantum_rate:.1%} → quantum ({stats['quantum']}/{stats['total']})")
    
    # Save detailed results
    results = {
        "timestamp": datetime.now().isoformat(),
        "scenarios": [
            {
                "scenario_id": s.scenario_id,
                "scenario_type": s.scenario_type,
                "query": s.query,
                "noise_level": s.multimodal_data.noise_level,
                "missing_data_pct": s.multimodal_data.missing_data_pct,
                "conflicts": s.multimodal_data.conflicts,
                "expected_classical_confidence": s.classical_expected_confidence,
                "quantum_advantage_reason": s.quantum_advantage_reason
            }
            for s in scenarios
        ],
        "routing_decisions": routing_decisions,
        "routing_stats": routing_stats,
        "scenario_routing": scenario_routing
    }
    
    with open("quantum_advantage_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nDetailed analysis saved to: quantum_advantage_analysis.json")
    
    # Recommendations
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATIONS FOR QUANTUM ADVANTAGE")
    logger.info("=" * 80)
    
    logger.info("\n🎯 OPTIMAL QUANTUM USE CASES:")
    high_quantum_scenarios = [d for d in routing_decisions if d["routed_to"] == "quantum"]
    for decision in high_quantum_scenarios:
        logger.info(f"  ✓ {decision['scenario_id']}: {decision['scenario_type']}")
        logger.info(f"    - Low classical confidence: {decision['classical_confidence']:.3f}")
        logger.info(f"    - High noise/uncertainty: {decision['noise_level']:.2f}/{decision['missing_data']:.2f}")
    
    logger.info("\n🔧 SYSTEM DESIGN RECOMMENDATIONS:")
    logger.info("  1. Use hybrid routing with confidence threshold ~0.6")
    logger.info("  2. Route to quantum for: noise >0.3, missing >0.2, conflicts >2")
    logger.info("  3. Focus quantum training on uncertainty representation")
    logger.info("  4. Design quantum circuits for multimodal fusion")
    logger.info("  5. Test quantum advantage in degraded/noisy conditions")
    
    logger.info("\n✅ Next steps: Implement quantum methods optimized for these scenarios!")


if __name__ == "__main__":
    main()
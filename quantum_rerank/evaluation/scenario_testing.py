"""
Scenario-Specific Testing Framework for Quantum Reranking.

This module implements specialized testing scenarios where quantum methods
should demonstrate advantages over classical approaches.

Based on:
- Quantum noise tolerance research
- Complex query processing studies  
- Selective quantum usage optimization
"""

import logging
import random
import re
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path

from .ir_metrics import QueryResult, RetrievalResult, IRMetricsCalculator
from .medical_relevance import MedicalQuery, MedicalDocument, MedicalRelevanceJudgments
from ..retrieval.two_stage_retriever import TwoStageRetriever, RetrieverConfig
from ..core.embeddings import EmbeddingProcessor

logger = logging.getLogger(__name__)


@dataclass
class NoiseConfig:
    """Configuration for noise injection."""
    noise_type: str  # ocr, abbreviation, typo, mixed
    noise_level: float  # 0.0 - 1.0
    target_chars: Optional[str] = None  # Specific characters to target
    preserve_medical_terms: bool = True  # Don't corrupt critical medical terms


@dataclass
class ScenarioResult:
    """Result from a scenario test."""
    scenario_name: str
    config: Dict[str, Any]
    baseline_metrics: Dict[str, float]
    quantum_metrics: Dict[str, float]
    performance_improvement: Dict[str, float]
    statistical_significance: Dict[str, Any]
    execution_time_ms: float
    metadata: Dict[str, Any]


class NoiseInjector:
    """Injects various types of noise into text documents."""
    
    def __init__(self):
        """Initialize noise injector."""
        # OCR error patterns (common substitutions)
        self.ocr_substitutions = {
            'a': ['o', 'e'], 'e': ['c', 'o'], 'i': ['l', '1'], 'o': ['0', 'a'],
            'u': ['v', 'n'], 'n': ['m', 'h'], 'm': ['rn', 'n'], 'cl': ['d'],
            'rn': ['m'], 'vv': ['w'], 'nn': ['m'], 'il': ['ll'], 'li': ['h']
        }
        
        # Medical abbreviation corruptions
        self.medical_corruptions = {
            'MI': ['Ml', 'M1', 'M!'], 'HTN': ['HT N', 'HTN.', 'H TN'],
            'DM': ['D M', 'OM', 'DM.'], 'COPD': ['C OPD', 'CO PD', 'CDPD'],
            'mg': ['mg.', 'rng', 'rag'], 'ml': ['ml.', 'rnl', 'mI']
        }
        
        # Common typos
        self.typo_patterns = {
            'the': ['teh', 'hte'], 'and': ['adn', 'nad'], 'patient': ['patinet', 'pateint'],
            'treatment': ['treatmetn', 'treatmnet'], 'diagnosis': ['diagnsois', 'diagnsis']
        }
        
        # Medical terms to preserve (don't corrupt these critical terms)
        self.protected_terms = {
            'heart', 'brain', 'liver', 'kidney', 'lung', 'cancer', 'tumor',
            'diabetes', 'insulin', 'blood', 'surgery', 'emergency', 'critical'
        }
    
    def inject_ocr_noise(self, text: str, noise_level: float) -> str:
        """Inject OCR-like errors into text."""
        words = text.split()
        modified_words = []
        
        for word in words:
            if random.random() < noise_level and word.lower() not in self.protected_terms:
                # Apply OCR substitutions
                modified_word = word
                for original, substitutes in self.ocr_substitutions.items():
                    if original in modified_word.lower():
                        substitute = random.choice(substitutes)
                        modified_word = modified_word.replace(original, substitute, 1)
                        break
                modified_words.append(modified_word)
            else:
                modified_words.append(word)
        
        return ' '.join(modified_words)
    
    def inject_abbreviation_noise(self, text: str, noise_level: float) -> str:
        """Inject medical abbreviation corruptions."""
        modified_text = text
        
        for abbrev, corruptions in self.medical_corruptions.items():
            if abbrev in text and random.random() < noise_level:
                corruption = random.choice(corruptions)
                modified_text = modified_text.replace(abbrev, corruption, 1)
        
        return modified_text
    
    def inject_typo_noise(self, text: str, noise_level: float) -> str:
        """Inject common typing errors."""
        words = text.split()
        modified_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in self.typo_patterns and random.random() < noise_level:
                typo = random.choice(self.typo_patterns[word_lower])
                # Preserve original capitalization
                if word[0].isupper():
                    typo = typo.capitalize()
                modified_words.append(typo)
            else:
                modified_words.append(word)
        
        return ' '.join(modified_words)
    
    def inject_mixed_noise(self, text: str, noise_level: float) -> str:
        """Inject multiple types of noise."""
        # Apply each noise type with reduced probability
        individual_noise_level = noise_level / 3
        
        # Apply in sequence
        noisy_text = self.inject_ocr_noise(text, individual_noise_level)
        noisy_text = self.inject_abbreviation_noise(noisy_text, individual_noise_level)
        noisy_text = self.inject_typo_noise(noisy_text, individual_noise_level)
        
        return noisy_text
    
    def inject_noise(self, text: str, config: NoiseConfig) -> str:
        """Inject noise according to configuration."""
        if config.noise_type == 'ocr':
            return self.inject_ocr_noise(text, config.noise_level)
        elif config.noise_type == 'abbreviation':
            return self.inject_abbreviation_noise(text, config.noise_level)
        elif config.noise_type == 'typo':
            return self.inject_typo_noise(text, config.noise_level)
        elif config.noise_type == 'mixed':
            return self.inject_mixed_noise(text, config.noise_level)
        else:
            raise ValueError(f"Unknown noise type: {config.noise_type}")


class ComplexQueryGenerator:
    """Generates complex medical queries for testing."""
    
    def __init__(self):
        """Initialize complex query generator."""
        # Multi-domain query templates
        self.multi_domain_templates = [
            "Management of {condition1} in patients with {condition2}",
            "Treatment of {condition1} and {condition2} comorbidity",
            "{condition1} with {condition2} complications",
            "Diagnosis of {condition1} versus {condition2}",
            "Risk factors for {condition1} in {condition2} patients"
        ]
        
        # Domain-specific conditions
        self.domain_conditions = {
            'cardiology': ['heart failure', 'myocardial infarction', 'hypertension', 'arrhythmia'],
            'diabetes': ['type 2 diabetes', 'diabetic neuropathy', 'hyperglycemia', 'insulin resistance'], 
            'respiratory': ['COPD', 'asthma', 'pneumonia', 'pulmonary embolism'],
            'neurology': ['stroke', 'alzheimer disease', 'parkinson disease', 'epilepsy'],
            'oncology': ['lung cancer', 'breast cancer', 'lymphoma', 'leukemia']
        }
        
        # Ambiguous medical terms
        self.ambiguous_terms = {
            'MI': ['myocardial infarction', 'mitral insufficiency', 'medical informatics'],
            'MS': ['multiple sclerosis', 'mitral stenosis', 'medical student'],
            'PE': ['pulmonary embolism', 'physical examination', 'pleural effusion'],
            'CA': ['cancer', 'cardiac arrest', 'coronary artery'],
            'AS': ['aortic stenosis', 'ankylosing spondylitis', 'aspirin']
        }
    
    def generate_multi_domain_query(self, domains: List[str]) -> str:
        """Generate query spanning multiple medical domains."""
        if len(domains) < 2:
            raise ValueError("Need at least 2 domains for multi-domain query")
        
        domain1, domain2 = random.sample(domains, 2)
        condition1 = random.choice(self.domain_conditions[domain1])
        condition2 = random.choice(self.domain_conditions[domain2])
        
        template = random.choice(self.multi_domain_templates)
        query = template.format(condition1=condition1, condition2=condition2)
        
        return query
    
    def generate_ambiguous_query(self) -> str:
        """Generate query with ambiguous medical terms."""
        ambiguous_term = random.choice(list(self.ambiguous_terms.keys()))
        meanings = self.ambiguous_terms[ambiguous_term]
        
        # Create query that could apply to multiple meanings
        query_templates = [
            f"Diagnosis and treatment of {ambiguous_term}",
            f"Management guidelines for {ambiguous_term}",
            f"Risk factors associated with {ambiguous_term}",
            f"Prognosis of patients with {ambiguous_term}",
            f"Complications of {ambiguous_term}"
        ]
        
        return random.choice(query_templates)
    
    def generate_long_complex_query(self) -> str:
        """Generate long, complex clinical scenario query."""
        scenarios = [
            "A 65-year-old male patient with a history of type 2 diabetes mellitus, hypertension, and coronary artery disease presents with acute chest pain, shortness of breath, and elevated troponin levels. What is the differential diagnosis and appropriate management strategy?",
            
            "A 45-year-old female with rheumatoid arthritis on methotrexate therapy develops fever, productive cough, and bilateral pulmonary infiltrates on chest X-ray. Discuss the approach to diagnosis and treatment considering her immunocompromised status.",
            
            "An 80-year-old patient with chronic kidney disease, heart failure, and atrial fibrillation requires anticoagulation therapy. What are the considerations for drug selection, dosing, and monitoring in this complex case?",
            
            "A pediatric patient presents with recurrent respiratory infections, failure to thrive, and steatorrhea. The family history is significant for cystic fibrosis. Outline the diagnostic workup and management approach."
        ]
        
        return random.choice(scenarios)


class QuantumAdvantageScenarios:
    """
    Testing framework for scenarios where quantum methods should excel.
    
    Tests quantum advantages in:
    1. Noise tolerance
    2. Complex query processing
    3. Fine-grained ranking
    4. Selective usage optimization
    """
    
    def __init__(self, retriever: TwoStageRetriever, 
                 medical_relevance: MedicalRelevanceJudgments):
        """Initialize scenario testing framework."""
        self.retriever = retriever
        self.medical_relevance = medical_relevance
        self.noise_injector = NoiseInjector()
        self.complex_query_generator = ComplexQueryGenerator()
        self.metrics_calculator = IRMetricsCalculator()
        
        logger.info("Quantum advantage scenario testing initialized")
    
    def test_noise_tolerance(self, queries: List[MedicalQuery],
                           documents: List[MedicalDocument],
                           noise_configs: List[NoiseConfig]) -> List[ScenarioResult]:
        """
        Test quantum vs classical tolerance to document noise.
        
        Args:
            queries: Test queries
            documents: Clean documents
            noise_configs: Different noise configurations to test
            
        Returns:
            List of scenario results for each noise configuration
        """
        logger.info(f"Testing noise tolerance with {len(noise_configs)} configurations")
        results = []
        
        for noise_config in noise_configs:
            logger.info(f"Testing {noise_config.noise_type} noise at {noise_config.noise_level} level")
            start_time = time.time()
            
            # Create noisy versions of documents
            noisy_documents = []
            for doc in documents:
                noisy_title = self.noise_injector.inject_noise(doc.title, noise_config)
                noisy_abstract = self.noise_injector.inject_noise(doc.abstract, noise_config)
                noisy_full_text = self.noise_injector.inject_noise(doc.full_text, noise_config)
                
                noisy_doc = MedicalDocument(
                    doc_id=doc.doc_id,
                    title=noisy_title,
                    abstract=noisy_abstract,
                    full_text=noisy_full_text,
                    medical_domain=doc.medical_domain,
                    key_terms=doc.key_terms,
                    sections=doc.sections
                )
                noisy_documents.append(noisy_doc)
            
            # Clear retriever and add noisy documents
            self.retriever.clear()
            self._add_medical_documents_to_retriever(noisy_documents)
            
            # Test both classical and quantum methods
            classical_results = self._run_retrieval_test(queries, method="classical")
            quantum_results = self._run_retrieval_test(queries, method="quantum") 
            
            # Create relevance judgments
            relevance_judgments = self.medical_relevance.create_relevance_judgments(queries, noisy_documents)
            self.metrics_calculator.add_relevance_judgments(relevance_judgments)
            
            # Calculate metrics
            classical_metrics = self.metrics_calculator.evaluate_method(classical_results)
            quantum_metrics = self.metrics_calculator.evaluate_method(quantum_results)
            
            # Calculate improvement
            improvements = self._calculate_improvements(classical_metrics, quantum_metrics)
            
            # Statistical significance test
            sig_test = self.metrics_calculator.statistical_significance_test(
                classical_results, quantum_results, "ndcg_10"
            )
            
            elapsed_time = (time.time() - start_time) * 1000
            
            result = ScenarioResult(
                scenario_name=f"noise_tolerance_{noise_config.noise_type}",
                config={
                    'noise_type': noise_config.noise_type,
                    'noise_level': noise_config.noise_level
                },
                baseline_metrics=self._metrics_to_dict(classical_metrics),
                quantum_metrics=self._metrics_to_dict(quantum_metrics),
                performance_improvement=improvements,
                statistical_significance=sig_test,
                execution_time_ms=elapsed_time,
                metadata={
                    'documents_processed': len(noisy_documents),
                    'queries_tested': len(queries)
                }
            )
            
            results.append(result)
            
            logger.info(f"Noise tolerance test completed: "
                       f"NDCG@10 improvement: {improvements.get('ndcg_10', 0):.3f}")
        
        return results
    
    def test_complex_queries(self, base_queries: List[MedicalQuery],
                           documents: List[MedicalDocument],
                           complexity_types: List[str] = None) -> List[ScenarioResult]:
        """
        Test quantum vs classical on complex queries.
        
        Args:
            base_queries: Base queries for domain context
            documents: Medical documents
            complexity_types: Types of complexity to test
            
        Returns:
            List of scenario results for each complexity type
        """
        if complexity_types is None:
            complexity_types = ['multi_domain', 'ambiguous', 'long_clinical']
        
        logger.info(f"Testing complex queries: {complexity_types}")
        results = []
        
        # Ensure documents are loaded
        self.retriever.clear()
        self._add_medical_documents_to_retriever(documents)
        
        for complexity_type in complexity_types:
            logger.info(f"Testing {complexity_type} queries")
            start_time = time.time()
            
            # Generate complex queries
            complex_queries = self._generate_complex_queries(complexity_type, base_queries)
            
            # Test both methods
            classical_results = self._run_retrieval_test(complex_queries, method="classical")
            quantum_results = self._run_retrieval_test(complex_queries, method="quantum")
            
            # Create relevance judgments
            relevance_judgments = self.medical_relevance.create_relevance_judgments(complex_queries, documents)
            self.metrics_calculator.add_relevance_judgments(relevance_judgments)
            
            # Calculate metrics
            classical_metrics = self.metrics_calculator.evaluate_method(classical_results)
            quantum_metrics = self.metrics_calculator.evaluate_method(quantum_results)
            
            # Calculate improvements and significance
            improvements = self._calculate_improvements(classical_metrics, quantum_metrics)
            sig_test = self.metrics_calculator.statistical_significance_test(
                classical_results, quantum_results, "ndcg_10"
            )
            
            elapsed_time = (time.time() - start_time) * 1000
            
            result = ScenarioResult(
                scenario_name=f"complex_queries_{complexity_type}",
                config={'complexity_type': complexity_type},
                baseline_metrics=self._metrics_to_dict(classical_metrics),
                quantum_metrics=self._metrics_to_dict(quantum_metrics),
                performance_improvement=improvements,
                statistical_significance=sig_test,
                execution_time_ms=elapsed_time,
                metadata={
                    'complex_queries_generated': len(complex_queries),
                    'avg_query_length': np.mean([len(q.query_text.split()) for q in complex_queries])
                }
            )
            
            results.append(result)
            
            logger.info(f"Complex query test completed: "
                       f"Type: {complexity_type}, NDCG@10 improvement: {improvements.get('ndcg_10', 0):.3f}")
        
        return results
    
    def test_selective_quantum_usage(self, queries: List[MedicalQuery],
                                   documents: List[MedicalDocument],
                                   selection_strategies: List[str] = None) -> List[ScenarioResult]:
        """
        Test selective quantum usage strategies.
        
        Args:
            queries: Test queries
            documents: Medical documents  
            selection_strategies: Different strategies for when to use quantum
            
        Returns:
            List of scenario results for each strategy
        """
        if selection_strategies is None:
            selection_strategies = ['complexity_based', 'confidence_based', 'hybrid_weighted']
        
        logger.info(f"Testing selective quantum usage: {selection_strategies}")
        results = []
        
        # Ensure documents are loaded
        self.retriever.clear()
        self._add_medical_documents_to_retriever(documents)
        
        for strategy in selection_strategies:
            logger.info(f"Testing {strategy} selection strategy")
            start_time = time.time()
            
            # Run selective quantum strategy
            selective_results = self._run_selective_quantum_test(queries, strategy)
            
            # Compare against always-classical and always-quantum
            classical_results = self._run_retrieval_test(queries, method="classical")
            quantum_results = self._run_retrieval_test(queries, method="quantum")
            
            # Create relevance judgments
            relevance_judgments = self.medical_relevance.create_relevance_judgments(queries, documents)
            self.metrics_calculator.add_relevance_judgments(relevance_judgments)
            
            # Calculate metrics
            classical_metrics = self.metrics_calculator.evaluate_method(classical_results)
            quantum_metrics = self.metrics_calculator.evaluate_method(quantum_results)
            selective_metrics = self.metrics_calculator.evaluate_method(selective_results)
            
            # Calculate improvements over classical baseline
            improvements = self._calculate_improvements(classical_metrics, selective_metrics)
            
            elapsed_time = (time.time() - start_time) * 1000
            
            result = ScenarioResult(
                scenario_name=f"selective_usage_{strategy}",
                config={'selection_strategy': strategy},
                baseline_metrics=self._metrics_to_dict(classical_metrics),
                quantum_metrics=self._metrics_to_dict(selective_metrics),
                performance_improvement=improvements,
                statistical_significance={},  # Could add detailed comparison
                execution_time_ms=elapsed_time,
                metadata={
                    'always_classical_ndcg_10': classical_metrics.ndcg_at_k.get(10, 0),
                    'always_quantum_ndcg_10': quantum_metrics.ndcg_at_k.get(10, 0),
                    'selective_ndcg_10': selective_metrics.ndcg_at_k.get(10, 0)
                }
            )
            
            results.append(result)
            
            logger.info(f"Selective usage test completed: "
                       f"Strategy: {strategy}, NDCG@10 improvement: {improvements.get('ndcg_10', 0):.3f}")
        
        return results
    
    def _add_medical_documents_to_retriever(self, documents: List[MedicalDocument]) -> None:
        """Add medical documents to retriever."""
        texts = []
        metadatas = []
        
        for doc in documents:
            # Combine title and abstract for retrieval
            text = f"{doc.title} {doc.abstract}"
            texts.append(text)
            
            metadata = {
                'doc_id': doc.doc_id,
                'medical_domain': doc.medical_domain,
                'title': doc.title
            }
            metadatas.append(metadata)
        
        # Add documents to retriever
        doc_ids = self.retriever.add_texts(texts, metadatas)
        logger.debug(f"Added {len(doc_ids)} documents to retriever")
    
    def _run_retrieval_test(self, queries: List[MedicalQuery], method: str) -> List[QueryResult]:
        """Run retrieval test with specified method."""
        # Temporarily change retriever method
        original_method = self.retriever.config.reranking_method
        self.retriever.config.reranking_method = method
        
        results = []
        
        for query in queries:
            start_time = time.time()
            
            # Perform retrieval
            retrieval_results = self.retriever.retrieve(query.query_text, k=10)
            
            query_time = (time.time() - start_time) * 1000
            
            # Convert to QueryResult format
            ir_results = []
            for i, result in enumerate(retrieval_results):
                ir_result = RetrievalResult(
                    doc_id=result.doc_id,
                    score=result.score,
                    rank=i + 1,
                    metadata=result.metadata
                )
                ir_results.append(ir_result)
            
            query_result = QueryResult(
                query_id=query.query_id,
                query_text=query.query_text,
                results=ir_results,
                method=method,
                metadata={'query_time_ms': query_time}
            )
            
            results.append(query_result)
        
        # Restore original method
        self.retriever.config.reranking_method = original_method
        
        return results
    
    def _generate_complex_queries(self, complexity_type: str, 
                                base_queries: List[MedicalQuery]) -> List[MedicalQuery]:
        """Generate complex queries of specified type."""
        complex_queries = []
        
        if complexity_type == 'multi_domain':
            domains = ['cardiology', 'diabetes', 'respiratory', 'neurology']
            for i in range(5):  # Generate 5 multi-domain queries
                query_text = self.complex_query_generator.generate_multi_domain_query(domains)
                query = self.medical_relevance.create_medical_query(f"multi_domain_{i}", query_text)
                complex_queries.append(query)
        
        elif complexity_type == 'ambiguous':
            for i in range(5):  # Generate 5 ambiguous queries
                query_text = self.complex_query_generator.generate_ambiguous_query()
                query = self.medical_relevance.create_medical_query(f"ambiguous_{i}", query_text)
                complex_queries.append(query)
        
        elif complexity_type == 'long_clinical':
            for i in range(3):  # Generate 3 long clinical scenarios
                query_text = self.complex_query_generator.generate_long_complex_query()
                query = self.medical_relevance.create_medical_query(f"long_clinical_{i}", query_text)
                complex_queries.append(query)
        
        return complex_queries
    
    def _run_selective_quantum_test(self, queries: List[MedicalQuery], 
                                  strategy: str) -> List[QueryResult]:
        """Run test with selective quantum usage."""
        results = []
        
        for query in queries:
            start_time = time.time()
            
            # Decide whether to use quantum based on strategy
            use_quantum = self._should_use_quantum(query, strategy)
            
            method = "quantum" if use_quantum else "classical"
            
            # Temporarily change method
            original_method = self.retriever.config.reranking_method
            self.retriever.config.reranking_method = method
            
            # Perform retrieval
            retrieval_results = self.retriever.retrieve(query.query_text, k=10)
            
            # Restore original method
            self.retriever.config.reranking_method = original_method
            
            query_time = (time.time() - start_time) * 1000
            
            # Convert to QueryResult format
            ir_results = []
            for i, result in enumerate(retrieval_results):
                ir_result = RetrievalResult(
                    doc_id=result.doc_id,
                    score=result.score,
                    rank=i + 1,
                    metadata=result.metadata
                )
                ir_results.append(ir_result)
            
            query_result = QueryResult(
                query_id=query.query_id,
                query_text=query.query_text,
                results=ir_results,
                method=f"selective_{strategy}",
                metadata={
                    'query_time_ms': query_time,
                    'used_quantum': use_quantum,
                    'selection_strategy': strategy
                }
            )
            
            results.append(query_result)
        
        return results
    
    def _should_use_quantum(self, query: MedicalQuery, strategy: str) -> bool:
        """Decide whether to use quantum for a query based on strategy."""
        if strategy == 'complexity_based':
            # Use quantum for moderate and complex queries
            return query.complexity_level in ['moderate', 'complex']
        
        elif strategy == 'confidence_based':
            # Use quantum for multi-domain queries or ambiguous terms
            return (query.medical_domain == 'general' or 
                   len(query.key_terms) > 5)
        
        elif strategy == 'hybrid_weighted':
            # Use quantum 50% of the time (random for testing)
            return random.random() > 0.5
        
        else:
            return False
    
    def _calculate_improvements(self, baseline_metrics, test_metrics) -> Dict[str, float]:
        """Calculate percentage improvements over baseline."""
        improvements = {}
        
        # Calculate improvements for each metric
        for k in [5, 10, 20]:
            if k in baseline_metrics.precision_at_k and k in test_metrics.precision_at_k:
                baseline_val = baseline_metrics.precision_at_k[k]
                test_val = test_metrics.precision_at_k[k]
                improvement = ((test_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
                improvements[f'precision_{k}'] = improvement
            
            if k in baseline_metrics.ndcg_at_k and k in test_metrics.ndcg_at_k:
                baseline_val = baseline_metrics.ndcg_at_k[k]
                test_val = test_metrics.ndcg_at_k[k]
                improvement = ((test_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
                improvements[f'ndcg_{k}'] = improvement
        
        # MRR improvement
        if baseline_metrics.mrr > 0:
            mrr_improvement = ((test_metrics.mrr - baseline_metrics.mrr) / baseline_metrics.mrr * 100)
            improvements['mrr'] = mrr_improvement
        
        # MAP improvement
        if baseline_metrics.map_score > 0:
            map_improvement = ((test_metrics.map_score - baseline_metrics.map_score) / baseline_metrics.map_score * 100)
            improvements['map'] = map_improvement
        
        return improvements
    
    def _metrics_to_dict(self, metrics) -> Dict[str, float]:
        """Convert metrics object to dictionary."""
        result = {}
        
        # Precision@K
        for k, value in metrics.precision_at_k.items():
            result[f'precision_{k}'] = value
        
        # NDCG@K
        for k, value in metrics.ndcg_at_k.items():
            result[f'ndcg_{k}'] = value
        
        # Other metrics
        result['mrr'] = metrics.mrr
        result['map'] = metrics.map_score
        
        if metrics.avg_query_time_ms:
            result['avg_query_time_ms'] = metrics.avg_query_time_ms
        
        return result


def run_comprehensive_scenario_tests(retriever: TwoStageRetriever,
                                   queries: List[MedicalQuery],
                                   documents: List[MedicalDocument]) -> Dict[str, List[ScenarioResult]]:
    """
    Run comprehensive scenario tests for quantum advantage evaluation.
    
    Args:
        retriever: Two-stage retriever to test
        queries: Medical queries for testing
        documents: Medical documents corpus
        
    Returns:
        Dictionary mapping scenario types to results
    """
    medical_relevance = MedicalRelevanceJudgments()
    scenario_tester = QuantumAdvantageScenarios(retriever, medical_relevance)
    
    logger.info("Running comprehensive scenario tests")
    
    all_results = {}
    
    # Test 1: Noise tolerance
    noise_configs = [
        NoiseConfig(noise_type='ocr', noise_level=0.1),
        NoiseConfig(noise_type='ocr', noise_level=0.2),
        NoiseConfig(noise_type='mixed', noise_level=0.15)
    ]
    
    noise_results = scenario_tester.test_noise_tolerance(queries, documents, noise_configs)
    all_results['noise_tolerance'] = noise_results
    
    # Test 2: Complex queries
    complex_results = scenario_tester.test_complex_queries(queries, documents)
    all_results['complex_queries'] = complex_results
    
    # Test 3: Selective usage
    selective_results = scenario_tester.test_selective_quantum_usage(queries, documents)
    all_results['selective_usage'] = selective_results
    
    logger.info(f"Comprehensive scenario testing completed: "
               f"{sum(len(results) for results in all_results.values())} total tests")
    
    return all_results
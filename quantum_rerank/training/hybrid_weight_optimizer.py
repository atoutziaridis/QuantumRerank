"""
Medical Domain Hybrid Weight Optimization.

This module implements optimization of quantum/classical hybrid weights for
medical document ranking, finding optimal weight combinations for different
medical scenarios (clean vs noisy data, different specialties, etc).

Based on QRF-04 requirements for hybrid weight optimization pipeline.
"""

import logging
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import pickle
from pathlib import Path
from scipy.optimize import differential_evolution, minimize
from scipy.stats import spearmanr
import itertools

from .medical_data_preparation import TrainingPair
from ..core.quantum_similarity_engine import QuantumSimilarityEngine
from ..evaluation.ir_metrics import IRMetricsCalculator, QueryResult, RetrievalResult
from ..evaluation.scenario_testing import NoiseInjector, NoiseConfig
from ..config.settings import QuantumConfig

logger = logging.getLogger(__name__)


@dataclass
class HybridWeightConfig:
    """Configuration for hybrid weight optimization."""
    weight_search_method: str = "grid_search"  # grid_search, differential_evolution, bayesian
    weight_granularity: float = 0.1  # Step size for grid search
    min_weight: float = 0.0
    max_weight: float = 1.0
    optimization_metric: str = "ndcg_10"  # ndcg_10, precision_5, mrr, ranking_correlation
    test_scenarios: List[str] = None  # clean, noisy_ocr, noisy_abbrev, complex_queries
    medical_domains: List[str] = None  # cardiology, diabetes, respiratory, etc
    save_intermediate_results: bool = True
    random_seed: int = 42
    
    def __post_init__(self):
        if self.test_scenarios is None:
            self.test_scenarios = ["clean", "noisy_ocr", "noisy_abbrev", "complex_queries"]
        if self.medical_domains is None:
            self.medical_domains = ["cardiology", "diabetes", "respiratory", "neurology", "general"]


@dataclass
class ScenarioWeightResult:
    """Results for a specific scenario weight optimization."""
    scenario_name: str
    optimal_quantum_weight: float
    optimal_classical_weight: float
    performance_metric: float
    detailed_metrics: Dict[str, float]
    weight_performance_map: Dict[Tuple[float, float], float]


@dataclass
class HybridWeightOptimizationResult:
    """Complete results from hybrid weight optimization."""
    scenario_results: Dict[str, ScenarioWeightResult]
    domain_results: Dict[str, ScenarioWeightResult]
    overall_optimal_weights: Tuple[float, float]
    optimization_time_seconds: float
    recommendations: List[str]


class MedicalHybridOptimizer:
    """
    Optimizer for quantum/classical hybrid weights on medical data.
    
    Systematically tests different weight combinations to find optimal
    balance for various medical scenarios and domains.
    """
    
    def __init__(self, similarity_engine: QuantumSimilarityEngine,
                 config: Optional[HybridWeightConfig] = None):
        """Initialize medical hybrid optimizer."""
        self.similarity_engine = similarity_engine
        self.config = config or HybridWeightConfig()
        
        # Components for evaluation
        self.metrics_calculator = IRMetricsCalculator()
        self.noise_injector = NoiseInjector()
        
        # Set random seed
        np.random.seed(self.config.random_seed)
        
        logger.info(f"Medical hybrid optimizer initialized with {self.config.weight_search_method}")
    
    def optimize_hybrid_weights(self, test_pairs: List[TrainingPair]) -> HybridWeightOptimizationResult:
        """
        Optimize hybrid weights on medical test data.
        
        Args:
            test_pairs: Test data for optimization
            
        Returns:
            Optimization results with scenario-specific weights
        """
        logger.info(f"Optimizing hybrid weights on {len(test_pairs)} test pairs")
        start_time = time.time()
        
        # Prepare relevance judgments
        self._prepare_relevance_judgments(test_pairs)
        
        # Optimize for different scenarios
        scenario_results = {}
        
        # 1. Clean data optimization
        if "clean" in self.config.test_scenarios:
            logger.info("Optimizing weights for clean medical documents")
            clean_result = self._optimize_for_scenario(test_pairs, "clean")
            scenario_results["clean"] = clean_result
        
        # 2. Noisy data scenarios
        if "noisy_ocr" in self.config.test_scenarios:
            logger.info("Optimizing weights for OCR-corrupted documents")
            noisy_pairs = self._create_noisy_pairs(test_pairs, NoiseConfig('ocr', 0.15))
            ocr_result = self._optimize_for_scenario(noisy_pairs, "noisy_ocr")
            scenario_results["noisy_ocr"] = ocr_result
        
        if "noisy_abbrev" in self.config.test_scenarios:
            logger.info("Optimizing weights for medical abbreviation corruption")
            noisy_pairs = self._create_noisy_pairs(test_pairs, NoiseConfig('abbreviation', 0.3))
            abbrev_result = self._optimize_for_scenario(noisy_pairs, "noisy_abbrev")
            scenario_results["noisy_abbrev"] = abbrev_result
        
        # 3. Complex queries
        if "complex_queries" in self.config.test_scenarios:
            logger.info("Optimizing weights for complex medical queries")
            complex_pairs = self._filter_complex_queries(test_pairs)
            if complex_pairs:
                complex_result = self._optimize_for_scenario(complex_pairs, "complex_queries")
                scenario_results["complex_queries"] = complex_result
        
        # Optimize for different medical domains
        domain_results = self._optimize_by_medical_domain(test_pairs)
        
        # Find overall optimal weights
        overall_optimal = self._find_overall_optimal_weights(scenario_results, domain_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(scenario_results, domain_results)
        
        optimization_time = time.time() - start_time
        
        result = HybridWeightOptimizationResult(
            scenario_results=scenario_results,
            domain_results=domain_results,
            overall_optimal_weights=overall_optimal,
            optimization_time_seconds=optimization_time,
            recommendations=recommendations
        )
        
        logger.info(f"Hybrid weight optimization completed in {optimization_time:.2f}s")
        logger.info(f"Overall optimal weights: Quantum={overall_optimal[0]:.2f}, "
                   f"Classical={overall_optimal[1]:.2f}")
        
        return result
    
    def _prepare_relevance_judgments(self, test_pairs: List[TrainingPair]):
        """Prepare relevance judgments for evaluation."""
        judgments = []
        
        for pair in test_pairs:
            judgment = {
                'query_id': pair.query_id,
                'doc_id': pair.doc_id,
                'relevance': pair.relevance_label
            }
            judgments.append(judgment)
        
        # Add to metrics calculator
        self.metrics_calculator.clear()
        for judgment in judgments:
            self.metrics_calculator.add_relevance_judgment(
                judgment['query_id'],
                judgment['doc_id'],
                judgment['relevance']
            )
    
    def _optimize_for_scenario(self, test_pairs: List[TrainingPair],
                             scenario_name: str) -> ScenarioWeightResult:
        """Optimize weights for a specific scenario."""
        logger.info(f"Optimizing for scenario: {scenario_name}")
        
        if self.config.weight_search_method == "grid_search":
            return self._grid_search_optimization(test_pairs, scenario_name)
        elif self.config.weight_search_method == "differential_evolution":
            return self._differential_evolution_optimization(test_pairs, scenario_name)
        else:
            return self._grid_search_optimization(test_pairs, scenario_name)
    
    def _grid_search_optimization(self, test_pairs: List[TrainingPair],
                                scenario_name: str) -> ScenarioWeightResult:
        """Grid search optimization for weights."""
        # Generate weight combinations
        quantum_weights = np.arange(
            self.config.min_weight,
            self.config.max_weight + self.config.weight_granularity,
            self.config.weight_granularity
        )
        
        best_performance = -float('inf')
        best_quantum_weight = 0.5
        best_classical_weight = 0.5
        weight_performance_map = {}
        
        # Test each weight combination
        for q_weight in quantum_weights:
            c_weight = 1.0 - q_weight  # Ensure weights sum to 1
            
            # Evaluate with these weights
            performance, metrics = self._evaluate_weights(
                test_pairs, q_weight, c_weight
            )
            
            weight_performance_map[(q_weight, c_weight)] = performance
            
            if performance > best_performance:
                best_performance = performance
                best_quantum_weight = q_weight
                best_classical_weight = c_weight
                best_metrics = metrics
            
            logger.debug(f"Weights Q={q_weight:.2f}, C={c_weight:.2f}: "
                        f"{self.config.optimization_metric}={performance:.4f}")
        
        return ScenarioWeightResult(
            scenario_name=scenario_name,
            optimal_quantum_weight=best_quantum_weight,
            optimal_classical_weight=best_classical_weight,
            performance_metric=best_performance,
            detailed_metrics=best_metrics,
            weight_performance_map=weight_performance_map
        )
    
    def _differential_evolution_optimization(self, test_pairs: List[TrainingPair],
                                          scenario_name: str) -> ScenarioWeightResult:
        """Differential evolution optimization for weights."""
        
        def objective(weights):
            q_weight = weights[0]
            c_weight = 1.0 - q_weight
            performance, _ = self._evaluate_weights(test_pairs, q_weight, c_weight)
            return -performance  # Minimize negative performance
        
        # Run optimization
        result = differential_evolution(
            objective,
            [(self.config.min_weight, self.config.max_weight)],
            seed=self.config.random_seed,
            maxiter=50
        )
        
        best_quantum_weight = result.x[0]
        best_classical_weight = 1.0 - best_quantum_weight
        
        # Get detailed metrics for best weights
        best_performance, best_metrics = self._evaluate_weights(
            test_pairs, best_quantum_weight, best_classical_weight
        )
        
        return ScenarioWeightResult(
            scenario_name=scenario_name,
            optimal_quantum_weight=best_quantum_weight,
            optimal_classical_weight=best_classical_weight,
            performance_metric=best_performance,
            detailed_metrics=best_metrics,
            weight_performance_map={}  # Not tracking all evaluations
        )
    
    def _evaluate_weights(self, test_pairs: List[TrainingPair],
                        quantum_weight: float,
                        classical_weight: float) -> Tuple[float, Dict[str, float]]:
        """Evaluate performance with specific weights."""
        # Set weights in similarity engine
        self.similarity_engine.config.quantum_weight = quantum_weight
        self.similarity_engine.config.classical_weight = classical_weight
        
        # Run similarity computations
        query_results = []
        
        # Group by query
        query_pairs = {}
        for pair in test_pairs:
            if pair.query_id not in query_pairs:
                query_pairs[pair.query_id] = []
            query_pairs[pair.query_id].append(pair)
        
        # Process each query
        for query_id, pairs in query_pairs.items():
            results = []
            
            for pair in pairs:
                # Compute hybrid similarity
                similarity = self.similarity_engine.compute_similarity(
                    pair.query_embedding,
                    pair.doc_embedding,
                    method="hybrid"
                )
                
                result = RetrievalResult(
                    doc_id=pair.doc_id,
                    score=similarity,
                    rank=0  # Will be set later
                )
                results.append(result)
            
            # Sort by score and assign ranks
            results.sort(key=lambda x: x.score, reverse=True)
            for i, result in enumerate(results):
                result.rank = i + 1
            
            query_result = QueryResult(
                query_id=query_id,
                query_text=pairs[0].query_text,
                results=results,
                method="hybrid"
            )
            query_results.append(query_result)
        
        # Evaluate metrics
        evaluation = self.metrics_calculator.evaluate_method(query_results)
        
        # Get optimization metric
        if self.config.optimization_metric == "ndcg_10":
            performance = evaluation.ndcg_at_k.get(10, 0.0)
        elif self.config.optimization_metric == "precision_5":
            performance = evaluation.precision_at_k.get(5, 0.0)
        elif self.config.optimization_metric == "mrr":
            performance = evaluation.mrr
        elif self.config.optimization_metric == "ranking_correlation":
            # Compute ranking correlation
            predicted_scores = []
            true_relevance = []
            for query_result in query_results:
                for result in query_result.results:
                    predicted_scores.append(result.score)
                    # Get relevance from test pairs
                    for pair in test_pairs:
                        if pair.query_id == query_result.query_id and pair.doc_id == result.doc_id:
                            true_relevance.append(pair.relevance_label)
                            break
            
            if predicted_scores and true_relevance:
                correlation, _ = spearmanr(predicted_scores, true_relevance)
                performance = correlation
            else:
                performance = 0.0
        else:
            performance = evaluation.ndcg_at_k.get(10, 0.0)
        
        # Collect all metrics
        metrics = {
            'ndcg_5': evaluation.ndcg_at_k.get(5, 0.0),
            'ndcg_10': evaluation.ndcg_at_k.get(10, 0.0),
            'precision_5': evaluation.precision_at_k.get(5, 0.0),
            'precision_10': evaluation.precision_at_k.get(10, 0.0),
            'mrr': evaluation.mrr,
            'map': evaluation.map_score
        }
        
        return performance, metrics
    
    def _create_noisy_pairs(self, test_pairs: List[TrainingPair],
                          noise_config: NoiseConfig) -> List[TrainingPair]:
        """Create noisy versions of test pairs."""
        noisy_pairs = []
        
        for pair in test_pairs:
            # Inject noise into text
            noisy_query = self.noise_injector.inject_noise(pair.query_text, noise_config)
            noisy_doc = self.noise_injector.inject_noise(pair.doc_text, noise_config)
            
            # Re-compute embeddings for noisy text
            # Note: In real implementation, would use embedding processor
            # For now, add small noise to existing embeddings
            noise_scale = 0.05 * noise_config.noise_level
            query_noise = np.random.normal(0, noise_scale, pair.query_embedding.shape)
            doc_noise = np.random.normal(0, noise_scale, pair.doc_embedding.shape)
            
            noisy_pair = TrainingPair(
                query_id=pair.query_id,
                doc_id=pair.doc_id,
                query_text=noisy_query,
                doc_text=noisy_doc,
                query_embedding=pair.query_embedding + query_noise,
                doc_embedding=pair.doc_embedding + doc_noise,
                relevance_label=pair.relevance_label,
                medical_domain=pair.medical_domain,
                pair_type=pair.pair_type
            )
            noisy_pairs.append(noisy_pair)
        
        return noisy_pairs
    
    def _filter_complex_queries(self, test_pairs: List[TrainingPair]) -> List[TrainingPair]:
        """Filter test pairs to only complex queries."""
        complex_pairs = []
        
        for pair in test_pairs:
            # Simple heuristic for complexity
            query_length = len(pair.query_text.split())
            has_multiple_terms = len(set(pair.query_text.lower().split())) > 10
            
            if query_length > 15 or has_multiple_terms:
                complex_pairs.append(pair)
        
        return complex_pairs
    
    def _optimize_by_medical_domain(self, test_pairs: List[TrainingPair]) -> Dict[str, ScenarioWeightResult]:
        """Optimize weights for each medical domain."""
        domain_results = {}
        
        # Group by domain
        domain_pairs = {}
        for pair in test_pairs:
            domain = pair.medical_domain
            if domain not in domain_pairs:
                domain_pairs[domain] = []
            domain_pairs[domain].append(pair)
        
        # Optimize for each domain
        for domain in self.config.medical_domains:
            if domain in domain_pairs and len(domain_pairs[domain]) >= 10:
                logger.info(f"Optimizing weights for {domain} domain")
                domain_result = self._optimize_for_scenario(
                    domain_pairs[domain],
                    f"domain_{domain}"
                )
                domain_results[domain] = domain_result
        
        return domain_results
    
    def _find_overall_optimal_weights(self, scenario_results: Dict[str, ScenarioWeightResult],
                                    domain_results: Dict[str, ScenarioWeightResult]) -> Tuple[float, float]:
        """Find overall optimal weights considering all scenarios."""
        all_results = list(scenario_results.values()) + list(domain_results.values())
        
        if not all_results:
            return (0.5, 0.5)
        
        # Weight scenarios by importance
        scenario_weights = {
            "clean": 0.3,
            "noisy_ocr": 0.2,
            "noisy_abbrev": 0.1,
            "complex_queries": 0.2
        }
        
        # Domain weights (equal for all)
        domain_weight = 0.2 / len(domain_results) if domain_results else 0
        
        # Calculate weighted average
        total_quantum_weight = 0.0
        total_weight = 0.0
        
        for scenario_name, result in scenario_results.items():
            weight = scenario_weights.get(scenario_name, 0.1)
            total_quantum_weight += result.optimal_quantum_weight * weight
            total_weight += weight
        
        for domain, result in domain_results.items():
            total_quantum_weight += result.optimal_quantum_weight * domain_weight
            total_weight += domain_weight
        
        # Normalize
        if total_weight > 0:
            optimal_quantum = total_quantum_weight / total_weight
        else:
            optimal_quantum = 0.5
        
        optimal_classical = 1.0 - optimal_quantum
        
        return (optimal_quantum, optimal_classical)
    
    def _generate_recommendations(self, scenario_results: Dict[str, ScenarioWeightResult],
                                domain_results: Dict[str, ScenarioWeightResult]) -> List[str]:
        """Generate deployment recommendations based on optimization results."""
        recommendations = []
        
        # Overall recommendation
        overall_q, overall_c = self._find_overall_optimal_weights(scenario_results, domain_results)
        recommendations.append(
            f"Overall recommended weights: Quantum={overall_q:.2f}, Classical={overall_c:.2f}"
        )
        
        # Scenario-specific recommendations
        if "clean" in scenario_results and "noisy_ocr" in scenario_results:
            clean_q = scenario_results["clean"].optimal_quantum_weight
            noisy_q = scenario_results["noisy_ocr"].optimal_quantum_weight
            
            if noisy_q > clean_q + 0.1:
                recommendations.append(
                    "Quantum methods show stronger performance on noisy medical documents. "
                    f"Consider using quantum weight={noisy_q:.2f} for OCR-processed documents."
                )
        
        # Domain-specific recommendations
        domain_variations = []
        for domain, result in domain_results.items():
            if abs(result.optimal_quantum_weight - overall_q) > 0.15:
                domain_variations.append(
                    f"{domain}: Q={result.optimal_quantum_weight:.2f}"
                )
        
        if domain_variations:
            recommendations.append(
                "Consider domain-specific weights: " + ", ".join(domain_variations)
            )
        
        # Performance-based recommendations
        best_scenario = max(scenario_results.values(), 
                          key=lambda x: x.performance_metric)
        if best_scenario.performance_metric > 0.8:
            recommendations.append(
                f"Strong performance achieved in {best_scenario.scenario_name} scenario "
                f"({self.config.optimization_metric}={best_scenario.performance_metric:.3f})"
            )
        
        # Stability recommendations
        weight_variations = [r.optimal_quantum_weight for r in scenario_results.values()]
        if weight_variations:
            variation = np.std(weight_variations)
            if variation < 0.1:
                recommendations.append(
                    "Weights are stable across scenarios - single configuration recommended"
                )
            else:
                recommendations.append(
                    "Consider adaptive weight selection based on input characteristics"
                )
        
        return recommendations


class HybridWeightOptimizationPipeline:
    """
    Complete pipeline for hybrid weight optimization on medical data.
    """
    
    def __init__(self, similarity_engine: QuantumSimilarityEngine,
                 config: Optional[HybridWeightConfig] = None):
        """Initialize hybrid weight optimization pipeline."""
        self.similarity_engine = similarity_engine
        self.config = config or HybridWeightConfig()
        self.optimizer = MedicalHybridOptimizer(similarity_engine, config)
        
        logger.info("Hybrid weight optimization pipeline initialized")
    
    def run(self, test_pairs: List[TrainingPair],
            output_dir: str = "hybrid_weight_optimization") -> Dict[str, Any]:
        """
        Run complete hybrid weight optimization pipeline.
        
        Args:
            test_pairs: Test data for optimization
            output_dir: Output directory
            
        Returns:
            Pipeline results
        """
        logger.info("Starting hybrid weight optimization pipeline")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Run optimization
        logger.info("Step 1: Optimizing hybrid weights")
        optimization_result = self.optimizer.optimize_hybrid_weights(test_pairs)
        
        # Step 2: Validate optimal weights
        logger.info("Step 2: Validating optimal weights")
        validation_results = self._validate_optimal_weights(
            test_pairs, optimization_result.overall_optimal_weights
        )
        
        # Step 3: Generate deployment configuration
        logger.info("Step 3: Generating deployment configuration")
        deployment_config = self._generate_deployment_config(optimization_result)
        
        # Step 4: Save results
        logger.info("Step 4: Saving optimization results")
        results_path = output_path / "hybrid_weight_results.pkl"
        config_path = output_path / "deployment_config.pkl"
        
        # Save comprehensive results
        results = {
            'optimization_result': optimization_result,
            'validation_results': validation_results,
            'deployment_config': deployment_config,
            'config': self.config,
            'data_summary': {
                'test_pairs': len(test_pairs),
                'scenarios_tested': len(optimization_result.scenario_results),
                'domains_tested': len(optimization_result.domain_results)
            },
            'file_paths': {
                'results': str(results_path),
                'config': str(config_path)
            }
        }
        
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        with open(config_path, 'wb') as f:
            pickle.dump(deployment_config, f)
        
        # Generate summary report
        self._generate_summary_report(optimization_result, output_path)
        
        logger.info("Hybrid weight optimization pipeline completed successfully")
        logger.info(f"Optimal weights: Q={optimization_result.overall_optimal_weights[0]:.2f}, "
                   f"C={optimization_result.overall_optimal_weights[1]:.2f}")
        
        return results
    
    def _validate_optimal_weights(self, test_pairs: List[TrainingPair],
                                optimal_weights: Tuple[float, float]) -> Dict[str, float]:
        """Validate optimal weights on test data."""
        performance, metrics = self.optimizer._evaluate_weights(
            test_pairs, optimal_weights[0], optimal_weights[1]
        )
        
        return {
            'optimal_weight_performance': performance,
            'detailed_metrics': metrics,
            'quantum_weight': optimal_weights[0],
            'classical_weight': optimal_weights[1]
        }
    
    def _generate_deployment_config(self, optimization_result: HybridWeightOptimizationResult) -> Dict[str, Any]:
        """Generate deployment configuration based on optimization."""
        config = {
            'default_weights': {
                'quantum': optimization_result.overall_optimal_weights[0],
                'classical': optimization_result.overall_optimal_weights[1]
            },
            'scenario_weights': {},
            'domain_weights': {},
            'adaptive_selection': {
                'enabled': False,
                'rules': []
            }
        }
        
        # Add scenario-specific weights
        for scenario, result in optimization_result.scenario_results.items():
            config['scenario_weights'][scenario] = {
                'quantum': result.optimal_quantum_weight,
                'classical': result.optimal_classical_weight,
                'performance': result.performance_metric
            }
        
        # Add domain-specific weights
        for domain, result in optimization_result.domain_results.items():
            config['domain_weights'][domain] = {
                'quantum': result.optimal_quantum_weight,
                'classical': result.optimal_classical_weight,
                'performance': result.performance_metric
            }
        
        # Check if adaptive selection is recommended
        weight_variations = [r.optimal_quantum_weight 
                           for r in optimization_result.scenario_results.values()]
        if weight_variations and np.std(weight_variations) > 0.15:
            config['adaptive_selection']['enabled'] = True
            
            # Add adaptive rules
            if "noisy_ocr" in optimization_result.scenario_results:
                noisy_result = optimization_result.scenario_results["noisy_ocr"]
                if noisy_result.optimal_quantum_weight > config['default_weights']['quantum'] + 0.1:
                    config['adaptive_selection']['rules'].append({
                        'condition': 'detected_ocr_noise',
                        'quantum_weight': noisy_result.optimal_quantum_weight,
                        'classical_weight': noisy_result.optimal_classical_weight
                    })
        
        return config
    
    def _generate_summary_report(self, optimization_result: HybridWeightOptimizationResult,
                               output_path: Path):
        """Generate human-readable summary report."""
        report_path = output_path / "optimization_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("Hybrid Weight Optimization Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Optimization completed in {optimization_result.optimization_time_seconds:.2f} seconds\n\n")
            
            f.write("Overall Optimal Weights:\n")
            f.write(f"  Quantum: {optimization_result.overall_optimal_weights[0]:.3f}\n")
            f.write(f"  Classical: {optimization_result.overall_optimal_weights[1]:.3f}\n\n")
            
            f.write("Scenario-Specific Results:\n")
            for scenario, result in optimization_result.scenario_results.items():
                f.write(f"  {scenario}:\n")
                f.write(f"    Optimal: Q={result.optimal_quantum_weight:.3f}, "
                       f"C={result.optimal_classical_weight:.3f}\n")
                f.write(f"    Performance: {result.performance_metric:.4f}\n")
            
            f.write("\nDomain-Specific Results:\n")
            for domain, result in optimization_result.domain_results.items():
                f.write(f"  {domain}:\n")
                f.write(f"    Optimal: Q={result.optimal_quantum_weight:.3f}, "
                       f"C={result.optimal_classical_weight:.3f}\n")
                f.write(f"    Performance: {result.performance_metric:.4f}\n")
            
            f.write("\nRecommendations:\n")
            for i, rec in enumerate(optimization_result.recommendations, 1):
                f.write(f"  {i}. {rec}\n")
        
        logger.info(f"Summary report saved to {report_path}")
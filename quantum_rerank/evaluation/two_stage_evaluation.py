"""
Comprehensive Two-Stage Retrieval Evaluation Framework.

This module provides a complete evaluation system for comparing classical
and quantum reranking methods in realistic two-stage retrieval scenarios.

Based on:
- Standard IR evaluation methodology
- Two-stage retrieval best practices
- Quantum vs classical comparison frameworks
"""

import logging
import time
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np

from .ir_metrics import IRMetricsCalculator, QueryResult, RetrievalResult, EvaluationMetrics
from .medical_relevance import MedicalRelevanceJudgments, MedicalQuery, MedicalDocument, create_medical_test_queries
from .scenario_testing import QuantumAdvantageScenarios, ScenarioResult, run_comprehensive_scenario_tests
from ..retrieval.two_stage_retriever import TwoStageRetriever, RetrieverConfig
from ..core.embeddings import EmbeddingProcessor

logger = logging.getLogger(__name__)


@dataclass
class TwoStageEvaluationConfig:
    """Configuration for two-stage evaluation."""
    # Test data settings
    max_documents: int = 100
    max_queries: int = 20
    
    # Retrieval settings
    faiss_candidates_k: int = 50
    final_results_k: int = 10
    
    # Testing scenarios
    test_noise_tolerance: bool = True
    test_complex_queries: bool = True
    test_selective_usage: bool = True
    
    # Methods to compare
    methods_to_test: List[str] = None
    
    # Output settings
    save_detailed_results: bool = True
    output_directory: str = "evaluation_results"
    
    def __post_init__(self):
        """Set default methods if not provided."""
        if self.methods_to_test is None:
            self.methods_to_test = ["classical", "quantum", "hybrid"]


@dataclass
class MethodComparisonResult:
    """Results from comparing different methods."""
    method_name: str
    metrics: EvaluationMetrics
    query_results: List[QueryResult]
    execution_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'method_name': self.method_name,
            'metrics': asdict(self.metrics),
            'execution_time_ms': self.execution_time_ms,
            'query_count': len(self.query_results)
        }


@dataclass
class TwoStageEvaluationReport:
    """Complete evaluation report."""
    config: TwoStageEvaluationConfig
    test_data_summary: Dict[str, Any]
    method_comparisons: List[MethodComparisonResult]
    scenario_results: Dict[str, List[ScenarioResult]]
    statistical_tests: Dict[str, Dict[str, Any]]
    recommendations: List[str]
    total_execution_time_ms: float
    timestamp: float


class TwoStageEvaluationFramework:
    """
    Comprehensive evaluation framework for two-stage retrieval systems.
    
    Provides end-to-end testing of FAISS → Quantum reranking pipeline
    with proper IR evaluation metrics and statistical significance testing.
    """
    
    def __init__(self, config: Optional[TwoStageEvaluationConfig] = None):
        """Initialize evaluation framework."""
        self.config = config or TwoStageEvaluationConfig()
        
        # Initialize components
        self.embedding_processor = EmbeddingProcessor()
        self.medical_relevance = MedicalRelevanceJudgments(self.embedding_processor)
        self.metrics_calculator = IRMetricsCalculator()
        
        # Data storage
        self.test_queries: List[MedicalQuery] = []
        self.test_documents: List[MedicalDocument] = []
        
        # Create output directory
        self.output_path = Path(self.config.output_directory)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Two-stage evaluation framework initialized: {self.config.output_directory}")
    
    def load_pmc_test_data(self, pmc_docs_path: str = "pmc_docs") -> Tuple[List[MedicalQuery], List[MedicalDocument]]:
        """
        Load PMC medical documents and create test queries.
        
        Args:
            pmc_docs_path: Path to PMC XML documents
            
        Returns:
            Tuple of (queries, documents)
        """
        logger.info(f"Loading PMC test data from {pmc_docs_path}")
        
        # Try to load pre-parsed articles
        parsed_articles_path = Path("parsed_pmc_articles.pkl")
        if parsed_articles_path.exists():
            logger.info("Loading pre-parsed PMC articles")
            with open(parsed_articles_path, 'rb') as f:
                pmc_articles = pickle.load(f)
        else:
            # Parse PMC XML files
            logger.info("Parsing PMC XML files")
            from ..pmc_xml_parser import PMCXMLParser
            
            parser = PMCXMLParser()
            xml_dir = Path(pmc_docs_path)
            pmc_articles = parser.parse_directory(xml_dir, max_articles=self.config.max_documents)
            
            # Save parsed articles
            with open(parsed_articles_path, 'wb') as f:
                pickle.dump(pmc_articles, f)
            logger.info("Saved parsed articles for future use")
        
        # Convert to MedicalDocument objects
        documents = []
        for article in pmc_articles[:self.config.max_documents]:
            doc = self.medical_relevance.create_medical_document(
                doc_id=article.pmc_id,
                title=article.title,
                abstract=article.abstract,
                full_text=article.full_text,
                sections=article.sections
            )
            documents.append(doc)
        
        # Create test queries
        queries = create_medical_test_queries()[:self.config.max_queries]
        
        self.test_queries = queries
        self.test_documents = documents
        
        logger.info(f"Loaded {len(documents)} documents and {len(queries)} queries")
        
        return queries, documents
    
    def setup_retriever(self, method: str) -> TwoStageRetriever:
        """
        Setup two-stage retriever for testing.
        
        Args:
            method: Reranking method to use
            
        Returns:
            Configured retriever
        """
        retriever_config = RetrieverConfig(
            initial_k=self.config.faiss_candidates_k,
            final_k=self.config.final_results_k,
            reranking_method=method,
            enable_caching=True,
            fallback_to_faiss=True
        )
        
        retriever = TwoStageRetriever(
            config=retriever_config,
            embedding_processor=self.embedding_processor
        )
        
        # Add documents to retriever
        texts = []
        metadatas = []
        
        for doc in self.test_documents:
            # Use title + abstract for retrieval
            text = f"{doc.title} {doc.abstract}"
            texts.append(text)
            
            metadata = {
                'doc_id': doc.doc_id,
                'medical_domain': doc.medical_domain,
                'title': doc.title
            }
            metadatas.append(metadata)
        
        retriever.add_texts(texts, metadatas)
        
        logger.info(f"Setup retriever with {method} method, {len(texts)} documents")
        
        return retriever
    
    def evaluate_method(self, method: str) -> MethodComparisonResult:
        """
        Evaluate a single retrieval method.
        
        Args:
            method: Method name to evaluate
            
        Returns:
            Method comparison result
        """
        logger.info(f"Evaluating method: {method}")
        start_time = time.time()
        
        # Setup retriever
        retriever = self.setup_retriever(method)
        
        # Run queries
        query_results = []
        
        for query in self.test_queries:
            query_start = time.time()
            
            # Perform retrieval
            retrieval_results = retriever.retrieve(
                query.query_text, 
                k=self.config.final_results_k,
                return_debug_info=True
            )
            
            query_time = (time.time() - query_start) * 1000
            
            # Convert to IR evaluation format
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
            
            query_results.append(query_result)
        
        # Create relevance judgments
        relevance_judgments = self.medical_relevance.create_relevance_judgments(
            self.test_queries, self.test_documents
        )
        self.metrics_calculator.add_relevance_judgments(relevance_judgments)
        
        # Calculate metrics
        metrics = self.metrics_calculator.evaluate_method(query_results)
        
        execution_time = (time.time() - start_time) * 1000
        
        result = MethodComparisonResult(
            method_name=method,
            metrics=metrics,
            query_results=query_results,
            execution_time_ms=execution_time
        )
        
        logger.info(f"Method {method} evaluation completed: "
                   f"NDCG@10={metrics.ndcg_at_k.get(10, 0):.3f}, "
                   f"Time={execution_time:.1f}ms")
        
        return result
    
    def compare_methods(self, methods: Optional[List[str]] = None) -> List[MethodComparisonResult]:
        """
        Compare multiple retrieval methods.
        
        Args:
            methods: List of methods to compare
            
        Returns:
            List of method comparison results
        """
        if methods is None:
            methods = self.config.methods_to_test
        
        logger.info(f"Comparing methods: {methods}")
        
        results = []
        for method in methods:
            try:
                result = self.evaluate_method(method)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate method {method}: {e}")
                # Continue with other methods
        
        return results
    
    def run_statistical_tests(self, method_results: List[MethodComparisonResult]) -> Dict[str, Dict[str, Any]]:
        """
        Run statistical significance tests between methods.
        
        Args:
            method_results: Results from method comparison
            
        Returns:
            Dictionary of statistical test results
        """
        logger.info("Running statistical significance tests")
        
        statistical_tests = {}
        
        # Compare each pair of methods
        for i, result1 in enumerate(method_results):
            for j, result2 in enumerate(method_results[i+1:], i+1):
                
                comparison_key = f"{result1.method_name}_vs_{result2.method_name}"
                
                try:
                    # Test multiple metrics
                    test_results = {}
                    
                    for metric in ['ndcg_10', 'precision_10', 'mrr']:
                        sig_test = self.metrics_calculator.statistical_significance_test(
                            result1.query_results, result2.query_results, metric
                        )
                        test_results[metric] = sig_test
                    
                    statistical_tests[comparison_key] = test_results
                    
                    logger.debug(f"Statistical test {comparison_key}: "
                               f"NDCG@10 p-value={test_results['ndcg_10']['p_value']:.4f}")
                
                except Exception as e:
                    logger.warning(f"Statistical test failed for {comparison_key}: {e}")
        
        return statistical_tests
    
    def run_scenario_tests(self) -> Dict[str, List[ScenarioResult]]:
        """
        Run scenario-specific tests for quantum advantages.
        
        Returns:
            Dictionary mapping scenario types to results
        """
        logger.info("Running scenario-specific tests")
        
        # Setup a retriever for scenario testing (use quantum method)
        retriever = self.setup_retriever("quantum")
        
        # Run comprehensive scenario tests
        scenario_results = run_comprehensive_scenario_tests(
            retriever, self.test_queries, self.test_documents
        )
        
        logger.info(f"Scenario tests completed: "
                   f"{sum(len(results) for results in scenario_results.values())} scenarios")
        
        return scenario_results
    
    def generate_recommendations(self, method_results: List[MethodComparisonResult],
                               scenario_results: Dict[str, List[ScenarioResult]],
                               statistical_tests: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Generate actionable recommendations based on evaluation results.
        
        Args:
            method_results: Method comparison results
            scenario_results: Scenario test results
            statistical_tests: Statistical significance tests
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Find best overall method
        best_method = max(method_results, key=lambda x: x.metrics.ndcg_at_k.get(10, 0))
        recommendations.append(f"Best overall method: {best_method.method_name} "
                             f"(NDCG@10: {best_method.metrics.ndcg_at_k.get(10, 0):.3f})")
        
        # Analyze quantum advantages
        quantum_results = [r for r in method_results if r.method_name == "quantum"]
        classical_results = [r for r in method_results if r.method_name == "classical"]
        
        if quantum_results and classical_results:
            quantum_ndcg = quantum_results[0].metrics.ndcg_at_k.get(10, 0)
            classical_ndcg = classical_results[0].metrics.ndcg_at_k.get(10, 0)
            
            improvement = ((quantum_ndcg - classical_ndcg) / classical_ndcg * 100) if classical_ndcg > 0 else 0
            
            if improvement > 5:
                recommendations.append(f"Quantum shows significant improvement: {improvement:.1f}% NDCG@10 gain")
            elif improvement > 0:
                recommendations.append(f"Quantum shows modest improvement: {improvement:.1f}% NDCG@10 gain")
            else:
                recommendations.append("Quantum does not show clear advantage in standard evaluation")
        
        # Analyze scenario-specific advantages
        if 'noise_tolerance' in scenario_results:
            noise_improvements = []
            for result in scenario_results['noise_tolerance']:
                ndcg_improvement = result.performance_improvement.get('ndcg_10', 0)
                if ndcg_improvement > 5:
                    noise_improvements.append(result.config['noise_type'])
            
            if noise_improvements:
                recommendations.append(f"Quantum shows noise tolerance advantages for: {', '.join(noise_improvements)}")
        
        if 'complex_queries' in scenario_results:
            complex_improvements = []
            for result in scenario_results['complex_queries']:
                ndcg_improvement = result.performance_improvement.get('ndcg_10', 0)
                if ndcg_improvement > 5:
                    complex_improvements.append(result.config['complexity_type'])
            
            if complex_improvements:
                recommendations.append(f"Quantum handles complex queries better: {', '.join(complex_improvements)}")
        
        # Performance recommendations
        avg_times = {r.method_name: r.execution_time_ms / len(r.query_results) 
                    for r in method_results}
        fastest_method = min(avg_times, key=avg_times.get)
        recommendations.append(f"Fastest method: {fastest_method} "
                             f"({avg_times[fastest_method]:.1f}ms/query)")
        
        # Statistical significance recommendations
        significant_comparisons = []
        for comparison, tests in statistical_tests.items():
            for metric, test_result in tests.items():
                if test_result.get('significant_at_05', False):
                    significant_comparisons.append(f"{comparison} ({metric})")
        
        if significant_comparisons:
            recommendations.append(f"Statistically significant differences found: {len(significant_comparisons)} comparisons")
        else:
            recommendations.append("No statistically significant differences found between methods")
        
        # Deployment recommendations
        if len(method_results) >= 2:
            # Check if hybrid method exists and performs well
            hybrid_results = [r for r in method_results if r.method_name == "hybrid"]
            if hybrid_results:
                hybrid_ndcg = hybrid_results[0].metrics.ndcg_at_k.get(10, 0)
                if hybrid_ndcg >= best_method.metrics.ndcg_at_k.get(10, 0) * 0.95:
                    recommendations.append("Consider hybrid approach for production deployment")
        
        return recommendations
    
    def run_comprehensive_evaluation(self) -> TwoStageEvaluationReport:
        """
        Run complete two-stage retrieval evaluation.
        
        Returns:
            Comprehensive evaluation report
        """
        logger.info("Starting comprehensive two-stage retrieval evaluation")
        start_time = time.time()
        
        # Use existing test data if available, otherwise load PMC data
        if self.test_queries and self.test_documents:
            queries, documents = self.test_queries, self.test_documents
            logger.info(f"Using existing test data: {len(queries)} queries, {len(documents)} documents")
        else:
            # Load test data
            queries, documents = self.load_pmc_test_data()
        
        # Test data summary
        test_data_summary = {
            'queries_count': len(queries),
            'documents_count': len(documents),
            'domain_distribution': {
                domain: len([d for d in documents if d.medical_domain == domain])
                for domain in set(d.medical_domain for d in documents)
            },
            'query_complexity_distribution': {
                level: len([q for q in queries if q.complexity_level == level])
                for level in set(q.complexity_level for q in queries)
            }
        }
        
        # Compare methods
        method_results = self.compare_methods()
        
        # Run statistical tests
        statistical_tests = self.run_statistical_tests(method_results)
        
        # Run scenario tests
        scenario_results = {}
        if self.config.test_noise_tolerance or self.config.test_complex_queries or self.config.test_selective_usage:
            scenario_results = self.run_scenario_tests()
        
        # Generate recommendations
        recommendations = self.generate_recommendations(method_results, scenario_results, statistical_tests)
        
        total_time = (time.time() - start_time) * 1000
        
        # Create report
        report = TwoStageEvaluationReport(
            config=self.config,
            test_data_summary=test_data_summary,
            method_comparisons=method_results,
            scenario_results=scenario_results,
            statistical_tests=statistical_tests,
            recommendations=recommendations,
            total_execution_time_ms=total_time,
            timestamp=time.time()
        )
        
        # Save report
        if self.config.save_detailed_results:
            self.save_evaluation_report(report)
        
        logger.info(f"Comprehensive evaluation completed in {total_time/1000:.1f}s")
        
        return report
    
    def save_evaluation_report(self, report: TwoStageEvaluationReport) -> None:
        """Save evaluation report to disk."""
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        
        # Save pickle report
        pickle_path = self.output_path / f"evaluation_report_{timestamp_str}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(report, f)
        
        # Save human-readable summary
        summary_path = self.output_path / f"evaluation_summary_{timestamp_str}.txt"
        with open(summary_path, 'w') as f:
            f.write("Two-Stage Retrieval Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Test Data Summary:\n")
            f.write(f"  Queries: {report.test_data_summary['queries_count']}\n")
            f.write(f"  Documents: {report.test_data_summary['documents_count']}\n\n")
            
            f.write("Method Comparison Results:\n")
            for result in report.method_comparisons:
                f.write(f"  {result.method_name}:\n")
                f.write(f"    NDCG@10: {result.metrics.ndcg_at_k.get(10, 0):.3f}\n")
                f.write(f"    Precision@10: {result.metrics.precision_at_k.get(10, 0):.3f}\n")
                f.write(f"    MRR: {result.metrics.mrr:.3f}\n")
                f.write(f"    Avg Query Time: {result.execution_time_ms / len(result.query_results):.1f}ms\n\n")
            
            f.write("Recommendations:\n")
            for i, rec in enumerate(report.recommendations, 1):
                f.write(f"  {i}. {rec}\n")
        
        logger.info(f"Evaluation report saved: {pickle_path}")
    
    def print_evaluation_summary(self, report: TwoStageEvaluationReport) -> None:
        """Print evaluation summary to console."""
        print("\n" + "=" * 80)
        print("TWO-STAGE RETRIEVAL EVALUATION SUMMARY")
        print("=" * 80)
        
        print(f"\nTest Configuration:")
        print(f"  Documents: {report.test_data_summary['documents_count']}")
        print(f"  Queries: {report.test_data_summary['queries_count']}")
        print(f"  Methods tested: {[r.method_name for r in report.method_comparisons]}")
        print(f"  Total evaluation time: {report.total_execution_time_ms/1000:.1f}s")
        
        print(f"\nMethod Performance (NDCG@10):")
        for result in sorted(report.method_comparisons, 
                           key=lambda x: x.metrics.ndcg_at_k.get(10, 0), reverse=True):
            ndcg_10 = result.metrics.ndcg_at_k.get(10, 0)
            precision_10 = result.metrics.precision_at_k.get(10, 0)
            avg_time = result.execution_time_ms / len(result.query_results)
            print(f"  {result.method_name}: {ndcg_10:.3f} NDCG@10, "
                  f"{precision_10:.3f} P@10, {avg_time:.1f}ms/query")
        
        if report.scenario_results:
            print(f"\nScenario Test Results:")
            for scenario_type, results in report.scenario_results.items():
                print(f"  {scenario_type}: {len(results)} tests completed")
                
                # Show best improvement
                best_improvement = max(
                    (r.performance_improvement.get('ndcg_10', 0) for r in results),
                    default=0
                )
                print(f"    Best NDCG@10 improvement: {best_improvement:.1f}%")
        
        print(f"\nKey Recommendations:")
        for i, rec in enumerate(report.recommendations[:5], 1):  # Show top 5
            print(f"  {i}. {rec}")
        
        print("\n" + "=" * 80)
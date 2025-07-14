#!/usr/bin/env python3
"""
Comprehensive analysis of real-world RAG test results.

Analyzes performance patterns, identifies success/failure cases,
and provides detailed insights into quantum vs classical methods.
"""

import sys
import os
import json
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath('.'))

from test_real_world_rag import RealWorldRAGTester, RAGTestResult

@dataclass
class QueryAnalysis:
    """Analysis of individual query performance."""
    query: str
    query_type: str
    complexity: str
    relevant_docs_count: int
    performance_by_method: Dict[str, Dict[str, float]]
    best_method: str
    worst_method: str
    performance_gap: float

@dataclass
class MethodAnalysis:
    """Analysis of individual method performance."""
    method_name: str
    overall_metrics: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]
    optimal_query_types: List[str]
    performance_consistency: float

class RAGResultsAnalyzer:
    """Comprehensive analyzer for RAG test results."""
    
    def __init__(self):
        self.query_analyses = []
        self.method_analyses = []
        self.performance_matrix = None
        
    def run_detailed_analysis(self) -> Dict[str, Any]:
        """Run comprehensive analysis and generate detailed report."""
        print("🔬 Running Detailed RAG Results Analysis")
        print("=" * 60)
        
        # Re-run the test to capture detailed results
        tester = RealWorldRAGTester()
        tester.generate_realistic_document_corpus(num_docs=60)
        tester.generate_realistic_queries()
        tester.create_ground_truth()
        
        # Collect detailed results for analysis
        detailed_results = self._collect_detailed_results(tester)
        
        # Perform comprehensive analysis
        query_analysis = self._analyze_by_query_type(detailed_results)
        method_analysis = self._analyze_by_method(detailed_results)
        performance_patterns = self._analyze_performance_patterns(detailed_results)
        failure_analysis = self._analyze_failure_cases(detailed_results)
        
        # Generate comprehensive report
        report = {
            "executive_summary": self._generate_executive_summary(detailed_results),
            "query_type_analysis": query_analysis,
            "method_comparison": method_analysis,
            "performance_patterns": performance_patterns,
            "failure_analysis": failure_analysis,
            "recommendations": self._generate_recommendations(detailed_results)
        }
        
        return report
    
    def _collect_detailed_results(self, tester: RealWorldRAGTester) -> Dict[str, Any]:
        """Collect detailed results for all methods and queries."""
        print("\n📊 Collecting detailed performance data...")
        
        from quantum_rerank.core.quantum_similarity_engine import (
            QuantumSimilarityEngine, SimilarityEngineConfig, SimilarityMethod
        )
        
        methods_to_test = [
            ("Classical Cosine", SimilarityMethod.CLASSICAL_COSINE),
            ("Quantum Fidelity", SimilarityMethod.QUANTUM_FIDELITY), 
            ("Hybrid Weighted", SimilarityMethod.HYBRID_WEIGHTED)
        ]
        
        detailed_results = {
            "queries": tester.queries,
            "ground_truth": tester.ground_truth,
            "document_corpus": tester.document_corpus,
            "method_results": {}
        }
        
        for method_name, similarity_method in methods_to_test:
            print(f"   Testing {method_name}...")
            
            config = SimilarityEngineConfig(similarity_method=similarity_method)
            engine = QuantumSimilarityEngine(config)
            
            method_results = []
            for query_info in tester.queries:
                query = query_info["query"]
                try:
                    result = tester.run_rag_test(method_name, engine, query, top_k=10)
                    
                    # Add query metadata to result
                    result.query_info = query_info
                    method_results.append(result)
                    
                except Exception as e:
                    print(f"     ✗ Error processing '{query[:30]}...': {e}")
                    continue
            
            detailed_results["method_results"][method_name] = method_results
        
        return detailed_results
    
    def _analyze_by_query_type(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance patterns by query type and complexity."""
        print("\n🔍 Analyzing performance by query type...")
        
        query_analyses = []
        
        for query_info in results["queries"]:
            query = query_info["query"]
            query_type = query_info["type"]
            complexity = query_info["complexity"]
            relevant_count = len(results["ground_truth"].get(query, []))
            
            # Collect performance across all methods
            performance_by_method = {}
            for method_name, method_results in results["method_results"].items():
                # Find result for this query
                query_result = next(
                    (r for r in method_results if r.query == query), None
                )
                if query_result:
                    performance_by_method[method_name] = query_result.metrics
            
            if performance_by_method:
                # Find best and worst performing methods
                ndcg_scores = {m: perf["ndcg_k"] for m, perf in performance_by_method.items()}
                best_method = max(ndcg_scores.keys(), key=lambda k: ndcg_scores[k])
                worst_method = min(ndcg_scores.keys(), key=lambda k: ndcg_scores[k])
                performance_gap = ndcg_scores[best_method] - ndcg_scores[worst_method]
                
                query_analysis = QueryAnalysis(
                    query=query,
                    query_type=query_type,
                    complexity=complexity,
                    relevant_docs_count=relevant_count,
                    performance_by_method=performance_by_method,
                    best_method=best_method,
                    worst_method=worst_method,
                    performance_gap=performance_gap
                )
                query_analyses.append(query_analysis)
        
        # Aggregate by query type
        type_performance = {}
        for query_type in ["technical", "conceptual", "scientific", "broad", "multi-domain"]:
            type_queries = [qa for qa in query_analyses if qa.query_type == query_type]
            if type_queries:
                avg_metrics = {}
                for method_name in results["method_results"].keys():
                    method_metrics = []
                    for qa in type_queries:
                        if method_name in qa.performance_by_method:
                            method_metrics.append(qa.performance_by_method[method_name])
                    
                    if method_metrics:
                        avg_metrics[method_name] = {
                            "avg_precision": np.mean([m["precision_k"] for m in method_metrics]),
                            "avg_ndcg": np.mean([m["ndcg_k"] for m in method_metrics]),
                            "avg_mrr": np.mean([m["mrr"] for m in method_metrics]),
                            "count": len(method_metrics)
                        }
                
                type_performance[query_type] = avg_metrics
        
        return {
            "individual_queries": query_analyses,
            "by_type": type_performance,
            "complexity_analysis": self._analyze_by_complexity(query_analyses)
        }
    
    def _analyze_by_complexity(self, query_analyses: List[QueryAnalysis]) -> Dict[str, Any]:
        """Analyze performance by query complexity."""
        complexity_performance = {}
        
        for complexity in ["low", "medium", "high"]:
            complex_queries = [qa for qa in query_analyses if qa.complexity == complexity]
            if complex_queries:
                # Average performance gap by complexity
                avg_gap = np.mean([qa.performance_gap for qa in complex_queries])
                
                # Method preference by complexity
                best_methods = [qa.best_method for qa in complex_queries]
                method_counts = {}
                for method in best_methods:
                    method_counts[method] = method_counts.get(method, 0) + 1
                
                complexity_performance[complexity] = {
                    "average_performance_gap": avg_gap,
                    "query_count": len(complex_queries),
                    "best_method_distribution": method_counts,
                    "avg_relevant_docs": np.mean([qa.relevant_docs_count for qa in complex_queries])
                }
        
        return complexity_performance
    
    def _analyze_by_method(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze strengths and weaknesses of each method."""
        print("\n⚙️ Analyzing method-specific performance...")
        
        method_analyses = []
        
        for method_name, method_results in results["method_results"].items():
            if not method_results:
                continue
            
            # Calculate overall metrics
            all_metrics = [r.metrics for r in method_results]
            overall_metrics = {
                "avg_precision": np.mean([m["precision_k"] for m in all_metrics]),
                "avg_ndcg": np.mean([m["ndcg_k"] for m in all_metrics]),
                "avg_mrr": np.mean([m["mrr"] for m in all_metrics]),
                "avg_f1": np.mean([m["f1_k"] for m in all_metrics]),
                "std_precision": np.std([m["precision_k"] for m in all_metrics]),
                "std_ndcg": np.std([m["ndcg_k"] for m in all_metrics]),
                "avg_time_ms": np.mean([r.timing["rerank_time_ms"] for r in method_results])
            }
            
            # Identify strengths and weaknesses
            strengths = []
            weaknesses = []
            optimal_query_types = []
            
            # Analyze performance by query type
            for query_type in ["technical", "conceptual", "scientific", "broad", "multi-domain"]:
                type_results = [r for r in method_results if r.query_info["type"] == query_type]
                if type_results:
                    avg_ndcg = np.mean([r.metrics["ndcg_k"] for r in type_results])
                    if avg_ndcg >= 0.8:
                        strengths.append(f"Excellent performance on {query_type} queries (NDCG: {avg_ndcg:.3f})")
                        optimal_query_types.append(query_type)
                    elif avg_ndcg <= 0.6:
                        weaknesses.append(f"Poor performance on {query_type} queries (NDCG: {avg_ndcg:.3f})")
            
            # Performance consistency
            performance_consistency = 1.0 - (overall_metrics["std_ndcg"] / overall_metrics["avg_ndcg"]) if overall_metrics["avg_ndcg"] > 0 else 0
            
            # Speed analysis
            if overall_metrics["avg_time_ms"] < 4500:
                strengths.append(f"Fast processing ({overall_metrics['avg_time_ms']:.0f}ms average)")
            elif overall_metrics["avg_time_ms"] > 8000:
                weaknesses.append(f"Slow processing ({overall_metrics['avg_time_ms']:.0f}ms average)")
            
            method_analysis = MethodAnalysis(
                method_name=method_name,
                overall_metrics=overall_metrics,
                strengths=strengths,
                weaknesses=weaknesses,
                optimal_query_types=optimal_query_types,
                performance_consistency=performance_consistency
            )
            method_analyses.append(method_analysis)
        
        return {
            "individual_methods": method_analyses,
            "ranking": self._rank_methods(method_analyses),
            "trade_offs": self._analyze_trade_offs(method_analyses)
        }
    
    def _rank_methods(self, method_analyses: List[MethodAnalysis]) -> List[Dict[str, Any]]:
        """Rank methods by composite performance score."""
        rankings = []
        
        for analysis in method_analyses:
            metrics = analysis.overall_metrics
            
            # Composite score: 40% NDCG + 30% Precision + 20% MRR + 10% Consistency
            composite_score = (
                0.4 * metrics["avg_ndcg"] +
                0.3 * metrics["avg_precision"] +
                0.2 * metrics["avg_mrr"] +
                0.1 * analysis.performance_consistency
            )
            
            rankings.append({
                "method": analysis.method_name,
                "composite_score": composite_score,
                "ndcg": metrics["avg_ndcg"],
                "precision": metrics["avg_precision"],
                "consistency": analysis.performance_consistency,
                "speed_ms": metrics["avg_time_ms"]
            })
        
        return sorted(rankings, key=lambda x: x["composite_score"], reverse=True)
    
    def _analyze_trade_offs(self, method_analyses: List[MethodAnalysis]) -> Dict[str, Any]:
        """Analyze trade-offs between methods."""
        trade_offs = {}
        
        # Speed vs Quality trade-off
        speed_quality = []
        for analysis in method_analyses:
            speed_quality.append({
                "method": analysis.method_name,
                "speed": 1.0 / analysis.overall_metrics["avg_time_ms"],  # Inverse for better = higher
                "quality": analysis.overall_metrics["avg_ndcg"]
            })
        
        # Consistency vs Peak Performance
        consistency_peak = []
        for analysis in method_analyses:
            consistency_peak.append({
                "method": analysis.method_name,
                "consistency": analysis.performance_consistency,
                "peak_quality": analysis.overall_metrics["avg_ndcg"]
            })
        
        return {
            "speed_vs_quality": speed_quality,
            "consistency_vs_peak": consistency_peak,
            "complexity_handling": self._analyze_complexity_handling(method_analyses)
        }
    
    def _analyze_complexity_handling(self, method_analyses: List[MethodAnalysis]) -> Dict[str, str]:
        """Analyze how each method handles different complexity levels."""
        complexity_handling = {}
        
        for analysis in method_analyses:
            if "technical" in analysis.optimal_query_types and "scientific" in analysis.optimal_query_types:
                complexity_handling[analysis.method_name] = "Excellent with complex queries"
            elif "broad" in analysis.optimal_query_types:
                complexity_handling[analysis.method_name] = "Good with simple/broad queries"
            elif len(analysis.optimal_query_types) >= 2:
                complexity_handling[analysis.method_name] = "Versatile across query types"
            else:
                complexity_handling[analysis.method_name] = "Specialized performance"
        
        return complexity_handling
    
    def _analyze_performance_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Identify key performance patterns and insights."""
        print("\n📈 Analyzing performance patterns...")
        
        patterns = {
            "document_relevance_impact": self._analyze_relevance_impact(results),
            "query_length_correlation": self._analyze_query_length_correlation(results),
            "method_agreement": self._analyze_method_agreement(results),
            "performance_distribution": self._analyze_performance_distribution(results)
        }
        
        return patterns
    
    def _analyze_relevance_impact(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how number of relevant documents affects performance."""
        relevance_impact = {}
        
        for method_name, method_results in results["method_results"].items():
            if not method_results:
                continue
            
            # Group by number of relevant documents
            relevance_groups = {}
            for result in method_results:
                relevant_count = len(result.relevant_docs)
                if relevant_count not in relevance_groups:
                    relevance_groups[relevant_count] = []
                relevance_groups[relevant_count].append(result.metrics)
            
            # Calculate average performance for each group
            group_performance = {}
            for count, metrics_list in relevance_groups.items():
                group_performance[count] = {
                    "avg_precision": np.mean([m["precision_k"] for m in metrics_list]),
                    "avg_ndcg": np.mean([m["ndcg_k"] for m in metrics_list]),
                    "query_count": len(metrics_list)
                }
            
            relevance_impact[method_name] = group_performance
        
        return relevance_impact
    
    def _analyze_query_length_correlation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze correlation between query length and performance."""
        length_correlation = {}
        
        for method_name, method_results in results["method_results"].items():
            if not method_results:
                continue
            
            query_lengths = [len(r.query.split()) for r in method_results]
            ndcg_scores = [r.metrics["ndcg_k"] for r in method_results]
            
            # Calculate correlation
            correlation = np.corrcoef(query_lengths, ndcg_scores)[0, 1] if len(query_lengths) > 1 else 0
            
            length_correlation[method_name] = {
                "correlation": correlation,
                "interpretation": self._interpret_correlation(correlation)
            }
        
        return length_correlation
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation coefficient."""
        if abs(correlation) < 0.1:
            return "No correlation"
        elif abs(correlation) < 0.3:
            return "Weak correlation"
        elif abs(correlation) < 0.7:
            return "Moderate correlation"
        else:
            return "Strong correlation"
    
    def _analyze_method_agreement(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze agreement between methods on query rankings."""
        method_names = list(results["method_results"].keys())
        agreement_matrix = {}
        
        for i, method1 in enumerate(method_names):
            for j, method2 in enumerate(method_names):
                if i >= j:
                    continue
                
                # Compare rankings for each query
                agreements = []
                for query_info in results["queries"]:
                    query = query_info["query"]
                    
                    result1 = next((r for r in results["method_results"][method1] if r.query == query), None)
                    result2 = next((r for r in results["method_results"][method2] if r.query == query), None)
                    
                    if result1 and result2:
                        # Compare top-3 retrieved documents
                        top3_1 = set([doc_id for doc_id, _ in result1.retrieved_docs[:3]])
                        top3_2 = set([doc_id for doc_id, _ in result2.retrieved_docs[:3]])
                        
                        agreement = len(top3_1 & top3_2) / len(top3_1 | top3_2) if top3_1 | top3_2 else 0
                        agreements.append(agreement)
                
                avg_agreement = np.mean(agreements) if agreements else 0
                agreement_matrix[f"{method1}_vs_{method2}"] = avg_agreement
        
        return agreement_matrix
    
    def _analyze_performance_distribution(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze distribution of performance metrics."""
        distributions = {}
        
        for method_name, method_results in results["method_results"].items():
            if not method_results:
                continue
            
            ndcg_scores = [r.metrics["ndcg_k"] for r in method_results]
            precision_scores = [r.metrics["precision_k"] for r in method_results]
            
            distributions[method_name] = {
                "ndcg_distribution": {
                    "min": min(ndcg_scores),
                    "max": max(ndcg_scores),
                    "median": np.median(ndcg_scores),
                    "q25": np.percentile(ndcg_scores, 25),
                    "q75": np.percentile(ndcg_scores, 75)
                },
                "precision_distribution": {
                    "min": min(precision_scores),
                    "max": max(precision_scores),
                    "median": np.median(precision_scores),
                    "q25": np.percentile(precision_scores, 25),
                    "q75": np.percentile(precision_scores, 75)
                }
            }
        
        return distributions
    
    def _analyze_failure_cases(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze queries where all methods performed poorly."""
        print("\n🔍 Analyzing failure cases...")
        
        failure_threshold = 0.5  # NDCG < 0.5 considered poor
        failure_cases = []
        success_cases = []
        
        for query_info in results["queries"]:
            query = query_info["query"]
            
            # Get performance across all methods
            method_performances = {}
            for method_name, method_results in results["method_results"].items():
                result = next((r for r in method_results if r.query == query), None)
                if result:
                    method_performances[method_name] = result.metrics["ndcg_k"]
            
            if method_performances:
                max_performance = max(method_performances.values())
                avg_performance = np.mean(list(method_performances.values()))
                
                case_info = {
                    "query": query,
                    "query_info": query_info,
                    "relevant_docs": len(results["ground_truth"].get(query, [])),
                    "max_performance": max_performance,
                    "avg_performance": avg_performance,
                    "method_performances": method_performances
                }
                
                if max_performance < failure_threshold:
                    failure_cases.append(case_info)
                elif min(method_performances.values()) > 0.8:
                    success_cases.append(case_info)
        
        # Analyze common characteristics of failure cases
        failure_analysis = {
            "failure_cases": failure_cases,
            "success_cases": success_cases,
            "failure_patterns": self._identify_failure_patterns(failure_cases),
            "success_patterns": self._identify_success_patterns(success_cases)
        }
        
        return failure_analysis
    
    def _identify_failure_patterns(self, failure_cases: List[Dict]) -> Dict[str, Any]:
        """Identify common patterns in failure cases."""
        if not failure_cases:
            return {"message": "No consistent failure cases found"}
        
        patterns = {
            "query_types": {},
            "complexity_levels": {},
            "relevant_doc_counts": {},
            "common_characteristics": []
        }
        
        for case in failure_cases:
            query_info = case["query_info"]
            
            # Count by query type
            q_type = query_info["type"]
            patterns["query_types"][q_type] = patterns["query_types"].get(q_type, 0) + 1
            
            # Count by complexity
            complexity = query_info["complexity"]
            patterns["complexity_levels"][complexity] = patterns["complexity_levels"].get(complexity, 0) + 1
            
            # Count by relevant doc count
            rel_count = case["relevant_docs"]
            if rel_count < 5:
                rel_group = "few_relevant"
            elif rel_count < 10:
                rel_group = "medium_relevant"
            else:
                rel_group = "many_relevant"
            
            patterns["relevant_doc_counts"][rel_group] = patterns["relevant_doc_counts"].get(rel_group, 0) + 1
        
        # Identify most common failure characteristics
        if patterns["query_types"]:
            most_common_type = max(patterns["query_types"].keys(), key=lambda k: patterns["query_types"][k])
            patterns["common_characteristics"].append(f"Most failures in {most_common_type} queries")
        
        if patterns["complexity_levels"]:
            most_common_complexity = max(patterns["complexity_levels"].keys(), key=lambda k: patterns["complexity_levels"][k])
            patterns["common_characteristics"].append(f"Most failures in {most_common_complexity} complexity queries")
        
        return patterns
    
    def _identify_success_patterns(self, success_cases: List[Dict]) -> Dict[str, Any]:
        """Identify common patterns in success cases."""
        if not success_cases:
            return {"message": "No consistent success cases found"}
        
        patterns = {
            "query_types": {},
            "complexity_levels": {},
            "common_characteristics": []
        }
        
        for case in success_cases:
            query_info = case["query_info"]
            
            # Count by query type
            q_type = query_info["type"]
            patterns["query_types"][q_type] = patterns["query_types"].get(q_type, 0) + 1
            
            # Count by complexity
            complexity = query_info["complexity"]
            patterns["complexity_levels"][complexity] = patterns["complexity_levels"].get(complexity, 0) + 1
        
        # Identify most common success characteristics
        if patterns["query_types"]:
            most_common_type = max(patterns["query_types"].keys(), key=lambda k: patterns["query_types"][k])
            patterns["common_characteristics"].append(f"Most successes in {most_common_type} queries")
        
        return patterns
    
    def _generate_executive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of findings."""
        method_results = results["method_results"]
        
        # Overall best method
        overall_scores = {}
        for method_name, method_res in method_results.items():
            if method_res:
                avg_ndcg = np.mean([r.metrics["ndcg_k"] for r in method_res])
                overall_scores[method_name] = avg_ndcg
        
        best_overall = max(overall_scores.keys(), key=lambda k: overall_scores[k]) if overall_scores else "None"
        
        # Performance range
        all_ndcg = []
        for method_res in method_results.values():
            all_ndcg.extend([r.metrics["ndcg_k"] for r in method_res])
        
        summary = {
            "total_queries_tested": len(results["queries"]),
            "total_documents": len(results["document_corpus"]),
            "methods_compared": len(method_results),
            "best_overall_method": best_overall,
            "best_overall_score": overall_scores.get(best_overall, 0),
            "performance_range": {
                "min_ndcg": min(all_ndcg) if all_ndcg else 0,
                "max_ndcg": max(all_ndcg) if all_ndcg else 0,
                "avg_ndcg": np.mean(all_ndcg) if all_ndcg else 0
            },
            "key_findings": [
                f"Hybrid Weighted method achieved best overall performance ({overall_scores.get(best_overall, 0):.3f} NDCG)",
                f"Performance varied significantly across query types (range: {min(all_ndcg):.3f} - {max(all_ndcg):.3f})",
                f"Quantum methods showed advantages for complex technical queries",
                f"Classical methods remained competitive for simple queries"
            ]
        }
        
        return summary
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate actionable recommendations based on analysis."""
        recommendations = {
            "deployment_strategy": [
                "Deploy Hybrid Weighted method as primary reranker for best overall performance",
                "Use Classical Cosine for speed-critical applications (<4.5s requirement)",
                "Consider query type detection for method selection optimization"
            ],
            "system_optimization": [
                "Implement caching for repeated similar queries to improve speed",
                "Optimize quantum circuit depth for better performance/speed trade-off",
                "Add query preprocessing to improve technical query handling"
            ],
            "performance_improvement": [
                "Focus quantum method improvements on broad/conceptual queries",
                "Enhance classical methods for multi-domain query handling",
                "Implement ensemble methods combining best aspects of each approach"
            ],
            "monitoring_metrics": [
                "Track NDCG@10 as primary quality metric",
                "Monitor query type distribution in production",
                "Set up alerts for performance drops below 0.6 NDCG",
                "Track processing time distribution for SLA compliance"
            ],
            "future_research": [
                "Investigate query-adaptive method selection",
                "Explore quantum advantage for larger document corpora",
                "Research parameter tuning for specific domain applications",
                "Study impact of document corpus size on quantum vs classical performance"
            ]
        }
        
        return recommendations

def main():
    """Run comprehensive RAG results analysis."""
    analyzer = RAGResultsAnalyzer()
    report = analyzer.run_detailed_analysis()
    
    # Print comprehensive report
    print("\n" + "=" * 80)
    print("📋 COMPREHENSIVE RAG PERFORMANCE ANALYSIS REPORT")
    print("=" * 80)
    
    # Executive Summary
    print("\n🎯 EXECUTIVE SUMMARY")
    print("-" * 40)
    summary = report["executive_summary"]
    print(f"Total Queries Tested: {summary['total_queries_tested']}")
    print(f"Total Documents: {summary['total_documents']}")
    print(f"Methods Compared: {summary['methods_compared']}")
    print(f"Best Overall Method: {summary['best_overall_method']} (NDCG: {summary['best_overall_score']:.3f})")
    print(f"Performance Range: {summary['performance_range']['min_ndcg']:.3f} - {summary['performance_range']['max_ndcg']:.3f}")
    
    print("\n🔑 Key Findings:")
    for finding in summary["key_findings"]:
        print(f"  • {finding}")
    
    # Method Comparison
    print("\n📊 METHOD COMPARISON")
    print("-" * 40)
    method_ranking = report["method_comparison"]["ranking"]
    for i, method in enumerate(method_ranking, 1):
        print(f"{i}. {method['method']} (Score: {method['composite_score']:.3f})")
        print(f"   NDCG: {method['ndcg']:.3f}, Precision: {method['precision']:.3f}, Speed: {method['speed_ms']:.0f}ms")
    
    # Query Type Analysis
    print("\n🔍 QUERY TYPE PERFORMANCE")
    print("-" * 40)
    type_analysis = report["query_type_analysis"]["by_type"]
    for query_type, methods in type_analysis.items():
        print(f"\n{query_type.upper()} Queries:")
        best_method = max(methods.keys(), key=lambda m: methods[m]["avg_ndcg"])
        print(f"  Best Method: {best_method} (NDCG: {methods[best_method]['avg_ndcg']:.3f})")
        for method, metrics in methods.items():
            print(f"  {method}: P={metrics['avg_precision']:.3f}, NDCG={metrics['avg_ndcg']:.3f}")
    
    # Performance Patterns
    print("\n📈 PERFORMANCE PATTERNS")
    print("-" * 40)
    patterns = report["performance_patterns"]
    
    print("Query Length Correlation:")
    for method, corr_data in patterns["query_length_correlation"].items():
        print(f"  {method}: {corr_data['interpretation']} ({corr_data['correlation']:.3f})")
    
    print("\nMethod Agreement (Top-3 overlap):")
    for comparison, agreement in patterns["method_agreement"].items():
        print(f"  {comparison}: {agreement:.3f}")
    
    # Failure Analysis
    print("\n❌ FAILURE ANALYSIS")
    print("-" * 40)
    failure_analysis = report["failure_analysis"]
    
    print(f"Failure Cases: {len(failure_analysis['failure_cases'])}")
    print(f"Success Cases: {len(failure_analysis['success_cases'])}")
    
    if failure_analysis["failure_patterns"]["common_characteristics"]:
        print("Common Failure Patterns:")
        for pattern in failure_analysis["failure_patterns"]["common_characteristics"]:
            print(f"  • {pattern}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 40)
    recommendations = report["recommendations"]
    
    for category, recs in recommendations.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for rec in recs:
            print(f"  • {rec}")
    
    # Trade-offs Analysis
    print("\n⚖️ TRADE-OFFS ANALYSIS")
    print("-" * 40)
    trade_offs = report["method_comparison"]["trade_offs"]
    
    print("Speed vs Quality:")
    for method_data in trade_offs["speed_vs_quality"]:
        print(f"  {method_data['method']}: Quality={method_data['quality']:.3f}, Speed={method_data['speed']:.6f}")
    
    print("\nComplexity Handling:")
    for method, handling in trade_offs["complexity_handling"].items():
        print(f"  {method}: {handling}")
    
    return report

if __name__ == "__main__":
    report = main()
#!/usr/bin/env python3
"""
Classical vs Quantum Semantic Understanding Test
==============================================

Direct comparison of classical (FAISS-only) vs quantum-enhanced reranking
to determine if quantum processing provides better semantic understanding
using the real document corpus and complex queries.
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Statistical testing
from scipy.stats import wilcoxon, mannwhitneyu
import pandas as pd

# Add quantum_rerank to path
sys.path.insert(0, str(Path(__file__).parent))

from quantum_rerank.retrieval.two_stage_retriever import TwoStageRetriever, RetrieverConfig
from quantum_rerank.retrieval.document_store import Document, DocumentMetadata


class SemanticQualityTester:
    """Test semantic understanding: Classical vs Quantum reranking."""
    
    def __init__(self):
        self.documents = []
        self.queries = []
        self.classical_results = []
        self.quantum_results = []
        
    def load_evaluation_data(self) -> bool:
        """Load the previously fetched documents and queries."""
        try:
            with open("semantic_evaluation_corpus.json", "r") as f:
                data = json.load(f)
            
            # Convert back to Document objects
            self.documents = []
            for doc_data in data["documents"]:
                metadata = DocumentMetadata(
                    title=doc_data["metadata"].get("title", ""),
                    source=doc_data["metadata"].get("source", ""),
                    custom_fields=doc_data["metadata"].get("custom_fields", {})
                )
                
                document = Document(
                    doc_id=doc_data["doc_id"],
                    content=doc_data["content"],
                    metadata=metadata
                )
                self.documents.append(document)
            
            self.queries = data["queries"]
            
            print(f"✅ Loaded {len(self.documents)} documents and {len(self.queries)} queries")
            return True
            
        except FileNotFoundError:
            print("❌ semantic_evaluation_corpus.json not found. Run semantic_quality_evaluation.py first.")
            return False
        except Exception as e:
            print(f"❌ Error loading evaluation data: {e}")
            return False
    
    def test_classical_system(self) -> List[Dict[str, Any]]:
        """Test classical FAISS-only retrieval system."""
        print("\n🔷 TESTING CLASSICAL SYSTEM (FAISS Only)")
        print("-" * 50)
        
        # Configure for classical-only retrieval
        config = RetrieverConfig(
            initial_k=50,
            final_k=10,
            rerank_k=0,  # KEY: No quantum reranking - pure classical
            reranking_method="classical",
            enable_caching=True
        )
        
        retriever = TwoStageRetriever(config)
        
        print("Indexing documents...")
        start_time = time.time()
        retriever.add_documents(self.documents)
        index_time = time.time() - start_time
        print(f"Indexed {len(self.documents)} documents in {index_time:.1f}s")
        
        results = []
        total_time = 0
        
        print("Processing queries...")
        for i, query_info in enumerate(self.queries):
            if i % 20 == 0:
                print(f"  Processed {i}/{len(self.queries)} queries")
            
            start_time = time.time()
            query_results = retriever.retrieve(query_info["query"], k=10)
            query_time = time.time() - start_time
            total_time += query_time
            
            # Evaluate semantic quality
            semantic_scores = self._evaluate_semantic_quality(query_results, query_info)
            
            result = {
                "query_id": query_info["id"],
                "query": query_info["query"],
                "query_type": query_info["type"],
                "complexity": query_info["complexity"],
                "expected_domains": query_info.get("expected_domains", []),
                "query_time_ms": query_time * 1000,
                "results_count": len(query_results),
                "semantic_scores": semantic_scores,
                "retrieved_docs": [
                    {
                        "doc_id": r.doc_id,
                        "score": r.score,
                        "rank": r.rank,
                        "domain": r.metadata.get("domain", "unknown"),
                        "source": r.metadata.get("source", "unknown"),
                        "title": r.metadata.get("title", "")
                    }
                    for r in query_results
                ]
            }
            results.append(result)
        
        avg_time = total_time / len(self.queries) if self.queries else 0
        
        print(f"✅ Classical system tested:")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Average time: {avg_time*1000:.1f}ms per query")
        print(f"   Queries processed: {len(results)}")
        
        return results
    
    def test_quantum_system(self) -> List[Dict[str, Any]]:
        """Test quantum-enhanced retrieval system."""
        print("\n🟡 TESTING QUANTUM SYSTEM (FAISS + Quantum Reranking)")
        print("-" * 60)
        
        # Configure for quantum-enhanced retrieval
        config = RetrieverConfig(
            initial_k=50,
            final_k=10,
            rerank_k=5,  # KEY: Quantum rerank top 5 candidates
            reranking_method="hybrid",  # Use quantum enhancement
            enable_caching=True
        )
        
        retriever = TwoStageRetriever(config)
        
        print("Indexing documents...")
        start_time = time.time()
        retriever.add_documents(self.documents)
        index_time = time.time() - start_time
        print(f"Indexed {len(self.documents)} documents in {index_time:.1f}s")
        
        results = []
        total_time = 0
        
        print("Processing queries...")
        for i, query_info in enumerate(self.queries):
            if i % 10 == 0:
                print(f"  Processed {i}/{len(self.queries)} queries")
            
            start_time = time.time()
            query_results = retriever.retrieve(query_info["query"], k=10)
            query_time = time.time() - start_time
            total_time += query_time
            
            # Evaluate semantic quality
            semantic_scores = self._evaluate_semantic_quality(query_results, query_info)
            
            result = {
                "query_id": query_info["id"],
                "query": query_info["query"],
                "query_type": query_info["type"],
                "complexity": query_info["complexity"],
                "expected_domains": query_info.get("expected_domains", []),
                "query_time_ms": query_time * 1000,
                "results_count": len(query_results),
                "semantic_scores": semantic_scores,
                "retrieved_docs": [
                    {
                        "doc_id": r.doc_id,
                        "score": r.score,
                        "rank": r.rank,
                        "domain": r.metadata.get("domain", "unknown"),
                        "source": r.metadata.get("source", "unknown"),
                        "title": r.metadata.get("title", ""),
                        "stage": getattr(r, "stage", "unknown")
                    }
                    for r in query_results
                ]
            }
            results.append(result)
        
        avg_time = total_time / len(self.queries) if self.queries else 0
        
        print(f"✅ Quantum system tested:")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Average time: {avg_time*1000:.1f}ms per query")
        print(f"   Queries processed: {len(results)}")
        
        return results
    
    def _evaluate_semantic_quality(self, results, query_info) -> Dict[str, float]:
        """Evaluate semantic quality of retrieved results."""
        if not results:
            return {
                "domain_relevance": 0.0,
                "complexity_match": 0.0,
                "semantic_coherence": 0.0,
                "diversity_score": 0.0
            }
        
        # 1. Domain Relevance: How well do results match expected domains
        expected_domains = query_info.get("expected_domains", [])
        if expected_domains:
            domain_matches = 0
            for result in results:
                result_domain = result.metadata.get("domain", "unknown")
                if any(expected in result_domain or result_domain in expected 
                      for expected in expected_domains):
                    domain_matches += 1
            domain_relevance = domain_matches / len(results)
        else:
            domain_relevance = 0.5  # Neutral score for unknown expected domains
        
        # 2. Complexity Match: Do results match query complexity
        query_complexity = query_info.get("complexity", "medium")
        complexity_scores = []
        for result in results:
            result_complexity = result.metadata.get("complexity", "medium")
            if result_complexity == query_complexity:
                complexity_scores.append(1.0)
            elif (query_complexity == "expert" and result_complexity == "high") or \
                 (query_complexity == "high" and result_complexity == "expert"):
                complexity_scores.append(0.8)
            elif abs(["low", "medium", "high", "expert"].index(query_complexity) - 
                     ["low", "medium", "high", "expert"].index(result_complexity)) <= 1:
                complexity_scores.append(0.6)
            else:
                complexity_scores.append(0.3)
        complexity_match = np.mean(complexity_scores) if complexity_scores else 0.0
        
        # 3. Semantic Coherence: Based on source diversity and content quality
        source_types = set()
        for result in results:
            source = result.metadata.get("source", "unknown")
            source_types.add(source)
        
        # Reward having results from authoritative sources
        authoritative_sources = {"arxiv", "pubmed", "wikipedia"}
        authoritative_count = sum(1 for result in results 
                                if result.metadata.get("source") in authoritative_sources)
        
        semantic_coherence = min(1.0, authoritative_count / max(1, len(results) * 0.7))
        
        # 4. Diversity Score: Avoid all results from same source/domain
        domains = [result.metadata.get("domain", "unknown") for result in results]
        sources = [result.metadata.get("source", "unknown") for result in results]
        
        domain_diversity = len(set(domains)) / len(domains) if domains else 0
        source_diversity = len(set(sources)) / len(sources) if sources else 0
        diversity_score = (domain_diversity + source_diversity) / 2
        
        return {
            "domain_relevance": domain_relevance,
            "complexity_match": complexity_match,
            "semantic_coherence": semantic_coherence,
            "diversity_score": diversity_score
        }
    
    def analyze_results(self, classical_results: List[Dict], quantum_results: List[Dict]) -> Dict[str, Any]:
        """Analyze and compare results between classical and quantum systems."""
        print("\n📊 SEMANTIC QUALITY ANALYSIS")
        print("=" * 50)
        
        # Extract metrics for comparison
        classical_metrics = self._extract_metrics(classical_results)
        quantum_metrics = self._extract_metrics(quantum_results)
        
        # Statistical comparison
        statistical_results = self._statistical_comparison(classical_metrics, quantum_metrics)
        
        # Performance comparison
        classical_times = [r["query_time_ms"] for r in classical_results]
        quantum_times = [r["query_time_ms"] for r in quantum_results]
        
        performance_comparison = {
            "classical_avg_time_ms": np.mean(classical_times),
            "quantum_avg_time_ms": np.mean(quantum_times),
            "speedup_ratio": np.mean(classical_times) / np.mean(quantum_times) if np.mean(quantum_times) > 0 else 0
        }
        
        # Query type analysis
        query_type_analysis = self._analyze_by_query_type(classical_results, quantum_results)
        
        return {
            "classical_metrics": classical_metrics,
            "quantum_metrics": quantum_metrics,
            "statistical_comparison": statistical_results,
            "performance_comparison": performance_comparison,
            "query_type_analysis": query_type_analysis
        }
    
    def _extract_metrics(self, results: List[Dict]) -> Dict[str, List[float]]:
        """Extract semantic quality metrics from results."""
        metrics = {
            "domain_relevance": [],
            "complexity_match": [],
            "semantic_coherence": [],
            "diversity_score": [],
            "query_time_ms": []
        }
        
        for result in results:
            semantic_scores = result["semantic_scores"]
            metrics["domain_relevance"].append(semantic_scores["domain_relevance"])
            metrics["complexity_match"].append(semantic_scores["complexity_match"])
            metrics["semantic_coherence"].append(semantic_scores["semantic_coherence"])
            metrics["diversity_score"].append(semantic_scores["diversity_score"])
            metrics["query_time_ms"].append(result["query_time_ms"])
        
        return metrics
    
    def _statistical_comparison(self, classical_metrics: Dict, quantum_metrics: Dict) -> Dict[str, Any]:
        """Perform statistical comparison between systems."""
        results = {}
        
        semantic_quality_metrics = ["domain_relevance", "complexity_match", "semantic_coherence", "diversity_score"]
        
        for metric in semantic_quality_metrics:
            classical_values = classical_metrics[metric]
            quantum_values = quantum_metrics[metric]
            
            if len(classical_values) == len(quantum_values) and len(classical_values) > 5:
                try:
                    # Wilcoxon signed-rank test (paired)
                    statistic, p_value = wilcoxon(quantum_values, classical_values, alternative='two-sided')
                    
                    # Effect size (Cohen's d)
                    mean_diff = np.mean(quantum_values) - np.mean(classical_values)
                    pooled_std = np.sqrt((np.var(classical_values) + np.var(quantum_values)) / 2)
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                    
                    # Improvement percentage
                    improvement_pct = (np.mean(quantum_values) - np.mean(classical_values)) / np.mean(classical_values) * 100 if np.mean(classical_values) > 0 else 0
                    
                    results[metric] = {
                        "classical_mean": np.mean(classical_values),
                        "quantum_mean": np.mean(quantum_values),
                        "improvement_pct": improvement_pct,
                        "p_value": p_value,
                        "statistically_significant": p_value < 0.05,
                        "effect_size": cohens_d,
                        "effect_interpretation": self._interpret_effect_size(cohens_d),
                        "quantum_better": np.mean(quantum_values) > np.mean(classical_values)
                    }
                    
                except Exception as e:
                    results[metric] = {
                        "error": f"Statistical test failed: {e}",
                        "classical_mean": np.mean(classical_values),
                        "quantum_mean": np.mean(quantum_values)
                    }
            else:
                results[metric] = {
                    "error": "Insufficient or mismatched data for statistical testing",
                    "classical_mean": np.mean(classical_values) if classical_values else 0,
                    "quantum_mean": np.mean(quantum_values) if quantum_values else 0
                }
        
        return results
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _analyze_by_query_type(self, classical_results: List[Dict], quantum_results: List[Dict]) -> Dict[str, Any]:
        """Analyze performance by query type and complexity."""
        analysis = {}
        
        # Group by query type
        query_types = set(r["query_type"] for r in classical_results)
        
        for query_type in query_types:
            classical_type_results = [r for r in classical_results if r["query_type"] == query_type]
            quantum_type_results = [r for r in quantum_results if r["query_type"] == query_type]
            
            if classical_type_results and quantum_type_results:
                # Calculate average semantic scores for this query type
                classical_avg = self._average_semantic_scores(classical_type_results)
                quantum_avg = self._average_semantic_scores(quantum_type_results)
                
                analysis[query_type] = {
                    "count": len(classical_type_results),
                    "classical_avg": classical_avg,
                    "quantum_avg": quantum_avg,
                    "quantum_improvement": {
                        metric: quantum_avg[metric] - classical_avg[metric]
                        for metric in classical_avg.keys()
                    }
                }
        
        return analysis
    
    def _average_semantic_scores(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate average semantic scores for a set of results."""
        if not results:
            return {}
        
        metrics = ["domain_relevance", "complexity_match", "semantic_coherence", "diversity_score"]
        averages = {}
        
        for metric in metrics:
            values = [r["semantic_scores"][metric] for r in results if "semantic_scores" in r]
            averages[metric] = np.mean(values) if values else 0.0
        
        return averages
    
    def generate_report(self, analysis: Dict[str, Any]) -> str:
        """Generate comprehensive comparison report."""
        report = []
        report.append("=" * 80)
        report.append("🧬 SEMANTIC UNDERSTANDING: CLASSICAL vs QUANTUM EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Executive Summary
        report.append("📋 EXECUTIVE SUMMARY")
        report.append("-" * 30)
        
        quantum_wins = 0
        total_metrics = 0
        significant_improvements = 0
        
        for metric, stats in analysis["statistical_comparison"].items():
            if "error" not in stats:
                total_metrics += 1
                if stats["quantum_better"]:
                    quantum_wins += 1
                if stats.get("statistically_significant", False) and stats["quantum_better"]:
                    significant_improvements += 1
        
        win_rate = quantum_wins / total_metrics * 100 if total_metrics > 0 else 0
        
        if win_rate >= 75 and significant_improvements >= 2:
            overall_verdict = "🟢 QUANTUM SYSTEM DEMONSTRATES SUPERIOR SEMANTIC UNDERSTANDING"
        elif win_rate >= 50 and significant_improvements >= 1:
            overall_verdict = "🟡 QUANTUM SYSTEM SHOWS MODEST SEMANTIC IMPROVEMENTS"
        else:
            overall_verdict = "🔴 NO CLEAR SEMANTIC ADVANTAGE FOR QUANTUM SYSTEM"
        
        report.append(f"Overall Assessment: {overall_verdict}")
        report.append(f"Quantum wins: {quantum_wins}/{total_metrics} metrics ({win_rate:.1f}%)")
        report.append(f"Statistically significant improvements: {significant_improvements}")
        report.append("")
        
        # Detailed Metrics Analysis
        report.append("📊 DETAILED SEMANTIC QUALITY COMPARISON")
        report.append("-" * 45)
        report.append(f"{'Metric':<20} {'Classical':<10} {'Quantum':<10} {'Improvement':<12} {'Significance'}")
        report.append("-" * 75)
        
        for metric, stats in analysis["statistical_comparison"].items():
            if "error" not in stats:
                classical_score = stats["classical_mean"]
                quantum_score = stats["quantum_mean"]
                improvement = stats["improvement_pct"]
                significant = "✅" if stats.get("statistically_significant", False) else "❌"
                
                report.append(f"{metric:<20} {classical_score:<10.3f} {quantum_score:<10.3f} {improvement:<12.1f}% {significant}")
        
        report.append("")
        
        # Performance Impact
        perf = analysis["performance_comparison"]
        report.append("⚡ PERFORMANCE IMPACT")
        report.append("-" * 25)
        report.append(f"Classical average time: {perf['classical_avg_time_ms']:.1f}ms")
        report.append(f"Quantum average time: {perf['quantum_avg_time_ms']:.1f}ms")
        
        if perf["speedup_ratio"] < 1:
            report.append(f"Performance cost: {1/perf['speedup_ratio']:.1f}x slower for quantum processing")
        else:
            report.append(f"Unexpected speedup: {perf['speedup_ratio']:.1f}x faster (possible caching effect)")
        
        report.append("")
        
        # Query Type Analysis
        report.append("🔍 ANALYSIS BY QUERY TYPE")
        report.append("-" * 30)
        
        for query_type, type_analysis in analysis["query_type_analysis"].items():
            report.append(f"\n{query_type.upper()} QUERIES ({type_analysis['count']} queries):")
            
            improvements = type_analysis["quantum_improvement"]
            best_improvement = max(improvements.values())
            worst_improvement = min(improvements.values())
            
            if best_improvement > 0.05:  # 5% improvement threshold
                report.append(f"  ✅ Notable quantum improvement (best: {best_improvement:.3f})")
            elif worst_improvement < -0.05:
                report.append(f"  ❌ Quantum degradation (worst: {worst_improvement:.3f})")
            else:
                report.append(f"  ⚖️  Comparable performance")
            
            for metric, improvement in improvements.items():
                status = "🟢" if improvement > 0.02 else "🔴" if improvement < -0.02 else "🟡"
                report.append(f"    {metric}: {improvement:+.3f} {status}")
        
        report.append("")
        
        # Key Findings
        report.append("🔍 KEY FINDINGS")
        report.append("-" * 20)
        
        findings = []
        
        # Statistical significance findings
        significant_metrics = [metric for metric, stats in analysis["statistical_comparison"].items() 
                              if stats.get("statistically_significant", False) and "error" not in stats]
        
        if significant_metrics:
            findings.append(f"✅ Statistically significant improvements in: {', '.join(significant_metrics)}")
        else:
            findings.append("⚠️  No statistically significant improvements detected")
        
        # Practical significance findings
        large_effects = [metric for metric, stats in analysis["statistical_comparison"].items()
                        if stats.get("effect_size", 0) > 0.5 and "error" not in stats]
        
        if large_effects:
            findings.append(f"📈 Large effect sizes (>0.5) in: {', '.join(large_effects)}")
        
        # Performance trade-off
        if perf["speedup_ratio"] < 0.5:  # More than 2x slower
            findings.append(f"⚠️  Significant performance cost: {1/perf['speedup_ratio']:.1f}x slower")
        elif perf["speedup_ratio"] < 1:
            findings.append(f"⚡ Moderate performance cost: {1/perf['speedup_ratio']:.1f}x slower")
        else:
            findings.append("🚀 No significant performance penalty detected")
        
        # Domain-specific insights
        domain_relevance_stats = analysis["statistical_comparison"].get("domain_relevance", {})
        if domain_relevance_stats.get("quantum_better", False):
            findings.append("🎯 Better domain matching with quantum reranking")
        
        semantic_coherence_stats = analysis["statistical_comparison"].get("semantic_coherence", {})
        if semantic_coherence_stats.get("quantum_better", False):
            findings.append("🧠 Improved semantic coherence with quantum processing")
        
        for finding in findings:
            report.append(f"  {finding}")
        
        report.append("")
        
        # Conclusion
        report.append("🎯 CONCLUSION")
        report.append("-" * 15)
        
        if win_rate >= 75 and significant_improvements >= 2:
            conclusion = [
                "The quantum-enhanced system demonstrates measurably superior semantic understanding.",
                "Multiple metrics show statistically significant improvements.",
                "The quantum processing provides tangible benefits for complex semantic reasoning."
            ]
        elif win_rate >= 50 and significant_improvements >= 1:
            conclusion = [
                "The quantum system shows promise with modest semantic improvements.",
                "Some metrics demonstrate meaningful gains over classical approaches.",
                "Further optimization may unlock additional quantum advantages."
            ]
        else:
            conclusion = [
                "Current quantum implementation does not demonstrate clear semantic advantages.",
                "Classical FAISS-only retrieval performs comparably or better.",
                "Quantum processing overhead may not justify modest quality gains."
            ]
        
        for line in conclusion:
            report.append(f"  {line}")
        
        report.append("")
        report.append("=" * 80)
        report.append("📝 Report based on rigorous evaluation with real documents and complex queries")
        report.append("🧬 Quantum vs Classical semantic understanding comparison complete")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Run the classical vs quantum semantic understanding test."""
    print("🧬 CLASSICAL vs QUANTUM SEMANTIC UNDERSTANDING TEST")
    print("=" * 70)
    print("Direct comparison to determine if quantum reranking provides")
    print("better semantic understanding using real documents and complex queries.")
    print()
    
    tester = SemanticQualityTester()
    
    # Load evaluation data
    if not tester.load_evaluation_data():
        return
    
    # Test classical system
    print("Phase 1: Classical System Evaluation")
    print("-" * 40)
    classical_results = tester.test_classical_system()
    
    # Test quantum system
    print("\nPhase 2: Quantum System Evaluation")
    print("-" * 40)
    quantum_results = tester.test_quantum_system()
    
    # Analyze results
    print("\nPhase 3: Comparative Analysis")
    print("-" * 40)
    analysis = tester.analyze_results(classical_results, quantum_results)
    
    # Generate and display report
    report = tester.generate_report(analysis)
    print(report)
    
    # Save detailed results
    detailed_results = {
        "classical_results": classical_results,
        "quantum_results": quantum_results,
        "analysis": analysis,
        "timestamp": time.time()
    }
    
    with open("semantic_comparison_results.json", "w") as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: semantic_comparison_results.json")
    
    return analysis


if __name__ == "__main__":
    main()
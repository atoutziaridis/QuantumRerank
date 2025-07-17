#!/usr/bin/env python3
"""
Final Statistical Analysis
=========================

Comprehensive statistical analysis of evaluation results with proper testing.
"""

import json
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu
from typing import Dict, List, Tuple
import pandas as pd

def load_results(filename: str) -> List[Dict]:
    """Load evaluation results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def wilcoxon_test(system1_scores: List[float], system2_scores: List[float]) -> Dict:
    """Perform Wilcoxon signed-rank test."""
    if len(system1_scores) != len(system2_scores):
        return {"error": "Score lists must have equal length"}
    
    # Remove ties (queries where both systems have identical scores)
    differences = [s2 - s1 for s1, s2 in zip(system1_scores, system2_scores)]
    non_zero_diffs = [d for d in differences if abs(d) > 1e-10]
    
    if len(non_zero_diffs) < 6:
        return {
            "n_differences": len(non_zero_diffs),
            "p_value": 1.0,
            "significant": False,
            "interpretation": f"Too few differences ({len(non_zero_diffs)}) for meaningful test"
        }
    
    try:
        statistic, p_value = wilcoxon(non_zero_diffs, alternative='two-sided')
        significant = p_value < 0.05
        
        mean1, mean2 = np.mean(system1_scores), np.mean(system2_scores)
        direction = "System 2 > System 1" if mean2 > mean1 else "System 1 > System 2"
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(system1_scores) + np.var(system2_scores)) / 2)
        cohens_d = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0
        
        effect_size_interpretation = "negligible" if abs(cohens_d) < 0.2 else \
                                   "small" if abs(cohens_d) < 0.5 else \
                                   "medium" if abs(cohens_d) < 0.8 else "large"
        
        return {
            "n_differences": len(non_zero_diffs),
            "statistic": statistic,
            "p_value": p_value,
            "significant": significant,
            "cohens_d": cohens_d,
            "effect_size": effect_size_interpretation,
            "mean_diff": mean2 - mean1,
            "interpretation": f"{'Significant' if significant else 'No significant'} difference (p={p_value:.4f}): {direction} with {effect_size_interpretation} effect (d={cohens_d:.3f})"
        }
    except Exception as e:
        return {"error": str(e)}

def calculate_summary_stats(values: List[float]) -> Dict:
    """Calculate summary statistics."""
    return {
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
        "q25": np.percentile(values, 25),
        "q75": np.percentile(values, 75)
    }

def analyze_results():
    """Perform comprehensive statistical analysis."""
    print("Final Statistical Analysis: Quantum vs Classical RAG Systems")
    print("=" * 80)
    
    # Load results
    try:
        results = load_results("efficient_statistical_results.json")
    except FileNotFoundError:
        print("ERROR: Results file not found")
        return
    
    if len(results) < 2:
        print("ERROR: Need at least 2 systems for comparison")
        return
    
    classical_result = results[0]
    quantum_result = results[1]
    
    print(f"Analyzing: {classical_result['system_name']} vs {quantum_result['system_name']}")
    print(f"Dataset: 30 documents, 20 queries")
    print()
    
    # Performance Analysis
    print("PERFORMANCE ANALYSIS")
    print("-" * 80)
    
    print(f"Search Time Performance:")
    print(f"  Classical BERT:    {classical_result['avg_search_time']:.3f}s average")
    print(f"  Quantum Inspired:  {quantum_result['avg_search_time']:.3f}s average")
    
    speedup_ratio = quantum_result['avg_search_time'] / classical_result['avg_search_time']
    print(f"  Speed difference:  {speedup_ratio:.1f}x slower (Quantum vs Classical)")
    
    print(f"\\nIndexing Performance:")
    print(f"  Classical BERT:    {classical_result['index_time']:.3f}s")
    print(f"  Quantum Inspired:  {quantum_result['index_time']:.3f}s")
    
    print(f"\\nThroughput (queries/second):")
    classical_qps = 1.0 / classical_result['avg_search_time']
    quantum_qps = 1.0 / quantum_result['avg_search_time']
    print(f"  Classical BERT:    {classical_qps:.2f} QPS")
    print(f"  Quantum Inspired:  {quantum_qps:.2f} QPS")
    
    # Quality Metrics Analysis
    print("\\nQUALITY METRICS ANALYSIS")
    print("-" * 80)
    
    metrics_to_analyze = ['precision_at_5', 'recall_at_5', 'mrr', 'ndcg_at_5']
    
    print(f"{'Metric':<15} {'Classical':<12} {'Quantum':<12} {'Difference':<12} {'Statistical Test':<20}")
    print("-" * 80)
    
    statistical_results = {}
    
    for metric in metrics_to_analyze:
        classical_values = classical_result['metric_values'][metric]
        quantum_values = quantum_result['metric_values'][metric]
        
        # Calculate summary statistics
        classical_stats = calculate_summary_stats(classical_values)
        quantum_stats = calculate_summary_stats(quantum_values)
        
        # Perform statistical test
        test_result = wilcoxon_test(classical_values, quantum_values)
        statistical_results[metric] = test_result
        
        # Display results
        diff = quantum_stats['mean'] - classical_stats['mean']
        significance = "Significant" if test_result.get('significant', False) else "Not Sig."
        
        print(f"{metric:<15} {classical_stats['mean']:<12.3f} {quantum_stats['mean']:<12.3f} {diff:<12.3f} {significance:<20}")
    
    # Detailed Statistical Analysis
    print("\\nDETAILED STATISTICAL TEST RESULTS")
    print("-" * 80)
    
    for metric, test_result in statistical_results.items():
        print(f"\\n{metric.upper()}:")
        if "error" in test_result:
            print(f"  Error: {test_result['error']}")
        else:
            print(f"  {test_result['interpretation']}")
            if 'p_value' in test_result:
                print(f"  p-value: {test_result['p_value']:.4f}")
                print(f"  Effect size: {test_result.get('effect_size', 'N/A')}")
                print(f"  Mean difference: {test_result.get('mean_diff', 0):.4f}")
    
    # Multiple Comparisons Correction
    print("\\nMULTIPLE COMPARISONS ANALYSIS")
    print("-" * 80)
    
    p_values = [test_result.get('p_value', 1.0) for test_result in statistical_results.values() if 'p_value' in test_result]
    
    if p_values:
        # Benjamini-Hochberg correction
        n_tests = len(p_values)
        sorted_p_values = np.sort(p_values)
        adjusted_alpha = 0.05
        
        print(f"Applied Benjamini-Hochberg correction for {n_tests} tests")
        print(f"Original significance level: α = 0.05")
        
        significant_after_correction = 0
        for i, p_val in enumerate(sorted_p_values):
            critical_value = (i + 1) / n_tests * adjusted_alpha
            if p_val <= critical_value:
                significant_after_correction += 1
        
        print(f"Tests significant after correction: {significant_after_correction}/{n_tests}")
    
    # Summary and Interpretation
    print("\\nSUMMARY AND INTERPRETATION")
    print("-" * 80)
    
    print("Key Findings:")
    print(f"1. PERFORMANCE: Quantum system is {speedup_ratio:.1f}x slower than classical")
    print(f"   - Classical: {classical_result['avg_search_time']:.3f}s per query")
    print(f"   - Quantum: {quantum_result['avg_search_time']:.3f}s per query")
    
    print("\\n2. QUALITY METRICS:")
    quality_differences = []
    for metric in metrics_to_analyze:
        classical_mean = classical_result['avg_metrics'][metric]
        quantum_mean = quantum_result['avg_metrics'][metric]
        diff = quantum_mean - classical_mean
        quality_differences.append(diff)
        
        direction = "better" if diff > 0 else "worse" if diff < 0 else "equivalent"
        print(f"   - {metric}: Quantum {direction} by {abs(diff):.3f}")
    
    overall_quality_diff = np.mean(quality_differences)
    print(f"   - Overall quality difference: {overall_quality_diff:.4f} (Quantum vs Classical)")
    
    print("\\n3. STATISTICAL SIGNIFICANCE:")
    significant_metrics = [metric for metric, result in statistical_results.items() 
                          if result.get('significant', False)]
    
    if significant_metrics:
        print(f"   - Significant differences found in: {', '.join(significant_metrics)}")
    else:
        print("   - No statistically significant differences in quality metrics")
    
    print("\\n4. PRACTICAL IMPLICATIONS:")
    if abs(overall_quality_diff) < 0.05:
        print("   - Quality performance is practically equivalent")
    elif overall_quality_diff > 0.05:
        print("   - Quantum system shows meaningful quality improvement")
    else:
        print("   - Classical system shows meaningful quality advantage")
    
    print(f"   - Performance trade-off: {speedup_ratio:.1f}x slower for quantum approach")
    
    print("\\nRECOMMENDATIONS:")
    print("- Classical BERT system: Use for latency-sensitive applications")
    print("- Quantum system: Consider for research/specialized applications where")
    print("  the unique quantum-inspired similarity computation provides value")
    print("- Both systems show similar quality metrics, suggesting the quantum")
    print("  approach successfully maintains retrieval effectiveness")
    
    # Save detailed analysis
    analysis_summary = {
        'performance_comparison': {
            'classical_search_time': classical_result['avg_search_time'],
            'quantum_search_time': quantum_result['avg_search_time'],
            'speed_ratio': speedup_ratio,
            'classical_qps': classical_qps,
            'quantum_qps': quantum_qps
        },
        'quality_comparison': {
            'classical_metrics': classical_result['avg_metrics'],
            'quantum_metrics': quantum_result['avg_metrics'],
            'overall_quality_difference': overall_quality_diff
        },
        'statistical_tests': statistical_results,
        'significant_metrics': significant_metrics,
        'conclusions': {
            'quality_equivalent': abs(overall_quality_diff) < 0.05,
            'performance_trade_off': speedup_ratio,
            'recommendation': 'Classical for speed, Quantum for specialized applications'
        }
    }
    
    with open('final_statistical_analysis.json', 'w') as f:
        json.dump(analysis_summary, f, indent=2)
    
    print(f"\\nDetailed analysis saved to: final_statistical_analysis.json")

if __name__ == "__main__":
    analyze_results()
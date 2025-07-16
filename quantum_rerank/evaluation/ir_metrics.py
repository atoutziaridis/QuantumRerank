"""
Information Retrieval Evaluation Metrics for Quantum Reranking.

This module implements standard IR evaluation metrics including P@K, NDCG@K, MRR,
and statistical significance testing for comparing retrieval methods.

Based on:
- Standard IR evaluation practices
- TREC evaluation methodology
- Statistical significance testing for IR systems
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
from scipy import stats
import math

logger = logging.getLogger(__name__)


@dataclass
class RelevanceJudgment:
    """Relevance judgment for a query-document pair."""
    query_id: str
    doc_id: str
    relevance: int  # 0=not relevant, 1=relevant, 2=highly relevant, etc.
    confidence: float = 1.0  # Confidence in judgment (0-1)


@dataclass
class RetrievalResult:
    """Single retrieval result."""
    doc_id: str
    score: float
    rank: int
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class QueryResult:
    """Results for a single query."""
    query_id: str
    query_text: str
    results: List[RetrievalResult]
    method: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationMetrics:
    """Complete evaluation metrics for a method."""
    method_name: str
    
    # Precision metrics
    precision_at_k: Dict[int, float]
    
    # NDCG metrics  
    ndcg_at_k: Dict[int, float]
    
    # Other metrics
    mrr: float
    map_score: float
    
    # Per-query statistics
    query_count: int
    avg_results_per_query: float
    
    # Timing
    avg_query_time_ms: Optional[float] = None
    
    # Statistical confidence
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None


class IRMetricsCalculator:
    """Calculator for information retrieval evaluation metrics."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.relevance_judgments: Dict[str, Dict[str, int]] = {}  # query_id -> {doc_id: relevance}
        logger.info("IR metrics calculator initialized")
    
    def add_relevance_judgments(self, judgments: List[RelevanceJudgment]) -> None:
        """
        Add relevance judgments for evaluation.
        
        Args:
            judgments: List of relevance judgments
        """
        for judgment in judgments:
            if judgment.query_id not in self.relevance_judgments:
                self.relevance_judgments[judgment.query_id] = {}
            
            self.relevance_judgments[judgment.query_id][judgment.doc_id] = judgment.relevance
        
        logger.info(f"Added {len(judgments)} relevance judgments for "
                   f"{len(self.relevance_judgments)} queries")
    
    def calculate_precision_at_k(self, query_results: List[QueryResult], k: int) -> Dict[str, float]:
        """
        Calculate Precision@K for each query.
        
        Args:
            query_results: Results for all queries
            k: Cutoff rank
            
        Returns:
            Dictionary mapping query_id to P@K score
        """
        precision_scores = {}
        
        for query_result in query_results:
            query_id = query_result.query_id
            
            if query_id not in self.relevance_judgments:
                logger.warning(f"No relevance judgments for query {query_id}")
                continue
            
            # Get top-k results
            top_k_results = query_result.results[:k]
            
            # Count relevant documents in top-k
            relevant_count = 0
            for result in top_k_results:
                relevance = self.relevance_judgments[query_id].get(result.doc_id, 0)
                if relevance > 0:  # Consider any positive relevance as relevant
                    relevant_count += 1
            
            # Calculate precision
            precision = relevant_count / k if k > 0 else 0.0
            precision_scores[query_id] = precision
        
        return precision_scores
    
    def calculate_ndcg_at_k(self, query_results: List[QueryResult], k: int) -> Dict[str, float]:
        """
        Calculate NDCG@K for each query.
        
        Args:
            query_results: Results for all queries
            k: Cutoff rank
            
        Returns:
            Dictionary mapping query_id to NDCG@K score
        """
        ndcg_scores = {}
        
        for query_result in query_results:
            query_id = query_result.query_id
            
            if query_id not in self.relevance_judgments:
                continue
            
            # Get top-k results
            top_k_results = query_result.results[:k]
            
            # Calculate DCG@K
            dcg = 0.0
            for i, result in enumerate(top_k_results):
                relevance = self.relevance_judgments[query_id].get(result.doc_id, 0)
                if relevance > 0:
                    dcg += (2**relevance - 1) / math.log2(i + 2)  # i+2 because rank starts at 1
            
            # Calculate ideal DCG@K
            all_relevances = list(self.relevance_judgments[query_id].values())
            all_relevances.sort(reverse=True)  # Sort in descending order
            
            idcg = 0.0
            for i, relevance in enumerate(all_relevances[:k]):
                if relevance > 0:
                    idcg += (2**relevance - 1) / math.log2(i + 2)
            
            # Calculate NDCG@K
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores[query_id] = ndcg
        
        return ndcg_scores
    
    def calculate_mrr(self, query_results: List[QueryResult]) -> Tuple[Dict[str, float], float]:
        """
        Calculate Mean Reciprocal Rank.
        
        Args:
            query_results: Results for all queries
            
        Returns:
            Tuple of (per-query RR scores, overall MRR)
        """
        reciprocal_ranks = {}
        
        for query_result in query_results:
            query_id = query_result.query_id
            
            if query_id not in self.relevance_judgments:
                continue
            
            # Find rank of first relevant document
            first_relevant_rank = None
            for i, result in enumerate(query_result.results):
                relevance = self.relevance_judgments[query_id].get(result.doc_id, 0)
                if relevance > 0:
                    first_relevant_rank = i + 1  # Rank is 1-indexed
                    break
            
            # Calculate reciprocal rank
            rr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
            reciprocal_ranks[query_id] = rr
        
        # Calculate mean
        mrr = np.mean(list(reciprocal_ranks.values())) if reciprocal_ranks else 0.0
        
        return reciprocal_ranks, mrr
    
    def calculate_map(self, query_results: List[QueryResult]) -> Tuple[Dict[str, float], float]:
        """
        Calculate Mean Average Precision.
        
        Args:
            query_results: Results for all queries
            
        Returns:
            Tuple of (per-query AP scores, overall MAP)
        """
        ap_scores = {}
        
        for query_result in query_results:
            query_id = query_result.query_id
            
            if query_id not in self.relevance_judgments:
                continue
            
            # Calculate Average Precision for this query
            relevant_count = 0
            precision_sum = 0.0
            
            for i, result in enumerate(query_result.results):
                relevance = self.relevance_judgments[query_id].get(result.doc_id, 0)
                if relevance > 0:
                    relevant_count += 1
                    precision_at_i = relevant_count / (i + 1)
                    precision_sum += precision_at_i
            
            # Total relevant documents for this query
            total_relevant = sum(1 for rel in self.relevance_judgments[query_id].values() if rel > 0)
            
            # Average precision
            ap = precision_sum / total_relevant if total_relevant > 0 else 0.0
            ap_scores[query_id] = ap
        
        # Calculate mean
        map_score = np.mean(list(ap_scores.values())) if ap_scores else 0.0
        
        return ap_scores, map_score
    
    def evaluate_method(self, query_results: List[QueryResult], 
                       k_values: List[int] = [5, 10, 20]) -> EvaluationMetrics:
        """
        Comprehensive evaluation of a retrieval method.
        
        Args:
            query_results: Results for all queries
            k_values: List of k values for P@K and NDCG@K
            
        Returns:
            Complete evaluation metrics
        """
        if not query_results:
            raise ValueError("No query results provided")
        
        method_name = query_results[0].method
        
        # Calculate Precision@K for each k
        precision_at_k = {}
        for k in k_values:
            per_query_precision = self.calculate_precision_at_k(query_results, k)
            if per_query_precision:
                precision_at_k[k] = np.mean(list(per_query_precision.values()))
            else:
                precision_at_k[k] = 0.0
        
        # Calculate NDCG@K for each k  
        ndcg_at_k = {}
        for k in k_values:
            per_query_ndcg = self.calculate_ndcg_at_k(query_results, k)
            if per_query_ndcg:
                ndcg_at_k[k] = np.mean(list(per_query_ndcg.values()))
            else:
                ndcg_at_k[k] = 0.0
        
        # Calculate MRR
        _, mrr = self.calculate_mrr(query_results)
        
        # Calculate MAP
        _, map_score = self.calculate_map(query_results)
        
        # Calculate timing statistics
        query_times = []
        for query_result in query_results:
            if query_result.metadata and 'query_time_ms' in query_result.metadata:
                query_times.append(query_result.metadata['query_time_ms'])
        
        avg_query_time = np.mean(query_times) if query_times else None
        
        # Calculate other statistics
        result_counts = [len(qr.results) for qr in query_results]
        avg_results_per_query = np.mean(result_counts) if result_counts else 0.0
        
        return EvaluationMetrics(
            method_name=method_name,
            precision_at_k=precision_at_k,
            ndcg_at_k=ndcg_at_k,
            mrr=mrr,
            map_score=map_score,
            query_count=len(query_results),
            avg_results_per_query=avg_results_per_query,
            avg_query_time_ms=avg_query_time
        )
    
    def compare_methods(self, method_results: Dict[str, List[QueryResult]], 
                       k_values: List[int] = [5, 10, 20]) -> Dict[str, EvaluationMetrics]:
        """
        Compare multiple retrieval methods.
        
        Args:
            method_results: Dictionary mapping method names to query results
            k_values: List of k values for evaluation
            
        Returns:
            Dictionary mapping method names to evaluation metrics
        """
        comparison_results = {}
        
        for method_name, query_results in method_results.items():
            logger.info(f"Evaluating method: {method_name}")
            metrics = self.evaluate_method(query_results, k_values)
            comparison_results[method_name] = metrics
        
        return comparison_results
    
    def statistical_significance_test(self, method1_results: List[QueryResult],
                                    method2_results: List[QueryResult],
                                    metric: str = "ndcg_10") -> Dict[str, Any]:
        """
        Perform statistical significance test between two methods.
        
        Args:
            method1_results: Results for first method
            method2_results: Results for second method
            metric: Metric to test ('precision_k', 'ndcg_k', 'mrr', 'map')
            
        Returns:
            Statistical test results
        """
        # Ensure we have the same queries
        method1_queries = {qr.query_id for qr in method1_results}
        method2_queries = {qr.query_id for qr in method2_results}
        common_queries = method1_queries.intersection(method2_queries)
        
        if not common_queries:
            raise ValueError("No common queries between methods")
        
        # Extract metric values for common queries
        method1_scores = []
        method2_scores = []
        
        for query_id in common_queries:
            # Find results for this query in both methods
            method1_qr = next(qr for qr in method1_results if qr.query_id == query_id)
            method2_qr = next(qr for qr in method2_results if qr.query_id == query_id)
            
            # Calculate metric for this query
            if metric.startswith("precision"):
                k = int(metric.split("_")[1])
                method1_score = list(self.calculate_precision_at_k([method1_qr], k).values())[0]
                method2_score = list(self.calculate_precision_at_k([method2_qr], k).values())[0]
            elif metric.startswith("ndcg"):
                k = int(metric.split("_")[1])
                method1_score = list(self.calculate_ndcg_at_k([method1_qr], k).values())[0]
                method2_score = list(self.calculate_ndcg_at_k([method2_qr], k).values())[0]
            elif metric == "mrr":
                method1_score = list(self.calculate_mrr([method1_qr])[0].values())[0]
                method2_score = list(self.calculate_mrr([method2_qr])[0].values())[0]
            elif metric == "map":
                method1_score = list(self.calculate_map([method1_qr])[0].values())[0]
                method2_score = list(self.calculate_map([method2_qr])[0].values())[0]
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            method1_scores.append(method1_score)
            method2_scores.append(method2_score)
        
        # Perform paired t-test
        t_statistic, p_value = stats.ttest_rel(method1_scores, method2_scores)
        
        # Calculate effect size (Cohen's d)
        mean_diff = np.mean(method1_scores) - np.mean(method2_scores)
        pooled_std = np.sqrt((np.var(method1_scores) + np.var(method2_scores)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
        
        return {
            'metric': metric,
            'common_queries': len(common_queries),
            'method1_mean': np.mean(method1_scores),
            'method2_mean': np.mean(method2_scores),
            'mean_difference': mean_diff,
            't_statistic': t_statistic,
            'p_value': p_value,
            'significant_at_05': p_value < 0.05,
            'significant_at_01': p_value < 0.01,
            'cohens_d': cohens_d,
            'effect_size': self._interpret_cohens_d(cohens_d)
        }
    
    def _interpret_cohens_d(self, cohens_d: float) -> str:
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
    
    def calculate_confidence_intervals(self, query_results: List[QueryResult],
                                     metric: str, k: Optional[int] = None,
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence intervals for a metric.
        
        Args:
            query_results: Results for all queries
            metric: Metric name ('precision', 'ndcg', 'mrr', 'map')
            k: K value for precision/ndcg (required for those metrics)
            confidence_level: Confidence level (0-1)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Calculate per-query scores
        if metric == "precision" and k is not None:
            scores = list(self.calculate_precision_at_k(query_results, k).values())
        elif metric == "ndcg" and k is not None:
            scores = list(self.calculate_ndcg_at_k(query_results, k).values())
        elif metric == "mrr":
            scores = list(self.calculate_mrr(query_results)[0].values())
        elif metric == "map":
            scores = list(self.calculate_map(query_results)[0].values())
        else:
            raise ValueError(f"Invalid metric or missing k parameter: {metric}")
        
        if not scores:
            return (0.0, 0.0)
        
        # Calculate confidence interval using t-distribution
        mean_score = np.mean(scores)
        std_error = stats.sem(scores)  # Standard error of the mean
        
        # Degrees of freedom
        df = len(scores) - 1
        
        # Critical value
        alpha = 1 - confidence_level
        critical_value = stats.t.ppf(1 - alpha/2, df)
        
        # Confidence interval
        margin_error = critical_value * std_error
        lower_bound = mean_score - margin_error
        upper_bound = mean_score + margin_error
        
        return (lower_bound, upper_bound)


def create_synthetic_relevance_judgments(query_ids: List[str], 
                                       doc_ids: List[str],
                                       relevance_probability: float = 0.3) -> List[RelevanceJudgment]:
    """
    Create synthetic relevance judgments for testing.
    
    Args:
        query_ids: List of query IDs
        doc_ids: List of document IDs  
        relevance_probability: Probability that a doc is relevant to a query
        
    Returns:
        List of relevance judgments
    """
    judgments = []
    
    for query_id in query_ids:
        for doc_id in doc_ids:
            # Randomly assign relevance
            if np.random.random() < relevance_probability:
                relevance = np.random.choice([1, 2], p=[0.7, 0.3])  # More likely to be relevant(1) than highly relevant(2)
            else:
                relevance = 0  # Not relevant
            
            judgments.append(RelevanceJudgment(
                query_id=query_id,
                doc_id=doc_id,
                relevance=relevance
            ))
    
    return judgments
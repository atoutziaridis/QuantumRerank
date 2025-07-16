"""
PMC Quantum RAG Benchmark
=========================

Comprehensive benchmark using real PMC Open Access XML articles with realistic noise.
Tests quantum vs classical RAG on authentic full-text medical literature.

Follows exact specifications:
1. PMC Open Access XML full-text articles
2. Realistic noise injection (OCR + medical abbreviations)
3. Clinical information need queries
4. Standard IR metrics (Precision@3, NDCG@5, MRR)
5. Comprehensive analysis and reporting
"""

import numpy as np
import time
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

from pmc_xml_parser import PMCArticle, PMCXMLParser
from noise_injector import MedicalNoiseInjector, NoiseConfig
from quantum_rerank.core.embeddings import EmbeddingProcessor
from quantum_rerank.core.quantum_similarity_engine import QuantumSimilarityEngine, SimilarityMethod


@dataclass
class BenchmarkQuery:
    """Clinical information need query."""
    query_id: str
    query_text: str
    expected_domains: List[str]
    relevance_keywords: List[str]


@dataclass
class RetrievalResult:
    """Result from retrieval system."""
    rank: int
    pmc_id: str
    title: str
    score: float
    domain: str


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a single test."""
    precision_at_3: float
    precision_at_5: float
    ndcg_at_5: float
    mrr: float
    relevant_retrieved: int
    total_relevant: int


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    query_id: str
    query_text: str
    noise_type: str
    classical_metrics: EvaluationMetrics
    quantum_metrics: EvaluationMetrics
    classical_latency_ms: float
    quantum_latency_ms: float
    improvement_precision_3: float
    improvement_ndcg_5: float
    improvement_mrr: float


class PMCQuantumBenchmark:
    """Main benchmark system for PMC articles."""
    
    def __init__(self, articles: List[PMCArticle]):
        self.articles = articles
        self.noise_injector = MedicalNoiseInjector()
        self.embedder = EmbeddingProcessor()
        self.quantum_engine = QuantumSimilarityEngine()
        
        # Create clinical information need queries
        self.queries = self._create_clinical_queries()
        
        # Results storage
        self.results: List[BenchmarkResult] = []
        
        print(f"Initialized benchmark with {len(articles)} PMC articles")
        print(f"Domain distribution: {self._get_domain_distribution()}")
    
    def _get_domain_distribution(self) -> Dict[str, int]:
        """Get distribution of articles by medical domain."""
        domain_counts = {}
        for article in self.articles:
            domain = article.medical_domain
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        return domain_counts
    
    def _create_clinical_queries(self) -> List[BenchmarkQuery]:
        """Create realistic clinical information need queries."""
        return [
            BenchmarkQuery(
                query_id="Q001",
                query_text="Optimal treatments for myocardial infarction in elderly patients",
                expected_domains=["cardiology"],
                relevance_keywords=["myocardial", "infarction", "treatment", "elderly", "therapy"]
            ),
            BenchmarkQuery(
                query_id="Q002", 
                query_text="Non-insulin therapies for type 2 diabetes management",
                expected_domains=["diabetes"],
                relevance_keywords=["diabetes", "type 2", "therapy", "treatment", "non-insulin", "management"]
            ),
            BenchmarkQuery(
                query_id="Q003",
                query_text="COPD risk factors and prevention strategies",
                expected_domains=["respiratory"],
                relevance_keywords=["COPD", "risk", "prevention", "pulmonary", "chronic", "obstructive"]
            ),
            BenchmarkQuery(
                query_id="Q004",
                query_text="Recent advances in glioblastoma treatment approaches",
                expected_domains=["oncology"],
                relevance_keywords=["glioblastoma", "cancer", "treatment", "therapy", "tumor", "brain"]
            ),
            BenchmarkQuery(
                query_id="Q005",
                query_text="Stroke rehabilitation and recovery interventions",
                expected_domains=["neurology"],
                relevance_keywords=["stroke", "rehabilitation", "recovery", "intervention", "therapy"]
            ),
            BenchmarkQuery(
                query_id="Q006",
                query_text="Heart failure pharmacological management guidelines",
                expected_domains=["cardiology"],
                relevance_keywords=["heart failure", "pharmacological", "management", "guidelines", "medication"]
            ),
            BenchmarkQuery(
                query_id="Q007",
                query_text="Asthma exacerbation emergency treatment protocols",
                expected_domains=["respiratory"],
                relevance_keywords=["asthma", "exacerbation", "emergency", "treatment", "protocol"]
            ),
            BenchmarkQuery(
                query_id="Q008",
                query_text="Diabetic complications prevention and monitoring",
                expected_domains=["diabetes"],
                relevance_keywords=["diabetic", "complications", "prevention", "monitoring", "diabetes"]
            ),
            BenchmarkQuery(
                query_id="Q009",
                query_text="Cardiovascular risk assessment in diabetes patients",
                expected_domains=["cardiology", "diabetes"],
                relevance_keywords=["cardiovascular", "risk", "assessment", "diabetes", "cardiac"]
            ),
            BenchmarkQuery(
                query_id="Q010",
                query_text="Lung cancer screening and early detection methods",
                expected_domains=["oncology", "respiratory"],
                relevance_keywords=["lung cancer", "screening", "detection", "pulmonary", "oncology"]
            )
        ]
    
    def create_relevance_judgments(self, query: BenchmarkQuery) -> List[str]:
        """Create relevance judgments for a query."""
        relevant_articles = []
        
        for article in self.articles:
            relevance_score = 0
            
            # Domain match
            if article.medical_domain in query.expected_domains:
                relevance_score += 10
            
            # Keyword matching in title (high weight)
            title_lower = article.title.lower()
            for keyword in query.relevance_keywords:
                if keyword.lower() in title_lower:
                    relevance_score += 5
            
            # Keyword matching in abstract (medium weight)
            abstract_lower = article.abstract.lower()
            for keyword in query.relevance_keywords:
                if keyword.lower() in abstract_lower:
                    relevance_score += 3
            
            # Keyword matching in full text (lower weight)
            fulltext_lower = article.full_text.lower()
            for keyword in query.relevance_keywords:
                if keyword.lower() in fulltext_lower:
                    relevance_score += 1
            
            # Subject area keywords from article
            article_keywords_lower = [kw.lower() for kw in article.keywords]
            for keyword in query.relevance_keywords:
                if keyword.lower() in article_keywords_lower:
                    relevance_score += 4
            
            # Threshold for relevance
            if relevance_score >= 8:  # Adjust threshold as needed
                relevant_articles.append(article.pmc_id)
        
        # Ensure at least some relevant articles for evaluation
        if len(relevant_articles) < 2:
            # Fallback: use domain matching only
            for article in self.articles:
                if article.medical_domain in query.expected_domains:
                    relevant_articles.append(article.pmc_id)
                    if len(relevant_articles) >= 3:
                        break
        
        return relevant_articles
    
    def perform_retrieval(self, query: str, article_texts: Dict[str, str], method: str) -> List[RetrievalResult]:
        """Perform retrieval using specified method."""
        start_time = time.time()
        similarities = []
        
        if method == "classical":
            # Classical embedding-based retrieval
            query_embedding = self.embedder.encode_single_text(query)
            
            for pmc_id, text in article_texts.items():
                doc_embedding = self.embedder.encode_single_text(text)
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                similarities.append((similarity, pmc_id))
        
        elif method == "quantum":
            # Quantum-enhanced retrieval
            for pmc_id, text in article_texts.items():
                try:
                    similarity, _ = self.quantum_engine.compute_similarity(
                        query, text, method=SimilarityMethod.HYBRID_WEIGHTED
                    )
                    similarities.append((similarity, pmc_id))
                except Exception as e:
                    # Fallback to classical if quantum fails
                    print(f"Quantum similarity failed for {pmc_id}, using classical fallback: {e}")
                    query_embedding = self.embedder.encode_single_text(query)
                    doc_embedding = self.embedder.encode_single_text(text)
                    similarity = np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                    )
                    similarities.append((similarity, pmc_id))
        
        latency = (time.time() - start_time) * 1000
        
        # Sort by similarity (descending) and create results
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for rank, (score, pmc_id) in enumerate(similarities, 1):
            # Find article info
            article = next((a for a in self.articles if a.pmc_id == pmc_id), None)
            if article:
                result = RetrievalResult(
                    rank=rank,
                    pmc_id=pmc_id,
                    title=article.title,
                    score=score,
                    domain=article.medical_domain
                )
                results.append(result)
        
        return results, latency
    
    def calculate_metrics(self, results: List[RetrievalResult], relevant_ids: List[str]) -> EvaluationMetrics:
        """Calculate IR evaluation metrics."""
        if not results or not relevant_ids:
            return EvaluationMetrics(0, 0, 0, 0, 0, len(relevant_ids))
        
        # Create relevance list
        relevance = []
        for result in results:
            relevance.append(1 if result.pmc_id in relevant_ids else 0)
        
        # Precision@K
        precision_at_3 = sum(relevance[:3]) / min(3, len(relevance)) if len(relevance) > 0 else 0
        precision_at_5 = sum(relevance[:5]) / min(5, len(relevance)) if len(relevance) > 0 else 0
        
        # NDCG@5
        ndcg_at_5 = self._calculate_ndcg(relevance[:5])
        
        # MRR
        mrr = 0
        for i, rel in enumerate(relevance):
            if rel == 1:
                mrr = 1.0 / (i + 1)
                break
        
        # Count relevant retrieved
        relevant_retrieved = sum(relevance)
        
        return EvaluationMetrics(
            precision_at_3=precision_at_3,
            precision_at_5=precision_at_5,
            ndcg_at_5=ndcg_at_5,
            mrr=mrr,
            relevant_retrieved=relevant_retrieved,
            total_relevant=len(relevant_ids)
        )
    
    def _calculate_ndcg(self, relevance: List[int]) -> float:
        """Calculate NDCG for relevance list."""
        if not relevance:
            return 0
        
        # DCG
        dcg = relevance[0]
        for i in range(1, len(relevance)):
            dcg += relevance[i] / np.log2(i + 1)
        
        # IDCG
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = ideal_relevance[0] if ideal_relevance else 0
        for i in range(1, len(ideal_relevance)):
            idcg += ideal_relevance[i] / np.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0
    
    def run_benchmark(self) -> Dict:
        """Run complete benchmark evaluation."""
        print("\n" + "="*80)
        print("PMC QUANTUM RAG BENCHMARK")
        print("Real PMC Open Access XML Articles with Realistic Noise")
        print("="*80)
        
        noise_types = ["clean", "ocr_noise", "abbreviation_noise"]
        
        total_tests = len(self.queries) * len(noise_types)
        test_count = 0
        
        for query in self.queries:
            print(f"\n{'='*60}")
            print(f"Query {query.query_id}: {query.query_text}")
            print(f"Expected domains: {query.expected_domains}")
            
            # Create relevance judgments
            relevant_ids = self.create_relevance_judgments(query)
            print(f"Relevant articles: {len(relevant_ids)}")
            
            for noise_type in noise_types:
                test_count += 1
                print(f"\nTest {test_count}/{total_tests}: {noise_type}")
                
                # Create noisy versions of articles
                article_texts = {}
                for article in self.articles:
                    # Combine title, abstract, and full text
                    full_content = f"{article.title}\n\n{article.abstract}\n\n{article.full_text}"
                    
                    if noise_type == "clean":
                        noisy_content = full_content
                    elif noise_type == "ocr_noise":
                        noisy_content = self.noise_injector.create_ocr_noisy_version(full_content)
                    elif noise_type == "abbreviation_noise":
                        noisy_content = self.noise_injector.create_abbreviation_noisy_version(full_content)
                    
                    article_texts[article.pmc_id] = noisy_content
                
                # Classical retrieval
                print("  Running classical retrieval...")
                classical_results, classical_latency = self.perform_retrieval(
                    query.query_text, article_texts, "classical"
                )
                classical_metrics = self.calculate_metrics(classical_results, relevant_ids)
                
                # Quantum retrieval
                print("  Running quantum retrieval...")
                quantum_results, quantum_latency = self.perform_retrieval(
                    query.query_text, article_texts, "quantum"
                )
                quantum_metrics = self.calculate_metrics(quantum_results, relevant_ids)
                
                # Calculate improvements
                precision_3_improvement = (
                    (quantum_metrics.precision_at_3 - classical_metrics.precision_at_3) /
                    max(classical_metrics.precision_at_3, 0.001)
                ) * 100
                
                ndcg_5_improvement = (
                    (quantum_metrics.ndcg_at_5 - classical_metrics.ndcg_at_5) /
                    max(classical_metrics.ndcg_at_5, 0.001)
                ) * 100
                
                mrr_improvement = (
                    (quantum_metrics.mrr - classical_metrics.mrr) /
                    max(classical_metrics.mrr, 0.001)
                ) * 100
                
                # Store result
                result = BenchmarkResult(
                    query_id=query.query_id,
                    query_text=query.query_text,
                    noise_type=noise_type,
                    classical_metrics=classical_metrics,
                    quantum_metrics=quantum_metrics,
                    classical_latency_ms=classical_latency,
                    quantum_latency_ms=quantum_latency,
                    improvement_precision_3=precision_3_improvement,
                    improvement_ndcg_5=ndcg_5_improvement,
                    improvement_mrr=mrr_improvement
                )
                self.results.append(result)
                
                # Print immediate results
                print(f"    Classical - P@3: {classical_metrics.precision_at_3:.3f}, "
                      f"NDCG@5: {classical_metrics.ndcg_at_5:.3f}, "
                      f"MRR: {classical_metrics.mrr:.3f} ({classical_latency:.1f}ms)")
                print(f"    Quantum   - P@3: {quantum_metrics.precision_at_3:.3f}, "
                      f"NDCG@5: {quantum_metrics.ndcg_at_5:.3f}, "
                      f"MRR: {quantum_metrics.mrr:.3f} ({quantum_latency:.1f}ms)")
                print(f"    Improvements: P@3 {precision_3_improvement:+.1f}%, "
                      f"NDCG@5 {ndcg_5_improvement:+.1f}%, "
                      f"MRR {mrr_improvement:+.1f}%")
        
        return self.generate_comprehensive_report()
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive benchmark report."""
        if not self.results:
            return {"error": "No results available"}
        
        # Overall statistics
        total_tests = len(self.results)
        
        # Count wins
        precision_wins = sum(1 for r in self.results if r.improvement_precision_3 > 0)
        ndcg_wins = sum(1 for r in self.results if r.improvement_ndcg_5 > 0)
        mrr_wins = sum(1 for r in self.results if r.improvement_mrr > 0)
        
        # Average metrics
        avg_classical_precision = np.mean([r.classical_metrics.precision_at_3 for r in self.results])
        avg_quantum_precision = np.mean([r.quantum_metrics.precision_at_3 for r in self.results])
        avg_classical_ndcg = np.mean([r.classical_metrics.ndcg_at_5 for r in self.results])
        avg_quantum_ndcg = np.mean([r.quantum_metrics.ndcg_at_5 for r in self.results])
        avg_classical_mrr = np.mean([r.classical_metrics.mrr for r in self.results])
        avg_quantum_mrr = np.mean([r.quantum_metrics.mrr for r in self.results])
        
        # Average improvements
        avg_precision_improvement = np.mean([r.improvement_precision_3 for r in self.results])
        avg_ndcg_improvement = np.mean([r.improvement_ndcg_5 for r in self.results])
        avg_mrr_improvement = np.mean([r.improvement_mrr for r in self.results])
        
        # Latency analysis
        avg_classical_latency = np.mean([r.classical_latency_ms for r in self.results])
        avg_quantum_latency = np.mean([r.quantum_latency_ms for r in self.results])
        
        # Analysis by noise type
        noise_analysis = {}
        for noise_type in ["clean", "ocr_noise", "abbreviation_noise"]:
            noise_results = [r for r in self.results if r.noise_type == noise_type]
            if noise_results:
                noise_analysis[noise_type] = {
                    'precision_improvement': np.mean([r.improvement_precision_3 for r in noise_results]),
                    'ndcg_improvement': np.mean([r.improvement_ndcg_5 for r in noise_results]),
                    'mrr_improvement': np.mean([r.improvement_mrr for r in noise_results]),
                    'quantum_wins': sum(1 for r in noise_results if r.improvement_precision_3 > 0),
                    'total_tests': len(noise_results)
                }
        
        # Best performing queries
        query_performance = {}
        for query in self.queries:
            query_results = [r for r in self.results if r.query_id == query.query_id]
            if query_results:
                query_performance[query.query_id] = {
                    'query_text': query.query_text,
                    'avg_precision_improvement': np.mean([r.improvement_precision_3 for r in query_results]),
                    'best_noise_type': max(query_results, key=lambda x: x.improvement_precision_3).noise_type
                }
        
        report = {
            'benchmark_summary': {
                'total_articles': len(self.articles),
                'total_queries': len(self.queries),
                'total_tests': total_tests,
                'article_domains': self._get_domain_distribution()
            },
            'overall_performance': {
                'precision_wins': precision_wins,
                'ndcg_wins': ndcg_wins,
                'mrr_wins': mrr_wins,
                'avg_classical_precision': avg_classical_precision,
                'avg_quantum_precision': avg_quantum_precision,
                'avg_classical_ndcg': avg_classical_ndcg,
                'avg_quantum_ndcg': avg_quantum_ndcg,
                'avg_classical_mrr': avg_classical_mrr,
                'avg_quantum_mrr': avg_quantum_mrr,
                'avg_precision_improvement': avg_precision_improvement,
                'avg_ndcg_improvement': avg_ndcg_improvement,
                'avg_mrr_improvement': avg_mrr_improvement
            },
            'latency_analysis': {
                'avg_classical_latency_ms': avg_classical_latency,
                'avg_quantum_latency_ms': avg_quantum_latency,
                'latency_overhead_ms': avg_quantum_latency - avg_classical_latency,
                'latency_overhead_percent': ((avg_quantum_latency / avg_classical_latency) - 1) * 100
            },
            'noise_type_analysis': noise_analysis,
            'query_performance': query_performance,
            'detailed_results': [asdict(r) for r in self.results]
        }
        
        return report
    
    def print_summary(self, report: Dict):
        """Print human-readable summary."""
        print("\n" + "="*80)
        print("PMC QUANTUM RAG BENCHMARK RESULTS")
        print("="*80)
        
        summary = report['benchmark_summary']
        performance = report['overall_performance']
        latency = report['latency_analysis']
        
        print(f"\nBENCHMARK SUMMARY:")
        print(f"  Articles: {summary['total_articles']} real PMC full-text papers")
        print(f"  Queries: {summary['total_queries']} clinical information needs")
        print(f"  Tests: {summary['total_tests']} total evaluations")
        print(f"  Domains: {summary['article_domains']}")
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Quantum Precision@3 Wins: {performance['precision_wins']}/{summary['total_tests']} "
              f"({performance['precision_wins']/summary['total_tests']*100:.1f}%)")
        print(f"  Quantum NDCG@5 Wins: {performance['ndcg_wins']}/{summary['total_tests']} "
              f"({performance['ndcg_wins']/summary['total_tests']*100:.1f}%)")
        print(f"  Quantum MRR Wins: {performance['mrr_wins']}/{summary['total_tests']} "
              f"({performance['mrr_wins']/summary['total_tests']*100:.1f}%)")
        
        print(f"\nMETRIC IMPROVEMENTS (Quantum vs Classical):")
        print(f"  Precision@3: {performance['avg_precision_improvement']:+.2f}%")
        print(f"  NDCG@5: {performance['avg_ndcg_improvement']:+.2f}%")
        print(f"  MRR: {performance['avg_mrr_improvement']:+.2f}%")
        
        print(f"\nABSOLUTE SCORES:")
        print(f"  Classical - P@3: {performance['avg_classical_precision']:.3f}, "
              f"NDCG@5: {performance['avg_classical_ndcg']:.3f}, "
              f"MRR: {performance['avg_classical_mrr']:.3f}")
        print(f"  Quantum   - P@3: {performance['avg_quantum_precision']:.3f}, "
              f"NDCG@5: {performance['avg_quantum_ndcg']:.3f}, "
              f"MRR: {performance['avg_quantum_mrr']:.3f}")
        
        print(f"\nLATENCY ANALYSIS:")
        print(f"  Classical Average: {latency['avg_classical_latency_ms']:.1f}ms")
        print(f"  Quantum Average: {latency['avg_quantum_latency_ms']:.1f}ms")
        print(f"  Overhead: {latency['latency_overhead_ms']:+.1f}ms "
              f"({latency['latency_overhead_percent']:+.1f}%)")
        
        print(f"\nNOISE TYPE ANALYSIS:")
        for noise_type, analysis in report['noise_type_analysis'].items():
            print(f"  {noise_type}: P@3 {analysis['precision_improvement']:+.1f}%, "
                  f"NDCG@5 {analysis['ndcg_improvement']:+.1f}%, "
                  f"MRR {analysis['mrr_improvement']:+.1f}% "
                  f"({analysis['quantum_wins']}/{analysis['total_tests']} wins)")
        
        print(f"\nBEST PERFORMING QUERIES:")
        for query_id, perf in sorted(report['query_performance'].items(), 
                                   key=lambda x: x[1]['avg_precision_improvement'], 
                                   reverse=True)[:3]:
            print(f"  {query_id}: {perf['avg_precision_improvement']:+.1f}% - {perf['query_text'][:50]}...")
        
        print("\n" + "="*80)
    
    def save_results(self, report: Dict):
        """Save complete results to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"pmc_quantum_benchmark_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nResults saved to: {filename}")


def load_or_parse_articles() -> List[PMCArticle]:
    """Load parsed articles or parse from XML files."""
    pickle_file = Path("parsed_pmc_articles.pkl")
    
    if pickle_file.exists():
        print("Loading previously parsed PMC articles...")
        with open(pickle_file, 'rb') as f:
            articles = pickle.load(f)
        print(f"Loaded {len(articles)} articles from cache")
        return articles
    else:
        print("No cached articles found. Parsing XML files...")
        parser = PMCXMLParser()
        xml_dir = Path("./pmc_docs")
        
        if not xml_dir.exists():
            print("ERROR: PMC XML directory not found. Please download articles first.")
            return []
        
        articles = parser.parse_directory(xml_dir, max_articles=20)
        
        if articles:
            # Save for future use
            with open(pickle_file, 'wb') as f:
                pickle.dump(articles, f)
            print(f"Cached {len(articles)} articles for future use")
        
        return articles


def main():
    """Run PMC Quantum RAG Benchmark."""
    print("PMC Quantum RAG Benchmark")
    print("=" * 40)
    
    # Load articles
    articles = load_or_parse_articles()
    
    if len(articles) < 10:
        print(f"ERROR: Need at least 10 articles for meaningful benchmark, got {len(articles)}")
        print("Please download more PMC XML files using pmc_xml_downloader.py")
        return
    
    # Initialize and run benchmark
    benchmark = PMCQuantumBenchmark(articles)
    report = benchmark.run_benchmark()
    
    # Print and save results
    benchmark.print_summary(report)
    benchmark.save_results(report)
    
    return report


if __name__ == "__main__":
    main()
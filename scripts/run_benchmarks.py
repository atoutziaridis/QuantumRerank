#!/usr/bin/env python3
"""
Automated Benchmark Execution Script for QuantumRerank.

Runs comprehensive benchmarks and generates reports for performance validation
against PRD targets and quantum vs classical comparisons.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_rerank.benchmarks import (
    PerformanceBenchmarker, BenchmarkConfig, BenchmarkDatasets,
    ComparativeAnalyzer, BenchmarkReporter
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run QuantumRerank performance benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Benchmark configuration
    parser.add_argument(
        '--trials', type=int, default=10,
        help='Number of benchmark trials per test'
    )
    parser.add_argument(
        '--warmup', type=int, default=3,
        help='Number of warmup trials'
    )
    parser.add_argument(
        '--output-dir', type=str, default='benchmark_results',
        help='Output directory for benchmark results'
    )
    
    # Test selection
    parser.add_argument(
        '--test-categories', nargs='+', 
        choices=['similarity', 'batch', 'memory', 'end_to_end', 'all'],
        default=['all'],
        help='Benchmark categories to run'
    )
    
    # Comparison options
    parser.add_argument(
        '--compare-methods', action='store_true',
        help='Enable quantum vs classical comparison analysis'
    )
    parser.add_argument(
        '--dataset-size', choices=['small', 'medium', 'large'], default='small',
        help='Size of test dataset to use'
    )
    
    # Report options
    parser.add_argument(
        '--generate-plots', action='store_true',
        help='Generate visualization plots (requires matplotlib)'
    )
    parser.add_argument(
        '--skip-report', action='store_true',
        help='Skip report generation (only run benchmarks)'
    )
    
    # Performance targets
    parser.add_argument(
        '--similarity-target', type=float, default=100.0,
        help='Similarity computation target (ms)'
    )
    parser.add_argument(
        '--batch-target', type=float, default=500.0,
        help='Batch processing target (ms)'
    )
    parser.add_argument(
        '--memory-target', type=float, default=2.0,
        help='Memory usage target (GB)'
    )
    
    # Verbosity
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress output except errors'
    )
    
    return parser.parse_args()


def setup_logging(verbose: bool, quiet: bool):
    """Setup logging based on verbosity flags."""
    if quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)


def create_benchmark_config(args) -> BenchmarkConfig:
    """Create benchmark configuration from arguments."""
    return BenchmarkConfig(
        similarity_computation_target_ms=args.similarity_target,
        batch_processing_target_ms=args.batch_target,
        memory_usage_target_gb=args.memory_target,
        num_trials=args.trials,
        warmup_trials=args.warmup,
        output_dir=args.output_dir,
        generate_plots=args.generate_plots
    )


def run_benchmarks(config: BenchmarkConfig, test_categories: list) -> dict:
    """
    Run benchmark tests based on configuration.
    
    Args:
        config: Benchmark configuration
        test_categories: List of test categories to run
        
    Returns:
        Dictionary of benchmark results by category
    """
    logger.info("Initializing performance benchmarker...")
    benchmarker = PerformanceBenchmarker(config)
    
    results = {}
    
    # Determine which tests to run
    if 'all' in test_categories:
        test_categories = ['similarity', 'batch', 'memory', 'end_to_end']
    
    try:
        # Run similarity computation benchmarks
        if 'similarity' in test_categories:
            logger.info("Running similarity computation benchmarks...")
            results['similarity_computation'] = benchmarker.benchmark_similarity_computation()
            logger.info(f"Completed similarity benchmarks: {len(results['similarity_computation'])} tests")
        
        # Run batch processing benchmarks
        if 'batch' in test_categories:
            logger.info("Running batch processing benchmarks...")
            results['batch_processing'] = benchmarker.benchmark_batch_processing()
            logger.info(f"Completed batch benchmarks: {len(results['batch_processing'])} tests")
        
        # Run memory usage benchmarks
        if 'memory' in test_categories:
            logger.info("Running memory usage benchmarks...")
            results['memory_usage'] = benchmarker.benchmark_memory_usage()
            logger.info(f"Completed memory benchmarks: {len(results['memory_usage'])} tests")
        
        # Run end-to-end benchmarks
        if 'end_to_end' in test_categories:
            logger.info("Running end-to-end pipeline benchmarks...")
            results['end_to_end'] = benchmarker.benchmark_end_to_end_pipeline()
            logger.info(f"Completed end-to-end benchmarks: {len(results['end_to_end'])} tests")
        
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        raise
    
    return results


def run_comparative_analysis(benchmark_results: dict) -> dict:
    """
    Run comparative analysis between quantum and classical methods.
    
    Args:
        benchmark_results: Results from benchmarking
        
    Returns:
        Dictionary of comparison results
    """
    logger.info("Running comparative analysis...")
    
    analyzer = ComparativeAnalyzer()
    
    # Extract quantum and classical results
    quantum_results = {}
    classical_results = {}
    
    for category, results in benchmark_results.items():
        for result in results:
            metric_key = f"{category}_{result.test_name}"
            
            if 'quantum' in result.test_name.lower():
                if metric_key not in quantum_results:
                    quantum_results[metric_key] = []
                quantum_results[metric_key].append(result.duration_ms)
            elif 'classical' in result.test_name.lower():
                if metric_key not in classical_results:
                    classical_results[metric_key] = []
                classical_results[metric_key].append(result.duration_ms)
    
    # Perform statistical comparisons
    comparison_results = {}
    if quantum_results and classical_results:
        comparison_results = analyzer.analyze_quantum_vs_classical(
            quantum_results, classical_results
        )
        logger.info(f"Completed comparative analysis: {len(comparison_results)} comparisons")
    else:
        logger.warning("Insufficient data for quantum vs classical comparison")
    
    return comparison_results


def generate_reports(benchmark_results: dict, 
                    comparison_results: dict, 
                    config: BenchmarkConfig) -> str:
    """
    Generate comprehensive benchmark reports.
    
    Args:
        benchmark_results: Benchmark test results
        comparison_results: Comparative analysis results
        config: Benchmark configuration
        
    Returns:
        Path to generated HTML report
    """
    logger.info("Generating comprehensive reports...")
    
    reporter = BenchmarkReporter(config.output_dir)
    
    try:
        report_path = reporter.generate_comprehensive_report(
            benchmark_results, comparison_results, config
        )
        logger.info(f"Reports generated successfully: {report_path}")
        return report_path
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise


def print_summary(benchmark_results: dict, comparison_results: dict):
    """Print benchmark summary to console."""
    print("\n" + "="*60)
    print("QUANTUMRERANK BENCHMARK SUMMARY")
    print("="*60)
    
    # Calculate overall statistics
    total_tests = sum(len(results) for results in benchmark_results.values())
    successful_tests = sum(
        sum(1 for r in results if r.success) 
        for results in benchmark_results.values()
    )
    
    print(f"Total Tests Executed: {total_tests}")
    print(f"Successful Tests: {successful_tests}")
    print(f"Success Rate: {successful_tests/total_tests*100:.1f}%")
    
    # PRD Compliance Summary
    print(f"\n{'-'*30}")
    print("PRD COMPLIANCE SUMMARY")
    print(f"{'-'*30}")
    
    # Check similarity computation compliance
    similarity_compliant = False
    batch_compliant = False
    memory_compliant = False
    
    for category, results in benchmark_results.items():
        for result in results:
            if 'similarity' in result.test_name and result.target_ms == 100.0:
                similarity_compliant = result.target_met
            elif 'batch' in result.test_name and result.target_ms == 500.0:
                batch_compliant = result.target_met
            elif result.metric_type == 'memory' and result.document_count == 100:
                memory_compliant = (result.memory_mb or 0) < 2048
    
    print(f"Similarity (<100ms): {'✅ PASS' if similarity_compliant else '❌ FAIL'}")
    print(f"Batch Processing (<500ms): {'✅ PASS' if batch_compliant else '❌ FAIL'}")
    print(f"Memory Usage (<2GB): {'✅ PASS' if memory_compliant else '❌ FAIL'}")
    
    overall_compliant = similarity_compliant and batch_compliant and memory_compliant
    print(f"\nOverall PRD Compliance: {'✅ PASS' if overall_compliant else '❌ FAIL'}")
    
    # Comparison Summary
    if comparison_results:
        print(f"\n{'-'*30}")
        print("QUANTUM VS CLASSICAL SUMMARY")
        print(f"{'-'*30}")
        
        quantum_wins = sum(1 for comp in comparison_results.values() if comp.winner == "Quantum")
        classical_wins = sum(1 for comp in comparison_results.values() if comp.winner == "Classical")
        ties = len(comparison_results) - quantum_wins - classical_wins
        
        print(f"Quantum Wins: {quantum_wins}")
        print(f"Classical Wins: {classical_wins}")
        print(f"Ties: {ties}")
        
        if quantum_wins > classical_wins:
            print("🔵 Quantum methods show overall advantage")
        elif classical_wins > quantum_wins:
            print("🔴 Classical methods show overall advantage")
        else:
            print("⚪ Methods show comparable performance")
    
    print("="*60)


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.verbose, args.quiet)
    
    logger.info("Starting QuantumRerank benchmark suite...")
    logger.info(f"Configuration: {args.trials} trials, {args.warmup} warmup, categories: {args.test_categories}")
    
    try:
        # Create benchmark configuration
        config = create_benchmark_config(args)
        
        # Run benchmarks
        benchmark_results = run_benchmarks(config, args.test_categories)
        
        # Run comparative analysis if requested
        comparison_results = {}
        if args.compare_methods:
            comparison_results = run_comparative_analysis(benchmark_results)
        
        # Generate reports unless skipped
        report_path = None
        if not args.skip_report:
            report_path = generate_reports(benchmark_results, comparison_results, config)
        
        # Print summary
        if not args.quiet:
            print_summary(benchmark_results, comparison_results)
        
        # Final status
        if report_path:
            print(f"\n📊 Full report available at: {report_path}")
        
        logger.info("Benchmark suite completed successfully")
        
    except KeyboardInterrupt:
        logger.warning("Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
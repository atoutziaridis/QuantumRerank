"""
Quick validation of the real-world medical benchmark approach.
Tests core functionality with smaller dataset.
"""

from real_world_medical_benchmark import (
    BenchmarkConfig, MedicalRAGBenchmark
)


def main():
    """Run a quick validation test."""
    # Minimal configuration for testing
    config = BenchmarkConfig(
        num_documents=10,   # Small set for quick testing
        num_queries=5,      # Few queries
        noise_levels=[0.0, 0.10, 0.20],  # Three noise levels
        noise_types=["mixed", "ocr"],     # Two noise types
        retrieval_k=5,
        save_results=True,
        detailed_analysis=True
    )
    
    print("Running quick validation of real-world medical benchmark...")
    
    # Run benchmark
    benchmark = MedicalRAGBenchmark(config)
    report = benchmark.run_benchmark()
    
    # Print summary
    benchmark.print_summary_report(report)
    
    # Validate key results
    stats = report['overall_statistics']
    perf = report['performance_analysis']
    
    print(f"\nVALIDATION RESULTS:")
    print(f"✓ Completed {stats['total_evaluations']} evaluations")
    print(f"✓ Average quantum precision improvement: {stats['avg_precision_improvement']:+.1f}%")
    print(f"✓ Quantum wins: {perf['quantum_wins']}/{stats['total_evaluations']}")
    print(f"✓ Average latency: {stats['avg_quantum_latency_ms']:.1f}ms")
    
    # Check if quantum shows advantages in noisy conditions
    noise_analysis = report['noise_level_analysis']
    high_noise_improvement = noise_analysis.get(0.20, {}).get('precision_improvement', 0)
    
    print(f"✓ High noise (20%) precision improvement: {high_noise_improvement:+.1f}%")
    
    if high_noise_improvement > 0:
        print("✅ SUCCESS: Quantum shows advantages in high-noise conditions!")
    else:
        print("⚠️  Note: Quantum advantages may need tuning for this test set")
    
    return report


if __name__ == "__main__":
    main()
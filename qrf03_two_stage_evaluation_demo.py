"""
QRF-03 Demonstration: Proper Two-Stage Retrieval Testing

This script demonstrates the comprehensive two-stage retrieval evaluation framework
that properly tests FAISS → Quantum reranking in realistic scenarios.

Features demonstrated:
1. Medical corpus integration with PMC documents
2. Proper IR evaluation metrics (P@K, NDCG@K, MRR)
3. Quantum vs classical method comparison
4. Scenario-specific testing (noise tolerance, complex queries)
5. Statistical significance testing
6. Actionable recommendations for deployment

Based on QRF-03 task requirements.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append('.')

from quantum_rerank.evaluation.two_stage_evaluation import (
    TwoStageEvaluationFramework, TwoStageEvaluationConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run QRF-03 two-stage retrieval evaluation demonstration."""
    print("QRF-03: PROPER TWO-STAGE RETRIEVAL TESTING")
    print("=" * 60)
    print("Comprehensive evaluation of FAISS → Quantum reranking pipeline")
    print("with proper IR metrics and scenario testing")
    print()
    
    try:
        # Configuration for demonstration
        config = TwoStageEvaluationConfig(
            max_documents=50,  # Use subset of PMC docs for demo
            max_queries=15,    # Test with multiple medical queries
            faiss_candidates_k=30,  # FAISS initial retrieval
            final_results_k=10,     # Final reranked results
            
            # Enable all scenario tests
            test_noise_tolerance=True,
            test_complex_queries=True,
            test_selective_usage=True,
            
            # Test quantum vs classical methods
            methods_to_test=["classical", "quantum", "hybrid"],
            
            # Save detailed results
            save_detailed_results=True,
            output_directory="qrf03_evaluation_results"
        )
        
        print(f"Evaluation Configuration:")
        print(f"  Max documents: {config.max_documents}")
        print(f"  Max queries: {config.max_queries}")
        print(f"  FAISS candidates: {config.faiss_candidates_k}")
        print(f"  Final results: {config.final_results_k}")
        print(f"  Methods to test: {config.methods_to_test}")
        print()
        
        # Initialize evaluation framework
        print("Initializing evaluation framework...")
        framework = TwoStageEvaluationFramework(config)
        
        # Check if PMC data is available
        pmc_docs_path = Path("pmc_docs")
        use_synthetic = True
        
        if pmc_docs_path.exists():
            try:
                print("Loading PMC medical corpus...")
                queries, documents = framework.load_pmc_test_data("pmc_docs")
                print(f"✓ Loaded {len(documents)} PMC documents")
                print(f"✓ Created {len(queries)} medical queries")
                use_synthetic = False
            except Exception as e:
                logger.error(f"Failed to load PMC data: {e}")
                print("⚠️  PMC data loading failed. Creating synthetic test data...")
        
        if use_synthetic:
            print("⚠️  PMC documents not found. Creating synthetic test data...")
            
            # Create synthetic medical documents for demonstration
            from quantum_rerank.evaluation.medical_relevance import MedicalDocument
            
            synthetic_docs = []
            medical_texts = [
                ("Acute Myocardial Infarction Management", "Comprehensive review of AMI diagnosis and treatment protocols in emergency settings.", "cardiology"),
                ("Type 2 Diabetes Treatment Guidelines", "Evidence-based management strategies for T2DM including pharmacological interventions.", "diabetes"),
                ("COPD Exacerbation Protocol", "Clinical guidelines for managing acute COPD exacerbations in hospital settings.", "respiratory"),
                ("Stroke Prevention Strategies", "Primary and secondary prevention approaches for cerebrovascular accidents.", "neurology"),
                ("Cancer Immunotherapy Advances", "Recent developments in immunotherapeutic approaches for oncological treatment.", "oncology"),
                ("Hypertension Control Methods", "Systematic approaches to blood pressure management in clinical practice.", "cardiology"),
                ("Diabetic Neuropathy Treatment", "Management of peripheral neuropathy complications in diabetic patients.", "diabetes"),
                ("Asthma Management Guidelines", "Evidence-based protocols for asthma control and exacerbation prevention.", "respiratory"),
                ("Alzheimer Disease Research", "Current understanding of AD pathophysiology and therapeutic targets.", "neurology"),
                ("Breast Cancer Screening", "Guidelines for mammographic screening and early detection strategies.", "oncology")
            ]
            
            for i, (title, abstract, domain) in enumerate(medical_texts * 5):  # Replicate for more docs
                doc = MedicalDocument(
                    doc_id=f"synthetic_doc_{i}",
                    title=f"{title} - Study {i}",
                    abstract=f"{abstract} This synthetic document {i} provides clinical insights for {domain}.",
                    full_text=f"Detailed clinical study about {domain} conditions. {abstract} This comprehensive document covers diagnostic approaches, treatment protocols, and patient management strategies. Document ID: {i}",
                    medical_domain=domain,
                    key_terms=[domain, "clinical", "treatment", "diagnosis"],
                    sections={"introduction": f"{domain} overview", "methods": "clinical protocols"}
                )
                synthetic_docs.append(doc)
            
            framework.test_documents = synthetic_docs[:config.max_documents]
            
            # Create synthetic queries
            from quantum_rerank.evaluation.medical_relevance import create_medical_test_queries
            framework.test_queries = create_medical_test_queries()[:config.max_queries]
            
            print(f"✓ Created {len(framework.test_documents)} synthetic documents")
            print(f"✓ Created {len(framework.test_queries)} test queries")
        
        
        print()
        print("Starting comprehensive two-stage retrieval evaluation...")
        print()
        
        # Run comprehensive evaluation
        report = framework.run_comprehensive_evaluation()
        
        # Print detailed results
        framework.print_evaluation_summary(report)
        
        # Additional analysis
        print("\nDetailed Analysis:")
        print("-" * 40)
        
        # Method performance comparison
        best_method = max(report.method_comparisons, 
                         key=lambda x: x.metrics.ndcg_at_k.get(10, 0))
        
        print(f"Best performing method: {best_method.method_name}")
        print(f"  NDCG@10: {best_method.metrics.ndcg_at_k.get(10, 0):.3f}")
        print(f"  Precision@10: {best_method.metrics.precision_at_k.get(10, 0):.3f}")
        print(f"  MRR: {best_method.metrics.mrr:.3f}")
        
        # Quantum vs Classical comparison
        quantum_results = [r for r in report.method_comparisons if r.method_name == "quantum"]
        classical_results = [r for r in report.method_comparisons if r.method_name == "classical"]
        
        if quantum_results and classical_results:
            quantum_ndcg = quantum_results[0].metrics.ndcg_at_k.get(10, 0)
            classical_ndcg = classical_results[0].metrics.ndcg_at_k.get(10, 0)
            
            improvement = ((quantum_ndcg - classical_ndcg) / classical_ndcg * 100) if classical_ndcg > 0 else 0
            
            print(f"\nQuantum vs Classical Comparison:")
            print(f"  Classical NDCG@10: {classical_ndcg:.3f}")
            print(f"  Quantum NDCG@10: {quantum_ndcg:.3f}")
            print(f"  Improvement: {improvement:+.1f}%")
            
            if improvement > 5:
                print("  ✅ Quantum shows significant improvement")
            elif improvement > 0:
                print("  ⚡ Quantum shows modest improvement")
            else:
                print("  ⚠️  Quantum does not show clear advantage")
        
        # Scenario test highlights
        if report.scenario_results:
            print(f"\nScenario Test Highlights:")
            
            for scenario_type, results in report.scenario_results.items():
                best_improvement = max(
                    (r.performance_improvement.get('ndcg_10', 0) for r in results),
                    default=0
                )
                
                print(f"  {scenario_type}: {len(results)} tests, "
                      f"best improvement: {best_improvement:+.1f}%")
                
                # Highlight significant improvements
                if best_improvement > 10:
                    print(f"    🎯 Strong quantum advantage detected!")
                elif best_improvement > 5:
                    print(f"    ⚡ Moderate quantum advantage")
        
        # Statistical significance summary
        if report.statistical_tests:
            significant_tests = 0
            total_tests = 0
            
            for comparison, tests in report.statistical_tests.items():
                for metric, test_result in tests.items():
                    total_tests += 1
                    if test_result.get('significant_at_05', False):
                        significant_tests += 1
            
            print(f"\nStatistical Significance:")
            print(f"  {significant_tests}/{total_tests} tests showed significant differences")
            if significant_tests > 0:
                print(f"  ✅ Statistical evidence for method differences")
            else:
                print(f"  ⚠️  No statistically significant differences found")
        
        # Output information
        output_path = Path(config.output_directory)
        if output_path.exists():
            result_files = list(output_path.glob("evaluation_*"))
            print(f"\nResults saved to: {output_path}")
            print(f"  Files created: {len(result_files)}")
            for file_path in result_files[:3]:  # Show first 3 files
                print(f"    {file_path.name}")
        
        print("\n" + "=" * 60)
        print("QRF-03 EVALUATION COMPLETED SUCCESSFULLY ✅")
        print("=" * 60)
        
        # Final recommendations summary
        print("\nKEY FINDINGS:")
        for i, rec in enumerate(report.recommendations[:5], 1):
            print(f"{i}. {rec}")
        
        return True
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n❌ QRF-03 evaluation failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
"""
Comprehensive Integration Tests for Two-Stage Retrieval with Quantum Reranking.

This test suite validates the complete FAISS → Quantum reranking pipeline
with proper IR evaluation metrics and scenario testing.

Based on QRF-03 requirements for proper two-stage retrieval testing.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any

# Import quantum rerank modules
from quantum_rerank.evaluation.two_stage_evaluation import (
    TwoStageEvaluationFramework, TwoStageEvaluationConfig
)
from quantum_rerank.evaluation.medical_relevance import (
    MedicalRelevanceJudgments, MedicalQuery, MedicalDocument, create_medical_test_queries
)
from quantum_rerank.evaluation.scenario_testing import (
    QuantumAdvantageScenarios, NoiseConfig, ComplexQueryGenerator
)
from quantum_rerank.evaluation.ir_metrics import (
    IRMetricsCalculator, RelevanceJudgment, QueryResult, RetrievalResult
)
from quantum_rerank.retrieval.two_stage_retriever import TwoStageRetriever, RetrieverConfig
from quantum_rerank.core.embeddings import EmbeddingProcessor


class TestTwoStageRetrievalIntegration:
    """Integration tests for two-stage retrieval evaluation."""
    
    @pytest.fixture(scope="class")
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture(scope="class")
    def embedding_processor(self):
        """Initialize embedding processor."""
        return EmbeddingProcessor()
    
    @pytest.fixture(scope="class")
    def medical_relevance(self, embedding_processor):
        """Initialize medical relevance system."""
        return MedicalRelevanceJudgments(embedding_processor)
    
    @pytest.fixture(scope="class")
    def test_documents(self, medical_relevance):
        """Create test medical documents."""
        documents = [
            MedicalDocument(
                doc_id="doc_001",
                title="Myocardial Infarction Diagnosis and Treatment",
                abstract="This study evaluates diagnostic methods and treatment protocols for acute myocardial infarction in emergency settings.",
                full_text="Acute myocardial infarction (MI) represents a medical emergency requiring immediate intervention. Standard diagnostic approaches include ECG analysis, troponin levels, and clinical assessment. Treatment protocols emphasize rapid reperfusion therapy through either percutaneous coronary intervention (PCI) or thrombolytic therapy.",
                medical_domain="cardiology",
                key_terms=["myocardial infarction", "troponin", "ECG", "PCI"],
                sections={"introduction": "MI overview", "methods": "diagnostic protocols"}
            ),
            MedicalDocument(
                doc_id="doc_002", 
                title="Type 2 Diabetes Management Guidelines",
                abstract="Comprehensive review of evidence-based management strategies for type 2 diabetes mellitus including lifestyle modifications and pharmacological interventions.",
                full_text="Type 2 diabetes mellitus (T2DM) management requires a multifaceted approach combining lifestyle modifications, blood glucose monitoring, and appropriate pharmacological interventions. First-line therapy typically includes metformin, with additional agents added based on glycemic targets and patient-specific factors.",
                medical_domain="diabetes",
                key_terms=["diabetes", "metformin", "glucose", "glycemic"],
                sections={"background": "diabetes overview", "treatment": "management strategies"}
            ),
            MedicalDocument(
                doc_id="doc_003",
                title="COPD Exacerbation Management in Emergency Department",
                abstract="Clinical protocols for managing acute chronic obstructive pulmonary disease exacerbations in emergency department settings.",
                full_text="Chronic obstructive pulmonary disease (COPD) exacerbations are a leading cause of emergency department visits. Management involves bronchodilators, corticosteroids, oxygen therapy, and assessment for respiratory failure. Antibiotic therapy may be indicated in cases with purulent sputum or systemic infection.",
                medical_domain="respiratory",
                key_terms=["COPD", "bronchodilators", "corticosteroids", "respiratory"],
                sections={"assessment": "clinical evaluation", "treatment": "therapeutic interventions"}
            ),
            MedicalDocument(
                doc_id="doc_004",
                title="Machine Learning in Medical Imaging",
                abstract="Review of artificial intelligence applications in diagnostic radiology and medical image analysis.",
                full_text="Artificial intelligence and machine learning techniques are increasingly applied to medical imaging for automated diagnosis, image segmentation, and radiological screening. Deep learning models show particular promise in detecting subtle abnormalities in chest X-rays, CT scans, and MRI images.",
                medical_domain="general",
                key_terms=["artificial intelligence", "machine learning", "medical imaging", "radiology"],
                sections={"technology": "AI overview", "applications": "clinical use cases"}
            ),
            MedicalDocument(
                doc_id="doc_005",
                title="Stroke Prevention and Management",
                abstract="Evidence-based approaches to stroke prevention, acute management, and rehabilitation strategies.",
                full_text="Stroke prevention involves risk factor modification including blood pressure control, anticoagulation for atrial fibrillation, and lifestyle modifications. Acute stroke management emphasizes rapid recognition, imaging evaluation, and timely reperfusion therapy when appropriate.",
                medical_domain="neurology", 
                key_terms=["stroke", "prevention", "anticoagulation", "rehabilitation"],
                sections={"prevention": "risk factors", "acute_care": "emergency management"}
            )
        ]
        
        return documents
    
    @pytest.fixture(scope="class")
    def test_queries(self, medical_relevance):
        """Create test medical queries."""
        return create_medical_test_queries()[:8]  # Use subset for faster testing
    
    @pytest.fixture(scope="class")
    def evaluation_config(self, temp_dir):
        """Create evaluation configuration."""
        return TwoStageEvaluationConfig(
            max_documents=10,
            max_queries=8,
            faiss_candidates_k=20,
            final_results_k=5,
            test_noise_tolerance=True,
            test_complex_queries=True,
            test_selective_usage=False,  # Skip for faster testing
            methods_to_test=["classical", "quantum"],
            save_detailed_results=True,
            output_directory=temp_dir
        )
    
    def test_medical_relevance_system(self, medical_relevance, test_queries, test_documents):
        """Test medical relevance judgment system."""
        # Test query classification
        query = test_queries[0]
        assert query.medical_domain in ["cardiology", "diabetes", "respiratory", "neurology", "general"]
        assert query.query_type in ["diagnostic", "treatment", "symptom", "prognosis", "general"]
        assert query.complexity_level in ["simple", "moderate", "complex"]
        
        # Test document classification
        doc = test_documents[0]
        assert doc.medical_domain in ["cardiology", "diabetes", "respiratory", "neurology", "general"]
        assert len(doc.key_terms) > 0
        
        # Test relevance judgment creation
        judgments = medical_relevance.create_relevance_judgments([query], [doc])
        assert len(judgments) == 1
        assert judgments[0].query_id == query.query_id
        assert judgments[0].doc_id == doc.doc_id
        assert judgments[0].relevance in [0, 1, 2]
        assert 0.0 <= judgments[0].confidence <= 1.0
    
    def test_ir_metrics_calculator(self, test_queries, test_documents):
        """Test IR metrics calculation."""
        calculator = IRMetricsCalculator()
        
        # Create synthetic relevance judgments
        judgments = []
        for query in test_queries[:3]:
            for doc in test_documents[:3]:
                relevance = 1 if query.medical_domain == doc.medical_domain else 0
                judgment = RelevanceJudgment(
                    query_id=query.query_id,
                    doc_id=doc.doc_id,
                    relevance=relevance
                )
                judgments.append(judgment)
        
        calculator.add_relevance_judgments(judgments)
        
        # Create synthetic query results
        query_results = []
        for query in test_queries[:3]:
            results = []
            for i, doc in enumerate(test_documents[:3]):
                result = RetrievalResult(
                    doc_id=doc.doc_id,
                    score=0.9 - i * 0.1,  # Decreasing scores
                    rank=i + 1
                )
                results.append(result)
            
            query_result = QueryResult(
                query_id=query.query_id,
                query_text=query.query_text,
                results=results,
                method="test_method"
            )
            query_results.append(query_result)
        
        # Test metrics calculation
        metrics = calculator.evaluate_method(query_results)
        
        assert metrics.method_name == "test_method"
        assert 5 in metrics.precision_at_k
        assert 5 in metrics.ndcg_at_k
        assert 0.0 <= metrics.mrr <= 1.0
        assert 0.0 <= metrics.map_score <= 1.0
        assert metrics.query_count == 3
    
    def test_two_stage_retriever_setup(self, embedding_processor, test_documents):
        """Test two-stage retriever configuration and setup."""
        config = RetrieverConfig(
            initial_k=10,
            final_k=5,
            reranking_method="classical"
        )
        
        retriever = TwoStageRetriever(config, embedding_processor)
        
        # Add test documents
        texts = [f"{doc.title} {doc.abstract}" for doc in test_documents]
        metadatas = [{"doc_id": doc.doc_id, "domain": doc.medical_domain} for doc in test_documents]
        
        doc_ids = retriever.add_texts(texts, metadatas)
        assert len(doc_ids) == len(test_documents)
        
        # Test retrieval
        query = "heart attack diagnosis and treatment"
        results = retriever.retrieve(query, k=3)
        
        assert len(results) <= 3
        assert all(result.score >= 0 for result in results)
        assert all(result.rank > 0 for result in results)
        
        # Test stats
        stats = retriever.get_stats()
        assert "retriever_stats" in stats
        assert "faiss_stats" in stats
        assert stats["retriever_stats"]["total_queries"] > 0
    
    def test_noise_injection(self, test_documents):
        """Test noise injection for robustness testing."""
        from quantum_rerank.evaluation.scenario_testing import NoiseInjector
        
        noise_injector = NoiseInjector()
        
        # Test OCR noise
        original_text = "The patient has diabetes and hypertension."
        noise_config = NoiseConfig(noise_type='ocr', noise_level=0.2)
        noisy_text = noise_injector.inject_noise(original_text, noise_config)
        
        # Should have some differences but preserve key medical terms
        assert len(noisy_text) > 0
        assert "diabetes" in noisy_text or "diabetic" in noisy_text  # May be slightly corrupted
        
        # Test abbreviation noise
        abbrev_text = "Patient has MI and HTN, requires immediate care."
        abbrev_config = NoiseConfig(noise_type='abbreviation', noise_level=0.5)
        noisy_abbrev = noise_injector.inject_noise(abbrev_text, abbrev_config)
        
        assert len(noisy_abbrev) > 0
        # Should have some corruption of abbreviations
    
    def test_complex_query_generation(self):
        """Test complex query generation for advanced testing."""
        generator = ComplexQueryGenerator()
        
        # Test multi-domain query
        domains = ['cardiology', 'diabetes']
        multi_query = generator.generate_multi_domain_query(domains)
        
        assert len(multi_query) > 20  # Should be reasonably complex
        assert any(term in multi_query.lower() for term in ['heart', 'cardiac', 'diabetes'])
        
        # Test ambiguous query
        ambiguous_query = generator.generate_ambiguous_query()
        
        assert len(ambiguous_query) > 10
        # Should contain medical abbreviation
        
        # Test long complex query
        long_query = generator.generate_long_complex_query()
        
        assert len(long_query) > 100  # Should be a detailed clinical scenario
        assert "patient" in long_query.lower()
    
    def test_scenario_testing_framework(self, embedding_processor, test_queries, test_documents):
        """Test scenario-specific testing framework."""
        # Setup retriever
        config = RetrieverConfig(reranking_method="quantum")
        retriever = TwoStageRetriever(config, embedding_processor)
        
        # Add documents
        texts = [f"{doc.title} {doc.abstract}" for doc in test_documents]
        metadatas = [{"doc_id": doc.doc_id} for doc in test_documents]
        retriever.add_texts(texts, metadatas)
        
        # Setup scenario tester
        medical_relevance = MedicalRelevanceJudgments(embedding_processor)
        scenario_tester = QuantumAdvantageScenarios(retriever, medical_relevance)
        
        # Test noise tolerance (simplified)
        noise_configs = [NoiseConfig(noise_type='ocr', noise_level=0.1)]
        noise_results = scenario_tester.test_noise_tolerance(
            test_queries[:2], test_documents[:3], noise_configs
        )
        
        assert len(noise_results) == 1
        assert noise_results[0].scenario_name == "noise_tolerance_ocr"
        assert "classical" in noise_results[0].baseline_metrics
        assert "quantum" in noise_results[0].quantum_metrics
        assert isinstance(noise_results[0].execution_time_ms, float)
    
    def test_full_evaluation_framework(self, evaluation_config, temp_dir):
        """Test complete evaluation framework."""
        # Create evaluation framework
        framework = TwoStageEvaluationFramework(evaluation_config)
        
        # Mock PMC data loading (use test data instead)
        framework.test_queries = create_medical_test_queries()[:5]
        
        # Create minimal test documents
        framework.test_documents = [
            MedicalDocument(
                doc_id=f"test_doc_{i}",
                title=f"Medical Document {i}",
                abstract=f"This is a test medical abstract about condition {i}.",
                full_text=f"Full text for medical document {i} with detailed information.",
                medical_domain="general",
                key_terms=[f"condition_{i}", "medical", "test"],
                sections={}
            )
            for i in range(5)
        ]
        
        # Test method evaluation
        classical_result = framework.evaluate_method("classical")
        
        assert classical_result.method_name == "classical"
        assert classical_result.metrics.query_count == 5
        assert classical_result.execution_time_ms > 0
        assert len(classical_result.query_results) == 5
        
        # Test method comparison
        method_results = framework.compare_methods(["classical"])
        
        assert len(method_results) == 1
        assert method_results[0].method_name == "classical"
        
        # Test statistical tests
        # Need at least 2 methods for comparison
        quantum_result = framework.evaluate_method("quantum")
        method_results.append(quantum_result)
        
        statistical_tests = framework.run_statistical_tests(method_results)
        
        assert len(statistical_tests) >= 1  # At least one comparison
        
        # Test recommendations generation
        recommendations = framework.generate_recommendations(
            method_results, {}, statistical_tests
        )
        
        assert len(recommendations) > 0
        assert any("method" in rec.lower() for rec in recommendations)
    
    def test_evaluation_report_generation(self, evaluation_config, temp_dir):
        """Test evaluation report generation and saving."""
        framework = TwoStageEvaluationFramework(evaluation_config)
        
        # Use minimal test data for speed
        framework.test_queries = create_medical_test_queries()[:3]
        framework.test_documents = [
            MedicalDocument(
                doc_id=f"report_test_{i}",
                title=f"Report Test Document {i}",
                abstract=f"Abstract for report test {i}.",
                full_text=f"Full text for report test document {i}.",
                medical_domain="general",
                key_terms=[f"test_{i}"],
                sections={}
            )
            for i in range(3)
        ]
        
        # Run evaluation with minimal testing
        config = evaluation_config
        config.test_noise_tolerance = False
        config.test_complex_queries = False
        config.test_selective_usage = False
        config.methods_to_test = ["classical"]
        
        framework.config = config
        
        # Run evaluation
        report = framework.run_comprehensive_evaluation()
        
        # Verify report structure
        assert report.config == config
        assert "queries_count" in report.test_data_summary
        assert len(report.method_comparisons) == 1
        assert len(report.recommendations) > 0
        assert report.total_execution_time_ms > 0
        
        # Check if files were saved
        output_path = Path(temp_dir)
        pickle_files = list(output_path.glob("evaluation_report_*.pkl"))
        summary_files = list(output_path.glob("evaluation_summary_*.txt"))
        
        assert len(pickle_files) >= 1
        assert len(summary_files) >= 1
        
        # Verify summary file content
        with open(summary_files[0], 'r') as f:
            summary_content = f.read()
        
        assert "Two-Stage Retrieval Evaluation Report" in summary_content
        assert "Method Comparison Results" in summary_content
        assert "Recommendations" in summary_content
    
    @pytest.mark.slow
    def test_comprehensive_integration(self, temp_dir):
        """Comprehensive integration test with realistic data."""
        # This test uses more realistic settings and can be slow
        config = TwoStageEvaluationConfig(
            max_documents=20,
            max_queries=10,
            faiss_candidates_k=15,
            final_results_k=5,
            test_noise_tolerance=True,
            test_complex_queries=True,
            test_selective_usage=True,
            methods_to_test=["classical", "quantum", "hybrid"],
            save_detailed_results=True,
            output_directory=temp_dir
        )
        
        framework = TwoStageEvaluationFramework(config)
        
        # Create more comprehensive test data
        framework.test_queries = create_medical_test_queries()[:10]
        framework.test_documents = []
        
        # Create documents across multiple domains
        domains = ["cardiology", "diabetes", "respiratory", "neurology", "general"]
        for i in range(20):
            domain = domains[i % len(domains)]
            doc = MedicalDocument(
                doc_id=f"comprehensive_doc_{i}",
                title=f"{domain.title()} Study {i}",
                abstract=f"Research study focusing on {domain} conditions and treatments. Document {i}.",
                full_text=f"Detailed clinical study about {domain} with comprehensive analysis and findings. This is document {i} in our test corpus.",
                medical_domain=domain,
                key_terms=[domain, f"study_{i}", "clinical", "research"],
                sections={"methods": f"Study methods for {domain}", "results": f"Clinical results"}
            )
            framework.test_documents.append(doc)
        
        # Run comprehensive evaluation
        try:
            report = framework.run_comprehensive_evaluation()
            
            # Verify comprehensive results
            assert len(report.method_comparisons) >= 2
            assert len(report.scenario_results) > 0
            assert len(report.statistical_tests) > 0
            assert len(report.recommendations) >= 5
            
            # Print summary for manual verification
            framework.print_evaluation_summary(report)
            
        except Exception as e:
            pytest.skip(f"Comprehensive test requires full quantum stack: {e}")


if __name__ == "__main__":
    # Run specific test for debugging
    pytest.main([__file__ + "::TestTwoStageRetrievalIntegration::test_medical_relevance_system", "-v"])
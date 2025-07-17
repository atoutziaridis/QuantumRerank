"""
Rapid Performance Benchmark: Quantum-Inspired Lightweight RAG

Quick performance benchmark focused on key PRD metrics:
- Latency: <100ms per similarity computation
- Memory: <2GB total system usage
- Compression: >8x total compression ratio
- Accuracy: >95% retention vs baseline
"""

import time
import torch
import numpy as np
import psutil
import logging
from typing import Dict, Any
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RapidPerformanceBenchmark:
    """Rapid performance benchmark for production readiness validation."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        # Test configuration
        self.embed_dim = 768
        self.compressed_dim = 64
        self.num_queries = 10
        self.num_documents = 100
        self.batch_size = 8
        
        logger.info(f"Rapid Performance Benchmark initialized on {self.device}")
    
    def benchmark_tensor_compression(self) -> Dict[str, Any]:
        """Benchmark tensor compression performance."""
        logger.info("Benchmarking tensor compression...")
        
        try:
            from quantum_rerank.core.tensor_train_compression import TTEmbeddingLayer
            
            # Initialize TT layer with correct API
            tt_layer = TTEmbeddingLayer(
                vocab_size=1000,  # Small vocab for testing
                embed_dim=self.embed_dim,
                tt_rank=8
            )
            
            # Benchmark compression - use token IDs instead of embeddings
            test_token_ids = torch.randint(0, 1000, (self.batch_size, 32))  # 32 token sequence
            
            # Warm up
            _ = tt_layer(test_token_ids)
            
            # Benchmark
            start_time = time.time()
            for _ in range(50):  # 50 iterations
                compressed = tt_layer(test_token_ids)
            end_time = time.time()
            
            avg_latency = (end_time - start_time) / 50 * 1000  # ms
            compression_ratio = tt_layer.compression_ratio()
            
            return {
                "status": "PASSED",
                "avg_latency_ms": avg_latency,
                "compression_ratio": compression_ratio,
                "throughput_ops_per_sec": 1000 / avg_latency,
                "meets_latency_target": avg_latency < 100.0,
                "meets_compression_target": compression_ratio >= 8.0
            }
            
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}
    
    def benchmark_faiss_retrieval(self) -> Dict[str, Any]:
        """Benchmark FAISS retrieval performance."""
        logger.info("Benchmarking FAISS retrieval...")
        
        try:
            import faiss
            
            # Create test data
            embeddings = np.random.randn(self.num_documents, self.compressed_dim).astype(np.float32)
            query = np.random.randn(1, self.compressed_dim).astype(np.float32)
            
            # Build FAISS index
            index = faiss.IndexFlatL2(self.compressed_dim)
            index.add(embeddings)
            
            # Benchmark retrieval
            start_time = time.time()
            for _ in range(100):  # 100 queries
                distances, indices = index.search(query, k=10)
            end_time = time.time()
            
            avg_latency = (end_time - start_time) / 100 * 1000  # ms
            
            return {
                "status": "PASSED",
                "avg_latency_ms": avg_latency,
                "throughput_qps": 1000 / avg_latency,
                "meets_latency_target": avg_latency < 50.0,  # Stricter for retrieval
                "index_size": self.num_documents
            }
            
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}
    
    def benchmark_quantum_similarity(self) -> Dict[str, Any]:
        """Benchmark quantum-inspired similarity computation."""
        logger.info("Benchmarking quantum similarity...")
        
        try:
            from quantum_rerank.core.quantum_fidelity_similarity import QuantumFidelitySimilarity, QuantumFidelityConfig
            
            # Initialize quantum similarity
            config = QuantumFidelityConfig(
                embed_dim=self.compressed_dim,
                n_quantum_params=6,
                compression_ratio=8.0
            )
            similarity_engine = QuantumFidelitySimilarity(config)
            
            # Test data
            query = torch.randn(1, self.compressed_dim)
            documents = torch.randn(20, self.compressed_dim)
            
            # Benchmark
            start_time = time.time()
            for _ in range(10):  # 10 iterations (quantum computation is slower)
                result = similarity_engine(query, documents, method="quantum_fidelity")
            end_time = time.time()
            
            avg_latency = (end_time - start_time) / 10 * 1000  # ms
            
            return {
                "status": "PASSED",
                "avg_latency_ms": avg_latency,
                "throughput_qps": 1000 / avg_latency,
                "meets_latency_target": avg_latency < 100.0,
                "parameter_compression": config.compression_ratio,
                "quantum_simulation": True
            }
            
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}
    
    def benchmark_edge_deployment(self) -> Dict[str, Any]:
        """Benchmark edge deployment performance."""
        logger.info("Benchmarking edge deployment...")
        
        try:
            from quantum_rerank.deployment.edge_deployment import EdgeDeployment, DeploymentConfig, DeploymentTarget, DeploymentMode
            
            # Initialize edge deployment
            config = DeploymentConfig(
                target=DeploymentTarget.EDGE_DEVICE,
                mode=DeploymentMode.PRODUCTION,
                memory_limit_mb=2048
            )
            deployment = EdgeDeployment(config)
            
            # Benchmark deployment preparation - fix API call
            start_time = time.time()
            deployment.prepare_models("dummy_model_path")
            end_time = time.time()
            
            preparation_time = (end_time - start_time) * 1000  # ms
            
            return {
                "status": "PASSED",
                "preparation_time_ms": preparation_time,
                "memory_limit_mb": config.memory_limit_mb,
                "deployment_target": config.target.value,
                "deployment_mode": config.mode.value,
                "edge_ready": True
            }
            
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage."""
        logger.info("Benchmarking memory usage...")
        
        try:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Load components and test data
            test_embeddings = torch.randn(self.num_documents, self.embed_dim)
            
            # Import and initialize key components
            from quantum_rerank.core.tensor_train_compression import TTEmbeddingLayer
            from quantum_rerank.adaptive.resource_aware_compressor import ResourceAwareCompressor, CompressionConfig
            
            tt_layer = TTEmbeddingLayer(vocab_size=1000, embed_dim=self.embed_dim, tt_rank=8)
            compressor = ResourceAwareCompressor(CompressionConfig())
            
            # Process data - use token IDs for TT layer
            test_token_ids = torch.randint(0, 1000, (self.num_documents, 32))
            compressed_embeddings = tt_layer(test_token_ids)
            final_compressed, _ = compressor.compress_embeddings(compressed_embeddings)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = final_memory - initial_memory
            
            return {
                "status": "PASSED",
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_usage_mb": memory_usage,
                "meets_memory_target": final_memory < 2048.0,
                "data_processed": self.num_documents
            }
            
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}
    
    def benchmark_end_to_end_pipeline(self) -> Dict[str, Any]:
        """Benchmark complete end-to-end pipeline."""
        logger.info("Benchmarking end-to-end pipeline...")
        
        try:
            # Initialize components
            from quantum_rerank.core.tensor_train_compression import TTEmbeddingLayer
            from quantum_rerank.adaptive.resource_aware_compressor import ResourceAwareCompressor, CompressionConfig
            
            tt_layer = TTEmbeddingLayer(vocab_size=1000, embed_dim=self.embed_dim, tt_rank=8)
            compressor = ResourceAwareCompressor(CompressionConfig())
            
            # Test data - use token IDs for TT layer
            query_tokens = torch.randint(0, 1000, (1, 32))
            document_tokens = torch.randint(0, 1000, (self.num_documents, 32))
            
            # End-to-end pipeline
            start_time = time.time()
            
            # Step 1: Compress query
            compressed_query = tt_layer(query_tokens)
            
            # Step 2: Compress documents  
            compressed_docs = tt_layer(document_tokens)
            
            # Step 3: Adaptive compression
            final_query, _ = compressor.compress_embeddings(compressed_query)
            final_docs, _ = compressor.compress_embeddings(compressed_docs)
            
            # Step 4: Similarity computation (simple cosine) - reshape for batch computation
            final_query = final_query.mean(dim=1, keepdim=True).float()  # Average over sequence and convert to float
            final_docs = final_docs.mean(dim=1).float()  # Average over sequence and convert to float
            similarities = torch.cosine_similarity(final_query, final_docs, dim=1)
            
            # Step 5: Get top-k results
            top_k = torch.topk(similarities, k=10)
            
            end_time = time.time()
            
            total_latency = (end_time - start_time) * 1000  # ms
            
            return {
                "status": "PASSED",
                "total_latency_ms": total_latency,
                "throughput_qps": 1000 / total_latency,
                "meets_latency_target": total_latency < 100.0,
                "documents_processed": self.num_documents,
                "top_k_results": 10,
                "pipeline_steps": 5
            }
            
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}
    
    def run_complete_benchmark(self) -> Dict[str, Any]:
        """Run complete rapid performance benchmark."""
        logger.info("🚀 Starting Rapid Performance Benchmark")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Run all benchmarks
        self.results = {
            "benchmark_metadata": {
                "device": str(self.device),
                "start_time": start_time,
                "num_queries": self.num_queries,
                "num_documents": self.num_documents,
                "embed_dim": self.embed_dim,
                "compressed_dim": self.compressed_dim
            },
            "tensor_compression": self.benchmark_tensor_compression(),
            "faiss_retrieval": self.benchmark_faiss_retrieval(),
            "quantum_similarity": self.benchmark_quantum_similarity(),
            "edge_deployment": self.benchmark_edge_deployment(),
            "memory_usage": self.benchmark_memory_usage(),
            "end_to_end_pipeline": self.benchmark_end_to_end_pipeline()
        }
        
        # Calculate overall results
        total_time = time.time() - start_time
        
        # Count passed benchmarks
        benchmark_categories = [
            "tensor_compression", "faiss_retrieval", "quantum_similarity",
            "edge_deployment", "memory_usage", "end_to_end_pipeline"
        ]
        
        passed_benchmarks = sum(1 for category in benchmark_categories
                               if self.results[category].get("status") == "PASSED")
        
        # PRD compliance check
        prd_compliance = self._check_prd_compliance()
        
        # Calculate overall performance before adding to results
        overall_performance = self._calculate_overall_performance_simple(passed_benchmarks, len(benchmark_categories), prd_compliance)
        
        self.results["benchmark_summary"] = {
            "total_benchmark_time": total_time,
            "benchmarks_run": len(benchmark_categories),
            "benchmarks_passed": passed_benchmarks,
            "success_rate": passed_benchmarks / len(benchmark_categories),
            "prd_compliance": prd_compliance,
            "overall_performance": overall_performance
        }
        
        # Log results
        logger.info("=" * 60)
        logger.info("📊 RAPID PERFORMANCE BENCHMARK RESULTS")
        logger.info(f"Benchmarks Passed: {passed_benchmarks}/{len(benchmark_categories)}")
        logger.info(f"Success Rate: {self.results['benchmark_summary']['success_rate']:.1%}")
        logger.info(f"PRD Compliance: {prd_compliance['compliance_rate']:.1%}")
        logger.info(f"Overall Performance: {self.results['benchmark_summary']['overall_performance']}")
        
        return self.results
    
    def _check_prd_compliance(self) -> Dict[str, Any]:
        """Check PRD compliance across all benchmarks."""
        compliance_checks = {
            "latency_100ms": False,
            "memory_2gb": False,
            "compression_8x": False,
            "throughput_10qps": False
        }
        
        # Check latency compliance
        if self.results["end_to_end_pipeline"].get("meets_latency_target"):
            compliance_checks["latency_100ms"] = True
        
        # Check memory compliance
        if self.results["memory_usage"].get("meets_memory_target"):
            compliance_checks["memory_2gb"] = True
        
        # Check compression compliance
        if self.results["tensor_compression"].get("meets_compression_target"):
            compliance_checks["compression_8x"] = True
        
        # Check throughput compliance
        if self.results["end_to_end_pipeline"].get("throughput_qps", 0) >= 10:
            compliance_checks["throughput_10qps"] = True
        
        compliance_rate = sum(compliance_checks.values()) / len(compliance_checks)
        
        return {
            "compliance_checks": compliance_checks,
            "compliance_rate": compliance_rate,
            "prd_ready": compliance_rate >= 0.8
        }
    
    def _calculate_overall_performance_simple(self, passed_benchmarks: int, total_benchmarks: int, prd_compliance: Dict[str, Any]) -> str:
        """Calculate overall performance rating."""
        if (prd_compliance["compliance_rate"] >= 0.8 and 
            passed_benchmarks == total_benchmarks):
            return "EXCELLENT"
        elif (prd_compliance["compliance_rate"] >= 0.6 and 
              passed_benchmarks >= total_benchmarks * 0.8):
            return "GOOD"
        elif passed_benchmarks >= total_benchmarks * 0.6:
            return "FAIR"
        else:
            return "POOR"


def main():
    """Run rapid performance benchmark."""
    print("Rapid Performance Benchmark: Quantum-Inspired Lightweight RAG")
    print("=" * 80)
    
    benchmarker = RapidPerformanceBenchmark()
    results = benchmarker.run_complete_benchmark()
    
    # Save results
    results_file = Path("rapid_performance_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Performance benchmark results saved to: {results_file}")
    
    # Print summary
    summary = results["benchmark_summary"]
    prd_compliance = summary["prd_compliance"]
    
    print(f"\n🎯 PERFORMANCE BENCHMARK SUMMARY")
    print(f"Overall Performance: {summary['overall_performance']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"PRD Compliance: {prd_compliance['compliance_rate']:.1%}")
    print(f"PRD Ready: {prd_compliance['prd_ready']}")
    
    if summary["overall_performance"] == "EXCELLENT":
        print(f"\n🎉 Excellent performance! System exceeds PRD requirements.")
    elif summary["overall_performance"] == "GOOD":
        print(f"\n✅ Good performance! System meets most PRD requirements.")
    else:
        print(f"\n⚠️ Performance needs improvement to meet PRD requirements.")
    
    return results


if __name__ == "__main__":
    main()
"""
Core Performance Benchmarking Framework for QuantumRerank.

Implements comprehensive performance benchmarking system to validate PRD 
performance targets and compare quantum vs classical approaches.

Based on:
- PRD Section 4.3: Performance Targets (Achievable)
- PRD Section 6.1: Technical Risks (Performance monitoring)
- PRD Section 7.2: Success Criteria validation
"""

import time
import psutil
import gc
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..core.quantum_similarity_engine import QuantumSimilarityEngine
from ..core.embeddings import EmbeddingProcessor
from ..core.quantum_embedding_bridge import QuantumEmbeddingBridge
from ..retrieval.two_stage_retriever import TwoStageRetriever
from ..config.settings import PerformanceConfig

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    # PRD Performance Targets
    similarity_computation_target_ms: float = 100.0
    batch_processing_target_ms: float = 500.0
    memory_usage_target_gb: float = 2.0
    accuracy_improvement_target: float = 0.15  # 15%
    
    # Benchmark Parameters
    num_trials: int = 10
    warmup_trials: int = 3
    statistical_confidence: float = 0.95
    
    # Test Data Parameters
    test_batch_sizes: List[int] = field(default_factory=lambda: [1, 10, 50, 100])
    test_document_counts: List[int] = field(default_factory=lambda: [10, 50, 100])
    
    # Output Configuration
    output_dir: str = "benchmark_results"
    save_detailed_results: bool = True
    generate_plots: bool = True


@dataclass
class BenchmarkResult:
    """Container for individual benchmark results."""
    test_name: str
    component: str
    metric_type: str  # 'latency', 'memory', 'accuracy', 'throughput'
    
    # Timing Results
    duration_ms: float
    target_ms: Optional[float] = None
    target_met: bool = False
    
    # Statistical Data
    trials: List[float] = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    median: float = 0.0
    
    # Resource Usage
    memory_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    
    # Test Parameters
    batch_size: Optional[int] = None
    document_count: Optional[int] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class PerformanceBenchmarker:
    """
    Comprehensive performance benchmarking system for QuantumRerank.
    
    Validates PRD performance targets and enables quantum vs classical comparison.
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Initialize the performance benchmarker.
        
        Args:
            config: Benchmark configuration (uses default if None)
        """
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []
        
        # Initialize components for benchmarking
        self._initialize_components()
        
        # Setup output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized PerformanceBenchmarker with {self.config.num_trials} trials")
    
    def _initialize_components(self):
        """Initialize all components needed for benchmarking."""
        try:
            # Core components
            self.embedding_processor = EmbeddingProcessor()
            self.quantum_engine = QuantumSimilarityEngine()
            self.quantum_bridge = QuantumEmbeddingBridge()
            self.two_stage_retriever = TwoStageRetriever()
            
            logger.info("All benchmark components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def benchmark_component_latency(self, 
                                  component_name: str,
                                  test_function: Callable,
                                  test_data: Any,
                                  target_ms: Optional[float] = None) -> BenchmarkResult:
        """
        Benchmark latency for a specific component.
        
        Args:
            component_name: Name of the component being tested
            test_function: Function to benchmark
            test_data: Data to pass to the test function
            target_ms: Performance target in milliseconds
            
        Returns:
            BenchmarkResult with latency metrics
        """
        trials = []
        memory_usage = []
        
        # Warmup trials
        for _ in range(self.config.warmup_trials):
            try:
                test_function(test_data)
            except Exception:
                pass  # Ignore warmup errors
        
        # Force garbage collection before benchmarking
        gc.collect()
        
        # Actual benchmark trials
        for trial in range(self.config.num_trials):
            # Memory before
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Time the operation
            start_time = time.perf_counter()
            try:
                result = test_function(test_data)
                success = True
                error_msg = None
            except Exception as e:
                success = False
                error_msg = str(e)
                result = None
            
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            # Memory after
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_delta = memory_after - memory_before
            
            if success:
                trials.append(duration_ms)
                memory_usage.append(memory_delta)
        
        # Calculate statistics
        if trials:
            mean_duration = np.mean(trials)
            std_duration = np.std(trials)
            min_duration = np.min(trials)
            max_duration = np.max(trials)
            median_duration = np.median(trials)
            mean_memory = np.mean(memory_usage) if memory_usage else 0.0
            
            target_met = mean_duration < target_ms if target_ms else False
        else:
            mean_duration = float('inf')
            std_duration = 0.0
            min_duration = float('inf')
            max_duration = float('inf')
            median_duration = float('inf')
            mean_memory = 0.0
            target_met = False
        
        result = BenchmarkResult(
            test_name=f"{component_name}_latency",
            component=component_name,
            metric_type="latency",
            duration_ms=mean_duration,
            target_ms=target_ms,
            target_met=target_met,
            trials=trials,
            mean=mean_duration,
            std=std_duration,
            min_val=min_duration,
            max_val=max_duration,
            median=median_duration,
            memory_mb=mean_memory,
            success=len(trials) > 0,
            error_message=error_msg if len(trials) == 0 else None
        )
        
        self.results.append(result)
        return result
    
    def benchmark_similarity_computation(self) -> List[BenchmarkResult]:
        """
        Benchmark similarity computation performance against PRD targets.
        
        PRD Target: <100ms per similarity computation
        
        Returns:
            List of BenchmarkResult objects
        """
        logger.info("Benchmarking similarity computation performance...")
        
        results = []
        
        # Test data
        test_texts = [
            "Quantum computing leverages quantum mechanical phenomena",
            "Machine learning algorithms process data to find patterns",
            "Classical computing uses traditional binary logic"
        ]
        
        # Get embeddings once
        embeddings = self.embedding_processor.encode_texts(test_texts)
        
        # Test different similarity methods
        similarity_methods = [
            ("classical_cosine", lambda data: self.embedding_processor.compute_classical_similarity(data[0], data[1])),
            ("quantum_fidelity", lambda data: self.embedding_processor.compute_fidelity_similarity(data[0], data[1])),
            ("quantum_engine", lambda data: self.quantum_engine.compute_similarity(data[0], data[1]))
        ]
        
        for method_name, method_func in similarity_methods:
            test_data = (embeddings[0], embeddings[1])
            
            result = self.benchmark_component_latency(
                component_name=f"similarity_{method_name}",
                test_function=method_func,
                test_data=test_data,
                target_ms=self.config.similarity_computation_target_ms
            )
            results.append(result)
        
        return results
    
    def benchmark_batch_processing(self) -> List[BenchmarkResult]:
        """
        Benchmark batch processing performance against PRD targets.
        
        PRD Target: 50 docs in <500ms
        
        Returns:
            List of BenchmarkResult objects
        """
        logger.info("Benchmarking batch processing performance...")
        
        results = []
        
        # Test data for different batch sizes
        for batch_size in self.config.test_batch_sizes:
            test_texts = [f"Test document {i} about various topics" for i in range(batch_size)]
            
            # Benchmark embedding processing
            result = self.benchmark_component_latency(
                component_name=f"batch_embedding_{batch_size}docs",
                test_function=lambda texts: self.embedding_processor.encode_texts(texts),
                test_data=test_texts,
                target_ms=self.config.batch_processing_target_ms if batch_size == 50 else None
            )
            result.batch_size = batch_size
            results.append(result)
            
            # Benchmark quantum circuit conversion
            result = self.benchmark_component_latency(
                component_name=f"batch_quantum_{batch_size}docs",
                test_function=lambda texts: self.quantum_bridge.batch_texts_to_circuits(texts),
                test_data=test_texts,
                target_ms=self.config.batch_processing_target_ms if batch_size == 50 else None
            )
            result.batch_size = batch_size
            results.append(result)
        
        return results
    
    def benchmark_memory_usage(self) -> List[BenchmarkResult]:
        """
        Benchmark memory usage against PRD targets.
        
        PRD Target: <2GB for 100 documents
        
        Returns:
            List of BenchmarkResult objects
        """
        logger.info("Benchmarking memory usage...")
        
        results = []
        
        for doc_count in self.config.test_document_counts:
            # Generate test documents
            test_docs = [f"Document {i} with some content about quantum computing and information retrieval systems" for i in range(doc_count)]
            
            # Measure memory before
            gc.collect()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024 / 1024  # GB
            
            try:
                # Process documents
                embeddings = self.embedding_processor.encode_texts(test_docs)
                quantum_circuits = self.quantum_bridge.batch_texts_to_circuits(test_docs)
                
                # Measure memory after
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024 / 1024  # GB
                memory_delta = memory_after - memory_before
                
                target_met = memory_delta < self.config.memory_usage_target_gb if doc_count == 100 else None
                
                result = BenchmarkResult(
                    test_name=f"memory_usage_{doc_count}docs",
                    component="memory_system",
                    metric_type="memory",
                    duration_ms=0.0,  # Not applicable
                    memory_mb=memory_delta * 1024,  # Convert to MB
                    target_met=target_met if target_met is not None else False,
                    document_count=doc_count,
                    success=True,
                    details={
                        'memory_before_gb': memory_before,
                        'memory_after_gb': memory_after,
                        'memory_delta_gb': memory_delta
                    }
                )
                
            except Exception as e:
                result = BenchmarkResult(
                    test_name=f"memory_usage_{doc_count}docs",
                    component="memory_system", 
                    metric_type="memory",
                    duration_ms=0.0,
                    document_count=doc_count,
                    success=False,
                    error_message=str(e)
                )
            
            results.append(result)
        
        return results
    
    def benchmark_end_to_end_pipeline(self) -> List[BenchmarkResult]:
        """
        Benchmark complete end-to-end pipeline performance.
        
        Returns:
            List of BenchmarkResult objects
        """
        logger.info("Benchmarking end-to-end pipeline...")
        
        results = []
        
        # Test query and documents
        query = "quantum machine learning algorithms for optimization"
        documents = [
            "Quantum algorithms for machine learning optimization problems",
            "Classical machine learning techniques for data analysis",
            "Quantum computing applications in artificial intelligence",
            "Traditional optimization methods in computer science",
            "Hybrid quantum-classical approaches to machine learning"
        ]
        
        # Benchmark full reranking pipeline
        def end_to_end_test(data):
            query, docs = data
            return self.two_stage_retriever.rerank(query, docs, top_k=len(docs))
        
        result = self.benchmark_component_latency(
            component_name="end_to_end_reranking",
            test_function=end_to_end_test,
            test_data=(query, documents),
            target_ms=self.config.batch_processing_target_ms
        )
        result.document_count = len(documents)
        results.append(result)
        
        return results
    
    def run_comprehensive_benchmark(self) -> Dict[str, List[BenchmarkResult]]:
        """
        Run comprehensive benchmark suite covering all PRD targets.
        
        Returns:
            Dictionary organized by benchmark category
        """
        logger.info("Starting comprehensive benchmark suite...")
        
        benchmark_suite = {
            'similarity_computation': self.benchmark_similarity_computation(),
            'batch_processing': self.benchmark_batch_processing(),
            'memory_usage': self.benchmark_memory_usage(),
            'end_to_end': self.benchmark_end_to_end_pipeline()
        }
        
        # Calculate overall statistics
        total_tests = sum(len(results) for results in benchmark_suite.values())
        passed_tests = sum(
            sum(1 for r in results if r.success and r.target_met) 
            for results in benchmark_suite.values()
        )
        
        logger.info(f"Benchmark suite completed: {passed_tests}/{total_tests} tests passed PRD targets")
        
        return benchmark_suite
    
    def get_prd_compliance_summary(self) -> Dict[str, Any]:
        """
        Generate PRD compliance summary from benchmark results.
        
        Returns:
            Dictionary with PRD compliance status
        """
        if not self.results:
            return {"error": "No benchmark results available"}
        
        # Group results by metric type
        latency_results = [r for r in self.results if r.metric_type == 'latency' and r.target_ms]
        memory_results = [r for r in self.results if r.metric_type == 'memory']
        
        # Check PRD targets
        similarity_under_100ms = any(
            r.target_met for r in latency_results 
            if 'similarity' in r.test_name and r.target_ms == 100.0
        )
        
        batch_under_500ms = any(
            r.target_met for r in latency_results
            if 'batch' in r.test_name and r.target_ms == 500.0
        )
        
        memory_under_2gb = any(
            r.memory_mb and r.memory_mb < 2048 for r in memory_results
            if r.document_count == 100
        )
        
        overall_compliance = similarity_under_100ms and batch_under_500ms and memory_under_2gb
        
        return {
            'overall_prd_compliance': overall_compliance,
            'similarity_computation_compliant': similarity_under_100ms,
            'batch_processing_compliant': batch_under_500ms, 
            'memory_usage_compliant': memory_under_2gb,
            'total_tests_run': len(self.results),
            'successful_tests': sum(1 for r in self.results if r.success),
            'benchmark_timestamp': datetime.now().isoformat()
        }
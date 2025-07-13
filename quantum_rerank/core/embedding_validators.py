"""
Embedding validation and performance monitoring for QuantumRerank.

This module provides validation and benchmarking tools for the embedding
processing pipeline, ensuring compliance with PRD requirements.

Based on:
- PRD Section 4.1: System Requirements and Performance Targets
- Task 03 specifications for embedding validation
"""

import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from .embeddings import EmbeddingProcessor
from .quantum_embedding_bridge import QuantumEmbeddingBridge

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result container for validation operations."""
    passed: bool
    score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class PerformanceBenchmark:
    """Container for performance benchmark results."""
    test_name: str
    duration_ms: float
    success: bool
    target_met: bool
    target_value: float
    actual_value: float
    details: Dict[str, Any] = field(default_factory=dict)


class EmbeddingValidator:
    """
    Validator for embedding processing pipeline.
    
    Validates embedding quality, quantum compatibility, and performance
    against PRD specifications.
    """
    
    def __init__(self, embedding_processor: Optional[EmbeddingProcessor] = None):
        """
        Initialize embedding validator.
        
        Args:
            embedding_processor: Optional EmbeddingProcessor instance
        """
        self.embedding_processor = embedding_processor or EmbeddingProcessor()
        
        # PRD performance targets
        self.prd_targets = {
            'single_encoding_ms': 100,
            'similarity_computation_ms': 100,
            'batch_processing_efficiency': 0.8,  # 80% efficiency target
            'memory_per_100_docs_gb': 2.0,
            'embedding_quality_threshold': 0.7,
            'quantum_compatibility_rate': 0.95
        }
        
        logger.info("Initialized EmbeddingValidator with PRD targets")
    
    def validate_embedding_basic_properties(self, embeddings: np.ndarray) -> ValidationResult:
        """
        Validate basic properties of embeddings.
        
        Args:
            embeddings: Embedding array to validate
            
        Returns:
            ValidationResult with basic property validation
        """
        errors = []
        warnings = []
        details = {}
        
        try:
            # Check shape and dimensions
            if embeddings.ndim != 2:
                errors.append(f"Expected 2D array, got {embeddings.ndim}D")
            else:
                details['shape'] = embeddings.shape
                details['num_embeddings'] = embeddings.shape[0]
                details['embedding_dim'] = embeddings.shape[1]
            
            # Check for finite values
            finite_mask = np.isfinite(embeddings)
            finite_rate = np.mean(finite_mask)
            details['finite_value_rate'] = finite_rate
            
            if finite_rate < 1.0:
                warnings.append(f"Non-finite values detected: {(1-finite_rate)*100:.2f}% invalid")
            
            # Check value ranges
            if embeddings.size > 0:
                details['value_stats'] = {
                    'min': float(np.min(embeddings)),
                    'max': float(np.max(embeddings)),
                    'mean': float(np.mean(embeddings)),
                    'std': float(np.std(embeddings))
                }
                
                # Check for extreme values
                if np.abs(details['value_stats']['min']) > 10 or details['value_stats']['max'] > 10:
                    warnings.append("Extreme embedding values detected (>10 or <-10)")
            
            # Check normalization
            if embeddings.ndim == 2 and embeddings.shape[0] > 0:
                norms = np.linalg.norm(embeddings, axis=1)
                details['norm_stats'] = {
                    'mean': float(np.mean(norms)),
                    'std': float(np.std(norms)),
                    'min': float(np.min(norms)),
                    'max': float(np.max(norms))
                }
                
                # Check if embeddings are normalized
                normalized = np.allclose(norms, 1.0, atol=1e-6)
                details['normalized'] = normalized
                
                if not normalized:
                    warnings.append("Embeddings are not L2 normalized")
            
            # Calculate overall score
            score = 1.0
            if errors:
                score = 0.0
            elif warnings:
                score = 0.7
            
            passed = len(errors) == 0
            
            return ValidationResult(
                passed=passed,
                score=score,
                details=details,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            return ValidationResult(
                passed=False,
                score=0.0,
                errors=[f"Validation failed: {str(e)}"]
            )
    
    def validate_quantum_compatibility(self, embeddings: np.ndarray, 
                                     n_qubits: int = 4) -> ValidationResult:
        """
        Validate embeddings are compatible with quantum processing.
        
        Args:
            embeddings: Embeddings to validate
            n_qubits: Number of qubits for quantum encoding
            
        Returns:
            ValidationResult for quantum compatibility
        """
        errors = []
        warnings = []
        details = {}
        
        try:
            # Test quantum preprocessing
            processed_embeddings, metadata = self.embedding_processor.preprocess_for_quantum(
                embeddings, n_qubits
            )
            
            details['preprocessing_metadata'] = metadata
            details['processed_shape'] = processed_embeddings.shape
            
            # Validate processed embeddings
            max_amplitudes = 2 ** n_qubits
            
            # Check dimensions
            if processed_embeddings.shape[1] != max_amplitudes:
                errors.append(f"Processed embedding dim {processed_embeddings.shape[1]} != expected {max_amplitudes}")
            
            # Check normalization (critical for quantum states)
            norms = np.linalg.norm(processed_embeddings, axis=1)
            normalized_correctly = np.allclose(norms, 1.0, atol=1e-6)
            details['quantum_normalized'] = normalized_correctly
            
            if not normalized_correctly:
                errors.append("Processed embeddings not properly normalized for quantum states")
            
            # Check for complex issues
            if np.any(np.iscomplex(processed_embeddings)):
                warnings.append("Complex values detected in processed embeddings")
            
            # Test compatibility rate
            compatibility_rate = np.mean(norms > 0)  # Non-zero embeddings
            details['compatibility_rate'] = compatibility_rate
            
            if compatibility_rate < self.prd_targets['quantum_compatibility_rate']:
                warnings.append(f"Compatibility rate {compatibility_rate:.2f} below target {self.prd_targets['quantum_compatibility_rate']}")
            
            # Calculate score
            score = 1.0
            if errors:
                score = 0.0
            elif compatibility_rate < self.prd_targets['quantum_compatibility_rate']:
                score = 0.6
            elif warnings:
                score = 0.8
            
            passed = len(errors) == 0
            
            return ValidationResult(
                passed=passed,
                score=score,
                details=details,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            return ValidationResult(
                passed=False,
                score=0.0,
                errors=[f"Quantum compatibility validation failed: {str(e)}"]
            )
    
    def validate_similarity_quality(self, test_texts: Optional[List[str]] = None) -> ValidationResult:
        """
        Validate quality of similarity computations.
        
        Args:
            test_texts: Optional test texts for validation
            
        Returns:
            ValidationResult for similarity quality
        """
        if test_texts is None:
            test_texts = [
                "quantum computing and quantum mechanics",
                "quantum physics and quantum theory",  # Should be similar to above
                "machine learning and artificial intelligence",
                "classical computing and traditional algorithms",  # Should be different from quantum
                "natural language processing"
            ]
        
        errors = []
        warnings = []
        details = {}
        
        try:
            # Generate embeddings
            embeddings = self.embedding_processor.encode_texts(test_texts)
            
            # Compute similarity matrix
            n_texts = len(test_texts)
            cosine_similarities = np.zeros((n_texts, n_texts))
            fidelity_similarities = np.zeros((n_texts, n_texts))
            
            for i in range(n_texts):
                for j in range(n_texts):
                    if i != j:
                        cosine_sim = self.embedding_processor.compute_classical_similarity(
                            embeddings[i], embeddings[j]
                        )
                        fidelity_sim = self.embedding_processor.compute_fidelity_similarity(
                            embeddings[i], embeddings[j]
                        )
                        cosine_similarities[i, j] = cosine_sim
                        fidelity_similarities[i, j] = fidelity_sim
                    else:
                        cosine_similarities[i, j] = 1.0
                        fidelity_similarities[i, j] = 1.0
            
            details['cosine_similarity_matrix'] = cosine_similarities.tolist()
            details['fidelity_similarity_matrix'] = fidelity_similarities.tolist()
            
            # Check expected relationships
            # Text 0 and 1 should be more similar (both about quantum)
            quantum_similarity_cosine = cosine_similarities[0, 1]
            quantum_similarity_fidelity = fidelity_similarities[0, 1]
            
            # Text 0 and 3 should be less similar (quantum vs classical)
            quantum_classical_cosine = cosine_similarities[0, 3]
            quantum_classical_fidelity = fidelity_similarities[0, 3]
            
            details['expected_relationships'] = {
                'quantum_quantum_cosine': quantum_similarity_cosine,
                'quantum_classical_cosine': quantum_classical_cosine,
                'quantum_quantum_fidelity': quantum_similarity_fidelity,
                'quantum_classical_fidelity': quantum_classical_fidelity,
                'relationship_correct': quantum_similarity_cosine > quantum_classical_cosine
            }
            
            # Validate similarity ranges
            cosine_valid = np.all((cosine_similarities >= 0) & (cosine_similarities <= 1))
            fidelity_valid = np.all((fidelity_similarities >= 0) & (fidelity_similarities <= 1))
            
            details['similarity_ranges_valid'] = {
                'cosine': cosine_valid,
                'fidelity': fidelity_valid
            }
            
            if not cosine_valid:
                errors.append("Cosine similarities outside valid range [0, 1]")
            if not fidelity_valid:
                errors.append("Fidelity similarities outside valid range [0, 1]")
            
            # Check semantic consistency
            if not details['expected_relationships']['relationship_correct']:
                warnings.append("Semantic relationships not preserved in similarities")
            
            # Calculate overall quality score
            avg_cosine = np.mean(cosine_similarities[np.triu_indices_from(cosine_similarities, k=1)])
            details['average_cosine_similarity'] = avg_cosine
            
            quality_score = avg_cosine
            if quality_score < self.prd_targets['embedding_quality_threshold']:
                warnings.append(f"Average similarity {quality_score:.3f} below quality threshold {self.prd_targets['embedding_quality_threshold']}")
            
            # Calculate final score
            score = 1.0
            if errors:
                score = 0.0
            elif quality_score < self.prd_targets['embedding_quality_threshold']:
                score = 0.6
            elif warnings:
                score = 0.8
            
            passed = len(errors) == 0
            
            return ValidationResult(
                passed=passed,
                score=score,
                details=details,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            return ValidationResult(
                passed=False,
                score=0.0,
                errors=[f"Similarity quality validation failed: {str(e)}"]
            )
    
    def run_performance_benchmarks(self) -> List[PerformanceBenchmark]:
        """
        Run comprehensive performance benchmarks against PRD targets.
        
        Returns:
            List of PerformanceBenchmark results
        """
        benchmarks = []
        test_texts = [
            "quantum computing research and development",
            "machine learning algorithms for classification",
            "information retrieval and search systems",
            "natural language processing applications"
        ]
        
        # Benchmark 1: Single text encoding
        start_time = time.time()
        try:
            embedding = self.embedding_processor.encode_single_text(test_texts[0])
            duration = (time.time() - start_time) * 1000
            success = True
            target_met = duration < self.prd_targets['single_encoding_ms']
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            success = False
            target_met = False
        
        benchmarks.append(PerformanceBenchmark(
            test_name="single_text_encoding",
            duration_ms=duration,
            success=success,
            target_met=target_met,
            target_value=self.prd_targets['single_encoding_ms'],
            actual_value=duration,
            details={'text_length': len(test_texts[0])}
        ))
        
        # Benchmark 2: Batch encoding
        start_time = time.time()
        try:
            embeddings = self.embedding_processor.encode_texts(test_texts)
            duration = (time.time() - start_time) * 1000
            per_text_duration = duration / len(test_texts)
            success = True
            target_met = per_text_duration < self.prd_targets['single_encoding_ms']
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            per_text_duration = duration
            success = False
            target_met = False
        
        benchmarks.append(PerformanceBenchmark(
            test_name="batch_text_encoding",
            duration_ms=duration,
            success=success,
            target_met=target_met,
            target_value=self.prd_targets['single_encoding_ms'] * len(test_texts),
            actual_value=duration,
            details={
                'batch_size': len(test_texts),
                'per_text_ms': per_text_duration
            }
        ))
        
        # Benchmark 3: Similarity computation
        if 'embeddings' in locals() and embeddings is not None:
            start_time = time.time()
            try:
                similarity = self.embedding_processor.compute_classical_similarity(
                    embeddings[0], embeddings[1]
                )
                duration = (time.time() - start_time) * 1000
                success = True
                target_met = duration < self.prd_targets['similarity_computation_ms']
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                success = False
                target_met = False
            
            benchmarks.append(PerformanceBenchmark(
                test_name="similarity_computation",
                duration_ms=duration,
                success=success,
                target_met=target_met,
                target_value=self.prd_targets['similarity_computation_ms'],
                actual_value=duration,
                details={'similarity_value': similarity if success else None}
            ))
        
        # Benchmark 4: Quantum preprocessing
        if 'embeddings' in locals() and embeddings is not None:
            start_time = time.time()
            try:
                processed, metadata = self.embedding_processor.preprocess_for_quantum(embeddings)
                duration = (time.time() - start_time) * 1000
                success = True
                target_met = duration < 50  # Reasonable target for preprocessing
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                success = False
                target_met = False
            
            benchmarks.append(PerformanceBenchmark(
                test_name="quantum_preprocessing",
                duration_ms=duration,
                success=success,
                target_met=target_met,
                target_value=50.0,
                actual_value=duration,
                details={'batch_size': len(embeddings)}
            ))
        
        logger.info(f"Completed {len(benchmarks)} performance benchmarks")
        return benchmarks
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Returns:
            Complete validation report dictionary
        """
        logger.info("Generating comprehensive validation report")
        
        report = {
            'timestamp': time.time(),
            'validator_config': {
                'prd_targets': self.prd_targets,
                'embedding_model': self.embedding_processor.config.model_name,
                'embedding_dim': self.embedding_processor.config.embedding_dim
            },
            'validations': {},
            'benchmarks': [],
            'summary': {}
        }
        
        # Test embeddings
        test_texts = [
            "quantum computing and quantum algorithms",
            "machine learning and artificial intelligence",
            "information retrieval and search systems",
            "natural language processing and linguistics"
        ]
        
        try:
            embeddings = self.embedding_processor.encode_texts(test_texts)
            
            # Run validations
            report['validations']['basic_properties'] = self.validate_embedding_basic_properties(embeddings)
            report['validations']['quantum_compatibility'] = self.validate_quantum_compatibility(embeddings)
            report['validations']['similarity_quality'] = self.validate_similarity_quality(test_texts)
            
            # Run benchmarks
            report['benchmarks'] = self.run_performance_benchmarks()
            
            # Generate summary
            all_validations_passed = all(
                v.passed for v in report['validations'].values()
            )
            avg_validation_score = np.mean([
                v.score for v in report['validations'].values()
            ])
            all_benchmarks_passed = all(
                b.target_met for b in report['benchmarks']
            )
            
            report['summary'] = {
                'overall_validation_passed': all_validations_passed,
                'average_validation_score': avg_validation_score,
                'all_benchmarks_passed': all_benchmarks_passed,
                'prd_compliance': all_validations_passed and all_benchmarks_passed,
                'total_validations': len(report['validations']),
                'total_benchmarks': len(report['benchmarks']),
                'passed_validations': sum(1 for v in report['validations'].values() if v.passed),
                'passed_benchmarks': sum(1 for b in report['benchmarks'] if b.target_met)
            }
            
        except Exception as e:
            report['error'] = str(e)
            report['summary'] = {
                'overall_validation_passed': False,
                'prd_compliance': False,
                'error': str(e)
            }
        
        logger.info(f"Validation report generated: {report['summary'].get('prd_compliance', False)} PRD compliant")
        return report
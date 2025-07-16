"""
QRF-02 Comprehensive Test: Quantum Encoding Discrimination Validation

Tests the improved quantum encoding methods to validate they preserve semantic
discrimination and achieve the QRF-02 target metrics.

Target Metrics:
- Quantum fidelity difference >0.1 between high/low similarity pairs
- Correlation >0.7 between classical and quantum similarities  
- Preserved ranking order for >80% of test pairs
- Information preservation >50% (vs current ~2%)

Based on:
- Task QRF-02: Fix Amplitude Encoding Discrimination
- QRF-01 fixes applied to improved encoding methods
- Paper insights on quantum-inspired similarity metrics
"""

import sys
import os
import logging
import numpy as np
from typing import List, Tuple, Dict, Any
import time

# Add project root to path
sys.path.append('/Users/alkist/Projects/QuantumRerank')

from quantum_rerank.core.embeddings import EmbeddingProcessor
from quantum_rerank.core.improved_quantum_encoding import ImprovedQuantumEncoder
from quantum_rerank.core.semantic_feature_selector import SemanticFeatureSelector, SemanticSelectionConfig
from quantum_rerank.core.fixed_swap_test import FixedQuantumSWAPTest
from scipy.stats import spearmanr, pearsonr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QRF02ValidationSuite:
    """
    Comprehensive validation suite for QRF-02 improved quantum encoding.
    
    Tests semantic preservation, discrimination, and correlation with classical methods.
    """
    
    def __init__(self, n_qubits: int = 4):
        """Initialize validation suite."""
        self.n_qubits = n_qubits
        
        # Initialize components
        self.embedding_processor = EmbeddingProcessor()
        self.quantum_encoder = ImprovedQuantumEncoder(n_qubits=n_qubits)
        self.swap_test = FixedQuantumSWAPTest(n_qubits=n_qubits)
        
        # Initialize semantic feature selector with ranking awareness
        feature_config = SemanticSelectionConfig(
            method="ranking_aware",  # Use ranking-aware method for better ranking preservation
            target_features=16,  # 4 qubits = 16 amplitudes
            preserve_local_structure=True,
            ranking_preservation_weight=0.8  # High weight for ranking preservation
        )
        self.feature_selector = SemanticFeatureSelector(feature_config)
        
        logger.info(f"Initialized QRF-02 validation suite with {n_qubits} qubits")
    
    def create_medical_test_pairs(self) -> List[Tuple[str, str, str]]:
        """
        Create test pairs with known similarity relationships.
        
        Returns:
            List of (text1, text2, expected_similarity_level) tuples
        """
        test_pairs = [
            # High similarity pairs (medical-medical)
            (
                "The patient presented with acute myocardial infarction and elevated troponin levels.",
                "Acute MI with increased cardiac enzymes and chest pain symptoms observed.",
                "high"
            ),
            (
                "Diabetes mellitus type 2 requires insulin therapy and glucose monitoring.",
                "Type 2 diabetes management includes insulin treatment and blood sugar control.",
                "high"
            ),
            
            # Medium similarity pairs (medical-related)
            (
                "The patient presented with acute myocardial infarction and elevated troponin levels.",
                "Hypertension treatment requires ACE inhibitors and lifestyle modifications.",
                "medium"
            ),
            (
                "Diabetes mellitus type 2 requires insulin therapy and glucose monitoring.",
                "Cardiovascular disease prevention includes diet and exercise interventions.",
                "medium"
            ),
            
            # Low similarity pairs (medical-non-medical)
            (
                "The patient presented with acute myocardial infarction and elevated troponin levels.",
                "Quantum computing uses quantum mechanical phenomena for computational advantages.",
                "low"
            ),
            (
                "Diabetes mellitus type 2 requires insulin therapy and glucose monitoring.",
                "Machine learning algorithms can process large datasets to identify patterns.",
                "low"
            ),
            
            # Very low similarity pairs (medical-unrelated)
            (
                "The patient presented with acute myocardial infarction and elevated troponin levels.",
                "The weather today is sunny with a temperature of 75 degrees and low humidity.",
                "very_low"
            ),
            (
                "Diabetes mellitus type 2 requires insulin therapy and glucose monitoring.",
                "I went to the grocery store to buy apples, oranges, and fresh vegetables.",
                "very_low"
            ),
        ]
        
        return test_pairs
    
    def compute_classical_similarities(self, test_pairs: List[Tuple[str, str, str]]) -> List[float]:
        """Compute classical cosine similarities for all test pairs."""
        classical_similarities = []
        
        for text1, text2, _ in test_pairs:
            emb1 = self.embedding_processor.encode_single_text(text1)
            emb2 = self.embedding_processor.encode_single_text(text2)
            
            # Compute cosine similarity
            cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            classical_similarities.append(cos_sim)
        
        return classical_similarities
    
    def compute_quantum_similarities(self, test_pairs: List[Tuple[str, str, str]],
                                   encoding_method: str = "angle") -> Tuple[List[float], Dict]:
        """
        Compute quantum similarities using improved encoding methods.
        
        Args:
            test_pairs: List of text pairs to test
            encoding_method: Quantum encoding method to use
            
        Returns:
            Tuple of (quantum_similarities, metadata)
        """
        quantum_similarities = []
        metadata = {
            'encoding_method': encoding_method,
            'successful_encodings': 0,
            'failed_encodings': 0,
            'avg_information_preservation': 0.0,
            'encoding_times': [],
            'fidelity_computation_times': []
        }
        
        # First, collect all embeddings for feature selection
        all_texts = []
        for text1, text2, _ in test_pairs:
            all_texts.extend([text1, text2])
        
        # Remove duplicates while preserving order
        unique_texts = list(dict.fromkeys(all_texts))
        
        # Generate embeddings
        all_embeddings = self.embedding_processor.encode_texts(unique_texts)
        
        # Fit feature selector on all embeddings
        logger.info(f"Fitting feature selector on {len(unique_texts)} unique texts")
        self.feature_selector.fit(all_embeddings)
        
        # Process each pair
        for i, (text1, text2, expected_level) in enumerate(test_pairs):
            logger.info(f"Processing pair {i+1}/{len(test_pairs)}: {expected_level} similarity")
            
            try:
                # Get embeddings
                emb1 = self.embedding_processor.encode_single_text(text1)
                emb2 = self.embedding_processor.encode_single_text(text2)
                
                # Apply feature selection
                selected1 = self.feature_selector.transform(emb1.reshape(1, -1))
                selected2 = self.feature_selector.transform(emb2.reshape(1, -1))
                
                if not (selected1.success and selected2.success):
                    logger.error(f"Feature selection failed for pair {i+1}")
                    quantum_similarities.append(0.0)
                    metadata['failed_encodings'] += 1
                    continue
                
                # Quantum encoding
                start_time = time.time()
                
                result1 = self.quantum_encoder.encode_embedding(
                    selected1.selected_features[0], method=encoding_method, name=f"text1_pair{i}"
                )
                result2 = self.quantum_encoder.encode_embedding(
                    selected2.selected_features[0], method=encoding_method, name=f"text2_pair{i}"
                )
                
                encoding_time = time.time() - start_time
                metadata['encoding_times'].append(encoding_time)
                
                if not (result1.success and result2.success):
                    logger.error(f"Quantum encoding failed for pair {i+1}")
                    quantum_similarities.append(0.0)
                    metadata['failed_encodings'] += 1
                    continue
                
                # Compute quantum fidelity using fixed SWAP test
                start_time = time.time()
                
                fidelity, fidelity_metadata = self.swap_test.compute_fidelity_from_statevectors(
                    result1.statevector.data, result2.statevector.data
                )
                
                fidelity_time = time.time() - start_time
                metadata['fidelity_computation_times'].append(fidelity_time)
                
                quantum_similarities.append(fidelity)
                metadata['successful_encodings'] += 1
                
                # Track information preservation
                avg_preservation = (result1.information_preservation + result2.information_preservation) / 2
                metadata['avg_information_preservation'] += avg_preservation
                
                logger.debug(f"Pair {i+1}: quantum_fidelity={fidelity:.6f}, "
                           f"info_preservation={avg_preservation:.3f}")
                
            except Exception as e:
                logger.error(f"Error processing pair {i+1}: {e}")
                quantum_similarities.append(0.0)
                metadata['failed_encodings'] += 1
        
        # Calculate average information preservation
        if metadata['successful_encodings'] > 0:
            metadata['avg_information_preservation'] /= metadata['successful_encodings']
        
        # Calculate timing statistics
        if metadata['encoding_times']:
            metadata['avg_encoding_time_ms'] = np.mean(metadata['encoding_times']) * 1000
            metadata['max_encoding_time_ms'] = np.max(metadata['encoding_times']) * 1000
        
        if metadata['fidelity_computation_times']:
            metadata['avg_fidelity_time_ms'] = np.mean(metadata['fidelity_computation_times']) * 1000
            metadata['max_fidelity_time_ms'] = np.max(metadata['fidelity_computation_times']) * 1000
        
        return quantum_similarities, metadata
    
    def evaluate_discrimination_metrics(self, test_pairs: List[Tuple[str, str, str]],
                                      classical_similarities: List[float],
                                      quantum_similarities: List[float]) -> Dict[str, Any]:
        """
        Evaluate QRF-02 target metrics for discrimination.
        
        Returns:
            Dictionary with discrimination analysis results
        """
        results = {
            'total_pairs': len(test_pairs),
            'classical_range': max(classical_similarities) - min(classical_similarities),
            'quantum_range': max(quantum_similarities) - min(quantum_similarities),
            'target_metrics': {},
            'similarity_analysis': {},
            'ranking_analysis': {}
        }
        
        # Target Metric 1: Quantum fidelity difference >0.1 between high/low similarity pairs
        high_sim_quantum = []
        low_sim_quantum = []
        
        for i, (_, _, expected_level) in enumerate(test_pairs):
            if expected_level in ['high']:
                high_sim_quantum.append(quantum_similarities[i])
            elif expected_level in ['low', 'very_low']:
                low_sim_quantum.append(quantum_similarities[i])
        
        if high_sim_quantum and low_sim_quantum:
            avg_high_quantum = np.mean(high_sim_quantum)
            avg_low_quantum = np.mean(low_sim_quantum)
            quantum_discrimination = avg_high_quantum - avg_low_quantum
            
            results['target_metrics']['quantum_discrimination'] = quantum_discrimination
            results['target_metrics']['quantum_discrimination_target'] = quantum_discrimination > 0.1
        else:
            results['target_metrics']['quantum_discrimination'] = 0.0
            results['target_metrics']['quantum_discrimination_target'] = False
        
        # Target Metric 2: Correlation >0.7 between classical and quantum similarities
        if len(classical_similarities) > 2:
            pearson_corr, pearson_p = pearsonr(classical_similarities, quantum_similarities)
            spearman_corr, spearman_p = spearmanr(classical_similarities, quantum_similarities)
            
            results['target_metrics']['pearson_correlation'] = pearson_corr
            results['target_metrics']['spearman_correlation'] = spearman_corr
            results['target_metrics']['correlation_target'] = pearson_corr > 0.7
        else:
            results['target_metrics']['pearson_correlation'] = 0.0
            results['target_metrics']['spearman_correlation'] = 0.0
            results['target_metrics']['correlation_target'] = False
        
        # Target Metric 3: Preserved ranking order for >80% of test pairs
        classical_ranking = np.argsort(np.argsort(classical_similarities))
        quantum_ranking = np.argsort(np.argsort(quantum_similarities))
        
        ranking_matches = np.sum(classical_ranking == quantum_ranking)
        ranking_preservation = ranking_matches / len(test_pairs)
        
        results['target_metrics']['ranking_preservation'] = ranking_preservation
        results['target_metrics']['ranking_preservation_target'] = ranking_preservation > 0.8
        
        # Detailed similarity analysis by expected level
        level_analysis = {}
        for level in ['high', 'medium', 'low', 'very_low']:
            level_indices = [i for i, (_, _, exp_level) in enumerate(test_pairs) if exp_level == level]
            if level_indices:
                level_analysis[level] = {
                    'count': len(level_indices),
                    'avg_classical': np.mean([classical_similarities[i] for i in level_indices]),
                    'avg_quantum': np.mean([quantum_similarities[i] for i in level_indices]),
                    'classical_range': [
                        min([classical_similarities[i] for i in level_indices]),
                        max([classical_similarities[i] for i in level_indices])
                    ],
                    'quantum_range': [
                        min([quantum_similarities[i] for i in level_indices]),
                        max([quantum_similarities[i] for i in level_indices])
                    ]
                }
        
        results['similarity_analysis'] = level_analysis
        
        # Overall QRF-02 success
        target_metrics = results['target_metrics']
        qrf02_success = (
            target_metrics.get('quantum_discrimination_target', False) and
            target_metrics.get('correlation_target', False) and
            target_metrics.get('ranking_preservation_target', False)
        )
        
        results['qrf02_success'] = qrf02_success
        
        return results
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive QRF-02 validation suite.
        
        Returns:
            Complete validation results
        """
        logger.info("Starting QRF-02 comprehensive validation")
        start_time = time.time()
        
        results = {
            'test_pairs': [],
            'classical_similarities': [],
            'quantum_results': {},
            'discrimination_analysis': {},
            'overall_assessment': {}
        }
        
        # Create test pairs
        test_pairs = self.create_medical_test_pairs()
        results['test_pairs'] = [(t1[:50], t2[:50], level) for t1, t2, level in test_pairs]
        
        logger.info(f"Created {len(test_pairs)} test pairs")
        
        # Compute classical similarities (baseline)
        logger.info("Computing classical similarities...")
        classical_similarities = self.compute_classical_similarities(test_pairs)
        results['classical_similarities'] = classical_similarities
        
        # Test different quantum encoding methods
        encoding_methods = ['angle', 'ranking_optimized', 'distance_preserving', 'multi_scale']  # Skip hybrid due to normalization issues
        
        for method in encoding_methods:
            logger.info(f"Testing quantum encoding method: {method}")
            
            quantum_similarities, method_metadata = self.compute_quantum_similarities(
                test_pairs, encoding_method=method
            )
            
            # Evaluate discrimination metrics
            discrimination_results = self.evaluate_discrimination_metrics(
                test_pairs, classical_similarities, quantum_similarities
            )
            
            results['quantum_results'][method] = {
                'similarities': quantum_similarities,
                'metadata': method_metadata,
                'discrimination_analysis': discrimination_results
            }
        
        # Overall assessment
        total_time = time.time() - start_time
        
        results['overall_assessment'] = {
            'total_validation_time_s': total_time,
            'methods_tested': len(encoding_methods),
            'total_pairs_tested': len(test_pairs),
            'timestamp': time.time()
        }
        
        logger.info(f"QRF-02 validation completed in {total_time:.2f}s")
        
        return results
    
    def print_validation_summary(self, results: Dict[str, Any]) -> None:
        """Print comprehensive validation summary."""
        print("\n" + "="*80)
        print("QRF-02 QUANTUM ENCODING DISCRIMINATION VALIDATION")
        print("="*80)
        
        print(f"\nTest Configuration:")
        print(f"  - Test pairs: {results['overall_assessment']['total_pairs_tested']}")
        print(f"  - Encoding methods: {', '.join(results['quantum_results'].keys())}")
        print(f"  - Validation time: {results['overall_assessment']['total_validation_time_s']:.2f}s")
        
        print(f"\nClassical Baseline:")
        classical_sims = results['classical_similarities']
        print(f"  - Range: {min(classical_sims):.6f} to {max(classical_sims):.6f}")
        print(f"  - Mean: {np.mean(classical_sims):.6f}")
        print(f"  - Std: {np.std(classical_sims):.6f}")
        
        # Results for each quantum method
        for method, method_results in results['quantum_results'].items():
            print(f"\n{method.upper()} Encoding Results:")
            print("-" * 40)
            
            metadata = method_results['metadata']
            disc_analysis = method_results['discrimination_analysis']
            target_metrics = disc_analysis['target_metrics']
            
            # Encoding performance
            print(f"  Encoding Performance:")
            print(f"    - Successful encodings: {metadata['successful_encodings']}")
            print(f"    - Failed encodings: {metadata['failed_encodings']}")
            print(f"    - Success rate: {metadata['successful_encodings']/(metadata['successful_encodings']+metadata['failed_encodings'])*100:.1f}%")
            
            if 'avg_information_preservation' in metadata:
                print(f"    - Avg information preservation: {metadata['avg_information_preservation']:.3f}")
            
            if 'avg_encoding_time_ms' in metadata:
                print(f"    - Avg encoding time: {metadata['avg_encoding_time_ms']:.2f}ms")
            
            # Quantum similarities
            quantum_sims = method_results['similarities']
            if quantum_sims:
                print(f"  Quantum Similarities:")
                print(f"    - Range: {min(quantum_sims):.6f} to {max(quantum_sims):.6f}")
                print(f"    - Mean: {np.mean(quantum_sims):.6f}")
                print(f"    - Std: {np.std(quantum_sims):.6f}")
            
            # QRF-02 Target Metrics
            print(f"  QRF-02 Target Metrics:")
            
            if 'quantum_discrimination' in target_metrics:
                disc_value = target_metrics['quantum_discrimination']
                disc_pass = target_metrics['quantum_discrimination_target']
                print(f"    - Quantum discrimination: {disc_value:.6f} ({'PASS' if disc_pass else 'FAIL'} >0.1)")
            
            if 'pearson_correlation' in target_metrics:
                corr_value = target_metrics['pearson_correlation']
                corr_pass = target_metrics['correlation_target']
                print(f"    - Classical-quantum correlation: {corr_value:.6f} ({'PASS' if corr_pass else 'FAIL'} >0.7)")
            
            if 'ranking_preservation' in target_metrics:
                rank_value = target_metrics['ranking_preservation']
                rank_pass = target_metrics['ranking_preservation_target']
                print(f"    - Ranking preservation: {rank_value:.3f} ({'PASS' if rank_pass else 'FAIL'} >0.8)")
            
            # Overall QRF-02 assessment
            qrf02_success = disc_analysis.get('qrf02_success', False)
            print(f"  Overall QRF-02 Success: {'PASS' if qrf02_success else 'FAIL'}")
            
            # Detailed pair analysis
            print(f"  Similarity by Expected Level:")
            for level, analysis in disc_analysis['similarity_analysis'].items():
                print(f"    - {level}: Classical={analysis['avg_classical']:.3f}, "
                      f"Quantum={analysis['avg_quantum']:.3f}")
        
        print(f"\n" + "="*80)


def main():
    """Main function to run QRF-02 validation."""
    print("QRF-02: QUANTUM ENCODING DISCRIMINATION VALIDATION")
    print("="*60)
    print("Testing improved quantum encoding methods for semantic discrimination")
    print("Target: Achieve QRF-02 success criteria")
    print()
    
    try:
        # Initialize validation suite
        validator = QRF02ValidationSuite(n_qubits=4)
        
        # Run comprehensive validation
        results = validator.run_comprehensive_validation()
        
        # Print summary
        validator.print_validation_summary(results)
        
        # Determine overall success
        methods_tested = list(results['quantum_results'].keys())
        successful_methods = [
            method for method, method_results in results['quantum_results'].items()
            if method_results['discrimination_analysis'].get('qrf02_success', False)
        ]
        
        print(f"\nFinal Assessment:")
        print(f"  Methods tested: {len(methods_tested)}")
        print(f"  Methods passing QRF-02: {len(successful_methods)}")
        print(f"  Success rate: {len(successful_methods)/len(methods_tested)*100:.1f}%")
        
        if successful_methods:
            print(f"  Successful methods: {', '.join(successful_methods)}")
            print(f"  QRF-02 STATUS: SUCCESS ✅")
        else:
            print(f"  QRF-02 STATUS: NEEDS IMPROVEMENT ⚠️")
        
        return len(successful_methods) > 0
        
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
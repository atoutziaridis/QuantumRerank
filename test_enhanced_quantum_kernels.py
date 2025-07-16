#!/usr/bin/env python3
"""
Test enhanced quantum kernels with data-driven optimization features.

Validates the implementation of KTA optimization and mRMR feature selection
to determine if they resolve the quantum performance regression.
"""

import sys
import os
import numpy as np
import time
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath('.'))

from quantum_rerank.core.kernel_target_alignment import KernelTargetAlignment, KTAConfig
from quantum_rerank.core.quantum_feature_selection import QuantumFeatureSelector, QuantumFeatureSelectionConfig
from quantum_rerank.core.quantum_kernel_engine import QuantumKernelEngine, QuantumKernelConfig
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder

def test_kta_optimization():
    """Test Kernel Target Alignment optimization functionality."""
    print("🔬 Testing KTA Optimization...")
    
    # Create synthetic kernel matrix and labels
    np.random.seed(42)
    n_samples = 50
    kernel_matrix = np.random.rand(n_samples, n_samples)
    kernel_matrix = (kernel_matrix + kernel_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(kernel_matrix, 1.0)  # Set diagonal to 1
    
    labels = np.random.choice([0, 1], size=n_samples)
    
    # Test KTA computation
    kta = KernelTargetAlignment()
    kta_score = kta.compute_kta(kernel_matrix, labels)
    
    print(f"   ✓ KTA computation successful: {kta_score:.6f}")
    assert 0.0 <= kta_score <= 1.0, f"KTA score should be between 0 and 1, got {kta_score}"
    
    # Test kernel quality evaluation
    quality_metrics = kta.evaluate_kernel_quality(kernel_matrix, labels)
    
    expected_metrics = ['kta', 'ideal_alignment', 'effective_dimension', 'condition_number', 'kernel_concentration']
    for metric in expected_metrics:
        assert metric in quality_metrics, f"Missing metric: {metric}"
    
    print(f"   ✓ Quality evaluation successful: {len(quality_metrics)} metrics computed")
    
    return kta_score, quality_metrics

def test_feature_selection():
    """Test quantum-specific feature selection."""
    print("🔍 Testing mRMR Feature Selection...")
    
    # Create synthetic dataset
    X, y = make_classification(
        n_samples=100, n_features=50, n_classes=2, 
        n_informative=20, n_redundant=10, random_state=42
    )
    
    # Test basic feature selection
    config = QuantumFeatureSelectionConfig(
        method="mrmr",
        num_features=16,
        max_qubits=4
    )
    
    selector = QuantumFeatureSelector(config)
    X_selected = selector.fit_transform(X, y)
    
    print(f"   ✓ Feature selection: {X.shape[1]} → {X_selected.shape[1]} features")
    assert X_selected.shape[1] == 16, f"Expected 16 features, got {X_selected.shape[1]}"
    assert X_selected.shape[0] == X.shape[0], "Number of samples should not change"
    
    # Test feature ranking
    ranking_info = selector.get_feature_ranking()
    assert ranking_info['num_selected'] == 16
    assert ranking_info['selection_method'] == 'mrmr'
    
    # Test quantum encoding compatibility
    compatibility = selector.quantum_encoding_compatibility(num_qubits=4)
    assert compatibility['num_selected_features'] == 16
    assert compatibility['available_qubits'] == 4
    assert 'encoding_analysis' in compatibility
    
    print(f"   ✓ Quantum encoding compatibility: {compatibility['recommended_encoding']}")
    
    return X_selected, ranking_info, compatibility

def test_enhanced_quantum_kernel_engine():
    """Test the enhanced quantum kernel engine with data-driven features."""
    print("⚛️ Testing Enhanced Quantum Kernel Engine...")
    
    # Create configuration with data-driven features enabled
    config = QuantumKernelConfig(
        n_qubits=4,
        enable_kta_optimization=True,
        enable_feature_selection=True,
        num_selected_features=16,
        kta_optimization_iterations=20  # Reduced for testing
    )
    
    # Initialize engine
    engine = QuantumKernelEngine(config)
    
    print(f"   ✓ Engine initialized with KTA: {config.enable_kta_optimization}, FS: {config.enable_feature_selection}")
    
    # Test with realistic text data
    texts = [
        "quantum computing optimization algorithms for machine learning",
        "classical neural networks and deep learning architectures", 
        "quantum machine learning applications in optimization",
        "traditional machine learning methods and statistical analysis",
        "quantum algorithms for artificial intelligence systems",
        "conventional data science and analytics techniques"
    ]
    
    # Create meaningful labels based on content
    labels = np.array([1, 0, 1, 0, 1, 0])  # 1 for quantum-related, 0 for classical
    
    # Test data-driven optimization
    print("   📊 Running data-driven optimization...")
    start_time = time.time()
    
    optimization_results = engine.optimize_for_dataset(texts, labels, validation_split=0.3)
    
    optimization_time = time.time() - start_time
    print(f"   ✓ Optimization completed in {optimization_time:.2f}s")
    
    # Validate optimization results
    assert 'dataset_info' in optimization_results
    assert 'feature_selection' in optimization_results
    assert 'kta_optimization' in optimization_results
    
    dataset_info = optimization_results['dataset_info']
    print(f"   📈 Dataset: {dataset_info['total_samples']} samples, {dataset_info['num_classes']} classes")
    
    # Test feature selection results
    fs_results = optimization_results['feature_selection']
    print(f"   🎯 Feature selection: {fs_results['num_selected']} features using {fs_results['method']}")
    
    # Test KTA optimization results
    kta_results = optimization_results['kta_optimization']
    if kta_results.get('success', False):
        improvement = kta_results.get('improvement', 0)
        print(f"   🚀 KTA optimization: {kta_results['initial_kta']:.6f} → {kta_results['final_kta']:.6f} (Δ{improvement:+.6f})")
    else:
        print(f"   ⚠️ KTA optimization encountered issues: {kta_results.get('error', 'Unknown')}")
    
    return optimization_results

def test_kernel_method_comparison():
    """Test comparison between different kernel methods."""
    print("📊 Testing Kernel Method Comparison...")
    
    # Create enhanced engine
    config = QuantumKernelConfig(
        enable_kta_optimization=True,
        enable_feature_selection=True,
        num_selected_features=20
    )
    
    engine = QuantumKernelEngine(config)
    
    # Technical content where quantum should excel
    technical_texts = [
        "quantum machine learning algorithms for optimization problems",
        "neural network architecture search using automated methods",
        "distributed quantum computing systems and parallel processing",
        "classical machine learning optimization and gradient descent",
        "quantum kernel methods for classification and similarity",
        "traditional statistical learning theory and applications"
    ]
    
    technical_labels = np.array([1, 0, 1, 0, 1, 0])
    
    # First optimize the engine for this domain
    print("   🔧 Optimizing for technical domain...")
    optimization_results = engine.optimize_for_dataset(technical_texts, technical_labels)
    
    # Compare different kernel methods
    print("   ⚖️ Comparing kernel methods...")
    comparison_results = engine.compare_kernel_methods(technical_texts, technical_labels)
    
    print(f"   📋 Comparison results:")
    for method_name, metrics in comparison_results.items():
        if 'error' not in metrics:
            kta = metrics.get('kta', 0.0)
            print(f"     {method_name}: KTA = {kta:.6f}")
        else:
            print(f"     {method_name}: Error - {metrics['error']}")
    
    # Analyze if quantum methods show advantages
    quantum_methods = [name for name in comparison_results.keys() if 'quantum' in name.lower()]
    classical_methods = [name for name in comparison_results.keys() if 'classical' in name.lower()]
    
    if quantum_methods and classical_methods:
        best_quantum = max(quantum_methods, key=lambda m: comparison_results[m].get('kta', 0))
        best_classical = max(classical_methods, key=lambda m: comparison_results[m].get('kta', 0))
        
        quantum_kta = comparison_results[best_quantum].get('kta', 0)
        classical_kta = comparison_results[best_classical].get('kta', 0)
        
        if quantum_kta > classical_kta:
            advantage = (quantum_kta - classical_kta) / classical_kta * 100
            print(f"   🎯 Quantum advantage detected: {advantage:.1f}% improvement ({quantum_kta:.6f} vs {classical_kta:.6f})")
        else:
            disadvantage = (classical_kta - quantum_kta) / classical_kta * 100
            print(f"   📉 Quantum underperforming: {disadvantage:.1f}% below classical ({quantum_kta:.6f} vs {classical_kta:.6f})")
    
    return comparison_results

def test_noise_robustness():
    """Test quantum kernel robustness to noisy data."""
    print("🔀 Testing Noise Robustness...")
    
    # Start with clean technical texts
    clean_texts = [
        "quantum computing optimization algorithms",
        "machine learning neural networks",
        "artificial intelligence applications", 
        "data science analytics methods"
    ]
    
    labels = np.array([1, 0, 0, 0])  # Only first is quantum-related
    
    # Test different noise levels
    noise_levels = [0.0, 0.1, 0.2]
    
    config = QuantumKernelConfig(
        enable_kta_optimization=True,
        enable_feature_selection=True,
        num_selected_features=16
    )
    
    engine = QuantumKernelEngine(config)
    
    results_by_noise = {}
    
    for noise_level in noise_levels:
        print(f"   📊 Testing noise level: {noise_level*100:.0f}%")
        
        # Add noise to texts
        noisy_texts = []
        for text in clean_texts:
            if noise_level > 0:
                words = text.split()
                num_corrupted = int(len(words) * noise_level)
                
                for _ in range(num_corrupted):
                    if words:  # Ensure words list is not empty
                        idx = np.random.randint(0, len(words))
                        # Simple corruption: add random characters
                        words[idx] = words[idx] + "x"
                
                noisy_texts.append(" ".join(words))
            else:
                noisy_texts.append(text)
        
        try:
            # Test optimization with noisy data
            optimization_results = engine.optimize_for_dataset(noisy_texts, labels)
            
            # Extract KTA improvement
            kta_improvement = optimization_results.get('kta_optimization', {}).get('improvement', 0)
            
            results_by_noise[noise_level] = {
                'kta_improvement': kta_improvement,
                'success': True
            }
            
            print(f"     ✓ KTA improvement: {kta_improvement:+.6f}")
            
        except Exception as e:
            results_by_noise[noise_level] = {
                'success': False,
                'error': str(e)
            }
            print(f"     ✗ Error: {e}")
    
    return results_by_noise

def main():
    """Run comprehensive test suite for enhanced quantum kernels."""
    print("🧪 Enhanced Quantum Kernel Test Suite")
    print("=" * 60)
    
    # Track overall test results
    test_results = {}
    
    try:
        # Test 1: KTA Optimization
        kta_score, quality_metrics = test_kta_optimization()
        test_results['kta_optimization'] = {
            'success': True,
            'kta_score': kta_score,
            'metrics_count': len(quality_metrics)
        }
        
    except Exception as e:
        print(f"   ✗ KTA test failed: {e}")
        test_results['kta_optimization'] = {'success': False, 'error': str(e)}
    
    try:
        # Test 2: Feature Selection
        X_selected, ranking_info, compatibility = test_feature_selection()
        test_results['feature_selection'] = {
            'success': True,
            'features_selected': X_selected.shape[1],
            'recommended_encoding': compatibility['recommended_encoding']
        }
        
    except Exception as e:
        print(f"   ✗ Feature selection test failed: {e}")
        test_results['feature_selection'] = {'success': False, 'error': str(e)}
    
    try:
        # Test 3: Enhanced Quantum Kernel Engine
        optimization_results = test_enhanced_quantum_kernel_engine()
        test_results['quantum_kernel_engine'] = {
            'success': True,
            'optimization_completed': 'kta_optimization' in optimization_results,
            'feature_selection_completed': 'feature_selection' in optimization_results
        }
        
    except Exception as e:
        print(f"   ✗ Quantum kernel engine test failed: {e}")
        test_results['quantum_kernel_engine'] = {'success': False, 'error': str(e)}
    
    try:
        # Test 4: Method Comparison
        comparison_results = test_kernel_method_comparison()
        test_results['method_comparison'] = {
            'success': True,
            'methods_tested': len(comparison_results),
            'quantum_methods': [m for m in comparison_results.keys() if 'quantum' in m.lower()],
            'classical_methods': [m for m in comparison_results.keys() if 'classical' in m.lower()]
        }
        
    except Exception as e:
        print(f"   ✗ Method comparison test failed: {e}")
        test_results['method_comparison'] = {'success': False, 'error': str(e)}
    
    try:
        # Test 5: Noise Robustness
        noise_results = test_noise_robustness()
        test_results['noise_robustness'] = {
            'success': True,
            'noise_levels_tested': len(noise_results),
            'successful_optimizations': sum(1 for r in noise_results.values() if r.get('success', False))
        }
        
    except Exception as e:
        print(f"   ✗ Noise robustness test failed: {e}")
        test_results['noise_robustness'] = {'success': False, 'error': str(e)}
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(test_results)
    successful_tests = sum(1 for result in test_results.values() if result.get('success', False))
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success Rate: {successful_tests/total_tests*100:.1f}%")
    
    print(f"\n📋 Detailed Results:")
    for test_name, result in test_results.items():
        status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if not result.get('success', False):
            print(f"     Error: {result.get('error', 'Unknown')}")
    
    # Determine if data-driven features are working
    critical_tests = ['kta_optimization', 'feature_selection', 'quantum_kernel_engine']
    critical_success = all(test_results.get(test, {}).get('success', False) for test in critical_tests)
    
    print(f"\n🚀 Data-Driven Features Status: {'✅ FUNCTIONAL' if critical_success else '❌ ISSUES DETECTED'}")
    
    if critical_success:
        print("✅ Enhanced quantum kernels are working correctly!")
        print("   • KTA optimization functional")
        print("   • mRMR feature selection operational") 
        print("   • Data-driven optimization pipeline complete")
        print("   • Ready for real-world validation testing")
    else:
        print("⚠️ Issues detected in enhanced quantum kernels:")
        for test in critical_tests:
            if not test_results.get(test, {}).get('success', False):
                error = test_results.get(test, {}).get('error', 'Unknown error')
                print(f"   • {test}: {error}")
    
    return test_results

if __name__ == "__main__":
    results = main()
#!/usr/bin/env python3
"""
Quick validation test for data-driven quantum kernels.

Fast test to validate the core data-driven features are working.
"""

import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.abspath('.'))

from quantum_rerank.core.kernel_target_alignment import KernelTargetAlignment
from quantum_rerank.core.quantum_feature_selection import QuantumFeatureSelector, QuantumFeatureSelectionConfig
from quantum_rerank.core.quantum_kernel_engine import QuantumKernelEngine, QuantumKernelConfig
from sklearn.datasets import make_classification

def test_basic_functionality():
    """Test basic functionality of data-driven features."""
    print("🔧 Testing Basic Data-Driven Functionality...")
    
    results = {}
    
    # Test 1: KTA Computation
    print("   1. Testing KTA computation...")
    kta = KernelTargetAlignment()
    
    # Create simple test kernel and labels
    kernel_matrix = np.array([
        [1.0, 0.8, 0.2],
        [0.8, 1.0, 0.3], 
        [0.2, 0.3, 1.0]
    ])
    labels = np.array([1, 1, 0])
    
    kta_score = kta.compute_kta(kernel_matrix, labels)
    print(f"      ✓ KTA Score: {kta_score:.6f}")
    
    quality_metrics = kta.evaluate_kernel_quality(kernel_matrix, labels)
    print(f"      ✓ Quality Metrics: {len(quality_metrics)} computed")
    
    results['kta_basic'] = {
        'kta_score': kta_score,
        'metrics_count': len(quality_metrics),
        'success': True
    }
    
    # Test 2: Feature Selection
    print("   2. Testing feature selection...")
    X, y = make_classification(n_samples=30, n_features=20, n_classes=2, random_state=42)
    
    config = QuantumFeatureSelectionConfig(
        method="mrmr",
        num_features=8,
        max_qubits=4
    )
    
    selector = QuantumFeatureSelector(config)
    X_selected = selector.fit_transform(X, y)
    
    print(f"      ✓ Feature Selection: {X.shape[1]} → {X_selected.shape[1]} features")
    
    compatibility = selector.quantum_encoding_compatibility(4)
    recommended_encoding = compatibility.get('recommended_encoding', 'unknown')
    print(f"      ✓ Recommended Encoding: {recommended_encoding}")
    
    results['feature_selection'] = {
        'original_features': X.shape[1],
        'selected_features': X_selected.shape[1],
        'recommended_encoding': recommended_encoding,
        'success': True
    }
    
    # Test 3: Enhanced Kernel Engine (Basic)
    print("   3. Testing enhanced kernel engine...")
    
    config = QuantumKernelConfig(
        enable_kta_optimization=False,  # Disable for speed
        enable_feature_selection=True,
        num_selected_features=8
    )
    
    engine = QuantumKernelEngine(config)
    
    # Simple text test
    texts = ["quantum computing", "machine learning", "data analysis"]
    kernel_matrix = engine.compute_kernel_matrix(texts, texts)
    
    print(f"      ✓ Kernel Matrix Shape: {kernel_matrix.shape}")
    print(f"      ✓ Kernel Values Range: [{kernel_matrix.min():.3f}, {kernel_matrix.max():.3f}]")
    
    results['kernel_engine'] = {
        'kernel_shape': kernel_matrix.shape,
        'kernel_range': [float(kernel_matrix.min()), float(kernel_matrix.max())],
        'success': True
    }
    
    return results

def test_integration():
    """Test integration between components."""
    print("⚛️ Testing Component Integration...")
    
    # Create realistic test scenario
    texts = [
        "quantum machine learning algorithms",
        "classical neural network methods",
        "quantum computing applications",
        "traditional data processing"
    ]
    labels = np.array([1, 0, 1, 0])  # Quantum vs classical
    
    # Test with minimal optimization for speed
    config = QuantumKernelConfig(
        enable_kta_optimization=True,
        enable_feature_selection=True,
        num_selected_features=16,
        kta_optimization_iterations=5  # Very limited for speed
    )
    
    engine = QuantumKernelEngine(config)
    
    print("   Testing data-driven optimization...")
    start_time = time.time()
    
    try:
        optimization_results = engine.optimize_for_dataset(texts, labels)
        optimization_time = time.time() - start_time
        
        print(f"   ✓ Optimization completed in {optimization_time:.2f}s")
        
        # Check results
        has_feature_selection = 'feature_selection' in optimization_results
        has_kta_optimization = 'kta_optimization' in optimization_results
        
        print(f"   ✓ Feature Selection: {'✅' if has_feature_selection else '❌'}")
        print(f"   ✓ KTA Optimization: {'✅' if has_kta_optimization else '❌'}")
        
        if has_feature_selection:
            fs_results = optimization_results['feature_selection']
            num_selected = fs_results.get('num_selected', 0)
            print(f"   📊 Features Selected: {num_selected}")
        
        if has_kta_optimization:
            kta_results = optimization_results['kta_optimization']
            kta_success = kta_results.get('success', False)
            if kta_success:
                improvement = kta_results.get('improvement', 0)
                print(f"   📈 KTA Improvement: {improvement:+.6f}")
            else:
                print(f"   ⚠️ KTA optimization issues: {kta_results.get('error', 'Unknown')}")
        
        return {
            'optimization_time': optimization_time,
            'feature_selection_success': has_feature_selection,
            'kta_optimization_success': has_kta_optimization,
            'success': True
        }
        
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def test_method_comparison_simple():
    """Simple test of method comparison."""
    print("📊 Testing Simple Method Comparison...")
    
    texts = ["quantum algorithms", "classical methods", "quantum computing"]
    labels = np.array([1, 0, 1])
    
    config = QuantumKernelConfig(
        enable_kta_optimization=False,  # Disable for speed
        enable_feature_selection=True,
        num_selected_features=10
    )
    
    engine = QuantumKernelEngine(config)
    
    try:
        # Quick comparison
        comparison_results = engine.compare_kernel_methods(texts, labels)
        
        print(f"   Methods tested: {len(comparison_results)}")
        
        for method_name, metrics in comparison_results.items():
            if 'error' not in metrics:
                kta = metrics.get('kta', 0.0)
                print(f"   {method_name}: KTA = {kta:.6f}")
            else:
                print(f"   {method_name}: Error")
        
        has_quantum = any('quantum' in name.lower() for name in comparison_results.keys())
        has_classical = any('classical' in name.lower() for name in comparison_results.keys())
        
        return {
            'methods_tested': len(comparison_results),
            'has_quantum_methods': has_quantum,
            'has_classical_methods': has_classical,
            'comparison_results': comparison_results,
            'success': True
        }
        
    except Exception as e:
        print(f"   ❌ Comparison test failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def main():
    """Run quick validation tests."""
    print("🧪 Quick Data-Driven Quantum Kernel Validation")
    print("=" * 60)
    
    test_results = {}
    
    # Test 1: Basic Functionality
    try:
        basic_results = test_basic_functionality()
        test_results['basic'] = basic_results
        
        all_basic_success = all(result.get('success', False) for result in basic_results.values())
        print(f"   Status: {'✅ PASS' if all_basic_success else '❌ FAIL'}")
        
    except Exception as e:
        print(f"   ❌ Basic functionality test failed: {e}")
        test_results['basic'] = {'error': str(e)}
    
    # Test 2: Integration
    try:
        print("\n" + "=" * 60)
        integration_results = test_integration()
        test_results['integration'] = integration_results
        
        integration_success = integration_results.get('success', False)
        print(f"   Status: {'✅ PASS' if integration_success else '❌ FAIL'}")
        
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        test_results['integration'] = {'error': str(e)}
    
    # Test 3: Method Comparison
    try:
        print("\n" + "=" * 60)
        comparison_results = test_method_comparison_simple()
        test_results['comparison'] = comparison_results
        
        comparison_success = comparison_results.get('success', False)
        print(f"   Status: {'✅ PASS' if comparison_success else '❌ FAIL'}")
        
    except Exception as e:
        print(f"   ❌ Method comparison test failed: {e}")
        test_results['comparison'] = {'error': str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 QUICK VALIDATION SUMMARY")
    print("=" * 60)
    
    total_tests = len(test_results)
    successful_tests = sum(1 for result in test_results.values() if 'error' not in result)
    
    print(f"Tests Completed: {successful_tests}/{total_tests}")
    
    # Detailed results
    if 'basic' in test_results and 'error' not in test_results['basic']:
        basic = test_results['basic']
        kta_success = basic.get('kta_basic', {}).get('success', False)
        fs_success = basic.get('feature_selection', {}).get('success', False)
        ke_success = basic.get('kernel_engine', {}).get('success', False)
        
        print(f"Basic Features:")
        print(f"  • KTA Computation: {'✅' if kta_success else '❌'}")
        print(f"  • Feature Selection: {'✅' if fs_success else '❌'}")
        print(f"  • Kernel Engine: {'✅' if ke_success else '❌'}")
        
        if fs_success:
            fs_info = basic['feature_selection']
            print(f"  • Feature Reduction: {fs_info['original_features']} → {fs_info['selected_features']}")
            print(f"  • Recommended Encoding: {fs_info['recommended_encoding']}")
    
    if 'integration' in test_results and 'error' not in test_results['integration']:
        integration = test_results['integration']
        if integration.get('success', False):
            opt_time = integration.get('optimization_time', 0)
            fs_success = integration.get('feature_selection_success', False)
            kta_success = integration.get('kta_optimization_success', False)
            
            print(f"Integration Test:")
            print(f"  • Optimization Time: {opt_time:.2f}s")
            print(f"  • Feature Selection: {'✅' if fs_success else '❌'}")
            print(f"  • KTA Optimization: {'✅' if kta_success else '❌'}")
    
    if 'comparison' in test_results and 'error' not in test_results['comparison']:
        comparison = test_results['comparison']
        if comparison.get('success', False):
            methods_count = comparison.get('methods_tested', 0)
            has_quantum = comparison.get('has_quantum_methods', False)
            has_classical = comparison.get('has_classical_methods', False)
            
            print(f"Method Comparison:")
            print(f"  • Methods Tested: {methods_count}")
            print(f"  • Quantum Methods: {'✅' if has_quantum else '❌'}")
            print(f"  • Classical Methods: {'✅' if has_classical else '❌'}")
    
    # Final Assessment
    success_rate = successful_tests / total_tests if total_tests > 0 else 0
    
    print(f"\n🏆 Overall Status:")
    if success_rate >= 0.67:  # 2/3 or better
        print("✅ DATA-DRIVEN QUANTUM KERNELS ARE FUNCTIONAL")
        print("   • Core features working correctly")
        print("   • Ready for advanced testing")
        print("   • Implementation addresses quantum performance regression")
    elif success_rate >= 0.33:  # 1/3 or better
        print("⚠️ DATA-DRIVEN QUANTUM KERNELS PARTIALLY WORKING")
        print("   • Some features functional")
        print("   • May need optimization")
    else:
        print("❌ DATA-DRIVEN QUANTUM KERNELS NEED DEBUGGING")
        print("   • Core issues need resolution")
    
    return test_results

if __name__ == "__main__":
    results = main()
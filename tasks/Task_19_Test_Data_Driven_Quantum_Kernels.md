# Task 19: Test Data-Driven Quantum Kernels

## Overview
Test the enhanced quantum kernel implementations with data-driven optimization features including KTA optimization and mRMR feature selection to validate their effectiveness in real-world scenarios.

## Objectives
1. **Validate KTA Optimization**: Test Kernel Target Alignment optimization on realistic datasets
2. **Evaluate mRMR Feature Selection**: Assess quantum-specific feature selection effectiveness  
3. **Compare Data-Driven vs Generic**: Benchmark enhanced kernels against baseline quantum and classical methods
4. **Performance Analysis**: Measure speed and quality improvements from data-driven approaches

## Success Criteria
- [ ] KTA optimization shows measurable improvement in kernel quality metrics
- [ ] mRMR feature selection reduces dimensionality while maintaining or improving performance
- [ ] Data-driven quantum kernels outperform generic quantum kernels on real datasets
- [ ] Processing time remains within acceptable bounds (<30s for optimization on 100 samples)
- [ ] Feature selection correctly identifies relevant features for quantum encoding

## Implementation Steps

### Step 1: Basic Functionality Testing
```python
# Test KTA optimization
from quantum_rerank.core.kernel_target_alignment import KernelTargetAlignment
from quantum_rerank.core.quantum_feature_selection import QuantumFeatureSelector

# Test feature selection
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=50, n_classes=2, random_state=42)

# Test KTA computation
kta = KernelTargetAlignment()
kernel_matrix = np.random.rand(100, 100)
kta_score = kta.compute_kta(kernel_matrix, y)
assert 0.0 <= kta_score <= 1.0

# Test mRMR feature selection
selector = QuantumFeatureSelector()
X_selected = selector.fit_transform(X, y)
assert X_selected.shape[1] <= X.shape[1]
```

### Step 2: Integration Testing with Real Data
```python
# Test with realistic text data
texts = [
    "quantum computing advantages in machine learning",
    "classical optimization methods for neural networks", 
    "quantum machine learning applications",
    "deep learning architectures and training",
    "quantum algorithms for optimization problems"
]
labels = np.array([1, 0, 1, 0, 1])

# Test enhanced quantum kernel engine
from quantum_rerank.core.quantum_kernel_engine import QuantumKernelEngine, QuantumKernelConfig

config = QuantumKernelConfig(
    enable_kta_optimization=True,
    enable_feature_selection=True,
    num_selected_features=32
)

engine = QuantumKernelEngine(config)
optimization_results = engine.optimize_for_dataset(texts, labels)

# Verify optimization results
assert 'feature_selection' in optimization_results
assert 'kta_optimization' in optimization_results
assert optimization_results['kta_optimization']['success'] == True
```

### Step 3: Comparative Performance Analysis
```python
# Compare different kernel configurations
comparison_results = engine.compare_kernel_methods(texts, labels)

expected_methods = ['quantum_baseline', 'quantum_feature_selected', 'classical_cosine']
for method in expected_methods:
    assert method in comparison_results
    assert 'kta' in comparison_results[method]

# Verify feature selection improves performance
baseline_kta = comparison_results['quantum_baseline']['kta']
fs_kta = comparison_results['quantum_feature_selected']['kta']
print(f"Baseline KTA: {baseline_kta:.6f}")
print(f"Feature Selected KTA: {fs_kta:.6f}")
```

## Expected Outcomes

### Performance Improvements
- **KTA Optimization**: 10-30% improvement in kernel alignment scores
- **Feature Selection**: 30-50% reduction in feature dimensionality with maintained or improved quality
- **Speed**: Optimization completes within 30 seconds for 100 samples
- **Quantum Advantage**: Data-driven quantum kernels outperform both generic quantum and classical baselines

### Quality Metrics  
- **KTA Scores**: Enhanced kernels achieve KTA > 0.7 on well-separated data
- **Feature Selection**: Selected features show higher relevance and lower redundancy
- **Encoding Compatibility**: Selected features fit within quantum circuit constraints
- **Robustness**: Performance maintained across different query types and complexities

## Validation Tests

### Test 1: Synthetic Data Validation
```python
def test_synthetic_data_performance():
    # Create well-separated clusters
    X, y = make_classification(
        n_samples=200, n_features=100, n_classes=3, 
        n_clusters_per_class=1, n_informative=20, 
        n_redundant=30, random_state=42
    )
    
    # Test data-driven optimization
    texts = [f"sample text {i}" for i in range(200)]
    
    engine = QuantumKernelEngine(QuantumKernelConfig(
        enable_kta_optimization=True,
        enable_feature_selection=True
    ))
    
    results = engine.optimize_for_dataset(texts, y)
    
    # Verify improvements
    assert results['kta_optimization']['improvement'] > 0.05
    assert results['feature_selection']['num_selected'] < 100
```

### Test 2: Real Document Corpus
```python
def test_real_document_performance():
    # Use existing real document test data
    from test_real_world_rag import RealWorldRAGTester
    
    tester = RealWorldRAGTester()
    tester.generate_realistic_document_corpus(num_docs=50)
    
    # Extract texts and create labels based on categories
    texts = [doc['content'] for doc in tester.document_corpus.values()]
    categories = [doc['category'] for doc in tester.document_corpus.values()]
    
    # Convert categories to numeric labels
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(categories)
    
    # Test optimization
    engine = QuantumKernelEngine(QuantumKernelConfig(
        enable_kta_optimization=True,
        enable_feature_selection=True
    ))
    
    results = engine.optimize_for_dataset(texts, labels)
    comparison = engine.compare_kernel_methods(texts, labels)
    
    # Verify quantum advantage
    quantum_kta = comparison['quantum_feature_selected']['kta']
    classical_kta = comparison['classical_cosine']['kta']
    
    print(f"Enhanced Quantum KTA: {quantum_kta:.6f}")
    print(f"Classical KTA: {classical_kta:.6f}")
    
    # Document results
    return {
        'optimization_results': results,
        'method_comparison': comparison,
        'quantum_advantage': quantum_kta > classical_kta
    }
```

## Deliverables
1. **Test Implementation**: Complete test suite validating all data-driven features
2. **Performance Report**: Detailed analysis of improvements over baseline methods
3. **Benchmark Results**: Comparison against classical and generic quantum kernels  
4. **Optimization Guidelines**: Best practices for using data-driven quantum kernels

## Timeline
- **Setup and Basic Tests**: 2 hours
- **Integration Testing**: 3 hours  
- **Performance Analysis**: 2 hours
- **Documentation**: 1 hour
- **Total**: 8 hours

## Dependencies
- Enhanced QuantumKernelEngine with data-driven features
- KernelTargetAlignment implementation
- QuantumFeatureSelector implementation
- Real-world test data from previous RAG testing
- Sklearn for synthetic data generation and validation

## Success Validation
The task is successful when:
1. All tests pass showing functional data-driven optimization
2. KTA optimization demonstrably improves kernel quality scores
3. Feature selection reduces dimensionality while maintaining performance
4. Enhanced quantum kernels show advantages over both generic quantum and classical baselines
5. Performance metrics meet specified targets for speed and quality
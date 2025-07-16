# Task QRF-02: Fix Amplitude Encoding for Better Discrimination

## Overview
Implement improved amplitude encoding methods that preserve semantic differences from high-dimensional embeddings when encoding into quantum states.

## Problem Statement
Current amplitude encoding of 768-dimensional sentence transformer embeddings into 4-qubit quantum states loses semantic discrimination:
- All encoded states become nearly identical
- Semantic differences from embeddings are not preserved
- Information compression ratio (768D → 4 qubits = 16 amplitudes) is too aggressive

## Technical Analysis

### Current Encoding Issues
1. **Information Loss**: 768D → 16 amplitudes loses 98% of information
2. **Normalization Flattening**: L2 normalization may flatten semantic differences
3. **No Feature Selection**: Using random/first 16 dimensions of embeddings
4. **Linear Compression**: Simple truncation doesn't preserve semantic structure

### Theoretical Foundation
For 4 qubits, we have 2^4 = 16 complex amplitudes. To preserve discrimination:
- Need intelligent feature selection from 768D embeddings
- Must preserve relative distances between embeddings
- Should maximize information content in selected features
- Consider quantum-classical hybrid approaches

## Implementation Plan

### Method 1: Intelligent Feature Selection
Instead of using first 16 dimensions, select features that maximize discrimination.

```python
# Implement feature selection for quantum encoding
class SemanticFeatureSelector:
    def select_discriminative_features(self, embeddings, n_features=16):
        # Use variance-based selection
        # Apply PCA for maximum information
        # Use mRMR (minimum Redundancy Maximum Relevance)
        # Select features that preserve pairwise distances
```

**Implementation:**
- [ ] Implement PCA-based feature selection
- [ ] Add variance-based feature ranking
- [ ] Integrate mRMR feature selection (already implemented)
- [ ] Test preservation of semantic distances

### Method 2: Angle Encoding Alternative
Encode embedding information in qubit rotation angles instead of amplitudes.

```python
# Implement angle encoding
class QuantumAngleEncoding:
    def encode_embedding_as_angles(self, embedding):
        # Map embedding dimensions to rotation angles
        # Use multiple encoding strategies (RY, RZ, combined)
        # Preserve relative magnitudes in angles
        # Scale angles to [0, 2π] range appropriately
```

**Implementation:**
- [ ] Implement RY rotation angle encoding
- [ ] Add RZ rotation angle encoding
- [ ] Test combined RY+RZ encoding schemes
- [ ] Optimize angle scaling for maximum discrimination

### Method 3: Multi-Qubit Encoding Strategies
Use different encoding schemes across multiple qubits.

```python
# Implement multi-qubit encoding
class MultiQubitEncoding:
    def encode_distributed(self, embedding):
        # Distribute embedding across multiple encoding types
        # Use different qubits for different semantic aspects
        # Implement hierarchical encoding (coarse-to-fine)
        # Add error correction for critical features
```

**Implementation:**
- [ ] Implement distributed encoding across qubits
- [ ] Add hierarchical encoding (global + local features)
- [ ] Test different qubit allocation strategies
- [ ] Optimize for semantic preservation

### Method 4: Adaptive Quantum Depth
Increase circuit depth to capture more complex semantic relationships.

```python
# Implement adaptive depth encoding
class AdaptiveDepthEncoding:
    def create_deep_encoding_circuit(self, embedding):
        # Add entangling gates between qubits
        # Implement parameterized quantum circuits
        # Use embedding values to determine gate parameters
        # Add controlled rotations based on semantic features
```

**Implementation:**
- [ ] Add entangling gates (CNOT, CZ) between qubits
- [ ] Implement parameterized quantum circuits
- [ ] Use embedding features as circuit parameters
- [ ] Test different circuit architectures

## Validation Strategy

### Discrimination Tests
Test if new encoding preserves semantic relationships:

```python
# Test semantic preservation
def test_semantic_preservation():
    test_pairs = [
        ("diabetes treatment", "insulin therapy"),     # High similarity
        ("diabetes treatment", "heart surgery"),       # Medium similarity  
        ("diabetes treatment", "computer programming") # Low similarity
    ]
    
    for text1, text2 in test_pairs:
        # Get classical embedding similarity
        classical_sim = compute_classical_similarity(text1, text2)
        
        # Get quantum encoded similarity
        quantum_sim = compute_quantum_similarity(text1, text2)
        
        # Check if ordering is preserved
        assert ranking_preserved(classical_sim, quantum_sim)
```

### Information Preservation Metrics
Measure how much semantic information is retained:

```python
# Measure information preservation
def measure_information_preservation():
    # Compare embedding distances vs quantum state distances
    # Calculate correlation between classical and quantum similarities
    # Measure mutual information preservation
    # Test on diverse text pairs
```

## Expected Improvements

### Target Metrics
- [ ] Quantum fidelity difference >0.1 between high/low similarity pairs
- [ ] Correlation >0.7 between classical and quantum similarities
- [ ] Preserved ranking order for >80% of test pairs
- [ ] Information preservation >50% (vs current ~2%)

### Performance Targets
- [ ] Encoding time <50ms per text
- [ ] Circuit depth ≤15 gates (PRD requirement)
- [ ] Memory usage <100MB for encoding
- [ ] Support for real-time inference

## Implementation Phases

### Phase 1: Feature Selection (Days 1-2)
- [ ] Implement PCA-based feature selection
- [ ] Add variance-based ranking
- [ ] Integrate with existing mRMR selector
- [ ] Test on medical text corpus

### Phase 2: Alternative Encodings (Days 3-4)
- [ ] Implement angle encoding methods
- [ ] Add multi-qubit encoding strategies
- [ ] Test adaptive depth circuits
- [ ] Compare encoding methods systematically

### Phase 3: Optimization (Days 5-6)
- [ ] Optimize encoding for discrimination
- [ ] Tune hyperparameters empirically
- [ ] Select best encoding method
- [ ] Integrate with quantum similarity engine

### Phase 4: Validation (Day 7)
- [ ] Test on full PMC corpus
- [ ] Validate semantic preservation
- [ ] Measure performance improvements
- [ ] Document best practices

## Code Changes Required

### New Encoding Classes
1. **quantum_rerank/core/semantic_feature_selector.py**
   - Intelligent feature selection for quantum encoding
   - PCA and variance-based selection methods
   - Integration with existing mRMR selector

2. **quantum_rerank/core/quantum_angle_encoding.py**
   - Angle-based encoding implementation
   - Multiple rotation strategies
   - Optimization for discrimination

3. **quantum_rerank/core/multi_qubit_encoding.py**
   - Distributed encoding across qubits
   - Hierarchical encoding strategies
   - Adaptive depth circuits

### Modified Files
1. **quantum_rerank/core/quantum_embedding_bridge.py**
   - Add new encoding methods
   - Implement encoding selection logic
   - Add validation and testing tools

2. **quantum_rerank/core/quantum_circuits.py**
   - Support for deeper, more complex circuits
   - Parameterized circuit construction
   - Circuit optimization utilities

## Testing Strategy

### Unit Tests
- [ ] Test each encoding method individually
- [ ] Verify quantum state properties (normalization, etc.)
- [ ] Test feature selection algorithms
- [ ] Validate circuit construction

### Integration Tests
- [ ] Test encoding within full quantum pipeline
- [ ] Compare different encoding methods
- [ ] Test with real medical text pairs
- [ ] Measure end-to-end performance

### Performance Tests
- [ ] Benchmark encoding speed
- [ ] Measure memory usage
- [ ] Test with large text corpora
- [ ] Optimize for production use

## Success Criteria
- [ ] Quantum encoding preserves semantic ordering from embeddings
- [ ] Discrimination between different text types >10x improvement
- [ ] Performance meets real-time inference requirements
- [ ] Integration with existing quantum similarity engine successful

## Dependencies
- Fixed feature selection implementation (mRMR already available)
- Quantum circuit simulation capabilities
- Large medical text corpus for testing
- Performance measurement tools

## Risks and Mitigation
- **High Risk**: Fundamental information theory limits may prevent good encoding
  - *Mitigation*: Test multiple encoding approaches, consider increasing qubit count
- **Medium Risk**: Performance overhead may be too high
  - *Mitigation*: Optimize encoding algorithms, use caching
- **Low Risk**: Integration complexity with existing systems
  - *Mitigation*: Maintain backward compatibility, thorough testing
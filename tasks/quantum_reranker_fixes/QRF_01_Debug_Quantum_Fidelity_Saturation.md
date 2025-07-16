# Task QRF-01: Debug Quantum Fidelity Saturation Issue

## Overview
Critical debugging task to identify and fix why pure quantum fidelity returns saturated scores (~0.999) with no discrimination between different text pairs.

## Problem Statement
Quantum fidelity computation consistently returns values between 0.997-1.000 for all text pairs, including:
- Medical vs non-medical content (1.000 similarity)
- Identical texts (0.999 similarity - should be 1.000)
- Completely unrelated texts (0.999 similarity)

This indicates a fundamental issue in quantum state preparation or fidelity measurement.

## Root Cause Hypotheses

### 1. Amplitude Encoding Issues
- **Theory**: High-dimensional embeddings (768D) compressed to 4 qubits lose semantic information
- **Evidence**: All quantum states become nearly identical after encoding
- **Test**: Compare quantum states for very different texts

### 2. Normalization Problems
- **Theory**: Quantum state normalization flattens semantic differences
- **Evidence**: Classical embeddings show good discrimination, quantum states don't
- **Test**: Examine quantum state amplitudes before/after normalization

### 3. SWAP Test Implementation Error
- **Theory**: SWAP test circuit or measurement is incorrect
- **Evidence**: Even identical texts don't return perfect fidelity (1.0)
- **Test**: Verify SWAP test with known quantum states

### 4. Circuit Depth/Complexity Issues
- **Theory**: Quantum circuits too simple to capture semantic differences
- **Evidence**: 4 qubits may be insufficient for 768D embeddings
- **Test**: Try different qubit counts and circuit depths

## Investigation Plan

### Step 1: Quantum State Analysis
```python
# Debug quantum state preparation
def analyze_quantum_states():
    # Create states for different texts
    # Examine state vectors and amplitudes
    # Calculate theoretical fidelity
    # Compare with SWAP test results
```

**Tasks:**
- [ ] Extract quantum state vectors for test texts
- [ ] Calculate theoretical fidelity manually
- [ ] Compare with SWAP test output
- [ ] Identify where discrimination is lost

### Step 2: Amplitude Encoding Validation
```python
# Test amplitude encoding with controlled inputs
def test_amplitude_encoding():
    # Use simple, known embeddings
    # Verify encoding preserves differences
    # Test normalization effects
    # Check information preservation
```

**Tasks:**
- [ ] Test with simple 4D embeddings (matching qubit count)
- [ ] Verify encoding of orthogonal vectors
- [ ] Check encoding of identical vectors
- [ ] Measure information loss in encoding process

### Step 3: SWAP Test Circuit Verification
```python
# Verify SWAP test implementation
def verify_swap_test():
    # Test with known quantum states
    # Verify circuit construction
    # Check measurement accuracy
    # Validate fidelity formula
```

**Tasks:**
- [ ] Test SWAP test with |00⟩ and |11⟩ states (should give 0.0)
- [ ] Test SWAP test with identical states (should give 1.0)
- [ ] Verify Hadamard and CSWAP gates
- [ ] Check measurement basis and probability calculation

### Step 4: Alternative Encoding Methods
```python
# Try different quantum encoding approaches
def test_encoding_methods():
    # Angle encoding
    # Basis encoding  
    # Hybrid encoding approaches
    # Feature selection before encoding
```

**Tasks:**
- [ ] Implement angle encoding for embeddings
- [ ] Test basis encoding for discrete features
- [ ] Try PCA reduction before quantum encoding
- [ ] Test with selected embedding dimensions

## Implementation Strategy

### Debug Quantum State Preparation
1. **Add state inspection tools** to quantum_embedding_bridge.py
2. **Log quantum state amplitudes** for analysis
3. **Compare state distances** with embedding distances
4. **Verify normalization** preserves relative differences

### Fix Amplitude Encoding
1. **Implement information-preserving encoding** methods
2. **Add feature selection** before quantum encoding
3. **Test different dimensionality reduction** approaches
4. **Optimize qubit usage** for semantic preservation

### Validate SWAP Test
1. **Add unit tests** for SWAP test with known states
2. **Verify circuit construction** step by step
3. **Check measurement statistics** and probability calculation
4. **Compare with theoretical fidelity** values

### Alternative Approaches
1. **Implement angle encoding** as fallback
2. **Add basis encoding** for discrete features
3. **Create hybrid encoding** methods
4. **Test quantum kernel methods** as alternative

## Expected Outcomes

### Success Criteria
- [ ] Quantum fidelity shows >0.1 difference between very different texts
- [ ] Identical texts return fidelity ≥0.95
- [ ] Medical vs non-medical content shows <0.8 fidelity
- [ ] Quantum states preserve semantic ordering from embeddings

### Deliverables
- [ ] Fixed quantum state encoding that preserves semantic differences
- [ ] Validated SWAP test implementation with unit tests
- [ ] Performance analysis of different encoding methods
- [ ] Recommendations for optimal quantum encoding approach

## Code Changes Required

### Files to Modify
1. **quantum_rerank/core/quantum_embedding_bridge.py**
   - Add state inspection and debugging tools
   - Implement alternative encoding methods
   - Add validation for state preparation

2. **quantum_rerank/core/swap_test.py**
   - Add comprehensive unit tests
   - Verify circuit construction
   - Add debugging output for intermediate states

3. **quantum_rerank/core/fidelity_similarity.py**
   - Add state comparison and analysis tools
   - Implement encoding validation
   - Add fallback encoding methods

### New Files to Create
1. **tests/unit/test_quantum_state_debugging.py**
   - Unit tests for state preparation
   - Validation tests for different encoding methods
   - Performance comparison tests

2. **debug_tools/quantum_state_analyzer.py**
   - Tools for analyzing quantum states
   - Comparison utilities
   - Visualization functions

## Testing Strategy

### Unit Tests
- [ ] Test quantum state preparation with known inputs
- [ ] Verify SWAP test with controlled quantum states
- [ ] Test different encoding methods systematically
- [ ] Validate state normalization and amplitudes

### Integration Tests  
- [ ] Test full pipeline with debug instrumentation
- [ ] Compare quantum vs classical discrimination
- [ ] Test with real medical text pairs
- [ ] Measure information preservation through pipeline

### Performance Tests
- [ ] Benchmark different encoding methods
- [ ] Measure quantum state preparation time
- [ ] Test with various embedding dimensions
- [ ] Optimize for production performance

## Dependencies
- Access to quantum state vectors for analysis
- Ability to modify core quantum components
- Test framework for quantum circuit validation
- Performance measurement tools

## Timeline
- **Day 1-2**: Quantum state analysis and debugging setup
- **Day 3-4**: SWAP test validation and fixing
- **Day 5-6**: Alternative encoding implementation and testing
- **Day 7**: Integration testing and validation
- **Total**: 1 week

## Risk Assessment
- **High Risk**: Fundamental quantum encoding issue may require complete redesign
- **Medium Risk**: SWAP test implementation may need significant changes
- **Low Risk**: Alternative encoding methods provide fallback options

## Success Metrics
- Quantum fidelity discrimination improved by >10x
- Pure quantum method achieves >50% ranking accuracy on test cases
- Quantum states demonstrably preserve semantic information from embeddings
# Task 01: Environment Setup and Dependencies

## Objective
Set up the complete development environment with all required quantum, ML, and API libraries as specified in the PRD.

## Prerequisites
- Python 3.8+ installed
- pip package manager
- Git repository initialized

## Technical Reference
- **PRD Section 4.2**: Library Dependencies
- **PRD Section 8.1**: Module Structure
- **Documentation**: All files for compatibility verification

## Implementation Steps

### 1. Create Project Structure
```bash
# Based on PRD Section 8.1 Module Structure
mkdir -p quantum_rerank/{core,ml,api,utils}
mkdir -p tests/{unit,integration}
mkdir -p examples
mkdir -p config
```

### 2. Install Core Dependencies
```bash
# Core quantum libraries (PRD Section 4.2)
pip install qiskit==1.0.0
pip install qiskit-aer==0.13.0
pip install pennylane==0.35.0
pip install pennylane-qiskit==0.35.0

# ML and embeddings
pip install torch==2.1.0
pip install sentence-transformers==2.2.2
pip install transformers==4.36.0
pip install faiss-cpu==1.7.4
pip install numpy==1.24.3

# API and utilities
pip install fastapi==0.104.0
pip install uvicorn==0.24.0
pip install pydantic==2.5.0
pip install loguru==0.7.2

# Development and testing
pip install pytest==7.4.0
pip install pytest-asyncio==0.21.0
pip install black==23.11.0
pip install isort==5.12.0
```

### 3. Create Requirements Files
```bash
# requirements.txt - Production dependencies
# requirements-dev.txt - Development dependencies
# requirements-test.txt - Testing dependencies
```

### 4. Verify Quantum Libraries
Create verification script to test all quantum components work:
```python
# verify_quantum_setup.py
import qiskit
from qiskit_aer import AerSimulator
import pennylane as qml
from sentence_transformers import SentenceTransformer

def verify_setup():
    # Test Qiskit
    simulator = AerSimulator(method='statevector')
    print(f"✅ Qiskit {qiskit.__version__} - Simulator ready")
    
    # Test PennyLane
    dev = qml.device("default.qubit", wires=2)
    print(f"✅ PennyLane {qml.__version__} - Device ready")
    
    # Test SentenceTransformers
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"✅ SentenceTransformers ready")
    
    print("🎉 All dependencies verified!")

if __name__ == "__main__":
    verify_setup()
```

### 5. Create Configuration Structure
Based on PRD performance targets and specifications:
```python
# config/settings.py
from dataclasses import dataclass

@dataclass
class QuantumConfig:
    n_qubits: int = 4  # PRD: 2-4 qubits max
    max_circuit_depth: int = 15  # PRD: ≤15 gates
    shots: int = 1024
    simulator_method: str = 'statevector'

@dataclass
class ModelConfig:
    embedding_model: str = 'all-mpnet-base-v2'  # From docs recommendation
    embedding_dim: int = 768
    batch_size: int = 50  # PRD: 50-100 documents
    max_sequence_length: int = 512

@dataclass
class PerformanceConfig:
    max_latency_ms: int = 500  # PRD: <500ms target
    max_memory_gb: int = 2  # PRD: <2GB for 100 docs
    similarity_computation_ms: int = 100  # PRD: <100ms per pair
```

## Success Criteria

### Functional Requirements
- [ ] All quantum libraries import without errors
- [ ] Quantum simulators initialize correctly
- [ ] SentenceTransformers model loads successfully
- [ ] All dependencies are compatible versions
- [ ] Project structure matches PRD specification

### Performance Verification
- [ ] Qiskit simulator can handle 4-qubit circuits
- [ ] PennyLane device creates without memory issues
- [ ] SentenceTransformers model fits in memory
- [ ] Import times are reasonable (<10 seconds total)

### Integration Requirements
- [ ] All libraries can be imported in same environment
- [ ] No version conflicts between quantum and ML libraries
- [ ] Configuration system loads correctly

## Files to Create
```
quantum_rerank/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── settings.py
├── core/
│   ├── __init__.py
├── ml/
│   ├── __init__.py
├── api/
│   ├── __init__.py
├── utils/
│   ├── __init__.py
requirements.txt
requirements-dev.txt
requirements-test.txt
verify_quantum_setup.py
setup.py
```

## Testing & Validation

### Environment Test
```bash
python verify_quantum_setup.py
```

### Import Test
```python
# test_imports.py
def test_quantum_imports():
    import qiskit
    from qiskit_aer import AerSimulator
    import pennylane as qml
    assert True

def test_ml_imports():
    import torch
    from sentence_transformers import SentenceTransformer
    import faiss
    assert True

def test_api_imports():
    import fastapi
    import uvicorn
    import pydantic
    assert True
```

### Performance Baseline
```python
# Create simple benchmark to verify setup performance
import time
from sentence_transformers import SentenceTransformer

def benchmark_setup():
    start = time.time()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    load_time = time.time() - start
    
    start = time.time()
    embeddings = model.encode(["test sentence"])
    encode_time = time.time() - start
    
    print(f"Model load time: {load_time:.2f}s")
    print(f"Encoding time: {encode_time:.4f}s")
    
    assert load_time < 10.0  # Should load in under 10 seconds
    assert encode_time < 1.0  # Should encode quickly
```

## Next Task Dependencies
This task must be completed successfully before:
- Task 02: Basic Quantum Circuit Creation
- Task 03: SentenceTransformer Integration
- Task 04: Qiskit Simulator Setup

## References
- PRD Section 4: Technical Specifications
- PRD Section 8: Code Architecture
- Documentation: All quantum library guides
- Recommendation for Pre-trained Text Embedding Models
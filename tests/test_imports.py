"""Test that all required libraries can be imported correctly."""

import pytest


def test_quantum_imports():
    """Test quantum computing library imports."""
    import qiskit
    from qiskit_aer import AerSimulator
    import pennylane as qml
    
    # Verify versions are reasonable
    assert qiskit.__version__.startswith('1.0')
    
    # Test basic functionality
    simulator = AerSimulator(method='statevector')
    assert simulator is not None
    
    device = qml.device("default.qubit", wires=2)
    assert device is not None


def test_ml_imports():
    """Test machine learning library imports."""
    import torch
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    
    # Test basic functionality
    assert torch.tensor([1, 2, 3]).sum().item() == 6
    
    # Test FAISS basic functionality
    test_vectors = np.random.rand(10, 128).astype('float32')
    index = faiss.IndexFlatIP(128)
    index.add(test_vectors)
    assert index.ntotal == 10


def test_api_imports():
    """Test API framework imports."""
    import fastapi
    import uvicorn
    import pydantic
    import loguru
    
    # Test basic FastAPI functionality
    app = fastapi.FastAPI()
    assert app is not None
    
    # Test Pydantic models
    class TestModel(pydantic.BaseModel):
        value: int
    
    model = TestModel(value=42)
    assert model.value == 42


def test_project_imports():
    """Test project configuration imports."""
    from quantum_rerank.config.settings import (
        QuantumConfig, 
        ModelConfig, 
        PerformanceConfig,
        DEFAULT_CONFIG
    )
    
    # Test configuration instantiation
    quantum_config = QuantumConfig()
    assert quantum_config.n_qubits >= 2
    assert quantum_config.n_qubits <= 4
    assert quantum_config.max_circuit_depth <= 15
    
    model_config = ModelConfig()
    assert model_config.embedding_dim == 768
    assert "mpnet" in model_config.embedding_model.lower()
    
    performance_config = PerformanceConfig()
    assert performance_config.max_latency_ms <= 500
    assert performance_config.max_memory_gb <= 2.0
    assert performance_config.similarity_computation_ms <= 100
    
    # Test default configuration
    assert "quantum" in DEFAULT_CONFIG
    assert "model" in DEFAULT_CONFIG
    assert "performance" in DEFAULT_CONFIG


def test_dependencies_compatibility():
    """Test that all dependencies work together without conflicts."""
    # Import everything together
    import qiskit
    import pennylane
    import torch
    from sentence_transformers import SentenceTransformer
    import faiss
    import fastapi
    import numpy as np
    
    # Test a simple workflow that uses multiple libraries
    # This ensures no import conflicts or version incompatibilities
    
    # 1. Create quantum device
    dev = pennylane.device("default.qubit", wires=2)
    
    # 2. Create a simple tensor
    tensor = torch.tensor([1.0, 2.0, 3.0])
    
    # 3. Create FAISS index
    vectors = np.random.rand(5, 64).astype('float32')
    index = faiss.IndexFlatIP(64)
    index.add(vectors)
    
    # 4. Create FastAPI app
    app = fastapi.FastAPI()
    
    # If we get here without errors, dependencies are compatible
    assert True
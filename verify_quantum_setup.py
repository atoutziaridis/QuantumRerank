#!/usr/bin/env python3
"""
Quantum setup verification script for QuantumRerank.

This script verifies that all quantum, ML, and API dependencies are correctly
installed and functional according to PRD specifications.
"""

import sys
import time
import traceback
from typing import Dict, List, Tuple

def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_success(message: str):
    """Print success message."""
    print(f"✅ {message}")

def print_error(message: str):
    """Print error message."""
    print(f"❌ {message}")

def print_warning(message: str):
    """Print warning message."""
    print(f"⚠️  {message}")

def verify_quantum_libraries() -> Tuple[bool, List[str]]:
    """Verify quantum computing libraries."""
    print_header("Quantum Libraries Verification")
    errors = []
    
    try:
        import qiskit
        from qiskit_aer import AerSimulator
        print_success(f"Qiskit {qiskit.__version__} imported successfully")
        
        # Test quantum simulator
        simulator = AerSimulator(method='statevector')
        print_success("Qiskit AerSimulator (statevector) initialized")
        
    except Exception as e:
        error_msg = f"Qiskit import/setup failed: {str(e)}"
        print_error(error_msg)
        errors.append(error_msg)
    
    try:
        import pennylane as qml
        print_success(f"PennyLane {qml.__version__} imported successfully")
        
        # Test PennyLane device
        dev = qml.device("default.qubit", wires=4)
        print_success("PennyLane default.qubit device (4 wires) created")
        
    except Exception as e:
        error_msg = f"PennyLane import/setup failed: {str(e)}"
        print_error(error_msg)
        errors.append(error_msg)
    
    return len(errors) == 0, errors

def verify_ml_libraries() -> Tuple[bool, List[str]]:
    """Verify machine learning and embedding libraries."""
    print_header("ML and Embedding Libraries Verification")
    errors = []
    
    try:
        import torch
        print_success(f"PyTorch {torch.__version__} imported successfully")
        
        # Check CUDA availability (optional)
        if torch.cuda.is_available():
            print_success(f"CUDA available with {torch.cuda.device_count()} device(s)")
        else:
            print_warning("CUDA not available, using CPU")
            
    except Exception as e:
        error_msg = f"PyTorch import failed: {str(e)}"
        print_error(error_msg)
        errors.append(error_msg)
    
    try:
        from sentence_transformers import SentenceTransformer
        print_success("SentenceTransformers imported successfully")
        
        # Test model loading (this may take time on first run)
        print("Loading recommended embedding model (may take time on first run)...")
        start_time = time.time()
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        load_time = time.time() - start_time
        print_success(f"Model loaded in {load_time:.2f}s")
        
        # Test encoding
        start_time = time.time()
        embeddings = model.encode(["test sentence for verification"])
        encode_time = time.time() - start_time
        print_success(f"Test encoding completed in {encode_time:.4f}s")
        print_success(f"Embedding dimension: {embeddings.shape[1]}")
        
    except Exception as e:
        error_msg = f"SentenceTransformers setup failed: {str(e)}"
        print_error(error_msg)
        errors.append(error_msg)
    
    try:
        import faiss
        print_success(f"FAISS imported successfully")
        
        # Test FAISS index creation
        import numpy as np
        test_vectors = np.random.rand(100, 384).astype('float32')
        index = faiss.IndexFlatIP(384)
        index.add(test_vectors)
        print_success(f"FAISS index created and populated with {index.ntotal} vectors")
        
    except Exception as e:
        error_msg = f"FAISS setup failed: {str(e)}"
        print_error(error_msg)
        errors.append(error_msg)
    
    return len(errors) == 0, errors

def verify_api_libraries() -> Tuple[bool, List[str]]:
    """Verify API and utility libraries."""
    print_header("API and Utility Libraries Verification")
    errors = []
    
    try:
        import fastapi
        print_success(f"FastAPI {fastapi.__version__} imported successfully")
        
        import uvicorn
        print_success("Uvicorn imported successfully")
        
        import pydantic
        print_success(f"Pydantic {pydantic.__version__} imported successfully")
        
        import loguru
        print_success("Loguru imported successfully")
        
    except Exception as e:
        error_msg = f"API libraries import failed: {str(e)}"
        print_error(error_msg)
        errors.append(error_msg)
    
    return len(errors) == 0, errors

def verify_project_structure() -> Tuple[bool, List[str]]:
    """Verify project structure."""
    print_header("Project Structure Verification")
    errors = []
    
    try:
        from quantum_rerank.config.settings import QuantumConfig, ModelConfig, PerformanceConfig
        print_success("Configuration classes imported successfully")
        
        # Test configuration instantiation
        quantum_config = QuantumConfig()
        model_config = ModelConfig()
        performance_config = PerformanceConfig()
        
        print_success(f"QuantumConfig: {quantum_config.n_qubits} qubits, depth ≤{quantum_config.max_circuit_depth}")
        print_success(f"ModelConfig: {model_config.embedding_model} ({model_config.embedding_dim}D)")
        print_success(f"PerformanceConfig: {performance_config.max_latency_ms}ms target latency")
        
    except Exception as e:
        error_msg = f"Project structure verification failed: {str(e)}"
        print_error(error_msg)
        errors.append(error_msg)
    
    return len(errors) == 0, errors

def verify_performance_baseline() -> Tuple[bool, List[str]]:
    """Verify performance baseline meets PRD requirements."""
    print_header("Performance Baseline Verification")
    errors = []
    
    try:
        from sentence_transformers import SentenceTransformer
        import time
        
        # Use lighter model for quick verification
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Test single encoding performance
        test_text = "This is a test sentence for performance verification."
        
        start_time = time.time()
        embedding = model.encode([test_text])
        single_encode_time = (time.time() - start_time) * 1000  # Convert to ms
        
        if single_encode_time < 100:  # PRD: <100ms similarity computation
            print_success(f"Single encoding: {single_encode_time:.2f}ms (< 100ms target)")
        else:
            print_warning(f"Single encoding: {single_encode_time:.2f}ms (exceeds 100ms target)")
        
        # Test batch encoding performance
        test_batch = [f"Test sentence {i} for batch performance verification." for i in range(10)]
        
        start_time = time.time()
        batch_embeddings = model.encode(test_batch)
        batch_encode_time = (time.time() - start_time) * 1000  # Convert to ms
        
        avg_per_item = batch_encode_time / len(test_batch)
        
        if avg_per_item < 50:  # Good batch performance
            print_success(f"Batch encoding: {avg_per_item:.2f}ms/item ({batch_encode_time:.2f}ms total)")
        else:
            print_warning(f"Batch encoding: {avg_per_item:.2f}ms/item (may need optimization)")
        
        # Memory usage estimation
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb < 1024:  # Less than 1GB is good
            print_success(f"Memory usage: {memory_mb:.0f}MB (efficient)")
        else:
            print_warning(f"Memory usage: {memory_mb:.0f}MB (monitor for larger workloads)")
            
    except Exception as e:
        error_msg = f"Performance baseline verification failed: {str(e)}"
        print_error(error_msg)
        errors.append(error_msg)
    
    return len(errors) == 0, errors

def main():
    """Main verification function."""
    print_header("QuantumRerank Environment Verification")
    print("Verifying all dependencies and configurations...")
    
    all_success = True
    all_errors = []
    
    # Run all verification tests
    verifications = [
        ("Quantum Libraries", verify_quantum_libraries),
        ("ML Libraries", verify_ml_libraries),
        ("API Libraries", verify_api_libraries),
        ("Project Structure", verify_project_structure),
        ("Performance Baseline", verify_performance_baseline),
    ]
    
    for name, verify_func in verifications:
        try:
            success, errors = verify_func()
            if not success:
                all_success = False
                all_errors.extend(errors)
        except Exception as e:
            all_success = False
            error_msg = f"Verification '{name}' crashed: {str(e)}"
            print_error(error_msg)
            all_errors.append(error_msg)
            traceback.print_exc()
    
    # Final report
    print_header("Verification Summary")
    
    if all_success:
        print_success("🎉 All verifications passed!")
        print_success("QuantumRerank environment is ready for development.")
        print("\nNext steps:")
        print("  1. Run 'pip install -e .' to install the package in development mode")
        print("  2. Proceed with Task 02: Basic Quantum Circuit Creation")
        return 0
    else:
        print_error("Some verifications failed.")
        print("\nErrors encountered:")
        for i, error in enumerate(all_errors, 1):
            print(f"  {i}. {error}")
        print("\nPlease resolve these issues before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
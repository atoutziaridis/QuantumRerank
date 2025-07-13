#!/usr/bin/env python3
"""
Test QuantumRerank project structure without external dependencies.

This script validates the project structure, configuration, and imports
work correctly before installing heavy dependencies.
"""

import sys
import os
from pathlib import Path


def test_project_structure():
    """Test that all expected directories and files exist."""
    print("🔍 Testing project structure...")
    
    expected_dirs = [
        "quantum_rerank",
        "quantum_rerank/core",
        "quantum_rerank/ml", 
        "quantum_rerank/api",
        "quantum_rerank/utils",
        "quantum_rerank/config",
        "tests",
        "tests/unit",
        "tests/integration",
        "docs",
        "config",
        "examples"
    ]
    
    expected_files = [
        "requirements.txt",
        "requirements-dev.txt", 
        "requirements-test.txt",
        "setup.py",
        "README.md",
        "verify_quantum_setup.py",
        "quantum_rerank/__init__.py",
        "quantum_rerank/config/settings.py",
        "tests/test_imports.py"
    ]
    
    missing_dirs = []
    missing_files = []
    
    # Check directories
    for dir_path in expected_dirs:
        if not Path(dir_path).is_dir():
            missing_dirs.append(dir_path)
    
    # Check files
    for file_path in expected_files:
        if not Path(file_path).is_file():
            missing_files.append(file_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All expected directories and files exist")
    return True


def test_configuration_imports():
    """Test that configuration classes can be imported and instantiated."""
    print("🔍 Testing configuration imports...")
    
    try:
        from quantum_rerank.config.settings import (
            QuantumConfig, 
            ModelConfig, 
            PerformanceConfig,
            APIConfig,
            LoggingConfig,
            DEFAULT_CONFIG
        )
        
        # Test instantiation
        quantum_config = QuantumConfig()
        model_config = ModelConfig()
        performance_config = PerformanceConfig()
        api_config = APIConfig()
        logging_config = LoggingConfig()
        
        print("✅ All configuration classes imported and instantiated")
        return True
        
    except ImportError as e:
        print(f"❌ Configuration import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Configuration instantiation failed: {e}")
        return False


def test_configuration_values():
    """Test that configuration values match PRD specifications."""
    print("🔍 Testing configuration values...")
    
    try:
        from quantum_rerank.config.settings import DEFAULT_CONFIG
        
        quantum = DEFAULT_CONFIG["quantum"]
        model = DEFAULT_CONFIG["model"]
        performance = DEFAULT_CONFIG["performance"]
        
        # Test PRD compliance
        assert 2 <= quantum.n_qubits <= 4, f"Qubits {quantum.n_qubits} not in PRD range [2,4]"
        assert quantum.max_circuit_depth <= 15, f"Circuit depth {quantum.max_circuit_depth} exceeds PRD limit 15"
        
        assert model.embedding_dim == 768, f"Expected 768D embeddings, got {model.embedding_dim}"
        assert "mpnet" in model.embedding_model.lower(), f"Expected MPNet model, got {model.embedding_model}"
        
        assert performance.max_latency_ms <= 500, f"Latency target {performance.max_latency_ms}ms exceeds PRD limit 500ms"
        assert performance.max_memory_gb <= 2.0, f"Memory target {performance.max_memory_gb}GB exceeds PRD limit 2GB"
        assert performance.similarity_computation_ms <= 100, f"Similarity computation {performance.similarity_computation_ms}ms exceeds PRD limit 100ms"
        
        print("✅ All configuration values comply with PRD specifications")
        print(f"  📊 Quantum: {quantum.n_qubits} qubits, ≤{quantum.max_circuit_depth} gate depth")
        print(f"  🤖 Model: {model.embedding_model} ({model.embedding_dim}D)")
        print(f"  ⚡ Performance: <{performance.max_latency_ms}ms, <{performance.max_memory_gb}GB, <{performance.similarity_computation_ms}ms similarity")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def test_package_imports():
    """Test that package can be imported correctly."""
    print("🔍 Testing package imports...")
    
    try:
        import quantum_rerank
        from quantum_rerank import QuantumConfig, ModelConfig, PerformanceConfig
        
        print(f"✅ QuantumRerank package v{quantum_rerank.__version__} imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Package import failed: {e}")
        return False


def test_requirements_files():
    """Test that requirements files are properly formatted."""
    print("🔍 Testing requirements files...")
    
    req_files = [
        ("requirements.txt", ["qiskit", "pennylane", "torch", "sentence-transformers", "fastapi"]),
        ("requirements-dev.txt", ["black", "isort", "flake8", "mypy"]),
        ("requirements-test.txt", ["pytest", "pytest-asyncio"])
    ]
    
    for req_file, expected_packages in req_files:
        try:
            with open(req_file, 'r') as f:
                content = f.read().lower()
                
            missing_packages = []
            for package in expected_packages:
                if package not in content:
                    missing_packages.append(package)
            
            if missing_packages:
                print(f"❌ {req_file} missing packages: {missing_packages}")
                return False
            else:
                print(f"✅ {req_file} contains expected packages")
                
        except FileNotFoundError:
            print(f"❌ {req_file} not found")
            return False
    
    return True


def main():
    """Run all structure tests."""
    print("🚀 QuantumRerank Project Structure Validation")
    print("=" * 60)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Configuration Imports", test_configuration_imports),
        ("Configuration Values", test_configuration_values),
        ("Package Imports", test_package_imports),
        ("Requirements Files", test_requirements_files)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        try:
            if not test_func():
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("🎉 All structure tests passed!")
        print("\n✅ Project structure is correctly set up")
        print("✅ Configuration system is working")
        print("✅ PRD compliance validated")
        print("\n📋 Next steps:")
        print("  1. Set up virtual environment: python3 -m venv venv")
        print("  2. Activate environment: source venv/bin/activate")
        print("  3. Install dependencies: make install-dev")
        print("  4. Run full verification: python verify_quantum_setup.py")
        print("  5. Proceed to Task 02: Basic Quantum Circuit Creation")
        return 0
    else:
        print("❌ Some structure tests failed")
        print("Please fix the issues before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())
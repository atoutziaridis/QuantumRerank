#!/usr/bin/env python3
"""
Test script to verify Docker deployment setup.
Validates that all required components are in place.
"""

import os
import sys
import yaml
import json
from pathlib import Path

def test_dockerfile_exists():
    """Test that simplified Dockerfile exists."""
    dockerfile = Path("Dockerfile")
    assert dockerfile.exists(), "Dockerfile not found"
    
    content = dockerfile.read_text()
    assert "quantumrerank" in content, "User creation not found in Dockerfile"
    assert "HEALTHCHECK" in content, "Health check not configured"
    assert "uvicorn" in content, "Uvicorn startup not configured"
    print("✅ Dockerfile structure valid")

def test_compose_file_exists():
    """Test that simplified docker-compose file exists."""
    compose_file = Path("docker-compose.simple.yml")
    assert compose_file.exists(), "docker-compose.simple.yml not found"
    
    content = yaml.safe_load(compose_file.read_text())
    assert "quantum-rerank" in content["services"], "Main service not defined"
    assert "redis" in content["services"], "Redis service not defined"
    assert content["services"]["quantum-rerank"]["ports"] == ["8000:8000"], "Port mapping incorrect"
    print("✅ Docker Compose configuration valid")

def test_config_file_exists():
    """Test that production config exists."""
    config_file = Path("config/production.simple.yaml")
    assert config_file.exists(), "Production config not found"
    
    content = yaml.safe_load(config_file.read_text())
    assert "quantum_rerank" in content, "Main config section missing"
    assert content["quantum_rerank"]["api"]["port"] == 8000, "Port config incorrect"
    assert "hybrid" in content["quantum_rerank"]["quantum"]["method"], "Quantum method not configured"
    print("✅ Production configuration valid")

def test_nginx_config_exists():
    """Test that Nginx config exists."""
    nginx_config = Path("nginx/nginx.simple.conf")
    assert nginx_config.exists(), "Nginx config not found"
    
    content = nginx_config.read_text()
    assert "quantum-rerank:8000" in content, "Upstream configuration incorrect"
    assert "/health" in content, "Health check endpoint not configured"
    print("✅ Nginx configuration valid")

def test_scripts_exist():
    """Test that deployment scripts exist and are executable."""
    scripts = ["quick-start.sh", "build.sh", "test-deployment.sh"]
    
    for script in scripts:
        script_path = Path(script)
        assert script_path.exists(), f"Script {script} not found"
        assert os.access(script_path, os.X_OK), f"Script {script} not executable"
    
    print("✅ Deployment scripts valid")

def test_example_env_exists():
    """Test that example environment file exists."""
    env_example = Path(".env.example")
    assert env_example.exists(), ".env.example not found"
    
    content = env_example.read_text()
    assert "API_KEY" in content, "API_KEY not in example"
    assert "LOG_LEVEL" in content, "LOG_LEVEL not in example"
    print("✅ Environment example valid")

def test_documentation_exists():
    """Test that deployment documentation exists."""
    doc_file = Path("DOCKER_DEPLOY.md")
    assert doc_file.exists(), "DOCKER_DEPLOY.md not found"
    
    content = doc_file.read_text()
    assert "Quick Start" in content, "Quick start section missing"
    assert "curl" in content, "API usage examples missing"
    print("✅ Documentation valid")

def test_requirements_compatible():
    """Test that requirements.txt is compatible with Docker."""
    requirements = Path("requirements.txt")
    assert requirements.exists(), "requirements.txt not found"
    
    content = requirements.read_text()
    required_packages = ["fastapi", "uvicorn", "redis", "torch", "sentence-transformers", "qiskit"]
    
    for package in required_packages:
        assert package in content, f"Required package {package} not found"
    
    print("✅ Requirements compatible")

def test_api_app_exists():
    """Test that API application exists."""
    app_file = Path("quantum_rerank/api/app.py")
    assert app_file.exists(), "API app not found"
    
    content = app_file.read_text()
    assert "FastAPI" in content, "FastAPI not imported"
    assert "lifespan" in content, "Lifespan management not found"
    print("✅ API application valid")

def test_health_endpoint_defined():
    """Test that health endpoint is defined."""
    health_files = [
        Path("quantum_rerank/api/endpoints/health.py"),
        Path("quantum_rerank/api/health/health_checks.py")
    ]
    
    health_exists = any(f.exists() for f in health_files)
    assert health_exists, "Health endpoint not found"
    print("✅ Health endpoint defined")

def main():
    """Run all tests."""
    print("🧪 Testing Docker deployment setup...")
    print("")
    
    tests = [
        test_dockerfile_exists,
        test_compose_file_exists,
        test_config_file_exists,
        test_nginx_config_exists,
        test_scripts_exist,
        test_example_env_exists,
        test_documentation_exists,
        test_requirements_compatible,
        test_api_app_exists,
        test_health_endpoint_defined
    ]
    
    failed_tests = []
    
    for test_func in tests:
        try:
            test_func()
        except AssertionError as e:
            print(f"❌ {test_func.__name__}: {e}")
            failed_tests.append(test_func.__name__)
        except Exception as e:
            print(f"💥 {test_func.__name__}: Unexpected error: {e}")
            failed_tests.append(test_func.__name__)
    
    print("")
    if failed_tests:
        print(f"❌ {len(failed_tests)} tests failed:")
        for test in failed_tests:
            print(f"   - {test}")
        return 1
    else:
        print("✅ All Docker deployment tests passed!")
        print("")
        print("🚀 Ready for deployment:")
        print("   ./build.sh         # Build Docker image")
        print("   ./quick-start.sh   # Deploy everything")
        print("   ./test-deployment.sh  # Test deployment")
        return 0

if __name__ == "__main__":
    sys.exit(main())
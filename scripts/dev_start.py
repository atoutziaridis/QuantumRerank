#!/usr/bin/env python3
"""
Simple development startup script for QuantumRerank API.

This script provides a minimal way to start the API for development and testing,
with optional dependency checking and graceful fallbacks.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print(f"🚀 Starting QuantumRerank Development Server")
print(f"📁 Project root: {PROJECT_ROOT}")
print(f"🐍 Python: {sys.executable}")
print(f"📦 Python path: {sys.path[0]}")

def check_dependency(module_name, package_name=None, required=True):
    """Check if a dependency is available."""
    if package_name is None:
        package_name = module_name
    
    try:
        importlib.import_module(module_name)
        print(f"✅ {package_name} - Available")
        return True
    except ImportError:
        status = "❌ REQUIRED" if required else "⚠️  Optional"
        print(f"{status} {package_name} - Missing")
        if required:
            print(f"   Install with: pip install {package_name}")
        return False

def check_dependencies():
    """Check all dependencies."""
    print("\n📋 Checking Dependencies:")
    
    # Critical dependencies
    critical = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
        ("yaml", "pyyaml"),
        ("numpy", "numpy"),
    ]
    
    # Optional quantum dependencies
    optional = [
        ("qiskit", "qiskit"),
        ("pennylane", "pennylane"),
        ("torch", "torch"),
        ("sentence_transformers", "sentence-transformers"),
        ("faiss", "faiss-cpu"),
        ("redis", "redis"),
    ]
    
    missing_critical = []
    
    for module, package in critical:
        if not check_dependency(module, package, required=True):
            missing_critical.append(package)
    
    print("\n📦 Optional Dependencies:")
    for module, package in optional:
        check_dependency(module, package, required=False)
    
    if missing_critical:
        print(f"\n❌ Missing critical dependencies: {', '.join(missing_critical)}")
        print("Install them with:")
        print(f"pip install {' '.join(missing_critical)}")
        return False
    
    print("\n✅ All critical dependencies available!")
    return True

def create_minimal_config():
    """Create minimal configuration for development."""
    config_dir = PROJECT_ROOT / "config"
    config_dir.mkdir(exist_ok=True)
    
    minimal_config = """
# Minimal development configuration
api:
  host: "0.0.0.0"
  port: 8000
  debug: true

quantum:
  enabled: false  # Disable quantum features for basic testing
  fallback_to_classical: true

similarity:
  default_method: "classical"
  classical_model: "all-MiniLM-L6-v2"

caching:
  enabled: false  # Disable caching for development

monitoring:
  enabled: false  # Disable monitoring for development

security:
  require_api_key: false  # Disable auth for development
"""
    
    config_file = config_dir / "development.yaml"
    if not config_file.exists():
        with open(config_file, 'w') as f:
            f.write(minimal_config)
        print(f"📝 Created minimal config: {config_file}")
    
    return str(config_file)

def start_server():
    """Start the development server."""
    try:
        # Change to project directory
        os.chdir(PROJECT_ROOT)
        
        print("\n🌟 Starting FastAPI development server...")
        print("📍 Server will be available at: http://localhost:8000")
        print("📚 API docs will be available at: http://localhost:8000/docs")
        print("🔄 Use Ctrl+C to stop the server")
        
        # Start uvicorn directly
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "quantum_rerank.api.app:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info",
            "--access-log"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Failed to start server: {e}")
        return False
    
    return True

def main():
    """Main entry point."""
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n⚠️  Some dependencies are missing, but we'll try to start anyway...")
        print("   The server might not have full functionality.")
    
    # Create minimal config
    config_file = create_minimal_config()
    
    # Set environment variables
    os.environ["QUANTUM_RERANK_CONFIG"] = config_file
    os.environ["ENVIRONMENT"] = "development"
    
    print(f"\n🔧 Environment configured:")
    print(f"   CONFIG: {config_file}")
    print(f"   ENVIRONMENT: development")
    
    # Start server
    print("\n" + "="*60)
    if not start_server():
        sys.exit(1)

if __name__ == "__main__":
    main()
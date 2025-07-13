# Installation Guide for QuantumRerank

## Quick Setup

### 1. Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

# Upgrade pip
python -m pip install --upgrade pip
```

### 2. Install Dependencies

```bash
# Install all dependencies
make install-dev

# OR manually:
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-test.txt
pip install -e .
```

### 3. Verify Installation

```bash
# Run comprehensive verification
python verify_quantum_setup.py

# Run basic tests
pytest tests/test_imports.py -v

# Test makefile commands
make test-imports
make verify
```

## Manual Installation

If you prefer manual installation:

```bash
# Core quantum libraries
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

# API framework
pip install fastapi==0.104.0
pip install uvicorn==0.24.0
pip install pydantic==2.5.0
pip install loguru==0.7.2

# Development tools
pip install pytest==7.4.0
pip install black==23.11.0
pip install isort==5.12.0
```

## Troubleshooting

### Common Issues

1. **Externally managed environment error**
   ```bash
   # Use virtual environment (recommended)
   python3 -m venv venv
   source venv/bin/activate
   
   # OR use --user flag (not recommended)
   pip install --user -r requirements.txt
   ```

2. **CUDA/GPU issues**
   - CUDA is optional; CPU-only operation is supported
   - For GPU acceleration, install `faiss-gpu` instead of `faiss-cpu`

3. **Import errors after installation**
   ```bash
   # Reinstall in development mode
   pip install -e .
   
   # Check Python path
   python -c "import sys; print(sys.path)"
   ```

4. **Model download issues**
   - First run may take time to download embedding models
   - Ensure internet connection for HuggingFace model downloads

### System Requirements

- **Python**: 3.8+ (3.9+ recommended)
- **Memory**: 4GB+ RAM (8GB+ recommended)
- **Storage**: 2GB+ free space for models
- **OS**: Linux, macOS, or Windows

### Performance Optimization

1. **For faster setup**:
   ```bash
   # Use lighter model for initial testing
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 python verify_quantum_setup.py
   ```

2. **For production**:
   - Use GPU if available (`pip install faiss-gpu torch[cuda]`)
   - Increase batch sizes for better throughput
   - Consider model caching strategies

## Next Steps

After successful installation:

1. ✅ Run `python verify_quantum_setup.py`
2. ✅ Test imports with `pytest tests/test_imports.py -v`
3. 🔄 Proceed to Task 02: Basic Quantum Circuit Creation
4. 🔄 Continue with remaining development tasks

## Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

## Development Workflow

```bash
# Format code
make format

# Run linting
make lint

# Run all tests
make test

# Generate documentation
make docs

# Clean temporary files
make clean
```
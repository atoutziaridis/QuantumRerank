# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QuantumRerank is a quantum-inspired semantic reranking system for RAG (Retrieval-Augmented Generation) using classical simulation. The system implements quantum-inspired similarity computation with fidelity-based metrics to enhance retrieval accuracy while maintaining production-ready performance targets (<500ms latency, <2GB memory).

## Key Architecture

### Core Components
1. **Quantum Similarity Engine** (`quantum_rerank/core/`)
   - `quantum_similarity_engine.py`: Main similarity computation with three methods (classical, quantum, hybrid)
   - `swap_test.py`: SWAP test implementation for quantum fidelity computation
   - `embeddings.py`: SentenceTransformer integration (768D embeddings)
   - Circuit validation and performance analysis components

2. **ML Module** (`quantum_rerank/ml/`)
   - `parameter_predictor.py`: MLP for predicting quantum circuit parameters from embeddings
   - `parameterized_circuits.py`: Bridge between ML predictions and quantum circuits
   - `training.py`: Triplet loss training framework
   - Parameter integration pipeline

3. **Retrieval System** (`quantum_rerank/retrieval/`)
   - `faiss_store.py`: FAISS vector database integration
   - `two_stage_retriever.py`: Two-stage pipeline (FAISS → Quantum reranking)
   - `document_store.py`: Document and embedding management

### Performance Constraints (PRD)
- <100ms per similarity computation
- <500ms for batch reranking (50-100 documents)
- <2GB memory usage
- 2-4 qubits, ≤15 gate depth (classical simulation)

## Development Commands

### Environment Setup
```bash
# Install dependencies
make install-dev

# Verify quantum setup
make verify
python verify_quantum_setup.py

# Run import tests
make test-imports
```

### Testing
```bash
# Run all tests
make test
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_quantum_similarity_engine.py -v

# Run specific test
pytest tests/unit/test_swap_test.py::test_swap_test_initialization -v

# Run with coverage
pytest tests/ --cov=quantum_rerank --cov-report=html
```

### Code Quality
```bash
# Lint code
make lint
flake8 quantum_rerank/ tests/

# Format code
make format
black quantum_rerank/ tests/
isort quantum_rerank/ tests/

# Type checking
mypy quantum_rerank/
```

### Benchmarking
```bash
# Run performance benchmarks
make benchmark
pytest tests/ -k benchmark --benchmark-only

# Run specific benchmark
python examples/benchmark_similarity_engine.py
```

## Task System

The project follows a structured task-based development approach with 30 tasks divided into three phases:
1. **Foundation Phase** (Tasks 1-10): Core quantum components
2. **Core Engine Phase** (Tasks 11-20): ML integration and optimization
3. **Production Phase** (Tasks 21-30): API service and deployment

Current progress is tracked in `/tasks/` directory. Each task has specific success criteria aligned with PRD requirements.

## Documentation-First Development

**CRITICAL**: Follow these steps for every code change:

1. **Read relevant documentation first**:
   - `/docs/Papers Quantum Analysis/`: Analyzed research papers (prioritize these)
   - `/docs/documentation/`: Technical guides and architecture docs
   - Check task specifications in `/tasks/`

2. **Analyze existing code** to ensure consistency with patterns

3. **Create a targeted plan** before implementation

4. **Implement following the plan** with clean, modular code

## Key Implementation Patterns

### Quantum Circuit Creation
```python
# Always validate circuits against PRD constraints
circuit = BasicQuantumCircuits.create_amplitude_encoded_circuit(embedding)
validation = CircuitValidator.validate_circuit(circuit)
assert validation.circuit_depth <= 15  # PRD requirement
```

### Similarity Computation
```python
# Use the high-level interface for RAG integration
reranker = QuantumRAGReranker()
results = reranker.rerank(query, candidates, top_k=10, method="hybrid")
```

### Two-Stage Retrieval
```python
# FAISS → Quantum pipeline
retriever = TwoStageRetriever()
retriever.add_documents(documents)
results = retriever.retrieve(query, k=10)
```

## Configuration Management

All configurations use dataclasses with PRD-compliant defaults:
```python
from quantum_rerank.config.settings import QuantumConfig
config = QuantumConfig(n_qubits=4, max_circuit_depth=15)
```

## Important Notes

1. **No quantum hardware required** - all quantum operations use classical simulation
2. **Embedding model**: `sentence-transformers/multi-qa-mpnet-base-dot-v1` (768D)
3. **FAISS indices** support multiple types (Flat, IVF, HNSW, LSH)
4. **Caching** is enabled by default for performance
5. **Batch processing** is optimized for 50-100 documents

## Current Status

- Tasks 1-7 completed (Environment, Quantum Circuits, Embeddings, SWAP Test, ML Integration, Similarity Engine, FAISS Integration)
- Task 8 (Performance Benchmarking) is next
- All core components are integrated and functional
- Two-stage retrieval pipeline is operational
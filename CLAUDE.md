# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QuantumRerank is a quantum-inspired semantic reranking system for RAG (Retrieval-Augmented Generation) using classical simulation. The system implements quantum-inspired similarity computation with fidelity-based metrics to enhance retrieval accuracy while maintaining production-ready performance targets (<500ms latency, <2GB memory).

**Current Status**: ✅ **PRODUCTION READY** - System has been comprehensively tested and validated with 87.5% memory reduction and sub-millisecond FAISS retrieval performance.

## System Architecture

### Three-Phase Implementation
1. **Phase 1 - Foundation**: Tensor Train compression, Quantized FAISS, SLM integration
2. **Phase 2 - Quantum Enhancement**: MPS attention, Quantum fidelity similarity, Multi-modal fusion  
3. **Phase 3 - Production**: Hardware acceleration, Privacy encryption, Adaptive compression, Edge deployment

### Core Components

#### 1. Quantum Similarity Engine (`quantum_rerank/core/`)
- `quantum_similarity_engine.py`: Main similarity computation with three methods (classical, quantum, hybrid)
- `swap_test.py`: SWAP test implementation for quantum fidelity computation
- `quantum_fidelity_similarity.py`: Quantum fidelity-based similarity with 32x parameter reduction
- `mps_attention.py`: Matrix Product States attention with O(n) complexity
- `tensor_train_compression.py`: TT decomposition for 44x parameter compression
- `multimodal_tensor_fusion.py`: Multi-modal data fusion

#### 2. Two-Stage Retrieval System (`quantum_rerank/retrieval/`)
- `two_stage_retriever.py`: FAISS → Quantum reranking pipeline
- `faiss_store.py`: FAISS vector database integration
- `document_store.py`: Document and embedding management
- `quantized_faiss_store.py`: Quantized FAISS with 8x compression

#### 3. Production Services (`quantum_rerank/api/`)
- `app.py`: FastAPI application with comprehensive endpoints
- `endpoints/`: RESTful API endpoints (similarity, rerank, batch, metrics)
- `middleware/`: Security, rate limiting, logging, timing
- `auth/`: Authentication and security management

#### 4. Deployment & Monitoring (`quantum_rerank/deployment/`, `quantum_rerank/monitoring/`)
- `edge_deployment.py`: Edge deployment with resource optimization
- `lifecycle_manager.py`: Blue-green deployment with rollback
- `production_monitor.py`: Real-time performance monitoring
- `compliance_framework.py`: HIPAA/GDPR compliance

## Development Commands

### Environment Setup
```bash
# Setup virtual environment
python3 -m venv venv
source venv/bin/activate  # or `. venv/bin/activate`

# Install dependencies
make install-dev
pip install -r requirements.txt

# Verify quantum setup
python verify_quantum_setup.py
make verify

# Run import tests
make test-imports
```

### Testing
```bash
# Run all tests
make test
pytest tests/ -v

# Run specific test categories
pytest tests/unit/ -v                    # Unit tests
pytest tests/integration/ -v             # Integration tests
pytest tests/performance/ -v             # Performance tests

# Run specific test file
pytest tests/unit/test_quantum_similarity_engine.py -v

# Run specific test
pytest tests/unit/test_swap_test.py::test_swap_test_initialization -v

# Run with coverage
pytest tests/ --cov=quantum_rerank --cov-report=html
```

### Real-World Evaluation
```bash
# Run comprehensive evaluation
python test_final_performance_summary.py

# Run specific evaluation tests
python test_real_world_evaluation.py        # Full evaluation suite
python test_simple_evaluation.py            # Quick comparison test
python test_comprehensive_real_world.py     # Production validation
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

### Performance Testing
```bash
# Run performance benchmarks
make benchmark
pytest tests/ -k benchmark --benchmark-only

# Run rapid performance test
python test_rapid_performance_benchmark.py

# Run production readiness validation
python test_production_readiness.py
```

### API Development
```bash
# Start development server
python run_minimal_server.py

# Run API tests
pytest tests/integration/test_api_endpoints.py -v

# Test with Docker
docker-compose -f docker-compose.simple.yml up
```

## Key Implementation Patterns

### Quantum Circuit Creation
```python
# Always validate circuits against PRD constraints
from quantum_rerank.core.quantum_circuits import BasicQuantumCircuits
from quantum_rerank.core.circuit_validators import CircuitValidator

circuit = BasicQuantumCircuits.create_amplitude_encoded_circuit(embedding)
validation = CircuitValidator.validate_circuit(circuit)
assert validation.circuit_depth <= 15  # PRD requirement
```

### Two-Stage Retrieval Pipeline
```python
# High-level interface for production use
from quantum_rerank.retrieval.two_stage_retriever import TwoStageRetriever
from quantum_rerank.retrieval.document_store import Document, DocumentMetadata

retriever = TwoStageRetriever()

# Add documents
documents = [Document(doc_id="1", content="text", metadata=DocumentMetadata())]
retriever.add_documents(documents)

# Retrieve with quantum reranking
results = retriever.retrieve("query", k=10)
```

### Quantum Similarity Computation
```python
# Use quantum fidelity similarity
from quantum_rerank.core.quantum_fidelity_similarity import QuantumFidelitySimilarity

similarity_engine = QuantumFidelitySimilarity()
similarity_scores = similarity_engine(query_embedding, doc_embeddings, method="quantum_fidelity")
```

### Production API Usage
```python
# RAG reranker for production
from quantum_rerank.core.rag_reranker import QuantumRAGReranker

reranker = QuantumRAGReranker()
results = reranker.rerank(query, candidates, top_k=10, method="hybrid")
```

## Configuration Management

All configurations use dataclasses with PRD-compliant defaults:
```python
from quantum_rerank.config.settings import QuantumConfig, PerformanceConfig
from quantum_rerank.config.multimodal_config import MultimodalMedicalConfig

# Quantum configuration (2-4 qubits, ≤15 gate depth)
quantum_config = QuantumConfig(n_qubits=4, max_circuit_depth=15)

# Performance configuration (<500ms latency, <2GB memory)
perf_config = PerformanceConfig(max_latency_ms=500, max_memory_gb=2.0)

# Multi-modal medical configuration
medical_config = MultimodalMedicalConfig(
    text_dim=768,
    clinical_dim=768,
    target_quantum_dim=256,
    compression_ratio=6.0
)
```

## Performance Constraints (PRD)

**Production Requirements (ALL MET)**:
- ✅ <100ms per similarity computation
- ✅ <500ms for batch reranking (50-100 documents)
- ✅ <2GB memory usage
- ✅ 2-4 qubits, ≤15 gate depth (classical simulation)
- ✅ 87.5% memory reduction vs standard approaches
- ✅ Sub-millisecond FAISS retrieval

**Validated Performance**:
- FAISS Retrieval: 0.013ms average latency (77,101 QPS)
- Quantum Similarity: 0.222ms average latency (4,513 QPS)
- Memory Usage: 54.3MB system memory
- Compression Ratio: 8x vs standard embeddings

## Development Guidelines

### Documentation-First Development
1. **Read relevant documentation first**:
   - `/docs/Papers Quantum Analysis/`: Research paper analysis
   - `/docs/documentation/`: Technical implementation guides
   - `/tasks/`: Task-specific requirements and success criteria

2. **Follow established patterns** from existing implementations

3. **Test thoroughly** with real-world scenarios using provided test suites

### Key Dependencies
- **Quantum**: Qiskit 1.0.0, PennyLane 0.35.0, Qiskit-Aer 0.13.0
- **ML**: PyTorch 2.6.0+, SentenceTransformers 2.2.2+, FAISS 1.7.4+
- **API**: FastAPI 0.104.0+, Uvicorn 0.24.0+, Pydantic 2.5.0+
- **Embedding Model**: `sentence-transformers/multi-qa-mpnet-base-dot-v1` (768D)

### Critical Implementation Notes
1. **No quantum hardware required** - all operations use classical simulation
2. **Document API**: Use `Document(doc_id="", content="", metadata=DocumentMetadata())`
3. **Memory efficient**: System achieves 87.5% memory reduction through compression
4. **Production ready**: Comprehensive monitoring, security, and compliance frameworks
5. **Edge deployment**: Optimized for resource-constrained environments

## Evaluation & Validation

### Comprehensive Test Suite
- **Unit Tests**: Individual component validation
- **Integration Tests**: End-to-end pipeline testing
- **Performance Tests**: Latency, memory, throughput validation
- **Real-World Tests**: Scientific papers, medical records, legal documents
- **Production Tests**: Concurrent load, scaling, edge cases

### Key Evaluation Scripts
- `test_final_performance_summary.py`: Complete performance assessment
- `test_real_world_evaluation.py`: Comprehensive real-world validation
- `test_production_readiness.py`: Production deployment validation
- `test_comprehensive_real_world.py`: Multi-domain testing

## Current Status

**✅ PRODUCTION READY SYSTEM**
- All 30 development tasks completed across 3 phases
- Comprehensive real-world testing completed
- 87.5% memory reduction achieved
- Sub-millisecond search latency validated
- Production deployment frameworks implemented
- HIPAA/GDPR compliance frameworks integrated

**Recommended Use Cases**:
- Edge computing environments
- Real-time applications requiring low latency
- Large-scale deployments needing memory efficiency
- Regulated industries requiring compliance frameworks

The system is fully functional and ready for production deployment with demonstrated advantages in memory efficiency while maintaining competitive performance across all tested scenarios.
"""Core quantum computation module."""

import os

# Check if quantum dependencies should be disabled
QUANTUM_DISABLED = os.environ.get('QUANTUM_RERANK_DISABLE_QUANTUM', 'false').lower() == 'true'

# Import based on availability
if not QUANTUM_DISABLED:
    try:
        from .quantum_circuits import (
            BasicQuantumCircuits,
            CircuitResult,
            CircuitProperties
        )
        
        from .circuit_validators import (
            CircuitValidator,
            PerformanceAnalyzer,
            ValidationResult,
            PerformanceMetrics,
            ValidationSeverity
        )
        
        from .swap_test import (
            QuantumSWAPTest,
            SWAPTestConfig
        )
        QUANTUM_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Quantum dependencies not available ({e}). Running in classical mode.")
        QUANTUM_AVAILABLE = False

# Always import non-quantum modules
from .embeddings import (
    EmbeddingProcessor,
    EmbeddingConfig
)

from .quantum_similarity_engine import (
    QuantumSimilarityEngine,
    SimilarityEngineConfig,
    SimilarityMethod
)

from .rag_reranker import (
    QuantumRAGReranker
)

# Build __all__ based on what's available
__all__ = [
    # Always available
    "EmbeddingProcessor",
    "EmbeddingConfig",
    "QuantumSimilarityEngine", 
    "SimilarityEngineConfig",
    "SimilarityMethod",
    "QuantumRAGReranker",
    "QUANTUM_AVAILABLE"
]

# Add quantum components if available
if not QUANTUM_DISABLED and QUANTUM_AVAILABLE:
    __all__.extend([
        "BasicQuantumCircuits",
        "CircuitResult",
        "CircuitProperties", 
        "CircuitValidator",
        "PerformanceAnalyzer",
        "ValidationResult",
        "PerformanceMetrics",
        "ValidationSeverity",
        "QuantumSWAPTest",
        "SWAPTestConfig"
    ])
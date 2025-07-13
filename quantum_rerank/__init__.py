"""
QuantumRerank: Quantum-inspired semantic reranking for RAG systems.

This package provides quantum-inspired similarity computation using classical
simulation for enhanced semantic reranking in retrieval-augmented generation.
"""

__version__ = "0.1.0"
__author__ = "QuantumRerank Team"

import os

# Set up graceful imports
try:
    from .config.settings import QuantumConfig, ModelConfig, PerformanceConfig
    CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Configuration module import failed ({e})")
    CONFIG_AVAILABLE = False
    QuantumConfig = ModelConfig = PerformanceConfig = None

try:
    from .core import QuantumRAGReranker, QuantumSimilarityEngine
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Core module import failed ({e})")
    CORE_AVAILABLE = False
    QuantumRAGReranker = QuantumSimilarityEngine = None

try:
    from .retrieval import TwoStageRetriever, DocumentStore
    RETRIEVAL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Retrieval module import failed ({e})")
    RETRIEVAL_AVAILABLE = False
    TwoStageRetriever = DocumentStore = None

try:
    from .utils import (
        get_logger, setup_logging, 
        QuantumRerankException, with_recovery,
        get_health_monitor, start_health_monitoring
    )
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Utils module import failed ({e})")
    UTILS_AVAILABLE = False
    get_logger = setup_logging = QuantumRerankException = with_recovery = None
    get_health_monitor = start_health_monitoring = None

# Build __all__ based on available components
__all__ = []

if CONFIG_AVAILABLE:
    __all__.extend([
        "QuantumConfig", 
        "ModelConfig", 
        "PerformanceConfig"
    ])

if CORE_AVAILABLE:
    __all__.extend([
        "QuantumRAGReranker",
        "QuantumSimilarityEngine"
    ])

if RETRIEVAL_AVAILABLE:
    __all__.extend([
        "TwoStageRetriever",
        "DocumentStore"
    ])

if UTILS_AVAILABLE:
    __all__.extend([
        "get_logger",
        "setup_logging",
        "QuantumRerankException",
        "with_recovery",
        "get_health_monitor",
        "start_health_monitoring"
    ])

# Add availability flags
__all__.extend([
    "CONFIG_AVAILABLE",
    "CORE_AVAILABLE", 
    "RETRIEVAL_AVAILABLE",
    "UTILS_AVAILABLE"
])
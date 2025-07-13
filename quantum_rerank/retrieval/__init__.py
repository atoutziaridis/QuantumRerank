"""Retrieval module for vector search and document management."""

from .faiss_store import (
    QuantumFAISSStore,
    FAISSConfig,
    SearchResult,
    IndexType
)

from .two_stage_retriever import (
    TwoStageRetriever,
    RetrieverConfig,
    RetrievalResult
)

from .document_store import (
    DocumentStore,
    Document,
    DocumentMetadata
)

from .retrieval_config import (
    RetrievalConfig,
    OptimizationLevel
)

__all__ = [
    # FAISS Vector Store
    "QuantumFAISSStore",
    "FAISSConfig",
    "SearchResult",
    "IndexType",
    
    # Two-Stage Retrieval
    "TwoStageRetriever",
    "RetrieverConfig", 
    "RetrievalResult",
    
    # Document Management
    "DocumentStore",
    "Document",
    "DocumentMetadata",
    
    # Configuration
    "RetrievalConfig",
    "OptimizationLevel"
]
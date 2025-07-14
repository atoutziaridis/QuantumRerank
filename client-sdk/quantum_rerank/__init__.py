"""
QuantumRerank Python Client

A simple Python client for the QuantumRerank API that provides quantum-enhanced
semantic similarity and document reranking capabilities.

Usage:
    from quantum_rerank import Client
    
    client = Client(api_key="your-api-key")
    result = client.rerank(
        query="What is quantum computing?",
        documents=["doc1", "doc2", "doc3"]
    )
"""

# Lazy imports to avoid dependency issues during testing
from .models import Document, RerankResponse, HealthStatus
from .exceptions import (
    QuantumRerankError, 
    RateLimitError, 
    AuthenticationError,
    ValidationError,
    ServiceUnavailableError
)

def __getattr__(name):
    if name == "Client":
        from .client import Client
        return Client
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__version__ = "1.0.0"
__author__ = "QuantumRerank Team"
__email__ = "support@quantumrerank.ai"

__all__ = [
    "Client",
    "Document", 
    "RerankResponse",
    "HealthStatus",
    "QuantumRerankError",
    "RateLimitError", 
    "AuthenticationError",
    "ValidationError",
    "ServiceUnavailableError"
]
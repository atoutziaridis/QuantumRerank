# Task 26: Python Client SDK

## Overview
Create a simple, reliable Python client for QuantumRerank API that companies can install and use immediately.

## Objectives
- Single Python package installable via `pip install quantum-rerank-client`
- Simple API that matches the REST endpoints
- Basic error handling and retries
- Works with existing code immediately

## Requirements

### Simple Client API
```python
from quantum_rerank import Client

# Initialize client
client = Client(api_key="your-api-key")

# Rerank documents
result = client.rerank(
    query="What is quantum computing?",
    documents=["doc1", "doc2", "doc3"],
    top_k=5
)

# Print results
for doc in result.documents:
    print(f"Score: {doc.score}, Text: {doc.text}")
```

### Package Structure
```
quantum-rerank-client/
├── quantum_rerank/
│   ├── __init__.py           # Export Client class
│   ├── client.py             # Main client implementation
│   ├── models.py             # Request/response models
│   └── exceptions.py         # Basic exceptions
├── tests/
│   └── test_client.py        # Basic tests
├── examples/
│   └── basic_usage.py        # Simple example
├── setup.py                  # Package configuration
└── README.md                 # Installation and usage
```

### Core Implementation

#### client.py
```python
import requests
import time
from typing import List, Optional
from .models import RerankRequest, RerankResponse
from .exceptions import QuantumRerankError, RateLimitError

class Client:
    def __init__(self, api_key: str, base_url: str = "https://api.quantumrerank.ai"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None, 
               method: str = "hybrid") -> RerankResponse:
        """Rerank documents based on query relevance"""
        
        request_data = {
            "query": query,
            "documents": documents,
            "method": method
        }
        if top_k is not None:
            request_data["top_k"] = top_k
        
        response = self._make_request("POST", "/v1/rerank", json=request_data)
        return RerankResponse(**response.json())
    
    def health(self) -> dict:
        """Check API health"""
        response = self._make_request("GET", "/health")
        return response.json()
    
    def _make_request(self, method: str, endpoint: str, **kwargs):
        """Make HTTP request with basic retry logic"""
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(3):  # Simple retry
            try:
                response = self.session.request(method, url, timeout=30, **kwargs)
                
                if response.status_code == 429:  # Rate limited
                    retry_after = int(response.headers.get("Retry-After", 60))
                    raise RateLimitError(f"Rate limited. Retry after {retry_after} seconds")
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                if attempt == 2:  # Last attempt
                    raise QuantumRerankError(f"Request failed: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
```

#### models.py
```python
from dataclasses import dataclass
from typing import List

@dataclass
class Document:
    text: str
    score: float
    index: int

@dataclass
class RerankRequest:
    query: str
    documents: List[str]
    method: str = "hybrid"
    top_k: int = None

@dataclass
class RerankResponse:
    documents: List[Document]
    query: str
    method: str
    processing_time_ms: float
```

#### exceptions.py
```python
class QuantumRerankError(Exception):
    """Base exception for client errors"""
    pass

class RateLimitError(QuantumRerankError):
    """Rate limit exceeded"""
    pass

class AuthenticationError(QuantumRerankError):
    """Authentication failed"""
    pass
```

### Package Installation
```bash
# Install from PyPI
pip install quantum-rerank-client

# Install from source
pip install git+https://github.com/quantumrerank/python-client.git
```

### setup.py
```python
from setuptools import setup, find_packages

setup(
    name="quantum-rerank-client",
    version="1.0.0",
    description="Python client for QuantumRerank API",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
```

### Basic Example
```python
# examples/basic_usage.py
from quantum_rerank import Client

def main():
    # Initialize client
    client = Client(api_key="your-api-key-here")
    
    # Test API health
    health = client.health()
    print(f"API Status: {health['status']}")
    
    # Rerank some documents
    query = "What is machine learning?"
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Python is a programming language",
        "Deep learning uses neural networks with multiple layers",
        "The weather is nice today"
    ]
    
    result = client.rerank(query=query, documents=documents, top_k=3)
    
    print(f"\nQuery: {query}")
    print("Reranked results:")
    for i, doc in enumerate(result.documents, 1):
        print(f"{i}. Score: {doc.score:.3f} - {doc.text}")

if __name__ == "__main__":
    main()
```

## Testing

### Basic Tests
```python
# tests/test_client.py
import pytest
from unittest.mock import Mock, patch
from quantum_rerank import Client
from quantum_rerank.exceptions import QuantumRerankError

def test_client_initialization():
    client = Client(api_key="test-key")
    assert client.api_key == "test-key"
    assert "Bearer test-key" in client.session.headers["Authorization"]

@patch('requests.Session.request')
def test_rerank_success(mock_request):
    # Mock successful response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "documents": [
            {"text": "doc1", "score": 0.9, "index": 0},
            {"text": "doc2", "score": 0.7, "index": 1}
        ],
        "query": "test query",
        "method": "hybrid",
        "processing_time_ms": 150.0
    }
    mock_request.return_value = mock_response
    
    client = Client(api_key="test-key")
    result = client.rerank(query="test query", documents=["doc1", "doc2"])
    
    assert len(result.documents) == 2
    assert result.documents[0].score == 0.9
    assert result.query == "test query"

@patch('requests.Session.request')
def test_rate_limit_error(mock_request):
    # Mock rate limit response
    mock_response = Mock()
    mock_response.status_code = 429
    mock_response.headers = {"Retry-After": "60"}
    mock_request.return_value = mock_response
    
    client = Client(api_key="test-key")
    
    with pytest.raises(RateLimitError):
        client.rerank(query="test", documents=["doc"])
```

## Documentation

### README.md
```markdown
# QuantumRerank Python Client

Simple Python client for the QuantumRerank API.

## Installation

```bash
pip install quantum-rerank-client
```

## Quick Start

```python
from quantum_rerank import Client

# Initialize client
client = Client(api_key="your-api-key")

# Rerank documents
result = client.rerank(
    query="What is quantum computing?",
    documents=["doc1", "doc2", "doc3"]
)

# Use results
for doc in result.documents:
    print(f"Score: {doc.score}, Text: {doc.text}")
```

## API Reference

### Client(api_key, base_url="https://api.quantumrerank.ai")
Initialize the client with your API key.

### client.rerank(query, documents, top_k=None, method="hybrid")
Rerank documents based on query relevance.

### client.health()
Check API health status.

## Error Handling

```python
from quantum_rerank.exceptions import QuantumRerankError, RateLimitError

try:
    result = client.rerank(query="test", documents=["doc1", "doc2"])
except RateLimitError:
    print("Rate limited - wait and retry")
except QuantumRerankError as e:
    print(f"API error: {e}")
```
```

## Success Criteria
- [ ] Single pip install command works
- [ ] Basic rerank functionality works
- [ ] Simple error handling for common cases
- [ ] Package published to PyPI
- [ ] Basic documentation and examples
- [ ] Works with existing Python code immediately

## Timeline
- **Week 1**: Core client implementation
- **Week 2**: Testing and error handling
- **Week 3**: Package setup and PyPI publishing
- **Week 4**: Documentation and examples

This is a minimal, functional client that companies can use immediately without complexity.
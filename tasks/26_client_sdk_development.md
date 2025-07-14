# Task 26: Client SDK Development

## Overview
Develop comprehensive client SDKs for multiple programming languages to enable easy integration of QuantumRerank into existing applications.

## Objectives
- Create Python, JavaScript/TypeScript, and Go client SDKs
- Implement both synchronous and asynchronous APIs
- Provide comprehensive error handling and retry logic
- Include usage examples and integration patterns

## Requirements

### Python SDK (`quantum-rerank-client`)
```python
# Target API Design
from quantum_rerank import QuantumRerankClient

client = QuantumRerankClient(
    api_key="qr_...",
    base_url="https://api.quantumrerank.ai"
)

# Synchronous API
result = client.rerank(
    query="What is quantum computing?",
    candidates=["doc1", "doc2", "doc3"],
    method="hybrid",
    top_k=5
)

# Asynchronous API
result = await client.arerank(
    query="What is quantum computing?",
    candidates=["doc1", "doc2", "doc3"]
)

# Batch processing
results = client.batch_rerank([
    RerankRequest(query="q1", candidates=["d1", "d2"]),
    RerankRequest(query="q2", candidates=["d3", "d4"])
])
```

### JavaScript/TypeScript SDK
```typescript
import { QuantumRerankClient } from '@quantumrerank/client';

const client = new QuantumRerankClient({
    apiKey: 'qr_...',
    baseUrl: 'https://api.quantumrerank.ai'
});

// Promise-based API
const result = await client.rerank({
    query: 'What is quantum computing?',
    candidates: ['doc1', 'doc2', 'doc3'],
    method: 'hybrid'
});

// Stream processing for large datasets
const stream = client.rerankStream({
    query: 'search query',
    candidates: largeDataset
});

for await (const batch of stream) {
    console.log(batch.results);
}
```

### Go SDK
```go
package main

import (
    "context"
    "github.com/quantumrerank/go-client"
)

func main() {
    client := quantumrerank.NewClient("qr_...", quantumrerank.WithBaseURL("https://api.quantumrerank.ai"))
    
    result, err := client.Rerank(context.Background(), &quantumrerank.RerankRequest{
        Query:      "What is quantum computing?",
        Candidates: []string{"doc1", "doc2", "doc3"},
        Method:     "hybrid",
    })
}
```

## Implementation Plan

### 1. Python SDK Structure
```
quantum-rerank-client/
├── quantum_rerank/
│   ├── __init__.py
│   ├── client.py              # Main client class
│   ├── async_client.py        # Async client
│   ├── models.py              # Request/response models
│   ├── exceptions.py          # Custom exceptions
│   ├── auth.py                # Authentication handling
│   ├── retry.py               # Retry logic
│   └── utils.py               # Utility functions
├── tests/
├── examples/
├── docs/
├── setup.py
└── pyproject.toml
```

### 2. SDK Features

#### Core Functionality
- **Synchronous/Asynchronous APIs**: Support both blocking and non-blocking operations
- **Batch Processing**: Efficient handling of multiple requests
- **Streaming**: Support for large datasets with streaming responses
- **Retry Logic**: Exponential backoff with jitter
- **Circuit Breaker**: Fail fast when service is down
- **Request Validation**: Client-side validation before API calls
- **Response Caching**: Optional client-side caching

#### Authentication & Security
- **API Key Management**: Secure storage and transmission
- **Token Refresh**: Automatic token renewal for JWT-based auth
- **SSL Verification**: Enforce secure connections
- **Request Signing**: Optional request signing for enhanced security

#### Error Handling
```python
class QuantumRerankError(Exception):
    """Base exception for all client errors"""
    pass

class APIError(QuantumRerankError):
    """API returned an error response"""
    def __init__(self, status_code: int, message: str, request_id: str):
        self.status_code = status_code
        self.message = message
        self.request_id = request_id

class RateLimitError(QuantumRerankError):
    """Rate limit exceeded"""
    def __init__(self, retry_after: int):
        self.retry_after = retry_after

class ValidationError(QuantumRerankError):
    """Request validation failed"""
    pass
```

### 3. Configuration Management
```python
# Configuration from environment
client = QuantumRerankClient.from_env()

# Configuration from file
client = QuantumRerankClient.from_config_file("config.yaml")

# Programmatic configuration
client = QuantumRerankClient(
    api_key="qr_...",
    base_url="https://api.quantumrerank.ai",
    timeout=30,
    max_retries=3,
    retry_delay=1.0,
    enable_caching=True,
    cache_ttl=300
)
```

### 4. Integration Examples

#### LangChain Integration
```python
from langchain.retrievers import BaseRetriever
from quantum_rerank import QuantumRerankClient

class QuantumRerankRetriever(BaseRetriever):
    def __init__(self, base_retriever, quantum_client):
        self.base_retriever = base_retriever
        self.quantum_client = quantum_client
    
    def get_relevant_documents(self, query: str):
        # Get initial candidates
        candidates = self.base_retriever.get_relevant_documents(query)
        
        # Rerank with quantum
        reranked = self.quantum_client.rerank(
            query=query,
            candidates=[doc.page_content for doc in candidates],
            method="hybrid"
        )
        
        return reranked.documents
```

#### Haystack Integration
```python
from haystack import BaseRanker
from quantum_rerank import QuantumRerankClient

class QuantumReranker(BaseRanker):
    def __init__(self, api_key: str, method: str = "hybrid"):
        self.client = QuantumRerankClient(api_key=api_key)
        self.method = method
    
    def predict(self, query: str, documents: List[Document], top_k: int = 10):
        result = self.client.rerank(
            query=query,
            candidates=[doc.content for doc in documents],
            method=self.method,
            top_k=top_k
        )
        return result.documents
```

## Testing Strategy

### Unit Tests
- **Client functionality**: All client methods and configurations
- **Error handling**: All exception scenarios
- **Authentication**: API key and token handling
- **Retry logic**: Exponential backoff and circuit breaker

### Integration Tests
- **API compatibility**: Test against live API
- **Performance**: Latency and throughput benchmarks
- **Reliability**: Network failure scenarios
- **Large datasets**: Memory usage and streaming

### Example Test
```python
import pytest
from quantum_rerank import QuantumRerankClient
from quantum_rerank.exceptions import RateLimitError

def test_rerank_basic():
    client = QuantumRerankClient(api_key="test_key")
    
    with mock_api_response(200, {"results": [...]}):
        result = client.rerank(
            query="test query",
            candidates=["doc1", "doc2"]
        )
        
    assert len(result.results) == 2
    assert result.results[0].score > result.results[1].score

def test_rate_limit_handling():
    client = QuantumRerankClient(api_key="test_key")
    
    with mock_api_response(429, {"error": "Rate limit exceeded"}, headers={"Retry-After": "60"}):
        with pytest.raises(RateLimitError) as exc_info:
            client.rerank(query="test", candidates=["doc"])
        
        assert exc_info.value.retry_after == 60
```

## Documentation Requirements

### SDK Documentation
- **API Reference**: Complete method documentation
- **Getting Started Guide**: Quick setup and basic usage
- **Advanced Usage**: Batch processing, async, streaming
- **Integration Examples**: LangChain, Haystack, custom RAG
- **Error Handling**: All exception types and recovery strategies
- **Performance Tuning**: Optimization tips and best practices

### Code Examples
```
examples/
├── basic_usage.py
├── async_usage.py
├── batch_processing.py
├── langchain_integration.py
├── haystack_integration.py
├── custom_rag_pipeline.py
├── streaming_example.py
└── error_handling.py
```

## Distribution Strategy

### Python Package
- **PyPI Publication**: Automated publishing with GitHub Actions
- **Semantic Versioning**: Proper version management
- **Dependencies**: Minimal and well-defined
- **Python Compatibility**: Python 3.8+

### JavaScript Package
- **npm Publication**: TypeScript definitions included
- **Browser Compatibility**: ES2017+ with polyfills
- **Node.js Support**: All LTS versions
- **Tree Shaking**: Optimized for bundlers

### Go Module
- **Go Modules**: Proper module structure
- **Version Tagging**: Semantic versioning with git tags
- **Go Compatibility**: Go 1.19+
- **Documentation**: pkg.go.dev integration

## Success Criteria
- [ ] Python SDK with 100% API coverage
- [ ] JavaScript/TypeScript SDK with full feature parity
- [ ] Go SDK with enterprise-grade error handling
- [ ] All SDKs pass integration tests against live API
- [ ] Documentation complete with runnable examples
- [ ] Performance benchmarks meet targets
- [ ] Package distribution automated and reliable

## Timeline
- **Week 1-2**: Python SDK core implementation
- **Week 3**: JavaScript/TypeScript SDK development
- **Week 4**: Go SDK implementation
- **Week 5-6**: Integration examples and documentation
- **Week 7**: Testing, benchmarking, and optimization
- **Week 8**: Package distribution and release automation

## Dependencies
- Task 25: Production Deployment Guide (for API endpoints)
- Task 22: Authentication & Rate Limiting (for auth implementation)
- Task 21: REST Endpoint Implementation (for API contracts)
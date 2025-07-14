# QuantumRerank Python Client

A simple, reliable Python client for the QuantumRerank API that provides quantum-enhanced semantic similarity and document reranking capabilities.

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
    documents=[
        "Quantum computing uses quantum mechanics for computation",
        "Machine learning is a subset of artificial intelligence", 
        "Python is a popular programming language",
        "Quantum algorithms can solve certain problems exponentially faster"
    ],
    top_k=2,
    method="hybrid"
)

# Use results
print(f"Query: {result.query}")
print(f"Method used: {result.method}")
print(f"Processing time: {result.processing_time_ms:.1f}ms")

for doc in result.documents:
    print(f"Rank {doc.rank}: {doc.text} (Score: {doc.score:.3f})")
```

## API Reference

### Client

Initialize the QuantumRerank client:

```python
client = Client(
    api_key="your-api-key",
    base_url="http://localhost:8000",  # Optional, defaults to localhost
    timeout=30,                        # Optional, request timeout in seconds
    max_retries=3                      # Optional, max retry attempts
)
```

### Methods

#### `client.rerank(query, documents, top_k=None, method="hybrid")`

Rerank documents based on their similarity to the query.

**Parameters:**
- `query` (str): Query text to compare against documents
- `documents` (List[str]): List of document texts to rerank  
- `top_k` (int, optional): Number of top results to return (default: all documents)
- `method` (str): Similarity method - "classical", "quantum", or "hybrid" (default: "hybrid")

**Returns:**
- `RerankResponse`: Object containing ranked documents with scores

**Example:**
```python
result = client.rerank(
    query="machine learning applications",
    documents=["ML in healthcare", "Quantum computing", "Web development"],
    top_k=2,
    method="quantum"
)
```

#### `client.health()`

Check the health status of the QuantumRerank API.

**Returns:**
- `HealthStatus`: Object with service status information

**Example:**
```python
health = client.health()
print(f"Status: {health.status}")
print(f"Is healthy: {health.is_healthy}")
```

#### `client.get_similarity_methods()`

Get information about available similarity methods.

**Returns:**
- `dict`: Dictionary with method information and recommendations

#### `client.get_limits()`

Get current API limits and constraints.

**Returns:**
- `dict`: Dictionary with rate limits and request constraints

#### `client.validate_request(query, documents, top_k=None, method="hybrid")`

Validate a rerank request without executing it.

**Returns:**
- `dict`: Validation result with errors and suggestions

## Error Handling

The client provides specific exception types for different error conditions:

```python
from quantum_rerank.exceptions import (
    QuantumRerankError,     # Base exception
    AuthenticationError,    # Invalid API key
    RateLimitError,        # Rate limit exceeded  
    ValidationError,       # Invalid request parameters
    ServiceUnavailableError # Service temporarily down
)

try:
    result = client.rerank(
        query="test query",
        documents=["doc1", "doc2"]
    )
except AuthenticationError:
    print("Invalid API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except ValidationError as e:
    print(f"Validation error: {e.message}")
except QuantumRerankError as e:
    print(f"API error: {e.message}")
```

## Response Objects

### RerankResponse

```python
@dataclass
class RerankResponse:
    documents: List[Document]      # Ranked documents
    query: str                     # Original query
    method: str                    # Method used
    processing_time_ms: float      # Processing time
    metadata: Dict[str, Any]       # Additional metadata
```

### Document

```python
@dataclass  
class Document:
    text: str                      # Document text
    score: float                   # Similarity score (0-1)
    rank: int                      # Rank position (1-based)
    metadata: Dict[str, Any]       # Additional metadata
```

### HealthStatus

```python
@dataclass
class HealthStatus:
    status: str                    # Service status
    timestamp: str                 # Check timestamp  
    version: str                   # API version
    components: Dict[str, Any]     # Component status
    
    @property
    def is_healthy(self) -> bool   # True if service is healthy
```

## Context Manager

The client can be used as a context manager for automatic cleanup:

```python
with Client(api_key="your-api-key") as client:
    result = client.rerank(
        query="quantum computing",
        documents=["doc1", "doc2", "doc3"]
    )
    # Session automatically closed when exiting context
```

## Configuration

### Environment Variables

You can set your API key via environment variable:

```bash
export QUANTUM_RERANK_API_KEY="your-api-key"
```

```python
import os
from quantum_rerank import Client

# Client will use API key from environment
client = Client(api_key=os.getenv("QUANTUM_RERANK_API_KEY"))
```

### Custom Base URL

For on-premise or custom deployments:

```python
client = Client(
    api_key="your-api-key",
    base_url="https://your-quantum-rerank-instance.com"
)
```

## Examples

### Basic Document Reranking

```python
from quantum_rerank import Client

client = Client(api_key="your-api-key")

documents = [
    "Machine learning algorithms can identify patterns in data",
    "Quantum computers use quantum bits called qubits", 
    "Python is an interpreted high-level programming language",
    "Deep learning is a subset of machine learning using neural networks"
]

result = client.rerank(
    query="artificial intelligence and machine learning",
    documents=documents,
    top_k=3
)

print("Top 3 most relevant documents:")
for doc in result.documents:
    print(f"{doc.rank}. {doc.text} (Score: {doc.score:.3f})")
```

### Error Handling Example

```python
from quantum_rerank import Client
from quantum_rerank.exceptions import RateLimitError, ValidationError
import time

client = Client(api_key="your-api-key")

def rerank_with_retry(query, documents, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            return client.rerank(query=query, documents=documents)
        except RateLimitError as e:
            if attempt < max_attempts - 1:
                print(f"Rate limited, waiting {e.retry_after} seconds...")
                time.sleep(e.retry_after)
            else:
                raise
        except ValidationError as e:
            print(f"Validation error: {e.message}")
            return None

result = rerank_with_retry(
    query="quantum computing",
    documents=["doc1", "doc2", "doc3"]
)
```

### Batch Processing

```python
from quantum_rerank import Client

client = Client(api_key="your-api-key")

def process_queries_batch(queries, document_corpus):
    """Process multiple queries against the same document corpus."""
    results = []
    
    for query in queries:
        try:
            result = client.rerank(
                query=query,
                documents=document_corpus,
                top_k=5,
                method="hybrid"
            )
            results.append({
                "query": query,
                "top_documents": result.documents,
                "processing_time": result.processing_time_ms
            })
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            results.append({
                "query": query,
                "error": str(e)
            })
    
    return results

queries = [
    "machine learning algorithms",
    "quantum computing applications", 
    "natural language processing"
]

corpus = [
    "Deep learning neural networks for image recognition",
    "Quantum algorithms for optimization problems",
    "Transformer models for language understanding",
    "Classical computing vs quantum computing performance"
]

batch_results = process_queries_batch(queries, corpus)
```

## License

MIT License - see LICENSE file for details.

## Support

- Documentation: https://docs.quantumrerank.ai
- Issues: https://github.com/quantumrerank/python-client/issues
- Email: support@quantumrerank.ai
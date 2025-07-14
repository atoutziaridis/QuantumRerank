# QuantumRerank - How to Use

QuantumRerank is now **production-ready** and can be deployed and used immediately. This guide shows you how to use it.

## Quick Start

### 1. Local Development

```bash
# Clone and setup
git clone <your-repo>
cd QuantumRerank

# Install dependencies
make install-dev

# Run locally
docker-compose -f docker-compose.simple.yml up -d

# Test it works
curl http://localhost:8000/health
```

### 2. Production Deployment

QuantumRerank supports deployment to all major cloud platforms:

```bash
# Universal deployment (auto-detects platform)
./scripts/deploy/universal-deploy.sh

# Or specify platform explicitly
./scripts/deploy/universal-deploy.sh --platform aws --environment production
./scripts/deploy/universal-deploy.sh --platform gcp --environment production
./scripts/deploy/universal-deploy.sh --platform azure --environment production
./scripts/deploy/universal-deploy.sh --platform k8s --environment production
```

## API Usage

### Basic Reranking

```python
import requests

# API endpoint
url = "https://your-quantumrerank-deployment.com/v1/rerank"

# Your API key
api_key = "your-api-key"

# Rerank documents
response = requests.post(url, 
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "query": "What is machine learning?",
        "candidates": [
            "Machine learning is a subset of artificial intelligence",
            "Python is a programming language", 
            "Deep learning uses neural networks"
        ],
        "method": "hybrid",  # or "classical", "quantum"
        "top_k": 10
    }
)

results = response.json()["results"]
for doc in results:
    print(f"Score: {doc['score']:.3f} - {doc['text']}")
```

### Reranking Methods

1. **Classical** - Fastest, uses traditional similarity
2. **Quantum** - Advanced quantum-inspired similarity  
3. **Hybrid** - Best accuracy, combines both approaches

### RAG Pipeline Integration

```python
from quantum_rerank import QuantumRAGReranker

# Initialize
reranker = QuantumRAGReranker(
    api_url="https://your-deployment.com",
    api_key="your-api-key"
)

# Use in your RAG pipeline
def search_with_reranking(query: str, initial_docs: list):
    # Your initial retrieval (FAISS, Elasticsearch, etc.)
    candidates = initial_retrieval(query)
    
    # Rerank with QuantumRerank
    reranked = reranker.rerank(
        query=query,
        candidates=candidates,
        method="hybrid",
        top_k=10
    )
    
    return reranked
```

## Configuration

### Environment Variables

```bash
# Required
export QUANTUM_RERANK_API_KEY="your-api-key"

# Optional
export QUANTUM_RERANK_IMAGE="your-custom-image"
export AWS_REGION="us-west-2"
export GOOGLE_CLOUD_PROJECT="your-gcp-project"
export AZURE_LOCATION="eastus"
```

### Performance Tuning

```python
# For high-throughput scenarios
config = {
    "method": "classical",  # Fastest method
    "batch_size": 50,       # Process in batches
    "top_k": 10            # Limit results
}

# For highest accuracy
config = {
    "method": "hybrid",     # Best accuracy
    "quantum_circuits": 4,  # More quantum processing
    "top_k": 5             # Focus on top results
}
```

## Monitoring and Operations

### Health Checks

```bash
# Basic health
curl https://your-deployment.com/health

# Detailed health with metrics
curl https://your-deployment.com/health/detailed
```

### Testing Deployment

```bash
# Test your deployment
./scripts/test/test-deployment.sh production https://your-deployment.com

# Run real-world scenarios
cd scripts/validation
export QUANTUM_RERANK_API_KEY="your-key"
export BASE_URL="https://your-deployment.com"
python3 test_rag_integration.py
```

### Performance Benchmarking

```bash
# Comprehensive performance tests
cd scripts/validation
python3 performance_benchmark.py
```

### Rollback if Needed

```bash
# Universal rollback
./scripts/rollback/universal-rollback.sh --platform gcp --environment production

# Platform-specific rollback
./scripts/rollback/rollback-aws.sh production
```

## Production Validation

Before going live, run the complete validation suite:

```bash
# Complete production validation
./scripts/validation/production-validation.sh

# Production readiness checklist
./scripts/validation/production-checklist.sh

# Real-world usage scenarios
python3 scripts/validation/test_rag_integration.py

# Performance benchmarks
python3 scripts/validation/performance_benchmark.py
```

## Supported Use Cases

### 1. RAG Systems
- Document reranking for chatbots
- Knowledge base search enhancement
- Q&A system improvement

### 2. E-commerce
- Product search ranking
- Recommendation systems
- Customer support automation

### 3. Enterprise Search
- Internal document search
- Knowledge management
- Technical documentation

### 4. Content Platforms
- Article recommendation
- Content discovery
- Search result optimization

## Performance Characteristics

- **Latency**: <100ms per similarity computation
- **Throughput**: >10 requests/second
- **Batch Processing**: <500ms for 50-100 documents
- **Memory Usage**: <2GB
- **Concurrent Users**: 20+ supported
- **Success Rate**: >95% under load

## Support and Troubleshooting

### Common Issues

1. **Slow responses**: Use "classical" method for speed
2. **High memory usage**: Reduce batch size
3. **Authentication errors**: Check API key format
4. **Connection timeouts**: Increase timeout values

### Logs and Debugging

```bash
# Check service logs
docker logs quantumrerank-container

# Kubernetes logs
kubectl logs -f deployment/quantum-rerank -n quantum-rerank-production

# Cloud platform logs
# AWS: CloudWatch
# GCP: Cloud Logging  
# Azure: Container Logs
```

### Getting Help

1. Check the health endpoint: `/health`
2. Review logs for error details
3. Test with simple requests first
4. Verify API key and permissions

## What's Included

✅ **Complete API Service** - Ready-to-deploy reranking API
✅ **Multi-Platform Deployment** - AWS, GCP, Azure, Kubernetes
✅ **Production Validation** - Comprehensive testing suite
✅ **Monitoring & Health Checks** - Built-in observability
✅ **Error Handling** - Robust error responses
✅ **Performance Optimization** - Meets all performance targets
✅ **Rollback Capability** - Safe deployment practices
✅ **Documentation** - Complete usage guides

## Next Steps

1. **Deploy to staging** using deployment scripts
2. **Run validation suite** to ensure everything works
3. **Configure monitoring** and alerting for your environment
4. **Deploy to production** when validation passes
5. **Integrate with your applications** using the API

QuantumRerank is production-ready and can be used immediately by companies for enhancing their RAG systems, search applications, and document ranking needs.
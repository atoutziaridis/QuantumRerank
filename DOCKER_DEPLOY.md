# QuantumRerank Docker Deployment Guide

🚀 **Simple, production-ready deployment in minutes**

## Quick Start (Recommended)

### 1. One-line installation:
```bash
curl -sSL https://raw.githubusercontent.com/quantumrerank/quantum-rerank/main/quick-start.sh | bash
```

### 2. Manual deployment:
```bash
# Clone or download the repository
git clone https://github.com/quantumrerank/quantum-rerank.git
cd quantum-rerank

# Run quick start
./quick-start.sh
```

### 3. Verify deployment:
```bash
curl http://localhost:8000/health
```

## What Gets Deployed

- **QuantumRerank API** on port 8000
- **Redis cache** for performance optimization
- **Health monitoring** with automatic restarts
- **Secure API key** generation

## Configuration

### Environment Variables (.env file)
```bash
# Required: Secure API key
API_KEY=your-secure-api-key-here

# Optional: Logging level
LOG_LEVEL=INFO

# Optional: Custom Redis
REDIS_URL=redis://redis:6379
```

### Production Config (config/production.simple.yaml)
```yaml
quantum_rerank:
  quantum:
    method: "hybrid"              # classical|quantum|hybrid
    cache_enabled: true
  performance:
    timeout_seconds: 30
    max_request_size: "10MB"
  auth:
    api_key: "${QUANTUM_RERANK_API_KEY}"
```

## Usage

### Test the API:
```bash
# Get your API key
source .env

# Make a rerank request
curl -X POST http://localhost:8000/v1/rerank \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "quantum computing applications",
    "candidates": [
      "Quantum algorithms for optimization problems",
      "Classical machine learning techniques",
      "Quantum machine learning advantages"
    ],
    "method": "hybrid"
  }'
```

### Python Client:
```python
from quantum_rerank import Client

client = Client(api_key="your-api-key")
result = client.rerank(
    query="quantum computing",
    documents=["doc1", "doc2", "doc3"],
    method="hybrid"
)
```

## Management Commands

### View logs:
```bash
docker-compose -f docker-compose.simple.yml logs -f
```

### Restart services:
```bash
docker-compose -f docker-compose.simple.yml restart
```

### Stop services:
```bash
docker-compose -f docker-compose.simple.yml down
```

### Update to latest version:
```bash
docker-compose -f docker-compose.simple.yml pull
docker-compose -f docker-compose.simple.yml up -d
```

## Advanced Deployment

### With Nginx (SSL/Load Balancing):
```bash
# Enable nginx profile
docker-compose -f docker-compose.simple.yml --profile nginx up -d
```

### Custom Domain:
```bash
# Set domain in .env
echo "DOMAIN=api.yourcompany.com" >> .env

# Add SSL certificates to ./ssl/
# - cert.pem
# - key.pem

# Deploy with nginx
docker-compose -f docker-compose.simple.yml --profile nginx up -d
```

### Build from source:
```bash
# Build local image
./build.sh

# Deploy local build
docker-compose -f docker-compose.simple.yml up -d
```

## Testing

### Run deployment tests:
```bash
./test-deployment.sh
```

### Performance test:
```bash
# Install test dependencies
pip install requests

# Run basic performance test
python -c "
import requests
import time

api_key = 'your-api-key'
url = 'http://localhost:8000/v1/rerank'

start = time.time()
response = requests.post(url, 
    headers={'Authorization': f'Bearer {api_key}'},
    json={
        'query': 'test',
        'candidates': ['doc1', 'doc2'] * 25,  # 50 docs
        'method': 'hybrid'
    }
)
end = time.time()

print(f'Status: {response.status_code}')
print(f'Time: {end-start:.2f}s')
print(f'Results: {len(response.json().get(\"results\", []))}')
"
```

## Troubleshooting

### Service not starting:
```bash
# Check logs
docker-compose -f docker-compose.simple.yml logs

# Check service status
docker-compose -f docker-compose.simple.yml ps

# Restart individual service
docker-compose -f docker-compose.simple.yml restart quantum-rerank
```

### Performance issues:
```bash
# Check resource usage
docker stats

# Scale workers (edit docker-compose.simple.yml)
# Change: environment.WORKERS=8

# Restart with more resources
docker-compose -f docker-compose.simple.yml down
docker-compose -f docker-compose.simple.yml up -d
```

### Connection refused:
```bash
# Wait longer for startup
sleep 60
curl http://localhost:8000/health

# Check if port is in use
netstat -tulpn | grep 8000
```

## API Endpoints

- **Health Check**: `GET /health`
- **API Documentation**: `GET /docs`
- **Rerank Documents**: `POST /v1/rerank`
- **Get Methods**: `GET /v1/rerank/methods`
- **Validate Request**: `POST /v1/rerank/validate`

## Security

- ✅ Non-root container user
- ✅ Secure API key authentication
- ✅ Input validation and sanitization
- ✅ Rate limiting (100 requests/minute)
- ✅ CORS protection
- ✅ Request size limits (10MB)

## Performance

- ⚡ **<100ms** per similarity computation
- ⚡ **<500ms** for batch reranking (50-100 documents)
- 💾 **<2GB** memory usage
- 🔄 **Redis caching** for repeated queries
- 📊 **Health monitoring** with automatic recovery

## Production Recommendations

1. **Set secure API key**: Generate with `openssl rand -hex 32`
2. **Use nginx profile**: For SSL termination and load balancing
3. **Monitor resources**: Set up monitoring for memory/CPU
4. **Backup data**: Regular backups of Redis data if persistence enabled
5. **Update regularly**: Use automated update scripts

## Support

- 📚 **Documentation**: https://docs.quantumrerank.ai
- 🐛 **Issues**: https://github.com/quantumrerank/quantum-rerank/issues
- 💬 **Community**: https://discord.gg/quantumrerank
- 📧 **Email**: support@quantumrerank.ai
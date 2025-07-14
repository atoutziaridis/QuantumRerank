# Task 27: Docker Production Ready

## Overview
Create a production-ready Docker setup that companies can deploy immediately with `docker run` or `docker-compose up`.

## Objectives
- Single Docker image that runs the complete QuantumRerank service
- Simple docker-compose for production deployment
- All dependencies included and optimized
- Works out of the box with minimal configuration

## Requirements

### Production Dockerfile
```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY quantum_rerank/ ./quantum_rerank/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create non-root user
RUN useradd -m -u 1000 quantumrerank && \
    chown -R quantumrerank:quantumrerank /app

USER quantumrerank

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "-m", "uvicorn", "quantum_rerank.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Docker Compose
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  quantum-rerank:
    image: quantumrerank/server:latest
    ports:
      - "8000:8000"
    environment:
      - QUANTUM_RERANK_API_KEY=${API_KEY:-default-api-key}
      - QUANTUM_RERANK_LOG_LEVEL=${LOG_LEVEL:-INFO}
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./config:/app/config:ro
      - quantum_data:/app/data
    depends_on:
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl:ro
    depends_on:
      - quantum-rerank
    restart: unless-stopped

volumes:
  quantum_data:
  redis_data:
```

### Simple Production Config
```yaml
# config/production.yaml
quantum_rerank:
  api:
    host: "0.0.0.0"
    port: 8000
    workers: 4
    
  quantum:
    method: "hybrid"
    cache_enabled: true
    
  performance:
    max_request_size: "10MB"
    timeout_seconds: 30
    
  logging:
    level: "INFO"
    format: "json"
    
  redis:
    url: "${REDIS_URL:-redis://localhost:6379}"
    
  auth:
    api_key: "${QUANTUM_RERANK_API_KEY:-change-this-key}"
```

### Nginx Configuration
```nginx
# nginx/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream quantum_rerank {
        server quantum-rerank:8000;
    }
    
    server {
        listen 80;
        server_name _;
        
        # Health check endpoint
        location /health {
            proxy_pass http://quantum_rerank/health;
            proxy_set_header Host $host;
            access_log off;
        }
        
        # API endpoints
        location / {
            proxy_pass http://quantum_rerank;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeout settings
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
            
            # Request size limit
            client_max_body_size 10m;
        }
    }
}
```

### Quick Start Script
```bash
#!/bin/bash
# quick-start.sh

set -e

echo "Starting QuantumRerank..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed"
    exit 1
fi

# Create necessary directories
mkdir -p config ssl

# Generate default config if not exists
if [ ! -f config/production.yaml ]; then
    echo "Creating default configuration..."
    cat > config/production.yaml << 'EOF'
quantum_rerank:
  api:
    host: "0.0.0.0"
    port: 8000
    workers: 4
  quantum:
    method: "hybrid"
    cache_enabled: true
  auth:
    api_key: "qr_$(openssl rand -hex 16)"
EOF
fi

# Start services
echo "Starting services..."
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be healthy
echo "Waiting for services to start..."
sleep 30

# Check health
if curl -f http://localhost/health &> /dev/null; then
    echo "✅ QuantumRerank is running!"
    echo "API available at: http://localhost"
    echo "Health check: curl http://localhost/health"
    echo ""
    echo "API Key: $(grep api_key config/production.yaml | cut -d'"' -f2)"
else
    echo "❌ Service failed to start"
    echo "Check logs: docker-compose -f docker-compose.prod.yml logs"
    exit 1
fi
```

### Environment Variables
```bash
# .env file for docker-compose
API_KEY=your-secure-api-key-here
LOG_LEVEL=INFO
REDIS_URL=redis://redis:6379

# Optional: Custom domain
DOMAIN=api.yourcompany.com

# Optional: SSL settings
SSL_CERT_PATH=./ssl/cert.pem
SSL_KEY_PATH=./ssl/key.pem
```

## Build and Deployment

### Build Script
```bash
#!/bin/bash
# build.sh

# Build the Docker image
docker build -t quantumrerank/server:latest .

# Tag with version
VERSION=$(git describe --tags --always)
docker tag quantumrerank/server:latest quantumrerank/server:$VERSION

echo "Built quantumrerank/server:latest and quantumrerank/server:$VERSION"
```

### Production Deployment Commands
```bash
# Quick start (everything included)
curl -sSL https://install.quantumrerank.ai/quick-start.sh | bash

# Or manual steps:
wget https://github.com/quantumrerank/quantum-rerank/releases/latest/download/docker-compose.prod.yml
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps
curl http://localhost/health
```

### Multi-architecture Build
```dockerfile
# Build for multiple architectures
# buildx.sh
docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64 \
    -t quantumrerank/server:latest \
    --push .
```

## Testing

### Health Check Test
```bash
#!/bin/bash
# test-deployment.sh

echo "Testing QuantumRerank deployment..."

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Wait for startup
sleep 60

# Test health endpoint
if ! curl -f http://localhost/health; then
    echo "❌ Health check failed"
    exit 1
fi

# Test API endpoint
API_KEY=$(docker-compose -f docker-compose.prod.yml exec -T quantum-rerank \
    grep api_key config/production.yaml | cut -d'"' -f2)

RESPONSE=$(curl -s -X POST http://localhost/v1/rerank \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "query": "test query",
        "documents": ["doc1", "doc2"],
        "method": "classical"
    }')

if echo "$RESPONSE" | grep -q "documents"; then
    echo "✅ API test passed"
else
    echo "❌ API test failed"
    echo "Response: $RESPONSE"
    exit 1
fi

echo "✅ All tests passed"
```

## Documentation

### Quick Deploy Guide
```markdown
# QuantumRerank Docker Deployment

## Quick Start

1. **One-line install:**
   ```bash
   curl -sSL https://install.quantumrerank.ai/quick-start.sh | bash
   ```

2. **Manual deployment:**
   ```bash
   wget https://github.com/quantumrerank/releases/latest/download/docker-compose.prod.yml
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Verify deployment:**
   ```bash
   curl http://localhost/health
   ```

## Configuration

Edit `config/production.yaml` to customize settings:
- API key
- Log level  
- Performance settings
- Redis connection

## Usage

```bash
curl -X POST http://localhost/v1/rerank \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "your search query",
    "documents": ["document 1", "document 2"],
    "method": "hybrid"
  }'
```

## Maintenance

- **View logs:** `docker-compose logs -f`
- **Restart:** `docker-compose restart`
- **Update:** `docker-compose pull && docker-compose up -d`
- **Stop:** `docker-compose down`
```

## Success Criteria
- [ ] Single `docker run` command starts the service
- [ ] `docker-compose up` deploys complete production stack
- [ ] Health checks work correctly
- [ ] All dependencies included in image
- [ ] Production configuration works out of the box
- [ ] Simple deployment documentation
- [ ] Service starts under 2 minutes

## Timeline
- **Week 1**: Production Dockerfile and base image
- **Week 2**: Docker-compose production stack
- **Week 3**: Quick start scripts and documentation
- **Week 4**: Testing and optimization

This creates a production-ready Docker deployment that companies can use immediately without complexity.
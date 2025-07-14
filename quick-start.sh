#!/bin/bash
# QuantumRerank Quick Start Script
# One-command deployment for production use

set -e

echo "🚀 Starting QuantumRerank..."
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed"
    echo "   Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Error: docker-compose is not installed"
    echo "   Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p config ssl logs

# Generate secure API key if not exists
if [ ! -f .env ]; then
    echo "🔑 Generating secure API key..."
    API_KEY="qr_$(openssl rand -hex 16)"
    cat > .env << EOF
# QuantumRerank Environment Variables
API_KEY=${API_KEY}
LOG_LEVEL=INFO
REDIS_URL=redis://redis:6379
EOF
    echo "   API key saved to .env file"
else
    echo "✅ Using existing .env configuration"
fi

# Create simple production config if not exists
if [ ! -f config/production.simple.yaml ]; then
    echo "⚙️  Creating default configuration..."
    cat > config/production.simple.yaml << 'EOF'
quantum_rerank:
  api:
    host: "0.0.0.0"
    port: 8000
    workers: 4
  quantum:
    method: "hybrid"
    cache_enabled: true
  performance:
    timeout_seconds: 30
    max_request_size: "10MB"
  logging:
    level: "INFO"
  auth:
    api_key: "${QUANTUM_RERANK_API_KEY:-qr-demo-key-change-this}"
EOF
fi

# Start services
echo "🐳 Starting Docker services..."
echo "   This may take a few minutes on first run..."

# Use simple compose file for quick deployment
docker-compose -f docker-compose.simple.yml up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to start..."
sleep 15

# Check health with retries
health_retries=0
max_retries=12
while [ $health_retries -lt $max_retries ]; do
    if curl -f -s http://localhost:8000/health &> /dev/null; then
        echo ""
        echo "✅ QuantumRerank is running successfully!"
        echo ""
        echo "📡 API Endpoint: http://localhost:8000"
        echo "🏥 Health Check: http://localhost:8000/health"
        echo "📚 API Docs: http://localhost:8000/docs"
        echo ""
        
        # Get API key from .env file
        source .env
        echo "🔑 Your API Key: ${API_KEY}"
        echo ""
        echo "🧪 Test the API:"
        echo "curl -X POST http://localhost:8000/v1/rerank \\"
        echo "  -H 'Authorization: Bearer ${API_KEY}' \\"
        echo "  -H 'Content-Type: application/json' \\"
        echo "  -d '{\"query\": \"quantum computing\", \"candidates\": [\"Quantum mechanics\", \"Classical computing\"], \"method\": \"hybrid\"}'"
        echo ""
        echo "🛠️  Management Commands:"
        echo "   View logs: docker-compose -f docker-compose.simple.yml logs -f"
        echo "   Restart: docker-compose -f docker-compose.simple.yml restart"
        echo "   Stop: docker-compose -f docker-compose.simple.yml down"
        echo "   Update: docker-compose -f docker-compose.simple.yml pull && docker-compose -f docker-compose.simple.yml up -d"
        echo ""
        
        exit 0
    fi
    
    health_retries=$((health_retries + 1))
    echo "   Health check ${health_retries}/${max_retries}..."
    sleep 10
done

echo ""
echo "❌ Service failed to start properly"
echo "   Check logs: docker-compose -f docker-compose.simple.yml logs"
echo "   Check status: docker-compose -f docker-compose.simple.yml ps"
exit 1
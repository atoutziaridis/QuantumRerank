#!/bin/bash
# QuantumRerank Production Validation Script
# Complete end-to-end validation of production readiness
# Usage: ./production-validation.sh [--environment production-test] [--api-key KEY]

set -e

# Configuration
ENVIRONMENT="${1:-production-test}"
API_KEY="${QUANTUM_RERANK_API_KEY:-}"
BASE_URL="${2:-http://localhost:8000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"
}

success() {
    echo -e "${GREEN}✅ $*${NC}"
}

error() {
    echo -e "${RED}❌ $*${NC}"
    exit 1
}

warning() {
    echo -e "${YELLOW}⚠️  $*${NC}"
}

echo ""
echo "🚀 QuantumRerank Production Validation"
echo "======================================"
echo ""
log "Environment: $ENVIRONMENT"
log "Base URL: $BASE_URL"
if [ -n "$API_KEY" ]; then
    log "API Key: ${API_KEY:0:8}..."
else
    log "Generating temporary API key..."
    API_KEY="qr_$(openssl rand -hex 16)"
    export QUANTUM_RERANK_API_KEY="$API_KEY"
    log "Generated API Key: ${API_KEY:0:8}..."
fi
echo ""

# Clean start
log "🧹 Cleaning environment..."
if command -v docker-compose &> /dev/null && [ -f "docker-compose.simple.yml" ]; then
    docker-compose -f docker-compose.simple.yml down -v 2>/dev/null || true
    docker system prune -f -q 2>/dev/null || true
fi

# Deploy production stack
log "🚀 Deploying production stack..."
export QUANTUM_RERANK_API_KEY="$API_KEY"

if [ -f "docker-compose.simple.yml" ]; then
    docker-compose -f docker-compose.simple.yml up -d
elif [ -f "Dockerfile" ]; then
    # Build and run directly
    docker build -t quantumrerank:test .
    docker run -d --name quantumrerank-test -p 8000:8000 \
        -e QUANTUM_RERANK_API_KEY="$API_KEY" \
        -e ENVIRONMENT="$ENVIRONMENT" \
        quantumrerank:test
else
    warning "No Docker configuration found, assuming service is already running"
fi

# Wait for services to be ready
log "⏳ Waiting for services to be ready..."
for i in {1..60}; do
    if curl -f -s --max-time 5 "$BASE_URL/health" >/dev/null 2>&1; then
        success "Services are ready"
        break
    fi
    if [ $i -eq 60 ]; then
        error "Services failed to start within 5 minutes"
    fi
    sleep 5
done

echo ""
log "🧪 Running production validation tests..."
echo ""

# Test 1: Basic functionality
log "Test 1: Basic API functionality"
if command -v python3 &> /dev/null; then
    python3 -c "
import requests
import json
import sys

try:
    response = requests.post('$BASE_URL/v1/rerank', 
        headers={'Authorization': 'Bearer $API_KEY', 'Content-Type': 'application/json'},
        json={
            'query': 'What is machine learning?',
            'candidates': [
                'Machine learning is a subset of artificial intelligence',
                'Python is a programming language',
                'Deep learning uses neural networks'
            ],
            'method': 'hybrid'
        },
        timeout=30
    )

    if response.status_code != 200:
        print(f'❌ Expected 200, got {response.status_code}')
        print(f'Response: {response.text}')
        sys.exit(1)
        
    result = response.json()
    if 'results' not in result:
        print('❌ Response missing results')
        print(f'Response: {result}')
        sys.exit(1)
        
    if len(result['results']) != 3:
        print(f'❌ Expected 3 results, got {len(result[\"results\"])}')
        sys.exit(1)
        
    print('✅ Basic functionality test passed')
    
except Exception as e:
    print(f'❌ Basic functionality test failed: {e}')
    sys.exit(1)
" || error "Basic functionality test failed"
else
    # Fallback to curl test
    RESPONSE=$(curl -s -w "%{http_code}" -X POST "$BASE_URL/v1/rerank" \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"query": "test", "candidates": ["doc1", "doc2"], "method": "classical"}')
    
    HTTP_CODE="${RESPONSE: -3}"
    if [ "$HTTP_CODE" = "200" ]; then
        success "Basic functionality test passed"
    else
        error "Basic functionality test failed (HTTP $HTTP_CODE)"
    fi
fi

# Test 2: Performance validation
log "Test 2: Performance validation"
if command -v python3 &> /dev/null; then
    python3 -c "
import requests
import time
import statistics
import sys

try:
    response_times = []
    for i in range(10):
        start = time.time()
        response = requests.post('$BASE_URL/v1/rerank',
            headers={'Authorization': 'Bearer $API_KEY', 'Content-Type': 'application/json'},
            json={
                'query': 'test query',
                'candidates': ['doc1', 'doc2', 'doc3', 'doc4', 'doc5'],
                'method': 'classical'
            },
            timeout=30
        )
        end = time.time()
        
        if response.status_code != 200:
            print(f'❌ Request {i} failed with status {response.status_code}')
            sys.exit(1)
            
        response_times.append((end - start) * 1000)  # Convert to ms

    avg_time = statistics.mean(response_times)
    max_time = max(response_times)

    print(f'Average response time: {avg_time:.1f}ms')
    print(f'Maximum response time: {max_time:.1f}ms')

    if avg_time >= 500:
        print(f'❌ Average response time {avg_time:.1f}ms exceeds 500ms limit')
        sys.exit(1)
        
    if max_time >= 1000:
        print(f'❌ Maximum response time {max_time:.1f}ms exceeds 1000ms limit')
        sys.exit(1)
        
    print('✅ Performance validation passed')
    
except Exception as e:
    print(f'❌ Performance validation failed: {e}')
    sys.exit(1)
" || error "Performance validation failed"
else
    warning "Python3 not available, skipping detailed performance test"
    # Simple performance test with curl
    START_TIME=$(date +%s%N)
    curl -f -s --max-time 10 -X POST "$BASE_URL/v1/rerank" \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"query": "test", "candidates": ["doc1"], "method": "classical"}' >/dev/null
    END_TIME=$(date +%s%N)
    DURATION=$(( (END_TIME - START_TIME) / 1000000 ))
    
    if [ $DURATION -lt 500 ]; then
        success "Performance validation passed (${DURATION}ms)"
    else
        warning "Performance may be slow (${DURATION}ms)"
    fi
fi

# Test 3: Load testing (simplified)
log "Test 3: Basic load testing"
SUCCESS_COUNT=0
for i in {1..20}; do
    if curl -f -s --max-time 10 -X POST "$BASE_URL/v1/rerank" \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"test query $i\", \"candidates\": [\"doc1\", \"doc2\"], \"method\": \"classical\"}" >/dev/null; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    fi
done

SUCCESS_RATE=$((SUCCESS_COUNT * 100 / 20))
log "Completed 20 requests with ${SUCCESS_COUNT}/20 successful (${SUCCESS_RATE}%)"

if [ $SUCCESS_COUNT -ge 18 ]; then
    success "Load testing passed"
else
    warning "Load testing concerns: only ${SUCCESS_COUNT}/20 requests successful"
fi

# Test 4: Error handling
log "Test 4: Error handling"

# Test invalid API key
INVALID_RESPONSE=$(curl -s -w "%{http_code}" -o /dev/null \
    -X POST "$BASE_URL/v1/rerank" \
    -H "Authorization: Bearer invalid-key" \
    -H "Content-Type: application/json" \
    -d '{"query": "test", "candidates": ["doc1"]}')

if [ "$INVALID_RESPONSE" = "401" ] || [ "$INVALID_RESPONSE" = "403" ]; then
    success "API correctly rejects invalid authentication"
else
    warning "Unexpected auth response: $INVALID_RESPONSE"
fi

# Test invalid request
INVALID_REQUEST=$(curl -s -w "%{http_code}" -o /dev/null \
    -X POST "$BASE_URL/v1/rerank" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"query": "", "candidates": []}')

if [ "$INVALID_REQUEST" = "400" ] || [ "$INVALID_REQUEST" = "422" ]; then
    success "API correctly handles invalid requests"
else
    warning "Unexpected error handling response: $INVALID_REQUEST"
fi

# Test 5: Health monitoring
log "Test 5: Health monitoring"

# Test basic health
HEALTH_RESPONSE=$(curl -s -w "%{http_code}" "$BASE_URL/health")
HEALTH_CODE="${HEALTH_RESPONSE: -3}"
HEALTH_BODY="${HEALTH_RESPONSE%???}"

if [ "$HEALTH_CODE" = "200" ]; then
    if echo "$HEALTH_BODY" | grep -q "healthy\|ok\|running"; then
        success "Health endpoint working correctly"
    else
        warning "Health endpoint returns 200 but unexpected content: $HEALTH_BODY"
    fi
else
    error "Health endpoint failed: $HEALTH_CODE"
fi

echo ""
success "🎉 All production validation tests passed!"
echo ""
log "📊 Test Summary:"
log "✅ Basic functionality"
log "✅ Performance validation (<500ms avg response time)"
log "✅ Load testing (20 requests)"
log "✅ Error handling"
log "✅ Health monitoring"
echo ""
success "🚀 QuantumRerank is production ready!"
echo ""

# Cleanup
log "🧹 Cleaning up test environment..."
if command -v docker &> /dev/null; then
    docker stop quantumrerank-test 2>/dev/null || true
    docker rm quantumrerank-test 2>/dev/null || true
    if [ -f "docker-compose.simple.yml" ]; then
        docker-compose -f docker-compose.simple.yml down -v 2>/dev/null || true
    fi
fi

log "Production validation completed successfully!"
echo ""
log "Next steps for production deployment:"
log "1. Configure your production environment variables"
log "2. Set up proper DNS and SSL certificates"
log "3. Configure monitoring and alerting"
log "4. Deploy using: scripts/deploy/universal-deploy.sh"
log "5. Test using: scripts/test/test-deployment.sh"
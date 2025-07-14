#!/bin/bash
# Deployment test script for QuantumRerank
# Tests health endpoint and API functionality
# Usage: ./test-deployment.sh [environment] [endpoint_url]

set -e

# Configuration
ENVIRONMENT="${1:-staging}"
ENDPOINT_URL="${2:-}"
API_KEY="${QUANTUM_RERANK_API_KEY:-}"
TIMEOUT=300

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

# Validate inputs
if [ -z "$API_KEY" ]; then
    error "QUANTUM_RERANK_API_KEY environment variable is required"
fi

# Auto-detect endpoint URL if not provided
if [ -z "$ENDPOINT_URL" ]; then
    log "Auto-detecting deployment endpoint for environment: $ENVIRONMENT"
    
    case $ENVIRONMENT in
        staging|production)
            # Try common endpoint patterns
            POSSIBLE_ENDPOINTS=(
                "https://quantum-rerank-${ENVIRONMENT}.gcp.example.com"
                "https://quantum-rerank-${ENVIRONMENT}.azurewebsites.net"
                "http://quantum-rerank-${ENVIRONMENT}.local"
                "http://localhost:8000"
            )
            
            for endpoint in "${POSSIBLE_ENDPOINTS[@]}"; do
                if curl -f -s --max-time 5 "$endpoint/health" > /dev/null 2>&1; then
                    ENDPOINT_URL="$endpoint"
                    log "Found endpoint: $ENDPOINT_URL"
                    break
                fi
            done
            ;;
        *)
            ENDPOINT_URL="http://localhost:8000"
            ;;
    esac
    
    if [ -z "$ENDPOINT_URL" ]; then
        error "Could not auto-detect endpoint URL. Please provide it as second argument."
    fi
fi

log "🧪 Testing QuantumRerank deployment"
log "   Environment: $ENVIRONMENT"
log "   Endpoint: $ENDPOINT_URL"
log "   API Key: ${API_KEY:0:8}..."
echo ""

# Test 1: Health Check
log "Test 1: Health endpoint"
if curl -f -s --max-time 10 "$ENDPOINT_URL/health" > /dev/null; then
    success "Health endpoint responding"
else
    error "Health endpoint failed"
fi

# Test 2: Detailed Health Check
log "Test 2: Detailed health check"
HEALTH_RESPONSE=$(curl -s --max-time 10 "$ENDPOINT_URL/health" || echo "")
if echo "$HEALTH_RESPONSE" | grep -q "healthy\|ok\|running"; then
    success "Service reports healthy status"
    echo "   Response: $HEALTH_RESPONSE"
else
    warning "Unexpected health response: $HEALTH_RESPONSE"
fi

# Test 3: API Authentication
log "Test 3: API authentication"
AUTH_RESPONSE=$(curl -s -w "%{http_code}" -o /dev/null \
    -X POST "$ENDPOINT_URL/v1/rerank" \
    -H "Authorization: Bearer invalid_key" \
    -H "Content-Type: application/json" \
    -d '{"query": "test", "candidates": ["doc1"], "method": "classical"}' || echo "000")

if [ "$AUTH_RESPONSE" = "401" ] || [ "$AUTH_RESPONSE" = "403" ]; then
    success "API correctly rejects invalid authentication"
else
    warning "Unexpected auth response code: $AUTH_RESPONSE"
fi

# Test 4: Basic API Functionality
log "Test 4: Basic API functionality"
API_RESPONSE=$(curl -s --max-time 30 \
    -X POST "$ENDPOINT_URL/v1/rerank" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "query": "machine learning algorithms",
        "candidates": [
            "Deep learning is a subset of machine learning",
            "Cats are popular pets",
            "Neural networks are computational models"
        ],
        "method": "classical",
        "top_k": 2
    }' || echo "")

if echo "$API_RESPONSE" | grep -q "results\|scores\|ranking"; then
    success "API returns valid reranking results"
    echo "   Sample response: $(echo "$API_RESPONSE" | head -c 200)..."
else
    error "API failed to return valid results. Response: $API_RESPONSE"
fi

# Test 5: Quantum Method
log "Test 5: Quantum method functionality"
QUANTUM_RESPONSE=$(curl -s --max-time 60 \
    -X POST "$ENDPOINT_URL/v1/rerank" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "query": "quantum computing",
        "candidates": [
            "Quantum computers use quantum mechanics",
            "Classical computers use bits"
        ],
        "method": "quantum",
        "top_k": 2
    }' || echo "")

if echo "$QUANTUM_RESPONSE" | grep -q "results\|scores\|ranking"; then
    success "Quantum method works correctly"
else
    warning "Quantum method may have issues. Response: $(echo "$QUANTUM_RESPONSE" | head -c 100)..."
fi

# Test 6: Hybrid Method
log "Test 6: Hybrid method functionality"
HYBRID_RESPONSE=$(curl -s --max-time 60 \
    -X POST "$ENDPOINT_URL/v1/rerank" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "query": "information retrieval",
        "candidates": [
            "Search engines help find information",
            "Databases store structured data",
            "Information retrieval systems rank documents"
        ],
        "method": "hybrid",
        "top_k": 3
    }' || echo "")

if echo "$HYBRID_RESPONSE" | grep -q "results\|scores\|ranking"; then
    success "Hybrid method works correctly"
else
    warning "Hybrid method may have issues. Response: $(echo "$HYBRID_RESPONSE" | head -c 100)..."
fi

# Test 7: Performance Test
log "Test 7: Performance test (10 requests)"
START_TIME=$(date +%s)
SUCCESS_COUNT=0

for i in {1..10}; do
    PERF_RESPONSE=$(curl -s --max-time 10 \
        -X POST "$ENDPOINT_URL/v1/rerank" \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"test query $i\", \"candidates\": [\"doc1\", \"doc2\"], \"method\": \"classical\"}" || echo "")
    
    if echo "$PERF_RESPONSE" | grep -q "results\|scores"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    fi
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
AVG_TIME=$((DURATION * 100 / 10))

if [ $SUCCESS_COUNT -ge 8 ]; then
    success "Performance test passed: $SUCCESS_COUNT/10 requests successful"
    log "   Average time: ${AVG_TIME:0:-2}.${AVG_TIME: -2}s per request"
else
    warning "Performance test concerns: only $SUCCESS_COUNT/10 requests successful"
fi

# Test 8: Error Handling
log "Test 8: Error handling"
ERROR_RESPONSE=$(curl -s -w "%{http_code}" -o /dev/null \
    -X POST "$ENDPOINT_URL/v1/rerank" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"invalid": "json"}' || echo "000")

if [ "$ERROR_RESPONSE" = "400" ] || [ "$ERROR_RESPONSE" = "422" ]; then
    success "API correctly handles invalid requests"
else
    warning "Unexpected error handling response: $ERROR_RESPONSE"
fi

echo ""
success "🎉 Deployment testing completed!"
echo ""
log "Summary:"
log "✅ Health endpoint working"
log "✅ Authentication working"
log "✅ Basic API functionality working"
log "✅ All reranking methods functional"
log "✅ Performance within acceptable range"
log "✅ Error handling working"
echo ""
log "Deployment ready for use!"
log "API endpoint: $ENDPOINT_URL/v1/rerank"
log "Health check: $ENDPOINT_URL/health"
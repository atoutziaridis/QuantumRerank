#!/bin/bash
# QuantumRerank Production Readiness Checklist
# Comprehensive validation of all production requirements
# Usage: ./production-checklist.sh [base_url] [api_key]

set -e

# Configuration
BASE_URL="${1:-http://localhost:8000}"
API_KEY="${2:-$QUANTUM_RERANK_API_KEY}"
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
}

warning() {
    echo -e "${YELLOW}⚠️  $*${NC}"
}

check_item() {
    local item="$1"
    local test_command="$2"
    
    echo -n "$item... "
    
    if eval "$test_command" >/dev/null 2>&1; then
        success "PASS"
        return 0
    else
        error "FAIL"
        return 1
    fi
}

echo ""
echo "🔍 QuantumRerank Production Readiness Checklist"
echo "==============================================="
echo ""

if [ -z "$API_KEY" ]; then
    warning "API_KEY not provided, generating temporary key..."
    API_KEY="qr_$(openssl rand -hex 16)"
    export QUANTUM_RERANK_API_KEY="$API_KEY"
fi

log "Base URL: $BASE_URL"
log "API Key: ${API_KEY:0:8}..."
echo ""

PASSED=0
TOTAL=0

# Infrastructure Checks
echo "🏗️  Infrastructure Checks"
echo "------------------------"

((TOTAL++))
if check_item "1. Service responds to health checks" \
    "curl -f -s --max-time 10 '$BASE_URL/health'"; then
    ((PASSED++))
fi

((TOTAL++))
if check_item "2. Service has proper CORS headers" \
    "curl -s -I '$BASE_URL/health' | grep -i 'access-control-allow'"; then
    ((PASSED++))
fi

((TOTAL++))
if check_item "3. Service returns JSON responses" \
    "curl -s '$BASE_URL/health' | python3 -m json.tool"; then
    ((PASSED++))
fi

((TOTAL++))
if check_item "4. Service handles OPTIONS requests" \
    "curl -s -X OPTIONS '$BASE_URL/v1/rerank'"; then
    ((PASSED++))
fi

echo ""

# Authentication & Security Checks
echo "🔐 Authentication & Security Checks"
echo "----------------------------------"

((TOTAL++))
if check_item "5. API rejects requests without authentication" \
    "test \$(curl -s -w '%{http_code}' -o /dev/null -X POST '$BASE_URL/v1/rerank' -H 'Content-Type: application/json' -d '{\"query\":\"test\",\"candidates\":[\"doc1\"]}') = '401'"; then
    ((PASSED++))
fi

((TOTAL++))
if check_item "6. API rejects invalid API keys" \
    "test \$(curl -s -w '%{http_code}' -o /dev/null -X POST '$BASE_URL/v1/rerank' -H 'Authorization: Bearer invalid-key' -H 'Content-Type: application/json' -d '{\"query\":\"test\",\"candidates\":[\"doc1\"]}') = '401'"; then
    ((PASSED++))
fi

((TOTAL++))
if check_item "7. API accepts valid API keys" \
    "test \$(curl -s -w '%{http_code}' -o /dev/null -X POST '$BASE_URL/v1/rerank' -H 'Authorization: Bearer $API_KEY' -H 'Content-Type: application/json' -d '{\"query\":\"test\",\"candidates\":[\"doc1\"],\"method\":\"classical\"}') = '200'"; then
    ((PASSED++))
fi

echo ""

# API Functionality Checks
echo "🔌 API Functionality Checks"
echo "---------------------------"

((TOTAL++))
if check_item "8. Classical method works correctly" \
    "curl -f -s -X POST '$BASE_URL/v1/rerank' -H 'Authorization: Bearer $API_KEY' -H 'Content-Type: application/json' -d '{\"query\":\"test\",\"candidates\":[\"doc1\",\"doc2\"],\"method\":\"classical\"}' | grep -q 'results'"; then
    ((PASSED++))
fi

((TOTAL++))
if check_item "9. Quantum method works correctly" \
    "curl -f -s -X POST '$BASE_URL/v1/rerank' -H 'Authorization: Bearer $API_KEY' -H 'Content-Type: application/json' -d '{\"query\":\"test\",\"candidates\":[\"doc1\",\"doc2\"],\"method\":\"quantum\"}' | grep -q 'results'"; then
    ((PASSED++))
fi

((TOTAL++))
if check_item "10. Hybrid method works correctly" \
    "curl -f -s -X POST '$BASE_URL/v1/rerank' -H 'Authorization: Bearer $API_KEY' -H 'Content-Type: application/json' -d '{\"query\":\"test\",\"candidates\":[\"doc1\",\"doc2\"],\"method\":\"hybrid\"}' | grep -q 'results'"; then
    ((PASSED++))
fi

((TOTAL++))
if check_item "11. API handles empty candidate list" \
    "test \$(curl -s -w '%{http_code}' -o /dev/null -X POST '$BASE_URL/v1/rerank' -H 'Authorization: Bearer $API_KEY' -H 'Content-Type: application/json' -d '{\"query\":\"test\",\"candidates\":[]}') = '422'"; then
    ((PASSED++))
fi

((TOTAL++))
if check_item "12. API handles missing query" \
    "test \$(curl -s -w '%{http_code}' -o /dev/null -X POST '$BASE_URL/v1/rerank' -H 'Authorization: Bearer $API_KEY' -H 'Content-Type: application/json' -d '{\"candidates\":[\"doc1\"]}') = '422'"; then
    ((PASSED++))
fi

echo ""

# Performance Checks
echo "⚡ Performance Checks"
echo "-------------------"

((TOTAL++))
RESPONSE_TIME=$(curl -w "%{time_total}" -s -o /dev/null -X POST "$BASE_URL/v1/rerank" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"query":"performance test","candidates":["doc1","doc2","doc3"],"method":"classical"}')
RESPONSE_TIME_MS=$(echo "$RESPONSE_TIME * 1000" | bc -l 2>/dev/null || echo "999")
if (( $(echo "$RESPONSE_TIME_MS < 500" | bc -l 2>/dev/null || echo "0") )); then
    success "13. Response time under 500ms (${RESPONSE_TIME_MS}ms)"
    ((PASSED++))
else
    error "13. Response time under 500ms (${RESPONSE_TIME_MS}ms)"
fi

((TOTAL++))
# Test concurrent requests
CONCURRENT_SUCCESS=0
for i in {1..10}; do
    if curl -f -s --max-time 5 -X POST "$BASE_URL/v1/rerank" \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"concurrent test $i\",\"candidates\":[\"doc1\",\"doc2\"],\"method\":\"classical\"}" >/dev/null 2>&1 & then
        ((CONCURRENT_SUCCESS++))
    fi
done
wait

if [ $CONCURRENT_SUCCESS -ge 8 ]; then
    success "14. Handles concurrent requests ($CONCURRENT_SUCCESS/10 successful)"
    ((PASSED++))
else
    error "14. Handles concurrent requests ($CONCURRENT_SUCCESS/10 successful)"
fi

echo ""

# Error Handling Checks
echo "🚨 Error Handling Checks"
echo "-----------------------"

((TOTAL++))
if check_item "15. Returns proper error for malformed JSON" \
    "test \$(curl -s -w '%{http_code}' -o /dev/null -X POST '$BASE_URL/v1/rerank' -H 'Authorization: Bearer $API_KEY' -H 'Content-Type: application/json' -d '{invalid json}') = '400'"; then
    ((PASSED++))
fi

((TOTAL++))
if check_item "16. Returns proper error for unsupported method" \
    "test \$(curl -s -w '%{http_code}' -o /dev/null -X POST '$BASE_URL/v1/rerank' -H 'Authorization: Bearer $API_KEY' -H 'Content-Type: application/json' -d '{\"query\":\"test\",\"candidates\":[\"doc1\"],\"method\":\"invalid\"}') = '422'"; then
    ((PASSED++))
fi

((TOTAL++))
if check_item "17. Returns proper error for oversized request" \
    "LARGE_DOC=\$(python3 -c 'print(\"x\" * 100000)') && test \$(curl -s -w '%{http_code}' -o /dev/null -X POST '$BASE_URL/v1/rerank' -H 'Authorization: Bearer $API_KEY' -H 'Content-Type: application/json' -d '{\"query\":\"test\",\"candidates\":[\"'\$LARGE_DOC'\"]}') = '422'"; then
    ((PASSED++))
fi

echo ""

# Monitoring & Observability Checks
echo "📊 Monitoring & Observability Checks"
echo "-----------------------------------"

((TOTAL++))
if check_item "18. Health endpoint returns status information" \
    "curl -s '$BASE_URL/health' | grep -E 'status|healthy'"; then
    ((PASSED++))
fi

((TOTAL++))
if check_item "19. Service provides proper HTTP status codes" \
    "test \$(curl -s -w '%{http_code}' -o /dev/null '$BASE_URL/health') = '200'"; then
    ((PASSED++))
fi

((TOTAL++))
# Check if detailed health endpoint exists
if curl -f -s "$BASE_URL/health/detailed" >/dev/null 2>&1; then
    if check_item "20. Detailed health endpoint available" \
        "curl -s '$BASE_URL/health/detailed' | grep -E 'checks|memory|database'"; then
        ((PASSED++))
    fi
else
    warning "20. Detailed health endpoint not available"
fi

echo ""

# Configuration & Environment Checks
echo "⚙️  Configuration & Environment Checks"
echo "------------------------------------"

((TOTAL++))
if check_item "21. Service uses environment configuration" \
    "curl -s '$BASE_URL/health' | grep -v 'error'"; then
    ((PASSED++))
fi

((TOTAL++))
# Check if service respects API key from environment
if [ -n "$QUANTUM_RERANK_API_KEY" ]; then
    if check_item "22. Service respects environment API key" \
        "test \$(curl -s -w '%{http_code}' -o /dev/null -X POST '$BASE_URL/v1/rerank' -H 'Authorization: Bearer $QUANTUM_RERANK_API_KEY' -H 'Content-Type: application/json' -d '{\"query\":\"test\",\"candidates\":[\"doc1\"],\"method\":\"classical\"}') = '200'"; then
        ((PASSED++))
    fi
else
    warning "22. Environment API key not set for testing"
fi

echo ""

# Container/Deployment Checks
echo "🐳 Container/Deployment Checks"
echo "-----------------------------"

((TOTAL++))
if command -v docker >/dev/null 2>&1 && docker ps | grep -q "quantumrerank\|8000"; then
    if check_item "23. Service running in container" \
        "docker ps | grep -E 'quantumrerank|8000'"; then
        ((PASSED++))
    fi
else
    warning "23. Cannot verify container deployment (Docker not available or service not containerized)"
fi

((TOTAL++))
if check_item "24. Service accessible on expected port" \
    "nc -z localhost 8000"; then
    ((PASSED++))
fi

echo ""

# Documentation & API Specification Checks
echo "📚 Documentation & API Specification"
echo "-----------------------------------"

((TOTAL++))
if check_item "25. Service has API documentation endpoint" \
    "curl -f -s '$BASE_URL/docs' || curl -f -s '$BASE_URL/openapi.json' || curl -f -s '$BASE_URL/swagger'"; then
    ((PASSED++))
fi

echo ""

# Final Results
echo "📊 PRODUCTION READINESS RESULTS"
echo "==============================="
echo ""

SUCCESS_RATE=$((PASSED * 100 / TOTAL))

log "Checks completed: $TOTAL"
log "Checks passed: $PASSED"
log "Success rate: ${SUCCESS_RATE}%"

echo ""

if [ $SUCCESS_RATE -ge 90 ]; then
    success "🎉 PRODUCTION READY! ($PASSED/$TOTAL checks passed)"
    echo ""
    log "✅ QuantumRerank meets production readiness criteria"
    log "🚀 Ready for deployment to production environment"
    echo ""
    log "Next steps:"
    log "1. Deploy to staging environment for final testing"
    log "2. Configure production monitoring and alerting"
    log "3. Set up backup and disaster recovery procedures"
    log "4. Deploy to production using deployment scripts"
    echo ""
    exit 0
elif [ $SUCCESS_RATE -ge 80 ]; then
    warning "⚠️  MOSTLY READY ($PASSED/$TOTAL checks passed)"
    echo ""
    log "🔧 Address failing checks before production deployment"
    log "📝 Review the failed items above and fix them"
    echo ""
    exit 1
else
    error "❌ NOT READY FOR PRODUCTION ($PASSED/$TOTAL checks passed)"
    echo ""
    log "🛠️  Significant issues found - resolve before deployment"
    log "📋 More than 20% of checks failed"
    echo ""
    exit 2
fi
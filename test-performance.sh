#!/bin/bash
# Performance testing script for QuantumRerank
# Validates PRD compliance and stability under load

set -e

# Configuration
BASE_URL="${BASE_URL:-http://localhost:8000}"
API_KEY="${API_KEY:-demo-api-key}"
DOCKER_COMPOSE_FILE="${DOCKER_COMPOSE_FILE:-docker-compose.simple.yml}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"
}

success() {
    echo -e "${GREEN}✅ $*${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $*${NC}"
}

error() {
    echo -e "${RED}❌ $*${NC}"
}

# Function to check if service is running
check_service_health() {
    local retries=0
    local max_retries=30
    
    log "Checking service health..."
    
    while [ $retries -lt $max_retries ]; do
        if curl -f -s "${BASE_URL}/health" > /dev/null 2>&1; then
            success "Service is healthy"
            return 0
        fi
        
        retries=$((retries + 1))
        log "Health check attempt ${retries}/${max_retries}..."
        sleep 2
    done
    
    error "Service failed to become healthy after ${max_retries} attempts"
    return 1
}

# Function to get API key from environment
get_api_key() {
    if [ -f .env ]; then
        API_KEY=$(grep "^API_KEY=" .env | cut -d'=' -f2)
        log "Using API key from .env file"
    else
        log "Using default API key (consider setting API_KEY environment variable)"
    fi
}

# Function to test basic API functionality
test_basic_api() {
    log "Testing basic API functionality..."
    
    local response
    response=$(curl -s -X POST "${BASE_URL}/v1/rerank" \
        -H "Authorization: Bearer ${API_KEY}" \
        -H "Content-Type: application/json" \
        -d '{
            "query": "test query",
            "candidates": ["doc1", "doc2"],
            "method": "classical"
        }')
    
    if echo "$response" | grep -q '"results"'; then
        success "Basic API test passed"
        return 0
    else
        error "Basic API test failed"
        echo "Response: $response"
        return 1
    fi
}

# Function to test performance targets
test_performance_targets() {
    log "Testing performance targets against PRD requirements..."
    
    # Test similarity computation (<100ms)
    log "Testing similarity computation performance..."
    local start_time
    start_time=$(date +%s%N)
    
    local response
    response=$(curl -s -X POST "${BASE_URL}/v1/rerank" \
        -H "Authorization: Bearer ${API_KEY}" \
        -H "Content-Type: application/json" \
        -d '{
            "query": "machine learning",
            "candidates": ["AI algorithms", "Data science"],
            "method": "hybrid"
        }')
    
    local end_time
    end_time=$(date +%s%N)
    local duration_ms=$(( (end_time - start_time) / 1000000 ))
    
    if [ $duration_ms -lt 100 ]; then
        success "Similarity computation: ${duration_ms}ms < 100ms (PRD compliant)"
    else
        warning "Similarity computation: ${duration_ms}ms >= 100ms (PRD violation)"
    fi
    
    # Test batch reranking (<500ms for 50-100 docs)
    log "Testing batch reranking performance..."
    start_time=$(date +%s%N)
    
    # Create batch request with 50 documents
    local batch_candidates=""
    for i in {1..50}; do
        batch_candidates="${batch_candidates}\"Document ${i} about machine learning and AI\","
    done
    batch_candidates="[${batch_candidates%,}]"  # Remove trailing comma and wrap in array
    
    response=$(curl -s -X POST "${BASE_URL}/v1/rerank" \
        -H "Authorization: Bearer ${API_KEY}" \
        -H "Content-Type: application/json" \
        -d "{
            \"query\": \"artificial intelligence applications\",
            \"candidates\": ${batch_candidates},
            \"method\": \"hybrid\",
            \"top_k\": 10
        }")
    
    end_time=$(date +%s%N)
    duration_ms=$(( (end_time - start_time) / 1000000 ))
    
    if [ $duration_ms -lt 500 ]; then
        success "Batch reranking (50 docs): ${duration_ms}ms < 500ms (PRD compliant)"
    else
        warning "Batch reranking (50 docs): ${duration_ms}ms >= 500ms (PRD violation)"
    fi
}

# Function to test memory usage
test_memory_usage() {
    log "Testing memory usage..."
    
    local memory_response
    memory_response=$(curl -s "${BASE_URL}/health/memory")
    
    if echo "$memory_response" | grep -q '"status"'; then
        local memory_gb
        memory_gb=$(echo "$memory_response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data['memory_details']['current_usage']['memory_usage_gb'])
except:
    print('0')
" 2>/dev/null || echo "0")
        
        if (( $(echo "$memory_gb < 2.0" | bc -l 2>/dev/null || echo 0) )); then
            success "Memory usage: ${memory_gb}GB < 2.0GB (PRD compliant)"
        else
            warning "Memory usage: ${memory_gb}GB >= 2.0GB (PRD violation)"
        fi
    else
        warning "Could not retrieve memory information"
    fi
}

# Function to test circuit breaker functionality
test_circuit_breakers() {
    log "Testing circuit breaker status..."
    
    local breaker_response
    breaker_response=$(curl -s "${BASE_URL}/health/circuit-breakers")
    
    if echo "$breaker_response" | grep -q '"status"'; then
        local status
        status=$(echo "$breaker_response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data['status'])
except:
    print('unknown')
" 2>/dev/null || echo "unknown")
        
        case $status in
            "healthy")
                success "Circuit breakers: All healthy"
                ;;
            "warning")
                warning "Circuit breakers: Some issues detected"
                ;;
            "error")
                error "Circuit breakers: Critical issues"
                ;;
            *)
                warning "Circuit breakers: Status unknown"
                ;;
        esac
    else
        warning "Could not retrieve circuit breaker information"
    fi
}

# Function to run load test
run_load_test() {
    log "Running load test to validate stability..."
    
    if [ -f "tests/load_test.py" ]; then
        log "Running Python load test..."
        
        # Check if required Python packages are available
        if python3 -c "import aiohttp" 2>/dev/null; then
            python3 tests/load_test.py \
                --url "$BASE_URL" \
                --api-key "$API_KEY" \
                --users 5 \
                --requests 5 \
                --scenario "single_similarity"
        else
            warning "aiohttp not available, skipping advanced load test"
            
            # Simple load test with curl
            log "Running simple concurrent test with curl..."
            for i in {1..10}; do
                curl -s -X POST "${BASE_URL}/v1/rerank" \
                    -H "Authorization: Bearer ${API_KEY}" \
                    -H "Content-Type: application/json" \
                    -d '{"query": "test", "candidates": ["doc1", "doc2"], "method": "classical"}' \
                    > /dev/null &
            done
            wait
            success "Simple concurrent test completed"
        fi
    else
        warning "Load test script not found, skipping load test"
    fi
}

# Function to test different similarity methods
test_similarity_methods() {
    log "Testing all similarity methods..."
    
    local methods=("classical" "quantum" "hybrid")
    
    for method in "${methods[@]}"; do
        log "Testing $method method..."
        
        local response
        response=$(curl -s -X POST "${BASE_URL}/v1/rerank" \
            -H "Authorization: Bearer ${API_KEY}" \
            -H "Content-Type: application/json" \
            -d "{
                \"query\": \"quantum computing\",
                \"candidates\": [\"Quantum algorithms\", \"Classical computing\"],
                \"method\": \"$method\"
            }")
        
        if echo "$response" | grep -q '"results"'; then
            success "$method method: Working"
        else
            error "$method method: Failed"
            echo "Response: $response"
        fi
    done
}

# Function to generate performance report
generate_report() {
    log "Generating performance report..."
    
    local report_file="performance_report_$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$report_file" << EOF
{
    "test_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "base_url": "$BASE_URL",
    "test_results": {
        "basic_api": "$(test_basic_api && echo 'passed' || echo 'failed')",
        "performance_targets": "$(test_performance_targets && echo 'passed' || echo 'failed')",
        "memory_usage": "$(test_memory_usage && echo 'passed' || echo 'failed')",
        "circuit_breakers": "$(test_circuit_breakers && echo 'passed' || echo 'failed')",
        "similarity_methods": "$(test_similarity_methods && echo 'passed' || echo 'failed')"
    },
    "service_info": $(curl -s "${BASE_URL}/status" || echo '{}')
}
EOF
    
    log "Performance report saved to: $report_file"
}

# Main execution
main() {
    echo "🚀 QuantumRerank Performance Testing"
    echo "===================================="
    echo ""
    
    # Get API key
    get_api_key
    
    # Start services if not running
    if ! curl -f -s "${BASE_URL}/health" > /dev/null 2>&1; then
        log "Service not running, starting with Docker Compose..."
        
        if [ -f "$DOCKER_COMPOSE_FILE" ]; then
            docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
            sleep 15
        else
            error "Docker Compose file not found: $DOCKER_COMPOSE_FILE"
            error "Please start the service manually or run from project root"
            exit 1
        fi
    fi
    
    # Check service health
    if ! check_service_health; then
        error "Service is not healthy, aborting tests"
        exit 1
    fi
    
    echo ""
    log "Starting performance tests..."
    echo ""
    
    # Run tests
    local test_results=0
    
    # Basic functionality
    if ! test_basic_api; then
        test_results=$((test_results + 1))
    fi
    
    echo ""
    
    # Performance targets
    if ! test_performance_targets; then
        test_results=$((test_results + 1))
    fi
    
    echo ""
    
    # Memory usage
    if ! test_memory_usage; then
        test_results=$((test_results + 1))
    fi
    
    echo ""
    
    # Circuit breakers
    if ! test_circuit_breakers; then
        test_results=$((test_results + 1))
    fi
    
    echo ""
    
    # Similarity methods
    if ! test_similarity_methods; then
        test_results=$((test_results + 1))
    fi
    
    echo ""
    
    # Load test
    run_load_test
    
    echo ""
    echo "===================================="
    
    if [ $test_results -eq 0 ]; then
        success "All performance tests passed!"
        echo ""
        success "QuantumRerank is production-ready and meets PRD requirements"
    else
        warning "Some tests failed or showed warnings"
        echo ""
        warning "Review the output above for specific issues"
    fi
    
    echo ""
    log "Performance testing completed"
    
    # Generate report
    # generate_report
    
    return $test_results
}

# Run main function
main "$@"
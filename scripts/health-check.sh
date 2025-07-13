#!/bin/bash
# Health check script for QuantumRerank API
# Used by Docker HEALTHCHECK and Kubernetes probes

set -euo pipefail

# Configuration
HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
TIMEOUT="${HEALTH_CHECK_TIMEOUT:-10}"
RETRIES="${HEALTH_CHECK_RETRIES:-3}"
INTERVAL="${HEALTH_CHECK_INTERVAL:-2}"

# Health check endpoints
LIVENESS_ENDPOINT="/health"
READINESS_ENDPOINT="/health/ready"
DETAILED_ENDPOINT="/health/detailed"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

# Colored logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $*" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

# Check if curl is available
if ! command -v curl >/dev/null 2>&1; then
    log_error "curl is not available. Installing curl..."
    # Try to install curl (works in most container environments)
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update && apt-get install -y curl
    elif command -v apk >/dev/null 2>&1; then
        apk add --no-cache curl
    else
        log_error "Cannot install curl. Health check failed."
        exit 1
    fi
fi

# Function to perform HTTP health check
http_check() {
    local endpoint="$1"
    local description="$2"
    local required_status="${3:-200}"
    
    log_info "Checking $description ($endpoint)"
    
    # Perform the HTTP request
    local response
    local http_status
    local response_time
    
    response=$(curl -s -w "\n%{http_code}\n%{time_total}" \
        --max-time "$TIMEOUT" \
        --connect-timeout 5 \
        "http://$HOST:$PORT$endpoint" 2>/dev/null || echo -e "\nERROR\n0")
    
    # Parse response
    local body=$(echo "$response" | head -n -2)
    http_status=$(echo "$response" | tail -n 2 | head -n 1)
    response_time=$(echo "$response" | tail -n 1)
    
    # Check HTTP status
    if [[ "$http_status" == "ERROR" ]]; then
        log_error "$description: Connection failed"
        return 1
    elif [[ "$http_status" != "$required_status" ]]; then
        log_error "$description: HTTP $http_status (expected $required_status)"
        return 1
    fi
    
    # Parse response body if it's JSON
    if echo "$body" | jq . >/dev/null 2>&1; then
        local status=$(echo "$body" | jq -r '.status // "unknown"' 2>/dev/null)
        log_info "$description: Status=$status, Response time=${response_time}s"
        
        # Check application status
        if [[ "$status" == "healthy" || "$status" == "ready" ]]; then
            return 0
        else
            log_warn "$description: Application reports status '$status'"
            return 1
        fi
    else
        log_info "$description: HTTP $http_status, Response time=${response_time}s"
        return 0
    fi
}

# Function to check system resources
system_check() {
    log_info "Checking system resources"
    
    # Check memory usage
    if command -v free >/dev/null 2>&1; then
        local memory_usage
        memory_usage=$(free | grep '^Mem:' | awk '{printf "%.1f", ($3/$2) * 100.0}')
        log_info "Memory usage: ${memory_usage}%"
        
        # Warn if memory usage is high
        if (( $(echo "$memory_usage > 90" | bc -l) )); then
            log_warn "High memory usage: ${memory_usage}%"
        fi
    fi
    
    # Check disk space
    if command -v df >/dev/null 2>&1; then
        local disk_usage
        disk_usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
        log_info "Disk usage: ${disk_usage}%"
        
        # Warn if disk usage is high
        if [[ "$disk_usage" -gt 90 ]]; then
            log_warn "High disk usage: ${disk_usage}%"
        fi
    fi
    
    # Check load average
    if [[ -f /proc/loadavg ]]; then
        local load_avg
        load_avg=$(cat /proc/loadavg | cut -d' ' -f1)
        log_info "Load average: $load_avg"
    fi
}

# Function to perform comprehensive health check
comprehensive_check() {
    local failed_checks=0
    
    log_info "Starting comprehensive health check"
    
    # Basic liveness check
    if ! http_check "$LIVENESS_ENDPOINT" "Liveness check"; then
        failed_checks=$((failed_checks + 1))
    fi
    
    # Readiness check
    if ! http_check "$READINESS_ENDPOINT" "Readiness check"; then
        failed_checks=$((failed_checks + 1))
    fi
    
    # System resource check
    system_check
    
    # Check if API key authentication is working (if enabled)
    if [[ -n "${API_KEY:-}" ]]; then
        log_info "Testing API key authentication"
        if ! curl -s -f --max-time "$TIMEOUT" \
            -H "X-API-Key: $API_KEY" \
            "http://$HOST:$PORT/health" >/dev/null; then
            log_warn "API key authentication test failed"
        else
            log_info "API key authentication test passed"
        fi
    fi
    
    # Check metrics endpoint (if enabled)
    if curl -s -f --max-time 5 "http://$HOST:8001/metrics" >/dev/null 2>&1; then
        log_info "Metrics endpoint is accessible"
    else
        log_warn "Metrics endpoint is not accessible"
    fi
    
    return $failed_checks
}

# Function to perform basic health check with retries
basic_check() {
    local attempt=1
    
    while [[ $attempt -le $RETRIES ]]; do
        log_info "Health check attempt $attempt/$RETRIES"
        
        if http_check "$LIVENESS_ENDPOINT" "Basic health check"; then
            log_info "Health check passed"
            return 0
        fi
        
        if [[ $attempt -lt $RETRIES ]]; then
            log_warn "Health check failed, retrying in ${INTERVAL}s..."
            sleep "$INTERVAL"
        fi
        
        attempt=$((attempt + 1))
    done
    
    log_error "Health check failed after $RETRIES attempts"
    return 1
}

# Main function
main() {
    local check_type="${1:-basic}"
    
    case "$check_type" in
        "basic"|"liveness")
            basic_check
            ;;
        "readiness")
            http_check "$READINESS_ENDPOINT" "Readiness check"
            ;;
        "comprehensive"|"detailed")
            comprehensive_check
            ;;
        "metrics")
            http_check "/metrics" "Metrics endpoint" "200"
            ;;
        *)
            log_error "Unknown check type: $check_type"
            echo "Usage: $0 [basic|liveness|readiness|comprehensive|metrics]"
            exit 1
            ;;
    esac
}

# Run the health check
main "$@"
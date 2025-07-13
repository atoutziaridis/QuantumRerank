#!/bin/bash
# Production startup script for QuantumRerank API
# This script starts the application with production-optimized settings

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${ENVIRONMENT:-production}"
WORKERS="${WORKERS:-4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

# Error handling
error_exit() {
    log "ERROR: $1"
    exit "${2:-1}"
}

# Cleanup function
cleanup() {
    log "Received shutdown signal, performing cleanup..."
    if [[ -n "${APP_PID:-}" ]]; then
        log "Stopping application (PID: $APP_PID)"
        kill -TERM "$APP_PID" 2>/dev/null || true
        wait "$APP_PID" 2>/dev/null || true
    fi
    log "Cleanup completed"
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT SIGQUIT

# Change to application directory
cd "$APP_DIR" || error_exit "Cannot change to application directory: $APP_DIR"

# Environment validation
log "Starting QuantumRerank API in $ENVIRONMENT environment"
log "Application directory: $APP_DIR"
log "Workers: $WORKERS"
log "Host: $HOST"
log "Port: $PORT"

# Validate required files
required_files=(
    "quantum_rerank/__init__.py"
    "quantum_rerank/api/app.py"
)

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        error_exit "Required file not found: $file"
    fi
done

# Pre-startup checks
log "Performing pre-startup checks..."

# Check Python version
python_version=$(python --version 2>&1 | cut -d' ' -f2)
log "Python version: $python_version"

# Check available memory
if command -v free >/dev/null 2>&1; then
    memory_info=$(free -h | grep '^Mem:')
    log "Memory info: $memory_info"
fi

# Check disk space
disk_usage=$(df -h . | tail -1)
log "Disk usage: $disk_usage"

# Environment-specific configuration
case "$ENVIRONMENT" in
    "production")
        # Production settings
        WORKER_CLASS="uvicorn.workers.UvicornWorker"
        WORKER_CONNECTIONS="1000"
        MAX_REQUESTS="1000"
        MAX_REQUESTS_JITTER="100"
        TIMEOUT="30"
        KEEPALIVE="5"
        PRELOAD="--preload"
        ACCESS_LOG="--access-logfile=-"
        ERROR_LOG="--error-logfile=-"
        LOG_LEVEL="info"
        ;;
    "staging")
        # Staging settings
        WORKER_CLASS="uvicorn.workers.UvicornWorker"
        WORKER_CONNECTIONS="500"
        MAX_REQUESTS="500"
        MAX_REQUESTS_JITTER="50"
        TIMEOUT="30"
        KEEPALIVE="5"
        PRELOAD="--preload"
        ACCESS_LOG="--access-logfile=-"
        ERROR_LOG="--error-logfile=-"
        LOG_LEVEL="debug"
        ;;
    "development")
        # Development settings
        WORKER_CLASS="uvicorn.workers.UvicornWorker"
        WORKER_CONNECTIONS="100"
        MAX_REQUESTS="100"
        MAX_REQUESTS_JITTER="10"
        TIMEOUT="60"
        KEEPALIVE="2"
        PRELOAD=""
        ACCESS_LOG="--access-logfile=-"
        ERROR_LOG="--error-logfile=-"
        LOG_LEVEL="debug"
        ;;
    *)
        error_exit "Unknown environment: $ENVIRONMENT"
        ;;
esac

# Health check function
health_check() {
    local retries=0
    local max_retries=30
    local wait_time=2
    
    log "Waiting for application to become healthy..."
    
    while [[ $retries -lt $max_retries ]]; do
        if curl -f -s "http://$HOST:$PORT/health" >/dev/null 2>&1; then
            log "Application is healthy"
            return 0
        fi
        
        retries=$((retries + 1))
        log "Health check attempt $retries/$max_retries failed, waiting ${wait_time}s..."
        sleep $wait_time
    done
    
    error_exit "Application failed to become healthy after $max_retries attempts"
}

# Application startup
log "Starting application with Gunicorn..."

# Create necessary directories
mkdir -p logs cache data

# Set up log rotation if logrotate is available
if command -v logrotate >/dev/null 2>&1; then
    log "Setting up log rotation"
    cat > /tmp/quantum-rerank-logrotate.conf << EOF
logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 quantum quantum
    postrotate
        kill -USR1 \$(cat /tmp/gunicorn.pid) 2>/dev/null || true
    endscript
}
EOF
fi

# Start application with Gunicorn
exec gunicorn \
    --bind "$HOST:$PORT" \
    --workers "$WORKERS" \
    --worker-class "$WORKER_CLASS" \
    --worker-connections "$WORKER_CONNECTIONS" \
    --max-requests "$MAX_REQUESTS" \
    --max-requests-jitter "$MAX_REQUESTS_JITTER" \
    --timeout "$TIMEOUT" \
    --keepalive "$KEEPALIVE" \
    --pid /tmp/gunicorn.pid \
    $PRELOAD \
    $ACCESS_LOG \
    $ERROR_LOG \
    --log-level "$LOG_LEVEL" \
    --worker-tmp-dir /dev/shm \
    --backlog 2048 \
    "quantum_rerank.api.app:app" &

APP_PID=$!
log "Application started with PID: $APP_PID"

# Wait for the application process
wait "$APP_PID"
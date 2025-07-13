#!/bin/bash
# Rollback script for QuantumRerank deployments
# Supports rollback for Docker and Kubernetes deployments

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_DIR="$PROJECT_DIR/deployment"

# Default configuration
ENVIRONMENT="${ENVIRONMENT:-staging}"
DEPLOYMENT_TYPE="${DEPLOYMENT_TYPE:-kubernetes}"
NAMESPACE="${NAMESPACE:-quantum-rerank}"
ROLLBACK_REVISION="${ROLLBACK_REVISION:-}"
DRY_RUN="${DRY_RUN:-false}"
VERBOSE="${VERBOSE:-false}"
FORCE="${FORCE:-false}"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_debug() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $*" >&2
    fi
}

# Error handling
error_exit() {
    log_error "$1"
    exit "${2:-1}"
}

# Help function
show_help() {
    cat << EOF
QuantumRerank Rollback Script

Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV     Target environment (development, staging, production)
    -t, --type TYPE          Deployment type (docker, kubernetes)
    -n, --namespace NS       Kubernetes namespace
    -r, --revision REV       Specific revision to rollback to (default: previous)
    -f, --force             Force rollback without confirmation
    -d, --dry-run           Show what would be done without executing
    --verbose               Enable verbose logging
    -h, --help              Show this help message

Examples:
    $0 --environment staging --type kubernetes
    $0 --environment production --type kubernetes --revision 3
    $0 --environment production --type docker --force
    $0 --dry-run --verbose

Supported deployment types:
    docker      - Rollback Docker Compose deployment
    kubernetes  - Rollback Kubernetes deployment

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -t|--type)
                DEPLOYMENT_TYPE="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -r|--revision)
                ROLLBACK_REVISION="$2"
                shift 2
                ;;
            -f|--force)
                FORCE="true"
                shift
                ;;
            -d|--dry-run)
                DRY_RUN="true"
                shift
                ;;
            --verbose)
                VERBOSE="true"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                error_exit "Unknown option: $1"
                ;;
        esac
    done
}

# Validation functions
validate_environment() {
    case "$ENVIRONMENT" in
        development|staging|production)
            log_info "Rolling back $ENVIRONMENT environment"
            ;;
        *)
            error_exit "Invalid environment: $ENVIRONMENT"
            ;;
    esac
}

validate_deployment_type() {
    case "$DEPLOYMENT_TYPE" in
        docker|kubernetes)
            log_info "Using $DEPLOYMENT_TYPE rollback"
            ;;
        *)
            error_exit "Invalid deployment type: $DEPLOYMENT_TYPE"
            ;;
    esac
}

validate_prerequisites() {
    log_info "Validating prerequisites..."
    
    # Check required commands
    local required_commands=("git")
    
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        required_commands+=("kubectl")
    elif [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        required_commands+=("docker" "docker-compose")
    fi
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            error_exit "Required command not found: $cmd"
        fi
        log_debug "Found command: $cmd"
    done
    
    # Check Kubernetes cluster (if applicable)
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        if ! kubectl cluster-info >/dev/null 2>&1; then
            error_exit "Cannot connect to Kubernetes cluster"
        fi
        
        # Check if namespace exists
        if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
            error_exit "Namespace '$NAMESPACE' does not exist"
        fi
        
        log_debug "Kubernetes cluster and namespace are accessible"
    fi
    
    # Check Docker (if applicable)
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        if ! docker info >/dev/null 2>&1; then
            error_exit "Docker daemon is not running"
        fi
        log_debug "Docker daemon is accessible"
    fi
}

# Confirmation function
confirm_rollback() {
    if [[ "$FORCE" == "true" ]] || [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    
    log_warn "This will rollback the $ENVIRONMENT environment deployment"
    
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        # Show current deployment status
        log_info "Current deployment status:"
        kubectl get deployment quantum-rerank -n "$NAMESPACE" -o wide 2>/dev/null || true
        
        # Show rollout history
        log_info "Rollout history:"
        kubectl rollout history deployment/quantum-rerank -n "$NAMESPACE" 2>/dev/null || true
    fi
    
    echo -n "Are you sure you want to proceed? (yes/no): "
    read -r response
    
    case "$response" in
        yes|YES|y|Y)
            log_info "Proceeding with rollback..."
            ;;
        *)
            log_info "Rollback cancelled"
            exit 0
            ;;
    esac
}

# Backup current state
backup_current_state() {
    log_info "Backing up current state..."
    
    local backup_dir="/tmp/quantum-rerank-backup-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"
    
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        # Backup Kubernetes resources
        kubectl get deployment quantum-rerank -n "$NAMESPACE" -o yaml > "$backup_dir/deployment.yaml" 2>/dev/null || true
        kubectl get service quantum-rerank-service -n "$NAMESPACE" -o yaml > "$backup_dir/service.yaml" 2>/dev/null || true
        kubectl get configmap quantum-rerank-config -n "$NAMESPACE" -o yaml > "$backup_dir/configmap.yaml" 2>/dev/null || true
        
        # Get pod logs
        kubectl logs -n "$NAMESPACE" -l app=quantum-rerank --tail=100 > "$backup_dir/pod-logs.txt" 2>/dev/null || true
        
        log_info "Kubernetes state backed up to: $backup_dir"
    elif [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        # Backup Docker Compose state
        cd "$DEPLOYMENT_DIR"
        docker-compose ps > "$backup_dir/compose-ps.txt" 2>/dev/null || true
        docker-compose logs --tail=100 > "$backup_dir/compose-logs.txt" 2>/dev/null || true
        
        log_info "Docker Compose state backed up to: $backup_dir"
    fi
    
    echo "$backup_dir" > "/tmp/quantum-rerank-last-backup"
    log_debug "Backup location saved to /tmp/quantum-rerank-last-backup"
}

# Kubernetes rollback
rollback_kubernetes() {
    log_info "Rolling back Kubernetes deployment..."
    
    # Check if deployment exists
    if ! kubectl get deployment quantum-rerank -n "$NAMESPACE" >/dev/null 2>&1; then
        error_exit "Deployment 'quantum-rerank' not found in namespace '$NAMESPACE'"
    fi
    
    # Build rollback command
    local rollback_cmd="kubectl rollout undo deployment/quantum-rerank -n $NAMESPACE"
    
    if [[ -n "$ROLLBACK_REVISION" ]]; then
        rollback_cmd="$rollback_cmd --to-revision=$ROLLBACK_REVISION"
        log_info "Rolling back to revision: $ROLLBACK_REVISION"
    else
        log_info "Rolling back to previous revision"
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would execute: $rollback_cmd"
        return 0
    fi
    
    # Perform rollback
    if ! eval "$rollback_cmd"; then
        error_exit "Failed to initiate rollback"
    fi
    
    # Wait for rollback to complete
    log_info "Waiting for rollback to complete..."
    if ! kubectl rollout status deployment/quantum-rerank -n "$NAMESPACE" --timeout=300s; then
        error_exit "Rollback did not complete within timeout"
    fi
    
    # Verify rollback
    log_info "Verifying rollback..."
    local current_revision
    current_revision=$(kubectl get deployment quantum-rerank -n "$NAMESPACE" -o jsonpath='{.metadata.annotations.deployment\.kubernetes\.io/revision}')
    log_info "Current revision after rollback: $current_revision"
    
    # Show pod status
    kubectl get pods -n "$NAMESPACE" -l app=quantum-rerank
}

# Docker rollback
rollback_docker() {
    log_info "Rolling back Docker Compose deployment..."
    
    cd "$DEPLOYMENT_DIR"
    
    # Check if compose file exists
    if [[ ! -f "docker-compose.yml" ]]; then
        error_exit "docker-compose.yml not found in $DEPLOYMENT_DIR"
    fi
    
    # Build compose command
    local compose_args=("-f" "docker-compose.yml")
    
    if [[ "$ENVIRONMENT" == "production" && -f "docker-compose.prod.yml" ]]; then
        compose_args+=("-f" "docker-compose.prod.yml")
    fi
    
    # Get current image tags
    log_info "Current image tags:"
    docker-compose "${compose_args[@]}" images || true
    
    # For Docker, we'll need to specify which image to rollback to
    if [[ -z "$ROLLBACK_REVISION" ]]; then
        log_warn "No specific revision specified for Docker rollback"
        log_info "Available images:"
        docker images quantum-rerank --format "table {{.Tag}}\t{{.CreatedAt}}\t{{.Size}}" || true
        
        # Try to find previous tag
        local previous_tag
        previous_tag=$(docker images quantum-rerank --format "{{.Tag}}" | grep -v latest | head -1 || echo "")
        
        if [[ -n "$previous_tag" ]]; then
            ROLLBACK_REVISION="$previous_tag"
            log_info "Using previous tag: $ROLLBACK_REVISION"
        else
            error_exit "No previous image found for rollback. Please specify --revision"
        fi
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would rollback to image: quantum-rerank:$ROLLBACK_REVISION"
        return 0
    fi
    
    # Update docker-compose.yml with rollback image
    local temp_compose="/tmp/docker-compose-rollback.yml"
    sed "s|quantum-rerank:latest|quantum-rerank:$ROLLBACK_REVISION|g" \
        "docker-compose.yml" > "$temp_compose"
    
    # Stop current services
    log_info "Stopping current services..."
    docker-compose "${compose_args[@]}" down
    
    # Start with rollback image
    log_info "Starting services with rollback image..."
    docker-compose -f "$temp_compose" up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    local retries=0
    local max_retries=30
    
    while [[ $retries -lt $max_retries ]]; do
        if docker-compose -f "$temp_compose" ps | grep -q "Up"; then
            log_info "Services are running"
            break
        fi
        
        retries=$((retries + 1))
        log_debug "Health check attempt $retries/$max_retries"
        sleep 10
    done
    
    # Show service status
    docker-compose -f "$temp_compose" ps
    
    # Cleanup
    rm -f "$temp_compose"
}

# Health check after rollback
post_rollback_check() {
    log_info "Performing post-rollback health checks..."
    
    local health_check_url=""
    local port_forward_pid=""
    
    case "$DEPLOYMENT_TYPE" in
        docker)
            health_check_url="http://localhost:8000/health"
            ;;
        kubernetes)
            # Port forward for health check
            kubectl port-forward -n "$NAMESPACE" service/quantum-rerank-service 8080:80 &
            port_forward_pid=$!
            sleep 5
            health_check_url="http://localhost:8080/health"
            ;;
    esac
    
    if [[ -n "$health_check_url" ]]; then
        local retries=0
        local max_retries=10
        
        while [[ $retries -lt $max_retries ]]; do
            if curl -f -s "$health_check_url" >/dev/null 2>&1; then
                log_info "Health check passed: $health_check_url"
                break
            fi
            
            retries=$((retries + 1))
            log_debug "Health check attempt $retries/$max_retries"
            sleep 10
        done
        
        if [[ $retries -eq $max_retries ]]; then
            log_warn "Health check failed after $max_retries attempts"
            log_warn "Service may not be fully healthy after rollback"
        else
            log_info "Service is healthy after rollback"
        fi
    fi
    
    # Clean up port forward
    if [[ -n "$port_forward_pid" ]]; then
        kill "$port_forward_pid" 2>/dev/null || true
    fi
}

# Main rollback function
main() {
    log_info "Starting QuantumRerank rollback"
    log_info "Environment: $ENVIRONMENT"
    log_info "Deployment type: $DEPLOYMENT_TYPE"
    
    # Parse arguments
    parse_args "$@"
    
    # Validate inputs
    validate_environment
    validate_deployment_type
    validate_prerequisites
    
    # Confirm rollback (unless force or dry-run)
    confirm_rollback
    
    # Backup current state
    if [[ "$DRY_RUN" != "true" ]]; then
        backup_current_state
    fi
    
    # Perform rollback based on type
    case "$DEPLOYMENT_TYPE" in
        kubernetes)
            rollback_kubernetes
            ;;
        docker)
            rollback_docker
            ;;
    esac
    
    # Post-rollback checks
    if [[ "$DRY_RUN" != "true" ]]; then
        post_rollback_check
    fi
    
    log_info "Rollback completed successfully!"
    
    # Show rollback information
    if [[ -f "/tmp/quantum-rerank-last-backup" ]]; then
        local backup_dir
        backup_dir=$(cat /tmp/quantum-rerank-last-backup)
        log_info "Previous state backed up to: $backup_dir"
    fi
    
    # Show next steps
    case "$DEPLOYMENT_TYPE" in
        kubernetes)
            log_info "Check deployment status: kubectl get pods -n $NAMESPACE"
            log_info "View rollout history: kubectl rollout history deployment/quantum-rerank -n $NAMESPACE"
            log_info "View logs: kubectl logs -n $NAMESPACE -l app=quantum-rerank"
            ;;
        docker)
            log_info "Check service status: docker-compose ps"
            log_info "View logs: docker-compose logs -f"
            ;;
    esac
}

# Run main function with all arguments
main "$@"
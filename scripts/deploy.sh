#!/bin/bash
# Comprehensive deployment script for QuantumRerank
# Supports Docker, Kubernetes, and cloud platforms

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_DIR="$PROJECT_DIR/deployment"

# Default configuration
ENVIRONMENT="${ENVIRONMENT:-staging}"
DEPLOYMENT_TYPE="${DEPLOYMENT_TYPE:-docker}"
VERSION="${VERSION:-latest}"
NAMESPACE="${NAMESPACE:-quantum-rerank}"
REGISTRY="${REGISTRY:-}"
BUILD_ARGS="${BUILD_ARGS:-}"
FORCE_BUILD="${FORCE_BUILD:-false}"
DRY_RUN="${DRY_RUN:-false}"
VERBOSE="${VERBOSE:-false}"

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
QuantumRerank Deployment Script

Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV     Target environment (development, staging, production)
    -t, --type TYPE          Deployment type (docker, kubernetes, cloud)
    -v, --version VERSION    Application version tag
    -n, --namespace NS       Kubernetes namespace
    -r, --registry REG       Container registry
    -f, --force-build       Force rebuild of container images
    -d, --dry-run           Show what would be done without executing
    --verbose               Enable verbose logging
    -h, --help              Show this help message

Examples:
    $0 --environment staging --type docker
    $0 --environment production --type kubernetes --version v1.2.3
    $0 --environment production --type cloud --registry my-registry.io
    $0 --dry-run --verbose

Supported deployment types:
    docker      - Deploy using Docker Compose
    kubernetes  - Deploy to Kubernetes cluster
    cloud       - Deploy to cloud platform (AWS, GCP, Azure)

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
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -r|--registry)
                REGISTRY="$2"
                shift 2
                ;;
            -f|--force-build)
                FORCE_BUILD="true"
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
            log_info "Deploying to $ENVIRONMENT environment"
            ;;
        *)
            error_exit "Invalid environment: $ENVIRONMENT"
            ;;
    esac
}

validate_deployment_type() {
    case "$DEPLOYMENT_TYPE" in
        docker|kubernetes|cloud)
            log_info "Using $DEPLOYMENT_TYPE deployment"
            ;;
        *)
            error_exit "Invalid deployment type: $DEPLOYMENT_TYPE"
            ;;
    esac
}

validate_prerequisites() {
    log_info "Validating prerequisites..."
    
    # Check required commands
    local required_commands=("docker" "git")
    
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        required_commands+=("kubectl")
    fi
    
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        required_commands+=("docker-compose")
    fi
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            error_exit "Required command not found: $cmd"
        fi
        log_debug "Found command: $cmd"
    done
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        error_exit "Docker daemon is not running"
    fi
    
    # Check Kubernetes cluster (if applicable)
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        if ! kubectl cluster-info >/dev/null 2>&1; then
            error_exit "Cannot connect to Kubernetes cluster"
        fi
        log_debug "Kubernetes cluster is accessible"
    fi
    
    # Check required files
    local required_files=(
        "$DEPLOYMENT_DIR/Dockerfile"
        "$DEPLOYMENT_DIR/environments/$ENVIRONMENT.yaml"
    )
    
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        required_files+=("$DEPLOYMENT_DIR/docker-compose.yml")
        if [[ "$ENVIRONMENT" == "production" ]]; then
            required_files+=("$DEPLOYMENT_DIR/docker-compose.prod.yml")
        fi
    fi
    
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        required_files+=(
            "$DEPLOYMENT_DIR/k8s/deployment.yaml"
            "$DEPLOYMENT_DIR/k8s/service.yaml"
            "$DEPLOYMENT_DIR/k8s/configmap.yaml"
        )
    fi
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            error_exit "Required file not found: $file"
        fi
        log_debug "Found file: $file"
    done
}

# Build functions
build_image() {
    log_info "Building container image..."
    
    local image_tag="${REGISTRY:+$REGISTRY/}quantum-rerank:$VERSION"
    local build_args_array=()
    
    # Add build arguments
    build_args_array+=(
        "--build-arg" "ENVIRONMENT=$ENVIRONMENT"
        "--build-arg" "VERSION=$VERSION"
    )
    
    # Add custom build args
    if [[ -n "$BUILD_ARGS" ]]; then
        IFS=' ' read -ra ADDR <<< "$BUILD_ARGS"
        for arg in "${ADDR[@]}"; do
            build_args_array+=("--build-arg" "$arg")
        done
    fi
    
    # Check if image exists and skip if not forcing rebuild
    if [[ "$FORCE_BUILD" != "true" ]] && docker image inspect "$image_tag" >/dev/null 2>&1; then
        log_info "Image $image_tag already exists, skipping build"
        return 0
    fi
    
    log_info "Building image: $image_tag"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would build: docker build ${build_args_array[*]} -f $DEPLOYMENT_DIR/Dockerfile -t $image_tag $PROJECT_DIR"
        return 0
    fi
    
    if ! docker build \
        "${build_args_array[@]}" \
        -f "$DEPLOYMENT_DIR/Dockerfile" \
        -t "$image_tag" \
        "$PROJECT_DIR"; then
        error_exit "Failed to build container image"
    fi
    
    log_info "Successfully built image: $image_tag"
    
    # Push to registry if specified
    if [[ -n "$REGISTRY" ]]; then
        push_image "$image_tag"
    fi
}

push_image() {
    local image_tag="$1"
    
    log_info "Pushing image to registry: $image_tag"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would push: docker push $image_tag"
        return 0
    fi
    
    if ! docker push "$image_tag"; then
        error_exit "Failed to push image to registry"
    fi
    
    log_info "Successfully pushed image: $image_tag"
}

# Deployment functions
deploy_docker() {
    log_info "Deploying with Docker Compose..."
    
    local compose_file="$DEPLOYMENT_DIR/docker-compose.yml"
    local compose_args=("-f" "$compose_file")
    
    # Use production compose file for production environment
    if [[ "$ENVIRONMENT" == "production" ]]; then
        compose_args+=("-f" "$DEPLOYMENT_DIR/docker-compose.prod.yml")
    fi
    
    # Set environment variables
    export ENVIRONMENT VERSION
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would run: docker-compose ${compose_args[*]} up -d"
        return 0
    fi
    
    # Stop existing services
    log_info "Stopping existing services..."
    docker-compose "${compose_args[@]}" down || true
    
    # Start services
    log_info "Starting services..."
    if ! docker-compose "${compose_args[@]}" up -d; then
        error_exit "Failed to start services with Docker Compose"
    fi
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    local retries=0
    local max_retries=30
    
    while [[ $retries -lt $max_retries ]]; do
        if docker-compose "${compose_args[@]}" ps | grep -q "Up (healthy)"; then
            log_info "Services are healthy"
            break
        fi
        
        retries=$((retries + 1))
        log_debug "Health check attempt $retries/$max_retries"
        sleep 10
    done
    
    if [[ $retries -eq $max_retries ]]; then
        log_warn "Services did not become healthy within expected time"
    fi
    
    # Show service status
    docker-compose "${compose_args[@]}" ps
}

deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Create namespace if it doesn't exist
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        log_info "Creating namespace: $NAMESPACE"
        if [[ "$DRY_RUN" != "true" ]]; then
            kubectl create namespace "$NAMESPACE"
        fi
    fi
    
    # Apply ConfigMaps first
    log_info "Applying ConfigMaps..."
    local configmap_file="$DEPLOYMENT_DIR/k8s/configmap.yaml"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would apply: kubectl apply -f $configmap_file -n $NAMESPACE"
    else
        kubectl apply -f "$configmap_file" -n "$NAMESPACE"
    fi
    
    # Apply Secrets
    log_info "Applying Secrets..."
    local secret_file="$DEPLOYMENT_DIR/k8s/secret.yaml"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would apply: kubectl apply -f $secret_file -n $NAMESPACE"
    else
        kubectl apply -f "$secret_file" -n "$NAMESPACE"
    fi
    
    # Apply Deployment
    log_info "Applying Deployment..."
    local deployment_file="$DEPLOYMENT_DIR/k8s/deployment.yaml"
    
    # Update image tag in deployment
    local temp_deployment="/tmp/deployment-$ENVIRONMENT.yaml"
    sed "s|quantum-rerank:latest|${REGISTRY:+$REGISTRY/}quantum-rerank:$VERSION|g" \
        "$deployment_file" > "$temp_deployment"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would apply: kubectl apply -f $temp_deployment -n $NAMESPACE"
    else
        kubectl apply -f "$temp_deployment" -n "$NAMESPACE"
    fi
    
    # Apply Service
    log_info "Applying Service..."
    local service_file="$DEPLOYMENT_DIR/k8s/service.yaml"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would apply: kubectl apply -f $service_file -n $NAMESPACE"
    else
        kubectl apply -f "$service_file" -n "$NAMESPACE"
    fi
    
    # Apply Ingress (if in production)
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log_info "Applying Ingress..."
        local ingress_file="$DEPLOYMENT_DIR/k8s/ingress.yaml"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY RUN] Would apply: kubectl apply -f $ingress_file -n $NAMESPACE"
        else
            kubectl apply -f "$ingress_file" -n "$NAMESPACE"
        fi
    fi
    
    # Wait for rollout to complete
    if [[ "$DRY_RUN" != "true" ]]; then
        log_info "Waiting for deployment rollout..."
        kubectl rollout status deployment/quantum-rerank -n "$NAMESPACE" --timeout=300s
        
        # Show deployment status
        kubectl get pods -n "$NAMESPACE" -l app=quantum-rerank
    fi
    
    # Cleanup temporary files
    rm -f "$temp_deployment"
}

deploy_cloud() {
    log_info "Deploying to cloud platform..."
    
    # Detect cloud platform
    local cloud_platform=""
    
    if kubectl get nodes -o wide 2>/dev/null | grep -q "eks"; then
        cloud_platform="aws"
    elif kubectl get nodes -o wide 2>/dev/null | grep -q "gke"; then
        cloud_platform="gcp"
    elif kubectl get nodes -o wide 2>/dev/null | grep -q "aks"; then
        cloud_platform="azure"
    else
        log_warn "Could not detect cloud platform, using generic Kubernetes deployment"
        deploy_kubernetes
        return
    fi
    
    log_info "Detected cloud platform: $cloud_platform"
    
    # Apply cloud-specific configurations
    local cloud_config_dir="$DEPLOYMENT_DIR/cloud/$cloud_platform"
    
    if [[ -d "$cloud_config_dir" ]]; then
        log_info "Applying cloud-specific configurations..."
        
        for config_file in "$cloud_config_dir"/*.yaml; do
            if [[ -f "$config_file" ]]; then
                log_info "Applying: $(basename "$config_file")"
                if [[ "$DRY_RUN" == "true" ]]; then
                    log_info "[DRY RUN] Would apply: kubectl apply -f $config_file -n $NAMESPACE"
                else
                    kubectl apply -f "$config_file" -n "$NAMESPACE"
                fi
            fi
        done
    fi
    
    # Deploy using Kubernetes
    deploy_kubernetes
}

# Health check after deployment
post_deployment_check() {
    log_info "Performing post-deployment health checks..."
    
    local health_check_url=""
    
    case "$DEPLOYMENT_TYPE" in
        docker)
            health_check_url="http://localhost:8000/health"
            ;;
        kubernetes)
            # Port forward for health check
            kubectl port-forward -n "$NAMESPACE" service/quantum-rerank-service 8080:80 &
            local pf_pid=$!
            sleep 5
            health_check_url="http://localhost:8080/health"
            ;;
        cloud)
            # Try to get external IP/URL
            local external_ip
            external_ip=$(kubectl get service -n "$NAMESPACE" quantum-rerank-loadbalancer -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
            
            if [[ -n "$external_ip" ]]; then
                health_check_url="http://$external_ip/health"
            else
                log_warn "Could not determine external URL, skipping health check"
                return 0
            fi
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
        fi
    fi
    
    # Clean up port forward
    if [[ -n "${pf_pid:-}" ]]; then
        kill "$pf_pid" 2>/dev/null || true
    fi
}

# Main deployment function
main() {
    log_info "Starting QuantumRerank deployment"
    log_info "Environment: $ENVIRONMENT"
    log_info "Deployment type: $DEPLOYMENT_TYPE"
    log_info "Version: $VERSION"
    
    # Parse arguments
    parse_args "$@"
    
    # Validate inputs
    validate_environment
    validate_deployment_type
    validate_prerequisites
    
    # Change to project directory
    cd "$PROJECT_DIR"
    
    # Build container image
    if [[ "$DEPLOYMENT_TYPE" != "local" ]]; then
        build_image
    fi
    
    # Deploy based on type
    case "$DEPLOYMENT_TYPE" in
        docker)
            deploy_docker
            ;;
        kubernetes)
            deploy_kubernetes
            ;;
        cloud)
            deploy_cloud
            ;;
    esac
    
    # Post-deployment checks
    if [[ "$DRY_RUN" != "true" ]]; then
        post_deployment_check
    fi
    
    log_info "Deployment completed successfully!"
    
    # Show next steps
    case "$DEPLOYMENT_TYPE" in
        docker)
            log_info "Access the application at: http://localhost:8000"
            log_info "View logs with: docker-compose logs -f"
            ;;
        kubernetes)
            log_info "Check deployment status: kubectl get pods -n $NAMESPACE"
            log_info "View logs: kubectl logs -n $NAMESPACE -l app=quantum-rerank"
            ;;
        cloud)
            log_info "Check deployment status: kubectl get all -n $NAMESPACE"
            log_info "Get external URL: kubectl get ingress -n $NAMESPACE"
            ;;
    esac
}

# Run main function with all arguments
main "$@"
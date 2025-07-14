#!/bin/bash
# Universal deployment script for QuantumRerank
# Auto-detects platform or allows manual selection
# Usage: ./universal-deploy.sh [--platform aws|gcp|azure|k8s|docker] [--environment production|staging]

set -e

# Default values
PLATFORM=""
ENVIRONMENT="production"
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

show_help() {
    cat << EOF
QuantumRerank Universal Deployment Script

Usage: $0 [OPTIONS]

Options:
    --platform PLATFORM    Target platform (aws|gcp|azure|k8s|docker)
    --environment ENV       Environment (production|staging|development)
    --help                  Show this help message

Platforms:
    aws     - Deploy to AWS ECS Fargate
    gcp     - Deploy to Google Cloud Run
    azure   - Deploy to Azure Container Instances
    k8s     - Deploy to Kubernetes cluster
    docker  - Deploy using Docker Compose locally

Environment Variables:
    QUANTUM_RERANK_API_KEY  - Required: API key for the service
    QUANTUM_RERANK_IMAGE    - Optional: Custom Docker image
    AWS_REGION              - AWS deployment region (default: us-west-2)
    GOOGLE_CLOUD_PROJECT    - GCP project ID (required for GCP)
    AZURE_LOCATION          - Azure region (default: eastus)

Examples:
    $0 --platform aws --environment production
    $0 --platform gcp --environment staging
    $0 --platform docker
    $0  # Auto-detect platform

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            error "Unknown option $1. Use --help for usage information."
            ;;
    esac
done

# Auto-detect platform if not specified
if [ -z "$PLATFORM" ]; then
    log "Auto-detecting deployment platform..."
    
    if command -v aws &> /dev/null && aws sts get-caller-identity &> /dev/null; then
        PLATFORM="aws"
        log "Detected AWS CLI with valid credentials"
    elif command -v gcloud &> /dev/null && [ -n "${GOOGLE_CLOUD_PROJECT:-}" ] && gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 &> /dev/null; then
        PLATFORM="gcp"
        log "Detected Google Cloud CLI with valid credentials"
    elif command -v az &> /dev/null && az account show &> /dev/null; then
        PLATFORM="azure"
        log "Detected Azure CLI with valid credentials"
    elif command -v kubectl &> /dev/null && kubectl cluster-info &> /dev/null; then
        PLATFORM="k8s"
        log "Detected Kubernetes cluster connection"
    elif command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
        PLATFORM="docker"
        log "Detected Docker and Docker Compose"
    else
        error "No supported platform detected. Please specify --platform or install/configure a supported platform."
    fi
fi

# Validate platform
case $PLATFORM in
    aws|gcp|azure|k8s|docker)
        success "Selected platform: $PLATFORM"
        ;;
    *)
        error "Unsupported platform: $PLATFORM. Supported platforms: aws, gcp, azure, k8s, docker"
        ;;
esac

# Validate environment
case $ENVIRONMENT in
    production|staging|development)
        success "Selected environment: $ENVIRONMENT"
        ;;
    *)
        error "Unsupported environment: $ENVIRONMENT. Supported environments: production, staging, development"
        ;;
esac

# Validate required environment variables
if [ -z "${QUANTUM_RERANK_API_KEY:-}" ]; then
    # Generate API key if not provided
    warning "QUANTUM_RERANK_API_KEY not set, generating random API key..."
    export QUANTUM_RERANK_API_KEY="qr_$(openssl rand -hex 16)"
    log "Generated API key: $QUANTUM_RERANK_API_KEY"
    echo "export QUANTUM_RERANK_API_KEY=$QUANTUM_RERANK_API_KEY" > .env.deploy
    log "API key saved to .env.deploy file"
fi

# Platform-specific validation
case $PLATFORM in
    gcp)
        if [ -z "${GOOGLE_CLOUD_PROJECT:-}" ]; then
            error "GOOGLE_CLOUD_PROJECT environment variable is required for GCP deployment"
        fi
        ;;
esac

echo ""
log "🚀 Starting QuantumRerank deployment"
log "   Platform: $PLATFORM"
log "   Environment: $ENVIRONMENT"
log "   API Key: ${QUANTUM_RERANK_API_KEY:0:8}..."
echo ""

# Execute platform-specific deployment
case $PLATFORM in
    aws)
        log "Deploying to AWS ECS..."
        $SCRIPT_DIR/deploy-aws.sh $ENVIRONMENT
        ;;
    gcp)
        log "Deploying to Google Cloud Run..."
        $SCRIPT_DIR/deploy-gcp.sh $ENVIRONMENT
        ;;
    azure)
        log "Deploying to Azure Container Instances..."
        $SCRIPT_DIR/deploy-azure.sh $ENVIRONMENT
        ;;
    k8s)
        log "Deploying to Kubernetes..."
        $SCRIPT_DIR/deploy-k8s.sh $ENVIRONMENT
        ;;
    docker)
        log "Deploying with Docker Compose..."
        if [ -f "docker-compose.simple.yml" ]; then
            docker-compose -f docker-compose.simple.yml down 2>/dev/null || true
            docker-compose -f docker-compose.simple.yml up -d
            
            # Wait for health check
            sleep 15
            if curl -f -s "http://localhost:8000/health" > /dev/null; then
                success "Docker deployment successful"
                log "   Health check: http://localhost:8000/health"
                log "   API endpoint: http://localhost:8000/v1/rerank"
                log "   API Key: $QUANTUM_RERANK_API_KEY"
            else
                error "Docker deployment health check failed"
            fi
        else
            error "docker-compose.simple.yml not found. Run from project root directory."
        fi
        ;;
    *)
        error "Unsupported platform: $PLATFORM"
        ;;
esac

echo ""
success "🎉 QuantumRerank deployment completed successfully!"
echo ""
log "Next steps:"
log "1. Test the health endpoint to verify the service is running"
log "2. Test the API with a sample request"
log "3. Set up monitoring and alerting for production use"
log "4. Configure DNS and SSL certificates if needed"
echo ""
log "For production use, consider:"
log "- Setting up proper DNS records"
log "- Configuring SSL/TLS certificates"
log "- Setting up monitoring and logging"
log "- Implementing backup and disaster recovery"
log "- Configuring auto-scaling policies"
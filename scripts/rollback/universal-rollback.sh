#!/bin/bash
# Universal rollback script for QuantumRerank
# Auto-detects platform or allows manual selection
# Usage: ./universal-rollback.sh [--platform aws|gcp|azure|k8s] [--environment production|staging] [--revision revision_number]

set -e

# Default values
PLATFORM=""
ENVIRONMENT="production"
REVISION=""
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
QuantumRerank Universal Rollback Script

Usage: $0 [OPTIONS]

Options:
    --platform PLATFORM    Target platform (aws|gcp|azure|k8s)
    --environment ENV       Environment (production|staging|development)
    --revision REV          Specific revision/tag to rollback to
    --help                  Show this help message

Platforms:
    aws     - Rollback AWS ECS Fargate deployment
    gcp     - Rollback Google Cloud Run deployment
    azure   - Rollback Azure Container Instances deployment
    k8s     - Rollback Kubernetes deployment

Examples:
    $0 --platform aws --environment production
    $0 --platform gcp --revision 5
    $0 --platform k8s --environment staging --revision 3
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
        --revision)
            REVISION="$2"
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
    else
        error "No supported platform detected. Please specify --platform or install/configure a supported platform."
    fi
fi

# Validate platform
case $PLATFORM in
    aws|gcp|azure|k8s)
        success "Selected platform: $PLATFORM"
        ;;
    *)
        error "Unsupported platform: $PLATFORM. Supported platforms: aws, gcp, azure, k8s"
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

echo ""
warning "🔄 ROLLBACK OPERATION"
log "   Platform: $PLATFORM"
log "   Environment: $ENVIRONMENT"
if [ -n "$REVISION" ]; then
    log "   Target revision: $REVISION"
else
    log "   Target revision: Previous (auto-detected)"
fi
echo ""
warning "⚠️  This will rollback your deployment to a previous version!"
echo ""
read -p "Do you want to continue with the rollback? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    log "Rollback cancelled by user"
    exit 0
fi

echo ""
log "🚀 Starting QuantumRerank rollback"
echo ""

# Execute platform-specific rollback
case $PLATFORM in
    aws)
        log "Rolling back AWS ECS deployment..."
        if [ -n "$REVISION" ]; then
            $SCRIPT_DIR/rollback-aws.sh $ENVIRONMENT $REVISION
        else
            $SCRIPT_DIR/rollback-aws.sh $ENVIRONMENT
        fi
        ;;
    gcp)
        log "Rolling back Google Cloud Run deployment..."
        if [ -n "$REVISION" ]; then
            $SCRIPT_DIR/rollback-gcp.sh $ENVIRONMENT $REVISION
        else
            $SCRIPT_DIR/rollback-gcp.sh $ENVIRONMENT
        fi
        ;;
    azure)
        log "Rolling back Azure Container Instances deployment..."
        if [ -n "$REVISION" ]; then
            $SCRIPT_DIR/rollback-azure.sh $ENVIRONMENT $REVISION
        else
            $SCRIPT_DIR/rollback-azure.sh $ENVIRONMENT
        fi
        ;;
    k8s)
        log "Rolling back Kubernetes deployment..."
        if [ -n "$REVISION" ]; then
            $SCRIPT_DIR/rollback-k8s.sh $ENVIRONMENT $REVISION
        else
            $SCRIPT_DIR/rollback-k8s.sh $ENVIRONMENT
        fi
        ;;
    *)
        error "Unsupported platform: $PLATFORM"
        ;;
esac

echo ""
success "🎉 QuantumRerank rollback completed successfully!"
echo ""
log "What to do next:"
log "1. Verify the service is working correctly"
log "2. Monitor logs for any issues"
log "3. Test the API endpoints"
log "4. Check monitoring dashboards"
echo ""
log "If issues persist:"
log "- Check the service logs"
log "- Verify the rollback target was correct"
log "- Consider rolling back to an earlier version"
log "- Contact your infrastructure team if needed"
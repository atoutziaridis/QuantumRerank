#!/bin/bash
# Google Cloud Run rollback script for QuantumRerank
# Usage: ./rollback-gcp.sh [environment] [revision_number]

set -e

# Configuration
ENVIRONMENT="${1:-production}"
REVISION="${2:-}"
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="quantum-rerank-${ENVIRONMENT}"

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

# Validate requirements
if [ -z "$PROJECT_ID" ]; then
    error "GOOGLE_CLOUD_PROJECT environment variable is required"
fi

if ! command -v gcloud &> /dev/null; then
    error "Google Cloud CLI is not installed. Please install it first."
fi

if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 &> /dev/null; then
    error "Not authenticated with Google Cloud. Run 'gcloud auth login' first."
fi

# Set project
gcloud config set project $PROJECT_ID

log "🔄 Rolling back QuantumRerank on Google Cloud Run"
log "   Environment: $ENVIRONMENT"
log "   Project: $PROJECT_ID"
log "   Service: $SERVICE_NAME"
log "   Region: $REGION"

# Check if service exists
if ! gcloud run services describe $SERVICE_NAME --region $REGION &> /dev/null; then
    error "Service $SERVICE_NAME not found in region $REGION"
fi

# Get current revision
CURRENT_REVISION=$(gcloud run services describe $SERVICE_NAME \
    --region $REGION \
    --format 'value(status.latestReadyRevisionName)')

log "Current revision: $CURRENT_REVISION"

# List available revisions
log "Available revisions:"
gcloud run revisions list \
    --service $SERVICE_NAME \
    --region $REGION \
    --format 'table(metadata.name,status.conditions[0].lastTransitionTime,spec.containers[0].image)'

# If no revision specified, get previous revision
if [ -z "$REVISION" ]; then
    log "Getting previous revision..."
    
    # Get all revisions sorted by creation time
    REVISIONS=($(gcloud run revisions list \
        --service $SERVICE_NAME \
        --region $REGION \
        --sort-by '~metadata.creationTimestamp' \
        --format 'value(metadata.name)'))
    
    # Find current index and get previous
    CURRENT_INDEX=-1
    for i in "${!REVISIONS[@]}"; do
        if [[ "${REVISIONS[$i]}" == "$CURRENT_REVISION" ]]; then
            CURRENT_INDEX=$i
            break
        fi
    done
    
    if [ $CURRENT_INDEX -le 0 ]; then
        error "No previous revision found for rollback"
    fi
    
    PREVIOUS_INDEX=$((CURRENT_INDEX + 1))
    ROLLBACK_REVISION="${REVISIONS[$PREVIOUS_INDEX]}"
    
    log "Auto-selected previous revision: $ROLLBACK_REVISION"
else
    ROLLBACK_REVISION="$REVISION"
    log "Rolling back to specified revision: $ROLLBACK_REVISION"
    
    # Verify the revision exists
    if ! gcloud run revisions describe "$ROLLBACK_REVISION" \
        --region $REGION &> /dev/null; then
        error "Revision $ROLLBACK_REVISION not found"
    fi
fi

# Confirm rollback
echo ""
warning "⚠️  ROLLBACK CONFIRMATION"
log "Current revision: $CURRENT_REVISION"
log "Rolling back to: $ROLLBACK_REVISION"
echo ""
read -p "Are you sure you want to proceed with rollback? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    log "Rollback cancelled"
    exit 0
fi

# Perform rollback by updating traffic allocation
log "Starting rollback..."
gcloud run services update-traffic $SERVICE_NAME \
    --to-revisions "$ROLLBACK_REVISION=100" \
    --region $REGION

success "Rollback initiated"

# Wait a moment for traffic to switch
log "Waiting for traffic to switch..."
sleep 10

# Verify rollback
NEW_REVISION=$(gcloud run services describe $SERVICE_NAME \
    --region $REGION \
    --format 'value(status.traffic[0].revisionName)')

if [ "$NEW_REVISION" = "$ROLLBACK_REVISION" ]; then
    success "Rollback completed successfully!"
else
    error "Rollback verification failed. Current: $NEW_REVISION, Expected: $ROLLBACK_REVISION"
fi

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
    --region $REGION \
    --format 'value(status.url)')

# Test rollback
log "Testing rolled back service..."
sleep 15

if curl -f -s --max-time 10 "$SERVICE_URL/health" > /dev/null; then
    success "Health check passed after rollback"
else
    warning "Health check failed after rollback"
fi

echo ""
success "🎉 Rollback completed!"
log "   Rolled back from: $CURRENT_REVISION"
log "   Rolled back to: $NEW_REVISION"
log "   Service URL: $SERVICE_URL"
log "   Health endpoint: $SERVICE_URL/health"
echo ""
log "Monitor the rolled back deployment:"
log "   gcloud run services describe $SERVICE_NAME --region $REGION"
log "   gcloud logging read \"resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME\" --limit 50"
#!/bin/bash
# Simple Google Cloud Run deployment for QuantumRerank
# Usage: ./deploy-gcp.sh [environment]

set -e

# Configuration
ENVIRONMENT="${1:-production}"
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="quantum-rerank-${ENVIRONMENT}"
IMAGE_NAME="quantum-rerank"
API_KEY="${QUANTUM_RERANK_API_KEY:-}"

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

if [ -z "$API_KEY" ]; then
    error "QUANTUM_RERANK_API_KEY environment variable is required"
fi

if ! command -v gcloud &> /dev/null; then
    error "Google Cloud CLI is not installed. Please install it first."
fi

if ! command -v docker &> /dev/null; then
    error "Docker is not installed. Please install it first."
fi

# Check authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 &> /dev/null; then
    error "Not authenticated with Google Cloud. Run 'gcloud auth login' first."
fi

# Set project
gcloud config set project $PROJECT_ID

IMAGE="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest"

log "Deploying QuantumRerank to Google Cloud Run..."
log "Environment: $ENVIRONMENT"
log "Project: $PROJECT_ID"
log "Service: $SERVICE_NAME"
log "Region: $REGION"
log "Image: $IMAGE"

# Enable required APIs
log "Enabling required Google Cloud APIs..."
gcloud services enable cloudbuild.googleapis.com --quiet
gcloud services enable run.googleapis.com --quiet
gcloud services enable containerregistry.googleapis.com --quiet
success "APIs enabled"

# Build and push image
log "Building Docker image..."
docker build -t $IMAGE .
success "Image built"

log "Pushing image to Google Container Registry..."
docker push $IMAGE
success "Image pushed"

# Deploy to Cloud Run
log "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --concurrency 100 \
    --max-instances 10 \
    --timeout 300 \
    --set-env-vars QUANTUM_RERANK_API_KEY=$API_KEY,ENVIRONMENT=$ENVIRONMENT \
    --port 8000 \
    --quiet

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')

# Test deployment
log "Testing deployment..."
sleep 10

if curl -f -s "$SERVICE_URL/health" > /dev/null; then
    success "Health check passed"
else
    error "Health check failed"
fi

success "Deployment completed successfully!"
echo ""
log "🚀 QuantumRerank is now running on Google Cloud Run"
log "   Service: $SERVICE_NAME"
log "   URL: $SERVICE_URL"
log "   Health check: $SERVICE_URL/health"
log "   API endpoint: $SERVICE_URL/v1/rerank"
log "   API Key: $API_KEY"
echo ""
log "Monitor deployment:"
log "   gcloud run services describe $SERVICE_NAME --region $REGION"
log "   gcloud logging read \"resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME\" --limit 50"
echo ""
log "Test the API:"
echo "curl -X POST $SERVICE_URL/v1/rerank \\"
echo "  -H 'Authorization: Bearer $API_KEY' \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"query\": \"test\", \"candidates\": [\"doc1\", \"doc2\"], \"method\": \"hybrid\"}'"
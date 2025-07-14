#!/bin/bash
# Simple Azure Container Instances deployment for QuantumRerank
# Usage: ./deploy-azure.sh [environment]

set -e

# Configuration
ENVIRONMENT="${1:-production}"
RESOURCE_GROUP="quantum-rerank-${ENVIRONMENT}-rg"
CONTAINER_NAME="quantum-rerank-${ENVIRONMENT}"
IMAGE="${QUANTUM_RERANK_IMAGE:-quantumrerank/server:latest}"
LOCATION="${AZURE_LOCATION:-eastus}"
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
if [ -z "$API_KEY" ]; then
    error "QUANTUM_RERANK_API_KEY environment variable is required"
fi

if ! command -v az &> /dev/null; then
    error "Azure CLI is not installed. Please install it first."
fi

# Check authentication
if ! az account show &> /dev/null; then
    error "Not authenticated with Azure. Run 'az login' first."
fi

SUBSCRIPTION_ID=$(az account show --query id --output tsv)

log "Deploying QuantumRerank to Azure Container Instances..."
log "Environment: $ENVIRONMENT"
log "Subscription: $SUBSCRIPTION_ID"
log "Resource Group: $RESOURCE_GROUP"
log "Container: $CONTAINER_NAME"
log "Location: $LOCATION"
log "Image: $IMAGE"

# Create resource group if it doesn't exist
log "Creating resource group..."
az group create --name $RESOURCE_GROUP --location $LOCATION --output none
success "Resource group ready"

# Generate random DNS label suffix
DNS_SUFFIX=$(openssl rand -hex 4)
DNS_NAME="quantum-rerank-${ENVIRONMENT}-${DNS_SUFFIX}"

# Deploy container
log "Creating container instance..."
az container create \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --image $IMAGE \
    --cpu 2 \
    --memory 2 \
    --restart-policy Always \
    --ports 8000 \
    --environment-variables \
        QUANTUM_RERANK_API_KEY=$API_KEY \
        ENVIRONMENT=$ENVIRONMENT \
    --dns-name-label $DNS_NAME \
    --output none

# Wait for container to be running
log "Waiting for container to start..."
for i in {1..60}; do
    STATE=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query instanceView.state --output tsv)
    if [ "$STATE" = "Running" ]; then
        success "Container is running"
        break
    fi
    if [ $i -eq 60 ]; then
        error "Container failed to start within 5 minutes"
    fi
    sleep 5
done

# Get container endpoint
CONTAINER_FQDN=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query ipAddress.fqdn --output tsv)
CONTAINER_IP=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query ipAddress.ip --output tsv)

# Test deployment
log "Testing deployment..."
sleep 30

HEALTH_URL="http://${CONTAINER_FQDN}:8000/health"
if curl -f -s "$HEALTH_URL" > /dev/null; then
    success "Health check passed"
else
    warning "Health check failed, checking container logs..."
    az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME
fi

success "Deployment completed successfully!"
echo ""
log "🚀 QuantumRerank is now running on Azure Container Instances"
log "   Resource Group: $RESOURCE_GROUP"
log "   Container: $CONTAINER_NAME"
log "   FQDN: $CONTAINER_FQDN"
log "   IP Address: $CONTAINER_IP"
log "   Health check: http://$CONTAINER_FQDN:8000/health"
log "   API endpoint: http://$CONTAINER_FQDN:8000/v1/rerank"
log "   API Key: $API_KEY"
echo ""
log "Monitor deployment:"
log "   az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
log "   az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
echo ""
log "Test the API:"
echo "curl -X POST http://$CONTAINER_FQDN:8000/v1/rerank \\"
echo "  -H 'Authorization: Bearer $API_KEY' \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"query\": \"test\", \"candidates\": [\"doc1\", \"doc2\"], \"method\": \"hybrid\"}'"
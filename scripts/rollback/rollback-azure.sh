#!/bin/bash
# Azure Container Instances rollback script for QuantumRerank
# Usage: ./rollback-azure.sh [environment] [image_tag]

set -e

# Configuration
ENVIRONMENT="${1:-production}"
IMAGE_TAG="${2:-}"
RESOURCE_GROUP="quantum-rerank-${ENVIRONMENT}-rg"
CONTAINER_NAME="quantum-rerank-${ENVIRONMENT}"
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

if ! az account show &> /dev/null; then
    error "Not authenticated with Azure. Run 'az login' first."
fi

SUBSCRIPTION_ID=$(az account show --query id --output tsv)

log "🔄 Rolling back QuantumRerank on Azure Container Instances"
log "   Environment: $ENVIRONMENT"
log "   Subscription: $SUBSCRIPTION_ID"
log "   Resource Group: $RESOURCE_GROUP"
log "   Container: $CONTAINER_NAME"

# Check if container exists
if ! az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME &> /dev/null; then
    error "Container $CONTAINER_NAME not found in resource group $RESOURCE_GROUP"
fi

# Get current container image
CURRENT_IMAGE=$(az container show \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --query 'containers[0].image' \
    --output tsv)

log "Current image: $CURRENT_IMAGE"

# If no image tag specified, try to find previous version
if [ -z "$IMAGE_TAG" ]; then
    log "No specific image tag provided. Available options:"
    echo "1. quantumrerank/server:latest"
    echo "2. quantumrerank/server:stable"
    echo "3. Custom image tag"
    echo ""
    read -p "Enter image tag to rollback to (or press Enter for 'stable'): " -r
    
    if [ -z "$REPLY" ]; then
        IMAGE_TAG="stable"
    else
        IMAGE_TAG="$REPLY"
    fi
fi

ROLLBACK_IMAGE="quantumrerank/server:$IMAGE_TAG"

# Confirm rollback
echo ""
warning "⚠️  ROLLBACK CONFIRMATION"
log "Current image: $CURRENT_IMAGE"
log "Rolling back to: $ROLLBACK_IMAGE"
echo ""
read -p "Are you sure you want to proceed with rollback? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    log "Rollback cancelled"
    exit 0
fi

# Stop current container
log "Stopping current container..."
az container stop \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_NAME

# Delete current container
log "Deleting current container..."
az container delete \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --yes

# Generate new DNS label suffix
DNS_SUFFIX=$(openssl rand -hex 4)
DNS_NAME="quantum-rerank-${ENVIRONMENT}-${DNS_SUFFIX}"

# Create new container with rollback image
log "Creating new container with rollback image..."
az container create \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --image $ROLLBACK_IMAGE \
    --cpu 2 \
    --memory 2 \
    --restart-policy Always \
    --ports 8000 \
    --environment-variables \
        QUANTUM_RERANK_API_KEY=$API_KEY \
        ENVIRONMENT=$ENVIRONMENT \
    --dns-name-label $DNS_NAME \
    --output none

success "Rollback container created"

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

# Test rollback
log "Testing rolled back service..."
sleep 30

HEALTH_URL="http://${CONTAINER_FQDN}:8000/health"
if curl -f -s --max-time 10 "$HEALTH_URL" > /dev/null; then
    success "Health check passed after rollback"
else
    warning "Health check failed after rollback"
    log "Checking container logs..."
    az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME
fi

echo ""
success "🎉 Rollback completed!"
log "   Rolled back from: $CURRENT_IMAGE"
log "   Rolled back to: $ROLLBACK_IMAGE"
log "   FQDN: $CONTAINER_FQDN"
log "   IP Address: $CONTAINER_IP"
log "   Health endpoint: http://$CONTAINER_FQDN:8000/health"
echo ""
log "Monitor the rolled back deployment:"
log "   az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
log "   az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
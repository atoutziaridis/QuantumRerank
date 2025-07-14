#!/bin/bash
# Kubernetes rollback script for QuantumRerank
# Usage: ./rollback-k8s.sh [environment] [revision_number]

set -e

# Configuration
ENVIRONMENT="${1:-production}"
REVISION="${2:-}"
NAMESPACE="quantum-rerank-${ENVIRONMENT}"
DEPLOYMENT_NAME="quantum-rerank"

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
if ! command -v kubectl &> /dev/null; then
    error "kubectl is not installed. Please install it first."
fi

if ! kubectl cluster-info &> /dev/null; then
    error "Cannot connect to Kubernetes cluster. Check your kubeconfig."
fi

log "🔄 Rolling back QuantumRerank on Kubernetes"
log "   Environment: $ENVIRONMENT"
log "   Namespace: $NAMESPACE"
log "   Deployment: $DEPLOYMENT_NAME"

# Check if namespace exists
if ! kubectl get namespace $NAMESPACE &> /dev/null; then
    error "Namespace $NAMESPACE not found"
fi

# Check if deployment exists
if ! kubectl get deployment $DEPLOYMENT_NAME -n $NAMESPACE &> /dev/null; then
    error "Deployment $DEPLOYMENT_NAME not found in namespace $NAMESPACE"
fi

# Get current deployment status
CURRENT_IMAGE=$(kubectl get deployment $DEPLOYMENT_NAME -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].image}')
log "Current image: $CURRENT_IMAGE"

# Show rollout history
log "Deployment rollout history:"
kubectl rollout history deployment/$DEPLOYMENT_NAME -n $NAMESPACE

# If no revision specified, rollback to previous
if [ -z "$REVISION" ]; then
    log "Rolling back to previous revision..."
    
    # Confirm rollback
    echo ""
    warning "⚠️  ROLLBACK CONFIRMATION"
    log "Rolling back deployment $DEPLOYMENT_NAME to previous revision"
    echo ""
    read -p "Are you sure you want to proceed with rollback? (yes/no): " -r
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log "Rollback cancelled"
        exit 0
    fi
    
    # Perform rollback to previous revision
    kubectl rollout undo deployment/$DEPLOYMENT_NAME -n $NAMESPACE
else
    log "Rolling back to revision $REVISION..."
    
    # Confirm rollback
    echo ""
    warning "⚠️  ROLLBACK CONFIRMATION"
    log "Rolling back deployment $DEPLOYMENT_NAME to revision $REVISION"
    echo ""
    read -p "Are you sure you want to proceed with rollback? (yes/no): " -r
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log "Rollback cancelled"
        exit 0
    fi
    
    # Perform rollback to specific revision
    kubectl rollout undo deployment/$DEPLOYMENT_NAME --to-revision=$REVISION -n $NAMESPACE
fi

success "Rollback initiated"

# Wait for rollout to complete
log "Waiting for rollback to complete..."
kubectl rollout status deployment/$DEPLOYMENT_NAME -n $NAMESPACE --timeout=300s

# Verify rollback
NEW_IMAGE=$(kubectl get deployment $DEPLOYMENT_NAME -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].image}')

success "Rollback completed successfully!"
log "   Rolled back from: $CURRENT_IMAGE"
log "   Rolled back to: $NEW_IMAGE"

# Wait for pods to be ready
log "Waiting for pods to be ready..."
kubectl wait --for=condition=ready --timeout=300s pod -l app=quantum-rerank -n $NAMESPACE

# Get service endpoint
SERVICE_TYPE=$(kubectl get service quantum-rerank-service -n $NAMESPACE -o jsonpath='{.spec.type}')

if [ "$SERVICE_TYPE" = "LoadBalancer" ]; then
    # Get LoadBalancer IP
    EXTERNAL_IP=$(kubectl get service quantum-rerank-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    if [ -z "$EXTERNAL_IP" ] || [ "$EXTERNAL_IP" = "<pending>" ]; then
        EXTERNAL_IP=$(kubectl get service quantum-rerank-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
    fi
fi

# Get node port if LoadBalancer IP is not available
if [ -z "$EXTERNAL_IP" ] || [ "$EXTERNAL_IP" = "<pending>" ]; then
    NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}' 2>/dev/null || kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
    NODE_PORT=$(kubectl get service quantum-rerank-service -n $NAMESPACE -o jsonpath='{.spec.ports[0].nodePort}')
    ENDPOINT="http://$NODE_IP:$NODE_PORT"
else
    ENDPOINT="http://$EXTERNAL_IP"
fi

# Test rollback
if [ -n "$ENDPOINT" ]; then
    log "Testing rolled back service..."
    sleep 15
    
    if curl -f -s --max-time 10 "$ENDPOINT/health" > /dev/null; then
        success "Health check passed after rollback"
    else
        warning "Health check failed after rollback"
        log "Checking pod status..."
        kubectl get pods -n $NAMESPACE -l app=quantum-rerank
    fi
fi

echo ""
success "🎉 Rollback completed!"
log "   Namespace: $NAMESPACE"
log "   Deployment: $DEPLOYMENT_NAME"
if [ -n "$EXTERNAL_IP" ] && [ "$EXTERNAL_IP" != "<pending>" ]; then
    log "   External IP: $EXTERNAL_IP"
    log "   Health endpoint: http://$EXTERNAL_IP/health"
elif [ -n "$NODE_IP" ] && [ -n "$NODE_PORT" ]; then
    log "   Node IP: $NODE_IP"
    log "   Node Port: $NODE_PORT"
    log "   Health endpoint: http://$NODE_IP:$NODE_PORT/health"
fi
echo ""
log "Monitor the rolled back deployment:"
log "   kubectl get all -n $NAMESPACE"
log "   kubectl logs -f deployment/quantum-rerank -n $NAMESPACE"
log "   kubectl describe deployment quantum-rerank -n $NAMESPACE"
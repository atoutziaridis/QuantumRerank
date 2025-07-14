#!/bin/bash
# Simple Kubernetes deployment for QuantumRerank
# Usage: ./deploy-k8s.sh [environment]

set -e

# Configuration
ENVIRONMENT="${1:-production}"
NAMESPACE="quantum-rerank-${ENVIRONMENT}"
IMAGE="${QUANTUM_RERANK_IMAGE:-quantumrerank/server:latest}"
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

if ! command -v kubectl &> /dev/null; then
    error "kubectl is not installed. Please install it first."
fi

# Check kubectl connection
if ! kubectl cluster-info &> /dev/null; then
    error "Cannot connect to Kubernetes cluster. Check your kubeconfig."
fi

log "Deploying QuantumRerank to Kubernetes..."
log "Environment: $ENVIRONMENT"
log "Namespace: $NAMESPACE"
log "Image: $IMAGE"

# Create namespace
log "Creating namespace..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
success "Namespace ready"

# Create API key secret
log "Creating API key secret..."
kubectl create secret generic quantum-rerank-secret \
    --from-literal=api-key="$API_KEY" \
    --namespace=$NAMESPACE \
    --dry-run=client -o yaml | kubectl apply -f -
success "Secret created"

# Create deployment manifest
log "Creating deployment manifest..."
cat > /tmp/k8s-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quantum-rerank
  namespace: $NAMESPACE
  labels:
    app: quantum-rerank
    environment: $ENVIRONMENT
spec:
  replicas: 3
  selector:
    matchLabels:
      app: quantum-rerank
  template:
    metadata:
      labels:
        app: quantum-rerank
        environment: $ENVIRONMENT
    spec:
      containers:
      - name: quantum-rerank
        image: $IMAGE
        ports:
        - containerPort: 8000
        env:
        - name: QUANTUM_RERANK_API_KEY
          valueFrom:
            secretKeyRef:
              name: quantum-rerank-secret
              key: api-key
        - name: ENVIRONMENT
          value: "$ENVIRONMENT"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
        startupProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 6
---
apiVersion: v1
kind: Service
metadata:
  name: quantum-rerank-service
  namespace: $NAMESPACE
  labels:
    app: quantum-rerank
    environment: $ENVIRONMENT
spec:
  selector:
    app: quantum-rerank
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  type: LoadBalancer
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: quantum-rerank-ingress
  namespace: $NAMESPACE
  labels:
    app: quantum-rerank
    environment: $ENVIRONMENT
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "false"
spec:
  rules:
  - host: quantum-rerank-${ENVIRONMENT}.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: quantum-rerank-service
            port:
              number: 80
EOF

# Apply deployment
log "Applying Kubernetes manifests..."
kubectl apply -f /tmp/k8s-deployment.yaml
success "Manifests applied"

# Wait for deployment to be ready
log "Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/quantum-rerank -n $NAMESPACE

# Wait for pods to be ready
kubectl wait --for=condition=ready --timeout=300s pod -l app=quantum-rerank -n $NAMESPACE

success "Deployment is ready"

# Get service information
log "Getting service information..."
SERVICE_TYPE=$(kubectl get service quantum-rerank-service -n $NAMESPACE -o jsonpath='{.spec.type}')

if [ "$SERVICE_TYPE" = "LoadBalancer" ]; then
    # Wait for LoadBalancer IP
    log "Waiting for LoadBalancer IP..."
    for i in {1..60}; do
        EXTERNAL_IP=$(kubectl get service quantum-rerank-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
        if [ -n "$EXTERNAL_IP" ] && [ "$EXTERNAL_IP" != "<pending>" ]; then
            break
        fi
        if [ $i -eq 60 ]; then
            warning "LoadBalancer IP not assigned, checking for hostname..."
            EXTERNAL_IP=$(kubectl get service quantum-rerank-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
        fi
        sleep 5
    done
fi

# Get node port if LoadBalancer IP is not available
if [ -z "$EXTERNAL_IP" ] || [ "$EXTERNAL_IP" = "<pending>" ]; then
    NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}' 2>/dev/null || kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
    NODE_PORT=$(kubectl get service quantum-rerank-service -n $NAMESPACE -o jsonpath='{.spec.ports[0].nodePort}')
    ENDPOINT="http://$NODE_IP:$NODE_PORT"
else
    ENDPOINT="http://$EXTERNAL_IP"
fi

# Test deployment
log "Testing deployment..."
sleep 10

if curl -f -s "$ENDPOINT/health" > /dev/null; then
    success "Health check passed"
else
    warning "Health check failed, checking pod status..."
    kubectl get pods -n $NAMESPACE -l app=quantum-rerank
fi

# Cleanup temp files
rm -f /tmp/k8s-deployment.yaml

success "Deployment completed successfully!"
echo ""
log "🚀 QuantumRerank is now running on Kubernetes"
log "   Namespace: $NAMESPACE"
log "   Deployment: quantum-rerank"
log "   Service: quantum-rerank-service"
if [ -n "$EXTERNAL_IP" ] && [ "$EXTERNAL_IP" != "<pending>" ]; then
    log "   External IP: $EXTERNAL_IP"
    log "   Health check: http://$EXTERNAL_IP/health"
    log "   API endpoint: http://$EXTERNAL_IP/v1/rerank"
elif [ -n "$NODE_IP" ] && [ -n "$NODE_PORT" ]; then
    log "   Node IP: $NODE_IP"
    log "   Node Port: $NODE_PORT"
    log "   Health check: http://$NODE_IP:$NODE_PORT/health"
    log "   API endpoint: http://$NODE_IP:$NODE_PORT/v1/rerank"
fi
log "   API Key: $API_KEY"
echo ""
log "Monitor deployment:"
log "   kubectl get all -n $NAMESPACE"
log "   kubectl logs -f deployment/quantum-rerank -n $NAMESPACE"
log "   kubectl describe deployment quantum-rerank -n $NAMESPACE"
echo ""
if [ -n "$ENDPOINT" ]; then
    log "Test the API:"
    echo "curl -X POST $ENDPOINT/v1/rerank \\"
    echo "  -H 'Authorization: Bearer $API_KEY' \\"
    echo "  -H 'Content-Type: application/json' \\"
    echo "  -d '{\"query\": \"test\", \"candidates\": [\"doc1\", \"doc2\"], \"method\": \"hybrid\"}'"
fi
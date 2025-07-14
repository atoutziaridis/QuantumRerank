# Task 29: Deployment Automation

## Overview
Create simple, reliable deployment automation that companies can use to deploy QuantumRerank to their infrastructure quickly.

## Objectives
- One-command deployment to major cloud providers
- Simple CI/CD pipeline that works out of the box
- Automated testing before deployment
- Easy rollback if something goes wrong

## Requirements

### Cloud Deployment Scripts

#### AWS Deployment (ECS + ALB)
```bash
#!/bin/bash
# deploy-aws.sh

set -e

# Configuration
CLUSTER_NAME="quantum-rerank"
SERVICE_NAME="quantum-rerank-service"
TASK_FAMILY="quantum-rerank-task"
IMAGE="quantumrerank/server:latest"
REGION="${AWS_REGION:-us-west-2}"

echo "Deploying QuantumRerank to AWS ECS..."

# Create ECS cluster if it doesn't exist
if ! aws ecs describe-clusters --clusters $CLUSTER_NAME --region $REGION &> /dev/null; then
    echo "Creating ECS cluster..."
    aws ecs create-cluster --cluster-name $CLUSTER_NAME --region $REGION
fi

# Create task definition
cat > task-definition.json << EOF
{
    "family": "$TASK_FAMILY",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "1024",
    "memory": "2048",
    "executionRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/ecsTaskExecutionRole",
    "containerDefinitions": [
        {
            "name": "quantum-rerank",
            "image": "$IMAGE",
            "portMappings": [
                {
                    "containerPort": 8000,
                    "protocol": "tcp"
                }
            ],
            "essential": true,
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/quantum-rerank",
                    "awslogs-region": "$REGION",
                    "awslogs-stream-prefix": "ecs"
                }
            },
            "environment": [
                {
                    "name": "QUANTUM_RERANK_API_KEY",
                    "value": "${API_KEY}"
                }
            ],
            "healthCheck": {
                "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
                "interval": 30,
                "timeout": 10,
                "retries": 3,
                "startPeriod": 60
            }
        }
    ]
}
EOF

# Register task definition
echo "Registering task definition..."
aws ecs register-task-definition --cli-input-json file://task-definition.json --region $REGION

# Create or update service
if aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $REGION &> /dev/null; then
    echo "Updating existing service..."
    aws ecs update-service \
        --cluster $CLUSTER_NAME \
        --service $SERVICE_NAME \
        --task-definition $TASK_FAMILY \
        --region $REGION
else
    echo "Creating new service..."
    aws ecs create-service \
        --cluster $CLUSTER_NAME \
        --service-name $SERVICE_NAME \
        --task-definition $TASK_FAMILY \
        --desired-count 2 \
        --launch-type FARGATE \
        --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_IDS],securityGroups=[$SECURITY_GROUP_ID],assignPublicIp=ENABLED}" \
        --region $REGION
fi

# Wait for deployment to complete
echo "Waiting for deployment to stabilize..."
aws ecs wait services-stable --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $REGION

echo "✅ Deployment complete!"
echo "Service URL: http://${LOAD_BALANCER_DNS}/health"
```

#### Google Cloud Deployment (Cloud Run)
```bash
#!/bin/bash
# deploy-gcp.sh

set -e

PROJECT_ID="${GOOGLE_CLOUD_PROJECT}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="quantum-rerank"
IMAGE="gcr.io/${PROJECT_ID}/quantum-rerank:latest"

echo "Deploying QuantumRerank to Google Cloud Run..."

# Build and push image
echo "Building and pushing Docker image..."
docker build -t $IMAGE .
docker push $IMAGE

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --concurrency 100 \
    --max-instances 10 \
    --set-env-vars QUANTUM_RERANK_API_KEY=${API_KEY} \
    --port 8000

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')

echo "✅ Deployment complete!"
echo "Service URL: ${SERVICE_URL}/health"
```

#### Azure Deployment (Container Instances)
```bash
#!/bin/bash
# deploy-azure.sh

set -e

RESOURCE_GROUP="quantum-rerank-rg"
CONTAINER_NAME="quantum-rerank"
IMAGE="quantumrerank/server:latest"
LOCATION="${AZURE_LOCATION:-eastus}"

echo "Deploying QuantumRerank to Azure Container Instances..."

# Create resource group if it doesn't exist
if ! az group show --name $RESOURCE_GROUP &> /dev/null; then
    echo "Creating resource group..."
    az group create --name $RESOURCE_GROUP --location $LOCATION
fi

# Deploy container
echo "Creating container instance..."
az container create \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --image $IMAGE \
    --cpu 2 \
    --memory 2 \
    --restart-policy Always \
    --ports 8000 \
    --environment-variables QUANTUM_RERANK_API_KEY=${API_KEY} \
    --dns-name-label quantum-rerank-${RANDOM}

# Get container IP
CONTAINER_IP=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query ipAddress.ip --output tsv)

echo "✅ Deployment complete!"
echo "Service URL: http://${CONTAINER_IP}:8000/health"
```

### Kubernetes Deployment (Simple)
```yaml
# k8s-deploy.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quantum-rerank
  labels:
    app: quantum-rerank
spec:
  replicas: 3
  selector:
    matchLabels:
      app: quantum-rerank
  template:
    metadata:
      labels:
        app: quantum-rerank
    spec:
      containers:
      - name: quantum-rerank
        image: quantumrerank/server:latest
        ports:
        - containerPort: 8000
        env:
        - name: QUANTUM_RERANK_API_KEY
          valueFrom:
            secretKeyRef:
              name: quantum-rerank-secret
              key: api-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: quantum-rerank-service
spec:
  selector:
    app: quantum-rerank
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: v1
kind: Secret
metadata:
  name: quantum-rerank-secret
type: Opaque
data:
  api-key: cXJfY2hhbmdlLXRoaXMta2V5  # base64 encoded API key
```

```bash
#!/bin/bash
# deploy-k8s.sh

set -e

echo "Deploying QuantumRerank to Kubernetes..."

# Apply the deployment
kubectl apply -f k8s-deploy.yaml

# Wait for deployment to be ready
echo "Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/quantum-rerank

# Get service URL
SERVICE_IP=$(kubectl get service quantum-rerank-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

echo "✅ Deployment complete!"
echo "Service URL: http://${SERVICE_IP}/health"
```

### GitHub Actions CI/CD
```yaml
# .github/workflows/deploy.yml
name: Deploy QuantumRerank

on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: quantumrerank/server

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest tests/ -v
        
    - name: Run linting
      run: |
        flake8 quantum_rerank/
        
    - name: Test Docker build
      run: |
        docker build -t test-image .
        docker run --rm test-image python -c "import quantum_rerank; print('Import successful')"

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=raw,value=latest,enable={{is_default_branch}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: staging
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to staging
      run: |
        # Deploy to staging environment
        echo "Deploying to staging..."
        # Add your staging deployment commands here
    
    - name: Run smoke tests
      run: |
        # Wait for deployment
        sleep 60
        
        # Test health endpoint
        curl -f ${{ secrets.STAGING_URL }}/health
        
        # Test API endpoint
        curl -f -X POST ${{ secrets.STAGING_URL }}/v1/rerank \
          -H "Authorization: Bearer ${{ secrets.STAGING_API_KEY }}" \
          -H "Content-Type: application/json" \
          -d '{"query": "test", "documents": ["doc1", "doc2"]}'

  deploy-production:
    needs: [build-and-push, deploy-staging]
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')
    environment: production
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to production
      run: |
        echo "Deploying to production..."
        # Add your production deployment commands here
    
    - name: Verify production deployment
      run: |
        sleep 60
        curl -f ${{ secrets.PRODUCTION_URL }}/health
```

### Universal Deployment Script
```bash
#!/bin/bash
# universal-deploy.sh

set -e

PLATFORM=""
ENVIRONMENT="production"

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
            echo "Usage: $0 --platform [aws|gcp|azure|k8s|docker] --environment [production|staging]"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Auto-detect platform if not specified
if [ -z "$PLATFORM" ]; then
    if command -v aws &> /dev/null && [ -n "$AWS_ACCOUNT_ID" ]; then
        PLATFORM="aws"
    elif command -v gcloud &> /dev/null && [ -n "$GOOGLE_CLOUD_PROJECT" ]; then
        PLATFORM="gcp"
    elif command -v az &> /dev/null; then
        PLATFORM="azure"
    elif command -v kubectl &> /dev/null; then
        PLATFORM="k8s"
    elif command -v docker &> /dev/null; then
        PLATFORM="docker"
    else
        echo "No supported platform detected. Please specify --platform"
        exit 1
    fi
fi

echo "Deploying QuantumRerank to $PLATFORM ($ENVIRONMENT environment)"

# Validate required environment variables
if [ -z "$API_KEY" ]; then
    echo "Error: API_KEY environment variable is required"
    exit 1
fi

# Platform-specific deployment
case $PLATFORM in
    aws)
        ./scripts/deploy-aws.sh
        ;;
    gcp)
        ./scripts/deploy-gcp.sh
        ;;
    azure)
        ./scripts/deploy-azure.sh
        ;;
    k8s)
        ./scripts/deploy-k8s.sh
        ;;
    docker)
        docker-compose -f docker-compose.prod.yml up -d
        ;;
    *)
        echo "Unsupported platform: $PLATFORM"
        exit 1
        ;;
esac

echo "✅ Deployment complete!"
```

### Rollback Script
```bash
#!/bin/bash
# rollback.sh

set -e

PLATFORM="$1"
VERSION="$2"

if [ -z "$PLATFORM" ] || [ -z "$VERSION" ]; then
    echo "Usage: $0 <platform> <version>"
    echo "Example: $0 aws v1.2.3"
    exit 1
fi

echo "Rolling back QuantumRerank on $PLATFORM to version $VERSION"

case $PLATFORM in
    aws)
        # Rollback ECS service to previous task definition
        aws ecs update-service \
            --cluster quantum-rerank \
            --service quantum-rerank-service \
            --task-definition quantum-rerank-task:$VERSION
        ;;
    gcp)
        # Rollback Cloud Run to specific revision
        gcloud run services update-traffic quantum-rerank \
            --to-revisions=$VERSION=100 \
            --region us-central1
        ;;
    k8s)
        # Rollback Kubernetes deployment
        kubectl rollout undo deployment/quantum-rerank --to-revision=$VERSION
        ;;
    docker)
        # Rollback Docker Compose
        export IMAGE_TAG=$VERSION
        docker-compose -f docker-compose.prod.yml up -d
        ;;
esac

echo "✅ Rollback complete!"
```

## Testing Automation

### Deployment Test Script
```bash
#!/bin/bash
# test-deployment.sh

ENDPOINT="$1"
API_KEY="$2"

if [ -z "$ENDPOINT" ] || [ -z "$API_KEY" ]; then
    echo "Usage: $0 <endpoint> <api_key>"
    exit 1
fi

echo "Testing deployment at $ENDPOINT"

# Test health endpoint
echo "Testing health endpoint..."
if ! curl -f "$ENDPOINT/health"; then
    echo "❌ Health check failed"
    exit 1
fi

# Test API endpoint
echo "Testing API endpoint..."
RESPONSE=$(curl -s -X POST "$ENDPOINT/v1/rerank" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "query": "test query",
        "documents": ["document 1", "document 2"],
        "method": "classical"
    }')

if echo "$RESPONSE" | grep -q "documents"; then
    echo "✅ API test passed"
else
    echo "❌ API test failed"
    echo "Response: $RESPONSE"
    exit 1
fi

echo "✅ All deployment tests passed!"
```

## Success Criteria
- [ ] One-command deployment to AWS, GCP, Azure
- [ ] Kubernetes deployment with single YAML file
- [ ] Automated CI/CD pipeline with testing
- [ ] Automated rollback capability
- [ ] Deployment verification tests
- [ ] Works without manual configuration

## Timeline
- **Week 1**: Cloud provider deployment scripts
- **Week 2**: Kubernetes deployment and universal script
- **Week 3**: CI/CD pipeline setup
- **Week 4**: Testing and rollback automation

This provides simple, reliable deployment automation that companies can use immediately.
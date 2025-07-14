#!/bin/bash
# Simple AWS ECS deployment for QuantumRerank
# Usage: ./deploy-aws.sh [environment]

set -e

# Configuration
ENVIRONMENT="${1:-production}"
CLUSTER_NAME="quantum-rerank-${ENVIRONMENT}"
SERVICE_NAME="quantum-rerank-service"
TASK_FAMILY="quantum-rerank-task"
IMAGE="${QUANTUM_RERANK_IMAGE:-quantumrerank/server:latest}"
REGION="${AWS_REGION:-us-west-2}"
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

if ! command -v aws &> /dev/null; then
    error "AWS CLI is not installed. Please install it first."
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    error "AWS credentials not configured. Run 'aws configure' first."
fi

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

log "Deploying QuantumRerank to AWS ECS..."
log "Environment: $ENVIRONMENT"
log "Cluster: $CLUSTER_NAME"
log "Image: $IMAGE"
log "Region: $REGION"

# Create ECS cluster if it doesn't exist
log "Checking/creating ECS cluster..."
if ! aws ecs describe-clusters --clusters $CLUSTER_NAME --region $REGION &> /dev/null; then
    log "Creating ECS cluster: $CLUSTER_NAME"
    aws ecs create-cluster --cluster-name $CLUSTER_NAME --region $REGION
    success "ECS cluster created"
else
    success "ECS cluster already exists"
fi

# Create CloudWatch log group
log "Creating CloudWatch log group..."
aws logs create-log-group --log-group-name "/ecs/quantum-rerank" --region $REGION 2>/dev/null || true

# Create task definition
log "Creating task definition..."
cat > /tmp/task-definition.json << EOF
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
                    "value": "$API_KEY"
                },
                {
                    "name": "ENVIRONMENT",
                    "value": "$ENVIRONMENT"
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
log "Registering task definition..."
TASK_DEFINITION_ARN=$(aws ecs register-task-definition \
    --cli-input-json file:///tmp/task-definition.json \
    --region $REGION \
    --query 'taskDefinition.taskDefinitionArn' \
    --output text)

success "Task definition registered: $TASK_DEFINITION_ARN"

# Get default VPC and subnets
log "Getting VPC configuration..."
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query 'Vpcs[0].VpcId' --output text --region $REGION)
SUBNET_IDS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query 'Subnets[0:2].SubnetId' --output text --region $REGION | tr '\t' ',')

# Create security group
log "Creating security group..."
SECURITY_GROUP_ID=$(aws ec2 create-security-group \
    --group-name "quantum-rerank-sg-$ENVIRONMENT" \
    --description "Security group for QuantumRerank $ENVIRONMENT" \
    --vpc-id $VPC_ID \
    --region $REGION \
    --query 'GroupId' \
    --output text 2>/dev/null || \
    aws ec2 describe-security-groups \
    --group-names "quantum-rerank-sg-$ENVIRONMENT" \
    --query 'SecurityGroups[0].GroupId' \
    --output text \
    --region $REGION)

# Add security group rules
aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 8000 \
    --cidr 0.0.0.0/0 \
    --region $REGION 2>/dev/null || true

success "Security group configured: $SECURITY_GROUP_ID"

# Create or update service
log "Deploying service..."
if aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $REGION &> /dev/null; then
    log "Updating existing service..."
    aws ecs update-service \
        --cluster $CLUSTER_NAME \
        --service $SERVICE_NAME \
        --task-definition $TASK_FAMILY \
        --region $REGION \
        --force-new-deployment
else
    log "Creating new service..."
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
log "Waiting for deployment to stabilize..."
aws ecs wait services-stable --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $REGION

# Get service endpoint
TASK_ARN=$(aws ecs list-tasks --cluster $CLUSTER_NAME --service-name $SERVICE_NAME --region $REGION --query 'taskArns[0]' --output text)
if [ "$TASK_ARN" != "None" ]; then
    PUBLIC_IP=$(aws ecs describe-tasks --cluster $CLUSTER_NAME --tasks $TASK_ARN --region $REGION --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value | [0]' --output text)
    if [ "$PUBLIC_IP" != "None" ]; then
        ENI_ID=$PUBLIC_IP
        PUBLIC_IP=$(aws ec2 describe-network-interfaces --network-interface-ids $ENI_ID --query 'NetworkInterfaces[0].Association.PublicIp' --output text --region $REGION)
    fi
fi

# Cleanup temp files
rm -f /tmp/task-definition.json

success "Deployment completed successfully!"
echo ""
log "🚀 QuantumRerank is now running on AWS ECS"
log "   Cluster: $CLUSTER_NAME"
log "   Service: $SERVICE_NAME"
if [ "$PUBLIC_IP" != "None" ] && [ -n "$PUBLIC_IP" ]; then
    log "   Health check: http://$PUBLIC_IP:8000/health"
    log "   API endpoint: http://$PUBLIC_IP:8000/v1/rerank"
else
    log "   Check AWS Console for service endpoint"
fi
log "   API Key: $API_KEY"
echo ""
log "Monitor deployment:"
log "   aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $REGION"
log "   aws logs tail /ecs/quantum-rerank --follow --region $REGION"
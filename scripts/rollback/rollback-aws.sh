#!/bin/bash
# AWS ECS rollback script for QuantumRerank
# Usage: ./rollback-aws.sh [environment] [revision_number]

set -e

# Configuration
ENVIRONMENT="${1:-production}"
REVISION="${2:-}"
CLUSTER_NAME="quantum-rerank-${ENVIRONMENT}"
SERVICE_NAME="quantum-rerank-service"
TASK_FAMILY="quantum-rerank-task"
REGION="${AWS_REGION:-us-west-2}"

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
if ! command -v aws &> /dev/null; then
    error "AWS CLI is not installed. Please install it first."
fi

if ! aws sts get-caller-identity &> /dev/null; then
    error "AWS credentials not configured. Run 'aws configure' first."
fi

log "🔄 Rolling back QuantumRerank on AWS ECS"
log "   Environment: $ENVIRONMENT"
log "   Cluster: $CLUSTER_NAME"
log "   Service: $SERVICE_NAME"
log "   Region: $REGION"

# Check if service exists
if ! aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $REGION &> /dev/null; then
    error "Service $SERVICE_NAME not found in cluster $CLUSTER_NAME"
fi

# Get current task definition
CURRENT_TASK_DEF=$(aws ecs describe-services \
    --cluster $CLUSTER_NAME \
    --services $SERVICE_NAME \
    --region $REGION \
    --query 'services[0].taskDefinition' \
    --output text)

log "Current task definition: $CURRENT_TASK_DEF"

# List available task definition revisions
log "Available task definition revisions:"
aws ecs list-task-definitions \
    --family-prefix $TASK_FAMILY \
    --region $REGION \
    --query 'taskDefinitionArns[]' \
    --output table

# If no revision specified, prompt for previous revision
if [ -z "$REVISION" ]; then
    log "Getting previous task definition revision..."
    
    # Get all task definitions for this family
    TASK_DEFINITIONS=($(aws ecs list-task-definitions \
        --family-prefix $TASK_FAMILY \
        --region $REGION \
        --query 'taskDefinitionArns[]' \
        --output text | sort -V))
    
    # Find current index and get previous
    CURRENT_INDEX=-1
    for i in "${!TASK_DEFINITIONS[@]}"; do
        if [[ "${TASK_DEFINITIONS[$i]}" == *"$CURRENT_TASK_DEF"* ]]; then
            CURRENT_INDEX=$i
            break
        fi
    done
    
    if [ $CURRENT_INDEX -le 0 ]; then
        error "No previous revision found for rollback"
    fi
    
    PREVIOUS_INDEX=$((CURRENT_INDEX - 1))
    ROLLBACK_TASK_DEF="${TASK_DEFINITIONS[$PREVIOUS_INDEX]}"
    
    # Extract revision number
    REVISION=$(echo "$ROLLBACK_TASK_DEF" | grep -o '[0-9]*$')
    
    log "Auto-selected previous revision: $REVISION"
    log "Rolling back to: $ROLLBACK_TASK_DEF"
else
    ROLLBACK_TASK_DEF="${TASK_FAMILY}:${REVISION}"
    log "Rolling back to specified revision: $ROLLBACK_TASK_DEF"
    
    # Verify the task definition exists
    if ! aws ecs describe-task-definition \
        --task-definition "$ROLLBACK_TASK_DEF" \
        --region $REGION &> /dev/null; then
        error "Task definition $ROLLBACK_TASK_DEF not found"
    fi
fi

# Confirm rollback
echo ""
warning "⚠️  ROLLBACK CONFIRMATION"
log "Current deployment: $CURRENT_TASK_DEF"
log "Rolling back to: $ROLLBACK_TASK_DEF"
echo ""
read -p "Are you sure you want to proceed with rollback? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    log "Rollback cancelled"
    exit 0
fi

# Perform rollback
log "Starting rollback deployment..."
aws ecs update-service \
    --cluster $CLUSTER_NAME \
    --service $SERVICE_NAME \
    --task-definition "$ROLLBACK_TASK_DEF" \
    --region $REGION \
    --force-new-deployment

success "Rollback initiated"

# Wait for deployment to stabilize
log "Waiting for rollback to complete..."
aws ecs wait services-stable \
    --cluster $CLUSTER_NAME \
    --services $SERVICE_NAME \
    --region $REGION

# Verify rollback
NEW_TASK_DEF=$(aws ecs describe-services \
    --cluster $CLUSTER_NAME \
    --services $SERVICE_NAME \
    --region $REGION \
    --query 'services[0].taskDefinition' \
    --output text)

if [[ "$NEW_TASK_DEF" == *"$ROLLBACK_TASK_DEF"* ]]; then
    success "Rollback completed successfully!"
else
    error "Rollback verification failed. Current: $NEW_TASK_DEF, Expected: $ROLLBACK_TASK_DEF"
fi

# Get service endpoint for testing
TASK_ARN=$(aws ecs list-tasks --cluster $CLUSTER_NAME --service-name $SERVICE_NAME --region $REGION --query 'taskArns[0]' --output text)
if [ "$TASK_ARN" != "None" ]; then
    PUBLIC_IP=$(aws ecs describe-tasks --cluster $CLUSTER_NAME --tasks $TASK_ARN --region $REGION --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value | [0]' --output text)
    if [ "$PUBLIC_IP" != "None" ]; then
        ENI_ID=$PUBLIC_IP
        PUBLIC_IP=$(aws ec2 describe-network-interfaces --network-interface-ids $ENI_ID --query 'NetworkInterfaces[0].Association.PublicIp' --output text --region $REGION)
    fi
fi

# Test rollback
if [ "$PUBLIC_IP" != "None" ] && [ -n "$PUBLIC_IP" ]; then
    log "Testing rolled back service..."
    sleep 30
    
    if curl -f -s --max-time 10 "http://$PUBLIC_IP:8000/health" > /dev/null; then
        success "Health check passed after rollback"
    else
        warning "Health check failed after rollback"
    fi
fi

echo ""
success "🎉 Rollback completed!"
log "   Rolled back from: $CURRENT_TASK_DEF"
log "   Rolled back to: $NEW_TASK_DEF"
if [ "$PUBLIC_IP" != "None" ] && [ -n "$PUBLIC_IP" ]; then
    log "   Health endpoint: http://$PUBLIC_IP:8000/health"
fi
echo ""
log "Monitor the rolled back deployment:"
log "   aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $REGION"
log "   aws logs tail /ecs/quantum-rerank --follow --region $REGION"
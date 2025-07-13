# AWS Production Deployment Guide for QuantumRerank

## Overview

This guide provides detailed procedures for deploying QuantumRerank on Amazon Web Services (AWS) using Amazon EKS (Elastic Kubernetes Service) and supporting AWS services.

## Prerequisites

### AWS Account Setup
- AWS CLI v2.x installed and configured
- kubectl v1.24+ installed
- eksctl v0.140+ installed
- Helm v3.x installed
- Docker installed for local testing

### Required Permissions
- EKS cluster creation and management
- EC2 instance management
- VPC and networking configuration
- IAM role and policy management
- ECR repository access
- Route53 DNS management
- ACM certificate management

## Infrastructure Setup

### 1. Network Infrastructure

**Create VPC and Subnets**
```bash
# Create dedicated VPC for production
aws ec2 create-vpc \
  --cidr-block 10.0.0.0/16 \
  --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=quantum-rerank-prod-vpc}]'

# Export VPC ID
export VPC_ID=$(aws ec2 describe-vpcs \
  --filters "Name=tag:Name,Values=quantum-rerank-prod-vpc" \
  --query 'Vpcs[0].VpcId' --output text)

# Create public subnets
aws ec2 create-subnet \
  --vpc-id $VPC_ID \
  --cidr-block 10.0.1.0/24 \
  --availability-zone us-west-2a \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=quantum-rerank-public-1}]'

aws ec2 create-subnet \
  --vpc-id $VPC_ID \
  --cidr-block 10.0.2.0/24 \
  --availability-zone us-west-2b \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=quantum-rerank-public-2}]'

# Create private subnets
aws ec2 create-subnet \
  --vpc-id $VPC_ID \
  --cidr-block 10.0.10.0/24 \
  --availability-zone us-west-2a \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=quantum-rerank-private-1}]'

aws ec2 create-subnet \
  --vpc-id $VPC_ID \
  --cidr-block 10.0.11.0/24 \
  --availability-zone us-west-2b \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=quantum-rerank-private-2}]'
```

**Configure Internet Gateway and NAT**
```bash
# Create and attach Internet Gateway
aws ec2 create-internet-gateway \
  --tag-specifications 'ResourceType=internet-gateway,Tags=[{Key=Name,Value=quantum-rerank-igw}]'

export IGW_ID=$(aws ec2 describe-internet-gateways \
  --filters "Name=tag:Name,Values=quantum-rerank-igw" \
  --query 'InternetGateways[0].InternetGatewayId' --output text)

aws ec2 attach-internet-gateway \
  --internet-gateway-id $IGW_ID \
  --vpc-id $VPC_ID

# Create NAT Gateway for private subnets
aws ec2 allocate-address --domain vpc
export EIP_ALLOC_ID=$(aws ec2 describe-addresses \
  --query 'Addresses[?Domain==`vpc`][AllocationId]' --output text | head -n1)

export PUBLIC_SUBNET_1=$(aws ec2 describe-subnets \
  --filters "Name=tag:Name,Values=quantum-rerank-public-1" \
  --query 'Subnets[0].SubnetId' --output text)

aws ec2 create-nat-gateway \
  --subnet-id $PUBLIC_SUBNET_1 \
  --allocation-id $EIP_ALLOC_ID \
  --tag-specifications 'ResourceType=nat-gateway,Tags=[{Key=Name,Value=quantum-rerank-nat}]'
```

### 2. EKS Cluster Setup

**Create EKS Cluster**
```bash
# Create cluster configuration
cat > cluster-config.yaml << EOF
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: quantum-rerank-prod
  region: us-west-2
  version: "1.28"

vpc:
  id: "$VPC_ID"
  subnets:
    public:
      us-west-2a: { id: "$PUBLIC_SUBNET_1" }
      us-west-2b: { id: "$PUBLIC_SUBNET_2" }
    private:
      us-west-2a: { id: "$PRIVATE_SUBNET_1" }
      us-west-2b: { id: "$PRIVATE_SUBNET_2" }

nodeGroups:
  - name: quantum-rerank-workers
    instanceType: m5.xlarge
    desiredCapacity: 3
    minSize: 3
    maxSize: 10
    privateNetworking: true
    ssh:
      allow: false
    iam:
      withAddonPolicies:
        autoScaler: true
        cloudWatch: true
        ebs: true
        efs: true
        albIngress: true
    labels:
      environment: production
      application: quantum-rerank
    tags:
      k8s.io/cluster-autoscaler/enabled: "true"
      k8s.io/cluster-autoscaler/quantum-rerank-prod: "owned"

addons:
  - name: vpc-cni
    version: latest
  - name: coredns
    version: latest
  - name: kube-proxy
    version: latest
  - name: aws-ebs-csi-driver
    version: latest

cloudWatch:
  clusterLogging:
    enable: ["api", "audit", "authenticator", "controllerManager", "scheduler"]
EOF

# Create the cluster
eksctl create cluster -f cluster-config.yaml
```

**Configure kubectl**
```bash
# Update kubeconfig
aws eks update-kubeconfig \
  --region us-west-2 \
  --name quantum-rerank-prod

# Verify cluster access
kubectl get nodes
kubectl get pods --all-namespaces
```

### 3. AWS Load Balancer Controller

**Install AWS Load Balancer Controller**
```bash
# Create IAM service account
eksctl create iamserviceaccount \
  --cluster=quantum-rerank-prod \
  --namespace=kube-system \
  --name=aws-load-balancer-controller \
  --role-name AmazonEKSLoadBalancerControllerRole \
  --attach-policy-arn=arn:aws:iam::aws:policy/ElasticLoadBalancingFullAccess \
  --override-existing-serviceaccounts \
  --approve

# Install controller
helm repo add eks https://aws.github.io/eks-charts
helm repo update

helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=quantum-rerank-prod \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller
```

## Application Deployment

### 1. Container Registry Setup

**Create ECR Repository**
```bash
# Create ECR repository
aws ecr create-repository \
  --repository-name quantum-rerank \
  --image-tag-mutability MUTABLE \
  --image-scanning-configuration scanOnPush=true

# Get repository URI
export ECR_REGISTRY=$(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-west-2.amazonaws.com
export IMAGE_URI=$ECR_REGISTRY/quantum-rerank:$(git describe --tags --always)

# Login to ECR
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin $ECR_REGISTRY
```

**Build and Push Image**
```bash
# Build production image
docker build \
  --target production \
  --build-arg ENVIRONMENT=production \
  --build-arg VERSION=$(git describe --tags --always) \
  -f deployment/Dockerfile \
  -t quantum-rerank:$(git describe --tags --always) .

# Tag and push to ECR
docker tag quantum-rerank:$(git describe --tags --always) $IMAGE_URI
docker push $IMAGE_URI
```

### 2. AWS Services Integration

**ElastiCache for Redis**
```bash
# Create Redis subnet group
aws elasticache create-cache-subnet-group \
  --cache-subnet-group-name quantum-rerank-redis-subnet \
  --cache-subnet-group-description "Subnet group for QuantumRerank Redis" \
  --subnet-ids $PRIVATE_SUBNET_1 $PRIVATE_SUBNET_2

# Create Redis cluster
aws elasticache create-replication-group \
  --replication-group-id quantum-rerank-redis \
  --description "Redis cluster for QuantumRerank" \
  --num-cache-clusters 2 \
  --cache-node-type cache.r6g.large \
  --engine redis \
  --engine-version 7.0 \
  --cache-subnet-group-name quantum-rerank-redis-subnet \
  --security-group-ids $REDIS_SECURITY_GROUP_ID \
  --at-rest-encryption-enabled \
  --transit-encryption-enabled \
  --multi-az-enabled \
  --automatic-failover-enabled
```

**RDS for PostgreSQL (if needed)**
```bash
# Create DB subnet group
aws rds create-db-subnet-group \
  --db-subnet-group-name quantum-rerank-db-subnet \
  --db-subnet-group-description "Subnet group for QuantumRerank database" \
  --subnet-ids $PRIVATE_SUBNET_1 $PRIVATE_SUBNET_2

# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier quantum-rerank-db \
  --db-instance-class db.r6g.large \
  --engine postgres \
  --engine-version 15.4 \
  --master-username quantum_admin \
  --master-user-password $(openssl rand -base64 32) \
  --allocated-storage 100 \
  --storage-type gp3 \
  --storage-encrypted \
  --db-subnet-group-name quantum-rerank-db-subnet \
  --vpc-security-group-ids $DB_SECURITY_GROUP_ID \
  --multi-az \
  --backup-retention-period 7 \
  --deletion-protection
```

### 3. SSL Certificate Management

**Request ACM Certificate**
```bash
# Request certificate
aws acm request-certificate \
  --domain-name api.quantumrerank.com \
  --subject-alternative-names "*.quantumrerank.com" \
  --validation-method DNS \
  --key-algorithm RSA_2048 \
  --tags Key=Name,Value=quantum-rerank-cert

# Get certificate ARN
export CERT_ARN=$(aws acm list-certificates \
  --query 'CertificateSummaryList[?DomainName==`api.quantumrerank.com`].CertificateArn' \
  --output text)
```

### 4. Kubernetes Deployment

**Create Namespace and Secrets**
```bash
# Create namespace
kubectl create namespace quantum-rerank
kubectl label namespace quantum-rerank environment=production

# Create secrets
kubectl create secret generic quantum-rerank-secrets \
  --namespace quantum-rerank \
  --from-literal=redis-password="$(aws elasticache describe-replication-groups \
    --replication-group-id quantum-rerank-redis \
    --query 'ReplicationGroups[0].AuthToken' --output text)" \
  --from-literal=database-password="$(aws ssm get-parameter \
    --name /quantum-rerank/db/password --with-decryption \
    --query 'Parameter.Value' --output text)" \
  --from-literal=jwt-secret="$(openssl rand -base64 64)" \
  --from-literal=api-key-salt="$(openssl rand -base64 32)"
```

**Deploy Application**
```bash
# Create AWS-specific deployment configuration
cat > aws-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quantum-rerank
  namespace: quantum-rerank
  labels:
    app: quantum-rerank
    environment: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: quantum-rerank
  template:
    metadata:
      labels:
        app: quantum-rerank
        environment: production
    spec:
      serviceAccountName: quantum-rerank
      containers:
      - name: quantum-rerank
        image: $IMAGE_URI
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: REDIS_HOST
          value: "quantum-rerank-redis.cache.amazonaws.com"
        - name: REDIS_PORT
          value: "6379"
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: quantum-rerank-secrets
              key: redis-password
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
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: quantum-rerank-service
  namespace: quantum-rerank
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-scheme: "internet-facing"
    service.beta.kubernetes.io/aws-load-balancer-ssl-cert: "$CERT_ARN"
    service.beta.kubernetes.io/aws-load-balancer-ssl-ports: "443"
    service.beta.kubernetes.io/aws-load-balancer-backend-protocol: "http"
spec:
  type: LoadBalancer
  ports:
  - name: https
    port: 443
    targetPort: 8000
    protocol: TCP
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  selector:
    app: quantum-rerank
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: quantum-rerank-ingress
  namespace: quantum-rerank
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/certificate-arn: $CERT_ARN
    alb.ingress.kubernetes.io/ssl-redirect: "443"
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTP": 80}, {"HTTPS":443}]'
spec:
  rules:
  - host: api.quantumrerank.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: quantum-rerank-service
            port:
              number: 8000
EOF

# Apply deployment
kubectl apply -f aws-deployment.yaml
```

## Auto-Scaling Configuration

### 1. Cluster Autoscaler

**Install Cluster Autoscaler**
```bash
# Create service account
kubectl create serviceaccount cluster-autoscaler \
  --namespace kube-system

# Apply RBAC
kubectl apply -f - <<EOF
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: cluster-autoscaler
rules:
- apiGroups: [""]
  resources: ["events", "endpoints"]
  verbs: ["create", "patch"]
- apiGroups: [""]
  resources: ["pods/eviction"]
  verbs: ["create"]
- apiGroups: [""]
  resources: ["pods/status"]
  verbs: ["update"]
- apiGroups: [""]
  resources: ["endpoints"]
  resourceNames: ["cluster-autoscaler"]
  verbs: ["get", "update"]
- apiGroups: [""]
  resources: ["nodes"]
  verbs: ["watch", "list", "get", "update"]
- apiGroups: [""]
  resources: ["pods", "services", "replicationcontrollers", "persistentvolumeclaims", "persistentvolumes"]
  verbs: ["watch", "list", "get"]
- apiGroups: ["extensions"]
  resources: ["replicasets", "daemonsets"]
  verbs: ["watch", "list", "get"]
- apiGroups: ["policy"]
  resources: ["poddisruptionbudgets"]
  verbs: ["watch", "list"]
- apiGroups: ["apps"]
  resources: ["statefulsets", "replicasets", "daemonsets"]
  verbs: ["watch", "list", "get"]
- apiGroups: ["storage.k8s.io"]
  resources: ["storageclasses", "csinodes", "csidrivers", "csistoragecapacities"]
  verbs: ["watch", "list", "get"]
- apiGroups: ["batch", "extensions"]
  resources: ["jobs"]
  verbs: ["get", "list", "watch", "patch"]
- apiGroups: ["coordination.k8s.io"]
  resources: ["leases"]
  verbs: ["create"]
- apiGroups: ["coordination.k8s.io"]
  resourceNames: ["cluster-autoscaler"]
  resources: ["leases"]
  verbs: ["get", "update"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: cluster-autoscaler
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: cluster-autoscaler
subjects:
- kind: ServiceAccount
  name: cluster-autoscaler
  namespace: kube-system
EOF

# Deploy cluster autoscaler
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
  labels:
    app: cluster-autoscaler
spec:
  selector:
    matchLabels:
      app: cluster-autoscaler
  template:
    metadata:
      labels:
        app: cluster-autoscaler
    spec:
      serviceAccountName: cluster-autoscaler
      containers:
      - image: registry.k8s.io/autoscaling/cluster-autoscaler:v1.28.0
        name: cluster-autoscaler
        resources:
          limits:
            cpu: 100m
            memory: 600Mi
          requests:
            cpu: 100m
            memory: 600Mi
        command:
        - ./cluster-autoscaler
        - --v=4
        - --stderrthreshold=info
        - --cloud-provider=aws
        - --skip-nodes-with-local-storage=false
        - --expander=least-waste
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/quantum-rerank-prod
        - --balance-similar-node-groups
        - --skip-nodes-with-system-pods=false
        env:
        - name: AWS_REGION
          value: us-west-2
EOF
```

### 2. Horizontal Pod Autoscaler

**Configure HPA**
```bash
kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: quantum-rerank-hpa
  namespace: quantum-rerank
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: quantum-rerank
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
EOF
```

## Monitoring and Logging

### 1. CloudWatch Integration

**Install CloudWatch Agent**
```bash
# Create CloudWatch namespace
kubectl create namespace amazon-cloudwatch

# Install CloudWatch agent
kubectl apply -f https://raw.githubusercontent.com/aws-samples/amazon-cloudwatch-container-insights/latest/k8s-deployment-manifest-templates/deployment-mode/daemonset/container-insights-monitoring/cloudwatch-namespace.yaml

kubectl apply -f https://raw.githubusercontent.com/aws-samples/amazon-cloudwatch-container-insights/latest/k8s-deployment-manifest-templates/deployment-mode/daemonset/container-insights-monitoring/cwagent/cwagent-serviceaccount.yaml

kubectl apply -f https://raw.githubusercontent.com/aws-samples/amazon-cloudwatch-container-insights/latest/k8s-deployment-manifest-templates/deployment-mode/daemonset/container-insights-monitoring/cwagent/cwagent-configmap.yaml

kubectl apply -f https://raw.githubusercontent.com/aws-samples/amazon-cloudwatch-container-insights/latest/k8s-deployment-manifest-templates/deployment-mode/daemonset/container-insights-monitoring/cwagent/cwagent-daemonset.yaml
```

### 2. Prometheus and Grafana

**Install kube-prometheus-stack**
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.service.type=LoadBalancer \
  --set grafana.service.type=LoadBalancer \
  --set grafana.adminPassword="$(openssl rand -base64 32)"
```

## Backup and Disaster Recovery

### 1. Velero Backup

**Install Velero**
```bash
# Create S3 bucket for backups
aws s3 mb s3://quantum-rerank-backups-$(date +%s)

# Install Velero
velero install \
  --provider aws \
  --plugins velero/velero-plugin-for-aws:v1.8.0 \
  --bucket quantum-rerank-backups-$(date +%s) \
  --backup-location-config region=us-west-2 \
  --snapshot-location-config region=us-west-2 \
  --secret-file ./credentials-velero
```

### 2. Database Backup

**RDS Automated Backups**
```bash
# Configure automated backups
aws rds modify-db-instance \
  --db-instance-identifier quantum-rerank-db \
  --backup-retention-period 30 \
  --preferred-backup-window "03:00-04:00" \
  --preferred-maintenance-window "sun:04:00-sun:05:00"
```

## Security Configuration

### 1. Network Security

**Security Groups**
```bash
# Create security group for EKS nodes
aws ec2 create-security-group \
  --group-name quantum-rerank-nodes-sg \
  --description "Security group for QuantumRerank EKS nodes" \
  --vpc-id $VPC_ID

# Create security group for Redis
aws ec2 create-security-group \
  --group-name quantum-rerank-redis-sg \
  --description "Security group for QuantumRerank Redis" \
  --vpc-id $VPC_ID

# Create security group for database
aws ec2 create-security-group \
  --group-name quantum-rerank-db-sg \
  --description "Security group for QuantumRerank database" \
  --vpc-id $VPC_ID
```

### 2. IAM Roles and Policies

**Create Service Account IAM Role**
```bash
eksctl create iamserviceaccount \
  --name quantum-rerank \
  --namespace quantum-rerank \
  --cluster quantum-rerank-prod \
  --attach-policy-arn arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy \
  --approve \
  --override-existing-serviceaccounts
```

## Validation and Testing

### 1. Deployment Validation

**Run Validation Scripts**
```bash
# Test EKS cluster connectivity
kubectl get nodes
kubectl get pods --all-namespaces

# Run smoke tests
python deployment/validation/smoke_tests.py \
  --base-url https://api.quantumrerank.com \
  --timeout 60

# Run performance validation
python deployment/validation/performance_validation.py \
  --base-url https://api.quantumrerank.com \
  --timeout 60
```

### 2. Load Testing

**Run Load Tests**
```bash
# Install k6
curl https://github.com/grafana/k6/releases/download/v0.45.0/k6-v0.45.0-linux-amd64.tar.gz -L | tar xvz --strip-components 1

# Run load test
./k6 run deployment/validation/load_test.js
```

## Troubleshooting

### Common AWS-Specific Issues

**EKS Cluster Issues**
```bash
# Check cluster status
aws eks describe-cluster --name quantum-rerank-prod

# Check node group status
aws eks describe-nodegroup \
  --cluster-name quantum-rerank-prod \
  --nodegroup-name quantum-rerank-workers

# Check CloudWatch logs
aws logs describe-log-groups --log-group-name-prefix /aws/eks/quantum-rerank-prod
```

**Load Balancer Issues**
```bash
# Check ALB status
aws elbv2 describe-load-balancers

# Check target group health
aws elbv2 describe-target-health --target-group-arn [TARGET_GROUP_ARN]
```

**ECR Issues**
```bash
# Check repository
aws ecr describe-repositories --repository-names quantum-rerank

# Check image scan results
aws ecr describe-image-scan-findings \
  --repository-name quantum-rerank \
  --image-id imageTag=$(git describe --tags --always)
```

## Cost Optimization

### 1. Right-Sizing Resources

**Instance Optimization**
```bash
# Use Spot instances for non-critical workloads
eksctl create nodegroup \
  --cluster quantum-rerank-prod \
  --name quantum-rerank-spot \
  --node-type m5.large,m5.xlarge,m4.large \
  --nodes 2 \
  --nodes-min 0 \
  --nodes-max 5 \
  --spot
```

### 2. Cost Monitoring

**Set up Cost Alerts**
```bash
aws budgets create-budget \
  --account-id $(aws sts get-caller-identity --query Account --output text) \
  --budget file://cost-budget.json
```

This comprehensive AWS deployment guide provides production-ready procedures for deploying QuantumRerank on AWS with all necessary security, monitoring, and operational considerations.
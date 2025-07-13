# Google Cloud Platform Deployment Guide for QuantumRerank

## Overview

This guide provides detailed procedures for deploying QuantumRerank on Google Cloud Platform (GCP) using Google Kubernetes Engine (GKE) and supporting GCP services.

## Prerequisites

### GCP Project Setup
- Google Cloud SDK (gcloud) installed and configured
- kubectl v1.24+ installed
- Helm v3.x installed
- Docker installed for local testing

### Required APIs and Permissions
- Kubernetes Engine API
- Compute Engine API
- Container Registry API
- Cloud DNS API
- Certificate Manager API
- Cloud Monitoring API
- Cloud Logging API
- Identity and Access Management (IAM) API

### IAM Roles Required
- Kubernetes Engine Admin
- Compute Admin
- Service Account Admin
- DNS Administrator
- Certificate Manager Editor

## Infrastructure Setup

### 1. Project and Network Configuration

**Set up Project Variables**
```bash
# Set project variables
export PROJECT_ID="quantum-rerank-prod"
export REGION="us-west1"
export ZONE="us-west1-a"
export CLUSTER_NAME="quantum-rerank-prod"

# Set default project
gcloud config set project $PROJECT_ID
gcloud config set compute/region $REGION
gcloud config set compute/zone $ZONE
```

**Enable Required APIs**
```bash
gcloud services enable \
  container.googleapis.com \
  compute.googleapis.com \
  containerregistry.googleapis.com \
  dns.googleapis.com \
  certificatemanager.googleapis.com \
  monitoring.googleapis.com \
  logging.googleapis.com
```

**Create VPC Network**
```bash
# Create custom VPC
gcloud compute networks create quantum-rerank-vpc \
  --subnet-mode custom \
  --bgp-routing-mode regional

# Create subnets
gcloud compute networks subnets create quantum-rerank-subnet \
  --network quantum-rerank-vpc \
  --range 10.0.0.0/16 \
  --region $REGION \
  --secondary-range pods=10.1.0.0/16,services=10.2.0.0/16

# Create firewall rules
gcloud compute firewall-rules create quantum-rerank-allow-internal \
  --network quantum-rerank-vpc \
  --allow tcp,udp,icmp \
  --source-ranges 10.0.0.0/8

gcloud compute firewall-rules create quantum-rerank-allow-ssh \
  --network quantum-rerank-vpc \
  --allow tcp:22 \
  --source-ranges 0.0.0.0/0

gcloud compute firewall-rules create quantum-rerank-allow-https \
  --network quantum-rerank-vpc \
  --allow tcp:443,tcp:80 \
  --source-ranges 0.0.0.0/0
```

### 2. GKE Cluster Setup

**Create GKE Cluster**
```bash
# Create GKE cluster with workload identity
gcloud container clusters create $CLUSTER_NAME \
  --region $REGION \
  --network quantum-rerank-vpc \
  --subnetwork quantum-rerank-subnet \
  --cluster-secondary-range-name pods \
  --services-secondary-range-name services \
  --enable-ip-alias \
  --enable-autorepair \
  --enable-autoupgrade \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 20 \
  --num-nodes 3 \
  --machine-type n2-standard-4 \
  --disk-type pd-ssd \
  --disk-size 100GB \
  --image-type COS_CONTAINERD \
  --enable-shielded-nodes \
  --enable-workload-identity \
  --workload-pool=${PROJECT_ID}.svc.id.goog \
  --enable-cloud-logging \
  --enable-cloud-monitoring \
  --logging=SYSTEM,WORKLOAD,API_SERVER \
  --monitoring=SYSTEM \
  --node-labels environment=production,application=quantum-rerank \
  --node-taints production=true:NoSchedule \
  --addons HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver
```

**Configure kubectl**
```bash
# Get cluster credentials
gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION

# Verify cluster access
kubectl get nodes
kubectl get pods --all-namespaces
```

### 3. Node Pool Configuration

**Create Production Node Pool**
```bash
# Remove default node pool taint and create production pool
kubectl taint nodes --all production=true:NoSchedule-

# Create optimized node pool for production workloads
gcloud container node-pools create quantum-rerank-production \
  --cluster $CLUSTER_NAME \
  --region $REGION \
  --machine-type n2-standard-4 \
  --disk-type pd-ssd \
  --disk-size 100GB \
  --num-nodes 3 \
  --enable-autorepair \
  --enable-autoupgrade \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 15 \
  --node-labels environment=production,tier=production \
  --node-taints dedicated=production:NoSchedule \
  --image-type COS_CONTAINERD \
  --enable-autorepair \
  --enable-autoupgrade
```

## Application Deployment

### 1. Container Registry Setup

**Configure Google Container Registry**
```bash
# Configure Docker to use gcloud as credential helper
gcloud auth configure-docker

# Build and tag image
export IMAGE_TAG=$(git describe --tags --always)
export GCR_IMAGE="gcr.io/${PROJECT_ID}/quantum-rerank:${IMAGE_TAG}"

# Build production image
docker build \
  --target production \
  --build-arg ENVIRONMENT=production \
  --build-arg VERSION=$IMAGE_TAG \
  -f deployment/Dockerfile \
  -t quantum-rerank:$IMAGE_TAG .

# Tag and push to GCR
docker tag quantum-rerank:$IMAGE_TAG $GCR_IMAGE
docker push $GCR_IMAGE
```

### 2. Google Cloud Services Integration

**Cloud Memorystore for Redis**
```bash
# Create Redis instance
gcloud redis instances create quantum-rerank-redis \
  --size 5 \
  --region $REGION \
  --network quantum-rerank-vpc \
  --redis-version redis_7_0 \
  --tier standard \
  --auth-enabled \
  --transit-encryption-mode SERVER_AUTHENTICATION \
  --replica-count 1 \
  --labels environment=production,application=quantum-rerank

# Get Redis connection info
export REDIS_HOST=$(gcloud redis instances describe quantum-rerank-redis \
  --region $REGION \
  --format="value(host)")

export REDIS_PORT=$(gcloud redis instances describe quantum-rerank-redis \
  --region $REGION \
  --format="value(port)")
```

**Cloud SQL for PostgreSQL (if needed)**
```bash
# Create Cloud SQL instance
gcloud sql instances create quantum-rerank-db \
  --database-version POSTGRES_15 \
  --tier db-custom-4-16384 \
  --region $REGION \
  --network quantum-rerank-vpc \
  --no-assign-ip \
  --storage-type SSD \
  --storage-size 100GB \
  --storage-auto-increase \
  --backup-start-time 03:00 \
  --maintenance-window-day SUN \
  --maintenance-window-hour 04 \
  --maintenance-release-channel production

# Create database and user
gcloud sql databases create quantumrerank --instance quantum-rerank-db
gcloud sql users create quantum_app --instance quantum-rerank-db \
  --password $(openssl rand -base64 32)
```

### 3. SSL Certificate Management

**Create Managed SSL Certificate**
```bash
# Create managed SSL certificate
gcloud compute ssl-certificates create quantum-rerank-ssl \
  --domains api.quantumrerank.com \
  --global

# Create certificate map (for newer certificate manager)
gcloud certificate-manager certificates create quantum-rerank-cert \
  --domains="api.quantumrerank.com" \
  --issuance-config="" \
  --global
```

### 4. Service Account Configuration

**Create Workload Identity Service Account**
```bash
# Create Google Service Account
gcloud iam service-accounts create quantum-rerank-gsa \
  --display-name "QuantumRerank Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member "serviceAccount:quantum-rerank-gsa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role "roles/monitoring.metricWriter"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member "serviceAccount:quantum-rerank-gsa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role "roles/logging.logWriter"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member "serviceAccount:quantum-rerank-gsa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role "roles/cloudtrace.agent"
```

### 5. Kubernetes Deployment

**Create Namespace and Configuration**
```bash
# Create namespace
kubectl create namespace quantum-rerank
kubectl label namespace quantum-rerank environment=production

# Create Kubernetes Service Account
kubectl create serviceaccount quantum-rerank \
  --namespace quantum-rerank

# Bind Kubernetes SA to Google SA
gcloud iam service-accounts add-iam-policy-binding \
  quantum-rerank-gsa@${PROJECT_ID}.iam.gserviceaccount.com \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:${PROJECT_ID}.svc.id.goog[quantum-rerank/quantum-rerank]"

kubectl annotate serviceaccount quantum-rerank \
  --namespace quantum-rerank \
  iam.gke.io/gcp-service-account=quantum-rerank-gsa@${PROJECT_ID}.iam.gserviceaccount.com

# Create secrets
kubectl create secret generic quantum-rerank-secrets \
  --namespace quantum-rerank \
  --from-literal=redis-auth-string="$(gcloud redis instances describe quantum-rerank-redis \
    --region $REGION --format='value(authString)')" \
  --from-literal=database-password="$(openssl rand -base64 32)" \
  --from-literal=jwt-secret="$(openssl rand -base64 64)" \
  --from-literal=api-key-salt="$(openssl rand -base64 32)"
```

**Deploy Application**
```bash
# Create GCP-specific deployment configuration
cat > gcp-deployment.yaml << EOF
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
      nodeSelector:
        environment: production
      tolerations:
      - key: dedicated
        operator: Equal
        value: production
        effect: NoSchedule
      containers:
      - name: quantum-rerank
        image: $GCR_IMAGE
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: REDIS_HOST
          value: "$REDIS_HOST"
        - name: REDIS_PORT
          value: "$REDIS_PORT"
        - name: REDIS_AUTH_STRING
          valueFrom:
            secretKeyRef:
              name: quantum-rerank-secrets
              key: redis-auth-string
        - name: GOOGLE_CLOUD_PROJECT
          value: "$PROJECT_ID"
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
    cloud.google.com/neg: '{"ingress": true}'
    cloud.google.com/backend-config: '{"ports": {"8000":"quantum-rerank-backendconfig"}}'
spec:
  type: ClusterIP
  ports:
  - port: 8000
    targetPort: 8000
    protocol: TCP
  selector:
    app: quantum-rerank
---
apiVersion: cloud.google.com/v1
kind: BackendConfig
metadata:
  name: quantum-rerank-backendconfig
  namespace: quantum-rerank
spec:
  healthCheck:
    checkIntervalSec: 10
    port: 8000
    type: HTTP
    requestPath: /health/ready
  timeoutSec: 30
  connectionDraining:
    drainingTimeoutSec: 60
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: quantum-rerank-ingress
  namespace: quantum-rerank
  annotations:
    kubernetes.io/ingress.class: "gce"
    kubernetes.io/ingress.global-static-ip-name: "quantum-rerank-ip"
    networking.gke.io/managed-certificates: "quantum-rerank-ssl"
    kubernetes.io/ingress.allow-http: "false"
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
kubectl apply -f gcp-deployment.yaml
```

## Auto-Scaling Configuration

### 1. Horizontal Pod Autoscaler

**Configure HPA with Custom Metrics**
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
  maxReplicas: 25
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

### 2. Cluster Autoscaler

**Configure Node Auto-Provisioning**
```bash
# Enable node auto-provisioning
gcloud container clusters update $CLUSTER_NAME \
  --region $REGION \
  --enable-autoprovisioning \
  --min-cpu 1 \
  --max-cpu 100 \
  --min-memory 1 \
  --max-memory 1000 \
  --autoprovisioning-node-pool-defaults-disk-size=100 \
  --autoprovisioning-node-pool-defaults-disk-type=pd-ssd \
  --autoprovisioning-node-pool-defaults-image-type=COS_CONTAINERD \
  --autoprovisioning-node-pool-defaults-shielded-secure-boot \
  --autoprovisioning-node-pool-defaults-shielded-integrity-monitoring
```

## Monitoring and Logging

### 1. Google Cloud Operations Suite

**Configure Monitoring and Logging**
```bash
# Install Google Cloud Monitoring
kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: google-cloud-monitoring
  namespace: quantum-rerank
  annotations:
    iam.gke.io/gcp-service-account: quantum-rerank-gsa@${PROJECT_ID}.iam.gserviceaccount.com
EOF

# Create monitoring dashboards
gcloud monitoring dashboards create --config-from-file=monitoring/gcp-dashboard.json
```

### 2. Prometheus and Grafana

**Install Prometheus Operator**
```bash
# Add Prometheus community Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install kube-prometheus-stack with GKE optimizations
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.storageClassName=standard-rwo \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi \
  --set grafana.persistence.enabled=true \
  --set grafana.persistence.storageClassName=standard-rwo \
  --set grafana.persistence.size=10Gi \
  --set grafana.adminPassword="$(openssl rand -base64 32)" \
  --set prometheus.service.type=LoadBalancer \
  --set grafana.service.type=LoadBalancer
```

## Backup and Disaster Recovery

### 1. Persistent Disk Snapshots

**Configure Automated Snapshots**
```bash
# Create snapshot schedule
gcloud compute resource-policies create snapshot-schedule quantum-rerank-daily \
  --region $REGION \
  --max-retention-days 30 \
  --on-source-disk-delete keep-auto-snapshots \
  --daily-schedule \
  --start-time 03:00 \
  --storage-location $REGION

# Apply to persistent disks
for disk in $(gcloud compute disks list --filter="zone:$ZONE AND labels.app=quantum-rerank" --format="value(name)"); do
  gcloud compute disks add-resource-policies $disk \
    --resource-policies quantum-rerank-daily \
    --zone $ZONE
done
```

### 2. Cloud SQL Backup

**Configure Automated Backups**
```bash
# Configure point-in-time recovery
gcloud sql instances patch quantum-rerank-db \
  --backup-start-time 03:00 \
  --enable-point-in-time-recovery \
  --retained-backups-count 30 \
  --retained-transaction-log-days 7
```

### 3. Application Backup with Velero

**Install Velero for Kubernetes Backup**
```bash
# Create backup bucket
gsutil mb gs://quantum-rerank-backups-$(date +%s)
export BACKUP_BUCKET=quantum-rerank-backups-$(date +%s)

# Create service account for Velero
gcloud iam service-accounts create velero \
  --display-name "Velero service account"

# Grant permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member serviceAccount:velero@${PROJECT_ID}.iam.gserviceaccount.com \
  --role roles/compute.storageAdmin

gsutil iam ch serviceAccount:velero@${PROJECT_ID}.iam.gserviceaccount.com:objectAdmin gs://$BACKUP_BUCKET

# Install Velero
velero install \
  --provider gcp \
  --plugins velero/velero-plugin-for-gcp:v1.8.0 \
  --bucket $BACKUP_BUCKET \
  --secret-file ./credentials-velero
```

## Security Configuration

### 1. Network Security

**Configure Private GKE Cluster**
```bash
# Update cluster to private
gcloud container clusters update $CLUSTER_NAME \
  --region $REGION \
  --enable-private-nodes \
  --master-ipv4-cidr 172.16.0.0/28 \
  --enable-ip-alias
```

**Configure Binary Authorization**
```bash
# Enable Binary Authorization
gcloud container binauthz policy import policy.yaml

# Create attestor for image verification
gcloud container binauthz attestors create quantum-rerank-attestor \
  --attestation-authority-note projects/${PROJECT_ID}/notes/quantum-rerank-note \
  --attestation-authority-note-public-key-id-override quantum-rerank-key
```

### 2. Workload Identity and IAM

**Configure Workload Identity**
```bash
# Already configured in deployment section above
# Verify workload identity binding
gcloud iam service-accounts get-iam-policy quantum-rerank-gsa@${PROJECT_ID}.iam.gserviceaccount.com
```

### 3. Pod Security Standards

**Apply Pod Security Standards**
```bash
kubectl apply -f - <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: quantum-rerank
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
EOF
```

## DNS and Load Balancing

### 1. Cloud DNS Configuration

**Create DNS Zone**
```bash
# Create DNS zone
gcloud dns managed-zones create quantumrerank-com \
  --dns-name quantumrerank.com \
  --description "DNS zone for QuantumRerank"

# Create static IP
gcloud compute addresses create quantum-rerank-ip --global

# Get static IP
export STATIC_IP=$(gcloud compute addresses describe quantum-rerank-ip --global --format="value(address)")

# Create DNS record
gcloud dns record-sets transaction start --zone quantumrerank-com
gcloud dns record-sets transaction add $STATIC_IP \
  --name api.quantumrerank.com \
  --ttl 300 \
  --type A \
  --zone quantumrerank-com
gcloud dns record-sets transaction execute --zone quantumrerank-com
```

### 2. Cloud CDN Configuration

**Enable Cloud CDN**
```bash
# Create CDN-enabled backend service
gcloud compute backend-services create quantum-rerank-backend \
  --global \
  --enable-cdn \
  --cache-mode USE_ORIGIN_HEADERS

# Configure CDN settings
gcloud compute backend-services update quantum-rerank-backend \
  --global \
  --signed-url-cache-max-age 3600 \
  --default-ttl 3600 \
  --max-ttl 86400 \
  --client-ttl 3600
```

## Performance Optimization

### 1. Node Pool Optimization

**Create High-Performance Node Pool**
```bash
gcloud container node-pools create quantum-rerank-compute \
  --cluster $CLUSTER_NAME \
  --region $REGION \
  --machine-type c2-standard-8 \
  --disk-type pd-ssd \
  --disk-size 200GB \
  --num-nodes 0 \
  --enable-autoscaling \
  --min-nodes 0 \
  --max-nodes 10 \
  --node-taints compute=intensive:NoSchedule \
  --node-labels workload=compute-intensive
```

### 2. GPU Support (if needed)

**Add GPU Node Pool**
```bash
gcloud container node-pools create quantum-rerank-gpu \
  --cluster $CLUSTER_NAME \
  --region $REGION \
  --machine-type n1-standard-4 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --disk-type pd-ssd \
  --disk-size 100GB \
  --num-nodes 0 \
  --enable-autoscaling \
  --min-nodes 0 \
  --max-nodes 5 \
  --node-taints gpu=nvidia:NoSchedule
```

## Validation and Testing

### 1. Deployment Validation

**Run GCP-Specific Validation**
```bash
# Verify cluster health
gcloud container clusters describe $CLUSTER_NAME --region $REGION

# Check node status
kubectl get nodes -o wide

# Verify services
kubectl get svc -n quantum-rerank
kubectl get ingress -n quantum-rerank

# Test external access
curl -H "Host: api.quantumrerank.com" http://$STATIC_IP/health

# Run smoke tests
python deployment/validation/smoke_tests.py \
  --base-url https://api.quantumrerank.com \
  --timeout 60

# Run performance validation
python deployment/validation/performance_validation.py \
  --base-url https://api.quantumrerank.com
```

### 2. Load Testing

**GCP-Specific Load Testing**
```bash
# Use Google Cloud Load Testing
gcloud builds submit --config cloudbuild-loadtest.yaml .

# Or use k6 with GCP monitoring
./k6 run --out cloud deployment/validation/load_test.js
```

## Troubleshooting

### Common GCP-Specific Issues

**GKE Cluster Issues**
```bash
# Check cluster events
gcloud logging read "resource.type=gke_cluster AND resource.labels.cluster_name=$CLUSTER_NAME" \
  --limit 50 \
  --format json

# Check node pool status
gcloud container node-pools describe quantum-rerank-production \
  --cluster $CLUSTER_NAME \
  --region $REGION
```

**Load Balancer Issues**
```bash
# Check backend service health
gcloud compute backend-services get-health quantum-rerank-backend --global

# Check firewall rules
gcloud compute firewall-rules list --filter="network:quantum-rerank-vpc"
```

**Monitoring Issues**
```bash
# Check monitoring agent status
kubectl get pods -n gke-system | grep metrics

# Check logs
gcloud logging read "resource.type=k8s_container AND resource.labels.namespace_name=quantum-rerank" \
  --limit 100
```

## Cost Optimization

### 1. Preemptible Instances

**Create Preemptible Node Pool**
```bash
gcloud container node-pools create quantum-rerank-preemptible \
  --cluster $CLUSTER_NAME \
  --region $REGION \
  --machine-type n2-standard-2 \
  --preemptible \
  --num-nodes 2 \
  --enable-autoscaling \
  --min-nodes 0 \
  --max-nodes 10 \
  --node-labels instance-type=preemptible
```

### 2. Resource Optimization

**Configure Resource Quotas**
```bash
kubectl apply -f - <<EOF
apiVersion: v1
kind: ResourceQuota
metadata:
  name: quantum-rerank-quota
  namespace: quantum-rerank
spec:
  hard:
    requests.cpu: "10"
    requests.memory: 20Gi
    limits.cpu: "20"
    limits.memory: 40Gi
    persistentvolumeclaims: "10"
EOF
```

This comprehensive GCP deployment guide provides production-ready procedures for deploying QuantumRerank on Google Cloud Platform with all necessary security, monitoring, and operational considerations.
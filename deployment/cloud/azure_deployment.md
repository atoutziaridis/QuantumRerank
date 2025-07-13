# Microsoft Azure Deployment Guide for QuantumRerank

## Overview

This guide provides detailed procedures for deploying QuantumRerank on Microsoft Azure using Azure Kubernetes Service (AKS) and supporting Azure services.

## Prerequisites

### Azure Account Setup
- Azure CLI (az) v2.50+ installed and configured
- kubectl v1.24+ installed
- Helm v3.x installed
- Docker installed for local testing

### Required Permissions
- Contributor role on the target subscription
- User Access Administrator role (for RBAC)
- Azure Kubernetes Service Cluster Admin Role

### Azure Resource Providers
- Microsoft.ContainerService
- Microsoft.Network
- Microsoft.Storage
- Microsoft.ContainerRegistry
- Microsoft.OperationalInsights
- Microsoft.Insights
- Microsoft.KeyVault

## Infrastructure Setup

### 1. Resource Group and Basic Configuration

**Set up Azure Variables**
```bash
# Set Azure variables
export SUBSCRIPTION_ID="your-subscription-id"
export RESOURCE_GROUP="quantum-rerank-prod-rg"
export LOCATION="westus2"
export CLUSTER_NAME="quantum-rerank-prod"
export ACR_NAME="quantumrerankprod"
export VNET_NAME="quantum-rerank-vnet"
export SUBNET_NAME="quantum-rerank-subnet"

# Set default subscription
az account set --subscription $SUBSCRIPTION_ID

# Create resource group
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION \
  --tags environment=production application=quantum-rerank
```

**Register Resource Providers**
```bash
az provider register --namespace Microsoft.ContainerService
az provider register --namespace Microsoft.Network
az provider register --namespace Microsoft.Storage
az provider register --namespace Microsoft.ContainerRegistry
az provider register --namespace Microsoft.OperationalInsights
az provider register --namespace Microsoft.Insights
az provider register --namespace Microsoft.KeyVault
```

### 2. Network Infrastructure

**Create Virtual Network**
```bash
# Create virtual network
az network vnet create \
  --resource-group $RESOURCE_GROUP \
  --name $VNET_NAME \
  --address-prefixes 10.0.0.0/16 \
  --subnet-name $SUBNET_NAME \
  --subnet-prefix 10.0.1.0/24 \
  --location $LOCATION

# Create additional subnets
az network vnet subnet create \
  --resource-group $RESOURCE_GROUP \
  --vnet-name $VNET_NAME \
  --name quantum-rerank-internal \
  --address-prefixes 10.0.2.0/24

az network vnet subnet create \
  --resource-group $RESOURCE_GROUP \
  --vnet-name $VNET_NAME \
  --name quantum-rerank-services \
  --address-prefixes 10.0.3.0/24

# Get subnet ID
export SUBNET_ID=$(az network vnet subnet show \
  --resource-group $RESOURCE_GROUP \
  --vnet-name $VNET_NAME \
  --name $SUBNET_NAME \
  --query id -o tsv)
```

**Configure Network Security Groups**
```bash
# Create network security group
az network nsg create \
  --resource-group $RESOURCE_GROUP \
  --name quantum-rerank-nsg \
  --location $LOCATION

# Add security rules
az network nsg rule create \
  --resource-group $RESOURCE_GROUP \
  --nsg-name quantum-rerank-nsg \
  --name AllowHTTPS \
  --priority 1000 \
  --protocol Tcp \
  --destination-port-ranges 443 \
  --access Allow

az network nsg rule create \
  --resource-group $RESOURCE_GROUP \
  --nsg-name quantum-rerank-nsg \
  --name AllowHTTP \
  --priority 1001 \
  --protocol Tcp \
  --destination-port-ranges 80 \
  --access Allow

# Associate NSG with subnet
az network vnet subnet update \
  --resource-group $RESOURCE_GROUP \
  --vnet-name $VNET_NAME \
  --name $SUBNET_NAME \
  --network-security-group quantum-rerank-nsg
```

### 3. Azure Container Registry

**Create Container Registry**
```bash
# Create Azure Container Registry
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Premium \
  --location $LOCATION \
  --admin-enabled false \
  --public-network-enabled true

# Enable container scanning
az acr update \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --quarantine-policy enabled

# Login to ACR
az acr login --name $ACR_NAME
```

**Build and Push Container Image**
```bash
# Get ACR login server
export ACR_LOGIN_SERVER=$(az acr show \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --query loginServer \
  --output tsv)

export IMAGE_TAG=$(git describe --tags --always)
export FULL_IMAGE_NAME="$ACR_LOGIN_SERVER/quantum-rerank:$IMAGE_TAG"

# Build production image
docker build \
  --target production \
  --build-arg ENVIRONMENT=production \
  --build-arg VERSION=$IMAGE_TAG \
  -f deployment/Dockerfile \
  -t quantum-rerank:$IMAGE_TAG .

# Tag and push to ACR
docker tag quantum-rerank:$IMAGE_TAG $FULL_IMAGE_NAME
docker push $FULL_IMAGE_NAME
```

### 4. Azure Kubernetes Service Setup

**Create AKS Cluster**
```bash
# Create service principal for AKS
export SP_NAME="quantum-rerank-sp"
export SP_JSON=$(az ad sp create-for-rbac \
  --name $SP_NAME \
  --role Contributor \
  --scopes /subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP)

export CLIENT_ID=$(echo $SP_JSON | jq -r .appId)
export CLIENT_SECRET=$(echo $SP_JSON | jq -r .password)

# Create AKS cluster
az aks create \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --location $LOCATION \
  --node-count 3 \
  --min-count 3 \
  --max-count 20 \
  --enable-cluster-autoscaler \
  --node-vm-size Standard_D4s_v3 \
  --node-osdisk-size 100 \
  --node-osdisk-type Managed \
  --network-plugin azure \
  --network-policy azure \
  --vnet-subnet-id $SUBNET_ID \
  --service-cidr 10.1.0.0/16 \
  --dns-service-ip 10.1.0.10 \
  --docker-bridge-address 172.17.0.1/16 \
  --enable-addons monitoring,http_application_routing \
  --workspace-resource-id "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.OperationalInsights/workspaces/quantum-rerank-workspace" \
  --enable-managed-identity \
  --attach-acr $ACR_NAME \
  --kubernetes-version 1.28.3 \
  --enable-pod-identity \
  --enable-pod-identity-with-kubenet \
  --tags environment=production application=quantum-rerank
```

**Configure kubectl**
```bash
# Get AKS credentials
az aks get-credentials \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --overwrite-existing

# Verify cluster access
kubectl get nodes
kubectl get pods --all-namespaces
```

### 5. Log Analytics Workspace

**Create Log Analytics Workspace**
```bash
# Create Log Analytics workspace
az monitor log-analytics workspace create \
  --resource-group $RESOURCE_GROUP \
  --workspace-name quantum-rerank-workspace \
  --location $LOCATION \
  --sku PerGB2018

# Get workspace ID
export WORKSPACE_ID=$(az monitor log-analytics workspace show \
  --resource-group $RESOURCE_GROUP \
  --workspace-name quantum-rerank-workspace \
  --query id \
  --output tsv)
```

## Application Deployment

### 1. Azure Services Integration

**Azure Cache for Redis**
```bash
# Create Redis cache
az redis create \
  --location $LOCATION \
  --resource-group $RESOURCE_GROUP \
  --name quantum-rerank-redis \
  --sku Premium \
  --vm-size P2 \
  --redis-configuration maxmemory-policy=allkeys-lru \
  --enable-non-ssl-port false \
  --minimum-tls-version 1.2 \
  --subnet-id "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Network/virtualNetworks/$VNET_NAME/subnets/quantum-rerank-services"

# Get Redis connection info
export REDIS_HOSTNAME=$(az redis show \
  --resource-group $RESOURCE_GROUP \
  --name quantum-rerank-redis \
  --query hostName \
  --output tsv)

export REDIS_KEY=$(az redis list-keys \
  --resource-group $RESOURCE_GROUP \
  --name quantum-rerank-redis \
  --query primaryKey \
  --output tsv)
```

**Azure Database for PostgreSQL (if needed)**
```bash
# Create PostgreSQL server
az postgres flexible-server create \
  --resource-group $RESOURCE_GROUP \
  --name quantum-rerank-db \
  --location $LOCATION \
  --admin-user quantum_admin \
  --admin-password "$(openssl rand -base64 32)" \
  --sku-name Standard_D4s_v3 \
  --tier GeneralPurpose \
  --storage-size 128 \
  --version 15 \
  --high-availability Enabled \
  --subnet $SUBNET_ID \
  --private-dns-zone quantum-rerank-db.private.postgres.database.azure.com

# Create database
az postgres flexible-server db create \
  --resource-group $RESOURCE_GROUP \
  --server-name quantum-rerank-db \
  --database-name quantumrerank
```

### 2. Azure Key Vault Integration

**Create Key Vault**
```bash
# Create Key Vault
az keyvault create \
  --resource-group $RESOURCE_GROUP \
  --name quantum-rerank-kv \
  --location $LOCATION \
  --sku premium \
  --enable-soft-delete true \
  --retention-days 30

# Store secrets
az keyvault secret set \
  --vault-name quantum-rerank-kv \
  --name redis-key \
  --value "$REDIS_KEY"

az keyvault secret set \
  --vault-name quantum-rerank-kv \
  --name jwt-secret \
  --value "$(openssl rand -base64 64)"

az keyvault secret set \
  --vault-name quantum-rerank-kv \
  --name api-key-salt \
  --value "$(openssl rand -base64 32)"
```

**Configure Key Vault Access**
```bash
# Get AKS managed identity
export AKS_IDENTITY=$(az aks show \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --query identityProfile.kubeletidentity.clientId \
  --output tsv)

# Grant Key Vault access
az keyvault set-policy \
  --name quantum-rerank-kv \
  --object-id $AKS_IDENTITY \
  --secret-permissions get list
```

### 3. SSL Certificate with Azure Application Gateway

**Create Application Gateway**
```bash
# Create public IP for Application Gateway
az network public-ip create \
  --resource-group $RESOURCE_GROUP \
  --name quantum-rerank-agw-pip \
  --allocation-method Static \
  --sku Standard \
  --location $LOCATION

# Create Application Gateway subnet
az network vnet subnet create \
  --resource-group $RESOURCE_GROUP \
  --vnet-name $VNET_NAME \
  --name agw-subnet \
  --address-prefixes 10.0.4.0/24

# Create Application Gateway
az network application-gateway create \
  --name quantum-rerank-agw \
  --location $LOCATION \
  --resource-group $RESOURCE_GROUP \
  --vnet-name $VNET_NAME \
  --subnet agw-subnet \
  --capacity 2 \
  --sku Standard_v2 \
  --http-settings-cookie-based-affinity Disabled \
  --frontend-port 80 \
  --http-settings-port 80 \
  --http-settings-protocol Http \
  --public-ip-address quantum-rerank-agw-pip
```

### 4. Kubernetes Deployment

**Create Namespace and Configuration**
```bash
# Create namespace
kubectl create namespace quantum-rerank
kubectl label namespace quantum-rerank environment=production

# Create Azure-specific secrets
kubectl create secret generic quantum-rerank-secrets \
  --namespace quantum-rerank \
  --from-literal=redis-key="$REDIS_KEY" \
  --from-literal=redis-hostname="$REDIS_HOSTNAME" \
  --from-literal=database-password="$(openssl rand -base64 32)" \
  --from-literal=jwt-secret="$(openssl rand -base64 64)" \
  --from-literal=api-key-salt="$(openssl rand -base64 32)"
```

**Deploy Application**
```bash
# Create Azure-specific deployment configuration
cat > azure-deployment.yaml << EOF
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
      containers:
      - name: quantum-rerank
        image: $FULL_IMAGE_NAME
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: REDIS_HOSTNAME
          valueFrom:
            secretKeyRef:
              name: quantum-rerank-secrets
              key: redis-hostname
        - name: REDIS_KEY
          valueFrom:
            secretKeyRef:
              name: quantum-rerank-secrets
              key: redis-key
        - name: REDIS_PORT
          value: "6380"
        - name: REDIS_SSL
          value: "true"
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
    service.beta.kubernetes.io/azure-load-balancer-internal: "true"
spec:
  type: LoadBalancer
  ports:
  - port: 80
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
    kubernetes.io/ingress.class: azure/application-gateway
    appgw.ingress.kubernetes.io/ssl-redirect: "true"
    appgw.ingress.kubernetes.io/cookie-based-affinity: "false"
    appgw.ingress.kubernetes.io/request-timeout: "30"
    appgw.ingress.kubernetes.io/connection-draining: "true"
    appgw.ingress.kubernetes.io/connection-draining-timeout: "30"
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
              number: 80
EOF

# Apply deployment
kubectl apply -f azure-deployment.yaml
```

## Auto-Scaling Configuration

### 1. Horizontal Pod Autoscaler

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
  maxReplicas: 30
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

**Configure Node Pool Autoscaling**
```bash
# Update node pool with autoscaling parameters
az aks nodepool update \
  --resource-group $RESOURCE_GROUP \
  --cluster-name $CLUSTER_NAME \
  --name nodepool1 \
  --enable-cluster-autoscaler \
  --min-count 3 \
  --max-count 30

# Create additional node pool for compute-intensive workloads
az aks nodepool add \
  --resource-group $RESOURCE_GROUP \
  --cluster-name $CLUSTER_NAME \
  --name computepool \
  --node-count 0 \
  --min-count 0 \
  --max-count 10 \
  --enable-cluster-autoscaler \
  --node-vm-size Standard_F8s_v2 \
  --node-taints compute=intensive:NoSchedule \
  --labels workload=compute-intensive
```

## Monitoring and Logging

### 1. Azure Monitor Integration

**Configure Container Insights**
```bash
# Enable Container Insights (if not enabled during cluster creation)
az aks enable-addons \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --addons monitoring \
  --workspace-resource-id $WORKSPACE_ID
```

**Create Custom Dashboards**
```bash
# Create Azure Monitor dashboard
az portal dashboard create \
  --resource-group $RESOURCE_GROUP \
  --name quantum-rerank-dashboard \
  --input-path monitoring/azure-dashboard.json
```

### 2. Application Insights

**Create Application Insights**
```bash
# Create Application Insights instance
az monitor app-insights component create \
  --app quantum-rerank-insights \
  --location $LOCATION \
  --resource-group $RESOURCE_GROUP \
  --workspace $WORKSPACE_ID

# Get instrumentation key
export APPINSIGHTS_KEY=$(az monitor app-insights component show \
  --app quantum-rerank-insights \
  --resource-group $RESOURCE_GROUP \
  --query instrumentationKey \
  --output tsv)

# Update deployment to include Application Insights
kubectl set env deployment/quantum-rerank \
  -n quantum-rerank \
  APPINSIGHTS_INSTRUMENTATIONKEY=$APPINSIGHTS_KEY
```

### 3. Prometheus and Grafana

**Install Prometheus Operator**
```bash
# Add Prometheus community Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install kube-prometheus-stack
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.storageClassName=managed-premium \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi \
  --set grafana.persistence.enabled=true \
  --set grafana.persistence.storageClassName=managed-premium \
  --set grafana.persistence.size=10Gi \
  --set grafana.adminPassword="$(openssl rand -base64 32)" \
  --set prometheus.service.type=LoadBalancer \
  --set grafana.service.type=LoadBalancer
```

## Backup and Disaster Recovery

### 1. Azure Backup for AKS

**Configure Backup**
```bash
# Create Recovery Services vault
az backup vault create \
  --resource-group $RESOURCE_GROUP \
  --name quantum-rerank-vault \
  --location $LOCATION

# Enable backup for AKS (Azure Backup for AKS is in preview)
# Manual backup approach using Velero
```

### 2. Velero Backup

**Install Velero with Azure**
```bash
# Create storage account for backups
export STORAGE_ACCOUNT="quantumrerankbackup$(date +%s)"

az storage account create \
  --name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --sku Standard_GRS \
  --encryption-services blob \
  --https-only true \
  --kind BlobStorage \
  --access-tier Hot

# Create blob container
az storage container create \
  --name velero \
  --public-access off \
  --account-name $STORAGE_ACCOUNT

# Get storage account key
export AZURE_STORAGE_ACCOUNT_ACCESS_KEY=$(az storage account keys list \
  --account-name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --query "[0].value" \
  --output tsv)

# Create credentials file for Velero
cat > credentials-velero << EOF
AZURE_SUBSCRIPTION_ID=$SUBSCRIPTION_ID
AZURE_TENANT_ID=$(az account show --query tenantId --output tsv)
AZURE_CLIENT_ID=$CLIENT_ID
AZURE_CLIENT_SECRET=$CLIENT_SECRET
AZURE_RESOURCE_GROUP=$RESOURCE_GROUP
AZURE_CLOUD_NAME=AzurePublicCloud
EOF

# Install Velero
velero install \
  --provider azure \
  --plugins velero/velero-plugin-for-microsoft-azure:v1.8.0 \
  --bucket velero \
  --secret-file ./credentials-velero \
  --backup-location-config resourceGroup=$RESOURCE_GROUP,storageAccount=$STORAGE_ACCOUNT \
  --snapshot-location-config resourceGroup=$RESOURCE_GROUP
```

### 3. Database Backup

**Configure PostgreSQL Backup**
```bash
# Configure automated backups
az postgres flexible-server parameter set \
  --resource-group $RESOURCE_GROUP \
  --server-name quantum-rerank-db \
  --name backup_retention_days \
  --value 30

# Create manual backup
az postgres flexible-server backup create \
  --resource-group $RESOURCE_GROUP \
  --server-name quantum-rerank-db \
  --backup-name quantum-rerank-backup-$(date +%Y%m%d)
```

## Security Configuration

### 1. Azure Active Directory Integration

**Configure AAD Integration**
```bash
# Enable AAD integration for AKS
az aks update \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --enable-aad \
  --aad-admin-group-object-ids $(az ad group show --group "AKS-Admins" --query objectId --output tsv)
```

### 2. Azure Policy for AKS

**Apply Security Policies**
```bash
# Enable Azure Policy for AKS
az aks enable-addons \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --addons azure-policy

# Apply policy definitions
az policy assignment create \
  --name "AKS Security Baseline" \
  --policy-set-definition "a8640138-9b0a-4a28-b8cb-1666c838647d" \
  --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.ContainerService/managedClusters/$CLUSTER_NAME"
```

### 3. Network Security

**Configure Private Cluster**
```bash
# Create private AKS cluster (for new deployments)
az aks create \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME-private \
  --enable-private-cluster \
  --private-dns-zone system \
  --node-count 3 \
  --enable-cluster-autoscaler \
  --min-count 3 \
  --max-count 20
```

## DNS and Traffic Management

### 1. Azure DNS Configuration

**Create DNS Zone**
```bash
# Create DNS zone
az network dns zone create \
  --resource-group $RESOURCE_GROUP \
  --name quantumrerank.com

# Get Application Gateway public IP
export AGW_PUBLIC_IP=$(az network public-ip show \
  --resource-group $RESOURCE_GROUP \
  --name quantum-rerank-agw-pip \
  --query ipAddress \
  --output tsv)

# Create DNS record
az network dns record-set a add-record \
  --resource-group $RESOURCE_GROUP \
  --zone-name quantumrerank.com \
  --record-set-name api \
  --ipv4-address $AGW_PUBLIC_IP
```

### 2. Azure Traffic Manager

**Configure Global Load Balancing**
```bash
# Create Traffic Manager profile
az network traffic-manager profile create \
  --resource-group $RESOURCE_GROUP \
  --name quantum-rerank-tm \
  --routing-method Performance \
  --unique-dns-name quantum-rerank-global \
  --ttl 30 \
  --protocol HTTPS \
  --port 443 \
  --path /health

# Add endpoint
az network traffic-manager endpoint create \
  --resource-group $RESOURCE_GROUP \
  --profile-name quantum-rerank-tm \
  --name westus2-endpoint \
  --type azureEndpoints \
  --target-resource-id "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Network/publicIPAddresses/quantum-rerank-agw-pip"
```

## Performance Optimization

### 1. Node Pool Optimization

**Create Specialized Node Pools**
```bash
# Create memory-optimized node pool
az aks nodepool add \
  --resource-group $RESOURCE_GROUP \
  --cluster-name $CLUSTER_NAME \
  --name memorypool \
  --node-count 0 \
  --min-count 0 \
  --max-count 8 \
  --enable-cluster-autoscaler \
  --node-vm-size Standard_E8s_v3 \
  --node-taints memory=optimized:NoSchedule \
  --labels workload=memory-intensive

# Create compute-optimized node pool
az aks nodepool add \
  --resource-group $RESOURCE_GROUP \
  --cluster-name $CLUSTER_NAME \
  --name computepool \
  --node-count 0 \
  --min-count 0 \
  --max-count 10 \
  --enable-cluster-autoscaler \
  --node-vm-size Standard_F8s_v2 \
  --node-taints compute=optimized:NoSchedule \
  --labels workload=compute-intensive
```

### 2. Azure CDN Integration

**Configure CDN**
```bash
# Create CDN profile
az cdn profile create \
  --resource-group $RESOURCE_GROUP \
  --name quantum-rerank-cdn \
  --sku Standard_Microsoft \
  --location Global

# Create CDN endpoint
az cdn endpoint create \
  --resource-group $RESOURCE_GROUP \
  --profile-name quantum-rerank-cdn \
  --name quantum-rerank-endpoint \
  --origin api.quantumrerank.com \
  --origin-host-header api.quantumrerank.com \
  --enable-compression true \
  --query-string-caching-behavior IgnoreQueryString
```

## Validation and Testing

### 1. Deployment Validation

**Run Azure-Specific Validation**
```bash
# Verify AKS cluster
az aks show --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME

# Check node status
kubectl get nodes -o wide

# Verify Azure services
az redis show --resource-group $RESOURCE_GROUP --name quantum-rerank-redis
az postgres flexible-server show --resource-group $RESOURCE_GROUP --name quantum-rerank-db

# Test application endpoints
curl -H "Host: api.quantumrerank.com" http://$AGW_PUBLIC_IP/health

# Run smoke tests
python deployment/validation/smoke_tests.py \
  --base-url https://api.quantumrerank.com \
  --timeout 60

# Run performance validation
python deployment/validation/performance_validation.py \
  --base-url https://api.quantumrerank.com
```

### 2. Load Testing with Azure

**Use Azure Load Testing**
```bash
# Create Azure Load Testing resource
az load create \
  --name quantum-rerank-loadtest \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

# Run load test (requires test configuration file)
az load test create \
  --load-test-resource quantum-rerank-loadtest \
  --resource-group $RESOURCE_GROUP \
  --test-id quantum-rerank-test \
  --display-name "QuantumRerank Production Load Test" \
  --test-plan load-test-config.yaml
```

## Troubleshooting

### Common Azure-Specific Issues

**AKS Cluster Issues**
```bash
# Check cluster health
az aks show --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME --query provisioningState

# Check node pool status
az aks nodepool list --resource-group $RESOURCE_GROUP --cluster-name $CLUSTER_NAME

# Check cluster logs
az aks get-upgrades --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME
```

**Application Gateway Issues**
```bash
# Check Application Gateway health
az network application-gateway show-health \
  --resource-group $RESOURCE_GROUP \
  --name quantum-rerank-agw

# Check backend health
az network application-gateway show-backend-health \
  --resource-group $RESOURCE_GROUP \
  --name quantum-rerank-agw
```

**Azure Services Issues**
```bash
# Check Redis status
az redis show --resource-group $RESOURCE_GROUP --name quantum-rerank-redis --query provisioningState

# Check PostgreSQL status
az postgres flexible-server show --resource-group $RESOURCE_GROUP --name quantum-rerank-db --query state
```

## Cost Optimization

### 1. Spot Instances

**Create Spot Node Pool**
```bash
az aks nodepool add \
  --resource-group $RESOURCE_GROUP \
  --cluster-name $CLUSTER_NAME \
  --name spotpool \
  --priority Spot \
  --eviction-policy Delete \
  --spot-max-price -1 \
  --node-count 2 \
  --min-count 0 \
  --max-count 10 \
  --enable-cluster-autoscaler \
  --node-vm-size Standard_D4s_v3 \
  --node-taints kubernetes.azure.com/scalesetpriority=spot:NoSchedule
```

### 2. Azure Reservations and Cost Management

**Configure Cost Alerts**
```bash
# Create budget alert
az consumption budget create \
  --resource-group $RESOURCE_GROUP \
  --budget-name quantum-rerank-budget \
  --amount 1000 \
  --time-grain Monthly \
  --time-period start-date=$(date -d "first day of this month" +%Y-%m-%d) \
  --threshold 80
```

This comprehensive Azure deployment guide provides production-ready procedures for deploying QuantumRerank on Microsoft Azure with all necessary security, monitoring, and operational considerations.
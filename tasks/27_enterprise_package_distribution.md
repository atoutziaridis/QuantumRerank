# Task 27: Enterprise Package Distribution & Installation

## Overview
Create comprehensive package distribution and installation systems to enable one-click deployment and easy adoption by enterprises of all sizes.

## Objectives
- Implement automated package publishing for multiple ecosystems
- Create one-line installation commands for different deployment scenarios
- Develop enterprise-specific distribution channels
- Provide comprehensive deployment templates and infrastructure-as-code

## Requirements

### 1. Package Distribution Ecosystems

#### Python Ecosystem
```bash
# Target Installation Commands
pip install quantum-rerank-server    # Full server package
pip install quantum-rerank-client    # Client SDK only
pip install quantum-rerank[enterprise]  # Enterprise features
```

#### Container Ecosystem
```bash
# Official Docker Images
docker pull quantumrerank/server:latest
docker pull quantumrerank/server:enterprise
docker pull quantumrerank/client-tools:latest

# Quick start
docker run -d -p 8000:8000 quantumrerank/server:latest
```

#### Kubernetes Ecosystem
```bash
# Helm Charts
helm repo add quantumrerank https://charts.quantumrerank.ai
helm install my-reranker quantumrerank/quantum-rerank

# Operator Installation
kubectl apply -f https://install.quantumrerank.ai/operator.yaml
```

#### Cloud Marketplaces
- **AWS Marketplace**: AMI and container listings
- **Azure Marketplace**: ARM templates and containers
- **GCP Marketplace**: Deployment Manager templates
- **Docker Hub**: Official verified images

### 2. Enterprise Distribution Strategy

#### Deployment Tiers
```yaml
# Starter Tier (Free/Open Source)
quantum_rerank_starter:
  features:
    - Basic reranking API
    - Standard authentication
    - Community support
    - Public cloud deployment only
  limits:
    requests_per_month: 10000
    concurrent_requests: 10
    quantum_methods: false

# Professional Tier
quantum_rerank_professional:
  features:
    - Full API access
    - Quantum and classical methods
    - Email support
    - Advanced caching
    - Performance monitoring
  limits:
    requests_per_month: 1000000
    concurrent_requests: 100
    custom_models: 1

# Enterprise Tier
quantum_rerank_enterprise:
  features:
    - All professional features
    - On-premise deployment
    - Custom model training
    - SSO integration
    - 24/7 support
    - SLA guarantees
    - Custom integrations
  limits:
    requests_per_month: unlimited
    concurrent_requests: unlimited
    custom_models: unlimited
```

### 3. Package Structure

#### Server Package (`quantum-rerank-server`)
```
quantum-rerank-server/
├── quantum_rerank/           # Core application
├── deployment/               # Deployment templates
│   ├── docker/
│   ├── kubernetes/
│   ├── terraform/
│   └── cloud-formation/
├── config/                   # Configuration templates
│   ├── development.yaml
│   ├── staging.yaml
│   └── production.yaml
├── scripts/                  # Deployment scripts
│   ├── install.sh
│   ├── upgrade.sh
│   ├── backup.sh
│   └── migrate.sh
├── docs/                     # Documentation
├── examples/                 # Usage examples
└── tests/                    # Test suite
```

#### Client Tools Package (`quantum-rerank-client-tools`)
```
quantum-rerank-client-tools/
├── quantum_rerank_cli/       # CLI tools
├── quantum_rerank_client/    # Python SDK
├── integrations/             # Framework integrations
│   ├── langchain/
│   ├── haystack/
│   └── llamaindex/
├── templates/                # Project templates
├── examples/                 # Integration examples
└── docs/                     # Client documentation
```

### 4. Infrastructure as Code Templates

#### Terraform Modules
```hcl
# modules/quantum-rerank/main.tf
module "quantum_rerank" {
  source = "quantumrerank/quantum-rerank/aws"
  
  # Basic configuration
  instance_type = "m5.xlarge"
  replicas      = 3
  
  # Networking
  vpc_id     = var.vpc_id
  subnet_ids = var.private_subnet_ids
  
  # Security
  ssl_certificate_arn = var.ssl_cert_arn
  api_keys = {
    admin = var.admin_api_key
  }
  
  # Performance
  enable_caching = true
  cache_size_gb  = 10
  
  # Monitoring
  enable_monitoring = true
  log_retention_days = 30
  
  tags = var.common_tags
}
```

#### CloudFormation Templates
```yaml
# templates/quantum-rerank-stack.yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'QuantumRerank Production Deployment'

Parameters:
  InstanceType:
    Type: String
    Default: m5.xlarge
    AllowedValues: [m5.large, m5.xlarge, m5.2xlarge]
  
  ReplicaCount:
    Type: Number
    Default: 3
    MinValue: 1
    MaxValue: 10

Resources:
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: quantum-rerank-cluster
      
  ApplicationLoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Scheme: internet-facing
      Type: application
      
  # ... additional resources
```

#### Kubernetes Helm Charts
```yaml
# charts/quantum-rerank/values.yaml
replicaCount: 3

image:
  repository: quantumrerank/server
  tag: "latest"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 8000

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: api.quantumrerank.example.com
      paths:
        - path: /
          pathType: Prefix

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 1000m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70

quantum:
  method: "hybrid"
  cache:
    enabled: true
    size: "10Gi"
```

### 5. Installation Scripts

#### Universal Installer
```bash
#!/bin/bash
# install.sh - Universal QuantumRerank installer

set -e

# Configuration
INSTALL_DIR="/opt/quantum-rerank"
SERVICE_USER="quantum-rerank"
CONFIG_DIR="/etc/quantum-rerank"

# Detect environment
detect_environment() {
    if command -v kubectl &> /dev/null; then
        echo "kubernetes"
    elif command -v docker &> /dev/null; then
        echo "docker"
    else
        echo "native"
    fi
}

# Install for Kubernetes
install_kubernetes() {
    echo "Installing QuantumRerank on Kubernetes..."
    
    # Add Helm repo
    helm repo add quantumrerank https://charts.quantumrerank.ai
    helm repo update
    
    # Install with default values
    helm install quantum-rerank quantumrerank/quantum-rerank \
        --namespace quantum-rerank \
        --create-namespace \
        --set ingress.enabled=true \
        --set autoscaling.enabled=true
        
    echo "Installation complete! Check status with:"
    echo "kubectl get pods -n quantum-rerank"
}

# Install with Docker
install_docker() {
    echo "Installing QuantumRerank with Docker..."
    
    # Create directories
    mkdir -p $CONFIG_DIR
    mkdir -p $INSTALL_DIR/data
    
    # Download configuration
    curl -o $CONFIG_DIR/config.yaml \
        https://install.quantumrerank.ai/config/production.yaml
    
    # Run container
    docker run -d \
        --name quantum-rerank \
        --restart unless-stopped \
        -p 8000:8000 \
        -v $CONFIG_DIR:/app/config \
        -v $INSTALL_DIR/data:/app/data \
        quantumrerank/server:latest
        
    echo "Installation complete! API available at http://localhost:8000"
}

# Install natively
install_native() {
    echo "Installing QuantumRerank natively..."
    
    # Install Python package
    pip install quantum-rerank-server[production]
    
    # Create service user
    useradd -r -s /bin/false $SERVICE_USER
    
    # Create directories
    mkdir -p $INSTALL_DIR $CONFIG_DIR
    chown $SERVICE_USER:$SERVICE_USER $INSTALL_DIR
    
    # Install systemd service
    cat > /etc/systemd/system/quantum-rerank.service << EOF
[Unit]
Description=QuantumRerank API Service
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=/usr/local/bin/quantum-rerank-server
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable quantum-rerank
    systemctl start quantum-rerank
    
    echo "Installation complete! Service started on port 8000"
}

# Main installation logic
main() {
    echo "QuantumRerank Universal Installer"
    echo "================================"
    
    ENV=$(detect_environment)
    echo "Detected environment: $ENV"
    
    case $ENV in
        kubernetes)
            install_kubernetes
            ;;
        docker)
            install_docker
            ;;
        native)
            install_native
            ;;
    esac
}

main "$@"
```

### 6. Configuration Management System

#### Configuration Templates
```yaml
# config/templates/enterprise.yaml
quantum_rerank:
  # Deployment information
  deployment:
    tier: "enterprise"
    organization: "${ORGANIZATION_NAME}"
    environment: "${ENVIRONMENT}"
    
  # API Configuration
  api:
    host: "0.0.0.0"
    port: 8000
    workers: ${CPU_CORES}
    max_request_size: "50MB"
    
  # Authentication
  auth:
    providers:
      - type: "api_key"
        enabled: true
      - type: "jwt"
        enabled: true
        issuer: "${OIDC_ISSUER}"
      - type: "saml"
        enabled: "${SAML_ENABLED:-false}"
        metadata_url: "${SAML_METADATA_URL}"
        
  # Performance
  performance:
    quantum_method: "hybrid"
    cache:
      enabled: true
      type: "${CACHE_TYPE:-redis}"
      ttl: 3600
      max_size: "${CACHE_SIZE:-10GB}"
    
  # Monitoring
  monitoring:
    metrics:
      enabled: true
      port: 9090
    tracing:
      enabled: true
      jaeger_endpoint: "${JAEGER_ENDPOINT}"
    logging:
      level: "${LOG_LEVEL:-INFO}"
      format: "json"
      
  # Compliance
  compliance:
    data_residency: "${DATA_RESIDENCY:-us-west-2}"
    encryption_at_rest: true
    audit_logging: true
    retention_days: ${RETENTION_DAYS:-90}
```

#### Configuration Validation CLI
```bash
# Validate configuration
quantum-rerank-config validate --config config.yaml

# Generate configuration from template
quantum-rerank-config generate \
    --template enterprise \
    --output config.yaml \
    --set organization=acme-corp \
    --set environment=production

# Test configuration
quantum-rerank-config test --config config.yaml --endpoint health
```

### 7. Enterprise Distribution Channels

#### Package Repositories
```bash
# Private PyPI for enterprise customers
pip install quantum-rerank-server \
    --extra-index-url https://enterprise.quantumrerank.ai/pypi/simple/ \
    --trusted-host enterprise.quantumrerank.ai

# Private Docker registry
docker pull enterprise.quantumrerank.ai/quantum-rerank:enterprise

# Private Helm repository
helm repo add quantumrerank-enterprise \
    https://enterprise.quantumrerank.ai/helm
```

#### License Management
```python
# License validation service
class LicenseValidator:
    def __init__(self, license_server: str):
        self.license_server = license_server
    
    def validate_license(self, license_key: str) -> LicenseInfo:
        """Validate enterprise license"""
        response = requests.post(
            f"{self.license_server}/validate",
            json={"license_key": license_key}
        )
        
        if response.status_code == 200:
            return LicenseInfo(**response.json())
        else:
            raise LicenseValidationError(response.json()["error"])
    
    def check_usage_limits(self, usage: UsageMetrics) -> bool:
        """Check if usage is within license limits"""
        license_info = self.get_license_info()
        return usage.requests_per_month <= license_info.max_requests
```

## Testing Strategy

### Package Testing
- **Installation Tests**: Verify all installation methods work correctly
- **Upgrade Tests**: Test seamless upgrades between versions
- **Configuration Tests**: Validate all configuration templates
- **Integration Tests**: Test with real cloud providers

### Distribution Testing
```bash
# Test matrix
environments:
  - ubuntu-20.04
  - ubuntu-22.04
  - centos-7
  - centos-8
  - amazonlinux-2
  - debian-11

installation_methods:
  - native_pip
  - docker
  - kubernetes_helm
  - terraform
  - cloudformation

test_scenarios:
  - fresh_install
  - upgrade_from_previous
  - configuration_migration
  - disaster_recovery
```

## Success Criteria
- [ ] One-line installation for all major deployment scenarios
- [ ] Automated package publishing to all ecosystems
- [ ] Enterprise-grade configuration management
- [ ] Infrastructure-as-code templates for all major clouds
- [ ] License management and validation system
- [ ] Comprehensive installation testing matrix
- [ ] Documentation for all installation methods

## Timeline
- **Week 1**: Package structure and PyPI distribution
- **Week 2**: Docker images and container distribution
- **Week 3**: Kubernetes Helm charts and operator
- **Week 4**: Infrastructure-as-code templates
- **Week 5**: Enterprise distribution channels
- **Week 6**: License management system
- **Week 7**: Installation scripts and automation
- **Week 8**: Testing, documentation, and release

## Dependencies
- Task 26: Client SDK Development (for client packages)
- Task 25: Production Deployment Guide (for deployment templates)
- Task 24: Deployment Configuration (for configuration management)
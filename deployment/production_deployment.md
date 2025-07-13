# QuantumRerank Production Deployment Guide

## Overview

This guide provides comprehensive procedures for deploying QuantumRerank in production environments. It covers infrastructure setup, application deployment, validation procedures, and operational best practices.

## Prerequisites

### System Requirements
- **Kubernetes cluster** (v1.24+) or **Docker with Compose** (v20.10+)
- **Container registry** access (Docker Hub, AWS ECR, GCR, etc.)
- **SSL certificates** for HTTPS endpoints
- **Resource allocation**: Minimum 8GB RAM, 4 CPU cores per instance
- **Storage**: 50GB+ for logs, cache, and temporary data

### Access Requirements
- **Infrastructure admin access** to target environment
- **Container registry** push/pull permissions
- **DNS management** access for domain configuration
- **SSL certificate** management access
- **Monitoring system** access (Prometheus, Grafana)

### Pre-Deployment Validation
- [ ] All Tasks 20-24 implemented and tested
- [ ] Staging environment successfully deployed and tested
- [ ] Load testing completed with production-level traffic
- [ ] Security assessment and penetration testing completed
- [ ] Performance targets validated (similarity <100ms, batch <500ms, memory <2GB)
- [ ] Monitoring and alerting systems configured
- [ ] Backup and recovery procedures documented and tested

## Deployment Phases

### Phase 1: Infrastructure Setup

#### 1.1 Environment Provisioning

**Cloud Infrastructure (Recommended)**
```bash
# AWS EKS cluster setup
eksctl create cluster \
  --name quantum-rerank-prod \
  --region us-west-2 \
  --node-type m5.xlarge \
  --nodes 3 \
  --nodes-min 3 \
  --nodes-max 10 \
  --managed

# Google GKE cluster setup
gcloud container clusters create quantum-rerank-prod \
  --zone us-west1-a \
  --machine-type n1-standard-4 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 10

# Azure AKS cluster setup
az aks create \
  --resource-group quantum-rerank-rg \
  --name quantum-rerank-prod \
  --node-count 3 \
  --enable-addons monitoring \
  --enable-cluster-autoscaler \
  --min-count 3 \
  --max-count 10
```

**On-Premises Kubernetes**
```bash
# Verify cluster is ready
kubectl cluster-info
kubectl get nodes

# Create namespace
kubectl create namespace quantum-rerank
kubectl label namespace quantum-rerank environment=production
```

#### 1.2 Network Security Configuration

**Network Policies**
```yaml
# Apply network isolation
kubectl apply -f deployment/k8s/network-policy.yaml

# Configure ingress rules
kubectl apply -f deployment/k8s/ingress.yaml
```

**SSL/TLS Setup**
```bash
# Install cert-manager (if not already installed)
kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Apply SSL certificate configuration
kubectl apply -f deployment/k8s/ssl-certificates.yaml
```

#### 1.3 Storage and Database Setup

**Persistent Storage**
```bash
# Create storage classes and persistent volumes
kubectl apply -f deployment/k8s/storage.yaml

# Verify storage is available
kubectl get pv,pvc -n quantum-rerank
```

**Redis Setup (for caching)**
```bash
# Deploy Redis cluster
helm install redis bitnami/redis \
  --namespace quantum-rerank \
  --set auth.enabled=true \
  --set auth.password="$(openssl rand -base64 32)" \
  --set master.persistence.enabled=true \
  --set replica.replicaCount=2
```

### Phase 2: Application Deployment

#### 2.1 Container Image Preparation

**Build Production Image**
```bash
# Navigate to project directory
cd /Users/alkist/Projects/QuantumRerank

# Build production image
docker build \
  --target production \
  --build-arg ENVIRONMENT=production \
  --build-arg VERSION=$(git describe --tags --always) \
  -f deployment/Dockerfile \
  -t quantum-rerank:$(git describe --tags --always) .

# Tag for registry
docker tag quantum-rerank:$(git describe --tags --always) \
  $REGISTRY/quantum-rerank:$(git describe --tags --always)

# Push to registry
docker push $REGISTRY/quantum-rerank:$(git describe --tags --always)
```

**Security Scanning**
```bash
# Scan for vulnerabilities
docker scan $REGISTRY/quantum-rerank:$(git describe --tags --always)

# Or use trivy
trivy image $REGISTRY/quantum-rerank:$(git describe --tags --always)
```

#### 2.2 Configuration Management

**Create Secrets**
```bash
# Create namespace secrets
kubectl create secret generic quantum-rerank-secrets \
  --namespace quantum-rerank \
  --from-literal=database-password="$(openssl rand -base64 32)" \
  --from-literal=jwt-secret="$(openssl rand -base64 64)" \
  --from-literal=api-key-salt="$(openssl rand -base64 32)"

# Create TLS secrets
kubectl create secret tls quantum-rerank-tls \
  --namespace quantum-rerank \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key
```

**Apply Configuration**
```bash
# Apply ConfigMaps
kubectl apply -f deployment/k8s/configmap.yaml -n quantum-rerank

# Update image in deployment
sed -i "s|quantum-rerank:latest|$REGISTRY/quantum-rerank:$(git describe --tags --always)|g" \
  deployment/k8s/deployment.yaml
```

#### 2.3 Service Deployment

**Deploy Core Services**
```bash
# Apply RBAC
kubectl apply -f deployment/k8s/secret.yaml -n quantum-rerank

# Deploy application
kubectl apply -f deployment/k8s/deployment.yaml -n quantum-rerank

# Deploy services
kubectl apply -f deployment/k8s/service.yaml -n quantum-rerank

# Deploy ingress
kubectl apply -f deployment/k8s/ingress.yaml -n quantum-rerank
```

**Verify Deployment**
```bash
# Check deployment status
kubectl rollout status deployment/quantum-rerank -n quantum-rerank

# Check pod status
kubectl get pods -n quantum-rerank -l app=quantum-rerank

# Check service endpoints
kubectl get svc -n quantum-rerank
kubectl get ingress -n quantum-rerank
```

### Phase 3: Production Validation

#### 3.1 Smoke Testing

**Basic Health Checks**
```bash
# Run health check script
./scripts/health-check.sh comprehensive

# Test API endpoints
curl -f https://api.quantumrerank.com/health
curl -f https://api.quantumrerank.com/health/ready
curl -f https://api.quantumrerank.com/metrics
```

**Functional Testing**
```bash
# Run deployment validation
python deployment/validation/smoke_tests.py

# Test with sample data
python deployment/validation/functional_tests.py
```

#### 3.2 Performance Validation

**Load Testing**
```bash
# Run performance validation
python deployment/validation/performance_validation.py

# Run load test
k6 run deployment/validation/load_test.js
```

**Performance Metrics Validation**
```bash
# Check PRD compliance
python deployment/validation/prd_compliance_check.py

# Verify response times
curl -w "@curl-format.txt" -s -o /dev/null https://api.quantumrerank.com/v1/similarity
```

#### 3.3 Security Validation

**Security Testing**
```bash
# Run security validation
python deployment/validation/security_validation.py

# SSL certificate validation
openssl s_client -connect api.quantumrerank.com:443 -servername api.quantumrerank.com

# Security headers check
curl -I https://api.quantumrerank.com/health
```

## Validation

### Post-Deployment Validation

**Infrastructure Validation**
```bash
# Verify all pods are running
kubectl get pods -n quantum-rerank
kubectl get svc -n quantum-rerank
kubectl get ingress -n quantum-rerank

# Check HPA status
kubectl get hpa -n quantum-rerank

# Verify RBAC configuration
kubectl auth can-i get pods --as=system:serviceaccount:quantum-rerank:quantum-rerank -n quantum-rerank
```

**Application Validation**
```bash
# Run comprehensive smoke tests
python deployment/validation/smoke_tests.py --base-url https://api.quantumrerank.com

# Run performance validation
python deployment/validation/performance_validation.py --base-url https://api.quantumrerank.com

# Test deployment pipeline
python deployment/test_deployment_pipeline.py --project-root .
```

**Security Validation**
```bash
# Verify SSL certificates
openssl s_client -connect api.quantumrerank.com:443 -servername api.quantumrerank.com

# Check security headers
curl -I https://api.quantumrerank.com/health

# Verify network policies
kubectl get networkpolicies -n quantum-rerank
```

## Post-Deployment Procedures

### Monitoring Setup

**Configure Dashboards**
```bash
# Import Grafana dashboards
curl -X POST \
  http://grafana.monitoring.svc.cluster.local:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @deployment/monitoring/grafana-dashboard.json
```

**Alert Configuration**
```bash
# Apply alert rules
kubectl apply -f deployment/monitoring/alert-rules.yaml -n quantum-rerank
```

### Operational Validation

**Team Readiness**
- [ ] Operations team trained on procedures
- [ ] Incident response procedures tested
- [ ] Escalation paths verified
- [ ] Documentation accessible to all team members

**Business Readiness**
- [ ] Service level agreements (SLAs) defined
- [ ] Customer onboarding procedures ready
- [ ] Billing and usage tracking operational
- [ ] Support procedures documented

## Deployment Automation

### Using Deployment Scripts

**Automated Deployment**
```bash
# Deploy to production
./scripts/deploy.sh \
  --environment production \
  --type kubernetes \
  --version $(git describe --tags --always) \
  --registry $REGISTRY

# Validate deployment
./scripts/validate_deployment.sh --environment production
```

### CI/CD Integration

**GitHub Actions Example**
```yaml
name: Deploy to Production
on:
  push:
    tags:
      - 'v*'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Deploy to Production
      run: |
        ./scripts/deploy.sh \
          --environment production \
          --type kubernetes \
          --version ${{ github.ref_name }} \
          --registry ${{ secrets.CONTAINER_REGISTRY }}
```

## Rollback Procedures

### Automated Rollback
```bash
# Rollback to previous version
./scripts/rollback.sh \
  --environment production \
  --type kubernetes

# Rollback to specific version
./scripts/rollback.sh \
  --environment production \
  --type kubernetes \
  --revision 3
```

### Manual Rollback
```bash
# Kubernetes rollback
kubectl rollout undo deployment/quantum-rerank -n quantum-rerank

# Verify rollback
kubectl rollout status deployment/quantum-rerank -n quantum-rerank
```

## Troubleshooting

### Common Issues

**Pod Startup Issues**
```bash
# Check pod logs
kubectl logs -n quantum-rerank -l app=quantum-rerank

# Check pod events
kubectl describe pod -n quantum-rerank [POD_NAME]

# Check resource constraints
kubectl top pods -n quantum-rerank
```

**Service Connectivity Issues**
```bash
# Test service endpoints
kubectl exec -n quantum-rerank -it [POD_NAME] -- curl localhost:8000/health

# Check service configuration
kubectl get svc -n quantum-rerank -o yaml

# Check ingress configuration
kubectl describe ingress -n quantum-rerank
```

**Performance Issues**
```bash
# Check resource usage
kubectl top pods -n quantum-rerank
kubectl top nodes

# Check application metrics
curl https://api.quantumrerank.com/metrics

# Review logs for performance issues
kubectl logs -n quantum-rerank -l app=quantum-rerank | grep -i "performance\|slow\|timeout"
```

## Maintenance Procedures

### Regular Maintenance

**Weekly Tasks**
- Review performance metrics and trends
- Check security alerts and update dependencies
- Validate backup and recovery procedures
- Review capacity planning metrics

**Monthly Tasks**
- Security vulnerability assessment
- Performance optimization review
- Disaster recovery testing
- Cost optimization analysis

### Updates and Patches

**Application Updates**
```bash
# Deploy new version
./scripts/deploy.sh \
  --environment production \
  --type kubernetes \
  --version [NEW_VERSION] \
  --registry $REGISTRY

# Monitor deployment
kubectl rollout status deployment/quantum-rerank -n quantum-rerank
```

**Security Patches**
```bash
# Update base images
docker build --no-cache \
  --target production \
  --build-arg ENVIRONMENT=production \
  -f deployment/Dockerfile \
  -t quantum-rerank:[NEW_VERSION] .

# Deploy updated image
kubectl set image deployment/quantum-rerank \
  quantum-rerank=$REGISTRY/quantum-rerank:[NEW_VERSION] \
  -n quantum-rerank
```

## Success Criteria

### Technical Success
- [ ] Service deployed and operational
- [ ] All health checks passing consistently
- [ ] Performance targets met (similarity <100ms, batch <500ms, memory <2GB)
- [ ] Security validations passed
- [ ] Monitoring and alerting functional
- [ ] Auto-scaling working as expected

### Operational Success
- [ ] Team trained and ready
- [ ] Incident response procedures tested
- [ ] Backup and recovery validated
- [ ] Documentation complete and accessible
- [ ] SLAs defined and monitored

### Business Success
- [ ] Customer onboarding ready
- [ ] Billing and usage tracking operational
- [ ] Support procedures documented
- [ ] Compliance requirements met
- [ ] Performance baselines established

## Support Contacts

### Emergency Contacts
- **Production Issues**: [production-alerts@company.com]
- **Security Incidents**: [security@company.com]
- **Infrastructure Issues**: [infrastructure@company.com]

### Escalation Procedures
1. **Level 1**: On-call engineer responds within 15 minutes
2. **Level 2**: Senior engineer engaged within 30 minutes
3. **Level 3**: Engineering manager and leadership within 1 hour

This deployment guide provides a comprehensive approach to production deployment while maintaining the reliability, performance, and security standards required for the QuantumRerank system.
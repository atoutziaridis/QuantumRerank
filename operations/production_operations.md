# QuantumRerank Production Operations Guide

## Overview

This guide provides comprehensive operational procedures for managing QuantumRerank in production environments. It covers daily operations, monitoring, incident response, and maintenance procedures.

## Service Architecture Overview

### Core Components
- **API Gateway**: Load balancing and SSL termination
- **Application Pods**: QuantumRerank API instances (3+ replicas)
- **Cache Layer**: Redis cluster for performance optimization  
- **Monitoring Stack**: Prometheus, Grafana, alerting
- **Load Balancer**: Cloud provider or Kubernetes ingress

### Key Metrics
- **Response Time**: <200ms (target)
- **Similarity Computation**: <100ms (PRD target)
- **Batch Processing**: <500ms (PRD target)
- **Memory Usage**: <2GB per instance (PRD target)
- **Error Rate**: <1% (target)
- **Uptime**: >99.9% (SLA target)

## Daily Operations

### Morning Health Check (5 minutes)
```bash
# 1. Check overall service health
kubectl get pods -n quantum-rerank
kubectl get svc -n quantum-rerank
kubectl get ingress -n quantum-rerank

# 2. Verify API responsiveness
curl -f https://api.quantumrerank.com/health
curl -f https://api.quantumrerank.com/health/ready

# 3. Check resource usage
kubectl top pods -n quantum-rerank
kubectl top nodes

# 4. Review overnight alerts
# Check Grafana dashboard and Slack/email alerts

# 5. Verify auto-scaling status
kubectl get hpa -n quantum-rerank
```

### Performance Monitoring (10 minutes)
```bash
# Check performance metrics
python deployment/validation/performance_validation.py --base-url https://api.quantumrerank.com

# Review Grafana dashboards
# - API Performance Dashboard: Response times, throughput
# - System Resources Dashboard: CPU, memory, disk usage
# - Business Metrics Dashboard: Usage patterns, success rates

# Check for performance degradation
curl -s https://api.quantumrerank.com/metrics | grep -E "(response_time|memory_usage|error_rate)"
```

### Log Review (10 minutes)
```bash
# Check recent error logs
kubectl logs -n quantum-rerank -l app=quantum-rerank --since=24h | grep -i error

# Review performance warnings
kubectl logs -n quantum-rerank -l app=quantum-rerank --since=24h | grep -i "slow\|timeout\|performance"

# Check security events
kubectl logs -n quantum-rerank -l app=quantum-rerank --since=24h | grep -i "security\|auth\|blocked"
```

## Monitoring and Alerting

### Key Dashboards

**1. Service Health Dashboard**
- Overall service status
- Pod health and readiness
- API response codes distribution
- Active connections and request rate

**2. Performance Dashboard**
- Similarity computation times (P50, P95, P99)
- Batch processing performance
- Memory and CPU utilization
- Cache hit rates

**3. Business Metrics Dashboard**
- API usage by endpoint
- User tier distribution
- Quantum vs classical method usage
- Cost and efficiency metrics

### Critical Alerts

**P0 (Critical - Immediate Response)**
- Service completely unavailable (>5 minutes)
- Error rate >10% for >2 minutes
- Response time >1000ms for >5 minutes
- Memory usage >90% for >5 minutes

**P1 (High - Response within 30 minutes)**
- Response time >500ms for >10 minutes
- Error rate >5% for >5 minutes
- CPU usage >80% for >15 minutes
- Cache miss rate >80% for >10 minutes

**P2 (Medium - Response within 2 hours)**
- Response time >200ms for >30 minutes
- Memory usage >70% for >30 minutes
- Disk usage >80%
- Security events detected

### Alert Response Procedures

**Service Unavailable (P0)**
```bash
# 1. Check pod status
kubectl get pods -n quantum-rerank -o wide

# 2. Check recent deployments
kubectl rollout history deployment/quantum-rerank -n quantum-rerank

# 3. Check resource constraints
kubectl describe pods -n quantum-rerank

# 4. Check node health
kubectl get nodes
kubectl describe nodes

# 5. If needed, restart service
kubectl rollout restart deployment/quantum-rerank -n quantum-rerank
```

**High Error Rate (P0/P1)**
```bash
# 1. Check error logs immediately
kubectl logs -n quantum-rerank -l app=quantum-rerank --tail=100 | grep -i error

# 2. Check recent configuration changes
kubectl get configmap quantum-rerank-config -n quantum-rerank -o yaml

# 3. Validate external dependencies
curl -f https://api.quantumrerank.com/health/detailed

# 4. Check resource exhaustion
kubectl top pods -n quantum-rerank

# 5. Consider rollback if recent deployment
./scripts/rollback.sh --environment production --type kubernetes
```

**Performance Degradation (P1/P2)**
```bash
# 1. Check current performance
python deployment/validation/performance_validation.py

# 2. Review resource usage trends
kubectl top pods -n quantum-rerank --sort-by=memory
kubectl top pods -n quantum-rerank --sort-by=cpu

# 3. Check for memory leaks
kubectl logs -n quantum-rerank -l app=quantum-rerank | grep -i "memory\|leak\|oom"

# 4. Scale up if needed
kubectl scale deployment quantum-rerank --replicas=5 -n quantum-rerank

# 5. Clear cache if applicable
kubectl exec -n quantum-rerank deployment/redis -- redis-cli FLUSHALL
```

## Incident Response

### Incident Severity Levels

**P0 (Critical)**
- Complete service outage
- Data loss or corruption
- Security breach
- Response time: Immediate (within 15 minutes)

**P1 (High)**
- Significant performance degradation
- Partial service disruption
- High error rates
- Response time: Within 30 minutes

**P2 (Medium)**
- Minor performance issues
- Non-critical feature problems
- Monitoring alerts
- Response time: Within 2 hours

**P3 (Low)**
- Cosmetic issues
- Enhancement requests
- Documentation updates
- Response time: Within 24 hours

### Incident Response Process

**1. Detection and Assessment (0-5 minutes)**
```bash
# Incident detected via:
# - Automated alerts (Prometheus/Grafana)
# - Customer reports
# - Monitoring dashboard abnormalities

# Immediate assessment:
# 1. Confirm the incident
curl -f https://api.quantumrerank.com/health

# 2. Determine severity
./scripts/health-check.sh comprehensive

# 3. Check scope of impact
kubectl get pods -n quantum-rerank
kubectl get svc -n quantum-rerank
```

**2. Initial Response (5-15 minutes)**
```bash
# 1. Notify team (P0/P1 incidents)
# - Slack: #quantum-rerank-alerts
# - Email: production-alerts@company.com
# - PagerDuty: For P0 incidents

# 2. Begin investigation
kubectl logs -n quantum-rerank -l app=quantum-rerank --tail=200

# 3. Document incident start time and symptoms
# Create incident record in issue tracking system

# 4. Engage additional resources if needed
```

**3. Investigation and Mitigation (15-60 minutes)**
```bash
# 1. Identify root cause
# - Check recent changes (deployments, config updates)
# - Review error logs and metrics
# - Test individual components

# 2. Implement immediate mitigation
# - Rollback if deployment-related
# - Restart services if needed
# - Scale resources if capacity issue
# - Apply configuration fixes

# 3. Monitor recovery
./scripts/health-check.sh comprehensive
python deployment/validation/performance_validation.py
```

**4. Recovery Validation (60-90 minutes)**
```bash
# 1. Verify full service restoration
python deployment/validation/smoke_tests.py

# 2. Confirm performance targets met
python deployment/validation/performance_validation.py

# 3. Monitor for stability (30 minutes minimum)
# 4. Notify stakeholders of resolution
```

**5. Post-Incident Review (Within 24 hours)**
- Document timeline and root cause
- Identify preventive measures
- Update runbooks and procedures
- Schedule technical debt tasks if needed

## Maintenance Procedures

### Weekly Maintenance Tasks

**Performance Review (30 minutes)**
```bash
# 1. Generate weekly performance report
python deployment/validation/performance_validation.py --output weekly_performance.json

# 2. Review trends in Grafana
# - Response time trends
# - Resource utilization trends
# - Error rate patterns
# - Usage growth patterns

# 3. Identify optimization opportunities
# - Slow queries or endpoints
# - Resource bottlenecks
# - Cache optimization potential

# 4. Plan capacity adjustments if needed
```

**Security Review (20 minutes)**
```bash
# 1. Review security events
kubectl logs -n quantum-rerank -l app=quantum-rerank --since=168h | grep -i security

# 2. Check for suspicious activity
# - Rate limiting violations
# - Authentication failures
# - Unusual access patterns

# 3. Validate SSL certificates
openssl s_client -connect api.quantumrerank.com:443 -servername api.quantumrerank.com | grep -E "(Verify|expire)"

# 4. Review access logs
```

**Backup Validation (15 minutes)**
```bash
# 1. Verify backup completion
# Check backup system status

# 2. Test restore procedure (monthly)
# Follow disaster recovery runbook

# 3. Validate backup integrity
# Check backup file sizes and checksums
```

### Monthly Maintenance Tasks

**Capacity Planning Review**
- Analyze usage growth trends
- Forecast resource requirements
- Plan infrastructure scaling
- Review cost optimization opportunities

**Security Assessment**
- Update dependencies and base images
- Review security configurations
- Conduct vulnerability scans
- Update access controls

**Performance Optimization**
- Analyze performance bottlenecks
- Optimize slow queries or operations
- Review caching strategies
- Update performance baselines

**Documentation Updates**
- Update operational procedures
- Review and update runbooks
- Update architecture diagrams
- Validate contact information

## Scaling Operations

### Manual Scaling

**Scale Up (High Load)**
```bash
# 1. Scale application pods
kubectl scale deployment quantum-rerank --replicas=8 -n quantum-rerank

# 2. Verify scaling
kubectl get pods -n quantum-rerank
kubectl rollout status deployment/quantum-rerank -n quantum-rerank

# 3. Monitor performance
./scripts/health-check.sh comprehensive

# 4. Update auto-scaling if needed
kubectl patch hpa quantum-rerank-hpa -n quantum-rerank -p '{"spec":{"maxReplicas":12}}'
```

**Scale Down (Low Load)**
```bash
# 1. Check current load
kubectl top pods -n quantum-rerank

# 2. Scale down gradually
kubectl scale deployment quantum-rerank --replicas=3 -n quantum-rerank

# 3. Monitor for performance impact
python deployment/validation/performance_validation.py

# 4. Adjust auto-scaling parameters
kubectl patch hpa quantum-rerank-hpa -n quantum-rerank -p '{"spec":{"minReplicas":3}}'
```

### Auto-Scaling Configuration

**Current Auto-Scaling Settings**
- Min replicas: 3
- Max replicas: 10
- CPU target: 70%
- Memory target: 80%
- Scale-up stabilization: 60 seconds
- Scale-down stabilization: 300 seconds

**Adjust Auto-Scaling**
```bash
# Update HPA configuration
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
  maxReplicas: 15
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
EOF
```

## Configuration Management

### Configuration Updates

**Safe Configuration Change Process**
```bash
# 1. Backup current configuration
kubectl get configmap quantum-rerank-config -n quantum-rerank -o yaml > config-backup-$(date +%Y%m%d).yaml

# 2. Test changes in staging first
# Apply to staging environment and validate

# 3. Apply to production with validation
kubectl apply -f deployment/k8s/configmap.yaml -n quantum-rerank

# 4. Monitor for issues
kubectl rollout status deployment/quantum-rerank -n quantum-rerank
./scripts/health-check.sh comprehensive

# 5. Rollback if needed
kubectl apply -f config-backup-$(date +%Y%m%d).yaml
```

### Secret Management

**Update Secrets Safely**
```bash
# 1. Backup current secrets (metadata only)
kubectl get secret quantum-rerank-secrets -n quantum-rerank -o yaml --export > secrets-backup-$(date +%Y%m%d).yaml

# 2. Update secrets
kubectl create secret generic quantum-rerank-secrets-new \
  --namespace quantum-rerank \
  --from-literal=api-key="new-api-key" \
  --dry-run=client -o yaml | kubectl apply -f -

# 3. Rolling restart to pick up new secrets
kubectl rollout restart deployment/quantum-rerank -n quantum-rerank

# 4. Verify functionality
./scripts/health-check.sh comprehensive
```

## Troubleshooting Common Issues

### Pod Startup Issues

**Pod Stuck in Pending**
```bash
# Check resource availability
kubectl describe pod [POD_NAME] -n quantum-rerank
kubectl get nodes
kubectl top nodes

# Check PVC status
kubectl get pv,pvc -n quantum-rerank
```

**Pod Stuck in CrashLoopBackOff**
```bash
# Check logs
kubectl logs [POD_NAME] -n quantum-rerank --previous

# Check startup probe configuration
kubectl describe pod [POD_NAME] -n quantum-rerank

# Check resource limits
kubectl get pod [POD_NAME] -n quantum-rerank -o yaml | grep -A 10 resources
```

### Performance Issues

**High Response Times**
```bash
# 1. Check resource usage
kubectl top pods -n quantum-rerank

# 2. Check for resource limits
kubectl describe pods -n quantum-rerank | grep -A 5 Limits

# 3. Scale up if needed
kubectl scale deployment quantum-rerank --replicas=6 -n quantum-rerank

# 4. Check cache performance
kubectl exec -n quantum-rerank deployment/redis -- redis-cli info stats
```

**Memory Issues**
```bash
# 1. Check memory usage
kubectl top pods -n quantum-rerank --sort-by=memory

# 2. Look for memory leaks
kubectl logs -n quantum-rerank -l app=quantum-rerank | grep -i "memory\|oom"

# 3. Check garbage collection
kubectl logs -n quantum-rerank -l app=quantum-rerank | grep -i "gc"

# 4. Restart pods if needed
kubectl delete pod -n quantum-rerank -l app=quantum-rerank
```

### Connectivity Issues

**Service Not Accessible**
```bash
# 1. Check service endpoints
kubectl get endpoints -n quantum-rerank

# 2. Check ingress configuration
kubectl describe ingress -n quantum-rerank

# 3. Test internal connectivity
kubectl exec -n quantum-rerank -it [POD_NAME] -- curl localhost:8000/health

# 4. Check DNS resolution
kubectl exec -n quantum-rerank -it [POD_NAME] -- nslookup quantum-rerank-service
```

## Emergency Procedures

### Complete Service Recovery

**If All Pods Are Down**
```bash
# 1. Check cluster health
kubectl get nodes
kubectl cluster-info

# 2. Check namespace status
kubectl get all -n quantum-rerank

# 3. Redeploy if necessary
kubectl apply -f deployment/k8s/ -n quantum-rerank

# 4. Monitor recovery
kubectl get pods -n quantum-rerank -w
```

### Disaster Recovery

**Data Center Outage**
1. Activate backup data center or region
2. Update DNS to point to backup location
3. Restore from last known good backup
4. Validate service functionality
5. Monitor for data consistency issues

**Complete Infrastructure Failure**
1. Deploy to alternative cloud provider or region
2. Restore configuration and secrets
3. Restore data from backups
4. Update external dependencies
5. Perform full validation suite

## Performance Baselines

### Normal Operating Ranges

**Response Times**
- Health endpoints: 10-50ms
- Similarity computation: 50-100ms
- Batch processing: 200-500ms
- Error responses: 5-20ms

**Resource Usage**
- CPU per pod: 10-70%
- Memory per pod: 0.5-1.5GB
- Cache hit rate: 15-30%
- Network I/O: <100MB/s per pod

**Business Metrics**
- Request rate: 50-200 RPS
- Error rate: <0.5%
- User distribution: 60% standard, 30% premium, 10% enterprise

## Contacts and Escalation

### On-Call Rotation
- **Primary**: Senior Engineer (Level 2)
- **Secondary**: Engineering Manager (Level 3)
- **Backup**: Principal Engineer (Level 3)

### Emergency Contacts
- **Production Issues**: production-alerts@company.com
- **Security Incidents**: security@company.com
- **Infrastructure**: infrastructure@company.com
- **Management**: engineering-mgmt@company.com

### Escalation Timeline
- **Level 1 Response**: 15 minutes (on-call engineer)
- **Level 2 Escalation**: 30 minutes (senior engineer/manager)
- **Level 3 Escalation**: 1 hour (principal engineer/director)
- **Executive Escalation**: 2 hours (VP Engineering/CTO)

This operations guide provides the foundation for reliable production operations of the QuantumRerank system.
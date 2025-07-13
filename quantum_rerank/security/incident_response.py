"""
Security Incident Response System for QuantumRerank.

This module provides automated security incident response including
incident classification, automated response actions, escalation procedures,
and notification systems for comprehensive security incident management.
"""

import time
import json
import smtplib
import hashlib
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
from datetime import datetime, timedelta

from ..utils.logging_config import get_logger
from ..utils.exceptions import SecurityError
from .security_monitor import ThreatDetection, ThreatSeverity, ThreatType

logger = get_logger(__name__)


class IncidentSeverity(Enum):
    """Incident severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(Enum):
    """Incident status values."""
    NEW = "new"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    RESOLVED = "resolved"
    CLOSED = "closed"


class ResponseAction(Enum):
    """Available response actions."""
    BLOCK_IP = "block_ip"
    DISABLE_USER = "disable_user"
    REVOKE_TOKEN = "revoke_token"
    INCREASE_LOGGING = "increase_logging"
    ALERT_ADMIN = "alert_admin"
    ISOLATE_SYSTEM = "isolate_system"
    BACKUP_DATA = "backup_data"
    ROTATE_KEYS = "rotate_keys"
    SCALE_RESOURCES = "scale_resources"
    ENABLE_DDOS_PROTECTION = "enable_ddos_protection"


class NotificationChannel(Enum):
    """Notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    DASHBOARD = "dashboard"


@dataclass
class SecurityIncident:
    """Security incident information."""
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    created_at: float
    updated_at: float
    source_threats: List[ThreatDetection] = field(default_factory=list)
    affected_systems: List[str] = field(default_factory=list)
    affected_users: List[str] = field(default_factory=list)
    response_actions: List[str] = field(default_factory=list)
    escalation_level: int = 0
    assigned_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityResponse:
    """Security response configuration and result."""
    response_id: str
    incident_id: str
    action: ResponseAction
    timestamp: float
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class NotificationConfig:
    """Notification configuration."""
    channel: NotificationChannel
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    severity_threshold: IncidentSeverity = IncidentSeverity.MEDIUM


class IncidentClassifier:
    """Classifies security threats into incidents."""
    
    def __init__(self):
        """Initialize incident classifier."""
        self.classification_rules = self._initialize_classification_rules()
        self.logger = logger
        
        logger.info("Initialized IncidentClassifier")
    
    def _initialize_classification_rules(self) -> Dict[ThreatType, Dict[str, Any]]:
        """Initialize threat-to-incident classification rules."""
        return {
            ThreatType.BRUTE_FORCE: {
                "base_severity": IncidentSeverity.MEDIUM,
                "escalation_factors": {
                    "multiple_ips": IncidentSeverity.HIGH,
                    "admin_targets": IncidentSeverity.HIGH,
                    "successful_breach": IncidentSeverity.CRITICAL
                },
                "response_actions": [
                    ResponseAction.BLOCK_IP,
                    ResponseAction.INCREASE_LOGGING,
                    ResponseAction.ALERT_ADMIN
                ]
            },
            ThreatType.DDOS: {
                "base_severity": IncidentSeverity.HIGH,
                "escalation_factors": {
                    "service_degradation": IncidentSeverity.CRITICAL,
                    "sustained_attack": IncidentSeverity.CRITICAL
                },
                "response_actions": [
                    ResponseAction.ENABLE_DDOS_PROTECTION,
                    ResponseAction.SCALE_RESOURCES,
                    ResponseAction.ALERT_ADMIN
                ]
            },
            ThreatType.INJECTION: {
                "base_severity": IncidentSeverity.HIGH,
                "escalation_factors": {
                    "data_access": IncidentSeverity.CRITICAL,
                    "privilege_escalation": IncidentSeverity.CRITICAL
                },
                "response_actions": [
                    ResponseAction.BLOCK_IP,
                    ResponseAction.INCREASE_LOGGING,
                    ResponseAction.BACKUP_DATA,
                    ResponseAction.ALERT_ADMIN
                ]
            },
            ThreatType.DATA_EXFILTRATION: {
                "base_severity": IncidentSeverity.CRITICAL,
                "escalation_factors": {
                    "large_volume": IncidentSeverity.CRITICAL,
                    "sensitive_data": IncidentSeverity.CRITICAL
                },
                "response_actions": [
                    ResponseAction.BLOCK_IP,
                    ResponseAction.DISABLE_USER,
                    ResponseAction.BACKUP_DATA,
                    ResponseAction.ROTATE_KEYS,
                    ResponseAction.ALERT_ADMIN
                ]
            },
            ThreatType.UNAUTHORIZED_ACCESS: {
                "base_severity": IncidentSeverity.HIGH,
                "escalation_factors": {
                    "admin_access": IncidentSeverity.CRITICAL,
                    "system_modification": IncidentSeverity.CRITICAL
                },
                "response_actions": [
                    ResponseAction.REVOKE_TOKEN,
                    ResponseAction.DISABLE_USER,
                    ResponseAction.INCREASE_LOGGING,
                    ResponseAction.ALERT_ADMIN
                ]
            },
            ThreatType.QUANTUM_ATTACK: {
                "base_severity": IncidentSeverity.MEDIUM,
                "escalation_factors": {
                    "side_channel_exploit": IncidentSeverity.HIGH,
                    "circuit_manipulation": IncidentSeverity.HIGH,
                    "parameter_tampering": IncidentSeverity.HIGH
                },
                "response_actions": [
                    ResponseAction.ISOLATE_SYSTEM,
                    ResponseAction.INCREASE_LOGGING,
                    ResponseAction.ALERT_ADMIN
                ]
            }
        }
    
    def classify_threat(self, threat: ThreatDetection) -> SecurityIncident:
        """
        Classify threat into security incident.
        
        Args:
            threat: Threat detection to classify
            
        Returns:
            SecurityIncident created from threat
        """
        rules = self.classification_rules.get(threat.threat_type, {})
        base_severity = rules.get("base_severity", IncidentSeverity.MEDIUM)
        
        # Determine final severity based on escalation factors
        final_severity = base_severity
        escalation_factors = rules.get("escalation_factors", {})
        
        for factor, severity in escalation_factors.items():
            if self._check_escalation_factor(threat, factor):
                if severity.value > final_severity.value:
                    final_severity = severity
        
        # Generate incident
        incident_id = f"inc_{hashlib.md5(f'{threat.threat_id}_{time.time()}'.encode()).hexdigest()[:12]}"
        
        incident = SecurityIncident(
            incident_id=incident_id,
            title=f"{threat.threat_type.value.replace('_', ' ').title()} Detected",
            description=f"Security threat detected with {threat.confidence:.2f} confidence",
            severity=final_severity,
            status=IncidentStatus.NEW,
            created_at=time.time(),
            updated_at=time.time(),
            source_threats=[threat],
            affected_systems=self._extract_affected_systems(threat),
            affected_users=self._extract_affected_users(threat),
            response_actions=rules.get("response_actions", []),
            metadata={
                "threat_confidence": threat.confidence,
                "threat_indicators": threat.indicators,
                "escalation_factors": [
                    factor for factor in escalation_factors.keys()
                    if self._check_escalation_factor(threat, factor)
                ]
            }
        )
        
        self.logger.info(f"Classified threat {threat.threat_id} as incident {incident_id} with severity {final_severity.value}")
        return incident
    
    def _check_escalation_factor(self, threat: ThreatDetection, factor: str) -> bool:
        """Check if escalation factor applies to threat."""
        indicators = threat.indicators
        
        escalation_checks = {
            "multiple_ips": lambda: len(indicators.get("top_source_ips", [])) > 1,
            "admin_targets": lambda: any("admin" in str(v).lower() for v in indicators.values()),
            "successful_breach": lambda: indicators.get("authentication_success", False),
            "service_degradation": lambda: indicators.get("request_rate", 0) > 5000,
            "sustained_attack": lambda: indicators.get("time_span", 0) > 300,
            "data_access": lambda: indicators.get("data_accessed", False),
            "privilege_escalation": lambda: indicators.get("privilege_change", False),
            "large_volume": lambda: indicators.get("total_data_bytes", 0) > 100 * 1024 * 1024,
            "sensitive_data": lambda: indicators.get("sensitive_data_accessed", False),
            "admin_access": lambda: indicators.get("admin_access_attempted", False),
            "system_modification": lambda: indicators.get("system_modified", False),
            "side_channel_exploit": lambda: "side_channel" in str(indicators),
            "circuit_manipulation": lambda: indicators.get("circuit_modified", False),
            "parameter_tampering": lambda: indicators.get("parameter_anomaly", 0) > 0.8
        }
        
        check_func = escalation_checks.get(factor)
        if check_func:
            try:
                return check_func()
            except:
                return False
        return False
    
    def _extract_affected_systems(self, threat: ThreatDetection) -> List[str]:
        """Extract affected systems from threat."""
        systems = []
        
        # Extract from threat indicators
        if "endpoint" in threat.indicators:
            systems.append(threat.indicators["endpoint"])
        
        if "service" in threat.indicators:
            systems.append(threat.indicators["service"])
        
        # Default systems based on threat type
        if threat.threat_type == ThreatType.QUANTUM_ATTACK:
            systems.append("quantum_engine")
        elif threat.threat_type == ThreatType.DDOS:
            systems.append("api_gateway")
        
        return list(set(systems))
    
    def _extract_affected_users(self, threat: ThreatDetection) -> List[str]:
        """Extract affected users from threat."""
        users = []
        
        # Extract from source events
        for event in threat.source_events:
            if event.user_id:
                users.append(event.user_id)
        
        # Extract from indicators
        if "user_id" in threat.indicators:
            users.append(threat.indicators["user_id"])
        
        return list(set(users))


class ResponseActionManager:
    """Manages automated response actions."""
    
    def __init__(self):
        """Initialize response action manager."""
        self.action_handlers = self._initialize_action_handlers()
        self.logger = logger
        
        logger.info("Initialized ResponseActionManager")
    
    def _initialize_action_handlers(self) -> Dict[ResponseAction, Callable]:
        """Initialize action handler functions."""
        return {
            ResponseAction.BLOCK_IP: self._block_ip,
            ResponseAction.DISABLE_USER: self._disable_user,
            ResponseAction.REVOKE_TOKEN: self._revoke_token,
            ResponseAction.INCREASE_LOGGING: self._increase_logging,
            ResponseAction.ALERT_ADMIN: self._alert_admin,
            ResponseAction.ISOLATE_SYSTEM: self._isolate_system,
            ResponseAction.BACKUP_DATA: self._backup_data,
            ResponseAction.ROTATE_KEYS: self._rotate_keys,
            ResponseAction.SCALE_RESOURCES: self._scale_resources,
            ResponseAction.ENABLE_DDOS_PROTECTION: self._enable_ddos_protection
        }
    
    def execute_response(self, incident: SecurityIncident, 
                        action: ResponseAction) -> SecurityResponse:
        """
        Execute response action for incident.
        
        Args:
            incident: Security incident
            action: Response action to execute
            
        Returns:
            SecurityResponse with execution results
        """
        response_id = f"resp_{hashlib.md5(f'{incident.incident_id}_{action.value}_{time.time()}'.encode()).hexdigest()[:12]}"
        
        self.logger.info(f"Executing response action {action.value} for incident {incident.incident_id}")
        
        try:
            handler = self.action_handlers.get(action)
            if not handler:
                raise SecurityError(f"No handler for action {action.value}")
            
            result = handler(incident)
            
            return SecurityResponse(
                response_id=response_id,
                incident_id=incident.incident_id,
                action=action,
                timestamp=time.time(),
                success=True,
                details=result
            )
            
        except Exception as e:
            self.logger.error(f"Response action {action.value} failed: {e}")
            return SecurityResponse(
                response_id=response_id,
                incident_id=incident.incident_id,
                action=action,
                timestamp=time.time(),
                success=False,
                error_message=str(e)
            )
    
    def _block_ip(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Block IP addresses involved in incident."""
        blocked_ips = []
        
        for threat in incident.source_threats:
            if "source_ip" in threat.indicators:
                ip = threat.indicators["source_ip"]
                # In real implementation, would integrate with firewall/WAF
                self.logger.warning(f"BLOCKING IP: {ip}")
                blocked_ips.append(ip)
            
            if "top_source_ips" in threat.indicators:
                for ip in threat.indicators["top_source_ips"]:
                    if isinstance(ip, str):
                        self.logger.warning(f"BLOCKING IP: {ip}")
                        blocked_ips.append(ip)
        
        return {"blocked_ips": blocked_ips}
    
    def _disable_user(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Disable user accounts involved in incident."""
        disabled_users = []
        
        for user in incident.affected_users:
            # In real implementation, would integrate with user management system
            self.logger.warning(f"DISABLING USER: {user}")
            disabled_users.append(user)
        
        return {"disabled_users": disabled_users}
    
    def _revoke_token(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Revoke authentication tokens."""
        revoked_tokens = []
        
        for threat in incident.source_threats:
            for event in threat.source_events:
                # Extract token information from events
                if "token" in event.details:
                    token = event.details["token"]
                    # In real implementation, would integrate with auth system
                    self.logger.warning(f"REVOKING TOKEN: {token[:8]}...")
                    revoked_tokens.append(token[:8])
        
        return {"revoked_tokens": len(revoked_tokens)}
    
    def _increase_logging(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Increase logging level for affected systems."""
        affected_systems = incident.affected_systems
        
        # In real implementation, would configure logging systems
        self.logger.info(f"INCREASING LOGGING for systems: {affected_systems}")
        
        return {"logging_increased": affected_systems}
    
    def _alert_admin(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Send alert to administrators."""
        # In real implementation, would send notifications
        self.logger.critical(f"ADMIN ALERT: Incident {incident.incident_id} - {incident.title}")
        
        return {"alert_sent": True, "admin_notified": True}
    
    def _isolate_system(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Isolate affected systems."""
        isolated_systems = []
        
        for system in incident.affected_systems:
            # In real implementation, would isolate systems
            self.logger.warning(f"ISOLATING SYSTEM: {system}")
            isolated_systems.append(system)
        
        return {"isolated_systems": isolated_systems}
    
    def _backup_data(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Create emergency data backup."""
        # In real implementation, would trigger backup systems
        self.logger.info(f"CREATING EMERGENCY BACKUP for incident {incident.incident_id}")
        
        return {"backup_created": True, "backup_timestamp": time.time()}
    
    def _rotate_keys(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Rotate cryptographic keys."""
        # In real implementation, would rotate keys
        self.logger.warning(f"ROTATING KEYS due to incident {incident.incident_id}")
        
        return {"keys_rotated": True, "rotation_timestamp": time.time()}
    
    def _scale_resources(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Scale resources to handle incident."""
        # In real implementation, would scale infrastructure
        self.logger.info(f"SCALING RESOURCES for incident {incident.incident_id}")
        
        return {"resources_scaled": True, "scale_factor": 2.0}
    
    def _enable_ddos_protection(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Enable DDoS protection mechanisms."""
        # In real implementation, would enable DDoS protection
        self.logger.warning(f"ENABLING DDOS PROTECTION for incident {incident.incident_id}")
        
        return {"ddos_protection_enabled": True}


class SecurityNotificationSystem:
    """Handles security notifications across multiple channels."""
    
    def __init__(self, notification_configs: Optional[List[NotificationConfig]] = None):
        """
        Initialize notification system.
        
        Args:
            notification_configs: Optional notification configurations
        """
        self.notification_configs = notification_configs or []
        self.logger = logger
        
        logger.info(f"Initialized SecurityNotificationSystem with {len(self.notification_configs)} channels")
    
    def send_incident_notification(self, incident: SecurityIncident) -> Dict[str, Any]:
        """
        Send incident notification across configured channels.
        
        Args:
            incident: Security incident to notify about
            
        Returns:
            Dictionary with notification results
        """
        results = {}
        
        for config in self.notification_configs:
            if not config.enabled:
                continue
            
            # Check severity threshold
            if incident.severity.value < config.severity_threshold.value:
                continue
            
            try:
                if config.channel == NotificationChannel.EMAIL:
                    result = self._send_email_notification(incident, config)
                elif config.channel == NotificationChannel.WEBHOOK:
                    result = self._send_webhook_notification(incident, config)
                elif config.channel == NotificationChannel.DASHBOARD:
                    result = self._send_dashboard_notification(incident, config)
                else:
                    result = {"success": False, "error": "Unsupported channel"}
                
                results[config.channel.value] = result
                
            except Exception as e:
                self.logger.error(f"Notification failed for {config.channel.value}: {e}")
                results[config.channel.value] = {"success": False, "error": str(e)}
        
        return results
    
    def _send_email_notification(self, incident: SecurityIncident, 
                               config: NotificationConfig) -> Dict[str, Any]:
        """Send email notification."""
        email_config = config.config
        
        # Create email content
        subject = f"[SECURITY ALERT] {incident.title} - {incident.severity.value.upper()}"
        
        body = f"""
Security Incident Detected

Incident ID: {incident.incident_id}
Severity: {incident.severity.value.upper()}
Status: {incident.status.value}
Created: {datetime.fromtimestamp(incident.created_at)}

Description:
{incident.description}

Affected Systems: {', '.join(incident.affected_systems)}
Affected Users: {', '.join(incident.affected_users)}

Response Actions Recommended:
{chr(10).join(f"- {action}" for action in incident.response_actions)}

Please investigate immediately.
"""
        
        # In real implementation, would send actual email
        self.logger.info(f"EMAIL NOTIFICATION: {subject}")
        
        return {"success": True, "subject": subject}
    
    def _send_webhook_notification(self, incident: SecurityIncident,
                                 config: NotificationConfig) -> Dict[str, Any]:
        """Send webhook notification."""
        webhook_config = config.config
        
        payload = {
            "incident_id": incident.incident_id,
            "title": incident.title,
            "severity": incident.severity.value,
            "status": incident.status.value,
            "created_at": incident.created_at,
            "description": incident.description,
            "affected_systems": incident.affected_systems,
            "affected_users": incident.affected_users,
            "response_actions": [action.value for action in incident.response_actions]
        }
        
        # In real implementation, would send HTTP POST to webhook URL
        self.logger.info(f"WEBHOOK NOTIFICATION: {webhook_config.get('url', 'unknown')}")
        
        return {"success": True, "payload_size": len(json.dumps(payload))}
    
    def _send_dashboard_notification(self, incident: SecurityIncident,
                                   config: NotificationConfig) -> Dict[str, Any]:
        """Send dashboard notification."""
        # In real implementation, would update dashboard
        self.logger.info(f"DASHBOARD NOTIFICATION: {incident.title}")
        
        return {"success": True, "dashboard_updated": True}


class SecurityIncidentResponse:
    """
    Comprehensive security incident response system.
    
    Integrates incident classification, automated response, and notifications
    for complete security incident management.
    """
    
    def __init__(self, notification_configs: Optional[List[NotificationConfig]] = None):
        """
        Initialize security incident response system.
        
        Args:
            notification_configs: Optional notification configurations
        """
        self.incident_classifier = IncidentClassifier()
        self.response_manager = ResponseActionManager()
        self.notification_system = SecurityNotificationSystem(notification_configs)
        
        self.active_incidents: Dict[str, SecurityIncident] = {}
        self.incident_history: List[SecurityIncident] = []
        self.lock = threading.Lock()
        
        self.logger = logger
        logger.info("Initialized SecurityIncidentResponse")
    
    def handle_threat(self, threat: ThreatDetection) -> SecurityIncident:
        """
        Handle security threat with full incident response.
        
        Args:
            threat: Threat detection to handle
            
        Returns:
            SecurityIncident created and handled
        """
        # Classify threat into incident
        incident = self.incident_classifier.classify_threat(threat)
        
        with self.lock:
            self.active_incidents[incident.incident_id] = incident
        
        # Send notifications
        notification_results = self.notification_system.send_incident_notification(incident)
        incident.metadata["notification_results"] = notification_results
        
        # Execute automated response actions
        for action in incident.response_actions:
            response = self.response_manager.execute_response(incident, action)
            
            # Update incident with response results
            incident.metadata.setdefault("responses", []).append({
                "action": action.value,
                "success": response.success,
                "timestamp": response.timestamp,
                "details": response.details
            })
        
        # Update incident status
        incident.status = IncidentStatus.INVESTIGATING
        incident.updated_at = time.time()
        
        self.logger.info(f"Handled threat {threat.threat_id} as incident {incident.incident_id}")
        return incident
    
    def update_incident_status(self, incident_id: str, 
                             status: IncidentStatus) -> Optional[SecurityIncident]:
        """Update incident status."""
        with self.lock:
            if incident_id in self.active_incidents:
                incident = self.active_incidents[incident_id]
                incident.status = status
                incident.updated_at = time.time()
                
                # Move to history if closed
                if status == IncidentStatus.CLOSED:
                    self.incident_history.append(incident)
                    del self.active_incidents[incident_id]
                
                self.logger.info(f"Updated incident {incident_id} status to {status.value}")
                return incident
        
        return None
    
    def get_active_incidents(self) -> List[SecurityIncident]:
        """Get all active incidents."""
        with self.lock:
            return list(self.active_incidents.values())
    
    def get_incident_statistics(self) -> Dict[str, Any]:
        """Get incident response statistics."""
        with self.lock:
            active_incidents = list(self.active_incidents.values())
            history = list(self.incident_history)
        
        all_incidents = active_incidents + history
        
        if not all_incidents:
            return {"total_incidents": 0}
        
        # Calculate statistics
        severity_counts = {severity.value: 0 for severity in IncidentSeverity}
        status_counts = {status.value: 0 for status in IncidentStatus}
        
        for incident in all_incidents:
            severity_counts[incident.severity.value] += 1
            status_counts[incident.status.value] += 1
        
        return {
            "total_incidents": len(all_incidents),
            "active_incidents": len(active_incidents),
            "closed_incidents": len(history),
            "severity_distribution": severity_counts,
            "status_distribution": status_counts,
            "avg_response_time": sum(
                incident.updated_at - incident.created_at 
                for incident in all_incidents
            ) / len(all_incidents) if all_incidents else 0
        }
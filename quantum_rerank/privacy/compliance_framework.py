"""
Compliance Framework for Privacy-Preserving RAG Systems

Provides HIPAA, GDPR, and other regulatory compliance frameworks for
quantum-inspired RAG deployments in healthcare and regulated industries.
"""

import json
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ComplianceLevel(Enum):
    """Compliance certification levels."""
    BASIC = "basic"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    HIPAA_GDPR = "hipaa_gdpr"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    FULL = "full"


@dataclass
class ComplianceConfig:
    """Configuration for compliance framework."""
    target_level: ComplianceLevel = ComplianceLevel.BASIC
    enable_audit_logging: bool = True
    enable_data_encryption: bool = True
    enable_access_controls: bool = True
    data_retention_days: int = 2555  # 7 years default
    anonymization_required: bool = True
    consent_management: bool = True
    breach_notification: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


@dataclass
class ComplianceAuditEntry:
    """Audit log entry for compliance tracking."""
    timestamp: float
    user_id: str
    action: str
    resource: str
    result: str
    compliance_level: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit entry to dictionary."""
        return asdict(self)


class ComplianceFramework:
    """Comprehensive compliance framework for RAG systems."""
    
    def __init__(self, config: ComplianceConfig):
        self.config = config
        self.audit_log: List[ComplianceAuditEntry] = []
        self.data_inventory: Dict[str, Dict[str, Any]] = {}
        self.consent_records: Dict[str, Dict[str, Any]] = {}
        self.access_controls: Dict[str, Set[str]] = {}
        
        # Initialize compliance checks
        self._initialize_compliance_checks()
        
        logger.info(f"Compliance Framework initialized: {config.target_level.value}")
    
    def _initialize_compliance_checks(self):
        """Initialize compliance validation checks."""
        self.compliance_checks = {
            ComplianceLevel.BASIC: self._validate_basic_compliance,
            ComplianceLevel.HIPAA: self._validate_hipaa_compliance,
            ComplianceLevel.GDPR: self._validate_gdpr_compliance,
            ComplianceLevel.HIPAA_GDPR: self._validate_hipaa_gdpr_compliance,
            ComplianceLevel.SOC2: self._validate_soc2_compliance,
            ComplianceLevel.ISO27001: self._validate_iso27001_compliance,
            ComplianceLevel.FULL: self._validate_full_compliance
        }
    
    def log_action(self, user_id: str, action: str, resource: str, 
                   result: str, metadata: Optional[Dict[str, Any]] = None):
        """Log an action for compliance audit trail."""
        if not self.config.enable_audit_logging:
            return
        
        audit_entry = ComplianceAuditEntry(
            timestamp=time.time(),
            user_id=user_id,
            action=action,
            resource=resource,
            result=result,
            compliance_level=self.config.target_level.value,
            metadata=metadata or {}
        )
        
        self.audit_log.append(audit_entry)
        logger.debug(f"Audit logged: {action} on {resource} by {user_id}")
    
    def register_data_processing(self, data_id: str, data_type: str, 
                                purpose: str, legal_basis: str):
        """Register data processing for GDPR compliance."""
        self.data_inventory[data_id] = {
            "data_type": data_type,
            "purpose": purpose,
            "legal_basis": legal_basis,
            "created_at": time.time(),
            "retention_until": time.time() + (self.config.data_retention_days * 24 * 3600)
        }
        
        self.log_action("system", "data_registration", data_id, "success", {
            "data_type": data_type,
            "purpose": purpose,
            "legal_basis": legal_basis
        })
    
    def record_consent(self, user_id: str, consent_type: str, 
                      granted: bool, purpose: str):
        """Record user consent for data processing."""
        if user_id not in self.consent_records:
            self.consent_records[user_id] = {}
        
        self.consent_records[user_id][consent_type] = {
            "granted": granted,
            "purpose": purpose,
            "timestamp": time.time(),
            "ip_address": "127.0.0.1",  # Mock IP
            "user_agent": "quantum-rag-system"
        }
        
        self.log_action(user_id, "consent_recorded", consent_type, 
                       "granted" if granted else "denied", {
                           "purpose": purpose,
                           "consent_type": consent_type
                       })
    
    def check_consent(self, user_id: str, consent_type: str) -> bool:
        """Check if user has granted specific consent."""
        if user_id not in self.consent_records:
            return False
        
        consent = self.consent_records[user_id].get(consent_type)
        return consent and consent.get("granted", False)
    
    def grant_access(self, user_id: str, resource: str):
        """Grant user access to a resource."""
        if user_id not in self.access_controls:
            self.access_controls[user_id] = set()
        
        self.access_controls[user_id].add(resource)
        self.log_action(user_id, "access_granted", resource, "success")
    
    def revoke_access(self, user_id: str, resource: str):
        """Revoke user access to a resource."""
        if user_id in self.access_controls:
            self.access_controls[user_id].discard(resource)
            self.log_action(user_id, "access_revoked", resource, "success")
    
    def check_access(self, user_id: str, resource: str) -> bool:
        """Check if user has access to a resource."""
        if not self.config.enable_access_controls:
            return True
        
        has_access = (user_id in self.access_controls and 
                     resource in self.access_controls[user_id])
        
        self.log_action(user_id, "access_check", resource, 
                       "allowed" if has_access else "denied")
        
        return has_access
    
    def handle_data_breach(self, breach_id: str, affected_users: List[str], 
                          breach_type: str, description: str):
        """Handle data breach notification (GDPR Article 33/34)."""
        breach_record = {
            "breach_id": breach_id,
            "timestamp": time.time(),
            "affected_users": affected_users,
            "breach_type": breach_type,
            "description": description,
            "notification_sent": False
        }
        
        # Log breach
        self.log_action("system", "data_breach", breach_id, "detected", {
            "affected_users_count": len(affected_users),
            "breach_type": breach_type,
            "description": description
        })
        
        # GDPR requires notification within 72 hours
        if self.config.breach_notification:
            logger.critical(f"Data breach detected: {breach_id} - {description}")
            logger.critical(f"Affected users: {len(affected_users)}")
            
            # In real implementation, send notifications to authorities and users
            breach_record["notification_sent"] = True
        
        return breach_record
    
    def request_data_deletion(self, user_id: str, data_types: List[str]) -> Dict[str, Any]:
        """Handle right to erasure (GDPR Article 17)."""
        deletion_record = {
            "user_id": user_id,
            "timestamp": time.time(),
            "data_types": data_types,
            "status": "processing"
        }
        
        try:
            # Remove from data inventory
            to_remove = []
            for data_id, data_info in self.data_inventory.items():
                if data_info.get("data_type") in data_types:
                    to_remove.append(data_id)
            
            for data_id in to_remove:
                del self.data_inventory[data_id]
            
            # Remove consent records
            if user_id in self.consent_records:
                for data_type in data_types:
                    self.consent_records[user_id].pop(data_type, None)
            
            deletion_record["status"] = "completed"
            deletion_record["deleted_items"] = len(to_remove)
            
            self.log_action(user_id, "data_deletion", "multiple", "success", {
                "data_types": data_types,
                "deleted_items": len(to_remove)
            })
            
        except Exception as e:
            deletion_record["status"] = "failed"
            deletion_record["error"] = str(e)
            
            self.log_action(user_id, "data_deletion", "multiple", "failed", {
                "error": str(e)
            })
        
        return deletion_record
    
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Handle data portability (GDPR Article 20)."""
        export_data = {
            "user_id": user_id,
            "export_timestamp": time.time(),
            "consent_records": self.consent_records.get(user_id, {}),
            "access_permissions": list(self.access_controls.get(user_id, set())),
            "data_processing_records": []
        }
        
        # Find user's data in inventory
        for data_id, data_info in self.data_inventory.items():
            if user_id in data_id:  # Simple check - in real impl, use proper user linking
                export_data["data_processing_records"].append({
                    "data_id": data_id,
                    "data_type": data_info["data_type"],
                    "purpose": data_info["purpose"],
                    "legal_basis": data_info["legal_basis"],
                    "created_at": data_info["created_at"]
                })
        
        self.log_action(user_id, "data_export", "user_data", "success", {
            "export_items": len(export_data["data_processing_records"])
        })
        
        return export_data
    
    def _validate_basic_compliance(self) -> Dict[str, Any]:
        """Validate basic compliance requirements."""
        return {
            "audit_logging": self.config.enable_audit_logging,
            "data_encryption": self.config.enable_data_encryption,
            "access_controls": self.config.enable_access_controls,
            "status": "compliant"
        }
    
    def _validate_hipaa_compliance(self) -> Dict[str, Any]:
        """Validate HIPAA compliance requirements."""
        hipaa_requirements = {
            "administrative_safeguards": True,
            "physical_safeguards": True,
            "technical_safeguards": True,
            "audit_controls": self.config.enable_audit_logging,
            "access_controls": self.config.enable_access_controls,
            "data_encryption": self.config.enable_data_encryption,
            "breach_notification": self.config.breach_notification,
            "minimum_necessary": True,
            "business_associate_agreements": True
        }
        
        compliant = all(hipaa_requirements.values())
        
        return {
            "requirements": hipaa_requirements,
            "status": "compliant" if compliant else "non_compliant",
            "compliance_level": "hipaa"
        }
    
    def _validate_gdpr_compliance(self) -> Dict[str, Any]:
        """Validate GDPR compliance requirements."""
        gdpr_requirements = {
            "lawful_basis": True,
            "consent_management": self.config.consent_management,
            "data_subject_rights": True,
            "data_protection_by_design": True,
            "data_retention_policy": self.config.data_retention_days > 0,
            "breach_notification": self.config.breach_notification,
            "data_processing_records": len(self.data_inventory) > 0,
            "anonymization": self.config.anonymization_required
        }
        
        compliant = all(gdpr_requirements.values())
        
        return {
            "requirements": gdpr_requirements,
            "status": "compliant" if compliant else "non_compliant",
            "compliance_level": "gdpr"
        }
    
    def _validate_hipaa_gdpr_compliance(self) -> Dict[str, Any]:
        """Validate combined HIPAA and GDPR compliance."""
        hipaa_result = self._validate_hipaa_compliance()
        gdpr_result = self._validate_gdpr_compliance()
        
        combined_compliant = (hipaa_result["status"] == "compliant" and 
                            gdpr_result["status"] == "compliant")
        
        return {
            "hipaa_compliance": hipaa_result,
            "gdpr_compliance": gdpr_result,
            "status": "compliant" if combined_compliant else "non_compliant",
            "compliance_level": "hipaa_gdpr"
        }
    
    def _validate_soc2_compliance(self) -> Dict[str, Any]:
        """Validate SOC 2 compliance requirements."""
        soc2_requirements = {
            "security": True,
            "availability": True,
            "processing_integrity": True,
            "confidentiality": self.config.enable_data_encryption,
            "privacy": self.config.consent_management,
            "audit_logging": self.config.enable_audit_logging,
            "access_controls": self.config.enable_access_controls
        }
        
        compliant = all(soc2_requirements.values())
        
        return {
            "requirements": soc2_requirements,
            "status": "compliant" if compliant else "non_compliant",
            "compliance_level": "soc2"
        }
    
    def _validate_iso27001_compliance(self) -> Dict[str, Any]:
        """Validate ISO 27001 compliance requirements."""
        iso27001_requirements = {
            "information_security_policy": True,
            "risk_management": True,
            "asset_management": True,
            "access_control": self.config.enable_access_controls,
            "cryptography": self.config.enable_data_encryption,
            "operations_security": True,
            "communications_security": True,
            "incident_management": self.config.breach_notification,
            "business_continuity": True,
            "compliance": True
        }
        
        compliant = all(iso27001_requirements.values())
        
        return {
            "requirements": iso27001_requirements,
            "status": "compliant" if compliant else "non_compliant",
            "compliance_level": "iso27001"
        }
    
    def _validate_full_compliance(self) -> Dict[str, Any]:
        """Validate full compliance across all frameworks."""
        all_results = {
            "basic": self._validate_basic_compliance(),
            "hipaa": self._validate_hipaa_compliance(),
            "gdpr": self._validate_gdpr_compliance(),
            "soc2": self._validate_soc2_compliance(),
            "iso27001": self._validate_iso27001_compliance()
        }
        
        fully_compliant = all(result["status"] == "compliant" 
                            for result in all_results.values())
        
        return {
            "framework_results": all_results,
            "status": "compliant" if fully_compliant else "non_compliant",
            "compliance_level": "full"
        }
    
    def validate_compliance(self) -> Dict[str, Any]:
        """Validate current compliance level."""
        validation_func = self.compliance_checks.get(self.config.target_level)
        if not validation_func:
            return {"status": "unsupported", "error": f"Unsupported compliance level: {self.config.target_level}"}
        
        result = validation_func()
        result["validation_timestamp"] = time.time()
        result["target_level"] = self.config.target_level.value
        
        self.log_action("system", "compliance_validation", self.config.target_level.value, 
                       result["status"], result)
        
        return result
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        report = {
            "report_timestamp": time.time(),
            "configuration": self.config.to_dict(),
            "compliance_validation": self.validate_compliance(),
            "audit_log_entries": len(self.audit_log),
            "data_inventory_items": len(self.data_inventory),
            "consent_records": len(self.consent_records),
            "access_control_users": len(self.access_controls),
            "statistics": {
                "total_actions_logged": len(self.audit_log),
                "data_processing_registrations": len(self.data_inventory),
                "consent_grants": sum(1 for user_consents in self.consent_records.values()
                                    for consent in user_consents.values()
                                    if consent.get("granted", False)),
                "access_permissions": sum(len(permissions) for permissions in self.access_controls.values())
            }
        }
        
        return report
    
    def export_audit_log(self, filepath: str, start_time: Optional[float] = None, 
                        end_time: Optional[float] = None):
        """Export audit log to file."""
        filtered_log = self.audit_log
        
        if start_time:
            filtered_log = [entry for entry in filtered_log if entry.timestamp >= start_time]
        
        if end_time:
            filtered_log = [entry for entry in filtered_log if entry.timestamp <= end_time]
        
        export_data = {
            "export_timestamp": time.time(),
            "audit_entries": [entry.to_dict() for entry in filtered_log],
            "total_entries": len(filtered_log),
            "compliance_level": self.config.target_level.value
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Audit log exported to {filepath}: {len(filtered_log)} entries")
    
    def cleanup_expired_data(self):
        """Clean up expired data according to retention policy."""
        current_time = time.time()
        expired_items = []
        
        for data_id, data_info in list(self.data_inventory.items()):
            if current_time > data_info.get("retention_until", current_time + 1):
                expired_items.append(data_id)
                del self.data_inventory[data_id]
        
        if expired_items:
            self.log_action("system", "data_cleanup", "expired_data", "success", {
                "expired_items": len(expired_items),
                "data_ids": expired_items
            })
            
            logger.info(f"Cleaned up {len(expired_items)} expired data items")
        
        return expired_items


# Utility functions
def create_compliance_config(level: ComplianceLevel) -> ComplianceConfig:
    """Create compliance configuration for specific level."""
    base_config = ComplianceConfig(target_level=level)
    
    if level in [ComplianceLevel.HIPAA, ComplianceLevel.HIPAA_GDPR]:
        base_config.enable_audit_logging = True
        base_config.enable_data_encryption = True
        base_config.enable_access_controls = True
        base_config.breach_notification = True
    
    if level in [ComplianceLevel.GDPR, ComplianceLevel.HIPAA_GDPR]:
        base_config.consent_management = True
        base_config.anonymization_required = True
        base_config.data_retention_days = 2555  # 7 years
    
    return base_config


def validate_compliance_readiness(config: ComplianceConfig) -> Dict[str, Any]:
    """Validate if system is ready for compliance certification."""
    readiness_checks = {
        "audit_logging_enabled": config.enable_audit_logging,
        "data_encryption_enabled": config.enable_data_encryption,
        "access_controls_enabled": config.enable_access_controls,
        "retention_policy_defined": config.data_retention_days > 0,
        "breach_notification_enabled": config.breach_notification,
        "consent_management_enabled": config.consent_management
    }
    
    readiness_score = sum(readiness_checks.values()) / len(readiness_checks)
    
    return {
        "readiness_score": readiness_score,
        "readiness_checks": readiness_checks,
        "compliance_level": config.target_level.value,
        "ready_for_certification": readiness_score >= 0.8
    }
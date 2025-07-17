"""
Privacy-Preserving Module for Quantum-Inspired RAG.

This module provides comprehensive privacy protection for edge-deployed RAG systems,
including homomorphic encryption, differential privacy, and HIPAA/GDPR compliance.

Components:
- Homomorphic encryption for embeddings
- Differential privacy mechanisms
- Secure multi-party computation
- HIPAA/GDPR compliance frameworks
"""

__version__ = "1.0.0"

from .homomorphic_encryption import (
    HomomorphicEncryption,
    EncryptionConfig,
    EncryptedEmbeddings
)

from .differential_privacy import (
    DifferentialPrivacy,
    PrivacyConfig,
    PrivacyMechanism
)

from .compliance_framework import (
    ComplianceFramework,
    ComplianceConfig,
    ComplianceLevel
)

__all__ = [
    "HomomorphicEncryption",
    "EncryptionConfig",
    "EncryptedEmbeddings",
    "DifferentialPrivacy", 
    "PrivacyConfig",
    "PrivacyMechanism",
    "ComplianceFramework",
    "ComplianceConfig",
    "ComplianceLevel"
]
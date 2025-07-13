"""
Comprehensive input validation and sanitization for QuantumRerank security.

This module provides robust input validation for all system inputs including
embeddings, text, quantum parameters, and API requests with security-focused
validation rules and sanitization.
"""

import re
import html
import numpy as np
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Pattern
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..utils.logging_config import get_logger
from pydantic import ValidationError
from ..utils.exceptions import (
    ConfigurationError
)

logger = get_logger(__name__)


class ValidationLevel(Enum):
    """Security validation levels."""
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"


@dataclass
class ValidationResult:
    """Result of security validation."""
    valid: bool
    sanitized_input: Optional[Any] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    security_score: float = 1.0
    validation_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationRule:
    """Security validation rule configuration."""
    name: str
    validator_function: callable
    error_message: str
    security_level: ValidationLevel = ValidationLevel.BASIC
    enabled: bool = True


class SecurityInputValidator:
    """
    Comprehensive input validation for security.
    
    Validates all inputs with security-focused rules, sanitization,
    and threat detection capabilities.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        """
        Initialize security input validator.
        
        Args:
            validation_level: Security validation strictness level
        """
        self.validation_level = validation_level
        self.logger = logger
        
        # Security validation rules configuration
        self.validation_rules = self._initialize_validation_rules()
        
        # Forbidden patterns for security
        self.forbidden_patterns = [
            # XSS prevention
            re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'data:text/html', re.IGNORECASE),
            re.compile(r'vbscript:', re.IGNORECASE),
            
            # SQL injection prevention  
            re.compile(r"(union|select|insert|update|delete|drop|create|alter)\s", re.IGNORECASE),
            re.compile(r"'.*'.*or.*'.*'", re.IGNORECASE),
            re.compile(r"--|#|/\*|\*/", re.IGNORECASE),
            
            # Command injection prevention
            re.compile(r"[;&|`$]", re.IGNORECASE),
            re.compile(r"(rm|cat|ls|ps|kill|wget|curl)\s", re.IGNORECASE),
            
            # Path traversal prevention
            re.compile(r"\.\./", re.IGNORECASE),
            re.compile(r"\.\.\\", re.IGNORECASE),
        ]
        
        # Security limits
        self.security_limits = {
            "max_text_length": 50000,
            "max_embedding_dimension": 4096,
            "max_batch_size": 1000,
            "max_parameter_count": 200,
            "max_circuit_depth": 100,
            "max_circuit_gates": 500
        }
        
        logger.info(f"Initialized SecurityInputValidator with {validation_level.value} validation level")
    
    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize security validation rules for different input types."""
        return {
            "embedding_vector": {
                "max_dimensions": 4096,
                "allowed_dtypes": [np.float32, np.float64, np.complex64, np.complex128],
                "value_range": (-1000.0, 1000.0),
                "required_finite": True,
                "normalization_tolerance": 0.1,
                "security_checks": True
            },
            "text_input": {
                "max_length": 50000,
                "encoding": "utf-8",
                "forbidden_patterns": True,
                "sanitization": True,
                "content_filtering": True
            },
            "api_parameters": {
                "max_batch_size": 1000,
                "allowed_methods": ["classical", "quantum", "hybrid"],
                "timeout_limits": {"min": 1, "max": 3600},
                "rate_limiting": True
            },
            "quantum_parameters": {
                "max_count": 200,
                "value_range": (-np.pi, np.pi),
                "required_finite": True,
                "gradient_safety": True
            },
            "file_input": {
                "max_size_mb": 100,
                "allowed_extensions": [".txt", ".json", ".csv"],
                "content_scanning": True,
                "virus_scanning": False  # Would require external integration
            }
        }
    
    def validate_embedding_input(self, embedding: np.ndarray, 
                                context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate embedding vector for security and correctness.
        
        Args:
            embedding: Embedding vector to validate
            context: Optional validation context
            
        Returns:
            ValidationResult with security validation results
        """
        start_time = time.time()
        validation_errors = []
        warnings = []
        metadata = {}
        
        try:
            rules = self.validation_rules["embedding_vector"]
            
            # Basic shape validation
            if not isinstance(embedding, np.ndarray):
                validation_errors.append("Input must be a numpy array")
                return self._create_validation_result(False, None, validation_errors, warnings, 0.0, start_time, metadata)
            
            if embedding.ndim == 0:
                validation_errors.append("Embedding must have at least 1 dimension")
            elif embedding.ndim > 2:
                validation_errors.append(f"Embedding has too many dimensions: {embedding.ndim}")
            
            # Dimension limits for security
            if embedding.size > rules["max_dimensions"]:
                validation_errors.append(f"Embedding dimension {embedding.size} exceeds security limit {rules['max_dimensions']}")
            
            # Data type validation
            if embedding.dtype not in rules["allowed_dtypes"]:
                validation_errors.append(f"Invalid embedding data type: {embedding.dtype}")
            
            # Value range validation for security
            if rules["security_checks"]:
                min_val, max_val = rules["value_range"]
                if np.any(embedding < min_val) or np.any(embedding > max_val):
                    validation_errors.append(f"Embedding values outside secure range [{min_val}, {max_val}]")
            
            # Check for malicious patterns in data
            if rules["required_finite"]:
                if np.any(~np.isfinite(embedding)):
                    validation_errors.append("Embedding contains non-finite values (NaN/Inf)")
            
            # Advanced security checks
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                # Check for potential data exfiltration patterns
                entropy = self._calculate_entropy(embedding.flatten())
                metadata["entropy"] = entropy
                
                if entropy < 0.1:  # Very low entropy might indicate crafted data
                    warnings.append("Suspiciously low entropy in embedding data")
                elif entropy > 8.0:  # Very high entropy might indicate encrypted data
                    warnings.append("Suspiciously high entropy in embedding data")
                
                # Check for unusual statistical properties
                if embedding.size > 10:
                    std_dev = np.std(embedding)
                    mean_val = np.mean(embedding)
                    metadata["statistics"] = {"std": std_dev, "mean": mean_val}
                    
                    if std_dev == 0:
                        warnings.append("Zero standard deviation - potential constant injection")
                    elif abs(mean_val) > 100:
                        warnings.append("Extreme mean value detected")
            
            # Paranoid security checks
            if self.validation_level == ValidationLevel.PARANOID:
                # Check for potential side-channel attack vectors
                if self._detect_timing_attack_patterns(embedding):
                    warnings.append("Potential timing attack patterns detected")
                
                # Check for adversarial perturbations
                if self._detect_adversarial_patterns(embedding):
                    warnings.append("Potential adversarial perturbation patterns detected")
            
            # Sanitize embedding if validation passes
            sanitized_embedding = None
            if not validation_errors:
                sanitized_embedding = self._sanitize_embedding(embedding)
            
            # Calculate security score
            security_score = self._calculate_security_score(validation_errors, warnings)
            
            return self._create_validation_result(
                len(validation_errors) == 0, 
                sanitized_embedding, 
                validation_errors, 
                warnings, 
                security_score, 
                start_time, 
                metadata
            )
            
        except Exception as e:
            self.logger.error(f"Embedding validation failed: {e}")
            return self._create_validation_result(
                False, None, [f"Validation exception: {str(e)}"], [], 0.0, start_time, {}
            )
    
    def validate_text_input(self, text: str, 
                           context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate text input for security threats.
        
        Args:
            text: Text input to validate
            context: Optional validation context
            
        Returns:
            ValidationResult with text security validation
        """
        start_time = time.time()
        validation_errors = []
        warnings = []
        metadata = {}
        
        try:
            rules = self.validation_rules["text_input"]
            
            # Basic type validation
            if not isinstance(text, str):
                validation_errors.append("Input must be a string")
                return self._create_validation_result(False, None, validation_errors, warnings, 0.0, start_time, metadata)
            
            # Length validation
            if len(text) > rules["max_length"]:
                validation_errors.append(f"Text length {len(text)} exceeds limit {rules['max_length']}")
            
            # Encoding validation
            try:
                text.encode(rules["encoding"])
            except UnicodeEncodeError:
                validation_errors.append(f"Text not valid {rules['encoding']} encoding")
            
            # Security pattern detection
            if rules["forbidden_patterns"]:
                for pattern in self.forbidden_patterns:
                    if pattern.search(text):
                        validation_errors.append(f"Forbidden security pattern detected: {pattern.pattern[:50]}...")
            
            # Content filtering for suspicious patterns
            if rules["content_filtering"]:
                # Check for potential data exfiltration
                if self._detect_data_exfiltration_patterns(text):
                    warnings.append("Potential data exfiltration patterns detected")
                
                # Check for potential injection attempts
                if self._detect_injection_patterns(text):
                    validation_errors.append("Potential injection attack patterns detected")
            
            # Advanced security checks for strict/paranoid levels
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                # Check character distribution for anomalies
                char_entropy = self._calculate_text_entropy(text)
                metadata["text_entropy"] = char_entropy
                
                if char_entropy > 7.5:  # Very high entropy might indicate encoded data
                    warnings.append("Suspiciously high character entropy")
                
                # Check for unusual character patterns
                unusual_chars = sum(1 for c in text if ord(c) > 127)
                if unusual_chars > len(text) * 0.1:  # >10% non-ASCII
                    warnings.append("High percentage of non-ASCII characters")
                
                metadata["non_ascii_ratio"] = unusual_chars / len(text) if text else 0
            
            # Sanitize text if validation passes
            sanitized_text = None
            if not validation_errors:
                sanitized_text = self._sanitize_text(text)
            
            # Calculate security score
            security_score = self._calculate_security_score(validation_errors, warnings)
            
            return self._create_validation_result(
                len(validation_errors) == 0,
                sanitized_text,
                validation_errors,
                warnings,
                security_score,
                start_time,
                metadata
            )
            
        except Exception as e:
            self.logger.error(f"Text validation failed: {e}")
            return self._create_validation_result(
                False, None, [f"Validation exception: {str(e)}"], [], 0.0, start_time, {}
            )
    
    def validate_quantum_parameters(self, parameters: np.ndarray,
                                  context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate quantum parameters for security and correctness.
        
        Args:
            parameters: Quantum parameters to validate
            context: Optional validation context
            
        Returns:
            ValidationResult with quantum parameter validation
        """
        start_time = time.time()
        validation_errors = []
        warnings = []
        metadata = {}
        
        try:
            rules = self.validation_rules["quantum_parameters"]
            
            # Basic validation
            if not isinstance(parameters, (np.ndarray, list, tuple)):
                validation_errors.append("Parameters must be array-like")
                return self._create_validation_result(False, None, validation_errors, warnings, 0.0, start_time, metadata)
            
            if isinstance(parameters, (list, tuple)):
                parameters = np.array(parameters)
            
            # Size limits for security
            if parameters.size > rules["max_count"]:
                validation_errors.append(f"Parameter count {parameters.size} exceeds security limit {rules['max_count']}")
            
            # Value range validation
            min_val, max_val = rules["value_range"]
            if np.any(parameters < min_val) or np.any(parameters > max_val):
                validation_errors.append(f"Parameters outside secure range [{min_val:.2f}, {max_val:.2f}]")
            
            # Finite value check
            if rules["required_finite"]:
                if np.any(~np.isfinite(parameters)):
                    validation_errors.append("Parameters contain non-finite values")
            
            # Gradient safety checks
            if rules["gradient_safety"]:
                # Check for potential gradient explosion/vanishing
                param_variance = np.var(parameters)
                metadata["parameter_variance"] = param_variance
                
                if param_variance > 10:  # High variance might indicate instability
                    warnings.append("High parameter variance detected")
                elif param_variance < 1e-6:  # Very low variance might indicate stagnation
                    warnings.append("Very low parameter variance detected")
            
            # Security-specific checks
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                # Check for potential adversarial parameter manipulation
                if self._detect_adversarial_parameters(parameters):
                    warnings.append("Potential adversarial parameter manipulation detected")
                
                # Check parameter distribution
                param_entropy = self._calculate_entropy(parameters)
                metadata["parameter_entropy"] = param_entropy
                
                if param_entropy < 1.0:
                    warnings.append("Low parameter entropy - potential security risk")
            
            # Sanitize parameters
            sanitized_parameters = None
            if not validation_errors:
                sanitized_parameters = self._sanitize_parameters(parameters)
            
            # Calculate security score
            security_score = self._calculate_security_score(validation_errors, warnings)
            
            return self._create_validation_result(
                len(validation_errors) == 0,
                sanitized_parameters,
                validation_errors,
                warnings,
                security_score,
                start_time,
                metadata
            )
            
        except Exception as e:
            self.logger.error(f"Parameter validation failed: {e}")
            return self._create_validation_result(
                False, None, [f"Validation exception: {str(e)}"], [], 0.0, start_time, {}
            )
    
    def _sanitize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Sanitize embedding for security."""
        # Clip to safe ranges
        sanitized = np.clip(embedding, -1000, 1000)
        
        # Replace non-finite values
        sanitized = np.where(np.isfinite(sanitized), sanitized, 0)
        
        # Normalize if required
        if sanitized.ndim == 1 and np.linalg.norm(sanitized) > 0:
            sanitized = sanitized / np.linalg.norm(sanitized)
        elif sanitized.ndim == 2:
            norms = np.linalg.norm(sanitized, axis=1, keepdims=True)
            sanitized = np.where(norms > 0, sanitized / norms, sanitized)
        
        return sanitized
    
    def _sanitize_text(self, text: str) -> str:
        """Sanitize text input for security."""
        # HTML escape
        sanitized = html.escape(text)
        
        # Remove or replace dangerous patterns
        for pattern in self.forbidden_patterns:
            sanitized = pattern.sub('', sanitized)
        
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        # Limit length
        max_length = self.validation_rules["text_input"]["max_length"]
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized
    
    def _sanitize_parameters(self, parameters: np.ndarray) -> np.ndarray:
        """Sanitize quantum parameters for security."""
        # Clip to valid quantum parameter range
        sanitized = np.clip(parameters, -np.pi, np.pi)
        
        # Replace non-finite values
        sanitized = np.where(np.isfinite(sanitized), sanitized, 0)
        
        return sanitized
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy of data."""
        try:
            # Discretize data for entropy calculation
            hist, _ = np.histogram(data, bins=50, density=True)
            hist = hist[hist > 0]  # Remove zero entries
            
            if len(hist) == 0:
                return 0.0
            
            # Calculate entropy
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            return float(entropy)
        except:
            return 0.0
    
    def _calculate_text_entropy(self, text: str) -> float:
        """Calculate character entropy of text."""
        if not text:
            return 0.0
        
        try:
            # Count character frequencies
            char_counts = {}
            for char in text:
                char_counts[char] = char_counts.get(char, 0) + 1
            
            # Calculate probabilities
            total_chars = len(text)
            probabilities = [count / total_chars for count in char_counts.values()]
            
            # Calculate entropy
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            return float(entropy)
        except:
            return 0.0
    
    def _detect_timing_attack_patterns(self, data: np.ndarray) -> bool:
        """Detect potential timing attack patterns in data."""
        # This is a simplified check - in practice would be more sophisticated
        try:
            # Check for repetitive patterns that might indicate timing exploitation
            if data.size > 10:
                autocorr = np.correlate(data.flatten(), data.flatten(), mode='full')
                max_autocorr = np.max(autocorr[len(autocorr)//2 + 1:])
                
                # High autocorrelation might indicate artificial patterns
                return max_autocorr > np.var(data) * data.size * 0.8
        except:
            pass
        return False
    
    def _detect_adversarial_patterns(self, data: np.ndarray) -> bool:
        """Detect potential adversarial perturbation patterns."""
        try:
            if data.size > 10:
                # Check for high-frequency noise patterns common in adversarial examples
                if data.ndim == 1:
                    diff = np.diff(data)
                    high_freq_energy = np.sum(diff**2)
                    total_energy = np.sum(data**2)
                    
                    # High ratio might indicate adversarial perturbations
                    return high_freq_energy / (total_energy + 1e-10) > 0.5
        except:
            pass
        return False
    
    def _detect_adversarial_parameters(self, parameters: np.ndarray) -> bool:
        """Detect potential adversarial parameter manipulation."""
        try:
            # Check for unusual parameter distributions
            if parameters.size > 5:
                # Check for parameters clustered at boundaries (common in attacks)
                boundary_threshold = 0.9 * np.pi
                near_boundaries = np.sum(np.abs(parameters) > boundary_threshold)
                
                return near_boundaries > parameters.size * 0.7  # >70% near boundaries
        except:
            pass
        return False
    
    def _detect_data_exfiltration_patterns(self, text: str) -> bool:
        """Detect potential data exfiltration patterns in text."""
        # Check for base64-like patterns
        base64_pattern = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')
        if base64_pattern.search(text):
            return True
        
        # Check for hex patterns
        hex_pattern = re.compile(r'[0-9a-fA-F]{32,}')
        if hex_pattern.search(text):
            return True
        
        return False
    
    def _detect_injection_patterns(self, text: str) -> bool:
        """Detect potential injection attack patterns."""
        # Check for common injection patterns beyond the basic forbidden patterns
        injection_indicators = [
            r'(eval|exec|system|shell_exec)\s*\(',
            r'(file_get_contents|fread|readfile)\s*\(',
            r'(include|require|include_once|require_once)\s*\(',
            r'<%.*%>',  # Server-side includes
            r'\{\{.*\}\}',  # Template injection
        ]
        
        for pattern_str in injection_indicators:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            if pattern.search(text):
                return True
        
        return False
    
    def _calculate_security_score(self, errors: List[str], warnings: List[str]) -> float:
        """Calculate security score based on validation results."""
        if errors:
            return 0.0
        
        # Start with perfect score
        score = 1.0
        
        # Deduct for warnings
        score -= len(warnings) * 0.1
        
        # Ensure score is in valid range
        return max(0.0, min(1.0, score))
    
    def _create_validation_result(self, valid: bool, sanitized_input: Any,
                                errors: List[str], warnings: List[str],
                                security_score: float, start_time: float,
                                metadata: Dict[str, Any]) -> ValidationResult:
        """Create validation result with timing information."""
        validation_time_ms = (time.time() - start_time) * 1000
        
        return ValidationResult(
            valid=valid,
            sanitized_input=sanitized_input,
            errors=errors,
            warnings=warnings,
            security_score=security_score,
            validation_time_ms=validation_time_ms,
            metadata=metadata
        )


# Specialized validators for different input types

class EmbeddingValidator(SecurityInputValidator):
    """Specialized validator for embedding inputs."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        super().__init__(validation_level)
        # Override with embedding-specific rules
        self.validation_rules["embedding_vector"].update({
            "max_dimensions": 2048,  # More restrictive for embeddings
            "normalization_required": True,
            "similarity_bounds_check": True
        })


class TextValidator(SecurityInputValidator):
    """Specialized validator for text inputs."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        super().__init__(validation_level)
        # Override with text-specific rules
        self.validation_rules["text_input"].update({
            "max_length": 10000,  # More restrictive for regular text
            "language_detection": True,
            "sentiment_filtering": False  # Can be enabled if needed
        })


class ParameterValidator(SecurityInputValidator):
    """Specialized validator for quantum parameters."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.PARANOID):
        super().__init__(validation_level)
        # Override with parameter-specific rules
        self.validation_rules["quantum_parameters"].update({
            "max_count": 50,  # More restrictive for parameters
            "gradient_explosion_check": True,
            "optimization_safety": True
        })
"""
Advanced error classification system for comprehensive error analysis.

This module provides multi-dimensional error classification with severity assessment,
pattern recognition, and proactive error detection capabilities.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import numpy as np
import re

from ..utils.exceptions import QuantumRerankException, ErrorContext
from ..utils.logging_config import get_logger


class ErrorSeverity(Enum):
    """Error severity levels with operational impact."""
    CRITICAL = "critical"     # System failure, immediate action required
    HIGH = "high"            # Significant impact, urgent attention needed  
    MEDIUM = "medium"        # Moderate impact, attention required
    LOW = "low"             # Minor impact, monitoring sufficient
    INFO = "info"           # Informational, no action required


class ErrorCategory(Enum):
    """Error categories for classification."""
    QUANTUM_COMPUTATION = "quantum_computation"
    CLASSICAL_COMPUTATION = "classical_computation"
    SYSTEM_RESOURCE = "system_resource"
    DATA_VALIDATION = "data_validation"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    NETWORK_CONNECTIVITY = "network_connectivity"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"


class ErrorPatternType(Enum):
    """Types of error patterns for recognition."""
    FREQUENCY_SPIKE = "frequency_spike"
    CASCADING_FAILURE = "cascading_failure"
    PERIODIC_FAILURE = "periodic_failure"
    GRADUAL_DEGRADATION = "gradual_degradation"
    THRESHOLD_BREACH = "threshold_breach"


@dataclass
class ErrorClassification:
    """Comprehensive error classification result."""
    error_type: str
    category: ErrorCategory
    severity: ErrorSeverity
    confidence: float  # Confidence in classification (0-1)
    recoverable: bool
    fallback_available: bool
    estimated_impact: Dict[str, float]  # Impact on different metrics
    recovery_time_estimate_s: float
    suggested_actions: List[str]
    related_components: List[str]
    pattern_indicators: List[str] = field(default_factory=list)


@dataclass
class ErrorPattern:
    """Detected error pattern information."""
    pattern_type: ErrorPatternType
    affected_components: List[str]
    frequency: float  # Errors per unit time
    severity_trend: str  # "increasing", "decreasing", "stable"
    first_occurrence: float
    last_occurrence: float
    pattern_confidence: float
    prediction: Optional[Dict[str, Any]] = None


@dataclass
class ErrorMetrics:
    """Error metrics for pattern analysis."""
    error_type: str
    timestamp: float
    component: str
    severity: ErrorSeverity
    recovery_time_s: Optional[float] = None
    context: Optional[Dict[str, Any]] = None


class ErrorClassifier:
    """
    Advanced error classification system with pattern recognition.
    
    Provides multi-dimensional error classification, severity assessment,
    and proactive pattern detection for comprehensive error management.
    """
    
    def __init__(self, pattern_detection_window_s: int = 3600):
        self.logger = get_logger(__name__)
        self.pattern_detection_window_s = pattern_detection_window_s
        
        # Error classification rules
        self.classification_rules = self._initialize_classification_rules()
        self.severity_rules = self._initialize_severity_rules()
        
        # Pattern detection
        self.error_history: deque = deque(maxlen=10000)
        self.detected_patterns: Dict[str, ErrorPattern] = {}
        self.pattern_detectors = self._initialize_pattern_detectors()
        
        # Component relationships for cascading analysis
        self.component_dependencies = self._initialize_component_dependencies()
        
        # Performance metrics integration
        self.performance_thresholds = self._initialize_performance_thresholds()
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info("Initialized ErrorClassifier")
    
    def classify_error(self, error: Exception, context: Dict[str, Any]) -> ErrorClassification:
        """
        Classify an error with comprehensive analysis.
        
        Args:
            error: Exception to classify
            context: Additional context information
            
        Returns:
            Comprehensive error classification
        """
        start_time = time.time()
        
        # Extract error information
        error_type = type(error).__name__
        error_message = str(error)
        component = context.get("component", "unknown")
        operation = context.get("operation", "unknown")
        
        # Basic classification
        category = self._classify_error_category(error, error_message, context)
        base_severity = self._assess_base_severity(error, category, context)
        
        # Context-aware severity adjustment
        adjusted_severity = self._adjust_severity_by_context(base_severity, context)
        
        # Recoverability assessment
        recoverable = self._assess_recoverability(error, category, context)
        fallback_available = self._check_fallback_availability(category, context)
        
        # Impact estimation
        estimated_impact = self._estimate_error_impact(error, category, context)
        
        # Recovery time estimation
        recovery_time_estimate = self._estimate_recovery_time(category, adjusted_severity, context)
        
        # Generate suggested actions
        suggested_actions = self._generate_suggested_actions(error, category, adjusted_severity, context)
        
        # Identify related components
        related_components = self._identify_related_components(component, category)
        
        # Pattern analysis
        pattern_indicators = self._analyze_error_patterns(error_type, component, context)
        
        # Calculate classification confidence
        confidence = self._calculate_classification_confidence(
            error, category, adjusted_severity, context
        )
        
        classification = ErrorClassification(
            error_type=error_type,
            category=category,
            severity=adjusted_severity,
            confidence=confidence,
            recoverable=recoverable,
            fallback_available=fallback_available,
            estimated_impact=estimated_impact,
            recovery_time_estimate_s=recovery_time_estimate,
            suggested_actions=suggested_actions,
            related_components=related_components,
            pattern_indicators=pattern_indicators
        )
        
        # Record for pattern detection
        self._record_error_for_pattern_analysis(error, classification, context)
        
        classification_time = (time.time() - start_time) * 1000
        self.logger.debug(f"Error classification completed in {classification_time:.1f}ms")
        
        return classification
    
    def detect_error_patterns(self, analysis_window_s: Optional[int] = None) -> List[ErrorPattern]:
        """
        Detect error patterns in recent history.
        
        Args:
            analysis_window_s: Time window for pattern analysis
            
        Returns:
            List of detected error patterns
        """
        window_s = analysis_window_s or self.pattern_detection_window_s
        cutoff_time = time.time() - window_s
        
        with self._lock:
            # Filter recent errors
            recent_errors = [
                error for error in self.error_history
                if error.timestamp >= cutoff_time
            ]
            
            patterns = []
            
            # Run pattern detectors
            for pattern_type, detector in self.pattern_detectors.items():
                detected_pattern = detector(recent_errors, window_s)
                if detected_pattern:
                    patterns.append(detected_pattern)
                    
                    # Store pattern for future reference
                    pattern_key = f"{pattern_type}_{detected_pattern.affected_components[0] if detected_pattern.affected_components else 'global'}"
                    self.detected_patterns[pattern_key] = detected_pattern
            
            return patterns
    
    def predict_error_likelihood(self, component: str, operation: str,
                               context: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict likelihood of errors based on patterns and context.
        
        Args:
            component: Component being operated on
            operation: Operation being performed
            context: Current context
            
        Returns:
            Dictionary of error types and their likelihood (0-1)
        """
        predictions = {}
        
        # Analyze historical patterns for this component/operation
        component_patterns = [
            pattern for pattern in self.detected_patterns.values()
            if component in pattern.affected_components
        ]
        
        for pattern in component_patterns:
            if pattern.prediction:
                error_type = pattern.prediction.get("error_type")
                likelihood = pattern.prediction.get("likelihood", 0.0)
                
                # Adjust based on current context
                adjusted_likelihood = self._adjust_likelihood_by_context(
                    likelihood, context, pattern
                )
                
                predictions[error_type] = adjusted_likelihood
        
        # Add baseline predictions for common error types
        baseline_predictions = self._get_baseline_error_predictions(component, operation, context)
        for error_type, likelihood in baseline_predictions.items():
            if error_type not in predictions:
                predictions[error_type] = likelihood
        
        return predictions
    
    def get_error_statistics(self, time_window_s: int = 3600) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        cutoff_time = time.time() - time_window_s
        
        with self._lock:
            recent_errors = [
                error for error in self.error_history
                if error.timestamp >= cutoff_time
            ]
            
            if not recent_errors:
                return {"error_count": 0, "time_window_s": time_window_s}
            
            # Basic statistics
            stats = {
                "time_window_s": time_window_s,
                "total_errors": len(recent_errors),
                "error_rate_per_hour": len(recent_errors) / (time_window_s / 3600),
                "severity_distribution": self._calculate_severity_distribution(recent_errors),
                "category_distribution": self._calculate_category_distribution(recent_errors),
                "component_distribution": self._calculate_component_distribution(recent_errors),
                "recovery_time_stats": self._calculate_recovery_time_stats(recent_errors),
                "pattern_summary": self._summarize_detected_patterns()
            }
            
            return stats
    
    def _classify_error_category(self, error: Exception, message: str, 
                               context: Dict[str, Any]) -> ErrorCategory:
        """Classify error into primary category."""
        error_type = type(error).__name__
        component = context.get("component", "").lower()
        
        # Apply classification rules
        for rule in self.classification_rules:
            if self._rule_matches(rule, error_type, message, component, context):
                return rule["category"]
        
        # Default classification based on error type and component
        if "quantum" in component or "circuit" in component:
            return ErrorCategory.QUANTUM_COMPUTATION
        elif "embedding" in component or "similarity" in component:
            return ErrorCategory.CLASSICAL_COMPUTATION
        elif "memory" in message.lower() or "timeout" in message.lower():
            return ErrorCategory.SYSTEM_RESOURCE
        elif "validation" in message.lower() or "invalid" in message.lower():
            return ErrorCategory.DATA_VALIDATION
        elif "performance" in message.lower() or "latency" in message.lower():
            return ErrorCategory.PERFORMANCE_DEGRADATION
        elif "connection" in message.lower() or "network" in message.lower():
            return ErrorCategory.NETWORK_CONNECTIVITY
        elif "config" in message.lower():
            return ErrorCategory.CONFIGURATION
        else:
            return ErrorCategory.SYSTEM_RESOURCE  # Default
    
    def _assess_base_severity(self, error: Exception, category: ErrorCategory,
                            context: Dict[str, Any]) -> ErrorSeverity:
        """Assess base severity before context adjustments."""
        error_type = type(error).__name__
        
        # Apply severity rules
        for rule in self.severity_rules:
            if self._severity_rule_matches(rule, error_type, category, context):
                return rule["severity"]
        
        # Default severity based on category and error type
        if category == ErrorCategory.QUANTUM_COMPUTATION:
            if "timeout" in str(error).lower():
                return ErrorSeverity.MEDIUM
            elif "circuit" in str(error).lower():
                return ErrorSeverity.HIGH
            else:
                return ErrorSeverity.MEDIUM
        elif category == ErrorCategory.SYSTEM_RESOURCE:
            if "memory" in str(error).lower():
                return ErrorSeverity.HIGH
            elif "timeout" in str(error).lower():
                return ErrorSeverity.MEDIUM
            else:
                return ErrorSeverity.LOW
        elif category == ErrorCategory.PERFORMANCE_DEGRADATION:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _adjust_severity_by_context(self, base_severity: ErrorSeverity,
                                  context: Dict[str, Any]) -> ErrorSeverity:
        """Adjust severity based on operational context."""
        severity_levels = [ErrorSeverity.INFO, ErrorSeverity.LOW, 
                          ErrorSeverity.MEDIUM, ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
        current_index = severity_levels.index(base_severity)
        
        # Check for severity escalation factors
        escalation_factors = 0
        
        # High system load
        if context.get("cpu_usage", 0) > 90:
            escalation_factors += 1
        if context.get("memory_usage", 0) > 95:
            escalation_factors += 1
        
        # Critical operation
        if context.get("critical_operation", False):
            escalation_factors += 1
        
        # Production environment
        if context.get("environment") == "production":
            escalation_factors += 1
        
        # Recent error frequency
        if context.get("recent_error_count", 0) > 10:
            escalation_factors += 1
        
        # Error during recovery
        if context.get("in_recovery", False):
            escalation_factors += 2
        
        # Apply escalation
        new_index = min(len(severity_levels) - 1, current_index + escalation_factors)
        
        return severity_levels[new_index]
    
    def _assess_recoverability(self, error: Exception, category: ErrorCategory,
                             context: Dict[str, Any]) -> bool:
        """Assess if error is recoverable."""
        error_type = type(error).__name__
        
        # Non-recoverable error types
        non_recoverable = {
            "SystemExit", "KeyboardInterrupt", "MemoryError",
            "RecursionError", "SyntaxError", "ImportError"
        }
        
        if error_type in non_recoverable:
            return False
        
        # Category-based recoverability
        if category == ErrorCategory.QUANTUM_COMPUTATION:
            return True  # Most quantum errors are recoverable via fallback
        elif category == ErrorCategory.CLASSICAL_COMPUTATION:
            return True  # Classical computations usually recoverable
        elif category == ErrorCategory.SYSTEM_RESOURCE:
            return "memory" not in str(error).lower()  # Memory errors harder to recover
        elif category == ErrorCategory.DATA_VALIDATION:
            return False  # Data validation errors need fixing
        elif category == ErrorCategory.CONFIGURATION:
            return False  # Config errors need manual fix
        else:
            return True  # Default to recoverable
    
    def _check_fallback_availability(self, category: ErrorCategory,
                                   context: Dict[str, Any]) -> bool:
        """Check if fallback options are available."""
        component = context.get("component", "").lower()
        
        if category == ErrorCategory.QUANTUM_COMPUTATION:
            return True  # Classical fallback available
        elif category == ErrorCategory.CLASSICAL_COMPUTATION:
            if "similarity" in component:
                return True  # Multiple similarity methods available
            return False
        elif category == ErrorCategory.PERFORMANCE_DEGRADATION:
            return True  # Can degrade gracefully
        else:
            return False
    
    def _estimate_error_impact(self, error: Exception, category: ErrorCategory,
                             context: Dict[str, Any]) -> Dict[str, float]:
        """Estimate impact on different system metrics."""
        impact = {
            "availability": 0.0,
            "performance": 0.0,
            "accuracy": 0.0,
            "user_experience": 0.0
        }
        
        if category == ErrorCategory.QUANTUM_COMPUTATION:
            impact["accuracy"] = 0.1  # Slight accuracy impact with classical fallback
            impact["performance"] = 0.05  # Minor performance impact
        elif category == ErrorCategory.SYSTEM_RESOURCE:
            impact["availability"] = 0.3
            impact["performance"] = 0.5
            impact["user_experience"] = 0.4
        elif category == ErrorCategory.PERFORMANCE_DEGRADATION:
            impact["performance"] = 0.8
            impact["user_experience"] = 0.6
        elif category == ErrorCategory.DATA_VALIDATION:
            impact["accuracy"] = 0.9
            impact["user_experience"] = 0.7
        
        return impact
    
    def _estimate_recovery_time(self, category: ErrorCategory, severity: ErrorSeverity,
                              context: Dict[str, Any]) -> float:
        """Estimate recovery time in seconds."""
        base_times = {
            ErrorSeverity.INFO: 0,
            ErrorSeverity.LOW: 5,
            ErrorSeverity.MEDIUM: 30,
            ErrorSeverity.HIGH: 120,
            ErrorSeverity.CRITICAL: 300
        }
        
        base_time = base_times[severity]
        
        # Category adjustments
        if category == ErrorCategory.QUANTUM_COMPUTATION:
            base_time *= 0.5  # Fast fallback to classical
        elif category == ErrorCategory.SYSTEM_RESOURCE:
            base_time *= 2.0  # Resource issues take longer
        elif category == ErrorCategory.CONFIGURATION:
            base_time *= 5.0  # Config issues need manual intervention
        
        return base_time
    
    def _generate_suggested_actions(self, error: Exception, category: ErrorCategory,
                                  severity: ErrorSeverity, context: Dict[str, Any]) -> List[str]:
        """Generate context-specific suggested actions."""
        actions = []
        
        # Category-specific actions
        if category == ErrorCategory.QUANTUM_COMPUTATION:
            actions.extend([
                "Switch to classical similarity computation",
                "Verify quantum circuit parameters",
                "Check quantum backend availability"
            ])
        elif category == ErrorCategory.SYSTEM_RESOURCE:
            actions.extend([
                "Monitor system resource usage",
                "Reduce batch size or concurrent operations",
                "Free up system resources"
            ])
        elif category == ErrorCategory.PERFORMANCE_DEGRADATION:
            actions.extend([
                "Enable performance optimization",
                "Use simplified computation methods",
                "Check system load and reduce if necessary"
            ])
        
        # Severity-specific actions
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            actions.insert(0, "Escalate to system administrator")
            actions.append("Consider service degradation to maintain availability")
        
        return actions[:5]  # Limit to top 5 actions
    
    def _identify_related_components(self, component: str, category: ErrorCategory) -> List[str]:
        """Identify components that might be affected."""
        related = []
        
        if component in self.component_dependencies:
            related.extend(self.component_dependencies[component])
        
        # Category-based relationships
        if category == ErrorCategory.QUANTUM_COMPUTATION:
            related.extend(["classical_similarity", "embedding_processor"])
        elif category == ErrorCategory.SYSTEM_RESOURCE:
            related.extend(["all_components"])  # Resource issues affect everything
        
        return list(set(related))
    
    def _analyze_error_patterns(self, error_type: str, component: str,
                              context: Dict[str, Any]) -> List[str]:
        """Analyze current error for pattern indicators."""
        indicators = []
        
        # Check recent error history for patterns
        recent_time = time.time() - 300  # Last 5 minutes
        recent_errors = [
            error for error in self.error_history
            if error.timestamp >= recent_time
        ]
        
        # Frequency pattern
        if len(recent_errors) > 10:
            indicators.append("high_frequency_errors")
        
        # Component pattern
        component_errors = [e for e in recent_errors if e.component == component]
        if len(component_errors) > 5:
            indicators.append("component_specific_pattern")
        
        # Error type pattern
        type_errors = [e for e in recent_errors if e.error_type == error_type]
        if len(type_errors) > 3:
            indicators.append("recurring_error_type")
        
        return indicators
    
    def _calculate_classification_confidence(self, error: Exception, category: ErrorCategory,
                                           severity: ErrorSeverity, context: Dict[str, Any]) -> float:
        """Calculate confidence in classification."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on clear indicators
        error_message = str(error).lower()
        
        # Clear category indicators
        if category == ErrorCategory.QUANTUM_COMPUTATION and "quantum" in error_message:
            confidence += 0.3
        elif category == ErrorCategory.SYSTEM_RESOURCE and ("memory" in error_message or "timeout" in error_message):
            confidence += 0.3
        
        # Context richness
        if len(context) > 5:
            confidence += 0.1
        
        # Known error patterns
        if any(indicator in ["recurring_error_type", "component_specific_pattern"] 
               for indicator in self._analyze_error_patterns(type(error).__name__, 
                                                           context.get("component", ""), context)):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _record_error_for_pattern_analysis(self, error: Exception, 
                                         classification: ErrorClassification,
                                         context: Dict[str, Any]) -> None:
        """Record error for pattern analysis."""
        with self._lock:
            error_metrics = ErrorMetrics(
                error_type=classification.error_type,
                timestamp=time.time(),
                component=context.get("component", "unknown"),
                severity=classification.severity,
                recovery_time_s=classification.recovery_time_estimate_s,
                context=context
            )
            
            self.error_history.append(error_metrics)
    
    def _rule_matches(self, rule: Dict[str, Any], error_type: str, message: str,
                     component: str, context: Dict[str, Any]) -> bool:
        """Check if classification rule matches."""
        # Error type matching
        if "error_types" in rule and error_type not in rule["error_types"]:
            return False
        
        # Message pattern matching
        if "message_patterns" in rule:
            if not any(pattern in message.lower() for pattern in rule["message_patterns"]):
                return False
        
        # Component matching
        if "components" in rule and component not in rule["components"]:
            return False
        
        return True
    
    def _severity_rule_matches(self, rule: Dict[str, Any], error_type: str,
                             category: ErrorCategory, context: Dict[str, Any]) -> bool:
        """Check if severity rule matches."""
        if "error_types" in rule and error_type not in rule["error_types"]:
            return False
        
        if "categories" in rule and category not in rule["categories"]:
            return False
        
        return True
    
    def _initialize_classification_rules(self) -> List[Dict[str, Any]]:
        """Initialize error classification rules."""
        return [
            {
                "error_types": ["QuantumCircuitError", "CircuitExecutionError"],
                "components": ["quantum_circuits", "quantum_engine"],
                "category": ErrorCategory.QUANTUM_COMPUTATION
            },
            {
                "message_patterns": ["memory", "out of memory", "allocation"],
                "category": ErrorCategory.SYSTEM_RESOURCE
            },
            {
                "message_patterns": ["timeout", "timed out"],
                "category": ErrorCategory.SYSTEM_RESOURCE
            },
            {
                "message_patterns": ["validation", "invalid", "format"],
                "category": ErrorCategory.DATA_VALIDATION
            },
            {
                "components": ["embedding_processor", "similarity_engine"],
                "category": ErrorCategory.CLASSICAL_COMPUTATION
            },
            {
                "message_patterns": ["performance", "latency", "slow"],
                "category": ErrorCategory.PERFORMANCE_DEGRADATION
            },
            {
                "message_patterns": ["connection", "network", "unreachable"],
                "category": ErrorCategory.NETWORK_CONNECTIVITY
            },
            {
                "message_patterns": ["config", "configuration", "setting"],
                "category": ErrorCategory.CONFIGURATION
            }
        ]
    
    def _initialize_severity_rules(self) -> List[Dict[str, Any]]:
        """Initialize severity assessment rules."""
        return [
            {
                "error_types": ["MemoryError", "SystemExit"],
                "severity": ErrorSeverity.CRITICAL
            },
            {
                "categories": [ErrorCategory.SYSTEM_RESOURCE],
                "severity": ErrorSeverity.HIGH
            },
            {
                "categories": [ErrorCategory.QUANTUM_COMPUTATION],
                "severity": ErrorSeverity.MEDIUM
            },
            {
                "categories": [ErrorCategory.DATA_VALIDATION],
                "severity": ErrorSeverity.LOW
            }
        ]
    
    def _initialize_pattern_detectors(self) -> Dict[str, Any]:
        """Initialize pattern detection functions."""
        return {
            "frequency_spike": self._detect_frequency_spike,
            "cascading_failure": self._detect_cascading_failure,
            "periodic_failure": self._detect_periodic_failure,
            "gradual_degradation": self._detect_gradual_degradation
        }
    
    def _initialize_component_dependencies(self) -> Dict[str, List[str]]:
        """Initialize component dependency mapping."""
        return {
            "quantum_engine": ["similarity_engine", "embedding_processor"],
            "similarity_engine": ["embedding_processor"],
            "search_engine": ["vector_index", "cache_manager"],
            "pipeline_manager": ["quantum_engine", "similarity_engine", "search_engine"]
        }
    
    def _initialize_performance_thresholds(self) -> Dict[str, float]:
        """Initialize performance thresholds for error detection."""
        return {
            "similarity_computation_ms": 150.0,
            "quantum_execution_ms": 100.0,
            "embedding_time_ms": 50.0,
            "memory_usage_percent": 90.0,
            "cpu_usage_percent": 85.0
        }
    
    def _detect_frequency_spike(self, errors: List[ErrorMetrics], window_s: int) -> Optional[ErrorPattern]:
        """Detect frequency spike patterns."""
        if len(errors) < 10:
            return None
        
        # Calculate error frequency
        frequency = len(errors) / (window_s / 3600)  # Errors per hour
        
        if frequency > 50:  # More than 50 errors per hour
            affected_components = list(set(error.component for error in errors))
            
            return ErrorPattern(
                pattern_type=ErrorPatternType.FREQUENCY_SPIKE,
                affected_components=affected_components,
                frequency=frequency,
                severity_trend="increasing",
                first_occurrence=min(error.timestamp for error in errors),
                last_occurrence=max(error.timestamp for error in errors),
                pattern_confidence=0.8
            )
        
        return None
    
    def _detect_cascading_failure(self, errors: List[ErrorMetrics], window_s: int) -> Optional[ErrorPattern]:
        """Detect cascading failure patterns."""
        if len(errors) < 5:
            return None
        
        # Group errors by component and check for cascade pattern
        component_errors = defaultdict(list)
        for error in errors:
            component_errors[error.component].append(error)
        
        # Check if errors spread across multiple components in sequence
        if len(component_errors) >= 3:
            # Sort components by first error time
            component_times = {
                comp: min(error.timestamp for error in errors)
                for comp, errors in component_errors.items()
            }
            
            sorted_components = sorted(component_times.keys(), key=lambda c: component_times[c])
            
            # Check if errors follow dependency chain
            cascade_detected = False
            for i in range(len(sorted_components) - 1):
                current_comp = sorted_components[i]
                next_comp = sorted_components[i + 1]
                
                time_diff = component_times[next_comp] - component_times[current_comp]
                if 0 < time_diff < 300:  # Errors within 5 minutes suggest cascade
                    cascade_detected = True
                    break
            
            if cascade_detected:
                return ErrorPattern(
                    pattern_type=ErrorPatternType.CASCADING_FAILURE,
                    affected_components=list(component_errors.keys()),
                    frequency=len(errors) / (window_s / 3600),
                    severity_trend="increasing",
                    first_occurrence=min(error.timestamp for error in errors),
                    last_occurrence=max(error.timestamp for error in errors),
                    pattern_confidence=0.9
                )
        
        return None
    
    def _detect_periodic_failure(self, errors: List[ErrorMetrics], window_s: int) -> Optional[ErrorPattern]:
        """Detect periodic failure patterns."""
        if len(errors) < 6:
            return None
        
        # Analyze time intervals between errors
        timestamps = sorted([error.timestamp for error in errors])
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        # Check for regular intervals (within 20% variance)
        if len(intervals) >= 5:
            avg_interval = np.mean(intervals)
            interval_std = np.std(intervals)
            
            # Periodic if standard deviation is less than 20% of mean
            if interval_std / avg_interval < 0.2 and avg_interval > 60:  # At least 1 minute intervals
                return ErrorPattern(
                    pattern_type=ErrorPatternType.PERIODIC_FAILURE,
                    affected_components=list(set(error.component for error in errors)),
                    frequency=1 / avg_interval * 3600,  # Frequency per hour
                    severity_trend="stable",
                    first_occurrence=timestamps[0],
                    last_occurrence=timestamps[-1],
                    pattern_confidence=0.7,
                    prediction={
                        "next_occurrence": timestamps[-1] + avg_interval,
                        "interval_seconds": avg_interval
                    }
                )
        
        return None
    
    def _detect_gradual_degradation(self, errors: List[ErrorMetrics], window_s: int) -> Optional[ErrorPattern]:
        """Detect gradual degradation patterns."""
        if len(errors) < 8:
            return None
        
        # Analyze severity trend over time
        severity_values = {
            ErrorSeverity.INFO: 1,
            ErrorSeverity.LOW: 2,
            ErrorSeverity.MEDIUM: 3,
            ErrorSeverity.HIGH: 4,
            ErrorSeverity.CRITICAL: 5
        }
        
        # Sort errors by time and check severity trend
        sorted_errors = sorted(errors, key=lambda e: e.timestamp)
        severity_scores = [severity_values[error.severity] for error in sorted_errors]
        
        # Calculate trend using linear regression
        times = np.array(range(len(severity_scores)))
        scores = np.array(severity_scores)
        
        if len(times) > 1:
            correlation = np.corrcoef(times, scores)[0, 1]
            
            if correlation > 0.6:  # Strong positive correlation indicates degradation
                return ErrorPattern(
                    pattern_type=ErrorPatternType.GRADUAL_DEGRADATION,
                    affected_components=list(set(error.component for error in errors)),
                    frequency=len(errors) / (window_s / 3600),
                    severity_trend="increasing",
                    first_occurrence=sorted_errors[0].timestamp,
                    last_occurrence=sorted_errors[-1].timestamp,
                    pattern_confidence=min(0.9, correlation),
                    prediction={
                        "degradation_rate": correlation,
                        "projected_critical_time": sorted_errors[-1].timestamp + (5 - scores[-1]) * 300
                    }
                )
        
        return None
    
    def _adjust_likelihood_by_context(self, base_likelihood: float, context: Dict[str, Any],
                                    pattern: ErrorPattern) -> float:
        """Adjust error likelihood based on current context."""
        adjusted_likelihood = base_likelihood
        
        # Time-based adjustment for periodic patterns
        if pattern.pattern_type == ErrorPatternType.PERIODIC_FAILURE and pattern.prediction:
            next_occurrence = pattern.prediction.get("next_occurrence", 0)
            current_time = time.time()
            time_to_next = abs(next_occurrence - current_time)
            
            # Higher likelihood as we approach predicted time
            if time_to_next < 300:  # Within 5 minutes
                adjusted_likelihood *= 2.0
            elif time_to_next < 900:  # Within 15 minutes
                adjusted_likelihood *= 1.5
        
        # Context-based adjustments
        if context.get("cpu_usage", 0) > 80:
            adjusted_likelihood *= 1.3
        if context.get("memory_usage", 0) > 85:
            adjusted_likelihood *= 1.4
        if context.get("recent_error_count", 0) > 5:
            adjusted_likelihood *= 1.2
        
        return min(1.0, adjusted_likelihood)
    
    def _get_baseline_error_predictions(self, component: str, operation: str,
                                      context: Dict[str, Any]) -> Dict[str, float]:
        """Get baseline error predictions for component/operation."""
        baseline = {}
        
        # Component-specific baselines
        if "quantum" in component:
            baseline["QuantumCircuitError"] = 0.05
            baseline["TimeoutError"] = 0.03
        elif "similarity" in component:
            baseline["SimilarityComputationError"] = 0.02
        elif "embedding" in component:
            baseline["EmbeddingProcessingError"] = 0.01
        
        # Operation-specific adjustments
        if operation == "batch_processing":
            for error_type in baseline:
                baseline[error_type] *= 1.5
        
        return baseline
    
    def _calculate_severity_distribution(self, errors: List[ErrorMetrics]) -> Dict[str, int]:
        """Calculate distribution of error severities."""
        distribution = defaultdict(int)
        for error in errors:
            distribution[error.severity.value] += 1
        return dict(distribution)
    
    def _calculate_category_distribution(self, errors: List[ErrorMetrics]) -> Dict[str, int]:
        """Calculate distribution of error categories."""
        # This would require storing category in ErrorMetrics
        # For now, return placeholder
        return {"quantum": 0, "classical": 0, "system": 0}
    
    def _calculate_component_distribution(self, errors: List[ErrorMetrics]) -> Dict[str, int]:
        """Calculate distribution of errors by component."""
        distribution = defaultdict(int)
        for error in errors:
            distribution[error.component] += 1
        return dict(distribution)
    
    def _calculate_recovery_time_stats(self, errors: List[ErrorMetrics]) -> Dict[str, float]:
        """Calculate recovery time statistics."""
        recovery_times = [
            error.recovery_time_s for error in errors 
            if error.recovery_time_s is not None
        ]
        
        if not recovery_times:
            return {"count": 0}
        
        return {
            "count": len(recovery_times),
            "mean": np.mean(recovery_times),
            "median": np.median(recovery_times),
            "p95": np.percentile(recovery_times, 95),
            "max": np.max(recovery_times)
        }
    
    def _summarize_detected_patterns(self) -> Dict[str, Any]:
        """Summarize currently detected patterns."""
        return {
            "active_patterns": len(self.detected_patterns),
            "pattern_types": list(set(
                pattern.pattern_type.value 
                for pattern in self.detected_patterns.values()
            )),
            "high_risk_patterns": len([
                pattern for pattern in self.detected_patterns.values()
                if pattern.pattern_confidence > 0.8
            ])
        }


__all__ = [
    "ErrorSeverity",
    "ErrorCategory", 
    "ErrorPatternType",
    "ErrorClassification",
    "ErrorPattern",
    "ErrorMetrics",
    "ErrorClassifier"
]
"""
Intelligent alerting and notification system for QuantumRerank monitoring.

This module provides comprehensive alerting capabilities with adaptive thresholds,
alert correlation, and intelligent notification management.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import numpy as np

from .metrics_collector import MetricsCollector
from ..utils import get_logger


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert lifecycle status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class AlertCategory(Enum):
    """Alert categories for classification."""
    PERFORMANCE = "performance"
    AVAILABILITY = "availability"
    RESOURCE = "resource"
    QUANTUM = "quantum"
    SECURITY = "security"
    CONFIGURATION = "configuration"


@dataclass
class AlertRule:
    """Definition of an alerting rule."""
    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # "greater_than", "less_than", "equals", "not_equals"
    threshold: float
    severity: AlertSeverity
    category: AlertCategory
    evaluation_window_seconds: int = 60
    min_samples: int = 3
    consecutive_breaches: int = 1
    enabled: bool = True
    auto_resolve: bool = True
    suppression_duration_seconds: int = 300  # 5 minutes
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Individual performance alert instance."""
    alert_id: str
    rule_id: str
    alert_name: str
    description: str
    severity: AlertSeverity
    category: AlertCategory
    status: AlertStatus
    metric_name: str
    current_value: float
    threshold_value: float
    triggered_at: float
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None
    suppressed_until: Optional[float] = None
    correlation_id: Optional[str] = None
    affected_components: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    notification_sent: bool = False


@dataclass
class AlertCorrelation:
    """Correlation between related alerts."""
    correlation_id: str
    related_alerts: List[str]  # Alert IDs
    correlation_type: str  # "cascade", "common_cause", "temporal"
    confidence_score: float
    description: str
    created_at: float


@dataclass
class NotificationChannel:
    """Notification channel configuration."""
    channel_id: str
    channel_type: str  # "email", "slack", "webhook", "log"
    config: Dict[str, Any]
    enabled: bool = True
    severity_filter: List[AlertSeverity] = field(default_factory=lambda: list(AlertSeverity))


class AlertManager:
    """
    Comprehensive alerting and notification system.
    
    Provides intelligent alerting with adaptive thresholds, alert correlation,
    and multi-channel notification capabilities.
    """
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.logger = get_logger(__name__)
        
        # Alert state
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # Alert correlation
        self.correlations: Dict[str, AlertCorrelation] = {}
        self.correlation_window_seconds = 300  # 5 minutes
        
        # Notification channels
        self.notification_channels: Dict[str, NotificationChannel] = {}
        
        # Alert evaluation
        self.evaluation_interval_seconds = 30
        self._evaluator_thread = None
        self._evaluation_active = False
        
        # Adaptive thresholds
        self.adaptive_thresholds_enabled = True
        self.threshold_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Suppression and throttling
        self.global_suppression_active = False
        self.alert_rate_limit = 10  # Max alerts per minute
        self.recent_alerts: deque = deque(maxlen=100)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize default rules and channels
        self._initialize_default_alert_rules()
        self._initialize_default_notification_channels()
        
        self.logger.info("Initialized AlertManager")
    
    def start_alert_evaluation(self) -> None:
        """Start continuous alert evaluation."""
        if self._evaluation_active:
            return
        
        self._evaluation_active = True
        self._evaluator_thread = threading.Thread(
            target=self._alert_evaluation_loop, daemon=True
        )
        self._evaluator_thread.start()
        
        self.logger.info("Started alert evaluation")
    
    def stop_alert_evaluation(self) -> None:
        """Stop continuous alert evaluation."""
        self._evaluation_active = False
        if self._evaluator_thread:
            self._evaluator_thread.join(timeout=10.0)
        
        self.logger.info("Stopped alert evaluation")
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add a new alert rule."""
        with self._lock:
            self.alert_rules[rule.rule_id] = rule
            self.logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_id: str) -> None:
        """Remove an alert rule."""
        with self._lock:
            if rule_id in self.alert_rules:
                del self.alert_rules[rule_id]
                self.logger.info(f"Removed alert rule: {rule_id}")
    
    def add_notification_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel."""
        with self._lock:
            self.notification_channels[channel.channel_id] = channel
            self.logger.info(f"Added notification channel: {channel.channel_id}")
    
    def trigger_alert(self, alert: PerformanceAlert) -> None:
        """Manually trigger an alert."""
        with self._lock:
            # Check rate limiting
            if self._is_rate_limited():
                self.logger.warning("Alert rate limit exceeded, skipping alert")
                return
            
            # Check for existing alert
            if alert.alert_id in self.active_alerts:
                # Update existing alert
                existing_alert = self.active_alerts[alert.alert_id]
                existing_alert.current_value = alert.current_value
                existing_alert.metadata.update(alert.metadata)
            else:
                # New alert
                self.active_alerts[alert.alert_id] = alert
                self.alert_history.append(alert)
                self.recent_alerts.append(time.time())
                
                # Detect correlations
                self._detect_alert_correlations(alert)
                
                # Send notifications
                self._send_alert_notifications(alert)
                
                self.logger.warning(f"Alert triggered: {alert.alert_name} ({alert.severity.value})")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an active alert."""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = time.time()
                alert.metadata["acknowledged_by"] = acknowledged_by
                
                self.logger.info(f"Alert acknowledged: {alert.alert_name}")
                return True
        
        return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an active alert."""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = time.time()
                alert.metadata["resolved_by"] = resolved_by
                
                # Move to history and remove from active
                del self.active_alerts[alert_id]
                
                self.logger.info(f"Alert resolved: {alert.alert_name}")
                return True
        
        return False
    
    def suppress_alerts(self, duration_seconds: int = 3600) -> None:
        """Globally suppress alerts for a duration."""
        self.global_suppression_active = True
        
        def clear_suppression():
            time.sleep(duration_seconds)
            self.global_suppression_active = False
            self.logger.info("Global alert suppression cleared")
        
        threading.Thread(target=clear_suppression, daemon=True).start()
        self.logger.info(f"Global alert suppression activated for {duration_seconds} seconds")
    
    def get_active_alerts(self, severity_filter: Optional[List[AlertSeverity]] = None) -> List[PerformanceAlert]:
        """Get currently active alerts."""
        with self._lock:
            alerts = list(self.active_alerts.values())
            
            if severity_filter:
                alerts = [a for a in alerts if a.severity in severity_filter]
            
            # Sort by severity and time
            severity_order = {AlertSeverity.CRITICAL: 0, AlertSeverity.ERROR: 1, 
                            AlertSeverity.WARNING: 2, AlertSeverity.INFO: 3}
            
            alerts.sort(key=lambda a: (severity_order[a.severity], -a.triggered_at))
            
            return alerts
    
    def get_alert_statistics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive alert statistics."""
        cutoff_time = time.time() - (time_window_hours * 3600)
        
        with self._lock:
            # Recent alerts from history
            recent_alerts = [
                alert for alert in self.alert_history
                if alert.triggered_at >= cutoff_time
            ]
            
            # Statistics
            stats = {
                "time_window_hours": time_window_hours,
                "total_alerts": len(recent_alerts),
                "active_alerts": len(self.active_alerts),
                "alerts_by_severity": defaultdict(int),
                "alerts_by_category": defaultdict(int),
                "top_alerting_metrics": defaultdict(int),
                "resolution_time_stats": {},
                "alert_frequency": len(recent_alerts) / max(1, time_window_hours),
                "correlations_detected": len(self.correlations)
            }
            
            # Analyze recent alerts
            for alert in recent_alerts:
                stats["alerts_by_severity"][alert.severity.value] += 1
                stats["alerts_by_category"][alert.category.value] += 1
                stats["top_alerting_metrics"][alert.metric_name] += 1
            
            # Resolution time analysis
            resolved_alerts = [a for a in recent_alerts if a.resolved_at is not None]
            if resolved_alerts:
                resolution_times = [
                    a.resolved_at - a.triggered_at for a in resolved_alerts
                ]
                stats["resolution_time_stats"] = {
                    "avg_resolution_minutes": np.mean(resolution_times) / 60,
                    "p95_resolution_minutes": np.percentile(resolution_times, 95) / 60,
                    "fastest_resolution_minutes": np.min(resolution_times) / 60,
                    "slowest_resolution_minutes": np.max(resolution_times) / 60
                }
            
            return dict(stats)
    
    def _alert_evaluation_loop(self) -> None:
        """Main alert evaluation loop."""
        while self._evaluation_active:
            try:
                self._evaluate_all_rules()
                self._check_auto_resolution()
                self._update_adaptive_thresholds()
                
                time.sleep(self.evaluation_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in alert evaluation loop: {e}")
                time.sleep(30.0)
    
    def _evaluate_all_rules(self) -> None:
        """Evaluate all active alert rules."""
        if self.global_suppression_active:
            return
        
        for rule in self.alert_rules.values():
            if rule.enabled:
                self._evaluate_rule(rule)
    
    def _evaluate_rule(self, rule: AlertRule) -> None:
        """Evaluate a single alert rule."""
        # Get recent samples for the metric
        samples = self.metrics_collector.get_samples_in_time_window(
            rule.metric_name, rule.evaluation_window_seconds
        )
        
        if len(samples) < rule.min_samples:
            return
        
        # Extract values
        values = [sample.value for sample in samples]
        current_value = np.mean(values)  # Use average of recent samples
        
        # Get effective threshold (may be adaptive)
        effective_threshold = self._get_effective_threshold(rule)
        
        # Evaluate condition
        condition_met = self._evaluate_condition(
            current_value, effective_threshold, rule.condition
        )
        
        alert_id = f"{rule.rule_id}_{rule.metric_name}"
        
        if condition_met:
            # Check if alert already exists
            if alert_id not in self.active_alerts:
                # Create new alert
                alert = PerformanceAlert(
                    alert_id=alert_id,
                    rule_id=rule.rule_id,
                    alert_name=rule.name,
                    description=rule.description.format(
                        metric=rule.metric_name,
                        value=current_value,
                        threshold=effective_threshold
                    ),
                    severity=rule.severity,
                    category=rule.category,
                    status=AlertStatus.ACTIVE,
                    metric_name=rule.metric_name,
                    current_value=current_value,
                    threshold_value=effective_threshold,
                    triggered_at=time.time(),
                    metadata={
                        "samples_count": len(samples),
                        "rule_tags": rule.tags
                    }
                )
                
                self.trigger_alert(alert)
        else:
            # Condition not met - check for auto-resolution
            if alert_id in self.active_alerts and rule.auto_resolve:
                self.resolve_alert(alert_id, "auto_resolved")
    
    def _evaluate_condition(self, value: float, threshold: float, condition: str) -> bool:
        """Evaluate alert condition."""
        if condition == "greater_than":
            return value > threshold
        elif condition == "less_than":
            return value < threshold
        elif condition == "equals":
            return abs(value - threshold) < 0.001
        elif condition == "not_equals":
            return abs(value - threshold) >= 0.001
        else:
            return False
    
    def _get_effective_threshold(self, rule: AlertRule) -> float:
        """Get effective threshold (possibly adaptive)."""
        if not self.adaptive_thresholds_enabled:
            return rule.threshold
        
        # Get historical values for adaptive threshold calculation
        threshold_key = f"{rule.rule_id}_{rule.metric_name}"
        
        if threshold_key not in self.threshold_history:
            return rule.threshold
        
        # Use statistical analysis to adapt thresholds
        historical_samples = self.metrics_collector.get_samples_in_time_window(
            rule.metric_name, 7 * 24 * 3600  # 7 days
        )
        
        if len(historical_samples) < 100:
            return rule.threshold
        
        values = [s.value for s in historical_samples]
        
        # Calculate adaptive threshold based on percentiles
        if rule.condition == "greater_than":
            # Use 95th percentile for upper bounds
            adaptive_threshold = np.percentile(values, 95)
            # Don't make threshold too aggressive
            return max(rule.threshold, adaptive_threshold * 0.8)
        elif rule.condition == "less_than":
            # Use 5th percentile for lower bounds
            adaptive_threshold = np.percentile(values, 5)
            return min(rule.threshold, adaptive_threshold * 1.2)
        
        return rule.threshold
    
    def _check_auto_resolution(self) -> None:
        """Check for auto-resolution of active alerts."""
        current_time = time.time()
        
        to_resolve = []
        for alert_id, alert in self.active_alerts.items():
            # Check if alert should be auto-resolved
            rule = self.alert_rules.get(alert.rule_id)
            if not rule or not rule.auto_resolve:
                continue
            
            # Check if suppression period has passed
            if (alert.suppressed_until and 
                current_time < alert.suppressed_until):
                continue
            
            # Re-evaluate the condition
            samples = self.metrics_collector.get_samples_in_time_window(
                alert.metric_name, rule.evaluation_window_seconds
            )
            
            if len(samples) >= rule.min_samples:
                values = [s.value for s in samples]
                current_value = np.mean(values)
                
                condition_met = self._evaluate_condition(
                    current_value, alert.threshold_value, rule.condition
                )
                
                if not condition_met:
                    to_resolve.append(alert_id)
        
        # Resolve alerts that no longer meet conditions
        for alert_id in to_resolve:
            self.resolve_alert(alert_id, "auto_resolved")
    
    def _update_adaptive_thresholds(self) -> None:
        """Update adaptive thresholds based on recent data."""
        if not self.adaptive_thresholds_enabled:
            return
        
        # This is a placeholder for more sophisticated adaptive threshold logic
        # In practice, this would use ML models to predict optimal thresholds
        pass
    
    def _detect_alert_correlations(self, new_alert: PerformanceAlert) -> None:
        """Detect correlations between alerts."""
        current_time = time.time()
        
        # Find alerts within correlation window
        related_alerts = []
        for alert_id, alert in self.active_alerts.items():
            if (alert_id != new_alert.alert_id and 
                current_time - alert.triggered_at <= self.correlation_window_seconds):
                related_alerts.append(alert)
        
        if len(related_alerts) >= 1:
            # Create correlation
            correlation_id = f"corr_{new_alert.alert_id}_{current_time}"
            
            # Determine correlation type
            correlation_type = "temporal"  # Default
            confidence = 0.5
            
            # Check for cascade (alerts triggered in sequence)
            if self._is_cascade_pattern(new_alert, related_alerts):
                correlation_type = "cascade"
                confidence = 0.8
            
            # Check for common cause (similar metrics/components)
            elif self._is_common_cause_pattern(new_alert, related_alerts):
                correlation_type = "common_cause"
                confidence = 0.9
            
            correlation = AlertCorrelation(
                correlation_id=correlation_id,
                related_alerts=[new_alert.alert_id] + [a.alert_id for a in related_alerts],
                correlation_type=correlation_type,
                confidence_score=confidence,
                description=f"Correlated alerts detected: {correlation_type}",
                created_at=current_time
            )
            
            self.correlations[correlation_id] = correlation
            new_alert.correlation_id = correlation_id
    
    def _is_cascade_pattern(self, new_alert: PerformanceAlert, related_alerts: List[PerformanceAlert]) -> bool:
        """Check if alerts follow a cascade pattern."""
        # Simple cascade detection: alerts in different components triggered in sequence
        components = set([new_alert.category.value])
        for alert in related_alerts:
            components.add(alert.category.value)
        
        return len(components) > 1
    
    def _is_common_cause_pattern(self, new_alert: PerformanceAlert, related_alerts: List[PerformanceAlert]) -> bool:
        """Check if alerts share a common cause."""
        # Simple common cause detection: similar metric names or categories
        for alert in related_alerts:
            if (alert.category == new_alert.category or 
                any(keyword in alert.metric_name for keyword in new_alert.metric_name.split('.'))):
                return True
        
        return False
    
    def _send_alert_notifications(self, alert: PerformanceAlert) -> None:
        """Send alert notifications through configured channels."""
        for channel in self.notification_channels.values():
            if (channel.enabled and 
                alert.severity in channel.severity_filter):
                self._send_notification(channel, alert)
    
    def _send_notification(self, channel: NotificationChannel, alert: PerformanceAlert) -> None:
        """Send notification through a specific channel."""
        try:
            if channel.channel_type == "log":
                self._send_log_notification(alert)
            elif channel.channel_type == "webhook":
                self._send_webhook_notification(channel, alert)
            # Add more notification types as needed
            
        except Exception as e:
            self.logger.error(f"Failed to send notification via {channel.channel_id}: {e}")
    
    def _send_log_notification(self, alert: PerformanceAlert) -> None:
        """Send notification to log."""
        log_message = (
            f"ALERT [{alert.severity.value.upper()}] {alert.alert_name}: "
            f"{alert.description} (value: {alert.current_value}, threshold: {alert.threshold_value})"
        )
        
        if alert.severity == AlertSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif alert.severity == AlertSeverity.ERROR:
            self.logger.error(log_message)
        elif alert.severity == AlertSeverity.WARNING:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _send_webhook_notification(self, channel: NotificationChannel, alert: PerformanceAlert) -> None:
        """Send notification via webhook."""
        # Placeholder for webhook implementation
        self.logger.info(f"Would send webhook notification for alert: {alert.alert_name}")
    
    def _is_rate_limited(self) -> bool:
        """Check if alert rate limiting is active."""
        current_time = time.time()
        recent_count = len([
            t for t in self.recent_alerts
            if current_time - t <= 60  # Last minute
        ])
        
        return recent_count >= self.alert_rate_limit
    
    def _initialize_default_alert_rules(self) -> None:
        """Initialize default alert rules."""
        default_rules = [
            AlertRule(
                rule_id="high_similarity_latency",
                name="High Similarity Computation Latency",
                description="Similarity computation latency is {value:.1f}ms (threshold: {threshold}ms)",
                metric_name="operation.similarity_computation.duration",
                condition="greater_than",
                threshold=150.0,
                severity=AlertSeverity.WARNING,
                category=AlertCategory.PERFORMANCE
            ),
            AlertRule(
                rule_id="critical_similarity_latency",
                name="Critical Similarity Computation Latency",
                description="Similarity computation latency is critically high: {value:.1f}ms",
                metric_name="operation.similarity_computation.duration",
                condition="greater_than",
                threshold=300.0,
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.PERFORMANCE
            ),
            AlertRule(
                rule_id="low_quantum_fidelity",
                name="Low Quantum Fidelity",
                description="Quantum fidelity dropped to {value:.3f} (threshold: {threshold})",
                metric_name="quantum.fidelity",
                condition="less_than",
                threshold=0.85,
                severity=AlertSeverity.ERROR,
                category=AlertCategory.QUANTUM
            ),
            AlertRule(
                rule_id="high_cpu_usage",
                name="High CPU Usage",
                description="System CPU usage is {value:.1f}% (threshold: {threshold}%)",
                metric_name="system.cpu_usage",
                condition="greater_than",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                category=AlertCategory.RESOURCE
            ),
            AlertRule(
                rule_id="high_memory_usage",
                name="High Memory Usage",
                description="System memory usage is {value:.1f}% (threshold: {threshold}%)",
                metric_name="system.memory_usage",
                condition="greater_than",
                threshold=90.0,
                severity=AlertSeverity.ERROR,
                category=AlertCategory.RESOURCE
            ),
            AlertRule(
                rule_id="quantum_error_rate",
                name="High Quantum Error Rate",
                description="Quantum error rate is {value:.1f} errors/sec",
                metric_name="quantum.errors",
                condition="greater_than",
                threshold=1.0,
                severity=AlertSeverity.ERROR,
                category=AlertCategory.QUANTUM
            )
        ]
        
        for rule in default_rules:
            self.add_alert_rule(rule)
    
    def _initialize_default_notification_channels(self) -> None:
        """Initialize default notification channels."""
        # Log channel (always enabled)
        log_channel = NotificationChannel(
            channel_id="default_log",
            channel_type="log",
            config={},
            severity_filter=list(AlertSeverity)
        )
        self.add_notification_channel(log_channel)


__all__ = [
    "AlertSeverity",
    "AlertStatus",
    "AlertCategory",
    "AlertRule",
    "PerformanceAlert",
    "AlertCorrelation",
    "NotificationChannel",
    "AlertManager"
]
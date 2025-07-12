"""
Comprehensive Testing Framework for QuantumRerank.

This module provides a multi-level testing architecture covering unit tests,
integration tests, performance tests, quantum testing, chaos engineering,
and automated validation with production readiness assessment.

Implements PRD Section 7.2: Success criteria requiring comprehensive testing validation.

Testing Levels:
- UNIT: Individual component testing
- INTEGRATION: Component interaction testing  
- SYSTEM: End-to-end system testing
- PERFORMANCE: Performance and scalability testing
- QUANTUM: Quantum-specific computation testing
- CHAOS: Resilience and fault tolerance testing

Features:
- Multi-format reporting (HTML, JSON, JUnit, Markdown)
- Performance monitoring and metrics collection
- PRD compliance validation
- Quantum fidelity and parameter testing
- Chaos engineering for resilience validation
- Production readiness assessment
"""

from .test_architecture import (
    TestLevel,
    TestCategory,
    TestFrameworkConfig,
    BaseTestFramework,
    TestCase,
    TestResult
)
from .test_runner import (
    TestRunner,
    TestExecutor,
    TestSuiteRunner,
    TestExecutionContext
)
from .test_reporter import (
    TestReporter,
    TestReport,
    TestSuiteReport,
    ProductionReadinessReport,
    ReportFormat
)
from .test_utilities import (
    TestTimer,
    TestMetricsCollector,
    TestDataValidator,
    TestEnvironmentManager,
    TimingResult,
    ResourceMetrics,
    time_function,
    measure_performance,
    validate_test_prerequisites
)

__all__ = [
    # Core Architecture
    "TestLevel",
    "TestCategory", 
    "TestFrameworkConfig",
    "BaseTestFramework",
    "TestCase",
    "TestResult",
    
    # Test Execution
    "TestRunner",
    "TestExecutor",
    "TestSuiteRunner", 
    "TestExecutionContext",
    
    # Reporting
    "TestReporter",
    "TestReport",
    "TestSuiteReport",
    "ProductionReadinessReport",
    "ReportFormat",
    
    # Utilities
    "TestTimer",
    "TestMetricsCollector",
    "TestDataValidator",
    "TestEnvironmentManager",
    "TimingResult",
    "ResourceMetrics",
    
    # Helper Functions
    "time_function",
    "measure_performance", 
    "validate_test_prerequisites"
]
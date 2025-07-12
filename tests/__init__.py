"""
Comprehensive Testing Suite for QuantumRerank.

This package provides complete testing coverage across all system components:

Testing Modules:
- framework: Core testing framework with multi-level architecture
- quantum: Specialized quantum computation testing
- performance: Performance and PRD compliance testing  
- chaos: Chaos engineering and resilience testing

Test Levels:
- Unit: Individual component testing
- Integration: Component interaction testing
- System: End-to-end system testing  
- Performance: Performance and scalability validation
- Quantum: Quantum-specific computation testing
- Chaos: Fault tolerance and resilience testing

Features:
- Multi-format test reporting (HTML, JSON, JUnit, Markdown)
- Performance monitoring and metrics collection
- PRD compliance validation
- Quantum fidelity and parameter accuracy testing
- Chaos engineering for system resilience
- Production readiness assessment
- Automated test discovery and execution
"""

from . import framework
from . import quantum
from . import performance  
from . import chaos

# Core framework components
from .framework import (
    TestLevel,
    TestCategory,
    TestRunner,
    TestReporter,
    TestSuiteRunner,
    ProductionReadinessReport
)

# Quantum testing components  
from .quantum import (
    QuantumTestFramework,
    FidelityAccuracyTester,
    CircuitOptimizationTester,
    ParameterPredictionTester
)

# Performance testing components
from .performance import (
    PRDComplianceFramework,
    PRDTarget,
    PerformanceTestCase
)

# Chaos testing components
from .chaos import (
    ChaosTestFramework, 
    ChaosType,
    ChaosExperiment
)

__all__ = [
    # Modules
    "framework",
    "quantum", 
    "performance",
    "chaos",
    
    # Core Framework
    "TestLevel",
    "TestCategory", 
    "TestRunner",
    "TestReporter",
    "TestSuiteRunner",
    "ProductionReadinessReport",
    
    # Quantum Testing
    "QuantumTestFramework",
    "FidelityAccuracyTester", 
    "CircuitOptimizationTester",
    "ParameterPredictionTester",
    
    # Performance Testing
    "PRDComplianceFramework",
    "PRDTarget",
    "PerformanceTestCase",
    
    # Chaos Testing
    "ChaosTestFramework",
    "ChaosType", 
    "ChaosExperiment"
]
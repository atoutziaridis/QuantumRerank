"""
Multi-level testing architecture for comprehensive validation.

This module defines the core testing framework architecture with support for
unit tests, integration tests, system tests, performance tests, and chaos tests.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Callable, Type, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import concurrent.futures
import time

from quantum_rerank.utils.logging_config import get_logger

logger = get_logger(__name__)


class TestLevel(Enum):
    """Testing framework levels."""
    UNIT = "unit"                    # Fast, isolated component tests
    INTEGRATION = "integration"      # Component interaction tests
    SYSTEM = "system"               # End-to-end system tests
    PERFORMANCE = "performance"     # Performance and load tests
    QUANTUM = "quantum"             # Quantum-specific tests
    CHAOS = "chaos"                 # Error injection and resilience tests


class TestCategory(Enum):
    """Test categories for organization."""
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    SECURITY = "security"
    RELIABILITY = "reliability"
    SCALABILITY = "scalability"
    COMPATIBILITY = "compatibility"


@dataclass
class TestFrameworkConfig:
    """Configuration for testing framework."""
    # Test organization
    test_root: str = "tests"
    test_levels: List[TestLevel] = field(default_factory=lambda: list(TestLevel))
    
    # Coverage targets
    unit_coverage_target: float = 0.90          # 90% code coverage
    integration_coverage_target: float = 0.85    # 85% integration coverage
    critical_path_coverage_target: float = 1.0   # 100% critical path coverage
    
    # Performance targets (from PRD)
    similarity_latency_target_ms: float = 100
    batch_reranking_latency_target_ms: float = 500
    memory_usage_target_gb: float = 2.0
    accuracy_improvement_target: float = 0.10
    
    # Execution configuration
    parallel_execution: bool = True
    max_workers: int = 4
    test_timeout_seconds: int = 300
    fail_fast: bool = False
    
    # Reporting
    generate_html_report: bool = True
    generate_json_report: bool = True
    generate_coverage_report: bool = True
    report_directory: str = "test_reports"
    
    # Environment
    isolated_test_environment: bool = True
    cleanup_after_tests: bool = True
    preserve_test_artifacts: bool = False


@dataclass
class TestCase:
    """Individual test case metadata."""
    name: str
    level: TestLevel
    category: TestCategory
    description: str
    test_function: Callable
    tags: List[str] = field(default_factory=list)
    timeout_seconds: Optional[int] = None
    skip_reason: Optional[str] = None
    expected_performance: Optional[Dict[str, float]] = None


@dataclass
class TestResult:
    """Result of a single test execution."""
    test_case: TestCase
    passed: bool
    execution_time_ms: float
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
    artifacts: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)


class BaseTestFramework(ABC):
    """
    Base class for all testing frameworks.
    
    Provides common functionality for test discovery, execution, and reporting
    across different testing levels.
    """
    
    def __init__(self, config: Optional[TestFrameworkConfig] = None):
        self.config = config or TestFrameworkConfig()
        self.test_cases: List[TestCase] = []
        self.test_results: List[TestResult] = []
        self.logger = logger
        
        # Test discovery
        self._discovered_tests: Dict[TestLevel, List[TestCase]] = {
            level: [] for level in TestLevel
        }
        
        # Performance baselines
        self.performance_baselines: Dict[str, float] = {}
        
        # Initialize test environment
        self._initialize_test_environment()
    
    def _initialize_test_environment(self):
        """Initialize testing environment."""
        # Create report directory
        os.makedirs(self.config.report_directory, exist_ok=True)
        
        # Load performance baselines if available
        baseline_file = os.path.join(self.config.report_directory, "performance_baselines.json")
        if os.path.exists(baseline_file):
            with open(baseline_file, 'r') as f:
                self.performance_baselines = json.load(f)
        
        self.logger.info("Initialized test environment")
    
    @abstractmethod
    def discover_tests(self, test_pattern: str = "test_*.py") -> List[TestCase]:
        """
        Discover tests based on pattern.
        
        Args:
            test_pattern: Pattern for test file discovery
            
        Returns:
            List of discovered test cases
        """
        pass
    
    @abstractmethod
    def setup_test(self, test_case: TestCase) -> None:
        """
        Setup before running a test.
        
        Args:
            test_case: Test case to setup
        """
        pass
    
    @abstractmethod
    def teardown_test(self, test_case: TestCase) -> None:
        """
        Teardown after running a test.
        
        Args:
            test_case: Test case to teardown
        """
        pass
    
    def run_test(self, test_case: TestCase) -> TestResult:
        """
        Run a single test case.
        
        Args:
            test_case: Test case to execute
            
        Returns:
            Test execution result
        """
        start_time = time.time()
        
        # Skip if marked
        if test_case.skip_reason:
            return TestResult(
                test_case=test_case,
                passed=True,
                execution_time_ms=0,
                error_message=f"Skipped: {test_case.skip_reason}"
            )
        
        try:
            # Setup
            self.setup_test(test_case)
            
            # Execute test with timeout if specified
            if test_case.timeout_seconds:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(test_case.test_function)
                    try:
                        result = future.result(timeout=test_case.timeout_seconds)
                    except concurrent.futures.TimeoutError:
                        raise TimeoutError(f"Test exceeded timeout of {test_case.timeout_seconds}s")
            else:
                result = test_case.test_function()
            
            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Extract performance metrics if available
            performance_metrics = None
            if hasattr(result, 'performance_metrics'):
                performance_metrics = result.performance_metrics
            
            return TestResult(
                test_case=test_case,
                passed=True,
                execution_time_ms=execution_time_ms,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            
            return TestResult(
                test_case=test_case,
                passed=False,
                execution_time_ms=execution_time_ms,
                error_message=str(e),
                stack_trace=self._get_stack_trace(e)
            )
            
        finally:
            # Always teardown
            try:
                self.teardown_test(test_case)
            except Exception as e:
                self.logger.error(f"Teardown failed for {test_case.name}: {e}")
    
    def run_tests(self, test_cases: Optional[List[TestCase]] = None,
                  parallel: Optional[bool] = None) -> List[TestResult]:
        """
        Run multiple test cases.
        
        Args:
            test_cases: List of test cases to run (or use discovered tests)
            parallel: Override parallel execution setting
            
        Returns:
            List of test results
        """
        if test_cases is None:
            test_cases = self.test_cases
        
        if not test_cases:
            self.logger.warning("No test cases to run")
            return []
        
        parallel = parallel if parallel is not None else self.config.parallel_execution
        results = []
        
        self.logger.info(f"Running {len(test_cases)} tests {'in parallel' if parallel else 'sequentially'}")
        
        if parallel and len(test_cases) > 1:
            # Parallel execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_test = {
                    executor.submit(self.run_test, test_case): test_case
                    for test_case in test_cases
                }
                
                for future in concurrent.futures.as_completed(future_to_test):
                    result = future.result()
                    results.append(result)
                    
                    # Fail fast if configured
                    if self.config.fail_fast and not result.passed:
                        self.logger.warning("Failing fast due to test failure")
                        executor.shutdown(wait=False)
                        break
        else:
            # Sequential execution
            for test_case in test_cases:
                result = self.run_test(test_case)
                results.append(result)
                
                # Fail fast if configured
                if self.config.fail_fast and not result.passed:
                    self.logger.warning("Failing fast due to test failure")
                    break
        
        self.test_results = results
        return results
    
    def filter_tests(self, level: Optional[TestLevel] = None,
                    category: Optional[TestCategory] = None,
                    tags: Optional[List[str]] = None) -> List[TestCase]:
        """
        Filter test cases based on criteria.
        
        Args:
            level: Filter by test level
            category: Filter by test category
            tags: Filter by tags (any match)
            
        Returns:
            Filtered list of test cases
        """
        filtered = self.test_cases
        
        if level:
            filtered = [tc for tc in filtered if tc.level == level]
        
        if category:
            filtered = [tc for tc in filtered if tc.category == category]
        
        if tags:
            filtered = [tc for tc in filtered if any(tag in tc.tags for tag in tags)]
        
        return filtered
    
    def validate_performance(self, test_result: TestResult) -> bool:
        """
        Validate test performance against baselines and targets.
        
        Args:
            test_result: Test result with performance metrics
            
        Returns:
            True if performance is acceptable
        """
        if not test_result.performance_metrics:
            return True
        
        # Check against expected performance
        if test_result.test_case.expected_performance:
            for metric, expected in test_result.test_case.expected_performance.items():
                if metric in test_result.performance_metrics:
                    actual = test_result.performance_metrics[metric]
                    if actual > expected * 1.1:  # 10% tolerance
                        self.logger.warning(
                            f"Performance degradation in {test_result.test_case.name}: "
                            f"{metric}={actual:.2f} (expected {expected:.2f})"
                        )
                        return False
        
        # Check against baselines
        test_key = f"{test_result.test_case.level.value}.{test_result.test_case.name}"
        if test_key in self.performance_baselines:
            baseline = self.performance_baselines[test_key]
            if test_result.execution_time_ms > baseline * 1.2:  # 20% regression threshold
                self.logger.warning(
                    f"Performance regression in {test_result.test_case.name}: "
                    f"{test_result.execution_time_ms:.2f}ms (baseline {baseline:.2f}ms)"
                )
                return False
        
        return True
    
    def update_performance_baselines(self) -> None:
        """Update performance baselines with current results."""
        for result in self.test_results:
            if result.passed:
                test_key = f"{result.test_case.level.value}.{result.test_case.name}"
                self.performance_baselines[test_key] = result.execution_time_ms
        
        # Save baselines
        baseline_file = os.path.join(self.config.report_directory, "performance_baselines.json")
        with open(baseline_file, 'w') as f:
            json.dump(self.performance_baselines, f, indent=2)
        
        self.logger.info("Updated performance baselines")
    
    def get_test_summary(self) -> Dict[str, Any]:
        """
        Get summary of test execution.
        
        Returns:
            Test execution summary
        """
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.passed)
        failed_tests = total_tests - passed_tests
        
        # Group by level
        level_summary = {}
        for level in TestLevel:
            level_results = [r for r in self.test_results if r.test_case.level == level]
            if level_results:
                level_summary[level.value] = {
                    "total": len(level_results),
                    "passed": sum(1 for r in level_results if r.passed),
                    "failed": sum(1 for r in level_results if not r.passed),
                    "avg_time_ms": sum(r.execution_time_ms for r in level_results) / len(level_results)
                }
        
        # Performance summary
        performance_issues = [
            r for r in self.test_results 
            if r.passed and not self.validate_performance(r)
        ]
        
        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "total_time_ms": sum(r.execution_time_ms for r in self.test_results),
            "level_summary": level_summary,
            "performance_issues": len(performance_issues),
            "timestamp": time.time()
        }
    
    def _get_stack_trace(self, exception: Exception) -> str:
        """Get formatted stack trace from exception."""
        import traceback
        return traceback.format_exc()


class TestOrganizer:
    """
    Organizes tests across multiple frameworks and levels.
    """
    
    def __init__(self, config: TestFrameworkConfig):
        self.config = config
        self.frameworks: Dict[TestLevel, BaseTestFramework] = {}
        self.logger = logger
        
        # Create test directory structure
        self._create_test_structure()
    
    def _create_test_structure(self):
        """Create organized test directory structure."""
        base_path = self.config.test_root
        
        # Create directories for each test level
        for level in TestLevel:
            level_path = os.path.join(base_path, level.value)
            os.makedirs(level_path, exist_ok=True)
            
            # Create __init__.py
            init_file = os.path.join(level_path, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write(f'"""Tests for {level.value} level."""\n')
        
        # Create fixtures directory
        fixtures_path = os.path.join(base_path, "fixtures")
        os.makedirs(fixtures_path, exist_ok=True)
        
        self.logger.info("Created test directory structure")
    
    def register_framework(self, level: TestLevel, framework: BaseTestFramework) -> None:
        """
        Register a test framework for a specific level.
        
        Args:
            level: Test level
            framework: Test framework instance
        """
        self.frameworks[level] = framework
        self.logger.info(f"Registered {level.value} test framework")
    
    def get_framework(self, level: TestLevel) -> Optional[BaseTestFramework]:
        """
        Get test framework for a specific level.
        
        Args:
            level: Test level
            
        Returns:
            Test framework instance or None
        """
        return self.frameworks.get(level)
    
    def discover_all_tests(self) -> Dict[TestLevel, List[TestCase]]:
        """
        Discover tests across all registered frameworks.
        
        Returns:
            Dictionary mapping test levels to discovered test cases
        """
        all_tests = {}
        
        for level, framework in self.frameworks.items():
            tests = framework.discover_tests()
            all_tests[level] = tests
            self.logger.info(f"Discovered {len(tests)} tests at {level.value} level")
        
        return all_tests
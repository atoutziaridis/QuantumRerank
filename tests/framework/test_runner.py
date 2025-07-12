"""
Test runner and executor for comprehensive test suite execution.

This module provides test execution capabilities with support for parallel execution,
timeout handling, and integration with various testing levels.
"""

import os
import sys
import time
import importlib
import inspect
from typing import Dict, List, Optional, Any, Callable, Type, Union, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
import threading
import queue

from .test_architecture import (
    TestCase, TestResult, TestLevel, TestCategory,
    BaseTestFramework, TestFrameworkConfig
)
from quantum_rerank.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TestExecutionContext:
    """Context for test execution with isolation and resource management."""
    test_case: TestCase
    environment_vars: Dict[str, str] = field(default_factory=dict)
    working_directory: Optional[str] = None
    resource_limits: Optional[Dict[str, Any]] = None
    setup_functions: List[Callable] = field(default_factory=list)
    teardown_functions: List[Callable] = field(default_factory=list)


class TestExecutor:
    """
    Executes individual tests with proper isolation and resource management.
    """
    
    def __init__(self, config: TestFrameworkConfig):
        self.config = config
        self.logger = logger
        self._execution_lock = threading.Lock()
        self._resource_manager = TestResourceManager()
    
    def execute_test(self, context: TestExecutionContext) -> TestResult:
        """
        Execute a test with proper context and isolation.
        
        Args:
            context: Test execution context
            
        Returns:
            Test execution result
        """
        start_time = time.time()
        test_case = context.test_case
        
        # Skip if marked
        if test_case.skip_reason:
            return TestResult(
                test_case=test_case,
                passed=True,
                execution_time_ms=0,
                error_message=f"Skipped: {test_case.skip_reason}"
            )
        
        # Setup execution environment
        original_env = os.environ.copy()
        original_cwd = os.getcwd()
        
        try:
            # Apply environment variables
            os.environ.update(context.environment_vars)
            
            # Change working directory if specified
            if context.working_directory:
                os.chdir(context.working_directory)
            
            # Apply resource limits
            if context.resource_limits:
                self._resource_manager.apply_limits(context.resource_limits)
            
            # Run setup functions
            for setup_fn in context.setup_functions:
                setup_fn()
            
            # Execute test
            result = self._run_test_function(test_case)
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            return TestResult(
                test_case=test_case,
                passed=result.get('passed', True),
                execution_time_ms=execution_time_ms,
                performance_metrics=result.get('performance_metrics'),
                artifacts=result.get('artifacts')
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
            # Run teardown functions
            for teardown_fn in reversed(context.teardown_functions):
                try:
                    teardown_fn()
                except Exception as e:
                    self.logger.error(f"Teardown failed: {e}")
            
            # Restore environment
            os.environ.clear()
            os.environ.update(original_env)
            os.chdir(original_cwd)
            
            # Release resource limits
            if context.resource_limits:
                self._resource_manager.release_limits()
    
    def _run_test_function(self, test_case: TestCase) -> Dict[str, Any]:
        """Run the actual test function with timeout handling."""
        if test_case.timeout_seconds:
            # Use multiprocessing for timeout with true isolation
            with multiprocessing.Pool(processes=1) as pool:
                async_result = pool.apply_async(test_case.test_function)
                try:
                    result = async_result.get(timeout=test_case.timeout_seconds)
                    return self._normalize_test_result(result)
                except multiprocessing.TimeoutError:
                    pool.terminate()
                    raise TimeoutError(f"Test exceeded timeout of {test_case.timeout_seconds}s")
        else:
            # Direct execution
            result = test_case.test_function()
            return self._normalize_test_result(result)
    
    def _normalize_test_result(self, result: Any) -> Dict[str, Any]:
        """Normalize test function result to standard format."""
        if isinstance(result, dict):
            return result
        elif isinstance(result, bool):
            return {"passed": result}
        elif result is None:
            return {"passed": True}
        else:
            return {"passed": True, "result": result}
    
    def _get_stack_trace(self, exception: Exception) -> str:
        """Get formatted stack trace from exception."""
        import traceback
        return traceback.format_exc()


class TestRunner:
    """
    High-level test runner coordinating test execution across frameworks.
    """
    
    def __init__(self, config: TestFrameworkConfig):
        self.config = config
        self.executor = TestExecutor(config)
        self.frameworks: Dict[TestLevel, BaseTestFramework] = {}
        self.logger = logger
        
        # Test discovery cache
        self._discovered_tests: Dict[TestLevel, List[TestCase]] = {}
        self._test_modules: Dict[str, Any] = {}
    
    def register_framework(self, level: TestLevel, framework: BaseTestFramework) -> None:
        """Register a test framework for a specific level."""
        self.frameworks[level] = framework
        self.logger.info(f"Registered {level.value} test framework")
    
    def discover_tests(self, levels: Optional[List[TestLevel]] = None,
                      pattern: str = "test_*.py") -> Dict[TestLevel, List[TestCase]]:
        """
        Discover tests across specified levels.
        
        Args:
            levels: Test levels to discover (None for all)
            pattern: Test file pattern
            
        Returns:
            Dictionary mapping levels to discovered tests
        """
        if levels is None:
            levels = list(self.frameworks.keys())
        
        discovered = {}
        
        for level in levels:
            if level in self.frameworks:
                framework = self.frameworks[level]
                tests = framework.discover_tests(pattern)
                discovered[level] = tests
                self._discovered_tests[level] = tests
                self.logger.info(f"Discovered {len(tests)} tests at {level.value} level")
            else:
                self.logger.warning(f"No framework registered for {level.value} level")
        
        return discovered
    
    def run_tests(self, test_cases: Optional[List[TestCase]] = None,
                  levels: Optional[List[TestLevel]] = None,
                  categories: Optional[List[TestCategory]] = None,
                  tags: Optional[List[str]] = None,
                  parallel: Optional[bool] = None) -> Dict[str, Any]:
        """
        Run tests with various filtering options.
        
        Args:
            test_cases: Specific test cases to run
            levels: Filter by test levels
            categories: Filter by test categories
            tags: Filter by tags
            parallel: Override parallel execution
            
        Returns:
            Test execution results and summary
        """
        # Determine tests to run
        if test_cases is None:
            test_cases = self._filter_tests(levels, categories, tags)
        
        if not test_cases:
            self.logger.warning("No tests to run after filtering")
            return {"results": [], "summary": {}}
        
        self.logger.info(f"Running {len(test_cases)} tests")
        
        # Group tests by level for proper framework execution
        tests_by_level = self._group_tests_by_level(test_cases)
        
        # Execute tests
        all_results = []
        
        for level, level_tests in tests_by_level.items():
            if level in self.frameworks:
                framework = self.frameworks[level]
                results = framework.run_tests(level_tests, parallel=parallel)
                all_results.extend(results)
            else:
                # Use default executor
                for test_case in level_tests:
                    context = TestExecutionContext(test_case=test_case)
                    result = self.executor.execute_test(context)
                    all_results.append(result)
        
        # Generate summary
        summary = self._generate_summary(all_results)
        
        return {
            "results": all_results,
            "summary": summary
        }
    
    def run_continuous(self, interval_seconds: int = 300,
                      levels: Optional[List[TestLevel]] = None) -> None:
        """
        Run tests continuously at specified interval.
        
        Args:
            interval_seconds: Interval between test runs
            levels: Test levels to run
        """
        self.logger.info(f"Starting continuous test execution with {interval_seconds}s interval")
        
        try:
            while True:
                self.logger.info("Starting test run")
                
                # Discover and run tests
                self.discover_tests(levels)
                results = self.run_tests(levels=levels)
                
                # Log summary
                summary = results["summary"]
                self.logger.info(
                    f"Test run complete: {summary['passed']}/{summary['total']} passed "
                    f"({summary['pass_rate']*100:.1f}%)"
                )
                
                # Wait for next run
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            self.logger.info("Continuous test execution stopped")
    
    def _filter_tests(self, levels: Optional[List[TestLevel]] = None,
                     categories: Optional[List[TestCategory]] = None,
                     tags: Optional[List[str]] = None) -> List[TestCase]:
        """Filter discovered tests based on criteria."""
        # Start with all discovered tests
        all_tests = []
        for level_tests in self._discovered_tests.values():
            all_tests.extend(level_tests)
        
        # Apply filters
        filtered = all_tests
        
        if levels:
            filtered = [tc for tc in filtered if tc.level in levels]
        
        if categories:
            filtered = [tc for tc in filtered if tc.category in categories]
        
        if tags:
            filtered = [tc for tc in filtered if any(tag in tc.tags for tag in tags)]
        
        return filtered
    
    def _group_tests_by_level(self, test_cases: List[TestCase]) -> Dict[TestLevel, List[TestCase]]:
        """Group test cases by their level."""
        grouped = {}
        
        for test_case in test_cases:
            level = test_case.level
            if level not in grouped:
                grouped[level] = []
            grouped[level].append(test_case)
        
        return grouped
    
    def _generate_summary(self, results: List[TestResult]) -> Dict[str, Any]:
        """Generate summary from test results."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        
        # Calculate statistics
        execution_times = [r.execution_time_ms for r in results]
        
        # Group by level
        level_stats = {}
        for level in TestLevel:
            level_results = [r for r in results if r.test_case.level == level]
            if level_results:
                level_passed = sum(1 for r in level_results if r.passed)
                level_stats[level.value] = {
                    "total": len(level_results),
                    "passed": level_passed,
                    "failed": len(level_results) - level_passed,
                    "pass_rate": level_passed / len(level_results)
                }
        
        # Failed test details
        failed_tests = [
            {
                "name": r.test_case.name,
                "level": r.test_case.level.value,
                "error": r.error_message
            }
            for r in results if not r.passed
        ]
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0,
            "total_time_ms": sum(execution_times),
            "avg_time_ms": sum(execution_times) / len(execution_times) if execution_times else 0,
            "level_stats": level_stats,
            "failed_tests": failed_tests,
            "timestamp": time.time()
        }


class TestSuiteRunner:
    """
    Coordinates execution of comprehensive test suites across all levels.
    """
    
    def __init__(self, config: TestFrameworkConfig):
        self.config = config
        self.runner = TestRunner(config)
        self.logger = logger
        
        # Test suite definitions
        self.test_suites = {
            "smoke": {
                "levels": [TestLevel.UNIT],
                "categories": [TestCategory.FUNCTIONAL],
                "tags": ["smoke", "quick"]
            },
            "regression": {
                "levels": [TestLevel.UNIT, TestLevel.INTEGRATION],
                "categories": [TestCategory.FUNCTIONAL, TestCategory.PERFORMANCE]
            },
            "full": {
                "levels": list(TestLevel),
                "categories": list(TestCategory)
            },
            "performance": {
                "levels": [TestLevel.PERFORMANCE],
                "categories": [TestCategory.PERFORMANCE, TestCategory.SCALABILITY]
            },
            "quantum": {
                "levels": [TestLevel.QUANTUM],
                "categories": [TestCategory.FUNCTIONAL, TestCategory.PERFORMANCE]
            }
        }
    
    def run_suite(self, suite_name: str) -> Dict[str, Any]:
        """
        Run a predefined test suite.
        
        Args:
            suite_name: Name of test suite to run
            
        Returns:
            Test suite execution results
        """
        if suite_name not in self.test_suites:
            raise ValueError(f"Unknown test suite: {suite_name}")
        
        suite_config = self.test_suites[suite_name]
        
        self.logger.info(f"Running '{suite_name}' test suite")
        
        # Discover tests for suite
        self.runner.discover_tests(levels=suite_config.get("levels"))
        
        # Run tests with suite filters
        results = self.runner.run_tests(
            levels=suite_config.get("levels"),
            categories=suite_config.get("categories"),
            tags=suite_config.get("tags")
        )
        
        return results
    
    def run_production_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive production validation suite.
        
        Returns:
            Production readiness validation results
        """
        self.logger.info("Running production validation suite")
        
        validation_results = {
            "functionality": self._validate_functionality(),
            "performance": self._validate_performance(),
            "reliability": self._validate_reliability(),
            "scalability": self._validate_scalability(),
            "security": self._validate_security()
        }
        
        # Overall readiness
        all_passed = all(r["passed"] for r in validation_results.values())
        
        return {
            "validation_results": validation_results,
            "production_ready": all_passed,
            "timestamp": time.time()
        }
    
    def _validate_functionality(self) -> Dict[str, Any]:
        """Validate core functionality."""
        results = self.runner.run_tests(
            levels=[TestLevel.UNIT, TestLevel.INTEGRATION],
            categories=[TestCategory.FUNCTIONAL]
        )
        
        summary = results["summary"]
        passed = summary["pass_rate"] >= 0.95  # 95% pass rate required
        
        return {
            "passed": passed,
            "pass_rate": summary["pass_rate"],
            "details": summary
        }
    
    def _validate_performance(self) -> Dict[str, Any]:
        """Validate performance targets."""
        results = self.runner.run_tests(
            levels=[TestLevel.PERFORMANCE],
            categories=[TestCategory.PERFORMANCE]
        )
        
        # Check PRD targets
        prd_compliance = self._check_prd_compliance(results["results"])
        
        return {
            "passed": prd_compliance["all_targets_met"],
            "prd_compliance": prd_compliance,
            "details": results["summary"]
        }
    
    def _validate_reliability(self) -> Dict[str, Any]:
        """Validate system reliability."""
        results = self.runner.run_tests(
            levels=[TestLevel.CHAOS],
            categories=[TestCategory.RELIABILITY]
        )
        
        summary = results["summary"]
        passed = summary["pass_rate"] >= 0.90  # 90% reliability required
        
        return {
            "passed": passed,
            "pass_rate": summary["pass_rate"],
            "details": summary
        }
    
    def _validate_scalability(self) -> Dict[str, Any]:
        """Validate scalability requirements."""
        results = self.runner.run_tests(
            levels=[TestLevel.PERFORMANCE],
            categories=[TestCategory.SCALABILITY]
        )
        
        summary = results["summary"]
        passed = summary["pass_rate"] >= 0.85  # 85% scalability tests pass
        
        return {
            "passed": passed,
            "pass_rate": summary["pass_rate"],
            "details": summary
        }
    
    def _validate_security(self) -> Dict[str, Any]:
        """Validate security requirements."""
        results = self.runner.run_tests(
            categories=[TestCategory.SECURITY]
        )
        
        summary = results["summary"]
        passed = summary["failed"] == 0  # No security test failures allowed
        
        return {
            "passed": passed,
            "pass_rate": summary["pass_rate"],
            "details": summary
        }
    
    def _check_prd_compliance(self, results: List[TestResult]) -> Dict[str, Any]:
        """Check compliance with PRD performance targets."""
        targets = {
            "similarity_latency_ms": self.config.similarity_latency_target_ms,
            "batch_reranking_latency_ms": self.config.batch_reranking_latency_target_ms,
            "memory_usage_gb": self.config.memory_usage_target_gb,
            "accuracy_improvement": self.config.accuracy_improvement_target
        }
        
        compliance = {}
        
        for result in results:
            if result.performance_metrics:
                for metric, target in targets.items():
                    if metric in result.performance_metrics:
                        actual = result.performance_metrics[metric]
                        meets_target = actual <= target if "latency" in metric or "memory" in metric else actual >= target
                        
                        compliance[metric] = {
                            "target": target,
                            "actual": actual,
                            "meets_target": meets_target
                        }
        
        all_targets_met = all(c["meets_target"] for c in compliance.values())
        
        return {
            "targets": targets,
            "compliance": compliance,
            "all_targets_met": all_targets_met
        }


class TestResourceManager:
    """Manages resource limits and isolation for test execution."""
    
    def __init__(self):
        self.original_limits = {}
    
    def apply_limits(self, limits: Dict[str, Any]) -> None:
        """Apply resource limits for test execution."""
        # This would implement actual resource limiting
        # For now, it's a placeholder
        pass
    
    def release_limits(self) -> None:
        """Release resource limits after test execution."""
        # This would release resource limits
        # For now, it's a placeholder
        pass
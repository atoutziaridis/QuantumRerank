#!/usr/bin/env python3
"""
Comprehensive test runner for QuantumRerank testing framework.

This script demonstrates the complete testing framework by running tests
across all levels: unit, integration, system, performance, quantum, and chaos.
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.framework import (
    TestRunner, TestSuiteRunner, TestReporter, TestLevel, TestCategory,
    TestFrameworkConfig
)
from tests.quantum import QuantumTestFramework
from tests.performance import PRDComplianceFramework
from tests.chaos import ChaosTestFramework
from quantum_rerank.utils.logging_config import get_logger

logger = get_logger(__name__)


def create_test_config() -> TestFrameworkConfig:
    """Create comprehensive test configuration."""
    return TestFrameworkConfig(
        # Coverage targets
        unit_test_coverage_target=0.90,
        integration_test_coverage_target=0.85,
        system_test_coverage_target=0.80,
        
        # Performance targets (PRD compliance)
        similarity_latency_target_ms=100,
        batch_reranking_latency_target_ms=500,
        memory_usage_target_gb=2.0,
        accuracy_improvement_target=0.15,
        
        # Execution settings
        parallel_execution=True,
        max_parallel_workers=4,
        default_timeout_seconds=300,
        
        # Reporting
        generate_html_reports=True,
        generate_json_reports=True,
        generate_junit_reports=True,
        
        # Output directory
        output_directory="test_reports"
    )


def run_unit_tests(test_runner: TestRunner) -> dict:
    """Run unit tests."""
    logger.info("=" * 60)
    logger.info("RUNNING UNIT TESTS")
    logger.info("=" * 60)
    
    # Discover and run unit tests
    discovered = test_runner.discover_tests(levels=[TestLevel.UNIT])
    results = test_runner.run_tests(levels=[TestLevel.UNIT])
    
    logger.info(f"Unit tests completed: {results['summary']['passed']}/{results['summary']['total']} passed")
    return results


def run_integration_tests(test_runner: TestRunner) -> dict:
    """Run integration tests."""
    logger.info("=" * 60)
    logger.info("RUNNING INTEGRATION TESTS")
    logger.info("=" * 60)
    
    # Discover and run integration tests
    discovered = test_runner.discover_tests(levels=[TestLevel.INTEGRATION])
    results = test_runner.run_tests(levels=[TestLevel.INTEGRATION])
    
    logger.info(f"Integration tests completed: {results['summary']['passed']}/{results['summary']['total']} passed")
    return results


def run_quantum_tests() -> dict:
    """Run quantum-specific tests."""
    logger.info("=" * 60)
    logger.info("RUNNING QUANTUM TESTS")
    logger.info("=" * 60)
    
    # Initialize quantum test framework
    quantum_framework = QuantumTestFramework()
    
    # Discover and run quantum tests
    test_cases = quantum_framework.discover_tests()
    results = quantum_framework.run_tests(test_cases)
    
    logger.info(f"Quantum tests completed: {len([r for r in results if r.passed])}/{len(results)} passed")
    
    # Convert to standard format
    return {
        "results": results,
        "summary": {
            "total": len(results),
            "passed": len([r for r in results if r.passed]),
            "failed": len([r for r in results if not r.passed]),
            "pass_rate": len([r for r in results if r.passed]) / len(results) if results else 0
        }
    }


def run_performance_tests() -> dict:
    """Run performance and PRD compliance tests."""
    logger.info("=" * 60)
    logger.info("RUNNING PERFORMANCE TESTS (PRD COMPLIANCE)")
    logger.info("=" * 60)
    
    # Initialize PRD compliance framework
    prd_framework = PRDComplianceFramework()
    
    # Discover and run PRD compliance tests
    test_cases = prd_framework.discover_tests()
    results = prd_framework.run_tests(test_cases)
    
    logger.info(f"Performance tests completed: {len([r for r in results if r.passed])}/{len(results)} passed")
    
    # Convert to standard format
    return {
        "results": results,
        "summary": {
            "total": len(results),
            "passed": len([r for r in results if r.passed]),
            "failed": len([r for r in results if not r.passed]),
            "pass_rate": len([r for r in results if r.passed]) / len(results) if results else 0
        }
    }


def run_chaos_tests() -> dict:
    """Run chaos engineering tests."""
    logger.info("=" * 60)
    logger.info("RUNNING CHAOS ENGINEERING TESTS")
    logger.info("=" * 60)
    
    # Initialize chaos test framework
    chaos_framework = ChaosTestFramework()
    
    # Discover and run chaos tests
    test_cases = chaos_framework.discover_tests()
    results = chaos_framework.run_tests(test_cases)
    
    logger.info(f"Chaos tests completed: {len([r for r in results if r.passed])}/{len(results)} passed")
    
    # Convert to standard format
    return {
        "results": results,
        "summary": {
            "total": len(results),
            "passed": len([r for r in results if r.passed]),
            "failed": len([r for r in results if not r.passed]),
            "pass_rate": len([r for r in results if r.passed]) / len(results) if results else 0
        }
    }


def run_production_validation(test_suite_runner: TestSuiteRunner) -> dict:
    """Run production readiness validation."""
    logger.info("=" * 60)
    logger.info("RUNNING PRODUCTION READINESS VALIDATION")
    logger.info("=" * 60)
    
    # Run comprehensive production validation
    validation_results = test_suite_runner.run_production_validation()
    
    production_ready = validation_results.get("production_ready", False)
    logger.info(f"Production readiness: {'✅ READY' if production_ready else '❌ NOT READY'}")
    
    return validation_results


def generate_comprehensive_reports(all_results: dict, config: TestFrameworkConfig) -> dict:
    """Generate comprehensive test reports."""
    logger.info("=" * 60)
    logger.info("GENERATING COMPREHENSIVE REPORTS")
    logger.info("=" * 60)
    
    # Initialize reporter
    reporter = TestReporter(output_directory=config.output_directory)
    
    # Combine all test results
    combined_results = []
    for test_type, results in all_results.items():
        if "results" in results:
            combined_results.extend(results["results"])
    
    # Generate reports in all formats
    report_files = reporter.generate_comprehensive_report(
        results=combined_results,
        coverage_data={"overall_coverage": 0.85},  # Mock coverage data
        performance_data={"performance_score": 0.88}  # Mock performance data
    )
    
    logger.info("Generated reports:")
    for format_name, file_path in report_files.items():
        logger.info(f"  {format_name.upper()}: {file_path}")
    
    return report_files


def print_final_summary(all_results: dict, validation_results: dict):
    """Print final test execution summary."""
    logger.info("=" * 60)
    logger.info("FINAL TEST EXECUTION SUMMARY")
    logger.info("=" * 60)
    
    total_tests = 0
    total_passed = 0
    
    for test_type, results in all_results.items():
        if "summary" in results:
            summary = results["summary"]
            total_tests += summary["total"]
            total_passed += summary["passed"]
            
            logger.info(f"{test_type.upper()}:")
            logger.info(f"  Tests: {summary['passed']}/{summary['total']} passed ({summary['pass_rate']*100:.1f}%)")
    
    overall_pass_rate = total_passed / total_tests if total_tests > 0 else 0
    production_ready = validation_results.get("production_ready", False)
    
    logger.info("-" * 60)
    logger.info(f"OVERALL RESULTS:")
    logger.info(f"  Total Tests: {total_passed}/{total_tests} passed ({overall_pass_rate*100:.1f}%)")
    logger.info(f"  Production Ready: {'✅ YES' if production_ready else '❌ NO'}")
    
    if production_ready:
        logger.info("🎉 System is ready for production deployment!")
    else:
        logger.info("⚠️  System requires fixes before production deployment.")
        
        # Show blocking issues
        blocking_issues = validation_results.get("validation_results", {})
        for component, result in blocking_issues.items():
            if not result.get("passed", True):
                logger.info(f"  ❌ {component}: {result.get('details', {}).get('pass_rate', 0)*100:.1f}% pass rate")


def main():
    """Main test execution function."""
    parser = argparse.ArgumentParser(description="Run comprehensive QuantumRerank tests")
    parser.add_argument("--level", choices=["unit", "integration", "quantum", "performance", "chaos", "all"], 
                       default="all", help="Test level to run")
    parser.add_argument("--output-dir", default="test_reports", help="Output directory for reports")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel test execution")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create test configuration
    config = create_test_config()
    config.output_directory = args.output_dir
    config.parallel_execution = args.parallel
    
    # Create output directory
    os.makedirs(config.output_directory, exist_ok=True)
    
    logger.info("🚀 Starting QuantumRerank Comprehensive Test Suite")
    logger.info(f"Output directory: {config.output_directory}")
    logger.info(f"Parallel execution: {config.parallel_execution}")
    
    start_time = time.time()
    
    # Initialize test frameworks
    test_runner = TestRunner(config)
    test_suite_runner = TestSuiteRunner(config)
    
    all_results = {}
    
    try:
        # Run tests based on selected level
        if args.level in ["unit", "all"]:
            all_results["unit"] = run_unit_tests(test_runner)
        
        if args.level in ["integration", "all"]:
            all_results["integration"] = run_integration_tests(test_runner)
        
        if args.level in ["quantum", "all"]:
            all_results["quantum"] = run_quantum_tests()
        
        if args.level in ["performance", "all"]:
            all_results["performance"] = run_performance_tests()
        
        if args.level in ["chaos", "all"]:
            all_results["chaos"] = run_chaos_tests()
        
        # Run production validation for comprehensive runs
        validation_results = {}
        if args.level == "all":
            validation_results = run_production_validation(test_suite_runner)
        
        # Generate comprehensive reports
        if all_results:
            report_files = generate_comprehensive_reports(all_results, config)
        
        # Print final summary
        if args.level == "all":
            print_final_summary(all_results, validation_results)
        
        execution_time = time.time() - start_time
        logger.info(f"✅ Test execution completed in {execution_time:.1f} seconds")
        
        # Exit with appropriate code
        overall_success = all(
            results.get("summary", {}).get("pass_rate", 0) > 0.8 
            for results in all_results.values()
        )
        
        if args.level == "all":
            overall_success = overall_success and validation_results.get("production_ready", False)
        
        sys.exit(0 if overall_success else 1)
        
    except KeyboardInterrupt:
        logger.info("❌ Test execution interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
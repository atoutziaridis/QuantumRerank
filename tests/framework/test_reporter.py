"""
Test reporting and metrics collection for comprehensive test analysis.

This module provides detailed test reporting with HTML, JSON, and coverage reports,
performance analysis, and production readiness assessment.
"""

import os
import json
import time
import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import html

from .test_architecture import TestResult, TestCase, TestLevel, TestCategory
from quantum_rerank.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TestReport:
    """Individual test result report."""
    test_name: str
    test_level: str
    test_category: str
    passed: bool
    execution_time_ms: float
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
    tags: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class TestSuiteReport:
    """Complete test suite execution report."""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_rate: float
    total_execution_time_ms: float
    reports: List[TestReport] = field(default_factory=list)
    level_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    performance_summary: Dict[str, Any] = field(default_factory=dict)
    coverage_summary: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProductionReadinessReport:
    """Production readiness assessment report."""
    overall_ready: bool
    functionality_score: float
    performance_score: float
    reliability_score: float
    security_score: float
    recommendations: List[str] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)
    prd_compliance: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class ReportFormat(Enum):
    """Supported report formats."""
    HTML = "html"
    JSON = "json"
    XML = "xml"
    JUNIT = "junit"
    MARKDOWN = "markdown"


class TestReporter:
    """
    Comprehensive test reporting with multiple output formats.
    """
    
    def __init__(self, output_directory: str = "test_reports"):
        self.output_directory = output_directory
        self.logger = logger
        
        # Create output directory
        os.makedirs(output_directory, exist_ok=True)
        
        # Report templates
        self.html_template = self._load_html_template()
        
        # Performance thresholds for scoring
        self.performance_thresholds = {
            "excellent": 0.95,
            "good": 0.85,
            "fair": 0.70,
            "poor": 0.50
        }
    
    def generate_test_report(self, results: List[TestResult],
                           format: ReportFormat = ReportFormat.HTML,
                           filename: Optional[str] = None) -> str:
        """
        Generate test report in specified format.
        
        Args:
            results: Test execution results
            format: Report format
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to generated report file
        """
        # Convert results to reports
        reports = [self._convert_result_to_report(result) for result in results]
        
        # Create suite report
        suite_report = self._create_suite_report("Test Execution", reports)
        
        # Generate report in specified format
        if format == ReportFormat.HTML:
            return self._generate_html_report(suite_report, filename)
        elif format == ReportFormat.JSON:
            return self._generate_json_report(suite_report, filename)
        elif format == ReportFormat.JUNIT:
            return self._generate_junit_report(suite_report, filename)
        elif format == ReportFormat.MARKDOWN:
            return self._generate_markdown_report(suite_report, filename)
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    def generate_comprehensive_report(self, results: List[TestResult],
                                    coverage_data: Optional[Dict[str, Any]] = None,
                                    performance_data: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Generate comprehensive report in multiple formats.
        
        Args:
            results: Test execution results
            coverage_data: Code coverage data
            performance_data: Performance metrics data
            
        Returns:
            Dictionary mapping format to file path
        """
        reports = [self._convert_result_to_report(result) for result in results]
        suite_report = self._create_suite_report("Comprehensive Test Report", reports)
        
        # Add coverage and performance data
        if coverage_data:
            suite_report.coverage_summary = coverage_data
        
        if performance_data:
            suite_report.performance_summary = performance_data
        
        # Generate all formats
        report_files = {}
        
        for format in ReportFormat:
            try:
                file_path = self.generate_test_report(results, format)
                report_files[format.value] = file_path
            except Exception as e:
                self.logger.error(f"Failed to generate {format.value} report: {e}")
        
        return report_files
    
    def generate_production_readiness_report(self, 
                                           validation_results: Dict[str, Any]) -> str:
        """
        Generate production readiness assessment report.
        
        Args:
            validation_results: Production validation results
            
        Returns:
            Path to generated readiness report
        """
        # Calculate scores
        functionality_score = validation_results.get("functionality", {}).get("pass_rate", 0.0)
        performance_score = validation_results.get("performance", {}).get("pass_rate", 0.0)
        reliability_score = validation_results.get("reliability", {}).get("pass_rate", 0.0)
        security_score = validation_results.get("security", {}).get("pass_rate", 0.0)
        
        # Overall readiness
        overall_score = (functionality_score + performance_score + reliability_score + security_score) / 4
        overall_ready = overall_score >= 0.85  # 85% threshold for production readiness
        
        # Generate recommendations
        recommendations = self._generate_readiness_recommendations(validation_results)
        critical_issues = self._identify_critical_issues(validation_results)
        
        # Create readiness report
        readiness_report = ProductionReadinessReport(
            overall_ready=overall_ready,
            functionality_score=functionality_score,
            performance_score=performance_score,
            reliability_score=reliability_score,
            security_score=security_score,
            recommendations=recommendations,
            critical_issues=critical_issues,
            prd_compliance=validation_results.get("performance", {}).get("prd_compliance", {})
        )
        
        # Generate HTML report
        return self._generate_readiness_html_report(readiness_report)
    
    def _convert_result_to_report(self, result: TestResult) -> TestReport:
        """Convert TestResult to TestReport."""
        return TestReport(
            test_name=result.test_case.name,
            test_level=result.test_case.level.value,
            test_category=result.test_case.category.value,
            passed=result.passed,
            execution_time_ms=result.execution_time_ms,
            error_message=result.error_message,
            performance_metrics=result.performance_metrics,
            tags=result.test_case.tags,
            timestamp=result.timestamp
        )
    
    def _create_suite_report(self, suite_name: str, reports: List[TestReport]) -> TestSuiteReport:
        """Create test suite report from individual reports."""
        total_tests = len(reports)
        passed_tests = sum(1 for r in reports if r.passed)
        failed_tests = total_tests - passed_tests
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        total_time = sum(r.execution_time_ms for r in reports)
        
        # Level summary
        level_summary = {}
        for level in TestLevel:
            level_reports = [r for r in reports if r.test_level == level.value]
            if level_reports:
                level_passed = sum(1 for r in level_reports if r.passed)
                level_summary[level.value] = {
                    "total": len(level_reports),
                    "passed": level_passed,
                    "failed": len(level_reports) - level_passed,
                    "pass_rate": level_passed / len(level_reports),
                    "avg_time_ms": sum(r.execution_time_ms for r in level_reports) / len(level_reports)
                }
        
        # Performance summary
        performance_summary = self._calculate_performance_summary(reports)
        
        return TestSuiteReport(
            suite_name=suite_name,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            pass_rate=pass_rate,
            total_execution_time_ms=total_time,
            reports=reports,
            level_summary=level_summary,
            performance_summary=performance_summary
        )
    
    def _calculate_performance_summary(self, reports: List[TestReport]) -> Dict[str, Any]:
        """Calculate performance summary from test reports."""
        performance_reports = [r for r in reports if r.performance_metrics]
        
        if not performance_reports:
            return {}
        
        # Aggregate metrics
        all_metrics = {}
        for report in performance_reports:
            for metric, value in report.performance_metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        # Calculate statistics
        summary = {}
        for metric, values in all_metrics.items():
            summary[metric] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "median": sorted(values)[len(values) // 2]
            }
        
        # Execution time statistics
        execution_times = [r.execution_time_ms for r in reports]
        summary["execution_time_ms"] = {
            "count": len(execution_times),
            "min": min(execution_times),
            "max": max(execution_times),
            "avg": sum(execution_times) / len(execution_times),
            "total": sum(execution_times)
        }
        
        return summary
    
    def _generate_html_report(self, suite_report: TestSuiteReport, filename: Optional[str] = None) -> str:
        """Generate HTML test report."""
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_report_{timestamp}.html"
        
        file_path = os.path.join(self.output_directory, filename)
        
        # Generate HTML content
        html_content = self._render_html_template(suite_report)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Generated HTML report: {file_path}")
        return file_path
    
    def _generate_json_report(self, suite_report: TestSuiteReport, filename: Optional[str] = None) -> str:
        """Generate JSON test report."""
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_report_{timestamp}.json"
        
        file_path = os.path.join(self.output_directory, filename)
        
        # Convert to dict and serialize
        report_dict = asdict(suite_report)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        self.logger.info(f"Generated JSON report: {file_path}")
        return file_path
    
    def _generate_junit_report(self, suite_report: TestSuiteReport, filename: Optional[str] = None) -> str:
        """Generate JUnit XML test report."""
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"junit_report_{timestamp}.xml"
        
        file_path = os.path.join(self.output_directory, filename)
        
        # Generate JUnit XML
        xml_content = self._render_junit_xml(suite_report)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        
        self.logger.info(f"Generated JUnit report: {file_path}")
        return file_path
    
    def _generate_markdown_report(self, suite_report: TestSuiteReport, filename: Optional[str] = None) -> str:
        """Generate Markdown test report."""
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_report_{timestamp}.md"
        
        file_path = os.path.join(self.output_directory, filename)
        
        # Generate Markdown content
        markdown_content = self._render_markdown_template(suite_report)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        self.logger.info(f"Generated Markdown report: {file_path}")
        return file_path
    
    def _generate_readiness_html_report(self, readiness_report: ProductionReadinessReport) -> str:
        """Generate production readiness HTML report."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"production_readiness_{timestamp}.html"
        file_path = os.path.join(self.output_directory, filename)
        
        # Generate HTML content
        html_content = self._render_readiness_html_template(readiness_report)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Generated production readiness report: {file_path}")
        return file_path
    
    def _render_html_template(self, suite_report: TestSuiteReport) -> str:
        """Render HTML template with test suite data."""
        # Get status color
        status_color = "success" if suite_report.pass_rate >= 0.90 else "warning" if suite_report.pass_rate >= 0.70 else "danger"
        
        # Format timestamp
        formatted_time = datetime.datetime.fromtimestamp(suite_report.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate test rows
        test_rows = ""
        for report in suite_report.reports:
            status_icon = "✓" if report.passed else "✗"
            row_class = "table-success" if report.passed else "table-danger"
            error_cell = html.escape(report.error_message or "") if not report.passed else ""
            
            test_rows += f"""
            <tr class="{row_class}">
                <td>{status_icon}</td>
                <td>{html.escape(report.test_name)}</td>
                <td><span class="badge bg-secondary">{report.test_level}</span></td>
                <td><span class="badge bg-info">{report.test_category}</span></td>
                <td>{report.execution_time_ms:.1f}ms</td>
                <td>{error_cell}</td>
            </tr>
            """
        
        # Generate level summary
        level_summary_rows = ""
        for level, data in suite_report.level_summary.items():
            level_summary_rows += f"""
            <tr>
                <td>{level}</td>
                <td>{data['total']}</td>
                <td>{data['passed']}</td>
                <td>{data['failed']}</td>
                <td>{data['pass_rate']*100:.1f}%</td>
                <td>{data['avg_time_ms']:.1f}ms</td>
            </tr>
            """
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>QuantumRerank Test Report</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                .metric-card {{ margin-bottom: 1rem; }}
                .status-{status_color} {{ color: {'green' if status_color == 'success' else 'orange' if status_color == 'warning' else 'red'}; }}
            </style>
        </head>
        <body>
            <div class="container mt-4">
                <h1>QuantumRerank Test Report</h1>
                <p class="text-muted">Generated on {formatted_time}</p>
                
                <div class="row mb-4">
                    <div class="col-md-3">
                        <div class="card metric-card">
                            <div class="card-body">
                                <h5 class="card-title">Total Tests</h5>
                                <h2 class="status-{status_color}">{suite_report.total_tests}</h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card metric-card">
                            <div class="card-body">
                                <h5 class="card-title">Pass Rate</h5>
                                <h2 class="status-{status_color}">{suite_report.pass_rate*100:.1f}%</h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card metric-card">
                            <div class="card-body">
                                <h5 class="card-title">Passed</h5>
                                <h2 class="text-success">{suite_report.passed_tests}</h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card metric-card">
                            <div class="card-body">
                                <h5 class="card-title">Failed</h5>
                                <h2 class="text-danger">{suite_report.failed_tests}</h2>
                            </div>
                        </div>
                    </div>
                </div>
                
                <h3>Test Results by Level</h3>
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>Level</th>
                            <th>Total</th>
                            <th>Passed</th>
                            <th>Failed</th>
                            <th>Pass Rate</th>
                            <th>Avg Time</th>
                        </tr>
                    </thead>
                    <tbody>
                        {level_summary_rows}
                    </tbody>
                </table>
                
                <h3>Individual Test Results</h3>
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>Status</th>
                            <th>Test Name</th>
                            <th>Level</th>
                            <th>Category</th>
                            <th>Time</th>
                            <th>Error</th>
                        </tr>
                    </thead>
                    <tbody>
                        {test_rows}
                    </tbody>
                </table>
            </div>
        </body>
        </html>
        """
    
    def _render_junit_xml(self, suite_report: TestSuiteReport) -> str:
        """Render JUnit XML format."""
        # Generate testcase elements
        testcases = ""
        for report in suite_report.reports:
            error_element = ""
            if not report.passed:
                error_element = f'<failure message="{html.escape(report.error_message or "Test failed")}" />'
            
            testcases += f"""
                <testcase name="{html.escape(report.test_name)}" 
                          classname="{report.test_level}.{report.test_category}" 
                          time="{report.execution_time_ms/1000:.3f}">
                    {error_element}
                </testcase>
            """
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
        <testsuite name="{html.escape(suite_report.suite_name)}" 
                   tests="{suite_report.total_tests}" 
                   failures="{suite_report.failed_tests}" 
                   time="{suite_report.total_execution_time_ms/1000:.3f}">
            {testcases}
        </testsuite>
        """
    
    def _render_markdown_template(self, suite_report: TestSuiteReport) -> str:
        """Render Markdown template."""
        formatted_time = datetime.datetime.fromtimestamp(suite_report.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate test results table
        test_table = "| Status | Test Name | Level | Category | Time | Error |\n"
        test_table += "|--------|-----------|-------|----------|------|-------|\n"
        
        for report in suite_report.reports:
            status = "✅" if report.passed else "❌"
            error = report.error_message or "" if not report.passed else ""
            test_table += f"| {status} | {report.test_name} | {report.test_level} | {report.test_category} | {report.execution_time_ms:.1f}ms | {error} |\n"
        
        return f"""# QuantumRerank Test Report
        
Generated on {formatted_time}

## Summary

- **Total Tests**: {suite_report.total_tests}
- **Passed**: {suite_report.passed_tests} ({suite_report.pass_rate*100:.1f}%)
- **Failed**: {suite_report.failed_tests}
- **Total Time**: {suite_report.total_execution_time_ms:.1f}ms

## Test Results

{test_table}

## Level Summary

| Level | Total | Passed | Failed | Pass Rate | Avg Time |
|-------|-------|--------|--------|-----------|----------|
"""
        
        for level, data in suite_report.level_summary.items():
            markdown_content += f"| {level} | {data['total']} | {data['passed']} | {data['failed']} | {data['pass_rate']*100:.1f}% | {data['avg_time_ms']:.1f}ms |\n"
        
        return markdown_content
    
    def _render_readiness_html_template(self, readiness_report: ProductionReadinessReport) -> str:
        """Render production readiness HTML template."""
        status_color = "success" if readiness_report.overall_ready else "danger"
        formatted_time = datetime.datetime.fromtimestamp(readiness_report.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate recommendations
        recommendations_html = ""
        for rec in readiness_report.recommendations:
            recommendations_html += f"<li>{html.escape(rec)}</li>\n"
        
        # Generate critical issues
        issues_html = ""
        for issue in readiness_report.critical_issues:
            issues_html += f"<li class='text-danger'>{html.escape(issue)}</li>\n"
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Production Readiness Report</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body>
            <div class="container mt-4">
                <h1>Production Readiness Report</h1>
                <p class="text-muted">Generated on {formatted_time}</p>
                
                <div class="alert alert-{status_color}">
                    <h4>Overall Status: {'✅ READY' if readiness_report.overall_ready else '❌ NOT READY'}</h4>
                </div>
                
                <div class="row mb-4">
                    <div class="col-md-3">
                        <div class="card">
                            <div class="card-body">
                                <h5>Functionality</h5>
                                <h2>{readiness_report.functionality_score*100:.1f}%</h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card">
                            <div class="card-body">
                                <h5>Performance</h5>
                                <h2>{readiness_report.performance_score*100:.1f}%</h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card">
                            <div class="card-body">
                                <h5>Reliability</h5>
                                <h2>{readiness_report.reliability_score*100:.1f}%</h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card">
                            <div class="card-body">
                                <h5>Security</h5>
                                <h2>{readiness_report.security_score*100:.1f}%</h2>
                            </div>
                        </div>
                    </div>
                </div>
                
                <h3>Recommendations</h3>
                <ul>
                    {recommendations_html}
                </ul>
                
                <h3>Critical Issues</h3>
                <ul>
                    {issues_html}
                </ul>
            </div>
        </body>
        </html>
        """
    
    def _generate_readiness_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate production readiness recommendations."""
        recommendations = []
        
        # Check each validation area
        for area, result in validation_results.items():
            if not result.get("passed", True):
                pass_rate = result.get("pass_rate", 0.0)
                
                if area == "functionality" and pass_rate < 0.95:
                    recommendations.append(f"Improve functionality test pass rate (current: {pass_rate*100:.1f}%, target: 95%)")
                
                elif area == "performance" and pass_rate < 0.90:
                    recommendations.append(f"Address performance issues (current: {pass_rate*100:.1f}%, target: 90%)")
                
                elif area == "reliability" and pass_rate < 0.90:
                    recommendations.append(f"Improve system reliability (current: {pass_rate*100:.1f}%, target: 90%)")
                
                elif area == "security" and pass_rate < 1.0:
                    recommendations.append(f"Fix all security issues (current: {pass_rate*100:.1f}%, target: 100%)")
        
        return recommendations
    
    def _identify_critical_issues(self, validation_results: Dict[str, Any]) -> List[str]:
        """Identify critical issues preventing production deployment."""
        critical_issues = []
        
        # Security failures are always critical
        if not validation_results.get("security", {}).get("passed", True):
            critical_issues.append("Security validation failed - deployment blocked")
        
        # Very low functionality score is critical
        functionality_score = validation_results.get("functionality", {}).get("pass_rate", 0.0)
        if functionality_score < 0.80:
            critical_issues.append(f"Functionality score too low: {functionality_score*100:.1f}% (minimum: 80%)")
        
        # Performance issues are critical if severe
        performance_score = validation_results.get("performance", {}).get("pass_rate", 0.0)
        if performance_score < 0.70:
            critical_issues.append(f"Performance score too low: {performance_score*100:.1f}% (minimum: 70%)")
        
        return critical_issues
    
    def _load_html_template(self) -> str:
        """Load HTML template for reports."""
        # This would load from a template file
        # For now, we generate inline
        return ""
"""
Automated Reporting System for QuantumRerank Benchmarks.

Generates comprehensive reports with visualizations, statistical analysis,
and PRD compliance validation for benchmark results.
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import asdict
import logging

# Plotting imports (optional, with fallbacks)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None

import numpy as np
from .benchmark_framework import BenchmarkResult, BenchmarkConfig
from .comparison import ComparisonResult, ComparativeAnalyzer
from .metrics import BenchmarkMetrics

logger = logging.getLogger(__name__)


class BenchmarkReporter:
    """
    Comprehensive benchmark reporting system.
    
    Generates detailed reports with statistics, visualizations,
    and PRD compliance analysis.
    """
    
    def __init__(self, output_dir: str = "benchmark_reports"):
        """
        Initialize benchmark reporter.
        
        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "html").mkdir(exist_ok=True)
        (self.output_dir / "json").mkdir(exist_ok=True)
        (self.output_dir / "csv").mkdir(exist_ok=True)
        if PLOTTING_AVAILABLE:
            (self.output_dir / "plots").mkdir(exist_ok=True)
        
        logger.info(f"Initialized BenchmarkReporter with output dir: {output_dir}")
    
    def generate_comprehensive_report(self,
                                    benchmark_results: Dict[str, List[BenchmarkResult]],
                                    comparison_results: Optional[Dict[str, ComparisonResult]] = None,
                                    config: Optional[BenchmarkConfig] = None) -> str:
        """
        Generate comprehensive benchmark report.
        
        Args:
            benchmark_results: Results organized by category
            comparison_results: Statistical comparison results
            config: Benchmark configuration
            
        Returns:
            Path to generated HTML report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"benchmark_report_{timestamp}"
        
        # Generate all report formats
        html_path = self._generate_html_report(
            benchmark_results, comparison_results, config, report_name
        )
        
        self._generate_json_report(
            benchmark_results, comparison_results, config, report_name
        )
        
        self._generate_csv_reports(benchmark_results, report_name)
        
        if PLOTTING_AVAILABLE:
            self._generate_plots(benchmark_results, comparison_results, report_name)
        
        logger.info(f"Generated comprehensive report: {html_path}")
        return str(html_path)
    
    def _generate_html_report(self,
                            benchmark_results: Dict[str, List[BenchmarkResult]],
                            comparison_results: Optional[Dict[str, ComparisonResult]],
                            config: Optional[BenchmarkConfig],
                            report_name: str) -> Path:
        """Generate HTML report with complete analysis."""
        html_path = self.output_dir / "html" / f"{report_name}.html"
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(benchmark_results)
        prd_compliance = self._analyze_prd_compliance(benchmark_results)
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>QuantumRerank Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ margin: 10px 0; }}
        .pass {{ color: green; font-weight: bold; }}
        .fail {{ color: red; font-weight: bold; }}
        .warning {{ color: orange; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .stats-table {{ margin: 20px 0; }}
        .comparison-section {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>QuantumRerank Benchmark Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Report ID:</strong> {report_name}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        {self._generate_executive_summary(summary_stats, prd_compliance)}
    </div>
    
    <div class="section">
        <h2>PRD Compliance Analysis</h2>
        {self._generate_prd_compliance_html(prd_compliance)}
    </div>
    
    <div class="section">
        <h2>Performance Results by Category</h2>
        {self._generate_performance_results_html(benchmark_results)}
    </div>
    
    {self._generate_comparison_html(comparison_results) if comparison_results else ""}
    
    <div class="section">
        <h2>Detailed Statistics</h2>
        {self._generate_detailed_stats_html(summary_stats)}
    </div>
    
    <div class="section">
        <h2>Configuration</h2>
        {self._generate_config_html(config) if config else "<p>No configuration provided</p>"}
    </div>
    
    <div class="section">
        <h2>Raw Data Summary</h2>
        <p>Total benchmark runs: {sum(len(results) for results in benchmark_results.values())}</p>
        <p>Categories tested: {', '.join(benchmark_results.keys())}</p>
        <p>Successful tests: {sum(sum(1 for r in results if r.success) for results in benchmark_results.values())}</p>
    </div>
</body>
</html>
"""
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return html_path
    
    def _generate_executive_summary(self, 
                                  summary_stats: Dict[str, Any],
                                  prd_compliance: Dict[str, Any]) -> str:
        """Generate executive summary section."""
        overall_compliance = prd_compliance.get('overall_compliant', False)
        compliance_class = 'pass' if overall_compliance else 'fail'
        
        return f"""
        <div class="metric">
            <strong>Overall PRD Compliance:</strong> 
            <span class="{compliance_class}">
                {'PASS' if overall_compliance else 'FAIL'}
            </span>
        </div>
        <div class="metric">
            <strong>Total Tests Executed:</strong> {summary_stats.get('total_tests', 0)}
        </div>
        <div class="metric">
            <strong>Success Rate:</strong> {summary_stats.get('success_rate', 0):.1%}
        </div>
        <div class="metric">
            <strong>Average Latency:</strong> {summary_stats.get('avg_latency_ms', 0):.2f}ms
        </div>
        <div class="metric">
            <strong>Peak Memory Usage:</strong> {summary_stats.get('peak_memory_mb', 0):.2f}MB
        </div>
        """
    
    def _generate_prd_compliance_html(self, prd_compliance: Dict[str, Any]) -> str:
        """Generate PRD compliance section."""
        html = "<table class='stats-table'>"
        html += "<tr><th>PRD Requirement</th><th>Target</th><th>Actual</th><th>Status</th></tr>"
        
        # Similarity computation
        sim_compliant = prd_compliance.get('similarity_compliant', False)
        sim_actual = prd_compliance.get('similarity_actual_ms', 0)
        html += f"""
        <tr>
            <td>Similarity Computation</td>
            <td>&lt;100ms</td>
            <td>{sim_actual:.2f}ms</td>
            <td><span class="{'pass' if sim_compliant else 'fail'}">
                {'PASS' if sim_compliant else 'FAIL'}
            </span></td>
        </tr>
        """
        
        # Batch processing
        batch_compliant = prd_compliance.get('batch_compliant', False)
        batch_actual = prd_compliance.get('batch_actual_ms', 0)
        html += f"""
        <tr>
            <td>Batch Processing (50 docs)</td>
            <td>&lt;500ms</td>
            <td>{batch_actual:.2f}ms</td>
            <td><span class="{'pass' if batch_compliant else 'fail'}">
                {'PASS' if batch_compliant else 'FAIL'}
            </span></td>
        </tr>
        """
        
        # Memory usage
        memory_compliant = prd_compliance.get('memory_compliant', False)
        memory_actual = prd_compliance.get('memory_actual_gb', 0)
        html += f"""
        <tr>
            <td>Memory Usage (100 docs)</td>
            <td>&lt;2GB</td>
            <td>{memory_actual:.2f}GB</td>
            <td><span class="{'pass' if memory_compliant else 'fail'}">
                {'PASS' if memory_compliant else 'FAIL'}
            </span></td>
        </tr>
        """
        
        html += "</table>"
        return html
    
    def _generate_performance_results_html(self, 
                                         benchmark_results: Dict[str, List[BenchmarkResult]]) -> str:
        """Generate performance results by category."""
        html = ""
        
        for category, results in benchmark_results.items():
            html += f"<h3>{category.replace('_', ' ').title()}</h3>"
            html += "<table class='stats-table'>"
            html += "<tr><th>Test</th><th>Duration (ms)</th><th>Target Met</th><th>Success</th></tr>"
            
            for result in results:
                target_class = 'pass' if result.target_met else 'fail'
                success_class = 'pass' if result.success else 'fail'
                
                html += f"""
                <tr>
                    <td>{result.test_name}</td>
                    <td>{result.duration_ms:.2f}</td>
                    <td><span class="{target_class}">
                        {'YES' if result.target_met else 'NO'}
                    </span></td>
                    <td><span class="{success_class}">
                        {'YES' if result.success else 'NO'}
                    </span></td>
                </tr>
                """
            
            html += "</table>"
        
        return html
    
    def _generate_comparison_html(self, comparison_results: Dict[str, ComparisonResult]) -> str:
        """Generate statistical comparison section."""
        html = """
        <div class="section">
            <h2>Statistical Comparisons</h2>
            <div class="comparison-section">
        """
        
        html += "<table class='stats-table'>"
        html += "<tr><th>Metric</th><th>Winner</th><th>Improvement</th><th>Significant</th><th>Effect Size</th></tr>"
        
        for metric_name, comparison in comparison_results.items():
            sig_class = 'pass' if comparison.statistically_significant else 'warning'
            
            html += f"""
            <tr>
                <td>{metric_name.replace('_', ' ').title()}</td>
                <td><strong>{comparison.winner}</strong></td>
                <td>{comparison.improvement_percent:.1f}%</td>
                <td><span class="{sig_class}">
                    {'YES' if comparison.statistically_significant else 'NO'}
                </span></td>
                <td>{comparison.effect_size:.3f}</td>
            </tr>
            """
        
        html += "</table></div></div>"
        return html
    
    def _generate_detailed_stats_html(self, summary_stats: Dict[str, Any]) -> str:
        """Generate detailed statistics section."""
        html = "<table class='stats-table'>"
        html += "<tr><th>Metric</th><th>Value</th></tr>"
        
        for key, value in summary_stats.items():
            if isinstance(value, float):
                value_str = f"{value:.3f}"
            else:
                value_str = str(value)
            
            html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value_str}</td></tr>"
        
        html += "</table>"
        return html
    
    def _generate_config_html(self, config: BenchmarkConfig) -> str:
        """Generate configuration section."""
        html = "<table class='stats-table'>"
        html += "<tr><th>Parameter</th><th>Value</th></tr>"
        
        config_dict = asdict(config)
        for key, value in config_dict.items():
            html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"
        
        html += "</table>"
        return html
    
    def _generate_json_report(self,
                            benchmark_results: Dict[str, List[BenchmarkResult]],
                            comparison_results: Optional[Dict[str, ComparisonResult]],
                            config: Optional[BenchmarkConfig],
                            report_name: str):
        """Generate JSON report for programmatic access."""
        json_path = self.output_dir / "json" / f"{report_name}.json"
        
        # Convert results to serializable format
        serializable_results = {}
        for category, results in benchmark_results.items():
            serializable_results[category] = [asdict(result) for result in results]
        
        serializable_comparisons = {}
        if comparison_results:
            for metric, comparison in comparison_results.items():
                serializable_comparisons[metric] = asdict(comparison)
        
        report_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_name": report_name,
                "generator": "QuantumRerank BenchmarkReporter"
            },
            "config": asdict(config) if config else None,
            "benchmark_results": serializable_results,
            "comparison_results": serializable_comparisons,
            "summary": self._calculate_summary_statistics(benchmark_results),
            "prd_compliance": self._analyze_prd_compliance(benchmark_results)
        }
        
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.debug(f"Generated JSON report: {json_path}")
    
    def _generate_csv_reports(self,
                            benchmark_results: Dict[str, List[BenchmarkResult]],
                            report_name: str):
        """Generate CSV reports for data analysis."""
        csv_path = self.output_dir / "csv" / f"{report_name}_results.csv"
        
        # Flatten all results into CSV format
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'category', 'test_name', 'component', 'metric_type',
                'duration_ms', 'target_ms', 'target_met', 'success',
                'memory_mb', 'batch_size', 'document_count', 'timestamp'
            ])
            
            # Data rows
            for category, results in benchmark_results.items():
                for result in results:
                    writer.writerow([
                        category,
                        result.test_name,
                        result.component,
                        result.metric_type,
                        result.duration_ms,
                        result.target_ms,
                        result.target_met,
                        result.success,
                        result.memory_mb,
                        result.batch_size,
                        result.document_count,
                        result.timestamp.isoformat()
                    ])
        
        logger.debug(f"Generated CSV report: {csv_path}")
    
    def _generate_plots(self,
                       benchmark_results: Dict[str, List[BenchmarkResult]],
                       comparison_results: Optional[Dict[str, ComparisonResult]],
                       report_name: str):
        """Generate visualization plots."""
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting libraries not available, skipping plot generation")
            return
        
        plots_dir = self.output_dir / "plots" / report_name
        plots_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        
        # Performance overview plot
        self._plot_performance_overview(benchmark_results, plots_dir)
        
        # PRD compliance plot
        self._plot_prd_compliance(benchmark_results, plots_dir)
        
        # Comparison plots
        if comparison_results:
            self._plot_comparisons(comparison_results, plots_dir)
        
        logger.debug(f"Generated plots in: {plots_dir}")
    
    def _plot_performance_overview(self, 
                                 benchmark_results: Dict[str, List[BenchmarkResult]],
                                 plots_dir: Path):
        """Generate performance overview plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('QuantumRerank Performance Overview', fontsize=16)
        
        # Collect data
        latencies = []
        memory_usage = []
        categories = []
        
        for category, results in benchmark_results.items():
            for result in results:
                if result.success and result.metric_type == 'latency':
                    latencies.append(result.duration_ms)
                    categories.append(category)
                if result.memory_mb is not None:
                    memory_usage.append(result.memory_mb)
        
        # Latency distribution
        if latencies:
            axes[0, 0].hist(latencies, bins=20, alpha=0.7, color='blue')
            axes[0, 0].axvline(100, color='red', linestyle='--', label='PRD Target (100ms)')
            axes[0, 0].set_xlabel('Latency (ms)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Latency Distribution')
            axes[0, 0].legend()
        
        # Memory usage
        if memory_usage:
            axes[0, 1].hist(memory_usage, bins=20, alpha=0.7, color='green')
            axes[0, 1].set_xlabel('Memory Usage (MB)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Memory Usage Distribution')
        
        # Success rates by category
        success_rates = {}
        for category, results in benchmark_results.items():
            if results:
                success_rates[category] = sum(1 for r in results if r.success) / len(results)
        
        if success_rates:
            categories = list(success_rates.keys())
            rates = list(success_rates.values())
            axes[1, 0].bar(categories, rates, alpha=0.7, color='orange')
            axes[1, 0].set_ylabel('Success Rate')
            axes[1, 0].set_title('Success Rate by Category')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # PRD compliance summary
        prd_compliance = self._analyze_prd_compliance(benchmark_results)
        compliance_metrics = ['similarity_compliant', 'batch_compliant', 'memory_compliant']
        compliance_values = [prd_compliance.get(metric, False) for metric in compliance_metrics]
        compliance_labels = ['Similarity', 'Batch', 'Memory']
        
        axes[1, 1].bar(compliance_labels, compliance_values, alpha=0.7, color='red')
        axes[1, 1].set_ylabel('Compliance (1=Pass, 0=Fail)')
        axes[1, 1].set_title('PRD Compliance Status')
        axes[1, 1].set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'performance_overview.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_prd_compliance(self, benchmark_results: Dict[str, List[BenchmarkResult]], plots_dir: Path):
        """Generate PRD compliance visualization."""
        prd_compliance = self._analyze_prd_compliance(benchmark_results)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = ['Similarity (<100ms)', 'Batch (<500ms)', 'Memory (<2GB)']
        actual_values = [
            prd_compliance.get('similarity_actual_ms', 0),
            prd_compliance.get('batch_actual_ms', 0),
            prd_compliance.get('memory_actual_gb', 0) * 1000  # Convert to MB for comparison
        ]
        target_values = [100, 500, 2000]  # All in consistent units
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, actual_values, width, label='Actual', alpha=0.7)
        ax.bar(x + width/2, target_values, width, label='Target', alpha=0.7)
        
        ax.set_xlabel('PRD Metrics')
        ax.set_ylabel('Value')
        ax.set_title('PRD Compliance: Actual vs Target')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'prd_compliance.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_comparisons(self, comparison_results: Dict[str, ComparisonResult], plots_dir: Path):
        """Generate comparison plots."""
        if not comparison_results:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = list(comparison_results.keys())
        improvements = [comp.improvement_percent for comp in comparison_results.values()]
        winners = [comp.winner for comp in comparison_results.values()]
        
        # Color bars based on winner
        colors = ['blue' if w == 'Quantum' else 'red' if w == 'Classical' else 'gray' for w in winners]
        
        bars = ax.bar(metrics, improvements, color=colors, alpha=0.7)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Performance Improvements: Quantum vs Classical')
        ax.tick_params(axis='x', rotation=45)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.7, label='Quantum Wins'),
            Patch(facecolor='red', alpha=0.7, label='Classical Wins'),
            Patch(facecolor='gray', alpha=0.7, label='Tie')
        ]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'comparisons.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _calculate_summary_statistics(self, 
                                    benchmark_results: Dict[str, List[BenchmarkResult]]) -> Dict[str, Any]:
        """Calculate overall summary statistics."""
        all_results = []
        for results in benchmark_results.values():
            all_results.extend(results)
        
        if not all_results:
            return {}
        
        latencies = [r.duration_ms for r in all_results if r.success and r.metric_type == 'latency']
        memory_values = [r.memory_mb for r in all_results if r.memory_mb is not None]
        
        return {
            'total_tests': len(all_results),
            'successful_tests': sum(1 for r in all_results if r.success),
            'success_rate': sum(1 for r in all_results if r.success) / len(all_results),
            'avg_latency_ms': np.mean(latencies) if latencies else 0,
            'median_latency_ms': np.median(latencies) if latencies else 0,
            'peak_memory_mb': np.max(memory_values) if memory_values else 0,
            'avg_memory_mb': np.mean(memory_values) if memory_values else 0
        }
    
    def _analyze_prd_compliance(self, 
                              benchmark_results: Dict[str, List[BenchmarkResult]]) -> Dict[str, Any]:
        """Analyze PRD compliance from benchmark results."""
        # Find similarity computation results
        similarity_results = []
        batch_results = []
        memory_results = []
        
        for category, results in benchmark_results.items():
            for result in results:
                if 'similarity' in result.test_name.lower() and result.success:
                    similarity_results.append(result.duration_ms)
                elif 'batch' in result.test_name.lower() and result.batch_size == 50 and result.success:
                    batch_results.append(result.duration_ms)
                elif result.memory_mb and result.document_count == 100:
                    memory_results.append(result.memory_mb / 1024)  # Convert to GB
        
        # Calculate compliance
        similarity_compliant = False
        similarity_actual = 0
        if similarity_results:
            similarity_actual = np.mean(similarity_results)
            similarity_compliant = similarity_actual < 100
        
        batch_compliant = False
        batch_actual = 0
        if batch_results:
            batch_actual = np.mean(batch_results)
            batch_compliant = batch_actual < 500
        
        memory_compliant = False
        memory_actual = 0
        if memory_results:
            memory_actual = np.mean(memory_results)
            memory_compliant = memory_actual < 2.0
        
        overall_compliant = similarity_compliant and batch_compliant and memory_compliant
        
        return {
            'overall_compliant': overall_compliant,
            'similarity_compliant': similarity_compliant,
            'similarity_actual_ms': similarity_actual,
            'batch_compliant': batch_compliant,
            'batch_actual_ms': batch_actual,
            'memory_compliant': memory_compliant,
            'memory_actual_gb': memory_actual
        }
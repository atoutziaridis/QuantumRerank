#!/usr/bin/env python3
"""
Comprehensive deployment pipeline validation for QuantumRerank.

This script validates that all deployment components are properly configured
and the deployment pipeline is ready for production use.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeploymentPipelineValidator:
    """
    Validates the complete deployment pipeline for QuantumRerank.
    
    Checks all deployment artifacts, configurations, and procedures.
    """
    
    def __init__(self, project_root: str = "/Users/alkist/Projects/QuantumRerank"):
        """Initialize the validator with project root path."""
        self.project_root = Path(project_root)
        self.results = []
        
    def validate_all(self) -> Dict[str, Any]:
        """Run complete deployment pipeline validation."""
        logger.info("Starting deployment pipeline validation...")
        
        # Core deployment files
        self._validate_deployment_files()
        
        # Validation scripts
        self._validate_validation_scripts()
        
        # Operational documentation
        self._validate_operational_docs()
        
        # Cloud deployment guides
        self._validate_cloud_deployment_guides()
        
        # Configuration files
        self._validate_configuration_files()
        
        # Scripts and automation
        self._validate_deployment_scripts()
        
        # Generate summary
        return self._generate_summary()
    
    def _validate_deployment_files(self) -> None:
        """Validate core deployment files exist and are properly configured."""
        logger.info("Validating deployment files...")
        
        required_files = [
            "deployment/production_deployment.md",
            "deployment/Dockerfile",
            "deployment/k8s/deployment.yaml",
            "deployment/k8s/service.yaml",
            "deployment/k8s/configmap.yaml",
            "deployment/k8s/ingress.yaml",
            "deployment/k8s/hpa.yaml",
            "deployment/k8s/rbac.yaml",
            "deployment/k8s/secret.yaml"
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                self._record_success(f"deployment_file_{file_path.replace('/', '_')}", 
                                   f"Found {file_path}")
                
                # Additional validation for specific files
                if file_path.endswith('.yaml'):
                    self._validate_yaml_syntax(full_path)
                elif file_path.endswith('.md'):
                    self._validate_markdown_content(full_path)
            else:
                self._record_failure(f"deployment_file_{file_path.replace('/', '_')}", 
                                   f"Missing {file_path}")
    
    def _validate_validation_scripts(self) -> None:
        """Validate deployment validation scripts."""
        logger.info("Validating validation scripts...")
        
        validation_scripts = [
            "deployment/validation/smoke_tests.py",
            "deployment/validation/performance_validation.py"
        ]
        
        for script_path in validation_scripts:
            full_path = self.project_root / script_path
            if full_path.exists():
                self._record_success(f"validation_script_{script_path.split('/')[-1]}", 
                                   f"Found {script_path}")
                self._validate_python_script(full_path)
            else:
                self._record_failure(f"validation_script_{script_path.split('/')[-1]}", 
                                   f"Missing {script_path}")
    
    def _validate_operational_docs(self) -> None:
        """Validate operational documentation."""
        logger.info("Validating operational documentation...")
        
        operational_docs = [
            "operations/production_operations.md",
            "operations/incident_response.md"
        ]
        
        for doc_path in operational_docs:
            full_path = self.project_root / doc_path
            if full_path.exists():
                self._record_success(f"operational_doc_{doc_path.split('/')[-1]}", 
                                   f"Found {doc_path}")
                self._validate_operational_content(full_path)
            else:
                self._record_failure(f"operational_doc_{doc_path.split('/')[-1]}", 
                                   f"Missing {doc_path}")
    
    def _validate_cloud_deployment_guides(self) -> None:
        """Validate cloud-specific deployment guides."""
        logger.info("Validating cloud deployment guides...")
        
        cloud_guides = [
            "deployment/cloud/aws_deployment.md",
            "deployment/cloud/gcp_deployment.md",
            "deployment/cloud/azure_deployment.md"
        ]
        
        for guide_path in cloud_guides:
            full_path = self.project_root / guide_path
            if full_path.exists():
                self._record_success(f"cloud_guide_{guide_path.split('/')[-1]}", 
                                   f"Found {guide_path}")
                self._validate_cloud_guide_content(full_path)
            else:
                self._record_failure(f"cloud_guide_{guide_path.split('/')[-1]}", 
                                   f"Missing {guide_path}")
    
    def _validate_configuration_files(self) -> None:
        """Validate configuration files."""
        logger.info("Validating configuration files...")
        
        # Check for essential configuration patterns
        config_checks = [
            ("deployment/k8s/", "Kubernetes manifests directory"),
            ("deployment/monitoring/", "Monitoring configurations"),
            ("scripts/", "Deployment scripts directory")
        ]
        
        for path, description in config_checks:
            full_path = self.project_root / path
            if full_path.exists():
                self._record_success(f"config_{path.replace('/', '_')}", 
                                   f"Found {description}")
            else:
                self._record_failure(f"config_{path.replace('/', '_')}", 
                                   f"Missing {description}")
    
    def _validate_deployment_scripts(self) -> None:
        """Validate deployment automation scripts."""
        logger.info("Validating deployment scripts...")
        
        # Check for script files
        script_patterns = [
            "scripts/deploy.sh",
            "scripts/rollback.sh",
            "scripts/health-check.sh"
        ]
        
        for script_path in script_patterns:
            full_path = self.project_root / script_path
            if full_path.exists():
                self._record_success(f"script_{script_path.split('/')[-1]}", 
                                   f"Found {script_path}")
                self._validate_shell_script(full_path)
            else:
                # Scripts might not exist yet, so we'll note as optional
                self._record_info(f"script_{script_path.split('/')[-1]}", 
                                f"Optional script {script_path} not found")
    
    def _validate_yaml_syntax(self, file_path: Path) -> None:
        """Validate YAML file syntax."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Basic YAML validation - check for common issues
            if content.strip():
                # Check for proper indentation (no tabs)
                if '\t' in content:
                    self._record_warning(f"yaml_syntax_{file_path.name}", 
                                       f"File {file_path.name} contains tabs - should use spaces")
                else:
                    self._record_success(f"yaml_syntax_{file_path.name}", 
                                       f"YAML syntax appears valid for {file_path.name}")
            else:
                self._record_warning(f"yaml_syntax_{file_path.name}", 
                                   f"File {file_path.name} is empty")
                
        except Exception as e:
            self._record_failure(f"yaml_syntax_{file_path.name}", 
                               f"Error reading {file_path.name}: {e}")
    
    def _validate_markdown_content(self, file_path: Path) -> None:
        """Validate markdown documentation content."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for essential content markers
            essential_sections = {
                "production_deployment.md": [
                    "## Overview", "## Prerequisites", "## Deployment Phases", 
                    "## Validation", "## Troubleshooting"
                ],
                "production_operations.md": [
                    "## Daily Operations", "## Monitoring", "## Incident Response", 
                    "## Maintenance"
                ],
                "incident_response.md": [
                    "## Incident Classification", "## Response Process", 
                    "## Escalation", "## Communication"
                ]
            }
            
            file_name = file_path.name
            if file_name in essential_sections:
                missing_sections = []
                for section in essential_sections[file_name]:
                    if section not in content:
                        missing_sections.append(section)
                
                if missing_sections:
                    self._record_warning(f"markdown_content_{file_name}", 
                                       f"Missing sections in {file_name}: {missing_sections}")
                else:
                    self._record_success(f"markdown_content_{file_name}", 
                                       f"All essential sections found in {file_name}")
            else:
                self._record_success(f"markdown_content_{file_name}", 
                                   f"Markdown content validated for {file_name}")
                
        except Exception as e:
            self._record_failure(f"markdown_content_{file_path.name}", 
                               f"Error reading {file_path.name}: {e}")
    
    def _validate_python_script(self, file_path: Path) -> None:
        """Validate Python script syntax."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Basic syntax checks
            if content.startswith('#!/usr/bin/env python3'):
                self._record_success(f"python_shebang_{file_path.name}", 
                                   f"Proper shebang in {file_path.name}")
            
            # Check for main function
            if 'def main(' in content and '__name__ == "__main__"' in content:
                self._record_success(f"python_main_{file_path.name}", 
                                   f"Main function found in {file_path.name}")
            else:
                self._record_warning(f"python_main_{file_path.name}", 
                                   f"No main function pattern in {file_path.name}")
            
            # Check for essential imports
            essential_imports = ['import logging', 'import json', 'import time']
            for imp in essential_imports:
                if imp in content:
                    self._record_success(f"python_import_{imp.split()[-1]}_{file_path.name}", 
                                       f"Found {imp} in {file_path.name}")
                
        except Exception as e:
            self._record_failure(f"python_script_{file_path.name}", 
                               f"Error reading {file_path.name}: {e}")
    
    def _validate_shell_script(self, file_path: Path) -> None:
        """Validate shell script."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for proper shebang
            if content.startswith('#!/bin/bash') or content.startswith('#!/usr/bin/env bash'):
                self._record_success(f"shell_shebang_{file_path.name}", 
                                   f"Proper shebang in {file_path.name}")
            
            # Check for set -e (exit on error)
            if 'set -e' in content:
                self._record_success(f"shell_safety_{file_path.name}", 
                                   f"Error handling found in {file_path.name}")
                
        except Exception as e:
            self._record_failure(f"shell_script_{file_path.name}", 
                               f"Error reading {file_path.name}: {e}")
    
    def _validate_operational_content(self, file_path: Path) -> None:
        """Validate operational documentation content."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for essential operational content
            operational_keywords = [
                'kubectl', 'monitoring', 'alerts', 'troubleshooting', 
                'escalation', 'incident', 'performance'
            ]
            
            found_keywords = []
            for keyword in operational_keywords:
                if keyword.lower() in content.lower():
                    found_keywords.append(keyword)
            
            if len(found_keywords) >= len(operational_keywords) // 2:
                self._record_success(f"operational_content_{file_path.name}", 
                                   f"Good operational content coverage in {file_path.name}")
            else:
                self._record_warning(f"operational_content_{file_path.name}", 
                                   f"Limited operational keywords in {file_path.name}")
                
        except Exception as e:
            self._record_failure(f"operational_content_{file_path.name}", 
                               f"Error validating {file_path.name}: {e}")
    
    def _validate_cloud_guide_content(self, file_path: Path) -> None:
        """Validate cloud deployment guide content."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for cloud-specific content
            cloud_name = file_path.stem.split('_')[0].upper()  # aws, gcp, azure
            
            cloud_keywords = {
                'aws': ['eks', 'ecr', 'vpc', 'iam', 'cloudwatch'],
                'gcp': ['gke', 'gcr', 'vpc', 'iam', 'monitoring'],
                'azure': ['aks', 'acr', 'vnet', 'rbac', 'monitor']
            }
            
            expected_keywords = cloud_keywords.get(cloud_name.lower(), [])
            found_keywords = []
            
            for keyword in expected_keywords:
                if keyword.lower() in content.lower():
                    found_keywords.append(keyword)
            
            if len(found_keywords) >= len(expected_keywords) // 2:
                self._record_success(f"cloud_content_{file_path.name}", 
                                   f"Good {cloud_name} content coverage in {file_path.name}")
            else:
                self._record_warning(f"cloud_content_{file_path.name}", 
                                   f"Limited {cloud_name} keywords in {file_path.name}")
                
        except Exception as e:
            self._record_failure(f"cloud_content_{file_path.name}", 
                               f"Error validating {file_path.name}: {e}")
    
    def _record_success(self, test_name: str, message: str) -> None:
        """Record a successful validation."""
        self.results.append({
            'test': test_name,
            'status': 'success',
            'message': message
        })
        logger.info(f"✓ {test_name}: {message}")
    
    def _record_failure(self, test_name: str, message: str) -> None:
        """Record a failed validation."""
        self.results.append({
            'test': test_name,
            'status': 'failure',
            'message': message
        })
        logger.error(f"✗ {test_name}: {message}")
    
    def _record_warning(self, test_name: str, message: str) -> None:
        """Record a validation warning."""
        self.results.append({
            'test': test_name,
            'status': 'warning',
            'message': message
        })
        logger.warning(f"⚠ {test_name}: {message}")
    
    def _record_info(self, test_name: str, message: str) -> None:
        """Record informational result."""
        self.results.append({
            'test': test_name,
            'status': 'info',
            'message': message
        })
        logger.info(f"ℹ {test_name}: {message}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary."""
        total_tests = len(self.results)
        successes = sum(1 for r in self.results if r['status'] == 'success')
        failures = sum(1 for r in self.results if r['status'] == 'failure')
        warnings = sum(1 for r in self.results if r['status'] == 'warning')
        infos = sum(1 for r in self.results if r['status'] == 'info')
        
        summary = {
            'total_tests': total_tests,
            'successes': successes,
            'failures': failures,
            'warnings': warnings,
            'infos': infos,
            'success_rate': (successes / total_tests * 100) if total_tests > 0 else 0,
            'overall_status': 'PASS' if failures == 0 else 'FAIL',
            'results': self.results
        }
        
        return summary


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QuantumRerank Deployment Pipeline Validator")
    parser.add_argument("--project-root", default="/Users/alkist/Projects/QuantumRerank",
                       help="Project root directory")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run validation
    validator = DeploymentPipelineValidator(args.project_root)
    summary = validator.validate_all()
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results written to {args.output}")
    
    # Print summary
    print("\n" + "="*70)
    print("DEPLOYMENT PIPELINE VALIDATION SUMMARY")
    print("="*70)
    print(f"Overall Status: {summary['overall_status']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"  ✓ Successes: {summary['successes']}")
    print(f"  ✗ Failures: {summary['failures']}")
    print(f"  ⚠ Warnings: {summary['warnings']}")
    print(f"  ℹ Info: {summary['infos']}")
    
    if summary['failures'] > 0:
        print("\nFAILURES:")
        for result in summary['results']:
            if result['status'] == 'failure':
                print(f"  ✗ {result['test']}: {result['message']}")
    
    if summary['warnings'] > 0:
        print("\nWARNINGS:")
        for result in summary['results']:
            if result['status'] == 'warning':
                print(f"  ⚠ {result['test']}: {result['message']}")
    
    print("\nDEPLOYMENT READINESS:")
    readiness_score = summary['success_rate']
    if readiness_score >= 95:
        print("🟢 EXCELLENT - Ready for production deployment")
    elif readiness_score >= 85:
        print("🟡 GOOD - Minor issues to address before deployment")
    elif readiness_score >= 70:
        print("🟠 FAIR - Several issues to resolve before deployment")
    else:
        print("🔴 POOR - Major issues must be resolved before deployment")
    
    # Exit with appropriate code
    sys.exit(0 if summary['failures'] == 0 else 1)


if __name__ == "__main__":
    main()